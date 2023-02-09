import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
from mayavi import mlab
from importlib.resources import files
from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel
from manifold_gp.models.riemann_gp import RiemannGP
from manifold_gp.utils.generate_truth import groundtruth_from_mesh

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

data_path = files('manifold_gp.data').joinpath('dragon10k.stl')
# data_path = files('manifold_gp.data').joinpath('dragon100k.msh')
nodes, faces, truth = groundtruth_from_mesh(data_path)
sampled_x = torch.from_numpy(nodes).float().to(device)
sampled_y = torch.from_numpy(truth).float().to(device)
(m, n) = sampled_x.shape

# mu, std = sampled_x.mean(0), sampled_x.std(0)
# sampled_x.sub_(mu).div_(std)

manifold_noise = 0.00
noisy_x = sampled_x + manifold_noise * torch.randn(m, n).to(device)
function_noise = 0.00
noisy_y = sampled_y + function_noise * torch.randn(m).to(device)

num_train = 500
train_idx = torch.randperm(m)[:num_train]
train_x = noisy_x[train_idx, :]
train_y = noisy_y[train_idx]

num_test = 100
test_idx = torch.randperm(m)[:num_test]
test_x = noisy_x[test_idx, :]
test_y = noisy_y[test_idx]

nu = 3
neighbors = 100
modes = 100
alpha = 1
laplacian = "normalized"
kernel = gpytorch.kernels.ScaleKernel(RiemannMaternKernel(
    nu=nu, nodes=noisy_x, neighbors=neighbors, modes=modes, alpha=alpha, laplacian=laplacian))

likelihood = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.GreaterThan(1e-8))
model = RiemannGP(train_x, train_y, likelihood, kernel, train_idx).to(device)

hypers = {
    'likelihood.noise_covar.noise': 1e-5,
    'covar_module.base_kernel.epsilon': 0.5027,
    'covar_module.base_kernel.lengthscale': 0.5054,
    'covar_module.outputscale': 1.0,
}
model.initialize(**hypers)

lr = 1e-1
iters = 30
verbose = True
loss = model.manifold_informed_train(lr, iters, verbose)

likelihood.eval()
model.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    preds = likelihood(model(noisy_x))
    mean = preds.mean
    std = preds.stddev
    posterior_sample = preds.sample()
    point = 3000
    kernel_eval = kernel(sampled_x[point, :].unsqueeze(
        0), sampled_x).evaluate().squeeze()
    var = kernel(sampled_x, sampled_x).diag()

    # Bring data to cpu
    sampled_x = sampled_x.cpu().numpy()
    sampled_y = sampled_y.cpu().numpy()
    train_x = train_x.cpu().numpy()
    test_x = test_x.cpu().numpy()
    kernel_eval = kernel_eval.cpu().numpy()
    posterior_sample = posterior_sample.cpu().numpy()
    mean = mean.cpu().numpy()
    std = std.cpu().numpy()
    var = var.cpu().numpy()

v_options = {'mode': 'sphere', 'scale_factor': 3e-3, 'color': (0, 0, 0)}

mlab.figure(size=(1920, 1360), fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
mlab.triangular_mesh(sampled_x[:, 0], sampled_x[:, 1],
                     sampled_x[:, 2], faces, scalars=mean)
bar = mlab.colorbar(orientation='vertical')
# bar.data_range = (0, 10)
mlab.points3d(train_x[:, 0], train_x[:, 1], train_x[:, 2], **v_options)
mlab.view(0.0, 180.0, 0.5139171204775793)

mlab.figure(size=(1920, 1360), fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
mlab.triangular_mesh(sampled_x[:, 0], sampled_x[:, 1],
                     sampled_x[:, 2], faces, scalars=std)
bar = mlab.colorbar(orientation='vertical')
# bar.data_range = (0, 10)
mlab.points3d(train_x[:, 0], train_x[:, 1], train_x[:, 2], **v_options)
mlab.view(0.0, 180.0, 0.5139171204775793)

# mlab.figure(size=(1920, 1360), fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
# mlab.triangular_mesh(sampled_x[:, 0], sampled_x[:, 1],
#                      sampled_x[:, 2], faces, scalars=kernel_eval)
# mlab.colorbar(orientation='vertical')
# mlab.points3d(sampled_x[point, 0], sampled_x[point, 1],
#               sampled_x[point, 2], **v_options)

# mlab.figure(size=(1920, 1360), fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
# mlab.triangular_mesh(sampled_x[:, 0], sampled_x[:, 1],
#                      sampled_x[:, 2], faces, scalars=var)
# mlab.colorbar(orientation='vertical')
