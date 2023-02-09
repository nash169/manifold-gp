import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
from importlib.resources import files
from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel
from manifold_gp.models.riemann_gp import RiemannGP
from manifold_gp.utils.generate_truth import groundtruth_from_mesh

# Set device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Load dataset
data_path = files('manifold_gp.data').joinpath('dragon10k.stl')
nodes, faces, truth = groundtruth_from_mesh(data_path)
sampled_x = torch.from_numpy(nodes).float().to(device)
sampled_y = torch.from_numpy(truth).float().to(device)
(m, n) = sampled_x.shape

# Normalization
mu, std = sampled_x.mean(0), sampled_x.std(0)
sampled_x.sub_(mu).div_(std)

# Noise Grid
min_noise_scale = 0
max_noise_scale = 0.5
resolution = 4
manifold_noise, function_noise = torch.meshgrid(torch.linspace(min_noise_scale, max_noise_scale, steps=resolution),
                                                torch.linspace(min_noise_scale, max_noise_scale, steps=resolution))

# Record Loss & MSE
count = 1
loss = torch.zeros_like(manifold_noise)
mse = torch.zeros_like(manifold_noise)

# Create Kernel
nu = 3
neighbors = 50
modes = 100
alpha = 1
laplacian = "normalized"
kernel = gpytorch.kernels.ScaleKernel(RiemannMaternKernel(
    nu=nu, nodes=sampled_x, neighbors=neighbors, modes=modes, alpha=1, laplacian=laplacian))

# Define Optimization Parameters
lr = 1e-1
iters = 100
verbose = False

# Model Hyperparameters
hypers = {
    'likelihood.noise_covar.noise': 1e-3**2,
    'covar_module.base_kernel.epsilon': 0.5,
    'covar_module.base_kernel.lengthscale': 0.5,
    'covar_module.outputscale': 1.0,
}

for i in range(resolution):
    for j in range(resolution):
        print(
            f"Iteration: {count}/{resolution**2}, Manifold Noise: {manifold_noise[i,j]}, Function Noise: {function_noise[i,j]}")
        # Add noise to samples
        noisy_x = sampled_x + \
            manifold_noise[i, j] * torch.randn(m, n).to(device)
        noisy_y = sampled_y + function_noise[i, j] * torch.randn(m).to(device)

        # Update Kernel
        if hasattr(kernel, 'base_kernel'):
            kernel.base_kernel.nodes = noisy_x
            kernel.base_kernel.generate_graph()
        else:
            kernel.nodes = noisy_x
            kernel.generate_graph()

        # Create model
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-8))
        model = RiemannGP(noisy_x, noisy_y, likelihood, kernel).to(device)

        # Init hyperparameters and train the model
        model.initialize(**hypers)
        loss[i, j] = model.manifold_informed_train(lr, iters, verbose)

        # Evaluate the model
        likelihood.eval()
        model.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = likelihood(model(noisy_x))
            mse[i, j] = torch.linalg.norm(
                sampled_y - preds.mean)/sampled_y.shape[0]

        count += 1

with torch.no_grad():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    contour_mse = ax.contourf(function_noise, manifold_noise, mse)
    fig.colorbar(contour_mse)
    ax.scatter(function_noise, manifold_noise, color='r')
    ax.axis('square')
    fig.savefig('mse_grid.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    contour_loss = ax.contourf(function_noise, manifold_noise, loss)
    ax.scatter(function_noise, manifold_noise, color='r')
    ax.axis('square')
    fig.colorbar(contour_loss)
    fig.savefig('loss_grid.png')
