# !/usr/bin/env python
# encoding: utf-8

if __name__ == "__main__":
    import torch
    import gpytorch
    import numpy as np
    from importlib.resources import files
    from manifold_gp.utils.file_read import get_data
    from manifold_gp.models.riemann_gp import RiemannGP
    from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel
    from manifold_gp.utils.mesh_helper import groundtruth_from_samples
    import matplotlib.pyplot as plt

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load mesh and generate ground truth
    data_path = files('manifold_gp.data').joinpath('dumbbell.msh')
    data = get_data(data_path, "Nodes", "Elements")

    vertices = data['Nodes'][:, 1:-1]
    edges = data['Elements'][:, -2:].astype(int) - 1
    truth, geodesics = groundtruth_from_samples(vertices, edges)

    sampled_x = torch.from_numpy(vertices).float().contiguous().to(device)
    sampled_y = torch.from_numpy(truth).float().contiguous().to(device)

    # kernel
    kernel = gpytorch.kernels.ScaleKernel(
        RiemannMaternKernel(
            nu=1,
            nodes=sampled_x,
            neighbors=100,
            operator="randomwalk",
            method="exact",
            modes=sampled_x.shape[0],
            ball_scale=3.0,
            prior_bandwidth=False,
        )
    )

    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-8))
    model = RiemannGP(sampled_x, sampled_y, likelihood, kernel).to(device)

    hypers = {
        'likelihood.noise_covar.noise': 1e-2,
        'covar_module.base_kernel.epsilon': 0.02,
        'covar_module.base_kernel.lengthscale': 0.5,
        'covar_module.outputscale': 1.0,
    }
    model.initialize(**hypers)

    likelihood.eval()
    model.eval()

    fig, ax = plt.subplots(figsize=(10, 10))
    cut = 100
    v = 600
    v1 = kernel.base_kernel.eigenvectors[:, v]
    v2 = kernel.base_kernel.features(sampled_x)[:, v]
    # v2 = v2/torch.linalg.norm(v2)*torch.linalg.norm(v1)
    print(torch.linalg.norm(v1-v2))
    ax.plot(np.arange(sampled_x.shape[0])[:cut], v1[:cut].cpu())
    ax.plot(np.arange(sampled_x.shape[0])[:cut], v2[:cut].detach().cpu())
    plt.show()
