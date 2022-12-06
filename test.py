#!/usr/bin/env python
# encoding: utf-8

import faiss
import math
import torch
import numpy as np
import gpytorch
import gpytorch.settings as settings
import matplotlib.pyplot as plt

from src.exact_gp import ExactGP
from src.knn_expansion import KnnExpansion

from src.precision_matrices.matern_precision import MaternPrecision
from src.kernels.riemann_matern import RiemannMatern
from src.utils import groundtruth_from_mesh

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigs

# Load dataset
file = 'rsc/dumbbell.msh'  # 'rsc/dragon10k.stl'

# Define scenario
supervised = True
function_noise = torch.linspace(0, 0.05, 5)
manifold_noise = torch.linspace(0, 0.05, 5)

# Set cuda
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Load data
try:
    data = np.loadtxt(file)
    X_sampled = torch.from_numpy(data[:, :2]).float().to(device)
    Y_sampled = torch.from_numpy(data[:, -1][:, np.newaxis]).float().to(device)
except:
    nodes, faces, truth = groundtruth_from_mesh(file)
    x, y, z = (nodes[:, i] for i in range(3))
    X_sampled = torch.from_numpy(nodes).float().to(device)
    Y_sampled = torch.from_numpy(truth).float().to(device)

# Dataset dimensions
(m, n) = X_sampled.shape

# Training & Test indices
num_train = 50
idx_train = torch.randint(m, (num_train,)).to(device)
num_test = 10
idx_test = torch.randint(m, (num_test,)).to(device)

# Logs
counter = 1
x_grid = np.zeros((manifold_noise.shape[0], function_noise.shape[0]))
y_grid = np.zeros((manifold_noise.shape[0], function_noise.shape[0]))
mse_log = np.zeros((manifold_noise.shape[0], function_noise.shape[0]))
loss_log = np.zeros((manifold_noise.shape[0], function_noise.shape[0]))

for count_m, noise_m in enumerate(manifold_noise):
    # Add noise to manifold
    X_noisy = X_sampled + noise_m * torch.randn(m, n).to(device)

    for count_f, noise_f in enumerate(function_noise):
        # Add noise to function
        Y_noisy = Y_sampled + noise_f * torch.randn(m, 1).to(device)

        # Training points
        X_train = X_noisy[idx_train, :]
        Y_train = Y_noisy[idx_train]

        # Test points
        X_test = X_noisy[idx_test, :]
        Y_test = Y_noisy[idx_test]

        # Precision matrix model
        k = 4
        nu = 2
        if use_cuda:
            model = MaternPrecision(X_sampled, k, nu).cuda()
        else:
            model = MaternPrecision(X_sampled, k, nu)
        model.train()  # Basically it clears the cache

        # Training
        training_iter = 1000
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

        # Print iteration
        print(
            f"Iteration: {counter}/{manifold_noise.shape[0]*function_noise.shape[0]}, Manifold noise: {noise_m:0.3f}, Function noise: {noise_f:0.3f}")

        with settings.fast_computations(log_prob=False) and settings.max_cholesky_size(300) and torch.autograd.set_detect_anomaly(True):
            for i in range(training_iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()

                if supervised:
                    # Output from model
                    output = model(Y_noisy)

                    # Calc loss and backprop gradients
                    loss = 0.5 * sum([torch.dot(Y_noisy.squeeze(), output.squeeze()),
                                      -model.to_operator().inv_quad_logdet(logdet=True)[1], Y_noisy.size(-1) * math.log(2 * math.pi)])
                else:
                    # Output from model
                    output = model(Y_train, idx_train)

                    # Calc loss and backprop gradients
                    loss = 0.5 * sum([torch.dot(Y_train.squeeze(), output.squeeze()),
                                      -model.to_operator().inv_quad_logdet(logdet=True)[1], Y_train.size(-1) * math.log(2 * math.pi)])

                loss.backward()

                # Step
                optimizer.step()

        print(f"Loss: {loss.item():0.3f}")

        # Laplacian & Extract Eigenvectors
        L = model.laplacian()

        indices = L.coalesce().indices().cpu().detach().numpy()
        values = L.coalesce().values().cpu().detach().numpy()
        Ls = coo_matrix(
            (values, (indices[0, :], indices[1, :])), shape=L.shape)

        num_eigs = 10
        T, V = eigs(Ls, k=num_eigs, which='SR')

        T = torch.from_numpy(T).float().to(device).requires_grad_(True)
        V = torch.from_numpy(V).float().to(device).requires_grad_(True)

        # Create KNN Eigenfunctions
        f = KnnExpansion()
        f.alpha = V
        f.knn = model.knn_
        f.k = k
        f.sigma = torch.sqrt(model.eps/2)

        # Create Riemann Kernel
        kernel = RiemannMatern((T, f), nu, 1)
        kernel.length = model.length
        kernel.signal = model.signal

        # Create GP model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        # likelihood.noise = torch.tensor(1e-3)
        likelihood.noise = model.noise
        gp_model = ExactGP(
            X_train, Y_train.squeeze(), likelihood, kernel)

        # Evaluation
        gp_model.eval()
        likelihood.eval()

        with torch.no_grad():
            # Exact prediction
            preds = likelihood(gp_model(X_noisy))

            mse_log[count_m, count_f] = np.linalg.norm(Y_noisy - preds.mean)
            loss_log[count_m, count_f] = loss
            x_grid[count_m, count_f] = noise_m
            y_grid[count_m, count_f] = noise_f
        counter += 1

fig = plt.figure()
ax = fig.add_subplot(121)
contour_mse = ax.contourf(x_grid, y_grid, mse_log)
fig.colorbar(contour_mse)

ax = fig.add_subplot(122)
contour_loss = ax.contourf(x_grid, y_grid, loss_log)
fig.colorbar(contour_loss)


fig.savefig("results/eval.png")
plt.show()
