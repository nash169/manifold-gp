#!/usr/bin/env python
# encoding: utf-8

import faiss
import math
import torch
import numpy as np
import gpytorch
import matplotlib.pyplot as plt
from src.knn_expansion import KnnExpansion
from src.operators.matern_precision import MaternPrecision
from src.kernels.riemann_knn import RiemannKNN
from exact_gp import ExactGP
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigs
import gpytorch.settings as settings

# Load data
data = np.loadtxt('rsc/dumbbell.msh')
# X = data[:, :2]
# Y = data[:, -1][:, np.newaxis]

use_cuda = False  # torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

X_sampled = torch.from_numpy(data[:, :2]).float().to(device)
Y_sampled = torch.from_numpy(data[:, -1][:, np.newaxis]).float().to(device)

# Re-order indices for visualization
knn = faiss.IndexFlatL2(X_sampled.shape[1])
knn.train(X_sampled)
knn.add(X_sampled)
distances, neighbors = knn.search(X_sampled, 3)
distances = distances[:, 1:]
neighbors = neighbors[:, 1:]
curr_index = 0
indices = np.array([0])
for i in range(X_sampled.shape[0]-1):
    if neighbors[curr_index, 0].numpy() in indices:
        indices = np.append(indices, neighbors[curr_index, 1])
        curr_index = neighbors[curr_index, 1]
    else:
        indices = np.append(indices, neighbors[curr_index, 0])
        curr_index = neighbors[curr_index, 0]
X_sampled = X_sampled[indices, :]
Y_sampled = Y_sampled[indices, :]

# Noise
X_noisy = X_sampled + 0.01*torch.randn(X_sampled.shape[0], X_sampled.shape[1])
Y_noisy = Y_sampled + 0.01*torch.randn(Y_sampled.shape[0], Y_sampled.shape[1])

# Training/Test points
num_train = 50
idx_train = torch.randint(X_noisy.shape[0], (num_train,))
X_train = X_noisy[idx_train, :]
Y_train = Y_noisy[idx_train]
num_test = 10
idx_test = torch.randint(X_noisy.shape[0], (num_test,))
X_test = X_noisy[idx_test, :]
Y_test = Y_noisy[idx_test]

# Precision matrix model
k = 4
nu = 2
model = MaternPrecision(X_sampled, k, nu)
model.train()  # Basically it clears the cache

# Training
training_iter = 5000
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

with settings.fast_computations(log_prob=False) and settings.max_cholesky_size(300):
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()

        # Output from model
        output = model(Y_noisy)

        # Calc loss and backprop gradients
        loss = 0.5 * sum([torch.dot(Y_noisy.squeeze(), output.squeeze()),
                          -model.to_operator().inv_quad_logdet(logdet=True)[1], Y_noisy.size(-1) * math.log(2 * math.pi)])
        loss.backward()

        # Print step information
        print(f"Iteration: {i}, Loss: {loss.item():0.3f}, eps: {model.eps.item():0.3f}, length: {model.length.item():0.3f}, signal: {model.signal.item():0.3f}, noise: {model.noise.item():0.3f}")

        # Step
        optimizer.step()

# Laplacian & Extract Eigenvectors
L = model.laplacian()

indices = L.coalesce().indices().cpu().detach().numpy()
values = L.coalesce().values().cpu().detach().numpy()
Ls = coo_matrix((values, (indices[0, :], indices[1, :])), shape=L.shape)

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
kernel = RiemannKNN((T, f), nu, 1)
kernel.length = model.length
kernel.signal = model.signal

# Create GP model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
# likelihood.noise = torch.tensor(1e-3)
likelihood.noise = model.noise
gp_model = ExactGP(X_train, Y_train.squeeze(), likelihood, kernel)

# Evaluation
gp_model.eval()
likelihood.eval()


# with torch.no_grad(), gpytorch.settings.fast_pred_var():
#     observed_pred = likelihood(gp_model(X_sampled))

f_preds = gp_model(X_sampled)
y_preds = likelihood(gp_model(X_sampled))


# f_mean = f_preds.mean
# f_var = f_preds.variance
# f_covar = f_preds.covariance_matrix
# f_samples = f_preds.sample(sample_shape=torch.Size(1000,))

with torch.no_grad():
    # # Initialize plot
    # f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    # pred = observed_pred.mean.numpy()
    # lower, upper = observed_pred.confidence_region()
    mean = f_preds.mean
    std = f_preds.variance.sqrt()

    fig = plt.figure()
    ax = fig.add_subplot(231)
    ax.scatter(X_sampled[:, 0], X_sampled[:, 1])
    ax.scatter(X_sampled[0, 0], X_sampled[0, 1], c='k')
    X_train = X_train.cpu().detach().numpy()
    ax.scatter(X_train[:, 0], X_train[:, 1], c="r", edgecolors="r")
    ax.axis('equal')
    ax.set_title('Training Points')

    ax = fig.add_subplot(232)
    plot = ax.scatter(X_sampled[:, 0], X_sampled[:, 1],
                      c=Y_sampled, vmin=-0.5, vmax=0.5)
    fig.colorbar(plot)
    ax.axis('equal')
    ax.set_title('Ground Truth')

    # fig = plt.figure()
    ax = fig.add_subplot(233)
    plot = ax.scatter(X_sampled[:, 0], X_sampled[:, 1],
                      c=mean, vmin=-0.5, vmax=0.5)
    fig.colorbar(plot)
    ax.axis('equal')
    ax.set_title('Mean')

    ax = fig.add_subplot(234)
    # ax.scatter(X[:, 0], X[:, 1])
    # X_test = X_test.cpu().detach().numpy()
    # ax.scatter(X_test[:, 0], X_test[:, 1], c="r", edgecolors="r")
    # ax.axis('equal')
    ax.plot(range(X_sampled.shape[0]), Y_sampled,
            linestyle='dashed', color="black")
    ax.scatter(idx_train, Y_train)
    ax.plot(range(X_sampled.shape[0]), mean)
    ax.fill_between(
        range(X_sampled.shape[0]), mean-std.numpy(), mean+std.numpy(), alpha=0.5)
    ax.set_title('Riemann GPR')

    # fig = plt.figure()
    ax = fig.add_subplot(235)
    # , vmin=-0.5, vmax=0.5)
    plot = ax.scatter(X_sampled[:, 0], X_sampled[:, 1], c=mean-std)
    fig.colorbar(plot)
    ax.axis('equal')
    ax.set_title('Mean - Standard Deviation')

    # fig = plt.figure()
    ax = fig.add_subplot(236)
    # , vmin=-0.5, vmax=0.5)
    plot = ax.scatter(X_sampled[:, 0], X_sampled[:, 1], c=mean+std)
    fig.colorbar(plot)
    ax.axis('equal')
    ax.set_title('Mean + Standard Deviation')

    plt.show()
