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
# from src.utils import groundtruth_from_mesh

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigs

# Load dataset
file = 'rsc/dumbbell.msh'  # 'rsc/dragon10k.stl'

# Define scenario
supervised, function_noise, manifold_noise = True, True, True

# Set cuda
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Load data
try:
    data = np.loadtxt(file)
    X_sampled = torch.from_numpy(data[:, :2]).float().to(device)
    Y_sampled = torch.from_numpy(data[:, -1][:, np.newaxis]).float().to(device)
except:
    pass
    # nodes, faces, truth = groundtruth_from_mesh(file)
    # x, y, z = (nodes[:, i] for i in range(3))
    # X_sampled = torch.from_numpy(nodes).float().to(device)
    # Y_sampled = torch.from_numpy(truth).float().to(device)

# Dataset dimensions
dim = X_sampled.shape[1]

# Noise
if manifold_noise:
    X_sampled = X_sampled + 0.01 * \
        torch.randn(X_sampled.shape[0], X_sampled.shape[1]).to(device)

if function_noise:
    Y_sampled = Y_sampled + 0.01 * \
        torch.randn(Y_sampled.shape[0], Y_sampled.shape[1]).to(device)

# Training points
num_train = 50
idx_train = torch.randint(X_sampled.shape[0], (num_train,))
X_train = X_sampled[idx_train, :]
Y_train = Y_sampled[idx_train]

# Test points
num_test = 10
idx_test = torch.randint(X_sampled.shape[0], (num_test,))
X_test = X_sampled[idx_test, :]
Y_test = Y_sampled[idx_test]

# Precision matrix model
k = 50
nu = 2
if use_cuda:
    model = MaternPrecision(X_sampled, k, nu).cuda()
else:
    model = MaternPrecision(X_sampled, k, nu)
model.train()  # Basically it clears the cache

# Training
training_iter = 100
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

with settings.fast_computations(log_prob=False) and settings.max_cholesky_size(300) and torch.autograd.set_detect_anomaly(True):
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()

        if supervised:
            # Output from model
            output = model(Y_sampled)

            # Calc loss and backprop gradients
            loss = 0.5 * sum([torch.dot(Y_sampled.squeeze(), output.squeeze()),
                              -model.to_operator().inv_quad_logdet(logdet=True)[1], Y_sampled.size(-1) * math.log(2 * math.pi)])
        else:
            # Output from model
            output = model(Y_train, idx_train)

            # Calc loss and backprop gradients
            loss = 0.5 * sum([torch.dot(Y_train.squeeze(), output.squeeze()),
                              -model.to_operator().inv_quad_logdet(logdet=True)[1], Y_train.size(-1) * math.log(2 * math.pi)])

        loss.backward()

        # Print step information
        print(f"Iteration: {i}, Loss: {loss.item():0.3f}, eps: {model.eps.item():0.3f}, length: {model.length.item():0.3f}, signal: {model.signal.item():0.3f}, noise: {model.noise.item():0.3f}")

        # Step
        optimizer.step()

# Laplacian & Extract Eigenvectors
L = model.laplacian()

# indices = L.coalesce().indices().cpu().detach().numpy()
# values = L.coalesce().values().cpu().detach().numpy()
# Ls = coo_matrix((values, (indices[0, :], indices[1, :])), shape=L.shape)

num_eigs = 10
# T, V = eigs(Ls, k=num_eigs, which='SR')

# T = torch.from_numpy(T).float().to(device).requires_grad_(True)
# V = torch.from_numpy(V).float().to(device).requires_grad_(True)

T, V = torch.lobpcg(L, k=num_eigs, largest=False)

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
if use_cuda:
    likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
    gp_model = ExactGP(X_train, Y_train.squeeze(), likelihood, kernel).cuda()
else:
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp_model = ExactGP(X_train, Y_train.squeeze(), likelihood, kernel)

likelihood.noise = model.noise

# Evaluation
gp_model.eval()
likelihood.eval()

with torch.no_grad():
    preds = likelihood(gp_model(X_sampled))
    mean = preds.mean.cpu().detach().numpy()
    std = preds.variance.sqrt().cpu().detach().numpy()

    X_sampled = X_sampled.cpu().detach().numpy()
    Y_sampled = Y_sampled.cpu().detach().numpy()
    X_train = X_train.cpu().detach().numpy()
    Y_train = Y_train.cpu().detach().numpy()
    X_test = X_test.cpu().detach().numpy()
    Y_test = Y_test.cpu().detach().numpy()

    if dim == 2:
        fig = plt.figure()
        ax = fig.add_subplot(221)
        ax.scatter(X_sampled[:, 0], X_sampled[:, 1])
        ax.scatter(X_sampled[0, 0], X_sampled[0, 1], c='k')
        ax.scatter(X_train[:, 0], X_train[:, 1], c="r", edgecolors="r")
        ax.axis('equal')
        ax.set_title('Training Points')

        ax = fig.add_subplot(222)
        plot = ax.scatter(X_sampled[:, 0], X_sampled[:, 1],
                          c=Y_sampled, vmin=-0.5, vmax=0.5)
        fig.colorbar(plot)
        ax.axis('equal')
        ax.set_title('Ground Truth')

        # fig = plt.figure()
        ax = fig.add_subplot(223)
        plot = ax.scatter(X_sampled[:, 0], X_sampled[:, 1],
                          c=mean, vmin=-0.5, vmax=0.5)
        fig.colorbar(plot)
        ax.axis('equal')
        ax.set_title('Mean')

        # fig = plt.figure()
        ax = fig.add_subplot(224)
        # , vmin=-0.5, vmax=0.5)
        plot = ax.scatter(X_sampled[:, 0], X_sampled[:, 1], c=std)
        fig.colorbar(plot)
        ax.axis('equal')
        ax.set_title('Standard Deviation')

        fig.savefig("results/dumbbell.png")
    elif dim == 3:
        pass

    plt.show()

# torch.save(model.state_dict(), 'model_state.pth')
# state_dict = torch.load('model_state.pth')
# model = ExactGPModel(train_x, train_y, likelihood)  # Create a new GP model
# model.load_state_dict(state_dict)
