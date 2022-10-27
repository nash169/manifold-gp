import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

import matplotlib.pyplot as plt

import faiss
import faiss.contrib.torch_utils

from src.gaussian_process import GaussianProcess
from src.kernels.squared_exp import SquaredExp

from src.parametrization import MaternCovariance

# CPU/GPU setting
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

res = 100
x = np.linspace(-5, 5, res)
y = np.sin(x)

num_samples = 30
x_s = x[np.random.randint(0, high=res, size=num_samples, dtype=int)]
y_s = np.sin(x_s) + 0.3*(2*np.random.rand(num_samples) - 1)

# Kernel
sigma = 0.5
k = SquaredExp(sigma)

# GPR
gp = GaussianProcess()

gp.samples = torch.from_numpy(x_s[:, np.newaxis]).float().to(
    device).requires_grad_(True)
gp.target = torch.from_numpy(y_s[:, np.newaxis]).float().to(
    device).requires_grad_(True)

gp.kernel_ = k
gp.signal = (torch.tensor(1.), True)
gp.noise = (torch.tensor(1e-3), True)

gp.update()
sol = gp(torch.from_numpy(x[:, np.newaxis]).float().to(
    device).requires_grad_(True)).cpu().detach().numpy()

opt = torch.optim.Adam(gp.parameters(), lr=1e-4)
for i in range(10000):
    K = gp.covariance()
    loss = 0.5*(torch.mm(gp.target.t(), torch.linalg.solve(K, gp.target)) +
                torch.log(torch.linalg.det(K)))
    loss.backward()
    if i % 100 == 0:
        print(
            f"Iteration: {i}, Loss: {loss.item():0.2f}")
    opt.step()
    opt.zero_grad()

gp.update()
sol_opt = gp(torch.from_numpy(x[:, np.newaxis]).float().to(
    device).requires_grad_(True)).cpu().detach().numpy()


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y, linestyle="dashed", color='k')
ax.scatter(x_s, y_s, color="r")
ax.plot(x, sol, color="g")
ax.plot(x, sol_opt, color="b")

plt.show()
