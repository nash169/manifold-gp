import torch
import torch_cluster
from src.gaussian_process import GaussianProcess
from src.utils import squared_exp, lanczos

# # CPU/GPU setting
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")


# x_train = torch.rand(10, 2).to(device).requires_grad_(True)
# x_test = torch.rand(5, 2).to(device).requires_grad_(True)

# y = torch.rand(10, 1).to(device).requires_grad_(True)

# gp = GaussianProcess()

# gp.samples = x_train
# gp.target = y

# gp.kernel = (squared_exp, torch.tensor([0.5]), True)
# gp.signal = (torch.tensor(0.8), True)
# gp.noise = (torch.tensor(0.2), True)

# gp.update()


N = 5  # your choice goes here
vals = torch.arange(N*(N+1)/2) + 1  # values

A = torch.zeros(N, N)
i, j = torch.triu_indices(N, N)
A[i, j] = vals
A.T[i, j] = vals
