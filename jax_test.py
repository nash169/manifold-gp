import numpy as np
import faiss
# import jax.numpy as jnp

# from jax import grad, jit, vmap
# from src.laplacian_jax import LaplacianJax
from scipy.sparse import coo_matrix

data = np.loadtxt('rsc/dumbbell.msh')
X = data[:, :2]
Y = data[:, -1][:, np.newaxis]

# # Training/Test points
# num_train = 50
# idx_train = torch.randint(X_sampled.shape[0], (num_train,))
# X_train = X_sampled[idx_train, :]
# Y_train = Y_sampled[idx_train]
# num_test = 10
# idx_test = torch.randint(X_sampled.shape[0], (num_test,))
# X_test = X_sampled[idx_test, :]
# Y_test = Y_sampled[idx_test]

# Build Diffusion Maps Laplacian
index = faiss.IndexFlatL2(X.shape[1])
index.train(X)
index.add(X)
k = 2
distances, neighbors = index.search(X, k+1)
distances = distances[:, 1:]
neighbors = neighbors[:, 1:]

row = np.arange(neighbors.shape[0]).repeat(neighbors.shape[1])
col = neighbors.reshape(1, -1).squeeze()
data = -distances.reshape(1, -1).squeeze()

L = coo_matrix((data, (row, col)), shape=(
    neighbors.shape[0], neighbors.shape[0]))

# lp = LaplacianJax(neighbors, distances)
# i = np.concatenate((np.repeat(np.arange(neighbors.shape[0]), neighbors.shape[1])[
#     np.newaxis, :], neighbors.reshape(1, -1)), axis=0)
# v = (X_sampled[i[0, :], :] - X_sampled[i[1, :], :]
#      ).pow(2).sum(dim=1).div(-lp.eps_).exp()
# L = torch.sparse_coo_tensor(
#     i, v, (neighbors.shape[0], neighbors.shape[0])).to(device)
# D = torch.sparse.sum(L, dim=1).pow(-1)
# index_diag = torch.cat((D.indices(), D.indices()), dim=0)
# D = torch.sparse_coo_tensor(index_diag, D.values(
# ), (neighbors.shape[0], neighbors.shape[0])).to(device)
# L = torch.sparse.mm(D, torch.sparse.mm(L, D))
# D = torch.sparse.sum(L, dim=1).pow(-1)
# D = torch.sparse_coo_tensor(index_diag, D.values(
# ), (neighbors.shape[0], neighbors.shape[0])).to(device)
# L = (torch.sparse_coo_tensor(index_diag, torch.ones(neighbors.shape[0]), (
#     neighbors.shape[0], neighbors.shape[0])).to(device) - torch.sparse.mm(D, L))/(1/4*lp.eps_)


# csp = sp.coalesce()
# torch.sparse_coo_tensor(csp.indices(), csp.values().exp(), csp.shape)

# W = LaplacianWeights(X.shape[0], i, v)

# layer = nn.Linear(X.shape[0], X.shape[0], bias=False)
# parametrize.register_parametrization(
#     layer, "weight", LaplacianWeights(X.shape[0], i, v))
