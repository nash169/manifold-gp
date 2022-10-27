import numpy as np
import faiss
import jax.numpy as jnp

from jax import grad, jit, vmap

use_cuda = False

X = np.random.rand(10, 3)
Y = np.random.rand(10, 1)

if use_cuda:
    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatL2(X.shape[1])
    index = faiss.index_cpu_to_gpu(
        res, 0, index_flat)
else:
    index = faiss.IndexFlatL2(X.shape[1])
index.train(X)
index.add(X)

k = 3
distances, indices = index.search(X, k+1)
distances = distances[:, 1:]
indices = indices[:, 1:]


# csp = sp.coalesce()
# torch.sparse_coo_tensor(csp.indices(), csp.values().exp(), csp.shape)

# W = LaplacianWeights(X.shape[0], i, v)

# layer = nn.Linear(X.shape[0], X.shape[0], bias=False)
# parametrize.register_parametrization(
#     layer, "weight", LaplacianWeights(X.shape[0], i, v))
