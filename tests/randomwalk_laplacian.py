# !/usr/bin/env python
# encoding: utf-8

from manifold_gp.utils.file_read import print_mat


def randomwalk_laplacian(x, epsilon, neighbors):
    import faiss
    import faiss.contrib.torch_utils
    from torch_geometric.utils import coalesce

    dim = x.shape[0]
    knn = faiss.IndexFlatL2(x.shape[1])
    knn.reset()
    knn.train(x)
    knn.add(x)
    val, idx = knn.search(x, neighbors)
    val, idx = val[:, 1:], idx[:, 1:]

    rows = torch.arange(idx.shape[0]).repeat_interleave(idx.shape[1])
    cols = idx.reshape(1, -1).squeeze()
    val = val.reshape(1, -1).squeeze()

    split = cols > rows
    rows, cols = torch.cat([rows[split], cols[~split]], dim=0), torch.cat([cols[split], rows[~split]], dim=0)

    idx = torch.stack([rows, cols], dim=0)
    val = torch.cat([val[split], val[~split]])
    idx, val = coalesce(idx, val, reduce='mean')
    rows = idx[0, :]
    cols = idx[1, :]

    val = val.div(-4*epsilon.square()).exp().squeeze()
    L = torch.sparse_coo_tensor(idx, val, (dim, dim)).to_dense()
    L = L + L.T + torch.eye(dim)
    D = L.sum(dim=1).pow(-1).diag()
    L = torch.mm(D, torch.mm(L, D))
    D = L.sum(dim=1).pow(-1).diag()
    L = torch.mm(D, L)

    return (torch.eye(dim)-L)/epsilon.square()


if __name__ == "__main__":
    import torch
    from importlib.resources import files
    from manifold_gp.utils.file_read import get_data, print_mat
    from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel
    from symmetric_laplacian import symmetric_laplacian

    # Load mesh and generate ground truth
    data_path = files('manifold_gp.data').joinpath('dumbbell.msh')
    data = get_data(data_path, "Nodes", "Elements")
    sampled_x = torch.from_numpy(data['Nodes'][:, 1:-1]).float()

    # kernel
    kernel = RiemannMaternKernel(
        nu=1,
        nodes=sampled_x,
        neighbors=50,
        operator="randomwalk",
        method="exact",
        modes=100,
        ball_scale=3.0,
        prior_bandwidth=False,
    )

    # laplacian matrix
    L_mat = randomwalk_laplacian(sampled_x, kernel.epsilon, kernel.neighbors)

    # laplacian operator
    L_opt = kernel.laplacian("randomwalk")

    # mul
    v = torch.rand(sampled_x.shape[0])
    print("(mul) Laplacian matrix")
    print(torch.mv(L_mat, v)[:10])

    print("(mul) Laplacian operator")
    print(L_opt.matmul(v.view(-1, 1)).squeeze()[:10])

    # solve
    v = torch.rand(sampled_x.shape[0])
    print("(solve) Laplacian matrix")
    print(torch.linalg.solve(L_mat, v)[:10])

    print("(solve) Laplacian operator")
    print(L_opt.solve(v.view(-1, 1)).squeeze()[:10])

    # eig
    evals_mat, evecs_mat = torch.linalg.eig(L_mat)
    evals_mat, evecs_mat = torch.real(evals_mat), torch.real(evecs_mat)
    evals_mat, evals_mat_idx = torch.sort(evals_mat)
    # evecs_mat = evecs_mat[:, evals_mat_idx]
    evals_opt, evecs_opt = L_opt.diagonalization()

    print("(evals) Laplacian matrix")
    print(evals_mat[:10])

    print("(evals) Laplacian operator")
    print(evals_opt[:10])

    print("(evecs) Laplacian matrix")
    print(evecs_mat[:10, evals_mat_idx[1]])

    print("(evecs) Laplacian operator")
    print(evecs_opt[:10, 1])
