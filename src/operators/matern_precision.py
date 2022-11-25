import torch
import gpytorch
import faiss
import faiss.contrib.torch_utils
from gpytorch.constraints import Positive
from src.operators.sparse_operator import SparseOperator
# from linear_operator import LinearOperator


# class PrecisionOperator(LinearOperator):
#     def __init__(self, X, module):
#         super(PrecisionOperator, self).__init__(
#             X.requires_grad_(True), module=module)
#         self.module_ = module

#     def _matmul(self, x):
#         return self.module_(x)

#     def _size(self):
#         return self.module_.size_

#     def _transpose_nonbatch(self):
#         return self

#     def _diagonal(self) -> torch.Tensor:
#         return self.module_.values_[:, 0]


class MaternPrecision(gpytorch.Module):

    def __init__(self, X, k, nu=5, taylor=3, **kwargs):
        super().__init__(**kwargs)

        if X.ndimension() == 1:
            self.X_ = X.unsqueeze(-1)
        else:
            self.X_ = X

        # Operator size
        self.size_ = torch.Size([X.shape[0], X.shape[0]])

        # Create graph
        self.knn_ = faiss.IndexFlatL2(X.shape[1])
        self.knn_.train(X)
        self.knn_.add(X)
        distances, neighbors = self.knn_.search(X, k+1)

        # Store indices and distances
        self.val_ = distances[:, 1:]
        self.idx_ = neighbors

        # Store smoothness hyperparameter
        self.nu_ = nu

        # Expansion terms
        self.taylor_ = taylor

        # Heat kernel length
        self.register_parameter(
            name='raw_eps', parameter=torch.nn.Parameter(torch.tensor(-1.), requires_grad=True)
        )
        self.register_constraint("raw_eps", Positive())

        # Riemann Matern kernel length
        self.register_parameter(
            name='raw_length', parameter=torch.nn.Parameter(torch.tensor(-1.), requires_grad=True)
        )
        self.register_constraint("raw_length", Positive())

        # Signal variance
        self.register_parameter(
            name='raw_signal', parameter=torch.nn.Parameter(torch.tensor(-1.), requires_grad=True)
        )
        self.register_constraint("raw_signal", Positive())

        # Noise variance
        self.register_parameter(
            name='raw_noise', parameter=torch.nn.Parameter(torch.tensor(-5.))
        )
        self.register_constraint("raw_noise", Positive())

    @property
    def eps(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_eps_constraint.transform(self.raw_eps)

    @eps.setter
    def eps(self, value):
        return self._set_eps(value)

    def _set_eps(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_eps)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(
            raw_eps=self.raw_eps_constraint.inverse_transform(value))

    @property
    def length(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_length_constraint.transform(self.raw_length)

    @length.setter
    def length(self, value):
        return self._set_length(value)

    def _set_length(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_length)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(
            raw_length=self.raw_length_constraint.inverse_transform(value))

    @property
    def signal(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_signal_constraint.transform(self.raw_signal)

    @signal.setter
    def signal(self, value):
        return self._set_signal(value)

    def _set_signal(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_signal)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(
            raw_signal=self.raw_signal_constraint.inverse_transform(value))

    @property
    def noise(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_noise_constraint.transform(self.raw_noise)

    @noise.setter
    def noise(self, value):
        return self._set_noise(value)

    def _set_noise(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_noise)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(
            raw_noise=self.raw_noise_constraint.inverse_transform(value))

    @property
    def dtype(self):
        for param in self.parameters():
            return param.dtype
        return torch.get_default_dtype()

    def to_operator(self):
        return SparseOperator(self.X_, self)

    def forward(self, x, **params):
        val = self.val_.div(-self.eps).exp()
        deg = val.sum(dim=1)
        val = val.div(deg.unsqueeze(1)).div(deg[self.idx_[:, 1:]])
        val = torch.cat((torch.ones(
            self.X_.shape[0], 1), -val.div(val.sum(dim=1).unsqueeze(1))), dim=1)*4/self.eps
        val[:, 0] += 2*self.nu_/self.length**2

        y = torch.zeros_like(x).to(x.device)

        for iter in range(1, self.nu_*self.taylor_+1):
            x = torch.sum(
                val * x[self.idx_].permute(2, 0, 1), dim=2).t()
            if (iter % self.nu_ == 0):
                power = iter/self.nu_
                y += pow(-1, power + 1)*self.signal.pow(-2*power) * \
                    self.noise.pow(2*(power-1))*x
        return y
        # return self.signal**2*x

    def laplacian(self):
        val = self.val_.div(-self.eps).exp()
        deg = val.sum(dim=1)
        val = val.div(deg.unsqueeze(1)).div(deg[self.idx_[:, 1:]])
        val = torch.cat((torch.ones(
            self.X_.shape[0], 1), -val.div(val.sum(dim=1).unsqueeze(1))), dim=1)*4/self.eps

        rows = torch.arange(self.idx_.shape[0]).repeat_interleave(
            self.idx_.shape[1]).unsqueeze(0)
        cols = self.idx_.reshape(1, -1)
        val = val.reshape(1, -1).squeeze()

        return torch.sparse_coo_tensor(torch.cat((rows, cols), dim=0), val, (self.idx_.shape[0], self.idx_.shape[0]))
