import torch
import gpytorch
import faiss
import faiss.contrib.torch_utils

from gpytorch.constraints import Positive
from src.operators.sparse_operator import SparseOperator
from src.operators.module_operator import ModuleOperator


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
            name='raw_eps', parameter=torch.nn.Parameter(torch.tensor(-2.), requires_grad=True)
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
        return ModuleOperator(self.X_, self)

    def precision_matrix(self, x, labels):
        val = self.val_.div(-self.eps**2).exp()
        deg = val.sum(dim=1)
        val = val.div(deg.sqrt().unsqueeze(1)).div(
            deg.sqrt()[self.idx_[:, 1:]])
        val = torch.cat(
            (torch.ones(self.X_.shape[0], 1) + 2*self.nu_/self.length**2, -val), dim=1)

        def matmul(x): return torch.sum(
            val * x[self.idx_].permute(2, 0, 1), dim=2).t()

        if labels is not None:
            labaled = torch.zeros(self.X_.shape[0], x.shape[1])
            labaled[labels, :] = 1.0

            not_labaled = torch.ones(self.X_.shape[0], x.shape[1])
            not_labaled[labels, :] = 0.0

            y = torch.zeros(self.X_.shape[0], x.shape[1])
            y[labels, :] = x

            for _ in range(self.nu_):
                y = matmul(y)

            Q_xx = y[labels, :]
            opt = SparseOperator(val, self.idx_, self.size_)
            # y *= not_labaled
            y[labels, :] = 0.0

            for _ in range(self.nu_):
                y = opt.solve(y)

            z = y*not_labaled

            for _ in range(1, self.nu_):
                z = matmul(z)

            return (Q_xx + z[labels, :])/self.signal**2
        else:
            y = x

            for iter in range(0, self.nu_):
                y = torch.sum(
                    val * y[self.idx_].permute(2, 0, 1), dim=2).t()

            return y

    def forward(self, x, labels=None, **params):
        return self.precision_matrix(x - self.noise.pow(2)*self.precision_matrix(x + self.noise.pow(4)*self.precision_matrix(x, labels), labels), labels)

    def laplacian(self):
        val = self.val_.div(-self.eps**2).exp()
        deg = val.sum(dim=1)
        val = val.div(deg.sqrt().unsqueeze(1)).div(
            deg.sqrt()[self.idx_[:, 1:]])
        val = torch.cat((torch.ones(self.X_.shape[0], 1), -val), dim=1)
        # val = self.val_.div(-self.eps**2).exp()
        # deg = val.sum(dim=1)
        # val = val.div(deg.unsqueeze(1)).div(deg[self.idx_[:, 1:]])
        # val = torch.cat((torch.ones(
        #     self.X_.shape[0], 1), -val.div(val.sum(dim=1).unsqueeze(1))), dim=1)*4/self.eps

        rows = torch.arange(self.idx_.shape[0]).repeat_interleave(
            self.idx_.shape[1]).unsqueeze(0)
        cols = self.idx_.reshape(1, -1)
        val = val.reshape(1, -1).squeeze()

        return torch.sparse_coo_tensor(torch.cat((rows, cols), dim=0), val, (self.idx_.shape[0], self.idx_.shape[0]))

    def _base(self, x, values, indices):
        return torch.sum(values * x[indices].permute(2, 0, 1), dim=2).t()
