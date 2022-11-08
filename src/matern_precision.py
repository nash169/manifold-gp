import torch
import gpytorch
import faiss
import faiss.contrib.torch_utils
from gpytorch.constraints import Positive
from linear_operator import LinearOperator

class PrecisionOperator(LinearOperator):
    def __init__(self, module):
        super(PrecisionOperator, self).__init__(module)
        self.module_ = module
        
    def _matmul(self, x):
        return self.module_(x)

    def _size(self):
        return self.module_.size_

    def _transpose_nonbatch(self):
        return self

class MaternPrecision(gpytorch.Module):

    def __init__(self, X, k, nu = 5, **kwargs):
        super().__init__(**kwargs)

        # Operator size
        self.size_ = torch.Size([X.shape[0], X.shape[0]])
        
        # Create graph
        index = faiss.IndexFlatL2(X.shape[1])
        index.train(X)
        index.add(X)
        distances, neighbors = index.search(X, k+1)

        # Store indices and distances
        self.values_ = distances[:, 1:]
        self.indices_ = neighbors[:, 1:]

        # Store smoothness hyperparameter
        self.nu_ = nu

        # register and constraint heat kernel length
        self.register_parameter(
            name='raw_eps', parameter=torch.nn.Parameter(torch.tensor(0.1))
        )
        self.register_constraint("raw_eps", Positive())

        # register and constraint Riemann Matern kernel length
        self.register_parameter(
            name='raw_length', parameter=torch.nn.Parameter(torch.tensor(1.))
        )
        self.register_constraint("raw_length", Positive())

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
        self.initialize(raw_eps=self.raw_eps_constraint.inverse_transform(value))

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
        self.initialize(raw_length=self.raw_length_constraint.inverse_transform(value))

    def forward(self, x, **params):
        values = self.values_.div(-self.eps).exp()
        D = values.sum(dim=1)
        values = values.div(D.unsqueeze(1)).div(D[self.indices_])
        values = torch.cat(
            (torch.ones(self.indices_.shape[0], 1), -values.div(values.sum(dim=1).unsqueeze(1))), dim=1)*4/self.eps
        indices = torch.cat(
            (torch.arange(self.indices_.shape[0]).unsqueeze(1), self.indices_), dim=1)
        values[:, 0] += 2*self.nu_/self.length**2

        for _ in range(self.nu_):
            x = torch.sum(
                values*x.t()[indices].permute(2, 0, 1), dim=2)

        return x