#!/usr/bin/env python
# encoding: utf-8

import torch
import gpytorch
from gpytorch.constraints import Positive


class RiemannMatern(gpytorch.kernels.Kernel):
    is_stationary = True

    def __init__(self, eigpairs, nu=5, d=1, **kwargs):
        super().__init__(**kwargs)

        # Store smoothness and dimension parameters
        self.nu_ = nu
        self.d_ = d

        # Store eigenvalues
        self.eigval_ = eigpairs[0]

        # Store eigenfunctions
        self.eigfun_ = eigpairs[1]

        # register the raw parameter and the constraint
        self.register_parameter(
            name='raw_length', parameter=torch.nn.Parameter(torch.tensor(1.))
        )
        self.register_constraint("raw_length", Positive())

        # register the raw parameter and the constraint
        self.register_parameter(
            name='raw_signal', parameter=torch.nn.Parameter(torch.tensor(1.))
        )
        self.register_constraint("raw_signal", Positive())

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

    def forward(self, x1, x2, **params):
        s = self.signal**2*(2*self.nu_ / self.length**2 +
             self.eigval_).pow(-self.nu_ - self.d_/2)
        s/=s.sum()

        return torch.mm(self.eigfun_(x1).T, s.unsqueeze(1)*self.eigfun_(x2))
