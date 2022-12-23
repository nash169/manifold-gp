#!/usr/bin/env python
# encoding: utf-8

import torch
from typing import Optional
from .riemann_kernel import RiemannKernel


class RiemannRBFKernel(RiemannKernel):
    has_lengthscale = True

    def __init__(self, **kwargs):
        super(RiemannRBFKernel, self).__init__(**kwargs)

    def spectral_density(self):
        s = (-0.5*self.lengthscale.square()*self.eigenvalues).exp()
        return s/s.sum()
