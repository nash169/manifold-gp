#!/usr/bin/env python
# encoding: utf-8

# import gc
import torch


def bump_function(x, alpha, beta):
    y = torch.zeros_like(x)
    y[x.abs() < alpha] = x[x.abs() < alpha].square().sub(alpha.square()).pow(-1).mul(beta).exp().div(alpha.square().pow(-1).mul(-beta).exp())
    return y


def grid_uniform(center, la, lb=None, samples=1):
    if lb == None:
        lb = la
    a = [center[0] - la, center[1] - lb]
    b = [center[0] + la, center[1] + lb]
    return torch.cat((torch.FloatTensor(samples, 1).uniform_(a[0], b[0]), torch.FloatTensor(samples, 1).uniform_(a[1], b[1])), dim=1)
