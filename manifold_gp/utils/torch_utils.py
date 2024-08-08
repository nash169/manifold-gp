#!/usr/bin/env python
# encoding: utf-8

import gc
import os
import torch


def torch_save(model, path):
    torch.save(model.state_dict(), os.path.join('', '{}.pt'.format(path)))


def torch_load(model, path):
    # model.load_state_dict(torch.load(os.path.join('', '{}.pt'.format(path)), map_location=torch.device(model.device)))
    model.load_state_dict(torch.load(os.path.join('', '{}.pt'.format(path))))
    model.eval()


def torch_set_grad(model, grad):
    for param in model.parameters():
        param.requires_grad = grad


def torch_set_zero(model):
    for p in model.parameters():
        p.data.fill_(0)


def torch_memory_allocation():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass


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
