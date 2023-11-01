#!/usr/bin/env python
# encoding: utf-8

import os
import torch


class TorchHelper():
    @staticmethod
    def save(model, path):
        torch.save(model.state_dict(), os.path.join('', '{}.pt'.format(path)))

    @staticmethod
    def load(model, path):
        # model.load_state_dict(torch.load(os.path.join('', '{}.pt'.format(path)), map_location=torch.device(model.device)))
        model.load_state_dict(torch.load(os.path.join('', '{}.pt'.format(path))))
        model.eval()

    @staticmethod
    def set_grad(model, grad):
        for param in model.parameters():
            param.requires_grad = grad

    @staticmethod
    def set_zero(model):
        for p in model.parameters():
            p.data.fill_(0)

    @staticmethod
    def grid_uniform(center, la, lb=None, samples=1):
        if lb == None:
            lb = la
        a = [center[0] - la, center[1] - lb]
        b = [center[0] + la, center[1] + lb]
        return torch.cat((torch.FloatTensor(samples, 1).uniform_(a[0], b[0]), torch.FloatTensor(samples, 1).uniform_(a[1], b[1])), dim=1)
    
    @staticmethod
    def memory_allocation():
        import gc
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    print(type(obj), obj.size())
            except:
                pass
