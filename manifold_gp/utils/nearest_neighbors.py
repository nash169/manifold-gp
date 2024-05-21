#!/usr/bin/env python
# encoding: utf-8

import torch
import faiss
import faiss.contrib.torch_utils
from torch_sparse import coalesce


class NearestNeighbors():
    def __init__(self, x=None, nlist=1) -> None:
        self.min_ivf = 5000

        if x is not None:
            self.train(x, nlist)

    def train(self, x, nlist=1):
        self.x = x
        (n, d) = self.x.shape

        if x.device.type == 'cuda':
            res = faiss.StandardGpuResources()
            self.index = faiss.GpuIndexIVFFlat(res, d, nlist, faiss.METRIC_L2) if n >= self.min_ivf else faiss.GpuIndexFlatL2(res, d)
        else:
            self.index = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, nlist) if n >= self.min_ivf else faiss.IndexFlatL2(d)

        if n >= self.min_ivf:
            assert not self.index.is_trained
            self.index.train(self.x)
            assert self.index.is_trained
        self.index.add(self.x)

        return self

    def search(self, x, k, nprobe=1):
        self.index.probe = nprobe
        return self.index.search(x, k)

    def graph(self, k, symmetric=True, self_loop=False, nprobe=1):
        val, idx = self.search(self.x, k, nprobe)

        if not self_loop:
            val, idx = val[:, 1:], idx[:, 1:]

        rows, cols = torch.arange(idx.shape[0]).repeat_interleave(idx.shape[1]).to(self.x.device), idx.reshape(1, -1).squeeze()
        val = val.reshape(1, -1).squeeze()

        if symmetric:
            split = cols > rows
            rows, cols = torch.cat([rows[split], cols[~split]], dim=0), torch.cat([cols[split], rows[~split]], dim=0)
            idx, val = coalesce(torch.stack([rows, cols], dim=0), torch.cat([val[split], val[~split]]), self.x.shape[0], self.x.shape[0], op='mean')
        else:
            idx = torch.stack([rows, cols], dim=0)

        return idx, val

    @property
    def min_ivf(self):
        return self._min_ivf

    @min_ivf.setter
    def min_ivf(self, value):
        self._min_ivf = value
