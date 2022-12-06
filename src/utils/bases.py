#!/usr/bin/env python
# encoding: utf-8

import torch


def radial_basis(distances, lengthscale):
    torch.exp(-0.5 * distances/lengthscale**2)
