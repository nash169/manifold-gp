#!/usr/bin/env python
# encoding: utf-8

import numpy as np


def get_line(name, fs):
    while True:
        try:
            line = next(fs)
        except StopIteration:
            return

        if name in line:
            try:
                next(fs)
            except StopIteration:
                return

            while True:
                try:
                    line = next(fs)
                except StopIteration:
                    return

                if line in ["\n", "\r\n"]:
                    break
                else:
                    yield line
        elif not line:
            break


def get_data(file_path, *args):
    M = {}
    for var in args:
        with open(file_path) as fs:
            g = get_line(var, fs)
            M[var] = np.loadtxt(g)

    return M


def print_mat(mat):
    s = [[str(e) for e in row] for row in mat]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))
