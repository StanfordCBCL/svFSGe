#!/usr/bin/env python
# coding=utf-8

import pdb
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.sparse.linalg as lg
import sys

import matplotlib as mpl

# mpl.use('Agg')


def read_mat(fname):
    # get dimension
    with open(fname) as f:
        for line in f:
            pass
        dim = int(line.split()[0])

    # read matrix
    with open(fname) as file:
        lines = file.read().splitlines()
        d = np.genfromtxt(lines, names=['i', 'j', 'val'], dtype=None, delimiter=' ')

    # convert to sparse
    return scipy.sparse.coo_matrix(((d['val']), (d['i'] - 1, d['j'] - 1)), shape=(dim, dim))


def is_sym(A):
    return not (np.abs(A - A.transpose()) > 1.0e-10).sum()


def sym(A):
    return 0.5 * (A + A.transpose())


def asym(A):
    return A - sym(A)


def eig_ratio(A):
    eig, _ = np.linalg.eig(A)
    n_pos = np.sum(eig > 0)
    n_neg = np.sum(eig < 0)
    return min(n_pos, n_neg) / (n_pos + n_neg)


def main():
    fnames = ['K_gr', 'K_pre']

    b = np.loadtxt('RHS.mat')
    colors = ['b', 'r']  #'b'

    # n = 100
    # mat = np.abs(asym(mg)[:n, :n])
    # ax.spy(mat, precision=1.0e-12)
    # ax.grid(visible=True, which='both')
    # ax.imshow(mat, cmap='Blues', interpolation='nearest')
    # ax.grid(visible=True, which='both')
    # fig.show()

    # pdb.set_trace()
    for f, col in zip(fnames, colors):
        A = read_mat(f + '.mat').toarray()
        if 'gr' in f:
            A = np.dot(A.transpose(), A)
            b = np.dot(A.transpose(), b)

        # pdb.set_trace()
        cond = np.linalg.cond(A)

        sign, logdet = np.linalg.slogdet(A)
        # det = sign * np.exp(logdet)

        print(cond, logdet)

        eig, _ = np.linalg.eig(A)

        fig, ax = plt.subplots(dpi=300, figsize=(6, 3))
        ax.plot(eig, color=col)
        ax.grid(visible=True, which='both')
        fig.savefig(f + '_eig.png', bbox_inches="tight")

        # pdb.set_trace()
        # x, stat = scipy.sparse.linalg.gmres(A, b, maxiter=10000)
        # x, stat = scipy.sparse.linalg.bicg(A, b)
        #
        # res = np.linalg.norm(b - np.dot(A, x))
        # print(res)
        # print(stat)


if __name__ == '__main__':
    main()
