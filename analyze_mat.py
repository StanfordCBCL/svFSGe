#!/usr/bin/env python
# coding=utf-8

import pdb
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.sparse.linalg as lg
import sys

import matplotlib as mpl
mpl.use('Agg')

# fname = sys.argv[1]
fnames = ['K_gr_t1', 'K_gr_t0', ]#'K_stvk',
colors = ['r', 'b']#'b'

# k = scipy.sparse.load_npz(fname)

# fig, ax = plt.subplots(1, len(fnames), dpi=100, figsize=(60, 30))
fig, ax = plt.subplots(dpi=100, figsize=(60, 30))
for i, (fname, color) in enumerate(zip(fnames, colors)):
    lines = []
    with open(fname + '.mat') as f:
        for line in f:
            if abs(float(line.split()[-1])) > 1.0e-16:
                lines += [line]
            pass
        dim = int(line.split()[0])
    with open(fname + '.mat', 'w') as f:
        f.writelines(lines)

    with open(fname + '.mat') as file:
        data = np.genfromtxt(file.read().splitlines(), names=['i', 'j', 'val'], dtype=None, delimiter=' ')
    M = scipy.sparse.coo_matrix(((data['val']), (data['i'] - 1, data['j'] - 1)), shape=(dim, dim))

    zero_diag = np.where(np.abs(M.diagonal()) < 1.0e-14)[0]
    if len(zero_diag) > 0:
        print(zero_diag + 1)

    #pdb.set_trace()
    ax.spy(M, color=color)#, precision=1.0e-15
    ax.grid()
    fig.savefig(fname, bbox_inches='tight')

    ew1, _ = lg.eigsh(M)
    ew2, _ = lg.eigsh(M, sigma=1e-8)  # <--- takes a long time

    ew1 = abs(ew1)
    ew2 = abs(ew2)

    cond = ew1.max() / ew2.min()
    print(fname + '\tcondition number\t' + str(cond) + '\tnnz\t' + str(M.nnz))

#plt.show()
fig.savefig('matrices.png', bbox_inches='tight')
