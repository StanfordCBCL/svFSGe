#!/usr/bin/env python
# coding=utf-8

import pdb
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.sparse.linalg as lg
import re
import os
import sys
import glob
import platform
import matplotlib as mpl

from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

from simulation import Simulation
from cylinder import generate_mesh
from fsg import rad

if platform.system() == "Darwin":
    usr = "/Users/pfaller/"
elif platform.system() == "Linux":
    usr = "/home/pfaller/"

# from https://github.com/StanfordCBCL/DataCuration
sys.path.append(os.path.join(usr, "work/repos/DataCuration"))
from vtk_functions import (
    read_geo,
    write_geo,
    calculator,
    extract_surface,
    clean,
    threshold,
    get_all_arrays,
)

names = ["gr", "poiseuille", "poiseuille_converged"]
fnames = [
    "gr/gr_*.vtu",
    "poiseuille_res_2022-07-20_13-51-11.931897/partitioned/disp_*.vtp",
    "poiseuille_res_2022-07-20_13-51-11.931897/partitioned/converged/disp_*.vtp",
]

for name, fname in zip(names, fnames):
    res = glob.glob(fname)
    res.sort()

    # inner point ids
    points = v2n(read_geo(res[0]).GetOutput().GetPoints().GetData())
    rd = rad(points)
    i_in = rd == np.min(rd)
    rd = rd[i_in]

    disp = []
    rads = []
    for f in res:
        geo = read_geo(f).GetOutput()
        disp += [v2n(geo.GetPointData().GetArray("Displacement"))[i_in]]
        rads += [rad(disp[-1])]
    disp = np.array(disp)
    rads = np.array(rads)

    disp_n = np.linalg.norm(disp, axis=2)
    # disp_n /= disp_n[-1, :]

    r_in = rd + rads
    wss = 1 / r_in**3
    wss /= wss[0, :]
    r_in /= r_in[0, :]

    fig, ax = plt.subplots(1, 2, dpi=100, figsize=(30, 30))

    plots = [r_in, wss]
    pnames = ["r_in", "wss"]

    for i in range(2):
        ax[i].plot(plots[i])
        ax[i].set_title(pnames[i])
        ax[i].set_xlabel("time")
        ax[i].set_ylabel(pnames[i])
        if plots[i].shape[0] <= 11:
            ax[i].set_xlim(0, 10)
        ax[i].set_ylim(0, 4)
        ax[i].grid(True)
    # plt.show()
    fig.savefig("time_" + name + ".png", bbox_inches="tight")
