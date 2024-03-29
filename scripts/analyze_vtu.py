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

# files = [1, 2, 38, 39]
files = np.arange(1, 40)

geo_inp = []
for f in files:
    geo_inp += [
        read_geo(
            os.path.join("partitioned", "solid_inp_" + str(f).zfill(3) + ".vtu")
        ).GetOutput()
    ]

geo_out = []
for f in files:
    geo_out += [
        read_geo(
            os.path.join("partitioned", "solid_out_" + str(f).zfill(3) + ".vtu")
        ).GetOutput()
    ]

wss = []
disp = []
for g in geo_inp:
    wss += [v2n(g.GetPointData().GetArray("varWallProps"))[:, 6]]
for g in geo_out:
    disp += [rad(v2n(g.GetPointData().GetArray("Displacement")))]

pdb.set_trace()
