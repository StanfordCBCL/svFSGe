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
from scipy.ndimage import gaussian_filter

from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

from simulation import Simulation
from cylinder import generate_mesh
from fsg import rad

if platform.system() == 'Darwin':
    usr = '/Users/pfaller/'
elif platform.system() == 'Linux':
    usr = '/home/pfaller/'

# from https://github.com/StanfordCBCL/DataCuration
sys.path.append(os.path.join(usr, 'work/repos/DataCuration'))
from vtk_functions import read_geo, write_geo, calculator, extract_surface, clean, threshold, get_all_arrays


def cart2rad(pts):
    phi = np.arctan2(pts[:, 0], pts[:, 1])
    axi = pts[:, 2]
    rad = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    return np.vstack((rad * phi, axi)).T


period = np.pi / 2.0
geo_inp = read_geo('wss_buckled.vtp').GetOutput()
points = v2n(geo_inp.GetPoints().GetData())
wss = v2n(geo_inp.GetPointData().GetArray('WSS'))

# 2d cylindrical surface coordinates
coord = cart2rad(points)
pdb.set_trace()
