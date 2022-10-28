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
from scipy.ndimage import gaussian_filter
from scipy.spatial import distance_matrix

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
from vtk_functions import read_geo, write_geo


def cart2rad(pts):
    phi = np.arctan2(pts[:, 0], pts[:, 1])
    axi = pts[:, 2]
    rad = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    return np.vstack((rad * phi, axi)).T

def add_array(geo, name, array):
    arr = n2v(array)
    arr.SetName(name)
    geo.GetPointData().AddArray(arr)

def grid_to_image(pts, val, res=10):
    assert pts.shape[1] == 2, 'points must be two-dimensional'
    
    # get spacing of points
    ds = []
    nn = []
    grid = []
    for i in range(2):
        grid += [np.unique(pts[:, i])]
        nn += [len(grid[-1]) - 1]

        diff = np.unique(np.abs(np.diff(pts[:,i]))).tolist()
        if 0.0 in diff:
            diff.remove(0.0)
        ds += [np.min(diff)]
    
    # get number of points in each dimension
    ni = []
    if ds[1] > ds[0]:
        ni += [res * nn[0]]
        ni += [int(ds[1] / ds[0] * res) * nn[1]]
    else:
        ni += [int(ds[0] / ds[1] * res) * nn[0]]
        ni += [res * nn[1]]

    # sample
    x = []
    for i in range(2):
        c = pts[:, i]
        x += [np.linspace(np.min(c), np.max(c), ni[i])]
    
    # meshgrid
    xv, yv = np.meshgrid(x[0], x[1])
    xi = np.vstack((xv.flatten(), yv.flatten())).T
    interp = scipy.interpolate.griddata(pts, wss, xi)
    return interp.reshape(xv.shape), xi

def image_to_grid(img, pts, xi):
    return scipy.interpolate.griddata(xi, img.flatten(), pts)

def smooth(pts, val, ns=1):
    # 2d cylindrical surface coordinates
    coord = cart2rad(pts)

    # resample to image
    img, xi = grid_to_image(coord, val, res=ns)

    # smooth image
    img_smooth = gaussian_filter(img, sigma=ns*2)

    # resample to grid
    return image_to_grid(img_smooth, coord, xi)

# read data
geo = read_geo('wss_buckled.vtp').GetOutput()
points = v2n(geo.GetPoints().GetData())
wss = v2n(geo.GetPointData().GetArray('WSS'))

# smooth
wss_smooth = smooth(points, wss)

add_array(geo, 'WSS', wss_smooth)
write_geo('wss_smooth.vtp', geo)
