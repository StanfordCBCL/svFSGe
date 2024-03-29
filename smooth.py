#!/usr/bin/env python
# coding=utf-8

import pdb
import os
import numpy as np
import vtk
import glob

from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata

from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

from vtk_functions import read_geo, write_geo

def cart2rad(pts):
    # convert cartesian coordinates to cylindrical coordinates
    phi = np.arctan2(pts[:, 0], pts[:, 1])
    axi = pts[:, 2]
    rad = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    return np.vstack((rad * phi, axi)).T


def add_array(geo, name, array):
    # add array to a geometry
    arr = n2v(array)
    arr.SetName(name)
    geo.GetPointData().AddArray(arr)


def grid_to_image(pts, val, res=10):
    # convert grid-based data to image array
    assert pts.shape[1] == 2, "points must be two-dimensional"

    # get spacing of points
    ds = []
    nn = []
    grid = []
    for i in range(2):
        grid += [np.unique(pts[:, i])]
        nn += [len(grid[-1]) - 1]

        diff = np.unique(np.abs(np.diff(pts[:, i]))).tolist()
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
    interp = griddata(pts, val, xi)
    return interp.reshape(xv.shape), xi


def image_to_grid(img, pts, xi):
    # convert image back to grid
    return griddata(xi, img.flatten(), pts)


def smooth_wss(pts, val, ns=1, smooth=2):
    # 2d cylindrical surface coordinates
    coord = cart2rad(pts)

    # resample to image
    img, xi = grid_to_image(coord, val, res=ns)

    # smooth image
    img_smooth = gaussian_filter(img, sigma=ns * smooth)

    # resample to grid
    return image_to_grid(img_smooth, coord, xi)


def smoothAttributes(data, relaxationFactor=0.1, numIterations=100):
    attributeSmoothingFilter = vtk.vtkAttributeSmoothingFilter()
    attributeSmoothingFilter.SetInputData(data)
    attributeSmoothingFilter.SetSmoothingStrategyToAllPoints()
    attributeSmoothingFilter.SetNumberOfIterations(numIterations)
    attributeSmoothingFilter.SetRelaxationFactor(relaxationFactor)
    attributeSmoothingFilter.SetWeightsTypeToDistance()
    attributeSmoothingFilter.AddExcludedArray("Pressure")
    attributeSmoothingFilter.AddExcludedArray("Velocity")
    attributeSmoothingFilter.AddExcludedArray("Displacement")
    attributeSmoothingFilter.Update()
    return attributeSmoothingFilter.GetOutput()

def smooth_gr(inp, n_smooth=10):
    c2p = inp
    for _ in range(n_smooth):
        # map point data to cell data
        p2c = vtk.vtkPointDataToCellData()
        p2c.SetInputData(c2p.GetOutput())
        p2c.Update()

        # map cell data to point data
        c2p = vtk.vtkCellDataToPointData()
        c2p.SetInputData(p2c.GetOutput())
        c2p.Update()
    return c2p.GetOutput()

if __name__ == "__main__":
    study = ""
    folder_in = "gr"
    folder_out = "gr_smooth"
    for fname in glob.glob(os.path.join(study, folder_in, "gr_*.vtu")):
        geo = read_geo(fname)
        out = os.path.join(study, folder_out, os.path.basename(fname))
        # write_geo(out, smoothAttributes(geo, 0.05, 100))
        write_geo(out, smooth_gr(geo))
    # pdb.set_trace()


# from simulation import Simulation
# from cylinder import generate_mesh
# from fsg import rad

# if platform.system() == 'Darwin':
#     usr = '/Users/pfaller/'
# elif platform.system() == 'Linux':
#     usr = '/home/pfaller/'

# # from https://github.com/StanfordCBCL/DataCuration
# sys.path.append(os.path.join(usr, 'work/repos/DataCuration'))

# # read data
# geo = read_geo('wss_buckled.vtp').GetOutput()
# points = v2n(geo.GetPoints().GetData())
# wss = v2n(geo.GetPointData().GetArray('WSS'))

# # smooth
# wss_smooth = smooth(points, wss)

# add_array(geo, 'WSS', wss_smooth)
# write_geo('wss_smooth.vtp', geo)
