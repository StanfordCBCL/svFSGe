#!/usr/bin/env python

import pdb
import numpy as np
import meshio
import sys
import vtk
import os

from collections import defaultdict
from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

# from https://github.com/StanfordCBCL/DataCuration
sys.path.append('/home/pfaller/work/osmsc/curation_scripts')
sys.path.append('/Users/pfaller/work/repos/DataCuration')
from vtk_functions import read_geo, write_geo, get_points_cells, extract_surface, threshold

# output folder
f_out = '/home/pfaller/work/repos/svFSI_examples_fork/05-struct/03-GR/mesh_tube'

# cylinder size
r_inner = 0.64678
r_outer = 0.687
height = 0.03
# height = 15.0

# number of cells in each dimension

# radial
n_tran = 10
n_rad_s = 3

# circumferential
n_cir = 8

# axial
n_axi = 1

# number of circle segments (1 = full circle, 2 = half circle, ...)
n_sec = 4

# -------------
assert n_cir // 2 == n_cir / 2, 'number of elements in cir direction must be divisible by two'
assert n_tran >= n_cir // 2, 'choose number of transition elements at least half the number of cir elements'

# size of quadratic mesh
n_quad = n_cir // 2 + 1

# number of layers in fluid mesh
n_rad_f = n_quad + n_tran

# number of cells in circumferential direction (one more if the circle is closed)
n_cell_cir = n_cir
n_point_cir = n_cir
n_point_eff = n_cir * n_sec
if n_sec > 1:
    n_point_cir += 1

# create points
# n_points = (n_axi + 1) * (n_rad_f + n_rad_s + 1) * n_point_cir
n_points = (n_axi + 1) * (n_quad ** 2 + (n_tran + n_rad_s) * n_point_cir)
pid = 0
points = np.zeros((n_points, 3))
cosy = np.zeros((n_points, 6))
vol_dict = defaultdict(list)
surf_dict = defaultdict(list)
fiber_dict = defaultdict(list)

# generate quadratic mesh
for ia in range(n_axi + 1):
    for iy in range(n_quad):
        for ix in range(n_quad):
            axi = height * ia / n_axi
            rad = r_inner / (n_rad_f - 1)
            points[pid, :] = [ix * rad, iy * rad, axi]
            pid += 1

# generate transition mesh
for ia in range(n_axi + 1):
    for ir in range(n_tran):
        for ic in range(n_point_cir):
            # cylindrical coordinate system
            axi = height * ia / n_axi
            cir = 2 * np.pi * ic / n_cell_cir / n_sec
            rad = r_inner * (ir + n_quad) / (n_rad_f - 1)

            i_trans = (ir + 1) / n_tran
            if ic <= n_cell_cir // 2:
                rad_mod = rad * ((1 - i_trans) / np.cos(cir) + i_trans)
            else:
                rad_mod = rad * ((1 - i_trans) / np.sin(cir) + i_trans)
            points[pid, :] = [rad_mod * np.cos(cir), rad_mod * np.sin(cir), axi]
            pid += 1

# generate circular g&r mesh
    for ir in range(n_rad_s):
        for ic in range(n_point_cir):
            # cylindrical coordinate system
            axi = height * ia / n_axi
            cir = 2 * np.pi * ic / n_cell_cir / n_sec
            rad = r_inner + (r_outer - r_inner) * (ir + 1) / n_rad_s

            points[pid, :] = [rad * np.cos(cir), rad * np.sin(cir), axi]
            pid += 1

#
#
# for ia in range(n_axi + 1):
#     for ir in range(n_rad_f + n_rad_s + 1):
#         for ic in range(n_point_cir):
#             if ir <= n_rad_f // 2:
#                 if ic > ir * 2:
#                     continue
#
#             # cylindrical coordinate system
#             axi = height * ia / n_axi
#             cir = 2 * np.pi * ic / n_cell_cir / n_sec
#             if ir < n_rad_f:
#                 rad = r_inner * ir / n_rad_f
#             else:
#                 rad = r_inner + (r_outer - r_inner) * (ir - n_rad_f) / n_rad_s
#
#             # store normalized coordinates
#             cosy[pid, 0] = rad # / r_outer
#             cosy[pid, 1] = ic / n_cell_cir / n_sec
#             cosy[pid, 2] = ia / n_axi
#
#             # grid
#             if ir <= n_rad_f // 2:
#                 if ic <= ir:
#                     xi = ir
#                     yi = ic
#                 else:
#                     xi = 2 * ir - ic
#                     yi = ir
#                 points[pid, :] = [xi * r_inner / n_rad_f, yi * r_inner / n_rad_f, axi]
#             # transition from grid to circular
#             elif ir < n_rad_f:
#                 i_curve = (ir - n_rad_f // 2) / (n_rad_f//2)
#                 if ic <= n_point_cir // 2:
#                     rad_mod = rad * ((1 - i_curve) / np.cos(cir) + i_curve)
#                 else:
#                     rad_mod = rad * ((1 - i_curve) / np.sin(cir) + i_curve)
#                 points[pid, :] = [rad_mod * np.cos(cir), rad_mod * np.sin(cir), axi]
#             # circular
#             else:
#                 points[pid, :] = [rad * np.cos(cir), rad * np.sin(cir), axi]
#             cosy[pid, 3:] = points[pid, :]
#
#             # store surfaces
#             if ir == 0:
#                 surf_dict['inside'] += [pid]
#             if ir == n_rad_f:
#                 surf_dict['interface'] += [pid]
#             if ir == n_rad_f + n_rad_s:
#                 surf_dict['outside'] += [pid]
#             if ia == 0:
#                 surf_dict['start'] += [pid]
#             if ia == n_axi:
#                 surf_dict['end'] += [pid]
#             # cut-surfaces only exist for cylinder sections, not the whole cylinder
#             if n_sec > 1:
#                 if ic == 0:
#                     surf_dict['y_zero'] += [pid]
#                 if ic == n_point_cir - 1:
#                     surf_dict['x_zero'] += [pid]
#
#             # store fibers
#             fiber_dict['axi'] += [[0, 0, 1]]
#             fiber_dict['rad'] += [[-np.cos(cir), -np.sin(cir), 0]]
#             fiber_dict['cir'] += [[-np.sin(cir), np.cos(cir), 0]]
#
#             pid += 1

# cell vertices in (cir, rad, axi)
coords = [[0, 1, 0],
          [0, 1, 1],
          [1, 1, 1],
          [1, 1, 0],
          [0, 0, 0],
          [0, 0, 1],
          [1, 0, 1],
          [1, 0, 0]]

coords_cart = [[]]

# create cells
cells = []

# generate quadratic mesh
for ia in range(n_axi):
    for iy in range(n_quad - 1):
        for ix in range(n_quad - 1):
            ids = []
            for c in coords:
                ids += [(iy + c[0]) * n_quad + ix + c[1] + (ia + c[2]) * n_quad ** 2]
            cells += [ids]

# generate transition mesh
for ia in range(n_axi):
    for ic in range(n_cell_cir):
        ids = []
        for c in coords:
                if c[1] == 1:
                    # circular side
                    ids += [ic + c[0] + (n_axi + 1) * n_quad ** 2 + (ia + c[2]) * (n_tran + n_rad_s) * n_point_cir]
                else:
                    # quadratic side
                    if ic < n_cell_cir // 2:
                        ids += [n_quad - 1 + (ic + c[0]) * n_quad + (ia + c[2]) * n_quad ** 2]
                    else:
                        ids += [n_quad ** 2 - 1 + n_cell_cir // 2 - ic - c[0] + (ia + c[2]) * n_quad ** 2]
        cells += [ids]

# generate circular g&r mesh
for ia in range(n_axi):
    for ir in range(n_tran + n_rad_s - 1):
        for ic in range(n_cell_cir):
            ids = []
            for c in coords:
                ids += [(n_axi + 1) * n_quad ** 2 + (ic + c[0]) % n_point_cir + (ir + c[1]) * n_point_cir + (ia + c[2]) * (n_tran + n_rad_s) * n_point_cir]
            cells += [ids]
cells = np.array(cells)

cell_data = {'GlobalElementID': [np.arange(len(cells)) + 1]}
point_data = {'GlobalNodeID': np.arange(len(points)) + 1}
# point_data = {'GlobalNodeID': np.arange(len(points)) + 1,
#               'FIB_DIR': np.array(fiber_dict['rad']),
#               'varWallProps': cosy}
# for name, ids in surf_dict.items():
#     point_data['ids_' + name] = np.zeros(len(points))
#     point_data['ids_' + name][ids] = 1

# export mesh
mesh = meshio.Mesh(points, {'hexahedron': cells}, point_data=point_data, cell_data=cell_data)
# mesh = meshio.Mesh(points, {'hexahedron': cells})
fname = 'tube_' + str(n_rad_f) + '+' + str(n_rad_s) + 'x' + str(n_cir) + 'x' + str(n_axi) + '.vtu'
mesh.write(fname)

sys.exit(0)

# read volume mesh in vtk
vol = read_geo(fname).GetOutput()

# make output dirs
os.makedirs(f_out, exist_ok=True)
os.makedirs(os.path.join(f_out, 'mesh-surfaces'), exist_ok=True)

# map point data to cell data
p2c = vtk.vtkPointDataToCellData()
p2c.SetInputData(vol)
p2c.PassPointDataOn()
p2c.Update()
vol = p2c.GetOutput()

# extract surfaces
extract = vtk.vtkGeometryFilter()
extract.SetInputData(vol)
# extract.SetNonlinearSubdivisionLevel(0)
extract.Update()
surfaces = extract.GetOutput()

# threshold surfaces
for name in surf_dict.keys():
    # select only current surface
    thresh = vtk.vtkThreshold()
    thresh.SetInputData(surfaces)
    thresh.SetInputArrayToProcess(0, 0, 0, 0, 'ids_' + name)
    thresh.ThresholdBetween(1, 1)
    thresh.Update()

    # export to file
    fout = os.path.join(f_out, 'mesh-surfaces', name + '.vtp')
    write_geo(fout, extract_surface(thresh.GetOutput()))

extract_edges = vtk.vtkExtractEdges()
extract_edges.SetInputData(vol)
extract_edges.Update()

# export volume mesh
write_geo(os.path.join(f_out, 'mesh-complete.mesh.vtu'), vol)

# generate quadratic mesh
convert_quad = False
if convert_quad:
    # read quadratic mesh
    f_quad = '/home/pfaller/work/repos/svFSI_examples_fork/05-struct/03-GR/mesh_tube_quad/mesh-complete.mesh.vtu'
    vol = read_geo(f_quad).GetOutput()

    # calculate cell centers
    centers = vtk.vtkCellCenters()
    centers.SetInputData(vol)
    centers.Update()
    centers.VertexCellsOn()
    centers.CopyArraysOn()
    points = v2n(centers.GetOutput().GetPoints().GetData())

    # radial vector
    rad = points
    rad[:, 2] = 0
    rad = (rad.T / np.linalg.norm(rad, axis=1)).T

    arr = n2v(rad)
    arr.SetName('FIB_DIR')
    vol.GetCellData().AddArray(arr)

    write_geo(f_quad, vol)
    # write_geo('test.vtu', vol)
