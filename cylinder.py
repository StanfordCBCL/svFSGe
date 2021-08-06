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
from vtk_functions import read_geo, write_geo, get_points_cells, extract_surface, threshold

# output folder
f_out = '/home/pfaller/work/repos/svFSI_examples_fork/05-struct/03-GR/mesh_tube'

# cylinder size
r_inner = 0.64678
r_outer = 0.687
height = 0.03
# height = 15.0

# number of cells in each dimension
# n_rad = 1
# n_cir = 4
# n_axi = 1
n_rad = 4
n_cir = 200
n_axi = 1
# n_rad = 1
# n_cir = 20
# n_axi = 20

n_cir_q = n_cir // 4
assert n_cir_q == n_cir / 4, 'radius must be divisible by four'

# create points
n_points = (n_axi + 1) * (n_rad + 1) * n_cir
pid = 0
points = np.zeros((n_points, 3))
cosy = np.zeros((n_points, 3))
line_dict = defaultdict(list)
surf_dict = defaultdict(list)
fiber_dict = defaultdict(list)
for ia in range(n_axi + 1):
    for ir in range(n_rad + 1):
        for ic in range(n_cir):
            # cylindrical coordinate system
            axi = height * ia / n_axi
            cir = 2 * np.pi * ic / n_cir
            rad = r_inner + (r_outer - r_inner) * ir / n_rad

            # store normalized coordinates
            cosy[pid, 0] = ia / n_axi
            cosy[pid, 1] = ic / n_cir
            cosy[pid, 2] = rad # / r_outer

            # cartesian coordinate system
            points[pid, :] = [rad * np.cos(cir), rad * np.sin(cir), axi]

            # store surfaces
            if ir == 0:
                surf_dict['inside'] += [pid]
                if ic == 0 or ic == 2 * n_cir_q:
                    line_dict['y_zero'] += [pid]
                if ic == n_cir_q or ic == 3 * n_cir_q:
                    line_dict['x_zero'] += [pid]
            if ir == n_rad:
                surf_dict['outside'] += [pid]
            if ia == 0:
                surf_dict['start'] += [pid]
            if ia == n_axi:
                surf_dict['end'] += [pid]

            # store fibers
            fiber_dict['axi'] += [[0, 0, 1]]
            fiber_dict['rad'] += [[np.cos(cir), np.sin(cir), 0]]
            fiber_dict['cir'] += [[np.sin(cir), -np.cos(cir), 0]]

            pid += 1

# cell vertices in (cir, rad, axi)
coords = [
          [0, 1, 0],
          [0, 1, 1],
          [1, 1, 1],
          [1, 1, 0],
          [0, 0, 0],
          [0, 0, 1],
          [1, 0, 1],
          [1, 0, 0]
]

# create cells
cells = []
for ia in range(n_axi):
    for ir in range(n_rad):
        for ic in range(n_cir):
            ids = []
            for c in coords:
                ids += [(ic + c[0]) % n_cir + (ir + c[1]) * n_cir + (ia + c[2]) * (n_rad + 1) * n_cir]
            cells += [ids]
cells = np.array(cells)

cell_data = {'GlobalElementID': [np.arange(len(cells)) + 1]}
point_data = {'GlobalNodeID': np.arange(len(points)) + 1,
              'FIB_DIR': np.array(fiber_dict['rad']),
              'varWallProps': cosy[:, 2]}
for name, ids in surf_dict.items():
    point_data['ids_' + name] = np.zeros(len(points))
    point_data['ids_' + name][ids] = 1
for name, ids in line_dict.items():
    point_data['ids_' + name] = np.zeros(len(points))
    point_data['ids_' + name][ids] = 1

# export mesh
mesh = meshio.Mesh(points, {'hexahedron': cells}, point_data=point_data, cell_data=cell_data)
fname = 'tube_' + str(n_rad) + 'x' + str(n_cir) + 'x' + str(n_axi) + '.vtu'
mesh.write(fname)

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

# threshold lines
for name in line_dict.keys():
    # select only current surface
    thresh = vtk.vtkThreshold()
    thresh.SetInputData(extract_edges.GetOutput())
    thresh.SetInputArrayToProcess(0, 0, 0, 0, 'ids_' + name)
    thresh.ThresholdBetween(1, 1)
    thresh.Update()

    # export surface mesh to file
    fout = os.path.join(f_out, 'mesh-surfaces', name + '.vtp')
    write_geo(fout, extract_surface(thresh.GetOutput()))

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
