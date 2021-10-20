#!/usr/bin/env python

import os
import sys
import vtk
import numpy as np
import pdb
from collections import OrderedDict

from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

sys.path.append('/home/pfaller/work/osmsc/curation_scripts')

from vtk_functions import read_geo, write_geo, get_points_cells, extract_surface, threshold


def convert_mesh(f_in, f_out, surf_dict):
    """
    :param f_in: input mesh
    :param f_out: folder for output meshes
    :param surf_dict: list surfaces: coordinate, treshhold value
    """
    vol = read_geo(f_in).GetOutput()

    # make output dirs
    os.makedirs(f_out, exist_ok=True)
    os.makedirs(os.path.join(f_out, 'mesh-surfaces'), exist_ok=True)

    # quad mesh -> linear mesh
    tess = vtk.vtkTessellatorFilter()
    tess.SetInputData(vol)
    tess.SetMaximumNumberOfSubdivisions(0)
    tess.Update()
    vol = tess.GetOutput()

    # add node and element ids
    arr = n2v(np.arange(vol.GetNumberOfPoints()) + 1)
    arr.SetName('GlobalNodeID')
    vol.GetPointData().AddArray(arr)
    arr = n2v(np.arange(vol.GetNumberOfCells()) + 1)
    arr.SetName('GlobalElementID')
    vol.GetCellData().AddArray(arr)

    # transform (x, y, z) to (r, phi, z)
    points_xyz, _ = get_points_cells(vol)
    points = np.zeros(points_xyz.shape)
    points[:, 0] = np.sqrt(points_xyz[:, 0]**2 + points_xyz[:, 1]**2)
    points[:, 1] = np.arctan2(points_xyz[:, 1], points_xyz[:, 0])
    points[:, 2] = points_xyz[:, 2]

    # create coordinate system
    fibers = OrderedDict()
    fiber_names = ['axi', 'cir', 'rad']
    for name in fiber_names:
        fibers[name] = np.zeros(points_xyz.shape)
    fibers['axi'][:, 2] = 1
    fibers['rad'][:, 0] = np.cos(points[:, 1])
    fibers['rad'][:, 1] = np.sin(points[:, 1])
    fibers['cir'] = np.cross(fibers['axi'], fibers['rad'])

    # # add fibers to geometry
    # for i, fiber in enumerate(fibers.values()):
    #     arr = n2v(fiber)
    #     arr.SetName('FIB_DIR' + str(i))
    #     vol.GetPointData().AddArray(arr)

    arr = n2v(fibers['rad'])
    arr.SetName('FIB_DIR')
    vol.GetPointData().AddArray(arr)

    # map point data to cell data
    p2c = vtk.vtkPointDataToCellData()
    p2c.SetInputData(vol)
    p2c.PassPointDataOn()
    p2c.Update()
    vol = p2c.GetOutput()

    # tolerance for finding surfaces
    eps = 1.0e-4

    # set node ids on surfaces
    for name, loc in surf_dict.items():
        # select nodes on surface
        node_ids = np.zeros(len(points))
        node_ids[np.where(np.abs(points[:, loc[0]] - loc[1]) <= eps)[0]] = 1

        # add array to geometry
        arr = n2v(node_ids)
        arr.SetName('ids_' + name)
        vol.GetPointData().AddArray(arr)

    # extract surfaces
    extract = vtk.vtkDataSetSurfaceFilter()
    extract.SetInputData(vol)
    extract.SetNonlinearSubdivisionLevel(0)
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

    write_geo(os.path.join(f_out, 'mesh-complete.mesh.vtu'), vol)
    # write_geo(os.path.join(out, 'mesh-complete.mesh.vtp'), surfaces)


if __name__ == '__main__':
    f_in_small = 'TAA-axi-4x200x1L-ht-t0.vtu'
    f_out_small = '/home/pfaller/work/repos/svFSI_examples_fork/05-struct/03-GR/mesh_small'
    surf_dict_small = {'start': [2, 0.0], 'end': [2, 0.03], 'inside': [0, 0.64678], 'outside': [0, 0.687]}

    f_in_large = 'TAA-axi-1x20x20Q-ht-t0.vtu'
    f_out_large = '/home/pfaller/work/repos/svFSI_examples_fork/05-struct/03-GR/mesh_large'
    surf_dict_large = {'start': [2, 0.0], 'end': [2, 15.0], 'inside': [0, 0.64678], 'outside': [0, 0.687]}

    convert_mesh(f_in_small, f_out_small, surf_dict_small)
    convert_mesh(f_in_large, f_out_large, surf_dict_large)
