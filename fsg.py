#!/usr/bin/env python
# coding=utf-8

import pdb
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.sparse.linalg
import sys
import subprocess
import shlex

from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

# from https://github.com/StanfordCBCL/DataCuration
sys.path.append('/home/pfaller/work/osmsc/curation_scripts')
sys.path.append('/Users/pfaller/work/repos/DataCuration')
from vtk_functions import read_geo, write_geo, get_points_cells, extract_surface, threshold
from simulation_io import map_meshes


exe_fluid = '/home/pfaller/work/repos/svFSI_clean/build/svFSI-build/bin/svFSI'
exe_solid = '/home/pfaller/work/repos/svFSI_direct/build/svFSI-build/bin/svFSI'
inp_fluid = 'steady_flow.inp'
inp_solid = 'gr.inp'
f_load = 'pressure_interface.vtp'

interface_f = read_geo('mesh_tube_fsi/fluid/mesh-surfaces/interface.vtp').GetOutput()
interface_s = read_geo('mesh_tube_fsi/solid/mesh-surfaces/interface.vtp').GetOutput()

def main():
    # step_fluid()
    # project_f2s()
    # step_solid()
    project_s2f(1)

def step_fluid():
    subprocess.run(shlex.split('mpirun -np 10 ' + exe_fluid + ' ' + inp_fluid))


def step_solid():
    subprocess.run(shlex.split(exe_solid + ' ' + inp_solid))


def project_f2s():
    res = read_geo('10-procs/steady_020.vtu').GetOutput()

    pressure_f = v2n(res.GetPointData().GetArray('Pressure'))
    nodes_f = v2n(interface_f.GetPointData().GetArray('GlobalNodeID'))

    points_f = v2n(interface_f.GetPoints().GetData())
    points_s = v2n(interface_s.GetPoints().GetData())

    tree = scipy.spatial.KDTree(points_s)
    _, i_fs = tree.query(points_f)

    # pdb.set_trace()
    array = n2v(pressure_f[nodes_f - 1][i_fs])
    array.SetName('Pressure')
    interface_s.GetPointData().AddArray(array)
    write_geo(f_load, interface_s)

def project_s2f(i):
    res = read_geo('1-procs/gr_' + str(i).zfill(3) + '.vtu').GetOutput()

    displacement_s = v2n(res.GetPointData().GetArray('Displacement'))
    pdb.set_trace()
    # # create point locator with old wave front
    # points = vtk.vtkPoints()
    # points.Initialize()
    # for i_old in pids_old:
    #     points.InsertNextPoint(geo.GetPoint(i_old))
    #
    # dataset = vtk.vtkPolyData()
    # dataset.SetPoints(points)


if __name__ == '__main__':
    main()
