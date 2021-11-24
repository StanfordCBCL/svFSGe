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
import shutil
import datetime
import os
import vtk

from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

from cylinder import generate_mesh

# from https://github.com/StanfordCBCL/DataCuration
sys.path.append('/home/pfaller/work/osmsc/curation_scripts')
sys.path.append('/Users/pfaller/work/repos/DataCuration')
from vtk_functions import read_geo, write_geo, get_points_cells, extract_surface, threshold
from simulation_io import map_meshes


exe_fluid = '/home/pfaller/work/repos/svFSI_clean/build/svFSI-build/bin/svFSI'
exe_solid = '/home/pfaller/work/repos/svFSI_direct/build/svFSI-build/bin/svFSI'
inp_fluid = 'steady_flow.inp'
inp_solid = 'gr_restart.inp'
inp_mesh = 'mesh.inp'
f_load = 'interface_pressure.vtp'
f_disp = 'interface_displacement.vtp'

interface_f = read_geo('mesh_tube_fsi/fluid/mesh-surfaces/interface.vtp').GetOutput()
interface_s = read_geo('mesh_tube_fsi/solid/mesh-surfaces/interface.vtp').GetOutput()

nodes_s = v2n(interface_s.GetPointData().GetArray('GlobalNodeID'))
nodes_f = v2n(interface_f.GetPointData().GetArray('GlobalNodeID'))

points_f = v2n(interface_f.GetPoints().GetData())
points_s = v2n(interface_s.GetPoints().GetData())

vol_f = read_geo('mesh_tube_fsi/fluid/mesh-complete.mesh.vtu').GetOutput()

# homeostatic pressure
p0 = 13.9868

# maximum number of time steps
nmax = 10

# maximum load increase
fmax = 1.5

# maximum asymmetry (one-way coupling only)
vmax = 0.1

def main_one_way():
    generate_mesh()
#    set_load(p0)
#    step_fluid(0)
    for i in list(range(0, nmax + 1)):
        f = 1.0 + np.max([i, 0]) / nmax * (fmax - 1.0)
        v = vmax * np.max([i, 0]) / nmax
        pressure_gradient(p0, f, v)
        step_gr()

    # time stamp
    ct = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-')

    # folder name
    f_out = 'gr_res_' + ct

    # archive
    shutil.move('1-procs', f_out)
    shutil.copyfile(f_load, os.path.join(f_out, f_load))
    shutil.copytree('mesh_tube_fsi', os.path.join(f_out, 'mesh_tube_fsi'))


def main_two_way():
#    generate_mesh()
    initialize_fluid()

    for i in list(range(0, nmax + 1)):
        # current load
        f = 1.0 + np.max([i, 0]) / nmax * (fmax - 1.0)

        # current index
        j = i + 1

        # step 0: set fluid distal pressure
        set_load(p0 * f)

        # step 1: steady-state fluid
        step_fluid()
        project_f2s(j)

        # step 2: solid g&r
        step_gr()
        project_s2f(j)

        # step 3: deform mesh
        step_mesh()
        project_disp(j)

    # time stamp
    ct = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-')

    # folder name
    f_out = 'fsg_res_' + ct

    # archive
    shutil.move('1-procs', f_out)
    shutil.copyfile('fluid', os.path.join(f_out, 'fluid'))
    shutil.copytree('mesh_tube_fsi', os.path.join(f_out, 'mesh_tube_fsi'))


def initialize():
    # write empty displacements
    displacement_f = np.zeros((vol_f.GetNumberOfPoints(), 3))
    array = n2v(displacement_f)
    array.SetName('Displacement')
    vol_f.GetPointData().AddArray(array)
    write_geo(f_disp, vol_f)


def initialize_fluid():
    shutil.copyfile('mesh_tube_fsi/fluid/mesh-complete.mesh.vtu', 'fluid/mesh.vtu')
    
def set_load(p):
    with open('steady_pressure.dat', 'w') as f:
        f.write('2 1\n')
        f.write('0.0 ' + str(p) + '\n')
        f.write('100.0 ' + str(p) + '\n')


def step_fluid():
    subprocess.run(shlex.split('mpirun -np 10 ' + exe_fluid + ' ' + inp_fluid))


def step_gr():
    subprocess.run(shlex.split(exe_solid + ' ' + inp_solid))


def step_mesh():
    subprocess.run(shlex.split('mpirun -np 10 ' + exe_fluid + ' ' + inp_mesh))


def pressure_gradient(p0, f, var):
    pressure_s = f * p0 * np.ones(interface_s.GetNumberOfPoints())

    # apply axial gradient
    if var > 0.0:
        points = v2n(interface_s.GetPoints().GetData())
        z_max = np.max(points[:, 2]) - np.min(points[:, 2])
        pressure_s *= 1.0 - var / 2.0 + var * points[:, 2] / z_max

    array = n2v(pressure_s)
    array.SetName('Pressure')
    interface_s.GetPointData().AddArray(array)
    write_geo(f_load, interface_s)


def project_f2s(i, f=None, p0=None):
    shutil.copyfile('10-procs/steady_030.vtu', 'fluid/steady_' + str(i) + '.vtu')

    res = read_geo('fluid/steady_' + str(i) + '.vtu').GetOutput()
    pressure_f = v2n(res.GetPointData().GetArray('Pressure'))
    if f is not None:
        # constant pressure
        if f < 1:
            pressure_s = p0 * np.ones(interface_s.GetNumberOfPoints())
        # pressure gradient from fluid
        else:
            pressure_f *= f
    else:
        tree = scipy.spatial.KDTree(points_s)
        _, i_fs = tree.query(points_f)
        pressure_s = pressure_f[nodes_f - 1][i_fs]

    array = n2v(pressure_s)
    array.SetName('Pressure')
    interface_s.GetPointData().AddArray(array)
    write_geo(f_load, interface_s)


def project_s2f(i):
    res = read_geo('1-procs/gr_' + str(i).zfill(3) + '.vtu').GetOutput()

    displacement_s = v2n(res.GetPointData().GetArray('Displacement'))

    tree = scipy.spatial.KDTree(points_f)
    _, i_sf = tree.query(points_s)

    displacement_f = displacement_s[nodes_s - 1][i_sf]
    array = n2v(displacement_f)
    array.SetName('Displacement')
    interface_f.GetPointData().AddArray(array)
    write_geo(f_disp, interface_f)

    # write general bc file
    with open('interface_displacement.dat', 'w') as f:
        f.write('3 2 ' + str(len(displacement_f)) + '\n')
        f.write('0.0\n')
        f.write('100.0\n')
        for n, d in zip(nodes_f, displacement_f):
            f.write(str(n) + '\n')
            for _ in range(2):
                for di in d:
                    f.write(str(di) + ' ')
                f.write('\n')

def project_disp(i):
    t_end = 1
    shutil.copyfile('10-procs/mesh_' + str(t_end).zfill(3) + '.vtu', 'fluid/mesh_' + str(i) + '.vtu')
    res = read_geo('fluid/mesh_' + str(i) + '.vtu').GetOutput()

    res.GetPointData().SetActiveVectors('Displacement')
    warp = vtk.vtkWarpVector()
    warp.SetInputData(res)
    warp.Update()
    write_geo('fluid/mesh.vtu', warp.GetOutput())


if __name__ == '__main__':
    main_two_way()
