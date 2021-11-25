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
from vtk_functions import read_geo, write_geo


class FSG():
    def __init__(self):
        self.p = {}

        # define file paths
        self.p['exe_fluid'] = '/home/pfaller/work/repos/svFSI_clean/build/svFSI-build/bin/svFSI'
        self.p['exe_solid'] = '/home/pfaller/work/repos/svFSI_direct/build/svFSI-build/bin/svFSI'
        self.p['inp_fluid'] = 'steady_flow.inp'
        self.p['inp_solid'] = 'gr_restart.inp'
        self.p['inp_mesh'] = 'mesh.inp'
        self.p['f_load'] = 'interface_pressure.vtp'
        self.p['f_disp'] = 'interface_displacement.vtp'

        # homeostatic pressure
        self.p['p0'] = 13.9868

        # fluid flow
        # todo: set in python
        self.p['q0'] = 0.01

        # maximum number of time steps
        self.p['nmax'] = 10

        # maximum load increase
        self.p['fmax'] = 1.5

        # maximum asymmetry (one-way coupling only)
        # todo: replace by q*resistance
        self.p['vmax'] = 0.1

        # generate and initialize mesh
        generate_mesh()
        self.initialize_mesh()

        self.interface_f = read_geo('mesh_tube_fsi/fluid/mesh-surfaces/interface.vtp').GetOutput()
        self.interface_s = read_geo('mesh_tube_fsi/solid/mesh-surfaces/interface.vtp').GetOutput()

        self.nodes_s = v2n(self.interface_s.GetPointData().GetArray('GlobalNodeID'))
        self.nodes_f = v2n(self.interface_f.GetPointData().GetArray('GlobalNodeID'))

        self.points_f = v2n(self.interface_f.GetPoints().GetData())
        self.points_s = v2n(self.interface_s.GetPoints().GetData())

    def main_one_way(self):
        for i in list(range(0, self.p['nmax'] + 1)):
            f = 1.0 + np.max([i, 0]) / self.p['nmax'] * (self.p['fmax'] - 1.0)
            v = self.p['vmax'] * np.max([i, 0]) / self.p['nmax']
            self.pressure_gradient(f, v)
            self.step_gr()

        # time stamp
        ct = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-')

        # folder name
        f_out = 'gr_res_' + ct

        # archive
        shutil.move('1-procs', f_out)
        shutil.copyfile(self.p['f_load'], os.path.join(f_out, self.p['f_load']))
        shutil.copytree('mesh_tube_fsi', os.path.join(f_out, 'mesh_tube_fsi'))

    def main_two_way(self):
        for i in list(range(0, self.p['nmax'] + 1)):
            # current load
            f = 1.0 + np.max([i, 0]) / self.p['nmax'] * (self.p['fmax'] - 1.0)

            # current index
            j = i + 1

            # step 0: set fluid distal pressure and initialize fluid solution
            self.set_load(self.p['p0'] * f)
            self.initialize_fluid(self.p['p0'] * f, self.p['q0'])

            # step 1: steady-state fluid
            self.step_fluid()
            self.project_f2s(j)

            # step 2: solid g&r
            self.step_gr()
            self.project_s2f(j)

            # step 3: deform mesh
            self.step_mesh()
            self.project_disp(j)

        # time stamp
        ct = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-')

        # folder name
        f_out = 'fsg_res_' + ct

        # archive
        shutil.move('1-procs', os.path.join(f_out, 'gr'))
        shutil.copytree('fluid', os.path.join(f_out, 'fluid'))
        shutil.copytree('mesh_tube_fsi', os.path.join(f_out, 'mesh_tube_fsi'))

    def initialize_mesh(self):
        shutil.copyfile('mesh_tube_fsi/fluid/mesh-complete.mesh.vtu', 'fluid/mesh.vtu')

    def initialize_fluid(self, p, q):
        vol = read_geo('fluid/mesh.vtu').GetOutput()
        arrays = {}

        # mesh points
        points =  v2n(vol.GetPoints().GetData())

        # normalized axial coordinate
        ax = points[:, 2].copy()
        amax = np.max(ax)
        ax /= amax

        # normalized radial coordinate
        rad = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        rmax = np.max(rad)
        rad /= rmax

        # estimate linear pressure gradient
        res = 8 * 0.04 * amax / np.pi / rmax**4
        arrays['Pressure'] = p * np.ones(vol.GetNumberOfPoints()) + res * q * (1 - ax)

        # estimate quadratic flow profile
        arrays['Velocity'] = np.zeros((vol.GetNumberOfPoints(), 3))
        arrays['Velocity'][:, 2] = 4 * q / (rmax**2 * np.pi) * 2 * (1 - rad**2)

        # add arrays and write to file
        for n, a in arrays.items():
            array = n2v(a)
            array.SetName(n)
            vol.GetPointData().AddArray(array)
        write_geo('fluid/mesh.vtu', vol)
        
    def set_load(self, p):
        with open('steady_pressure.dat', 'w') as f:
            f.write('2 1\n')
            f.write('0.0 ' + str(p) + '\n')
            f.write('100.0 ' + str(p) + '\n')

    def step_fluid(self):
        subprocess.run(shlex.split('mpirun -np 10 ' + self.p['exe_fluid'] + ' ' + self.p['inp_fluid']))

    def step_gr(self):
        subprocess.run(shlex.split(self.p['exe_solid'] + ' ' + self.p['inp_solid']))

    def step_mesh(self):
        subprocess.run(shlex.split('mpirun -np 10 ' + self.p['exe_fluid'] + ' ' + self.p['inp_mesh']))

    def pressure_gradient(self, f, var):
        # todo: replace by q*resistance
        pressure_s = f * self.p['p0'] * np.ones(self.interface_s.GetNumberOfPoints())

        # apply axial gradient
        if var > 0.0:
            points = v2n(self.interface_s.GetPoints().GetData())
            z_max = np.max(points[:, 2]) - np.min(points[:, 2])
            pressure_s *= 1.0 - var / 2.0 + var * points[:, 2] / z_max

        array = n2v(pressure_s)
        array.SetName('Pressure')
        self.interface_s.GetPointData().AddArray(array)
        write_geo(self.p['f_load'], self.interface_s)

    def project_f2s(self, i, f=None, p0=None):
        t_end = 30
        shutil.copyfile('10-procs/steady_' + str(t_end).zfill(3) + '.vtu', 'fluid/steady_' + str(i) + '.vtu')

        res = read_geo('fluid/steady_' + str(i) + '.vtu').GetOutput()
        pressure_f = v2n(res.GetPointData().GetArray('Pressure'))
        if f is not None:
            # constant pressure
            if f < 1:
                pressure_s = p0 * np.ones(self.interface_s.GetNumberOfPoints())
            # pressure gradient from fluid
            else:
                pressure_f *= f
        else:
            tree = scipy.spatial.KDTree(self.points_s)
            _, i_fs = tree.query(self.points_f)
            pressure_s = pressure_f[self.nodes_f - 1][i_fs]

        array = n2v(pressure_s)
        array.SetName('Pressure')
        self.interface_s.GetPointData().AddArray(array)
        write_geo(self.p['f_load'], self.interface_s)

    def project_s2f(self, i):
        res = read_geo('1-procs/gr_' + str(i).zfill(3) + '.vtu').GetOutput()

        displacement_s = v2n(res.GetPointData().GetArray('Displacement'))

        tree = scipy.spatial.KDTree(self.points_f)
        _, i_sf = tree.query(self.points_s)

        displacement_f = displacement_s[self.nodes_s - 1][i_sf]
        array = n2v(displacement_f)
        array.SetName('Displacement')
        self.interface_f.GetPointData().AddArray(array)
        write_geo(self.p['f_disp'], self.interface_f)

        # write general bc file
        with open('interface_displacement.dat', 'w') as f:
            f.write('3 2 ' + str(len(displacement_f)) + '\n')
            f.write('0.0\n')
            f.write('100.0\n')
            for n, d in zip(self.nodes_f, displacement_f):
                f.write(str(n) + '\n')
                for _ in range(2):
                    for di in d:
                        f.write(str(di) + ' ')
                    f.write('\n')

    def project_disp(self, i):
        t_end = 1
        shutil.copyfile('10-procs/mesh_' + str(t_end).zfill(3) + '.vtu', 'fluid/mesh_' + str(i) + '.vtu')
        res = read_geo('fluid/mesh_' + str(i) + '.vtu').GetOutput()

        res.GetPointData().SetActiveVectors('Displacement')
        warp = vtk.vtkWarpVector()
        warp.SetInputData(res)
        warp.Update()
        write_geo('fluid/mesh.vtu', warp.GetOutput())


if __name__ == '__main__':
    fsg = FSG()
    fsg.main_two_way()
