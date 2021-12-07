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
import json

from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

from simulation import Simulation
from cylinder import generate_mesh

# from https://github.com/StanfordCBCL/DataCuration
sys.path.append('/home/pfaller/work/osmsc/curation_scripts')
sys.path.append('/Users/pfaller/work/repos/DataCuration')
from vtk_functions import read_geo, write_geo


class FSG(Simulation):
    def __init__(self, f_params=None):
        # simulation parameters
        Simulation.__init__(self, f_params)

        # make folders
        os.makedirs('fluid', exist_ok=True)

        # generate and initialize mesh
        generate_mesh()
        self.initialize_mesh()

        self.interface_f = read_geo('mesh_tube_fsi/fluid/mesh-surfaces/interface.vtp').GetOutput()
        self.interface_s = read_geo('mesh_tube_fsi/solid/mesh-surfaces/interface.vtp').GetOutput()

        self.nodes_s = v2n(self.interface_s.GetPointData().GetArray('GlobalNodeID'))
        self.nodes_f = v2n(self.interface_f.GetPointData().GetArray('GlobalNodeID'))

        self.points_f = v2n(self.interface_f.GetPoints().GetData())
        self.points_s = v2n(self.interface_s.GetPoints().GetData())

        # map fluid mesh to solid mesh
        tree = scipy.spatial.KDTree(self.points_s)
        _, self.i_fs = tree.query(self.points_f)

        # map solid mesh to fluid mesh
        tree = scipy.spatial.KDTree(self.points_f)
        _, self.i_sf = tree.query(self.points_s)

    def run(self, mode):
        try:
            if mode == '1-way':
                self.main_one_way()
            elif mode == '2-way':
                self.main_two_way()
            else:
                raise ValueError('Unknown mode ' + mode)
        # in case anything fails, still proceed with archiving the results
        except:
            pass

        # archive results
        self.archive(mode + '_res')

    def set_params(self):
        # define file paths
        self.p['exe_fluid'] = '/home/pfaller/work/repos/svFSI_clean/build/svFSI-build/bin/svFSI'
        self.p['exe_solid'] = '/home/pfaller/work/repos/svFSI_direct/build/svFSI-build/bin/svFSI'
        self.p['inp_fluid'] = 'steady_flow.inp'
        self.p['inp_solid'] = 'gr_restart.inp'
        self.p['inp_mesh'] = 'mesh.inp'
        self.p['f_load_pressure'] = 'interface_pressure.vtp'
        self.p['f_load_wss'] = 'interface_wss.vtp'
        self.p['f_disp'] = 'interface_displacement.vtp'

        # maximum number of time steps in fluid and solid simulations
        self.p['n_max_fluid'] = 10
        self.p['n_max_solid'] = 10

        # homeostatic pressure
        self.p['p0'] = 13.9868

        # fluid flow
        self.p['q0'] = 0.1

        # maximum number of G&R time steps (excluding prestress)
        self.p['nmax'] = 100

        # maximum load factor
        self.p['fmax'] = 1.0

    def validate_params(self):
        pass

    def main_one_way(self):
        for i in list(range(0, self.p['nmax'] + 1)):
            # calculate pressure load increase
            fp = 1.0 + np.max([i, 0]) / self.p['nmax'] * (self.p['fmax'] - 1.0)

            # step 1: steady-state fluid (analytical solution)
            self.initialize_fluid(self.p['p0'] * fp, self.p['q0'], 'interface')
            shutil.copyfile(self.p['f_load_pressure'], 'fluid/steady_' + str(i).zfill(3) + '.vtu')

            # step 2: solid g&r
            self.step_gr()

    def main_two_way(self):
        for i in list(range(0, self.p['nmax'] + 1)):
            # current load
            f = 1.0 + np.max([i, 0]) / self.p['nmax'] * (self.p['fmax'] - 1.0)

            # current index
            j = i + 1

            # step 0: set fluid distal pressure and initialize fluid solution
            self.set_pressure(self.p['p0'] * f)
            self.initialize_fluid(self.p['p0'] * f, self.p['q0'], 'vol')

            # step 1: steady-state fluid
            self.set_flow(self.p['q0'])
            self.step_fluid()
            self.project_f2s(j - 1)

            # step 2: solid g&r
            self.step_gr()
            self.project_s2f(j)

            # step 3: deform mesh
            self.step_mesh()
            self.project_disp(j)
    
    def archive(self, name):
        # time stamp
        ct = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-')

        # folder name
        f_out = name + '_' + ct
        self.p['f_out'] = f_out

        # move results
        shutil.move('1-procs', os.path.join(f_out, 'gr'))
        shutil.move('fluid', os.path.join(f_out, 'fluid'))
        shutil.move('mesh_tube_fsi', os.path.join(f_out, 'mesh_tube_fsi'))

        # save parameters
        self.save_params('fsg.json')

    def initialize_mesh(self):
        # initial fluid mesh (zero displacements)
        shutil.copyfile('mesh_tube_fsi/fluid/mesh-complete.mesh.vtu', 'fluid/mesh.vtu')

        # initial zero mesh displacements
        geo = read_geo('fluid/mesh.vtu').GetOutput()
        array = n2v(np.zeros((geo.GetNumberOfPoints(), 3)))
        array.SetName('Displacement')
        geo.GetPointData().AddArray(array)

        os.makedirs('10-procs', exist_ok=True)
        write_geo('10-procs/mesh_' + str(self.p['n_max_solid']).zfill(3) + '.vtu', geo)

    def initialize_fluid(self, p, q, mode):
        if mode == 'vol':
            geo = read_geo('fluid/mesh.vtu').GetOutput()
        elif mode == 'interface':
            geo = self.interface_s
        arrays = {}

        # mesh points
        points = v2n(geo.GetPoints().GetData())

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
        arrays['Pressure'] = p * np.ones(geo.GetNumberOfPoints()) + res * q * (1 - ax)

        # estimate quadratic flow profile
        arrays['Velocity'] = np.zeros((geo.GetNumberOfPoints(), 3))
        arrays['Velocity'][:, 2] = 4 * q / (rmax**2 * np.pi) * 2 * (1 - rad**2)

        # add arrays
        for n, a in arrays.items():
            array = n2v(a)
            array.SetName(n)
            geo.GetPointData().AddArray(array)

        # write to file
        if mode == 'vol':
            write_geo('fluid/mesh.vtu', geo)
        elif mode == 'interface':
            write_geo(self.p['f_load_pressure'], geo)
        
    def set_pressure(self, p):
        with open('steady_pressure.dat', 'w') as f:
            f.write('2 1\n')
            f.write('0.0 ' + str(p) + '\n')
            f.write('100.0 ' + str(p) + '\n')
        
    def set_flow(self, q):
        with open('steady_flow.dat', 'w') as f:
            f.write('2 1\n')
            f.write('0.0 ' + str(-q) + '\n')
            f.write('100.0 ' + str(-q) + '\n')

    def step_fluid(self):
        subprocess.run(shlex.split('mpirun -np 10 ' + self.p['exe_fluid'] + ' ' + self.p['inp_fluid']))

    def step_gr(self):
        subprocess.run(shlex.split(self.p['exe_solid'] + ' ' + self.p['inp_solid']))

    def step_mesh(self):
        subprocess.run(shlex.split('mpirun -np 10 ' + self.p['exe_fluid'] + ' ' + self.p['inp_mesh']))

    def project_f2s(self, i):
        src = '10-procs/steady_' + str(self.p['n_max_fluid']).zfill(3) + '.vtu'
        trg = 'fluid/steady_' + str(i).zfill(3) + '.vtu'
        shutil.copyfile(src, trg)

        # read fluid pressure
        res = read_geo(trg).GetOutput()
        for n in ['Pressure', 'WSS']:
            # read from fluid mesh
            res_f = v2n(res.GetPointData().GetArray(n))

            # map onto solid mesh
            res_s = res_f[self.nodes_f - 1][self.i_fs]

            # create VTK array
            array = n2v(res_s)
            array.SetName(n)
            self.interface_s.GetPointData().AddArray(array)

            # write to file
            write_geo(self.p['f_load_' + n.lower()], self.interface_s)

    def project_s2f(self, i):
        res = read_geo('1-procs/gr_' + str(i).zfill(3) + '.vtu').GetOutput()

        # read from solid mesh
        res_s = v2n(res.GetPointData().GetArray('Displacement'))

        # map onto fluid mesh
        res_f = res_s[self.nodes_s - 1][self.i_sf]

        # create VTK array
        array = n2v(res_f)
        array.SetName('Displacement')
        self.interface_f.GetPointData().AddArray(array)

        # write to file
        write_geo(self.p['f_disp'], self.interface_f)

        # write general bc file
        with open('interface_displacement.dat', 'w') as f:
            f.write('3 2 ' + str(len(res_f)) + '\n')
            f.write('0.0\n')
            f.write('1.0\n')
            for n, d in zip(self.nodes_f, res_f):
                f.write(str(n) + '\n')
                # for _ in range(2):
                f.write('0.0 0.0 0.0\n')
                for di in d:
                    f.write(str(di) + ' ')
                f.write('\n')

    def project_disp(self, i):
        src = '10-procs/mesh_' + str(self.p['n_max_solid']).zfill(3) + '.vtu'
        trg = 'fluid/mesh_' + str(i).zfill(3) + '.vtu'
        shutil.copyfile(src, trg)
        res = read_geo(trg).GetOutput()

        res.GetPointData().SetActiveVectors('Displacement')
        warp = vtk.vtkWarpVector()
        warp.SetInputData(res)
        warp.Update()
        write_geo('fluid/mesh.vtu', warp.GetOutput())


if __name__ == '__main__':
    fsg = FSG()
    # fsg.run('1-way')
    fsg.run('2-way')
