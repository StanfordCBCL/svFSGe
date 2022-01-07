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
import platform

if platform.system() == 'Darwin':
    usr = '/Users/pfaller/'
    sys.path.append('/Users/pfaller/work/repos/DataCuration')
else:
    usr = '/home/pfaller'
    sys.path.append('/home/pfaller/work/osmsc/curation_scripts')

import scipy.interpolate

from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

from simulation import Simulation
from cylinder import generate_mesh

# from https://github.com/StanfordCBCL/DataCuration
from vtk_functions import read_geo, write_geo, calculator, extract_surface, clean, threshold, get_all_arrays


class FSG(Simulation):
    def __init__(self, f_params=None):
        # simulation parameters
        Simulation.__init__(self, f_params)

        # make folders
        os.makedirs('fsg', exist_ok=True)

        # generate and initialize mesh
        generate_mesh()
        self.initialize_mesh()

        # intialize fluid/solid interface meshes
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
        except Exception as e:
            print(e)

        # archive results
        self.archive(mode + '_res')

    def set_params(self):
        # define file paths
        self.p['exe_fluid'] = usr + '/work/repos/svFSI_clean/build/svFSI-build/bin/svFSI'
        self.p['exe_solid'] = usr + '/work/repos/svFSI_direct/build/svFSI-build/bin/svFSI'
        self.p['inp_fluid'] = 'steady_flow.inp'
        self.p['inp_solid'] = 'gr_restart.inp'
        self.p['inp_mesh'] = 'mesh.inp'
        self.p['f_load_pressure'] = 'interface_pressure.vtp'
        self.p['f_load_wss'] = 'interface_wss.vtp'
        self.p['f_disp'] = 'interface_displacement.vtp'

        # number of processors
        self.p['n_procs_fluid'] = 10
        self.p['n_procs_mesh'] = 10

        # maximum number of time steps in fluid and solid simulations
        self.p['n_max_fluid'] = 30
        self.p['n_max_mesh'] = 10

        # homeostatic pressure
        self.p['p0'] = 13.9868

        # fluid flow
        self.p['q0'] = 0.0

        # maximum number of coupling iterations
        self.p['imax'] = 50

        # relaxation constant
        self.p['omega'] = 0.5

        # maximum number of G&R time steps (excluding prestress)
        self.p['nmax'] = 20

        # maximum load factor
        self.p['fmax'] = 1.5

    def validate_params(self):
        pass

    def main_one_way(self):
        for i in list(range(0, self.p['nmax'] + 1)):
            # calculate pressure load increase
            fp = 1.0 + np.max([i, 0]) / self.p['nmax'] * (self.p['fmax'] - 1.0)

            print('t ' + str(i) + '\tfp ' + str(fp))

            # step 1: steady-state fluid (analytical solution)
            self.initialize_fluid(self.p['p0'] * fp, self.p['q0'], 'interface')

            # step 2: solid g&r
            for j in range(self.p['imax']):
                k = i * self.p['imax'] + j
                self.step_gr()
                if i == 0 and j == 0:
                    i_res = [-1, k + 1]
                else:
                    # i_res = [(i - 1) * self.p['imax'] + j + 1, k + 1]
                    i_res = [k, k + 1]
                res = self.initialize_wss(i_res)
                print('  j\t' + str(j) + '\t' + '{:.2e}'.format(res))

            # copy files for logging
            src = '1-procs/gr_' + str((i + 1) * self.p['imax']).zfill(3) + '.vtu'
            trg = 'fsg/gr_' + str(i).zfill(3) + '.vtu'
            shutil.copyfile(src, trg)
            shutil.copyfile(self.p['f_load_pressure'], 'fsg/steady_' + str(i).zfill(3) + '.vtp')
            shutil.copyfile('fsg/solid.vtu', 'fsg/solid_' + str(i).zfill(3) + '.vtu')

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
            self.post_wss(j - 1)

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
        shutil.move('fsg', os.path.join(f_out, 'fsg'))
        shutil.move('mesh_tube_fsi', os.path.join(f_out, 'mesh_tube_fsi'))

        # save parameters
        self.save_params('fsg.json')

    def initialize_mesh(self):
        # initial fluid mesh (zero displacements)
        for f in ['fluid', 'solid']:
            shutil.copyfile('mesh_tube_fsi/' + f + '/mesh-complete.mesh.vtu', 'fsg/' + f + '.vtu')

        # initial zero mesh displacements
        geo = read_geo('fsg/fluid.vtu').GetOutput()
        array = n2v(np.zeros((geo.GetNumberOfPoints(), 3)))
        array.SetName('Displacement')
        geo.GetPointData().AddArray(array)

        os.makedirs(str(self.p['n_procs_mesh']) + '-procs', exist_ok=True)
        write_geo(str(self.p['n_procs_mesh']) + '-procs/mesh_' + str(self.p['n_max_mesh']).zfill(3) + '.vtu', geo)

    def initialize_fluid(self, p, q, mode):
        if mode == 'vol':
            geo = read_geo('fsg/fluid.vtu').GetOutput()
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
            write_geo('fsg/fluid.vtu', geo)
        elif mode == 'interface':
            write_geo(self.p['f_load_pressure'], geo)

    def initialize_wss(self, i_res):
        # read solid mesh
        geo = read_geo('mesh_tube_fsi/solid/mesh-complete.mesh.vtu').GetOutput()
        arrays, _ = get_all_arrays(geo)

        # read results
        disp_list = []
        for j in i_res:
            if j < 0:
                disp_list += [np.zeros((geo.GetNumberOfPoints(), 3))]
            else:
                fname = '1-procs/gr_' + str(j).zfill(3) + '.vtu'
                res = read_geo(fname).GetOutput()
                disp_list += [v2n(res.GetPointData().GetArray('Displacement'))]

        # relax displacement increment
        disp = n2v((1.0 - self.p['omega']) * disp_list[0] + self.p['omega'] * disp_list[1])
        disp.SetName('Displacement')
        res.GetPointData().AddArray(disp)

        # warp mesh by displacements
        res.GetPointData().SetActiveVectors('Displacement')
        warp = vtk.vtkWarpVector()
        warp.SetInputData(res)
        warp.Update()
        res = warp.GetOutput()

        # mesh points
        points = v2n(res.GetPoints().GetData())

        # radial coordinate
        rad = np.sqrt(points[:, 0]**2 + points[:, 1]**2)

        # wss (assume Q = 1.0 = const)
        arrays['varWallProps'][:, 6] = 4 * 0.04 * 1.0 / np.pi / np.min(rad)**3
        # arrays['varWallProps'][:, 6] = 4 * 0.04 * 1.0 / np.pi / rad**3

        # create VTK array
        array = n2v(arrays['varWallProps'])
        array.SetName('varWallProps')
        geo.GetPointData().AddArray(array)

        # write to file
        write_geo('fsg/solid.vtu', geo)

        return np.linalg.norm(disp_list[1] - disp_list[0])

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
        subprocess.run(shlex.split('mpirun -np ' + str(self.p['n_procs_fluid']) + ' ' + self.p['exe_fluid'] + ' ' + self.p['inp_fluid']))

    def step_gr(self):
        subprocess.run(shlex.split(self.p['exe_solid'] + ' ' + self.p['inp_solid']),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def step_mesh(self):
        subprocess.run(shlex.split('mpirun -np ' + str(self.p['n_procs_mesh']) + ' ' + self.p['exe_fluid'] + ' ' + self.p['inp_mesh']))

    def project_f2s(self, i):
        src = str(self.p['n_procs_fluid']) + '-procs/steady_' + str(self.p['n_max_fluid']).zfill(3) + '.vtu'
        trg = 'fsg/steady_' + str(i).zfill(3) + '.vtu'
        shutil.copyfile(src, trg)

        # read fluid pressure
        res = read_geo(trg).GetOutput()
        for n in ['Pressure']: #, 'WSS'
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
        # read from solid mesh
        res = read_geo('1-procs/gr_' + str(i).zfill(3) + '.vtu').GetOutput()
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
        src = str(self.p['n_procs_mesh']) + '-procs/mesh_' + str(self.p['n_max_mesh']).zfill(3) + '.vtu'
        trg = 'fsg/mesh_' + str(i).zfill(3) + '.vtu'
        shutil.copyfile(src, trg)

        # warp mesh by displacements
        res = read_geo(trg).GetOutput()
        res.GetPointData().SetActiveVectors('Displacement')
        warp = vtk.vtkWarpVector()
        warp.SetInputData(res)
        warp.Update()
        write_geo('fsg/fluid.vtu', warp.GetOutput())

    def post_wss(self, i):
        # read solid mesh
        solid = read_geo('mesh_tube_fsi/solid/mesh-complete.mesh.vtu').GetOutput()

        # read fluid pressure
        trg = 'fsg/steady_' + str(i).zfill(3) + '.vtu'
        res = read_geo(trg)

        # calculate WSS
        calc1 = calculator(res, 'mag(Velocity)', ['Velocity'], 'u')
        grad = vtk.vtkGradientFilter()
        grad.SetInputData(calc1.GetOutput())
        grad.SetInputScalars(0, 'u')
        grad.Update()
        surf = extract_surface(grad.GetOutput())
        norm = vtk.vtkPolyDataNormals()
        norm.SetInputData(surf)
        norm.Update()
        calc2 = calculator(norm, '0.04*abs(Gradients.Normals)', ['Gradients', 'Normals'], 'WSS')
        p2c = vtk.vtkPointDataToCellData()
        p2c.SetInputData(calc2.GetOutput())
        p2c.Update()
        cl = clean(p2c.GetOutput())
        thr = threshold(cl, 0.0, 'u')
        c2p = vtk.vtkCellDataToPointData()
        c2p.SetInputData(thr.GetOutput())
        c2p.Update()
        wss_f = v2n(c2p.GetOutput().GetPointData().GetArray('WSS'))

        # get wall properties
        props = v2n(solid.GetPointData().GetArray('varWallProps'))

        # map fluid mesh to solid mesh
        tree = scipy.spatial.KDTree(v2n(c2p.GetOutput().GetPoints().GetData()))
        _, i_ws = tree.query(self.points_s)
        wss_is = wss_f[i_ws]

        # write wss to solid mesh
        props[:, 6] = scipy.interpolate.griddata(props[self.nodes_s - 1][:, 1:3], wss_is, (props[:, 1], props[:, 2]))

        # create VTK array
        array = n2v(props)
        array.SetName('varWallProps')
        solid.GetPointData().AddArray(array)

        write_geo('fsg/wss_' + str(i).zfill(3) + '.vtu', solid)
        write_geo('fsg/solid.vtu', solid)


if __name__ == '__main__':
    fsg = FSG()
    fsg.run('1-way')
    # fsg.run('2-way')
