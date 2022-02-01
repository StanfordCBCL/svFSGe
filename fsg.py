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
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

from simulation import Simulation
from cylinder import generate_mesh

# from https://github.com/StanfordCBCL/DataCuration
from vtk_functions import read_geo, write_geo, calculator, extract_surface, clean, threshold, get_all_arrays


class FSG(Simulation):
    def __init__(self, mode, f_params=None):
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

        # remove old output folders
        for f in ['1-procs', 'fsg']:
            if os.path.exists(f) and os.path.isdir(f):
                shutil.rmtree(f)

        # time stamp
        ct = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-')

        # folder name
        self.p['f_out'] = mode + '_res_' + ct

        # create output folders
        os.makedirs(self.p['f_out'])
        os.makedirs('fsg')
        os.makedirs(os.path.join(self.p['f_out'], 'fsg'))

        # logging
        self.wss = [[]]
        self.rad = [[]]

    def run(self):
        self.main_one_way()
        sys.exit(0)
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
        self.archive()

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

        # coupling tolerance
        self.p['coup_tol'] = 1.0e-3

        # maximum number of coupling iterations
        self.p['coup_imax'] = 100

        # relaxation constant
        self.p['coup_damp'] = 15/16

        # maximum number of G&R time steps (excluding prestress)
        self.p['nmax'] = 10

        # maximum load factor
        self.p['fmax'] = 1.5

    def validate_params(self):
        pass

    def main_one_way(self):
        # generate load vector
        # todo: do globally
        p_vec = np.linspace(1.0, self.p['fmax'], self.p['nmax'] + 1)

        # initialize fluid
        self.initialize_fluid(self.p['p0'], self.p['q0'], 'interface')

        # wss in reference configuration
        self.initialize_wss(0, False)

        # loop load steps
        i = 0
        load_vec = []
        for t in range(self.p['nmax'] + 1):
            # pick next load factor
            fp = p_vec[t]

            # pressure update
            self.initialize_fluid(self.p['p0'] * fp, self.p['q0'], 'interface')

            print('==== t ' + str(t) + ' ==== fp ' + '{:.2f}'.format(fp) + ' ' + '=' * 40)

            # loop sub-iterations
            for n in range(self.p['coup_imax']):
                # count total iterations (load + sub-iterations)
                i += 1
                if n==0:
                    load_vec.append([])
                load_vec[-1].append(fp)

                # solid update
                self.step_gr()

                # wss update
                res = self.initialize_wss(i, n==0)

                # logging
                str_i = str(i).zfill(3)
                src = ['fsg/solid.vtu',
                       '1-procs/gr_' + str_i + '.vtu',
                       self.p['f_load_pressure']]
                trg = ['fsg/solid_' + str_i + '.vtu',
                       'fsg/gr_' + str_i + '.vtu',
                       'fsg/steady_' + str_i + '.vtp']
                for sr, tg in zip(src, trg):
                    shutil.copyfile(sr, os.path.join(self.p['f_out'], tg))
                print('i ' + str(i) + ' \tn ' + str(n) + '\terr ' + '{:.2e}'.format(res))

                # check if coupling converged
                if res < self.p['coup_tol']:
                    break
            else:
                print('\tcoupling unconverged')

        self.plot_wss(load_vec)

    def plot_wss(self, load_vec):
        fig, ax = plt.subplots(1, 2 , figsize=(20, 10), dpi=300)

        labels = ['wss', 'rad']
        data = [self.wss, self.rad]

        for i, (d, l) in enumerate(zip(data, labels)):
            ax[i].set_xlabel('sub-iterations $n$')
            ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))
            ax[i].set_ylabel(l)
            for j in d:
                ax[i].plot(j, linestyle='-', marker='o')

        # fig.tight_layout()
        plt.show()


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
    
    def archive(self):
        # move results
        # shutil.move('1-procs', os.path.join(self.p['f_out'], 'gr'))
        # shutil.move('fsg', os.path.join(f_out, 'fsg'))
        shutil.move('mesh_tube_fsi', os.path.join(self.p['f_out'], 'mesh_tube_fsi'))

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

        # estimate Poiseuille resistance
        res = 8 * 0.04 * amax / np.pi / rmax**4

        # estimate linear pressure gradient
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

    def initialize_wss(self, i_res, ini):
        # read solid mesh
        geo = read_geo('mesh_tube_fsi/solid/mesh-complete.mesh.vtu').GetOutput()
        arrays, _ = get_all_arrays(geo)

        # read results
        if i_res == 0:
            # get reference configuration
            points = v2n(geo.GetPoints().GetData())
        else:
            # get displacement from g&r solution
            fname = '1-procs/gr_' + str(i_res).zfill(3) + '.vtu'
            res = read_geo(fname).GetOutput()

            # warp mesh by displacements
            res.GetPointData().SetActiveVectors('Displacement')
            warp = vtk.vtkWarpVector()
            warp.SetInputData(res)
            warp.Update()

            # mesh points
            points = v2n(warp.GetOutput().GetPoints().GetData())

        # get radial coordinate
        rad_new = np.sqrt(points[:, 0]**2 + points[:, 1]**2)

        # wss (assume Q = 1.0 = const)
        wss_new = 4 * 0.04 * 1.0 / np.pi / np.min(rad)**3

        if i_res == 0:
            # prestress: initialzie wss from reference configuration
            wss_relax = wss_new
        else:
            # if ini: converged wss of last load step. else: wss of last sub-iteration
            wss_old = self.wss[-1][-1]
            if ini and len(self.wss) > 2:
                # linearly extrapolate new wss from previous load increment
                wss_old_old = self.wss[-2][-1]
                wss_relax = 2.0 * wss_old - wss_old_old
            else:
                # damp with wss from previous iteration
                wss_relax = (1.0 - self.p['coup_damp']) * wss_new + self.p['coup_damp'] * wss_old

            # start a new sub-list for new load step
            if ini:
                self.wss.append([])
                self.rad.append([])

        # append current (damped) wss
        self.wss[-1].append(wss_relax)
        self.rad[-1].append(np.min(rad))

        # store wss in geometry
        arrays['varWallProps'][:, 6] = wss_relax

        # create VTK array
        array = n2v(arrays['varWallProps'])
        array.SetName('varWallProps')
        geo.GetPointData().AddArray(array)

        # write to file
        write_geo('fsg/solid.vtu', geo)

        # calculate wss norm
        return abs(wss_relax / wss_new - 1.0)

    def coup_relax(self):
        pass

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
    mode = '1-way'
    fsg = FSG(mode)
    fsg.run()
    # fsg.run('2-way')
