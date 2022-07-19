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
import glob
from collections import defaultdict

import scipy.interpolate
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

from simulation import Simulation
from cylinder import generate_mesh


if platform.system() == 'Darwin':
    usr = '/Users/pfaller/'
elif platform.system() == 'Linux':
    usr = '/home/pfaller/'

# from https://github.com/StanfordCBCL/DataCuration
sys.path.append(os.path.join(usr, 'work/repos/DataCuration'))
from vtk_functions import read_geo, write_geo, calculator, extract_surface, clean, threshold, get_all_arrays


class svFSI(Simulation):
    """
    svFSI base class (handles simulation runs)
    """

    def __init__(self, fluid, f_params=None):
        # simulation parameters
        Simulation.__init__(self, f_params)

        # fluid mode (fsi or poiseuille)
        if fluid not in ['fsi', 'poiseuille']:
            raise ValueError('Unknown fluid option ' + fluid)
        self.p['fluid'] = fluid

        # remove old output folders
        self.fields = ['fluid', 'solid', 'mesh', 'fsi']
        for f in self.fields + [self.p['root']]:
            if f is not self.p['root']:
                f = str(self.p['n_procs'][f]) + '-procs'
            if os.path.exists(f) and os.path.isdir(f):
                shutil.rmtree(f)

        # make folders
        os.makedirs(self.p['root'])
        os.makedirs(os.path.join(self.p['root'], 'converged'))

        # generate and initialize mesh
        generate_mesh()
        self.initialize_mesh()

        # intialize fluid/solid interface meshes
        self.tube = read_geo(glob.glob('mesh_tube_fsi/tube_*.vtu')[0]).GetOutput()
        self.interface_f = read_geo('mesh_tube_fsi/fluid/mesh-surfaces/interface.vtp').GetOutput()
        self.interface_s = read_geo('mesh_tube_fsi/solid/mesh-surfaces/interface.vtp').GetOutput()
        self.vol_f = read_geo('mesh_tube_fsi/fluid/mesh-complete.mesh.vtu').GetOutput()
        self.vol_s = read_geo('mesh_tube_fsi/solid/mesh-complete.mesh.vtu').GetOutput()

        self.nodes_f = v2n(self.interface_f.GetPointData().GetArray('GlobalNodeID'))
        self.nodes_s = v2n(self.interface_s.GetPointData().GetArray('GlobalNodeID'))

        self.points_tube = v2n(self.tube.GetPoints().GetData())
        self.points_f = v2n(self.interface_f.GetPoints().GetData())
        self.points_s = v2n(self.interface_s.GetPoints().GetData())
        self.points_vol_f = v2n(self.vol_f.GetPoints().GetData())
        self.points_vol_s = v2n(self.vol_s.GetPoints().GetData())

        # map nodes
        self.i_fs = map_ids(self.points_f, self.points_s)
        self.i_sf = map_ids(self.points_s, self.points_f)
        self.i_vol_f = map_ids(self.points_vol_f, self.points_tube)
        self.i_vol_s = map_ids(self.points_vol_s, self.points_tube)

        # time stamp
        ct = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-')

        # output folder name
        self.p['f_out'] = fluid + '_res_' + ct

        # create output folder
        os.makedirs(self.p['f_out'])

        # logging
        self.log = defaultdict(list)
        self.err = defaultdict(list)

        # current interface solution vector
        self.sol = {'disp': np.zeros(self.points_f.shape),
                    'wss': np.zeros(len(self.points_f))}

        # generate load vector
        self.p_vec = np.linspace(1.0, self.p['fmax'], self.p['nmax'] + 1)

    def validate_params(self):
        pass

    def set_bc_pressure(self, p):
        with open('steady_pressure.dat', 'w') as f:
            f.write('2 1\n')
            f.write('0.0 ' + str(p) + '\n')
            f.write('100.0 ' + str(p) + '\n')
        if self.p['fluid'] == 'poiseuille':
            self.sol['press'] = p * np.ones(self.interface_s.GetNumberOfPoints())

    def set_bc_flow(self, q):
        with open('steady_flow.dat', 'w') as f:
            f.write('2 1\n')
            f.write('0.0 ' + str(-q) + '\n')
            f.write('100.0 ' + str(-q) + '\n')

    def step(self, name, i):
        if name not in self.fields:
            raise ValueError('Unknown step option ' + name)
        exe = 'mpirun -np'
        for k in ['n_procs', 'exe', 'inp']:
            exe += ' ' + str(self.p[k][name])
        with open(os.path.join(self.p['root'], name + '_' + str(i).zfill(3) + '.log'), 'w') as f:
            if self.p['debug']:
                print(exe)
                subprocess.run(shlex.split(exe))
            else:
                subprocess.run(shlex.split(exe), stdout=f, stderr=subprocess.DEVNULL)


class FSG(svFSI):
    """
    FSG-specific stuff
    """

    def __init__(self, fluid, f_params=None):
        # svFSI simulations
        svFSI.__init__(self, fluid, f_params)

    def run(self):
        # run simulation
        self.main()

        # plot convergence
        try:
            self.plot_convergence()
        except:
            pass

        # archive results
        self.archive()

    def set_params(self):
        # debug mode?
        self.p['debug'] = False

        # simulation folder
        self.p['root'] = 'partitioned'

        # define file paths
        self.p['exe'] = {'fluid': usr + '/work/repos/svFSI_clean/build/svFSI-build/bin/svFSI',
                         'solid': usr + '/work/repos/svFSI_fork/build/svFSI-build/bin/svFSI'}
        self.p['exe']['mesh'] = self.p['exe']['fluid']
        self.p['exe']['fsi'] = self.p['exe']['solid']

        # input files
        self.p['inp'] = {'fluid': 'steady_flow.inp', 'solid': 'gr_restart.inp', 'mesh': 'mesh.inp', 'fsi': 'fsi.inp'}

        # number of processors
        self.p['n_procs'] = {'solid': 1, 'fluid': 10, 'mesh': 10, 'fsi': 10}

        # maximum number of time steps
        self.p['n_max'] = {'fluid': 10, 'mesh': 10, 'fsi': 100}

        # interface loads
        self.p['f_load_pressure'] = 'interface_pressure.vtp'
        self.p['f_load_wss'] = 'interface_wss.vtp'
        self.p['f_disp'] = 'interface_displacement'

        # homeostatic pressure
        self.p['p0'] = 13.9868

        # fluid flow
        self.p['q0'] = 0.1

        # coupling tolerance
        self.p['coup_tol'] = 1.0e-3

        # maximum number of coupling iterations
        self.p['coup_imax'] = 100

        # relaxation constant
        exp = 2
        self.p['coup_omega0'] = 1/2**exp
        self.p['coup_omega'] = self.p['coup_omega0']

        # maximum number of G&R time steps (excluding prestress)
        self.p['nmax'] = 10

        # maximum load factor
        self.p['fmax'] = 1.0

    def main(self):
        # initialize fluid
        self.initialize_fluid(self.p['p0'], self.p['q0'], 'interface')

        # loop load steps
        i = 0
        for t in range(self.p['nmax'] + 1):
            # pick next load factor
            fp = self.p_vec[t]

            print('=' * 30 + ' t ' + str(t + 1) + ' ==== fp ' + '{:.2f}'.format(fp) + ' ' + '=' * 30)

            # loop sub-iterations
            for n in range(self.p['coup_imax']):
                # count total iterations (load + sub-iterations)
                i += 1

                # update simulation
                self.initialize_fluid(self.p['p0'] * fp, self.p['q0'], 'vol')
                self.initialize_fluid(self.p['p0'] * fp, self.p['q0'], 'interface')
                self.set_bc_flow(self.p['q0'])
                self.set_bc_pressure(self.p['p0'] * fp)

                # store current load
                if n == 0:
                    self.log['load'].append([])
                self.log['load'][-1].append(fp)

                # perform coupling step
                self.coup_step(i, t, n == 0)

                # check if simulation failed
                for name, s in self.sol.items():
                    if s is None:
                        print(name + ' simulation failed')
                        return

                # screen output
                out = 'i ' + str(i) + ' \tn ' + str(n + 1)
                for name, e in self.err.items():
                    out += '\t' + name + ' ' + '{:.2e}'.format(e[-1][-1])
                out += '\tomega ' + '{:.2e}'.format(self.p['coup_omega'])
                print(out)

                # combine meshes
                if self.p['fluid'] == 'fsi':
                    self.combined_vtu(i)

                # check if coupling converged
                if np.all(np.array([e[-1][-1] for e in self.err.values()]) < self.p['coup_tol']):
                    # save converged steps
                    i_conv = str(i).zfill(3)
                    t_conv = str(t).zfill(3)
                    for src in glob.glob(os.path.join(self.p['root'], '*_' + i_conv + '.vt*')):
                        trg = src.replace(i_conv, t_conv).replace(self.p['root'], self.p['root'] + '/converged')
                        shutil.copyfile(src, trg)

                    # terminate coupling
                    break
            else:
                print('\tcoupling unconverged')

    def plot_convergence(self):
        fields = ['disp', 'wss']
        colors = ['b', 'r']

        # fig, ax = plt.subplots(1, len(fields), figsize=(40, 10), dpi=200)
        fig, ax1 = plt.subplots(figsize=(40, 10), dpi=200)
        ax = [ax1, ax1.twinx()]
        for i, (name, col) in enumerate(zip(fields, colors)):
            ax[i].set_xlabel('sub-iteration $n$')
            ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))
            ax[i].set_ylabel(name, color=col)
            ax[i].tick_params(axis='y', colors=col)
            ax[i].grid(True)
            if name == 'r':
                ax[i].set_yscale('log')
            plot = []
            for res in self.log[name]:
                for v in res:
                    if 'disp' in name:
                        plot += [np.mean(rad(v))]
                    else:
                        plot += [np.mean(v)]
            ax[i].plot(plot, linestyle='-', color=col)#, marker='o'
        fig.savefig(os.path.join(self.p['f_out'], 'convergence.png'), bbox_inches='tight')
        plt.show()
        plt.close(fig)

    def archive(self):
        # move results
        shutil.move(self.p['root'], os.path.join(self.p['f_out'], self.p['root']))
        shutil.move('mesh_tube_fsi', os.path.join(self.p['f_out'], 'mesh_tube_fsi'))

        # save stored results
        file_name = os.path.join(self.p['f_out'], 'solution.json')
        np.save(file_name, self.log)

        # save parameters
        self.save_params(self.p['root'] + '.json')

    def initialize_mesh(self):
        # initial fluid mesh (zero displacements)
        for f in ['fluid', 'solid']:
            shutil.copyfile('mesh_tube_fsi/' + f + '/mesh-complete.mesh.vtu', self.p['root'] + '/' + f + '.vtu')

        # initialize with zero displacements
        names_in = [self.p['root'] + '/fluid.vtu', self.p['root'] + '/solid.vtu']
        names_out = ['mesh_' + str(self.p['n_max']['mesh']).zfill(3) + '.vtu', 'gr_000.vtu']
        dirs_out = [str(self.p['n_procs']['mesh']) + '-procs', str(self.p['n_procs']['solid']) + '-procs']
        for n_in, n_out, d_out in zip(names_in, names_out, dirs_out):
            geo = read_geo(n_in).GetOutput()
            array = n2v(np.zeros((geo.GetNumberOfPoints(), 3)))
            array.SetName('Displacement')
            geo.GetPointData().AddArray(array)

            os.makedirs(d_out, exist_ok=True)
            write_geo(os.path.join(d_out, n_out), geo)

        shutil.copyfile('mesh_tube_fsi/fluid/mesh-surfaces/interface.vtp', self.p['root'] + '/wss_000.vtp')

    def initialize_fluid(self, p, q, mode):
        if mode == 'vol':
            geo = read_geo(self.p['root'] + '/fluid.vtu').GetOutput()
        elif mode == 'interface':
            geo = self.interface_s
        else:
            raise ValueError('Unknown mode ' + mode)
        arrays = {}

        # mesh points
        points = v2n(geo.GetPoints().GetData())

        # normalized axial coordinate
        ax = points[:, 2].copy()
        amax = np.max(ax)
        ax /= amax

        # normalized radial coordinate
        rad = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        rmax = np.max(rad)
        rad /= rmax

        # estimate Poiseuille resistance
        res = 8 * 0.04 * amax / np.pi / rmax ** 4

        # estimate linear pressure gradient
        arrays['Pressure'] = p * np.ones(geo.GetNumberOfPoints()) + res * q * (1 - ax)

        # estimate quadratic flow profile
        arrays['Velocity'] = np.zeros((geo.GetNumberOfPoints(), 3))
        arrays['Velocity'][:, 2] = 4 * q / (rmax ** 2 * np.pi) * 2 * (1 - rad ** 2)

        # add arrays
        for n, a in arrays.items():
            array = n2v(a)
            array.SetName(n)
            geo.GetPointData().AddArray(array)

        # write to file
        if mode == 'vol':
            write_geo(self.p['root'] + '/fluid.vtu', geo)
        elif mode == 'interface':
            write_geo(self.p['f_load_pressure'], geo)
            write_geo(self.p['root'] + '/fluid_001.vtp', geo)

    def coup_step(self, i, t, ini):
        if self.p['fluid'] == 'fsi':
            # step 1: fluid update
            self.step('fluid', i)

            # step 1.5: fsi update
            # self.step('fsi', i)
            self.project_f2s(i)
        if self.sol['wss'] is None or self.sol['press'] is None:
            return
        self.post_wss(i)

        # relax pressure update
        # self.coup_relax('press', i, ini)
        self.write_pressure(i)

        # relax wss update
        self.coup_relax('wss', i, ini)
        self.write_wss(i)

        # step 2: solid update
        self.set_time(t + 1)
        self.step('solid', i)
        self.project_s2f(i)
        if self.sol['disp'] is None:
            return

        # relax displacement update
        self.coup_relax('disp', i, ini)
        self.write_disp(i)

        # step 3: deform mesh
        if self.p['fluid'] == 'fsi':
            self.step('mesh', i)
        self.apply_disp(i)

        # compute relaxation constant
        # self.coup_aitken()

        # for archiving
        self.coup_relax('disp_vol_solid', i, ini)
        # self.coup_relax('press_vol', i, ini)

    def coup_relax(self, name, i, ini):
        if i == 1:
            # first step: no old solution
            vec_relax = self.sol[name]
        else:
            # if ini: converged last load step. else: last sub-iteration
            vec_m0 = self.log[name][-1][-1]
            if ini and len(self.log[name]) > 2:
                # quadratically extrapolate from previous two load increments
                vec_m1 = self.log[name][-2][-1]
                vec_m2 = self.log[name][-3][-1]
                vec_relax = 3.0 * vec_m0 - 3.0 * vec_m1 + vec_m2
            elif ini and len(self.log[name]) > 1:
                # linearly extrapolate from previous load increment
                vec_m1 = self.log[name][-2][-1]
                vec_relax = 2.0 * vec_m0 - vec_m1
            else:
                # damp with previous iteration
                vec_relax = self.p['coup_omega'] * self.sol[name] + (1.0 - self.p['coup_omega']) * vec_m0

        # start a new sub-list for new load step
        if ini:
            self.log[name + '_new'].append([])
            self.log[name].append([])
            self.err[name].append([])

        # append current (damped) name
        self.log[name + '_new'][-1].append(self.sol[name])
        self.log[name][-1].append(vec_relax)

        # calculate error norm
        if i == 1 or ini:
            err = 1.0
        else:
            err = np.linalg.norm(vec_m0 - self.sol[name]) / np.linalg.norm(self.sol[name])
        self.err[name][-1].append(err)

        # update solution
        self.sol[name] = vec_relax

    def coup_aitken(self):
        if len(self.log['r'][-1]) > 2:
            r = np.array(self.log['r'][-1][-1])
            r_old = np.array(self.log['r'][-1][-2])
            self.p['coup_omega'] = - self.p['coup_omega'] * np.dot(r_old, r - r_old) / np.linalg.norm(r - r_old) ** 2
        else:
            self.p['coup_omega'] = self.p['coup_omega0']

        self.p['coup_omega'] = np.max([self.p['coup_omega'], 0.25])
        self.p['coup_omega'] = np.min([self.p['coup_omega'], 0.75])

    def project_f2s(self, i):
        # retrieve fluid solution
        src = 'steady/steady_' + str(self.p['n_max']['fluid']).zfill(3) + '.vtu'
        if not os.path.exists(src):
            self.sol['wss'] = None
            self.sol['press'] = None
            return
        trg = self.p['root'] + '/fluid_' + str(i).zfill(3) + '.vtu'
        shutil.copyfile(src, trg)

        # export interface pressure
        res = read_geo(trg).GetOutput()

        # read from fluid mesh
        res_f = v2n(res.GetPointData().GetArray('Pressure'))

        # archive
        self.sol['press_vol'] = res_f
        self.sol['velo_vol'] = v2n(res.GetPointData().GetArray('Velocity'))

        # map onto solid mesh
        self.sol['press'] = res_f[self.nodes_f - 1][self.i_fs]

    def project_s2f(self, i):
        # retrieve solid solution
        src = str(self.p['n_procs']['solid']) + '-procs/gr_' + str(i).zfill(3) + '.vtu'
        if not os.path.exists(src):
            self.sol['disp'] = None
            return
        trg = self.p['root'] + '/solid_' + str(i).zfill(3) + '.vtu'
        shutil.copyfile(src, trg)

        # export interface displacement
        res = read_geo(trg).GetOutput()

        # read from solid mesh
        res_s = v2n(res.GetPointData().GetArray('Displacement'))

        # archive
        self.sol['disp_vol_solid'] = res_s

        # map onto fluid mesh
        self.sol['disp'] = res_s[self.nodes_s - 1][self.i_sf]

    def write_pressure(self, i):
        # write pressure to file
        array = n2v(self.sol['press'])
        array.SetName('Pressure')
        self.interface_s.GetPointData().AddArray(array)
        write_geo(self.p['f_load_pressure'], self.interface_s)

        # archive
        shutil.copyfile(self.p['f_load_pressure'], os.path.join(self.p['root'], 'press_' + str(i).zfill(3) + '.vtp'))

    def write_disp(self, i):
        # create VTK array
        array = n2v(self.sol['disp'])
        array.SetName('Displacement')
        self.interface_f.GetPointData().AddArray(array)

        # write to file
        write_geo(self.p['f_disp'] + '.vtp', self.interface_f)

        # write general bc file
        with open(self.p['f_disp'] + '.dat', 'w') as f:
            f.write('3 2 ' + str(len(self.sol['disp'])) + '\n')
            f.write('0.0\n')
            f.write('1.0\n')
            for n, d in zip(self.nodes_f, self.sol['disp']):
                f.write(str(n) + '\n')
                # for _ in range(2):
                f.write('0.0 0.0 0.0\n')
                for di in d:
                    f.write(str(di) + ' ')
                f.write('\n')

        # archive
        shutil.copyfile(self.p['f_disp'] + '.vtp', os.path.join(self.p['root'], 'disp_' + str(i).zfill(3) + '.vtp'))

    def apply_disp(self, i):
        if self.p['fluid'] == 'fsi':
            # get mesh displacement solution
            src = str(self.p['n_procs']['mesh']) + '-procs/mesh_' + str(self.p['n_max']['mesh']).zfill(3) + '.vtu'
            f_out = self.p['root'] + '/fluid.vtu'
        elif self.p['fluid'] == 'poiseuille':
            # get (relaxed) displacement from g&r solution
            src = self.p['f_disp'] + '.vtp'
            f_out = self.p['root'] + '/fluid_' + str(i + 1).zfill(3) + '.vtp'
        else:
            raise ValueError('Unknown fluid option ' + self.p['fluid'])

        # warp mesh by displacements
        res = read_geo(src).GetOutput()
        res.GetPointData().SetActiveVectors('Displacement')
        warp = vtk.vtkWarpVector()
        warp.SetInputData(res)
        warp.Update()
        write_geo(f_out, warp.GetOutput())

        # archive
        self.sol['disp_vol_fluid'] = v2n(res.GetPointData().GetArray('Displacement'))
        if self.p['fluid'] == 'fsi':
            shutil.copyfile(src, self.p['root'] + '/mesh_' + str(i).zfill(3) + '.vtu')
        elif self.p['fluid'] == 'poiseuille':
            shutil.copyfile(f_out, self.p['root'] + '/mesh_' + str(i).zfill(3) + '.vtp')

    def post_wss(self, i):
        if self.p['fluid'] == 'fsi':
            # read fluid solution
            trg = self.p['root'] + '/fluid_' + str(i).zfill(3) + '.vtu'
            res = read_geo(trg)

            # calculate velocity magnitude
            calc1 = calculator(res, 'mag(Velocity)', ['Velocity'], 'u')
            grad = vtk.vtkGradientFilter()
            grad.SetInputData(calc1.GetOutput())
            grad.SetInputScalars(0, 'u')
            grad.Update()

            # extract surface
            surf = extract_surface(grad.GetOutput())
            norm = vtk.vtkPolyDataNormals()

            # generate surface normal vectors
            norm.SetInputData(surf)
            norm.Update()

            # calculate wss
            calc2 = calculator(norm, '0.04*abs(dot(Gradients,Normals))', ['Gradients', 'Normals'], 'WSS')

            # average onto cells
            p2c = vtk.vtkPointDataToCellData()
            p2c.SetInputData(calc2.GetOutput())
            p2c.Update()

            # clean (not sure why I did this)
            cl = clean(p2c.GetOutput())

            # extract FS-interface (only surface where all cells have velocity zero)
            thr = threshold(cl, 0.0, 'u')

            # map results back to point data (could also leave original point data... during p2c)
            c2p = vtk.vtkCellDataToPointData()
            c2p.SetInputData(thr.GetOutput())
            c2p.Update()
            out = c2p.GetOutput()

            # extract wss and point coordinates
            wss_f = v2n(out.GetPointData().GetArray('WSS'))
            points = v2n(out.GetPoints().GetData())

            # add wss to fluid mesh
            wss_f_vol = np.zeros(res.GetNumberOfPoints())
            wss_f_vol[self.nodes_f - 1] = wss_f
            array = n2v(wss_f_vol)
            array.SetName('WSS Python')
            res.GetOutput().GetPointData().AddArray(array)
            write_geo(trg, res.GetOutput())
        elif self.p['fluid'] == 'poiseuille':
            # read deformed interface
            res = read_geo(self.p['root'] + '/fluid_' + str(i).zfill(3) + '.vtp')

            # mesh points
            points = v2n(res.GetOutput().GetPoints().GetData())

            # get radial coordinate
            r = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)

            # calculate wss from const Poiseuille flow (assume q = q0 = const)
            wss_f = 4.0 * 0.04 / np.pi / r ** 3.0

            # archive
            array = n2v(wss_f)
            array.SetName('WSS Python')
            self.interface_f.GetPointData().AddArray(array)
            write_geo(os.path.join(self.p['root'], 'wss_' + str(i).zfill(3) + '.vtp'), self.interface_f)
        else:
            raise ValueError('Unknown fluid option ' + self.p['fluid'])

        # map fluid mesh to solid mesh
        tree = scipy.spatial.KDTree(points)
        _, i_ws = tree.query(self.points_s)
        wss_is = wss_f[i_ws]

        # get wall properties
        props = v2n(self.vol_s.GetPointData().GetArray('varWallProps'))

        # interpolate wss to solid mesh
        wss = scipy.interpolate.griddata(props[self.nodes_s - 1][:, 1:3], wss_is, (props[:, 1], props[:, 2]))
        self.sol['wss'] = wss

    def write_wss(self, i):
        # get wall properties
        props = v2n(self.vol_s.GetPointData().GetArray('varWallProps'))
        props[:, 6] = self.sol['wss']

        # create VTK array
        array = n2v(props)
        array.SetName('varWallProps')
        self.vol_s.GetPointData().AddArray(array)

        # write to file
        write_geo(self.p['root'] + '/solid.vtu', self.vol_s)

        # archive
        shutil.copyfile(self.p['root'] + '/solid.vtu', os.path.join(self.p['root'], 'wss_' + str(i).zfill(3) + '.vtu'))

    def set_time(self, t):
        # read solid mesh
        f = self.p['root'] + '/solid.vtu'
        n = 'varWallProps'
        solid = read_geo(f).GetOutput()

        # set wall properties
        props = v2n(solid.GetPointData().GetArray(n))
        props[:, 7] = t

        # create VTK array
        array = n2v(props)
        array.SetName(n)
        solid.GetPointData().AddArray(array)

        # write to file
        write_geo(f, solid)

    def combined_vtu(self, i):
        # displacements (fluid and solid)
        disp = np.zeros((self.tube.GetNumberOfPoints(), 3))
        disp[self.i_vol_f] = self.sol['disp_vol_fluid']
        disp[self.i_vol_s] = self.sol['disp_vol_solid']
        add_array(self.tube, disp, 'Displacement')

        # wss (solid only)
        wss = np.ones(self.tube.GetNumberOfPoints()) * np.nan
        wss[self.i_vol_s] = self.sol['wss']
        add_array(self.tube, wss, 'WSS')

        # pressure (fluid only)
        press = np.ones(self.tube.GetNumberOfPoints()) * np.nan
        press[self.i_vol_f] = self.sol['press_vol']
        add_array(self.tube, press, 'Pressure')

        # velocity (fluid only)
        velo = np.ones((self.tube.GetNumberOfPoints(), 3)) * np.nan
        velo[self.i_vol_f] = self.sol['velo_vol']
        add_array(self.tube, velo, 'Velocity')

        # archive
        write_geo(os.path.join(self.p['root'], 'tube_' + str(i).zfill(3) + '.vtu'), self.tube)


def rad(x):
    sign = - (x[:, 0] < 0.0).astype(int)
    sign += (x[:, 0] > 0.0).astype(int)
    return sign * np.sqrt(x[:, 0]**2 + x[:, 1]**2)


def add_array(geo, num, name):
    array = n2v(num)
    array.SetName(name)
    geo.GetPointData().AddArray(array)


# todo: this could be done easier with GlobalNodeID
def map_ids(src, trg):
    tree = scipy.spatial.KDTree(trg)
    _, res = tree.query(src)
    return res


if __name__ == '__main__':
    fluid = 'fsi'
    fsg = FSG(fluid)
    fsg.run()
