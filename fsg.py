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
from collections import defaultdict

if platform.system() == 'Darwin':
    usr = '/Users/pfaller/'
    sys.path.append(os.path.join(usr, 'work/repos/DataCuration'))
else:
    usr = '/home/pfaller'
    sys.path.append(os.path.join(usr, 'work/osmsc/curation_scripts'))

import scipy.interpolate
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

from simulation import Simulation
from cylinder import generate_mesh

# from https://github.com/StanfordCBCL/DataCuration
from vtk_functions import read_geo, write_geo, calculator, extract_surface, clean, threshold, get_all_arrays


class svFSI(Simulation):
    """
    svFSI base class (handles simulation runs)
    """

    def __init__(self, mode, f_params=None):
        # simulation parameters
        Simulation.__init__(self, f_params)

        # remove old output folders
        for f in ['1-procs', 'fsg']:
            if os.path.exists(f) and os.path.isdir(f):
                shutil.rmtree(f)

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

        # time stamp
        ct = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-')

        # output folder name
        self.p['f_out'] = mode + '_res_' + ct

        # create output folders
        os.makedirs(self.p['f_out'])
        os.makedirs(os.path.join(self.p['f_out'], 'fsg'))

        # logging
        self.log = defaultdict(list)

        # generate load vector
        self.p_vec = np.linspace(1.0, self.p['fmax'], self.p['nmax'] + 1)

    def validate_params(self):
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
        subprocess.run(shlex.split(
            'mpirun -np ' + str(self.p['n_procs_fluid']) + ' ' + self.p['exe_fluid'] + ' ' + self.p['inp_fluid']))#,
                       #stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def step_gr(self):
        subprocess.run(shlex.split(self.p['exe_solid'] + ' ' + self.p['inp_solid']),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def step_mesh(self):
        subprocess.run(shlex.split(
            'mpirun -np ' + str(self.p['n_procs_mesh']) + ' ' + self.p['exe_fluid'] + ' ' + self.p['inp_mesh']))#,
                       #stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class FSG(svFSI):
    """
    FSG-specific stuff
    """

    def __init__(self, mode, f_params=None):
        # svFSI simulations
        svFSI.__init__(self, mode, f_params)

    def run(self):
        if mode == '1-way':
            self.main_one_way()
        elif mode == '2-way':
            self.main_two_way()
        else:
            raise ValueError('Unknown mode ' + mode)

        # archive results
        self.archive()
        
        # plot convergence
        self.plot_convergence()

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
        exp = 1
        self.p['coup_omega0'] = 1/2**exp
        self.p['coup_omega'] = self.p['coup_omega0']

        # maximum number of G&R time steps (excluding prestress)
        self.p['nmax'] = 10

        # maximum load factor
        self.p['fmax'] = 1.5

    def main_one_way(self, fluid='poiseuille'):
        # initialize fluid
        self.initialize_fluid(self.p['p0'], self.p['q0'], 'interface')

        # loop load steps
        i = 0
        for t in range(self.p['nmax'] + 1):
            # pick next load factor
            fp = self.p_vec[t]

            # pressure update
            self.initialize_fluid(self.p['p0'] * fp, self.p['q0'], 'interface')

            print('==== t ' + str(t) + ' ==== fp ' + '{:.2f}'.format(fp) + ' ' + '=' * 40)

            # loop sub-iterations
            for n in range(self.p['coup_imax']):
                # count total iterations (load + sub-iterations)
                i += 1
                if n == 0:
                    self.log['load'].append([])
                self.log['load'][-1].append(fp)

                # solid update
                self.step_gr()

                # wss update
                disp_err, wss_err = self.coup_step(i, n == 0, fluid)
                if disp_err is None:
                    print('Solid simulation failed')
                    return
                if wss_err is None:
                    print('Fluid simulation failed')
                    return

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

                # screen output
                out = 'i ' + str(i) + ' \tn ' + str(n)
                out += '\tdisp ' + '{:.2e}'.format(disp_err)
                out += '\twss ' + '{:.2e}'.format(wss_err)
                out += '\tomega ' + '{:.2e}'.format(self.p['coup_omega'])
                print(out)

                # check if coupling converged
                if disp_err < self.p['coup_tol'] and wss_err < self.p['coup_tol']:
                    break
            else:
                print('\tcoupling unconverged')

    def plot_convergence(self):
        fig, ax = plt.subplots(1, len(self.log), figsize=(40, 10), dpi=200)
        for i, name in enumerate(self.log.keys()):
            ax[i].set_xlabel('sub-iteration $n$')
            ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))
            ax[i].set_ylabel(name)
            ax[i].grid(True)
            if name == 'r':
                ax[i].set_yscale('log')
            for res in self.log[name]:
                plot = []
                for v in res:
                    if 'disp' in name:
                        plot += [-rad(v)]
                    else:
                        plot += [np.mean(v)]
                ax[i].plot(plot, linestyle='-', marker='o')
        fig.savefig(os.path.join(self.p['f_out'], 'convergence.png'), bbox_inches='tight')
        plt.show()
        plt.close(fig)

    def main_two_way(self):
        # loop load steps
        i = 0
        for t in range(self.p['nmax'] + 1):
            # pick load factor
            fp = self.p_vec[t]

            # step 0: set fluid distal pressure and initialize fluid solution
            self.set_pressure(self.p['p0'] * fp)
            self.initialize_fluid(self.p['p0'] * fp, self.p['q0'], 'vol')

            print('==== t ' + str(t) + ' ==== fp ' + '{:.2f}'.format(fp) + ' ' + '=' * 40)

            # loop sub-iterations
            for n in range(self.p['coup_imax']):
                # count total iterations (load + sub-iterations)
                i += 1

                # step 1: steady-state fluid
                self.set_flow(self.p['q0'])
                self.step_fluid()
                self.project_f2s(i - 1)
                self.post_wss(i - 1)

                # step 2: solid g&r
                self.step_gr()
                self.project_s2f(i)

                # step 3: deform mesh
                self.step_mesh()
                self.apply_disp(i)

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

        # initialize with zero displacements
        names_in = ['fsg/fluid.vtu', 'fsg/solid.vtu']
        names_out = ['mesh_' + str(self.p['n_max_mesh']).zfill(3) + '.vtu', 'gr_000.vtu']
        dirs_out = [str(self.p['n_procs_mesh']) + '-procs', '1-procs']
        for n_in, n_out, d_out in zip(names_in, names_out, dirs_out):
            geo = read_geo(n_in).GetOutput()
            array = n2v(np.zeros((geo.GetNumberOfPoints(), 3)))
            array.SetName('Displacement')
            geo.GetPointData().AddArray(array)

            os.makedirs(d_out, exist_ok=True)
            write_geo(os.path.join(d_out, n_out), geo)

    def initialize_fluid(self, p, q, mode):
        if mode == 'vol':
            geo = read_geo('fsg/fluid.vtu').GetOutput()
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
            write_geo('fsg/fluid.vtu', geo)
        elif mode == 'interface':
            write_geo(self.p['f_load_pressure'], geo)

    def coup_step(self, i, ini, fluid):
        # get solid displacement
        disp = self.post_disp(i)
        if disp is None:
            return None, 0.0

        # relax displacement update
        disp_relax, disp_err = self.coup_relax('disp', disp, i, ini)

        # apply relaxed displacements to fluid
        self.project_s2f(disp_relax)
        self.apply_disp(i, fluid)

        # get wss
        wss = self.post_wss(i, fluid)
        if disp is None:
            return 0.0, None

        # relax wss update
        wss_relax, wss_err = self.coup_relax('wss', wss, i, ini)

        # store wss in geometry
        self.write_wss(wss_relax)

        # store residual
        if ini:
            self.log['r'].append([])
        self.log['r'][-1].append([disp_err, wss_err])

        # compute relaxation constant
        # self.coup_aitken()

        return disp_err, wss_err

    def coup_relax(self, name, vec_new, i, ini):
        if i == 1:
            # first step: no old solution
            vec_relax = vec_new
        else:
            # if ini: converged name of last load step. else: name of last sub-iteration
            vec_old = self.log[name][-1][-1]
            if ini and len(self.log[name]) > 1:
                # linearly extrapolate new name from previous load increment
                vec_old_old = self.log[name][-2][-1]
                vec_relax = 2.0 * vec_old - vec_old_old
            else:
                # damp with previous iteration
                vec_relax = self.p['coup_omega'] * vec_new + (1.0 - self.p['coup_omega']) * vec_old

        # start a new sub-list for new load step
        if ini:
            self.log[name + '_new'].append([])
            self.log[name].append([])

        # append current (damped) name
        self.log[name + '_new'][-1].append(vec_new)
        self.log[name][-1].append(vec_relax)

        # calculate error norm
        if i == 1:
            err = 1.0
        else:
            err = abs(norm_relax / norm_new - 1.0)

        return vec_relax, err

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
        src = str(self.p['n_procs_fluid']) + '-procs/steady_' + str(self.p['n_max_fluid']).zfill(3) + '.vtu'
        trg = 'fsg/steady_' + str(i).zfill(3) + '.vtu'
        shutil.copyfile(src, trg)

        # read fluid pressure
        res = read_geo(trg).GetOutput()
        for n in ['Pressure']:  # , 'WSS'
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

    def post_disp(self, i):
        # read from solid mesh
        fpath = '1-procs/gr_' + str(i).zfill(3) + '.vtu'
        if not os.path.exists(fpath):
            return None
        res = read_geo(fpath).GetOutput()
        res_s = v2n(res.GetPointData().GetArray('Displacement'))
        # points = v2n(res.GetPoints().GetData())

        # map onto fluid mesh
        # return (res_s + points)[self.nodes_s - 1][self.i_sf]
        return res_s[self.nodes_s - 1][self.i_sf]

    def project_s2f(self, res_f):
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

    def apply_disp(self, i, fluid, disp=None):
        if fluid == 'fsi':
            # get mesh displacement solution
            src = str(self.p['n_procs_mesh']) + '-procs/mesh_' + str(self.p['n_max_mesh']).zfill(3) + '.vtu'
            trg = 'fsg/mesh_' + str(i).zfill(3) + '.vtu'
            f_out = 'fsg/fluid.vtu'
            shutil.copyfile(src, trg)
            res = read_geo(trg).GetOutput()
        elif fluid == 'poiseuille':
            # get (relaxed) displacement from g&r solution
            res = self.interface_f
            f_out = 'fsg/wss_' + str(i).zfill(3) + '.vtp'
        else:
            raise ValueError('Unknown fluid option ' + fluid)

        # warp mesh by displacements
        res.GetPointData().SetActiveVectors('Displacement')
        warp = vtk.vtkWarpVector()
        warp.SetInputData(res)
        warp.Update()
        write_geo(f_out, warp.GetOutput())

        return v2n(warp.GetOutput().GetPoints().GetData())

    def post_wss(self, i, fluid):
        if fluid == 'fsi':
            # read fluid pressure
            trg = 'fsg/steady_' + str(i).zfill(3) + '.vtu'
            if not os.path.exists(trg):
                return None
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
            points = v2n(c2p.GetPoints().GetData())
        elif fluid == 'poiseuille':
            # read deformed geometry
            res = read_geo('fsg/wss_' + str(i).zfill(3) + '.vtp').GetOutput()

            # mesh points
            points = v2n(res.GetPoints().GetData())

            # get radial coordinate
            rad = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)

            # wss from const Poiseuille flow (assume q = q0 = const)
            # return 4.0 * 0.04 * self.p['q0'] / np.pi / rad ** 3.0
            wss_f = 4.0 * 0.04 / np.pi / rad ** 3.0
        else:
            raise ValueError('Unknown fluid option ' + fluid)

        # map fluid mesh to solid mesh
        tree = scipy.spatial.KDTree(points)
        _, i_ws = tree.query(self.points_s)
        wss_is = wss_f[i_ws]

        # read solid mesh
        solid = read_geo('mesh_tube_fsi/solid/mesh-complete.mesh.vtu').GetOutput()

        # get wall properties
        props = v2n(solid.GetPointData().GetArray('varWallProps'))

        # interpolate wss to solid mesh
        return scipy.interpolate.griddata(props[self.nodes_s - 1][:, 1:3], wss_is, (props[:, 1], props[:, 2]))

    def write_wss(self, wss):
        # read solid mesh
        solid = read_geo('mesh_tube_fsi/solid/mesh-complete.mesh.vtu').GetOutput()

        # get wall properties
        props = v2n(solid.GetPointData().GetArray('varWallProps'))
        props[:, 6] = wss

        # create VTK array
        array = n2v(props)
        array.SetName('varWallProps')
        solid.GetPointData().AddArray(array)

        # write to file
        write_geo('fsg/solid.vtu', solid)


def rad(x):
    return np.mean(np.sqrt(x[:, 0]**2 + x[:, 1]**2))


if __name__ == '__main__':
    mode = '1-way'
    fsg = FSG(mode)
    fsg.run()
