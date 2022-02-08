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

    def __init__(self, fluid, f_params=None):
        # simulation parameters
        Simulation.__init__(self, f_params)

        # fluid mode (fsi or poiseuille)
        if fluid not in ['fsi', 'poiseuille']:
            raise ValueError('Unknown fluid option ' + fluid)
        self.p['fluid'] = fluid

        # remove old output folders
        self.fields = ['fluid', 'solid', 'mesh']
        for f in self.fields + ['fsg']:
            if f is not 'fsg':
                f = str(self.p['n_procs_' + f]) + '-procs'
            if os.path.exists(f) and os.path.isdir(f):
                shutil.rmtree(f)

        # make folders
        os.makedirs('fsg')
        os.makedirs(os.path.join('fsg', 'converged'))

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
        self.p['f_out'] = fluid + '_res_' + ct

        # create output folder
        os.makedirs(self.p['f_out'])

        # logging
        self.log = defaultdict(list)
        self.sol = {'disp': np.zeros(self.points_f.shape),
                    'wss': np.zeros(len(self.points_f))}

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

    def step(self, name, i):
        if name not in self.fields:
            raise ValueError('Unknown step option ' + name)
        exe = 'mpirun -np ' + str(self.p['n_procs_' + name]) + ' ' + self.p['exe_' + name] + ' ' + self.p['inp_' + name]
        with open(os.path.join('fsg', name + '_' + str(i).zfill(3) + '.log'), 'w') as f:
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

        # archive results
        self.archive()
        
        # plot convergence
        self.plot_convergence()

    def set_params(self):
        # define file paths
        self.p['exe_fluid'] = usr + '/work/repos/svFSI_clean/build/svFSI-build/bin/svFSI'
        self.p['exe_solid'] = usr + '/work/repos/svFSI_direct/build/svFSI-build/bin/svFSI'
        self.p['exe_mesh'] = self.p['exe_fluid']

        # input files
        self.p['inp_fluid'] = 'steady_flow.inp'
        self.p['inp_solid'] = 'gr_restart.inp'
        self.p['inp_mesh'] = 'mesh.inp'

        # number of processors
        self.p['n_procs_fluid'] = 10
        self.p['n_procs_solid'] = 1
        self.p['n_procs_mesh'] = 10

        # maximum number of time steps
        self.p['n_max_fluid'] = 30
        self.p['n_max_mesh'] = 10

        # interface loads
        self.p['f_load_pressure'] = 'interface_pressure.vtp'
        self.p['f_load_wss'] = 'interface_wss.vtp'
        self.p['f_disp'] = 'interface_displacement.vtp'

        # homeostatic pressure
        self.p['p0'] = 13.9868

        # fluid flow
        self.p['q0'] = 0.001

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

    def main(self):
        # initialize fluid
        self.initialize_fluid(self.p['p0'], self.p['q0'], 'interface')

        # loop load steps
        i = 0
        for t in range(self.p['nmax'] + 1):
            # pick next load factor
            fp = self.p_vec[t]

            # pressure update
            self.initialize_fluid(self.p['p0'] * fp, self.p['q0'], 'interface')
            self.set_flow(self.p['q0'])
            self.set_pressure(self.p['p0'] * fp)

            print('==== t ' + str(t) + ' ==== fp ' + '{:.2f}'.format(fp) + ' ' + '=' * 40)

            # loop sub-iterations
            for n in range(self.p['coup_imax']):
                # count total iterations (load + sub-iterations)
                i += 1

                # store current load
                if n == 0:
                    self.log['load'].append([])
                self.log['load'][-1].append(fp)

                # update
                disp_err, wss_err = self.coup_step(i, n == 0)

                # check for errors
                if disp_err is None:
                    print('Solid simulation failed')
                    return
                if wss_err is None:
                    print('Fluid simulation failed')
                    return

                # screen output
                out = 'i ' + str(i) + ' \tn ' + str(n)
                out += '\tdisp ' + '{:.2e}'.format(disp_err)
                out += '\twss ' + '{:.2e}'.format(wss_err)
                out += '\tomega ' + '{:.2e}'.format(self.p['coup_omega'])
                print(out)

                # check if coupling converged
                if disp_err < self.p['coup_tol'] and wss_err < self.p['coup_tol']:
                    # save converged steps
                    i_conv = str(i).zfill(3)
                    t_conv = str(t).zfill(3)
                    for src in glob.glob(os.path.join('fsg', '*_' + i_conv + '.vt*')):
                        trg = src.replace(i_conv, t_conv).replace('fsg', 'fsg/converged')
                        shutil.copyfile(src, trg)

                    # terminate coupling
                    break
            else:
                print('\tcoupling unconverged')

    def plot_convergence(self):
        fields = ['wss', 'disp', 'r']

        fig, ax = plt.subplots(1, len(fields), figsize=(40, 10), dpi=200)
        for i, name in enumerate(fields):
            ax[i].set_xlabel('sub-iteration $n$')
            ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))
            ax[i].set_ylabel(name)
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
            ax[i].plot(plot, linestyle='-')#, marker='o'
        fig.savefig(os.path.join(self.p['f_out'], 'convergence.png'), bbox_inches='tight')
        plt.show()
        plt.close(fig)

    def archive(self):
        # move results
        shutil.move('fsg', os.path.join(self.p['f_out'], 'fsg'))
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
        dirs_out = [str(self.p['n_procs_mesh']) + '-procs', str(self.p['n_procs_solid']) + '-procs']
        for n_in, n_out, d_out in zip(names_in, names_out, dirs_out):
            geo = read_geo(n_in).GetOutput()
            array = n2v(np.zeros((geo.GetNumberOfPoints(), 3)))
            array.SetName('Displacement')
            geo.GetPointData().AddArray(array)

            os.makedirs(d_out, exist_ok=True)
            write_geo(os.path.join(d_out, n_out), geo)

        shutil.copyfile('mesh_tube_fsi/fluid/mesh-surfaces/interface.vtp', 'fsg/wss_000.vtp')

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
            write_geo('fsg/fluid_001.vtp', geo)

    def coup_step(self, i, ini):
        # step 1: fluid update
        if self.p['fluid'] == 'fsi':
            self.step('fluid', i)
            self.project_f2s(i)
        self.post_wss(i)
        if self.sol['wss'] is None:
            return 0.0, None

        # relax wss update
        # print(str(np.mean(self.sol['wss'])) + '\t' + str(np.mean(rad(self.sol['disp']))))
        wss_err = self.coup_relax('wss', i, ini)
        self.write_wss(i)

        # step 2: solid update
        self.step('solid', i)
        self.project_s2f(i)
        if self.sol['disp'] is None:
            return None, 0.0

        # relax displacement update
        disp_err = self.coup_relax('disp', i, ini)
        self.write_disp(i)

        # step 3: deform mesh
        if self.p['fluid'] == 'fsi':
            self.step('mesh', i)
        self.apply_disp(i)

        # store residual
        if ini:
            self.log['r'].append([])
        self.log['r'][-1].append([disp_err, wss_err])

        # compute relaxation constant
        # self.coup_aitken()

        return disp_err, wss_err

    def coup_relax(self, name, i, ini):
        if i == 1:
            # first step: no old solution
            vec_relax = self.sol[name]
        else:
            # if ini: converged last load step. else: last sub-iteration
            vec_old = self.log[name][-1][-1]
            if ini and len(self.log[name]) > 1:
                # linearly extrapolate new name from previous load increment
                vec_old_old = self.log[name][-2][-1]
                vec_relax = 2.0 * vec_old - vec_old_old
            else:
                # damp with previous iteration
                vec_relax = self.p['coup_omega'] * self.sol[name] + (1.0 - self.p['coup_omega']) * vec_old

        # start a new sub-list for new load step
        if ini:
            self.log[name + '_new'].append([])
            self.log[name].append([])

        # append current (damped) name
        self.log[name + '_new'][-1].append(self.sol[name])
        self.log[name][-1].append(vec_relax)

        # calculate error norm
        if i == 1:
            err = 1.0
        else:
            err = abs(np.linalg.norm(vec_relax) / np.linalg.norm(self.sol[name]) - 1.0)

        # update solution
        self.sol[name] = vec_relax

        return err

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
        src = str(self.p['n_procs_fluid']) + '-procs/steady_' + str(self.p['n_max_fluid']).zfill(3) + '.vtu'
        if not os.path.exists(src):
            self.sol['wss'] = None
            return
        trg = 'fsg/fluid_' + str(i).zfill(3) + '.vtu'
        shutil.copyfile(src, trg)

        # export interface pressure
        res = read_geo(trg).GetOutput()

        # read from fluid mesh
        n = 'Pressure'
        res_f = v2n(res.GetPointData().GetArray(n))

        # map onto solid mesh
        res_s = res_f[self.nodes_f - 1][self.i_fs]

        # write pressure to file (no relaxation)
        array = n2v(res_s)
        array.SetName(n)
        self.interface_s.GetPointData().AddArray(array)
        write_geo(self.p['f_load_' + n.lower()], self.interface_s)

    def project_s2f(self, i):
        # retrieve solid solution
        src = str(self.p['n_procs_solid']) + '-procs/gr_' + str(i).zfill(3) + '.vtu'
        if not os.path.exists(src):
            self.sol['disp'] = None
            return
        trg = 'fsg/solid_' + str(i).zfill(3) + '.vtu'
        shutil.copyfile(src, trg)

        # export interface displacement
        res = read_geo(trg).GetOutput()

        # read from solid mesh
        n = 'Displacement'
        res_s = v2n(res.GetPointData().GetArray(n))

        # map onto fluid mesh
        self.sol['disp'] = res_s[self.nodes_s - 1][self.i_sf]

    def write_disp(self, i):
        # create VTK array
        array = n2v(self.sol['disp'])
        array.SetName('Displacement')
        self.interface_f.GetPointData().AddArray(array)

        # write to file
        write_geo(self.p['f_disp'], self.interface_f)

        # write general bc file
        with open('interface_displacement.dat', 'w') as f:
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
        shutil.copyfile(self.p['f_disp'], os.path.join('fsg', 'disp_' + str(i).zfill(3) + '.vtp'))

    def apply_disp(self, i):
        if self.p['fluid'] == 'fsi':
            # get mesh displacement solution
            src = str(self.p['n_procs_mesh']) + '-procs/mesh_' + str(self.p['n_max_mesh']).zfill(3) + '.vtu'
            f_out = 'fsg/fluid.vtu'
        elif self.p['fluid'] == 'poiseuille':
            # get (relaxed) displacement from g&r solution
            src = self.p['f_disp']
            f_out = 'fsg/fluid_' + str(i + 1).zfill(3) + '.vtp'
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
        if self.p['fluid'] == 'fsi':
            shutil.copyfile(src, 'fsg/mesh_' + str(i).zfill(3) + '.vtu')
        elif self.p['fluid'] == 'poiseuille':
            shutil.copyfile(f_out, 'fsg/mesh_' + str(i).zfill(3) + '.vtp')

    def post_wss(self, i):
        if self.p['fluid'] == 'fsi':
            # read fluid solution
            trg = 'fsg/fluid_' + str(i).zfill(3) + '.vtu'
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
            calc2 = calculator(norm, '0.04*abs(Gradients.Normals)', ['Gradients', 'Normals'], 'WSS')

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
        elif self.p['fluid'] == 'poiseuille':
            # read deformed interface
            res = read_geo('fsg/fluid_' + str(i).zfill(3) + '.vtp')

            # mesh points
            points = v2n(res.GetOutput().GetPoints().GetData())

            # get radial coordinate
            r = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)

            # calculate wss from const Poiseuille flow (assume q = q0 = const)
            wss_f = 4.0 * 0.04 / np.pi / r ** 3.0
        else:
            raise ValueError('Unknown fluid option ' + self.p['fluid'])

        # map fluid mesh to solid mesh
        tree = scipy.spatial.KDTree(points)
        _, i_ws = tree.query(self.points_s)
        wss_is = wss_f[i_ws]

        # read solid mesh
        solid = read_geo('mesh_tube_fsi/solid/mesh-complete.mesh.vtu').GetOutput()

        # get wall properties
        props = v2n(solid.GetPointData().GetArray('varWallProps'))

        # interpolate wss to solid mesh
        wss = scipy.interpolate.griddata(props[self.nodes_s - 1][:, 1:3], wss_is, (props[:, 1], props[:, 2]))
        self.sol['wss'] = wss

        # archive
        array = n2v(wss_f)
        array.SetName('WSS Python')
        self.interface_f.GetPointData().AddArray(array)
        write_geo(os.path.join('fsg', 'wss_' + str(i).zfill(3) + '.vtp'), self.interface_f)

    def write_wss(self, i):
        # read solid mesh
        solid = read_geo('mesh_tube_fsi/solid/mesh-complete.mesh.vtu').GetOutput()

        # get wall properties
        props = v2n(solid.GetPointData().GetArray('varWallProps'))
        props[:, 6] = self.sol['wss']

        # create VTK array
        array = n2v(props)
        array.SetName('varWallProps')
        solid.GetPointData().AddArray(array)

        # write to file
        write_geo('fsg/solid.vtu', solid)

        # archive
        shutil.copyfile('fsg/solid.vtu', os.path.join('fsg', 'wss_' + str(i).zfill(3) + '.vtu'))


def rad(x):
    sign = - (x[:, 0] < 0.0).astype(int)
    sign += (x[:, 0] > 0.0).astype(int)
    return sign * np.sqrt(x[:, 0]**2 + x[:, 1]**2)


if __name__ == '__main__':
    fluid = 'fsi'
    fsg = FSG(fluid)
    fsg.run()
