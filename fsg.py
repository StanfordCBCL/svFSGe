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
import os
import vtk
import json
import platform
import glob
from copy import deepcopy
from collections import defaultdict

import scipy.interpolate
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

from svfsi import svFSI


if platform.system() == 'Darwin':
    usr = '/Users/pfaller/'
elif platform.system() == 'Linux':
    usr = '/home/pfaller/'

# from https://github.com/StanfordCBCL/DataCuration
sys.path.append(os.path.join(usr, 'work/repos/DataCuration'))
from vtk_functions import read_geo, write_geo, calculator, extract_surface, clean, threshold, get_all_arrays


class FSG(svFSI):
    """
    FSG-specific stuff
    """

    def __init__(self, f_params=None):
        # svFSI simulations
        svFSI.__init__(self, f_params)

    def run(self):
        # run simulation
        self.main()

        # plot convergence
        try:
            self.plot_convergence()
        except:
            pass

        # archive results
        pdb.set_trace()
        self.archive()

    def set_params(self):
        # debug mode?
        self.p['debug'] = False

        # run 3D fsi?
        self.p['fsi'] = False

        # simulation folder
        self.p['root'] = 'partitioned'

        # define file paths
        self.p['exe'] = {'fluid': usr + '/work/repos/svFSI_test/build/svFSI-build/bin/svFSI',
                         'solid': usr + '/work/repos/svFSI_fork/build/svFSI-build/bin/svFSI',
                         'mesh': usr + '/work/repos/svFSI_clean/build/svFSI-build/bin/svFSI'}

        # input files
        self.p['inp'] = {'fluid': 'steady_flow.inp', 'solid': 'gr_restart.inp', 'mesh': 'mesh.inp'}

        # output folders
        self.p['out'] = {'fluid': 'steady', 'solid': 'gr_restart', 'mesh': 'mesh'}

        # number of processors
        self.p['n_procs'] = {'solid': 1, 'fluid': 6, 'mesh': 6}

        # maximum number of time steps
        self.p['n_max'] = {'solid': 1, 'fluid': 30, 'mesh': 10}

        # interface loads
        self.p['f_load_pressure'] = 'interface_pressure.vtp'
        self.p['f_load_wss'] = 'interface_wss.vtp'
        self.p['f_disp'] = 'interface_displacement'
        self.p['f_solid_geo'] = 'solid.vtu'
        self.p['f_fluid_geo'] = 'fluid.vtu'

        # homeostatic pressure
        self.p['p0'] = 13.9868

        # fluid flow
        self.p['q0'] = 0.0

        # coupling tolerance
        self.p['coup_tol'] = 1.0e-4

        # maximum number of coupling iterations
        self.p['coup_nmin'] = 1
        self.p['coup_nmax'] = 200

        # relaxation constant
        exp = 1
        self.p['coup_omega0'] = 1/2**exp
        self.p['coup_omega'] = self.p['coup_omega0']

        # maximum number of G&R time steps (excluding prestress)
        self.p['nmax'] = 10

        # maximum load factor
        self.p['fmax'] = 1.0

    def main(self):
        # loop load steps
        i = 0
        for t in range(self.p['nmax'] + 1):
            print('=' * 30 + ' t ' + str(t) + ' ==== fp ' + '{:.2f}'.format(self.p_vec[t]) + ' ' + '=' * 30)

            # loop sub-iterations
            for n in range(self.p['coup_nmax']):
                # count total iterations (load + sub-iterations)
                i += 1

                # perform coupling step
                self.coup_step(i, t, n)

                # check if simulation failed
                for name, s in self.curr.sol.items():
                    if s is None:
                        print(name + ' simulation failed')
                        return

                # get error
                self.coup_err('solid', 'disp', i, t, n)
                self.coup_err('fluid', 'wss', i, t, n)

                # screen output
                out = 'i ' + str(i) + ' \tn ' + str(n)
                for name, e in self.err.items():
                    out += '\t' + name + ' ' + '{:.2e}'.format(e[-1][-1])
                out += '\tomega ' + '{:.2e}'.format(self.p['coup_omega'])
                print(out)

                # archive solution
                self.curr.archive('tube', os.path.join(self.p['root'], 'tube_' + str(i).zfill(3) + '.vtu'))

                # check if coupling converged
                check_tol = np.all(np.array([e[-1][-1] for e in self.err.values()]) < self.p['coup_tol'])
                check_n = n >= self.p['coup_nmin']
                if check_tol and check_n:
                    # save converged steps
                    i_conv = str(i).zfill(3)
                    t_conv = str(t).zfill(3)
                    for src in glob.glob(os.path.join(self.p['root'], '*_' + i_conv + '.*')):
                        trg = src.replace(i_conv, t_conv).replace(self.p['root'], self.p['root'] + '/converged')
                        shutil.copyfile(src, trg)

                    # archive
                    self.log += [self.curr.copy()]

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

        # # save stored results
        # file_name = os.path.join(self.p['f_out'], 'solution.json')
        # np.save(file_name, self.log)

        # save parameters
        self.save_params(self.p['root'] + '.json')

    def coup_step(self, i, t, n):
        if t == 0:
            self.p['coup_omega'] = 1.0
        else:
            self.p['coup_omega'] = self.p['coup_omega0']

        # copy previous solution
        self.prev = self.curr.copy()

        # replace solution with prediction
        if n == 0:
            self.coup_predict(i, t, n)

        if self.p['fsi']:
            if self.step('fluid', i, t):
                return
        else:
            self.poiseuille(t)

        # relax pressure update
        # self.coup_relax('fluid', 'press', i, t, ini)

        # relax wss update (prestress does not depend on wss)
        if t > 0:
            self.coup_relax('fluid', 'wss', i, t, n)

        # step 2: solid update
        if self.step('solid', i, t):
            return

        # relax displacement update
        self.coup_relax('solid', 'disp', i, t, n)

        # step 3: deform mesh
        if self.p['fsi']:
            if self.step('mesh', i, t):
                return

        # compute relaxation constant
        # self.coup_aitken()

    def coup_predict(self, i, t, n):
        fields = [('solid', 'disp', 'vol'), ('fluid', 'wss', 'vol')]
        for f in fields:
            # sol = self.predictor_gr(f, t)
            # if sol is None:
            sol = self.predictor(f, t)
            self.curr.add(f, sol)

    def predictor(self, kind, t):
        # fluid, solid, tube
        # disp, velo, wss, press
        # vol, int
        d, f, p = kind

        # number of old solutions
        n_sol = len(self.log)

        if n_sol == 0:
            if f == 'disp':
                # zero displacements
                return np.zeros(self.points[(p, d)].shape)
            elif f == 'wss':
                # wss from poiseuille flow through reference configuration
                self.poiseuille(t)
                return self.curr.get(kind)
            else:
                raise ValueError('No predictor for field ' + f)

        # previous solution
        vec_m0 = self.log[-1].get(kind)
        if n_sol == 1:
            return vec_m0

        # linearly extrapolate from previous load increment
        vec_m1 = self.log[-2].get(kind)
        if n_sol == 2:
            return 2.0 * vec_m0 - vec_m1

        # quadratically extrapolate from previous two load increments
        vec_m2 = self.log[-3].get(kind)
        return 3.0 * vec_m0 - 3.0 * vec_m1 + vec_m2

    def predictor_tube(self, kind, t):
        d, f, p = kind
        fname = 'gr_partitioned/tube_' + str(t).zfill(3) + '.vtu'
        if not os.path.exists(fname):
            return None
        geo = read_geo(fname).GetOutput()
        if f == 'disp':
            return v2n(geo.GetPointData().GetArray('Displacement'))[self.map(((p, d), ('vol', 'tube')))]
        elif f == 'wss':
            return v2n(geo.GetPointData().GetArray('WSS'))[self.map(((p, d), ('vol', 'tube')))]

    def predictor_gr(self, kind, t):
        d, f, p = kind
        fname = 'gr_minimal_tube/gr_' + str(t+1).zfill(3) + '.vtu'
        if not os.path.exists(fname):
            return None
        geo = read_geo(fname).GetOutput()
        if f == 'disp':
            return v2n(geo.GetPointData().GetArray('Displacement'))[self.map(((p, d), ('vol', 'solid')))]
        elif f == 'wss':
            self.poiseuille(t)
            self.curr.get(kind)

    def coup_relax(self, domain, name, i, t, n):
        curri = deepcopy(self.curr.get((domain, name, 'vol')))
        previ = deepcopy(self.prev.get((domain, name, 'vol')))
        if i == 1 or n == 0:
            # first step: no old solution
            vec_relax = curri
        else:
            # damp with previous iteration
            vec_relax = self.p['coup_omega'] * curri + (1.0 - self.p['coup_omega']) * previ

        # update solution
        self.curr.add((domain, name, 'vol'), vec_relax)

    def coup_err(self, domain, name, i, t, n):
        curr = deepcopy(self.curr.get((domain, name, 'vol')))
        # prev = deepcopy(self.prev.get((domain, name, 'vol')))
        curri = deepcopy(self.curr.get((domain, name, 'int')))
        previ = deepcopy(self.prev.get((domain, name, 'int')))
        if i == 1 or n == 0:
            # first step: no old solution
            vec_relax = curri
            err = 1.0
        else:
            # difference
            diff = np.abs(previ - curri)
            if len(diff.shape) == 1:
                diff_n = np.max(diff)
            else:
                diff_n = np.max(np.linalg.norm(diff, axis=1))

            # norm
            if t == 0:
                # normalize w.r.t. mean radius
                norm = np.mean(rad(self.points[('int', 'solid')]))
            else:
                # normalize w.r.t. displacement norm
                norm = np.max(np.abs(curri))

            # normalized error
            err = np.linalg.norm(diff_n) / norm

        # start a new sub-list for new load step
        if n == 0:
            self.err[name].append([])

        # append error norm
        self.err[name][-1].append(err)

    def coup_aitken(self):
        if len(self.log['r'][-1]) > 2:
            r = np.array(self.log['r'][-1][-1])
            r_old = np.array(self.log['r'][-1][-2])
            self.p['coup_omega'] = - self.p['coup_omega'] * np.dot(r_old, r - r_old) / np.linalg.norm(r - r_old) ** 2
        else:
            self.p['coup_omega'] = self.p['coup_omega0']

        self.p['coup_omega'] = np.max([self.p['coup_omega'], 0.25])
        self.p['coup_omega'] = np.min([self.p['coup_omega'], 0.75])


def rad(x):
    sign = - (x[:, 0] < 0.0).astype(int)
    sign += (x[:, 0] > 0.0).astype(int)
    return sign * np.sqrt(x[:, 0]**2 + x[:, 1]**2)


if __name__ == '__main__':
    fsg = FSG()
    fsg.run()
