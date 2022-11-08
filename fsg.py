#!/usr/bin/env python
# coding=utf-8

import pdb
import numpy as np
import matplotlib.pyplot as plt
import sys
import shutil
import os
import json
import platform
import distro
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
    if distro.name() == 'CentOS Linux':
        usr = '/home/users/pfaller/'
    else:
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

    def main(self):
        # print reynolds number
        c1 = 2.0 * self.p['fluid']['rho'] * self.p['fluid']['q0']
        c2 = self.mesh_p['r_inner'] * np.pi * self.p['fluid']['mu']
        print('Re = ' + str(int(c1 / c2)))

        # offset pressure to match homeostatic pressure in the middle of the domain
        # self.poiseuille(0)
        # p = self.curr.get(('fluid', 'press', 'vol'))
        # self.p['p_offset'] = - (np.max(p) - np.min(p)) / 2.0
        self.p['p_offset'] = 0.0

        # loop load steps
        i = 0
        for t in range(self.p['nmax'] + 1):
            print('=' * 40 + ' t ' + str(t) + ' ==== fp ' + '{:.2f}'.format(self.p_vec[t]) + ' ' + '=' * 40)

            # loop sub-iterations
            for n in range(self.p['coup']['nmax']):
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
                self.coup_err('fluid', 'press', i, t, n)

                # screen output
                out = 'i ' + str(i) + ' \tn ' + str(n)
                for name, e in self.err.items():
                    out += '\t' + name + ' ' + '{:.2e}'.format(e[-1][-1])
                # out += '\t\t'
                # for name in self.err.keys():
                #     if n == 0:
                #         e_log = 0.0
                #     else:
                #         e_log = np.log10(self.err[name][-1][-1] / self.err[name][-1][-2])
                #     out += '\t' + name + ' log ' + '{:.1f}'.format(e_log)
                print(out)

                # archive solution
                self.curr.archive('tube', os.path.join(self.p['root'], 'tube_' + str(i).zfill(3) + '.vtu'))

                # check if coupling converged
                check_tol = np.all(np.array([e[-1][-1] for e in self.err.values()]) < self.p['coup']['tol'])
                check_n = n >= self.p['coup']['nmin']
                if check_tol and check_n:
                    # save converged steps
                    i_conv = str(i).zfill(3)
                    t_conv = str(t).zfill(3)
                    for src in glob.glob(os.path.join(self.p['root'], '*_' + i_conv + '.*')):
                        trg = src.replace(i_conv, t_conv).replace(self.p['root'], self.p['root'] + '/converged')
                        shutil.copyfile(src, trg)

                    # archive
                    self.converged += [self.curr.copy()]

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
            for res in self.converged[name]:
                for v in res:
                    if 'disp' in name:
                        plot += [np.mean(rad(v))]
                    else:
                        plot += [np.mean(v)]
            ax[i].plot(plot, linestyle='-', color=col)  # , marker='o'
        fig.savefig(os.path.join(self.p['f_out'], 'convergence.png'), bbox_inches='tight')
        plt.show()
        plt.close(fig)

    def archive(self):
        # move results
        shutil.move(self.p['root'], os.path.join(self.p['f_out'], self.p['root']))
        shutil.move('mesh_tube_fsi', os.path.join(self.p['f_out'], 'mesh_tube_fsi'))

        # # save stored results
        np.save(os.path.join(self.p['f_out'], 'err.npy'), self.err)
        # np.save(os.path.join(self.p['f_out'], 'log.npy'), self.log)
        # np.save(os.path.join(self.p['f_out'], 'converged.npy'), self.converged)

        # save parameters
        self.save_params(self.p['root'] + '.json')

        # save input files
        for src in self.p['inp'].values():
            trg = os.path.join(self.p['f_out'], self.p['root'], src)
            shutil.copyfile(src, trg)

        # save python scripts
        for src in ['fsg.py', 'svfsi.py']:
            trg = os.path.join(self.p['f_out'], self.p['root'], src)
            shutil.copyfile(src, trg)

        # save material model
        src = usr + os.path.split(self.p['exe']['solid'])[0] + '/../../../Code/Source/svFSI/FEMbeCmm.cpp'
        trg = os.path.join(self.p['f_out'], self.p['root'], 'FEMbeCmm.cpp')
        shutil.copyfile(src, trg)

    def coup_step(self, i, t, n):
        # store previous solutions
        self.bfor = self.prev.copy()
        self.prev = self.curr.copy()

        # step 1: fluid update
        if self.p['fsi']:
            if self.step('fluid', i, t):
                return
        else:
            self.poiseuille(t)

        # relax fluid update
        self.coup_relax('fluid', 'press', i, t, n)
        self.coup_relax('fluid', 'wss', i, t, n)

        # step 2: solid update
        if self.step('solid', i, t):
            return

        # relax solid update
        self.coup_relax('solid', 'disp', i, t, n)

        # step 3: deform mesh
        if self.p['fsi']:
            if self.step('mesh', i, t):
                return

    def coup_predict(self, i, t, n):
        if self.p['fsi']:
            # predict wss
            kind = ('fluid', 'wss', 'vol')
        else:
            # predict displacements
            kind = ('solid', 'disp', 'vol')

        if t == 0 or not self.p['predict_file']:
            # extrapolate from previous time step(s)
            sol = self.predictor(kind, t)
        else:
            # predict from file
            sol = self.predictor_tube(kind, t)
        self.curr.add(kind, sol)

        # calculate wss from poiseuille flow
        if not self.p['fsi']:
            self.poiseuille(t)

    def predictor(self, kind, t):
        # fluid, solid, tube
        # disp, velo, wss, press
        # vol, int
        d, f, p = kind

        # number of old solutions
        n_sol = len(self.converged)

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
        vec_m0 = self.converged[-1].get(kind)
        if n_sol == 1:
            return vec_m0

        # linearly extrapolate from previous load increment
        vec_m1 = self.converged[-2].get(kind)
        if n_sol == 2:
            return 2.0 * vec_m0 - vec_m1

        # quadratically extrapolate from previous two load increments
        vec_m2 = self.converged[-3].get(kind)
        # return 3.0 * vec_m0 - 3.0 * vec_m1 + vec_m2
        return 2.5 * vec_m0 - 2.0 * vec_m1 + 0.5 * vec_m2

    def predictor_tube(self, kind, t):
        d, f, p = kind
        fname = 'gr_partitioned/tube_' + str(t).zfill(3) + '.vtu'
        # fname = 'gr/gr_' + str(t + 1).zfill(3) + '.vtu'
        if not os.path.exists(fname):
            return None
        geo = read_geo(fname).GetOutput()
        if f == 'disp':
            return v2n(geo.GetPointData().GetArray('Displacement'))[self.map(((p, d), ('vol', 'tube')))]
        elif f == 'wss':
            if geo.GetPointData().HasArray('WSS'):
                return v2n(geo.GetPointData().GetArray('WSS'))[self.map(((p, d), ('vol', 'tube')))]
            else:
                disp = v2n(geo.GetPointData().GetArray('Displacement'))[self.map(((p, d), ('vol', 'solid')))]
                self.curr.add((d, 'disp', p), disp)
                self.poiseuille(t)
                return self.curr.get(kind)

    def coup_relax(self, domain, name, i, t, n):
        # volume increment
        curr_v = deepcopy(self.curr.get((domain, name, 'vol')))
        prev_v = deepcopy(self.prev.get((domain, name, 'vol')))
        bfor_v = deepcopy(self.bfor.get((domain, name, 'vol')))

        # interface increment
        curr_i = deepcopy(self.curr.get((domain, name, 'int')))
        prev_i = deepcopy(self.prev.get((domain, name, 'int')))

        # log increments for aitken relaxation
        self.log[name] += [curr_i - prev_i]

        # calculate new relaxation factor
        omega = self.coup_omega(name, t)
        self.p['coup']['omega'] = omega

        if t == 0:
            vec_relax = curr_v
        else:
            # vec_relax = 0.5 * curr_v + 0.375 * prev_v + 0.125 * bfor_v # best so far but starts oscillating at t=8
            # vec_relax = 0.25 * curr_v + 0.625 * prev_v + 0.125 * bfor_v # more damping but more robust
            # vec_relax = 0.5 * curr_v + 0.4 * prev_v + 0.1 * bfor_v # even better (but crashes at t=6)
            # vec_relax = 0.5 * curr_v + 0.5 * prev_v # good but takes many iterations and oscillates
            # vec_relax = 0.6 * curr_v + 0.3 * prev_v + 0.1 * bfor_v # oscillates

            # if t == 6:
            #     vec_relax = 0.25 * curr_v + 0.625 * prev_v + 0.125 * bfor_v
            # elif t >= 7:
            #     vec_relax = 0.125 * curr_v + 0.625 * prev_v + 0.25 * bfor_v
            # else:
            #     vec_relax = 0.5 * curr_v + 0.4 * prev_v + 0.1 * bfor_v

            # vec_relax = 0.25 * curr_v + 0.625 * prev_v + 0.125 * bfor_v
            # if t >= 5:
            #     vec_relax = 0.125 * curr_v + 0.625 * prev_v + 0.25 * bfor_v

            vec_relax = 0.5 * curr_v + 0.4 * prev_v + 0.1 * bfor_v
            if t >= 2:
                vec_relax = 0.25 * curr_v + 0.625 * prev_v + 0.125 * bfor_v
            if t >= 4:
                vec_relax = 0.125 * curr_v + 0.625 * prev_v + 0.25 * bfor_v

        # update solution
        self.curr.add((domain, name, 'vol'), vec_relax)

    def coup_err(self, domain, name, i, t, n):
        curri = deepcopy(self.curr.get((domain, name, 'int')))
        if i == 1 or n == 0:
            # first step: no old solution
            err = 1.0
        else:
            # difference
            diff = np.abs(self.log[name][-1])
            if len(diff.shape) == 1:
                diff_n = np.max(diff)
            else:
                diff_n = np.max(np.linalg.norm(diff, axis=1))

            # norm
            if t <= 1:
                # normalize w.r.t. mean radius
                norm = np.mean(np.abs(rad(self.points[('int', 'solid')])))
            else:
                # normalize w.r.t. solution norm
                norm = np.max(np.abs(curri))

            # normalized error
            err = np.linalg.norm(diff_n) / norm

        # start a new sub-list for new load step
        if n == 0:
            self.err[name].append([])

        # append error norm
        self.err[name][-1].append(err)

    def coup_omega(self, name, t):
        # no relaxation necessary during prestressing (prestress does not depend on wss)
        if t == 0:
            return 1.0

        # if name == 'disp':
        #     return 0.25
        # elif name == 'wss':
        #     return 0.25
        return 0.5

        if len(self.log[name]) < 2:
            return self.p['coup']['omega0']

        # current and old solution
        m1 = self.log[name][-1]
        m2 = self.log[name][-2]
        if len(m1.shape) == 2:
            m1 = np.linalg.norm(m1, axis=1)
            m2 = np.linalg.norm(m2, axis=1)
        diff = m1 - m2
        norm = np.linalg.norm(diff)

        if norm == 0.0:
            return self.p['coup']['omega0']

        # aitken relaxation
        omega = - self.p['coup']['omega'] * np.dot(m2, diff) / norm ** 2

        # lower bound
        omega = np.max([omega, 0.1])

        # upper bound
        omega = np.min([omega, 0.9])

        return omega


def rad(x):
    sign = - (x[:, 0] < 0.0).astype(int)
    sign += (x[:, 0] > 0.0).astype(int)
    return sign * np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)


if __name__ == '__main__':
    fsg = FSG('in_sim/partitioned.json')
    fsg.run()
