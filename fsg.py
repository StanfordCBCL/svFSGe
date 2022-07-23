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

        # input files
        self.p['inp'] = {'fluid': 'steady_flow.inp', 'solid': 'gr_restart.inp', 'mesh': 'mesh.inp'}

        # output folders
        self.p['out'] = {'fluid': 'steady', 'solid': 'gr_restart', 'mesh': 'mesh'}

        # number of processors
        self.p['n_procs'] = {'solid': 1, 'fluid': 10, 'mesh': 10}

        # maximum number of time steps
        self.p['n_max'] = {'fluid': 20, 'mesh': 10}

        # interface loads
        self.p['f_load_pressure'] = 'interface_pressure.vtp'
        self.p['f_load_wss'] = 'interface_wss.vtp'
        self.p['f_disp'] = 'interface_displacement'
        self.p['f_solid_geo'] = 'solid.vtu'
        self.p['f_fluid_geo'] = 'fluid.vtu'

        # homeostatic pressure
        self.p['p0'] = 13.9868

        # fluid flow
        self.p['q0'] = 0

        # coupling tolerance
        self.p['coup_tol'] = 1.0e-3

        # maximum number of coupling iterations
        self.p['coup_imax'] = 100

        # relaxation constant
        exp = 2
        self.p['coup_omega0'] = 1/2**exp
        self.p['coup_omega'] = self.p['coup_omega0']

        # maximum number of G&R time steps (excluding prestress)
        self.p['nmax'] = 20

        # maximum load factor
        self.p['fmax'] = 1.0

    def main(self):
        # loop load steps
        i = 0
        for t in range(self.p['nmax'] + 1):
            # pick next load factor
            fp = self.p_vec[t]

            print('=' * 30 + ' t ' + str(t) + ' ==== fp ' + '{:.2f}'.format(fp) + ' ' + '=' * 30)

            # loop sub-iterations
            for n in range(self.p['coup_imax']):
                # count total iterations (load + sub-iterations)
                i += 1

                # update simulation
                # self.initialize_fluid(self.p['p0'] * fp, self.p['q0'], 'vol')
                # self.initialize_fluid(self.p['p0'] * fp, self.p['q0'], 'interface')
                self.set_fluid(self.p['q0'], self.p['p0'] * fp)

                # store current load
                if n == 0:
                    self.log['load'].append([])
                self.log['load'][-1].append(fp)

                # perform coupling step
                self.coup_step(i, t, n == 0)

                # check if simulation failed
                for name, s in self.curr.sol.items():
                    if s is None:
                        print(name + ' simulation failed')
                        return

                # screen output
                out = 'i ' + str(i) + ' \tn ' + str(n)
                for name, e in self.err.items():
                    out += '\t' + name + ' ' + '{:.2e}'.format(e[-1][-1])
                out += '\tomega ' + '{:.2e}'.format(self.p['coup_omega'])
                print(out)

                # archive solution
                self.curr.archive(os.path.join(self.p['root'], 'tube_' + str(i).zfill(3) + '.vtu'), i)

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

    # def initialize_mesh(self):
    #     # initial fluid mesh (zero displacements)
    #     for f in ['fluid', 'solid']:
    #         shutil.copyfile('mesh_tube_fsi/' + f + '/mesh-complete.mesh.vtu', self.p['root'] + '/' + f + '.vtu')
    #
    #     # initialize with zero displacements
    #     names_in = [self.p['root'] + '/fluid.vtu', self.p['root'] + '/solid.vtu']
    #     names_out = ['mesh_' + str(self.p['n_max']['mesh']).zfill(3) + '.vtu', 'gr_000.vtu']
    #     dirs_out = [str(self.p['n_procs']['mesh']) + '-procs', str(self.p['n_procs']['solid']) + '-procs']
    #     for n_in, n_out, d_out in zip(names_in, names_out, dirs_out):
    #         geo = read_geo(n_in).GetOutput()
    #         array = n2v(np.zeros((geo.GetNumberOfPoints(), 3)))
    #         array.SetName('Displacement')
    #         geo.GetPointData().AddArray(array)
    #
    #         os.makedirs(d_out, exist_ok=True)
    #         write_geo(os.path.join(d_out, n_out), geo)
    #
    #     shutil.copyfile('mesh_tube_fsi/fluid/mesh-surfaces/interface.vtp', self.p['root'] + '/wss_000.vtp')

    def coup_step(self, i, t, ini):
        # copy previous solution
        self.prev.sol = self.curr.sol.copy()

        # # reset solution
        # self.curr.reset()

        # predict solution for new time step
        if ini:
            self.coup_predict(i, t)

        # step 1: fluid update
        if self.p['fluid'] == 'fsi':
            self.step('fluid', i)
        elif self.p['fluid'] == 'poiseuille':
            self.poiseuille(self.p['p0'], self.p['q0'])
        if not self.curr.check(['wss', 'press']):
            return

        # relax pressure update
        # self.coup_relax('fluid', 'press', i, t, ini)

        # relax wss update
        # self.coup_relax('fluid', 'wss', i, t, ini)

        # step 2: solid update
        self.set_solid(t + 1)
        self.step('solid', i)
        if not self.curr.check(['disp']):
            return

        # relax displacement update
        self.coup_relax('solid', 'disp', i, t, ini)

        # step 3: deform mesh
        if self.p['fluid'] == 'fsi':
            self.set_mesh()
            self.step('mesh', i)

        # compute relaxation constant
        # self.coup_aitken()

    def coup_predict(self, i, t):
        if i == 1:
            # zero displacements
            self.curr.init('disp')
        else:
            # solution of converged last load step
            vec_m0 = self.log['disp'][-1][-1]
            # if ini and len(self.log[name]) > 2:
            #     # quadratically extrapolate from previous two load increments
            #     vec_m1 = self.log[name][-2][-1]
            #     vec_m2 = self.log[name][-3][-1]
            #     vec_relax = 3.0 * vec_m0 - 3.0 * vec_m1 + vec_m2
            # elif ini and len(self.log[name]) > 1:
            #     # linearly extrapolate from previous load increment
            #     vec_m1 = self.log[name][-2][-1]
            #     vec_relax = 2.0 * vec_m0 - vec_m1
            f = 'gr/gr_' + str(t + 1).zfill(3) + '.vtu'
            geo = read_geo(f).GetOutput()
            self.curr.add(('solid', 'disp', 'vol'), v2n(geo.GetPointData().GetArray('Displacement')))

    def coup_relax(self, domain, name, i, t, ini):
        curr = self.curr.get((domain, name, 'vol'))
        prev = self.prev.get((domain, name, 'vol'))
        if i == 1 or ini:
            # first step: no old solution
            vec_relax = curr
            err = 1.0
        else:
            # damp with previous iteration
            vec_relax = self.p['coup_omega'] * curr + (1.0 - self.p['coup_omega']) * prev
            err = np.linalg.norm(prev - curr) / np.linalg.norm(curr)

        # start a new sub-list for new load step
        if ini:
            self.log[name + '_new'].append([])
            self.log[name].append([])
            self.err[name].append([])

        # append current (damped) name
        self.log[name + '_new'][-1].append(curr)
        self.log[name][-1].append(vec_relax)

        # append error norm
        self.err[name][-1].append(err)

        # update solution
        self.curr.add((domain, name, 'vol'), vec_relax)

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
    fluid = 'poiseuille'
    fsg = FSG(fluid)
    fsg.run()
