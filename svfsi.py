# coding=utf-8

import pdb
import vtk
import os
import sys
import shutil
import datetime
import platform
import scipy
import shlex
import subprocess
import numpy as np
from copy import deepcopy
from collections import defaultdict

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


# names of fields in SimVascular
sv_names = {'disp': 'Displacement', 'press': 'Pressure', 'velo': 'Velocity', 'wss': 'WSS'}


class svFSI(Simulation):
    """
    svFSI base class (handles simulation runs)
    """

    def __init__(self, f_params=None):
        # simulation parameters
        Simulation.__init__(self, f_params)

        # remove old output folders
        self.fields = ['fluid', 'solid', 'mesh']
        for f in self.fields + [self.p['root']]:
            if f is not self.p['root']:
                f = str(self.p['out'][f])
            if os.path.exists(f) and os.path.isdir(f):
                shutil.rmtree(f)

        # make folders
        os.makedirs(self.p['root'])
        os.makedirs(os.path.join(self.p['root'], 'converged'))

        # generate and initialize mesh
        self.mesh_p = generate_mesh(self.p['mesh'])

        # intialize meshes
        self.mesh = {}
        for d in ['fluid', 'solid']:
            self.mesh[('int', d)] = read_geo('mesh_tube_fsi/' + d + '/mesh-surfaces/interface.vtp').GetOutput()
            self.mesh[('vol', d)] = read_geo('mesh_tube_fsi/' + d + '/mesh-complete.mesh.vtu').GetOutput()
        self.mesh[('vol', 'tube')] = read_geo('mesh_tube_fsi/' + self.mesh_p['fname']).GetOutput()

        # read points
        self.points = {}
        for d in self.mesh.keys():
            self.points[d] = v2n(self.mesh[d].GetPoints().GetData())

        # stored map nodes [src][trg]
        self.maps = {}

        # time stamp
        ct = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-')

        # output folder name
        if self.p['fsi']:
            fluid = 'fsi'
        else:
            fluid = 'poiseuille'
        self.p['f_out'] = fluid + '_res_' + ct

        # create output folder
        os.makedirs(self.p['f_out'])

        # logging
        self.converged = []
        self.log = []
        self.err = defaultdict(list)

        # current/previous solution vector at interface and in volume
        self.curr = Solution(self)
        self.prev = Solution(self)

        # generate load vector
        self.p_vec = np.linspace(1.0, self.p['fmax'], self.p['nmax'] + 1)

    def validate_params(self):
        pass

    def map(self, m):
        # if not exists, generate new map from src to trg
        if m not in self.maps:
            self.maps[m] = map_ids(self.points[m[0]], self.points[m[1]])
        return self.maps[m]

    def set_fluid(self, t):
        # fluid flow (scale by number of tube segments)
        q = self.p['q0'] / self.mesh_p['n_seg']

        # fluid pressure (scale by current pressure load step)
        p = self.p['p0'] * self.p_vec[t]

        # set bc pressure
        with open('steady_pressure.dat', 'w') as f:
            f.write('2 1\n')
            f.write('0.0 ' + str(p) + '\n')
            f.write('100.0 ' + str(p) + '\n')

        # set bc flow
        with open('steady_flow.dat', 'w') as f:
            f.write('2 1\n')
            f.write('0.0 ' + str(q) + '\n')
            f.write('100.0 ' + str(q) + '\n')

        # initialize with poiseuille solution
        # self.poiseuille(t)

        # add solution to fluid mesh
        fluid = self.mesh[('vol', 'fluid')]
        add_array(fluid, self.curr.get(('fluid', 'disp', 'vol')), sv_names['disp'])
        for f in ['press', 'velo', 'disp']:
            add_array(fluid, self.curr.get(('fluid', f, 'vol')), sv_names[f])

        # warp mesh by displacements
        fluid.GetPointData().SetActiveVectors(sv_names['disp'])
        warp = vtk.vtkWarpVector()
        warp.SetInputData(fluid)
        warp.Update()

        # write geometry to file
        write_geo(os.path.join(self.p['root'], self.p['f_fluid_geo']), warp.GetOutput())

    def set_mesh(self):
        # write general bc file
        sol = self.curr.get(('solid', 'disp', 'int'))
        points = v2n(self.mesh[('int', 'fluid')].GetPointData().GetArray('GlobalNodeID'))
        with open(self.p['f_disp'] + '.dat', 'w') as f:
            f.write('3 2 ' + str(len(sol)) + '\n')
            f.write('0.0\n')
            f.write('1.0\n')
            for n, d in zip(points, sol):
                f.write(str(n) + '\n')
                # for _ in range(2):
                f.write('0.0 0.0 0.0\n')
                for di in d:
                    f.write(str(di) + ' ')
                f.write('\n')

    def set_solid(self, t):
        # name of wall properties array
        n = 'varWallProps'

        # read solid volume mesh
        solid = self.mesh[('vol', 'solid')]

        # set wss and time
        props = v2n(solid.GetPointData().GetArray(n))
        props[:, 6] = self.curr.get(('solid', 'wss', 'vol'))
        props[:, 7] = t + 1
        add_array(solid, props, n)

        # write geometry to file
        write_geo(os.path.join(self.p['root'], self.p['f_solid_geo']), solid)

        # write interface pressure to file
        geo = self.mesh[('int', 'solid')]
        num = self.curr.get(('solid', 'press', 'int'))
        name = 'Pressure'
        add_array(geo, num, name)
        write_geo(self.p['f_load_pressure'], geo)

    def step(self, name, i, t):
        if name not in self.fields:
            raise ValueError('Unknown step option ' + name)

        # set up input files
        if name == 'fluid':
            self.set_fluid(t)
        elif name == 'solid':
            self.set_solid(t)
        elif name == 'mesh':
            self.set_mesh()

        # execute svFSI
        exe = 'mpirun -np'
        for k in ['n_procs', 'exe', 'inp']:
            exe += ' ' + str(self.p[k][name])
        with open(os.path.join(self.p['root'], name + '_' + str(i).zfill(3) + '.log'), 'w') as f:
            if self.p['debug']:
                print(exe)
                child = subprocess.run(shlex.split(exe))
            else:
                child = subprocess.run(shlex.split(exe), stdout=f, stderr=subprocess.DEVNULL)

        # check if simulation crashed and return error
        if child.returncode != 0:
            for f in self.curr.sol.keys():
                self.curr.sol[f] = None
            return True

        # read and store results
        return self.post(name, i)

    def post(self, domain, i):
        out = self.p['out'][domain]
        fname = os.path.join(out, out + '_')
        phys = domain
        i_str = str(i).zfill(3)
        src = fname + str(self.p['n_max'][domain]).zfill(3) + '.vtu'
        if domain == 'solid':
            # read current iteration
            fields = ['disp']
            src = fname + i_str + '.vtu'
        elif domain == 'fluid':
            # read converged steady state flow
            fields = ['velo', 'wss', 'press']
        elif domain == 'mesh':
            # read fully displaced mesh
            fields = ['disp']
            phys = 'fluid'
        else:
            raise ValueError('Unknown domain ' + domain)

        # check if simulation crashed
        if not os.path.exists(src):
            for f in fields:
                self.curr.sol[f] = None
                return True
        else:
            # archive results
            trg = os.path.join(self.p['root'], domain + '_out_' + i_str + '.vtu')
            shutil.copyfile(src, trg)

            # read results
            res = read_geo(src).GetOutput()

            # extract fields
            for f in fields:
                sol = v2n(res.GetPointData().GetArray(sv_names[f]))
                if f == 'wss':
                    # points on fluid interface
                    map_int = self.map((('int', 'fluid'), ('vol', 'fluid')))
                    self.curr.add((phys, f, 'int'), np.linalg.norm(sol, axis=1)[map_int])
                else:
                    self.curr.add((phys, f, 'vol'), sol)

        # archive input
        if domain in ['fluid', 'solid']:
            src = os.path.join(self.p['root'], self.p['f_' + domain + '_geo'])
            trg = os.path.join(self.p['root'], domain + '_inp_' + i_str + '.vtu')
            shutil.copyfile(src, trg)
            os.remove(src)

        return False

    def poiseuille(self, t):
        # fluid flow and pressure
        q = self.p['q0']
        p = self.p['p0'] * self.p_vec[t]

        # fluid mesh points
        points_f = deepcopy(self.points[('vol', 'fluid')]) + deepcopy(self.curr.get(('fluid', 'disp', 'vol')))

        # normalized axial coordinate
        ax = deepcopy(points_f[:, 2])
        amax = np.max(ax)
        ax /= amax

        # normalized radial coordinate
        rad = np.sqrt(points_f[:, 0] ** 2 + points_f[:, 1] ** 2)
        rmax = np.max(rad)
        rad_norm = rad / rmax

        # estimate Poiseuille resistance
        res = 8.0 * self.p['mu'] * amax / np.pi / rmax ** 4

        # estimate linear pressure gradient
        press = p * np.ones(len(rad)) + res * q * (1.0 - ax)
        self.curr.add(('fluid', 'press', 'vol'), press)

        # estimate quadratic flow profile
        velo = np.zeros(points_f.shape)
        velo[:, 2] = q / (rmax ** 2.0 * np.pi) * 2.0 * (1.0 - rad_norm ** 2.0)
        self.curr.add(('fluid', 'velo', 'vol'), velo)

        # points on fluid interface
        map_int = self.map((('int', 'fluid'), ('vol', 'fluid')))

        # make sure wss is nonzero even for q=0 (only ratio is important for g&r)
        if q == 0.0:
            q = 1.0

        # calculate wss from const Poiseuille flow
        wss = 4.0 * self.p['mu'] * q / np.pi / rad[map_int] ** 3.0
        self.curr.add(('fluid', 'wss', 'int'), wss)


class Solution:
    """
    Object to handle solutions
    """
    def __init__(self, sim):
        self.sim = sim
        self.sol = {}

        # physics of fields
        self.field2phys = {'disp': 'solid', 'press': 'fluid', 'velo': 'fluid', 'wss': 'fluid'}

        dim_vec = self.sim.points[('vol', 'tube')].shape
        dim_sca = dim_vec[0]

        # "zero" vectors. use nan where quantity is not defined
        self.zero = {'disp': np.zeros(dim_vec),
                     'velo': np.zeros(dim_vec),
                     'wss': np.ones(dim_sca) * np.nan,
                     'press': np.zeros(dim_sca) * np.nan}
        self.fields = self.zero.keys()

        # initialize everything to zero
        for f in self.fields:
            self.init(f)

    def reset(self):
        for f in self.fields:
            self.sol[f] = None

    def check(self, fields):
        for f in fields:
            if self.sol[f] is None:
                return False
            if f == 'disp':
                if np.any(np.isnan(self.sol[f])):
                    return False
        return True

    def init(self, f):
        self.sol[f] = deepcopy(self.zero[f])

    def add(self, kind, sol):
        # fluid, solid, tube
        # disp, velo, wss, press
        # vol, int
        d, f, p = kind

        map_v = self.sim.map(((p, d), ('vol', 'tube')))
        if f in ['disp', 'velo', 'press']:
            self.sol[f][map_v] = deepcopy(sol)
        elif f == 'wss':
            # wss in tube volume
            self.sol[f][map_v] = deepcopy(sol)

            # wss at fluid interface
            sol_int = self.sol[f][self.sim.map((('int', 'fluid'), ('vol', 'tube')))]

            # wss in solid volume (assume wss is constant radially)
            map_src = self.sim.map((('vol', 'solid'), ('int', 'fluid')))
            map_trg = self.sim.map((('vol', 'solid'), ('vol', 'tube')))
            self.sol[f][map_trg] = deepcopy(sol_int[map_src])
        else:
            raise ValueError(f + ' not in fields ' + self.fields)

    def get(self, kind):
        # fluid, solid, tube
        # disp, velo, wss, press
        # vol, int
        d, f, p = kind
        if self.sol[f] is None:
            raise ValueError('no solution ' + ','.join(kind))

        map_s = self.sim.map(((p, d), ('vol', 'tube')))
        return deepcopy(self.sol[f][map_s])

    def archive(self, domain, fname):
        geo = self.sim.mesh[('vol', domain)]
        for f in self.fields:
            add_array(geo, self.sol[f][self.sim.map((('vol', domain), ('vol', 'tube')))], sv_names[f])
        write_geo(fname, geo)

    def copy(self):
        solution = Solution(self.sim)
        solution.sol = deepcopy(self.sol)
        return solution


def map_ids(src, trg):
    tree = scipy.spatial.KDTree(trg)
    _, res = tree.query(src)
    return res


def add_array(geo, num, name):
    array = n2v(num)
    array.SetName(name)
    geo.GetPointData().AddArray(array)
