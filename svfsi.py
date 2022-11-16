# coding=utf-8

import pdb
import vtk
import os
import sys
import shutil
import datetime
import platform
import scipy
import scipy.stats
import shlex
import subprocess
import numpy as np
import distro
from copy import deepcopy
from collections import defaultdict

from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

from simulation import Simulation
from cylinder import generate_mesh
from smooth import smooth_wss

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


# names of fields in SimVascular
sv_names = {'disp': 'Displacement',
            'press': 'Pressure',
            'velo': 'Velocity',
            'wss': 'WSS',
            'pwss': 'pWSS',
            'jac': 'Jacobian',
            'cauchy': 'Cauchy_stress',
            'stress': 'Stress',
            'strain': 'Strain'}


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
        self.mesh[('int', 'inlet')] = read_geo('mesh_tube_fsi/fluid/mesh-surfaces/start.vtp').GetOutput()
        if self.p['tortuosity']:
            self.mesh[('int', 'perturbation')] = read_geo('mesh_tube_fsi/' + d + '/mesh-surfaces/tortuosity.vtp').GetOutput()

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
        self.err = defaultdict(list)
        self.res = []
        self.mat_V = []
        self.mat_W = []
        self.dk = defaultdict(list)
        self.dtk = defaultdict(list)

        # current/previous solution vector at interface and in volume
        self.curr = Solution(self)
        self.prev = Solution(self)

        # generate load vector
        self.p_vec = np.linspace(1.0, self.p['fmax'], self.p['nmax'] + 1)

        # relaxation parameter
        self.p['coup']['omega'] = defaultdict(list)

        # calculate reynolds number
        c1 = 2.0 * self.p['fluid']['rho'] * self.p['fluid']['q0']
        c2 = self.mesh_p['r_inner'] * np.pi * self.p['fluid']['mu']
        self.p['re'] = c1 / c2

    def set_defaults(self):
        pass

    def validate_params(self):
        pass

    def map(self, m):
        # if not exists, generate new map from src to trg
        if m not in self.maps:
            self.maps[m] = map_ids(self.points[m[0]], self.points[m[1]])
        return self.maps[m]

    def set_fluid(self, i, t):
        # fluid flow (scale by number of tube segments)
        q = deepcopy(self.p['fluid']['q0'] / self.mesh_p['n_seg'])

        # ramp up flow over the first iterations
        if t == 0:
            q *= np.min([i / 5.0, 1.0])

        # fluid pressure (scale by current pressure load step)
        p = self.p['fluid']['p0'] * self.p_vec[t]

        # set bc pressure
        with open(self.p['interfaces']['bc_pressure'], 'w') as f:
            f.write('2 1\n')
            f.write('0.0 ' + str(p) + '\n')
            f.write('100.0 ' + str(p) + '\n')

        # set bc flow
        with open(self.p['interfaces']['bc_flow'], 'w') as f:
            f.write('2 1\n')
            f.write('0.0 ' + str(-q) + '\n')
            f.write('100.0 ' + str(-q) + '\n')

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
        write_geo(os.path.join(self.p['root'], self.p['interfaces']['geo_fluid']), warp.GetOutput())

        # write inflow profile
        self.write_profile(t)

    def set_mesh(self, i):
        # write general bc file
        pre = self.prev.get(('fluid', 'disp', 'int'))
        sol = self.curr.get(('fluid', 'disp', 'int'))
        points = v2n(self.mesh[('int', 'fluid')].GetPointData().GetArray('GlobalNodeID'))
        with open(self.p['interfaces']['disp'] + '.dat', 'w') as f:
            # don't add time zero twice
            if i > 2:
                f.write('3 4 ' + str(len(sol)) + '\n')
            else:
                f.write('3 3 ' + str(len(sol)) + '\n')

            # time steps of mesh displacement (subtract 1 since no mesh sim in first first iteration)
            if i > 2:
                f.write('0.0\n')
            f.write(str(float(i - 2)) + '\n')
            f.write(str(float(i - 1)) + '\n')
            f.write(str(float(i)) + '\n')

            # write displacements of previous and current iteration
            for n, disp_new, disp_old in zip(points, sol, pre):
                f.write(str(n) + '\n')
                if i > 2:
                    dlist = [np.zeros(3), disp_old, disp_new, disp_new]
                else:
                    dlist = [disp_old, disp_new, disp_new]
                for d in dlist:
                    for di in d:
                        f.write(str(di) + ' ')
                    f.write('\n')

        # add solution to fluid mesh
        mesh = self.mesh[('vol', 'fluid')]
        disp = self.curr.get(('fluid', 'disp', 'vol'))
        if i == 1:
            disp = np.zeros(disp.shape)
        add_array(mesh, disp, sv_names['disp'])

        # write geometry to file
        write_geo(os.path.join(self.p['root'], self.p['interfaces']['geo_mesh']), mesh)

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
        write_geo(os.path.join(self.p['root'], self.p['interfaces']['geo_solid']), solid)

        # write interface pressure to file
        geo = self.mesh[('int', 'solid')]
        num = self.curr.get(('solid', 'press', 'int'))
        name = 'Pressure'
        add_array(geo, num, name)
        write_geo(self.p['interfaces']['load_pressure'], geo)

        # write interface pressure perturbation to file
        if self.p['tortuosity']:
            geo = self.mesh[('int', 'perturbation')]
            if t == 0:
                perturb = 0.0
            else:
                perturb = 0.01 * self.p['fluid']['p0']
            num = perturb * np.ones(geo.GetNumberOfPoints())
            name = 'Pressure'
            add_array(geo, num, name)
            write_geo(self.p['interfaces']['load_perturbation'], geo)

    def step(self, name, i, t):
        if name not in self.fields:
            raise ValueError('Unknown step option ' + name)

        # set up input files
        if name == 'fluid':
            self.set_fluid(i, t)
        elif name == 'solid':
            self.set_solid(t)
        elif name == 'mesh':
            self.set_mesh(i)

        # execute svFSI
        exe = 'mpirun -np '
        for k in ['n_procs', 'exe', 'inp']:
            if k == 'exe':
                exe += usr
            exe += str(self.p[k][name]) + ' '
            if k == 'n_procs':
                exe += '--use-hwthread-cpus '
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
            fields = ['disp', 'jac', 'cauchy', 'stress', 'strain']
            src = fname + i_str + '.vtu'
        elif domain == 'fluid':
            # read converged steady state flow
            fields = ['velo', 'wss', 'press']
        elif domain == 'mesh':
            # read fully displaced mesh
            fields = ['disp']
            phys = 'fluid'
            i_str = str(self.p['n_max'][domain] * (i - 1)).zfill(3)
            src = fname + i_str + '.vtu'
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
                if f == 'wss':
                    # map point data to cell data
                    c2p = vtk.vtkCellDataToPointData()
                    c2p.SetInputData(res)
                    c2p.Update()

                    # get element-wise wss maped to point data
                    sol = v2n(c2p.GetOutput().GetPointData().GetArray('E_WSS'))

                    # points on fluid interface
                    map_int = self.map((('int', 'fluid'), ('vol', 'fluid')))

                    # only store magnitude of wss at interface (doesn't make sense elsewhere)
                    self.curr.add((phys, f, 'int'), sol[map_int])

                    # only for logging, store svFSI point-wise wss
                    sol = v2n(res.GetPointData().GetArray(sv_names[f]))
                    self.curr.add((phys, 'pwss', 'int'), np.linalg.norm(sol[map_int], axis=1))
                else:
                    sol = v2n(res.GetPointData().GetArray(sv_names[f]))
                    self.curr.add((phys, f, 'vol'), sol)

        # archive input
        # todo: remove all outputs after post-processing
        if domain in ['fluid', 'solid']:
            src = os.path.join(self.p['root'], self.p['interfaces']['geo_' + domain])
            trg = os.path.join(self.p['root'], domain + '_inp_' + i_str + '.vtu')
            shutil.copyfile(src, trg)
            # os.remove(src)
        return False

    def get_profile(self, x_norm, rad_norm, t):
        # quadratic flow profile (integrates to one, zero on the FS-interface)
        u_profile = 2.0 * (1.0 - rad_norm ** 2.0)

        # custom flow profile
        if 'profile' in self.p:
            # time factor
            f_time = t / self.p['nmax']

            # limits
            beta_min = self.p['profile']['beta_min']
            beta_max = self.p['profile']['beta_max']

            # beta distribution for x-bias
            beta = beta_min + (beta_max - beta_min) * f_time
            bias = scipy.stats.beta.pdf(x_norm, 2, beta)
            bias0 = scipy.stats.beta.pdf(x_norm, 2, beta_min)

            # normalize with initial profile
            pos = bias0 != 0.0
            bias[pos] /= bias0[pos]

            u_profile *= bias
        return u_profile

    def write_profile(self, t):
        # GlobalNodeID of inlet within fluid mesh
        i_inlet = self.map((('int', 'inlet'), ('vol', 'fluid')))

        # inlet points in current configuration
        points = deepcopy(self.points[('vol', 'fluid')])[i_inlet]

        # radial coordinate [0, 1]
        rad = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        rad_norm = rad / np.max(rad)
        area = np.max(rad)**2 * np.pi

        # normalized x coordinate [0, 1]
        x = points[:, 0]
        x_norm = (1.0 + x / np.max(x)) / 2.0

        # get flow profile at inlet
        u_profile = 1.0 / area * self.get_profile(x_norm, rad_norm, t)

        # export inflow profile: GlobalNodeID, weight
        with open('inflow_profile.dat', 'w') as file:
            for line, (i, v) in enumerate(zip(i_inlet, u_profile)):
                file.write(str(i + 1) + ' ' + str(- v))
                if line < len(i_inlet) - 1:
                    file.write('\n')

    def poiseuille(self, t):
        # fluid flow and pressure
        q = self.p['fluid']['q0']
        p = self.p['fluid']['p0'] * self.p_vec[t]

        # fluid mesh points in reference configuration
        points_r = deepcopy(self.points[('vol', 'fluid')])

        # fluid mesh points in current configuration
        points_f = deepcopy(points_r) + deepcopy(self.curr.get(('fluid', 'disp', 'vol')))
        n_points = points_f.shape[0]

        # normalized axial coordinate
        ax = deepcopy(points_f[:, 2])
        amax = np.max(ax)
        ax /= amax

        # normalized x coordinate [0, 1]
        x = points_f[:, 0]
        x_norm = (1.0 + x / np.max(x)) / 2.0

        # radial coordinate of all points
        rad = np.sqrt(points_f[:, 0] ** 2 + points_f[:, 1] ** 2)

        # minimum interface radius
        rmin = np.min(rad[self.map((('int', 'fluid'), ('vol', 'fluid')))])

        # estimate Poiseuille resistance
        res = 8.0 * self.p['fluid']['mu'] * amax / np.pi / rmin ** 4

        # estimate linear pressure gradient
        press = p * np.ones(len(rad)) + res * q * (1.0 - ax)
        self.curr.add(('fluid', 'press', 'vol'), press)

        # get local cross-sectional area and maximum radius (assuming a structured mesh)
        z_slices = np.unique(points_r[:, 2])
        areas = np.zeros(n_points)
        rad_norm = np.zeros(n_points)
        for z in z_slices:
            i_slice = points_r[:, 2] == z
            rmax = np.max(rad[i_slice])
            areas[i_slice] = rmax ** 2.0 * np.pi
            rad_norm[i_slice] = rad[i_slice] / rmax
        assert not np.any(areas == 0.0), 'area zero'

        # estimate flow profile
        velo = np.zeros(points_f.shape)
        velo[:, 2] = q / areas * self.get_profile(x_norm, rad_norm, t)
        self.curr.add(('fluid', 'velo', 'vol'), velo)

        # points on fluid interface
        map_int = self.map((('int', 'fluid'), ('vol', 'fluid')))

        # make sure wss is nonzero even for q=0 (only ratio is important for g&r)
        if q == 0.0:
            q = 1.0

        # calculate wss from const Poiseuille flow
        # todo: use actual profile (and local gradient??)
        wss = 4.0 * self.p['fluid']['mu'] * q / np.pi / rad[map_int] ** 3.0
        self.curr.add(('fluid', 'wss', 'int'), wss)

    def get_re(self):
        return


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
        dim_ten = (dim_sca, 6)

        # "zero" vectors. use nan where quantity is not defined
        self.zero = {'disp': np.zeros(dim_vec),
                     'velo': np.zeros(dim_vec),
                     'wss': np.ones(dim_sca) * np.nan,
                     'pwss': np.ones(dim_sca) * np.nan,
                     'press': np.zeros(dim_sca) * np.nan,
                     'jac': np.zeros(dim_sca) * np.nan,
                     'cauchy': np.zeros(dim_ten) * np.nan,
                     'stress': np.zeros(dim_ten) * np.nan,
                     'strain': np.zeros(dim_ten) * np.nan}
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
        if f in ['disp', 'velo', 'press', 'jac', 'cauchy', 'stress', 'strain']:
            self.sol[f][map_v] = deepcopy(sol)
        elif 'wss' in f:
            # wss in tube volume
            self.sol[f][map_v] = deepcopy(sol)

            # wss at fluid interface
            sol_int = self.sol[f][self.sim.map((('int', 'fluid'), ('vol', 'tube')))]

            # wss in solid volume (assume wss is constant radially)
            map_src = self.sim.map((('vol', 'solid'), ('int', 'fluid')))
            map_trg = self.sim.map((('vol', 'solid'), ('vol', 'tube')))
            self.sol[f][map_trg] = deepcopy(sol_int[map_src])
        else:
            raise ValueError(f + ' not in fields ' + str(list(self.fields)))

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
