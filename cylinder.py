#!/usr/bin/env python

import pdb
import numpy as np
import meshio
import sys
import vtk
import os
import json
import shutil
import platform

if platform.system() == 'Darwin':
    usr = '/Users/pfaller/'
    sys.path.append('/Users/pfaller/work/repos/DataCuration')
else:
    usr = '/home/pfaller'
    sys.path.append('/home/pfaller/work/osmsc/curation_scripts')

from collections import defaultdict
from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

from simulation import Simulation

# from https://github.com/StanfordCBCL/DataCuration
from vtk_functions import read_geo, write_geo, get_points_cells, extract_surface, threshold
from simulation_io import map_meshes

# cell vertices in (cir, rad, axi)
coords = [[0, 1, 0],
        [0, 1, 1],
        [1, 1, 1],
        [1, 1, 0],
        [0, 0, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 0, 0]]


class Mesh(Simulation):
    def __init__(self, f_params=None):
        # mesh parameters
        Simulation.__init__(self, f_params)

        # axial mesh function
        self.f = {}
        self.f['axi'] = lambda z: z
        # self.f['axi'] = lambda z: np.sqrt(z)
        # self.f['axi'] = lambda z: z**2
        # self.f['axi'] = lambda z: np.log(z + 1) / np.log(2)
        # self.f['axi'] = lambda z: np.exp(z ** 2) - 1

        # size of quadratic mesh
        self.p['n_quad'] = self.p['n_cir'] // 2 + 1

        # number of layers in fluid mesh
        self.p['n_rad_f'] = self.p['n_quad'] + self.p['n_rad_tran']

        # number of cells in circumferential direction (one more if the circle is closed)
        self.p['n_cell_cir'] = self.p['n_cir']
        self.p['n_point_cir'] = self.p['n_cir']
        self.p['n_point_eff'] = self.p['n_cir'] * self.p['n_seg']
        if self.p['n_seg'] > 1:
            self.p['n_point_cir'] += 1

        # total number of points
        n_points = (self.p['n_axi'] + 1) * (self.p['n_quad'] ** 2 + (self.p['n_rad_tran'] + self.p['n_rad_gr']) * self.p['n_point_cir'])

        # total number of cells
        n_cells = self.p['n_axi'] * ((self.p['n_quad'] - 1) ** 2 + (self.p['n_rad_tran'] + self.p['n_rad_gr']) * self.p['n_cell_cir'])

        # initialize arrays
        self.points = np.zeros((n_points, 3))
        self.cells = np.zeros((n_cells, 8))
        self.cosy = np.zeros((n_points, 9))
        self.fiber_dict = defaultdict(lambda: np.zeros((n_points, 3)))
        self.vol_dict = defaultdict(list)
        self.surf_dict = defaultdict(list)

        self.point_data = {}
        self.cell_data = {}
        
        # file name
        self.p['fname'] = 'tube_' + str(self.p['n_rad_f']) + '+' + str(self.p['n_rad_gr']) + 'x' + str(self.p['n_cir']) + 'x' + str(self.p['n_axi']) + '.vtu'

    def set_params(self):
        # output folder
        self.p['f_out'] = 'mesh_tube_fsi'

        # cylinder size
        self.p['r_inner'] = 0.64678
        self.p['r_outer'] = 0.687
        self.p['height'] = 0.3

        # number of cells in each dimension

        # radial g&r layer
        self.p['n_rad_gr'] = 4

        # radial transition layer
        self.p['n_rad_tran'] = 40

        # circumferential
        self.p['n_cir'] = 40

        # axial
        self.p['n_axi'] = 10

        # number of circle segments (1 = full circle, 2 = half circle, ...)
        self.p['n_seg'] = 4

    def validate_params(self):
        assert self.p['n_cir'] // 2 == self.p['n_cir'] / 2, 'number of elements in cir direction must be divisible by two'
        assert self.p['n_rad_tran'] >= self.p['n_cir'] // 2, 'choose number of transition elements at least half the number of cir elements'

    def get_surfaces_cyl(self, pid, ia, ir, ic):
        # store surfaces
        if ir == self.p['n_rad_tran'] - 1:
            self.surf_dict['interface'] += [pid]
        if ir == self.p['n_rad_tran'] + self.p['n_rad_gr'] - 1:
            self.surf_dict['outside'] += [pid]
        if ia == 0:
            self.surf_dict['start'] += [pid]
        if ia == self.p['n_axi']:
            self.surf_dict['end'] += [pid]
        
        # cut-surfaces only exist for cylinder sections, not the whole cylinder
        if self.p['n_seg'] > 1:
            if ic == 0:
                self.surf_dict['y_zero'] += [pid]
            if ic == self.p['n_point_cir'] - 1:
                self.surf_dict['x_zero'] += [pid]

    def get_surfaces_cart(self, pid, ia, ix, iy):
        # store surfaces
        if ia == 0:
            self.surf_dict['start'] += [pid]
        if ia == self.p['n_axi']:
            self.surf_dict['end'] += [pid]

        # cut-surfaces only exist for cylinder sections, not the whole cylinder
        if self.p['n_seg'] > 1:
            if iy == 0:
                self.surf_dict['y_zero'] += [pid]
            if ix == 0:
                self.surf_dict['x_zero'] += [pid]
    
    def generate_points(self):
        pid = 0

        # generate quadratic mesh
        for ia in range(self.p['n_axi'] + 1):
            for iy in range(self.p['n_quad']):
                for ix in range(self.p['n_quad']):
                    axi = self.p['height'] * self.f['axi'](ia / self.p['n_axi'])
                    rad = self.p['r_inner'] / (self.p['n_rad_f'] - 1)
                    
                    self.points[pid] = [ix * rad, iy * rad, axi]

                    self.get_surfaces_cart(pid, ia, ix, iy)
                    pid += 1

        # generate transition mesh
        for ia in range(self.p['n_axi'] + 1):
            for ir in range(self.p['n_rad_tran'] - 1):
                for ic in range(self.p['n_point_cir']):
                    # transition between two radii
                    i_rad = (ir + 1) / self.p['n_rad_tran']
                    rad_0 = self.p['r_inner'] * (self.p['n_quad'] - 1) / (self.p['n_rad_f'] - 1)
                    rad_1 = self.p['r_inner']

                    # cylindrical coordinate system
                    axi = self.p['height'] * self.f['axi'](ia / self.p['n_axi'])
                    cir = 2 * np.pi * ic / self.p['n_cell_cir'] / self.p['n_seg']
                    rad = rad_0 + (rad_1 - rad_0) * i_rad

                    # transition from quad mesh to circular mesh
                    i_trans = (ir + 1) / self.p['n_rad_tran']
                    if ic <= self.p['n_cell_cir'] // 2:
                        rad_mod = rad * ((1 - i_trans)**2 / np.cos(cir) + 2*i_trans - i_trans**2)
                    else:
                        rad_mod = rad * ((1 - i_trans)**2 / np.sin(cir) + 2*i_trans - i_trans**2)
                    self.points[pid] = [rad_mod * np.cos(cir), rad_mod * np.sin(cir), axi]

                    self.get_surfaces_cyl(pid, ia, ir, ic)
                    pid += 1

        # generate circular g&r mesh
            for ir in range(self.p['n_rad_gr'] + 1):
                for ic in range(self.p['n_point_cir']):
                    # cylindrical coordinate system
                    axi = self.p['height'] * self.f['axi'](ia / self.p['n_axi'])
                    cir = 2 * np.pi * ic / self.p['n_cell_cir'] / self.p['n_seg']
                    rad = self.p['r_inner'] + (self.p['r_outer'] - self.p['r_inner']) * (ir) / self.p['n_rad_gr']

                    self.points[pid] = [rad * np.cos(cir), rad * np.sin(cir), axi]
        
                    # store (normalized) coordinates
                    self.cosy[pid, 0] = rad # / r_outer
                    self.cosy[pid, 1] = ic / self.p['n_cell_cir'] / self.p['n_seg']
                    self.cosy[pid, 2] = ia / self.p['n_axi']
                    self.cosy[pid, 3:6] = self.points[pid, :]
                    # wss
                    self.cosy[pid, 6] = self.p['n_seg'] * 4.0 * 0.04 * 0.1 / np.pi / self.p['r_inner']**3
                    # time
                    self.cosy[pid, 7] = 0.0
                    # interface id
                    self.cosy[pid, 8] = ia * self.p['n_point_cir'] + ic
        
                    # store fibers
                    self.fiber_dict['axi'][pid] = [0, 0, 1]
                    self.fiber_dict['rad'][pid] = [-np.cos(cir), -np.sin(cir), 0]
                    self.fiber_dict['cir'][pid] = [-np.sin(cir), np.cos(cir), 0]

                    self.get_surfaces_cyl(pid, ia, self.p['n_rad_tran'] + ir - 1, ic)
                    pid += 1

    def generate_cells(self):
        cid = 0

        # generate quadratic mesh
        for ia in range(self.p['n_axi']):
            for iy in range(self.p['n_quad'] - 1):
                for ix in range(self.p['n_quad'] - 1):
                    ids = []
                    for c in coords:
                        ids += [(iy + c[0]) * self.p['n_quad'] + ix + c[1] + (ia + c[2]) * self.p['n_quad'] ** 2]
                    self.cells[cid] = ids
                    self.vol_dict['fluid'] += [cid]
                    cid += 1

        # generate transition mesh
        for ia in range(self.p['n_axi']):
            for ic in range(self.p['n_cell_cir']):
                ids = []
                for c in coords:
                        if c[1] == 1:
                            # circular side
                            ids += [ic + c[0] + (self.p['n_axi'] + 1) * self.p['n_quad'] ** 2 + (ia + c[2]) * (self.p['n_rad_tran'] + self.p['n_rad_gr']) * self.p['n_point_cir']]
                        else:
                            # quadratic side
                            if ic < self.p['n_cell_cir'] // 2:
                                ids += [self.p['n_quad'] - 1 + (ic + c[0]) * self.p['n_quad'] + (ia + c[2]) * self.p['n_quad'] ** 2]
                            else:
                                ids += [self.p['n_quad'] ** 2 - 1 + self.p['n_cell_cir'] // 2 - ic - c[0] + (ia + c[2]) * self.p['n_quad'] ** 2]
                self.cells[cid] = ids
                self.vol_dict['fluid'] += [cid]
                cid += 1

        # generate circular g&r mesh
        for ia in range(self.p['n_axi']):
            for ir in range(self.p['n_rad_tran'] + self.p['n_rad_gr'] - 1):
                for ic in range(self.p['n_cell_cir']):
                    ids = []
                    for c in coords:
                        ids += [(self.p['n_axi'] + 1) * self.p['n_quad'] ** 2 + (ic + c[0]) % self.p['n_point_cir'] + (ir + c[1]) * self.p['n_point_cir'] + (ia + c[2]) * (self.p['n_rad_tran'] + self.p['n_rad_gr']) * self.p['n_point_cir']]
                    self.cells[cid] = ids

                    if ir < self.p['n_rad_tran'] - 1:
                        self.vol_dict['fluid'] += [cid]
                    else:
                        self.vol_dict['solid'] += [cid]
                    cid += 1

        # assemble point data
        self.point_data = {'GlobalNodeID': np.arange(len(self.points)) + 1,
                      'FIB_DIR': np.array(self.fiber_dict['rad']),
                      'varWallProps': self.cosy}
        for name, ids in self.surf_dict.items():
            self.point_data['ids_' + name] = np.zeros(len(self.points))
            self.point_data['ids_' + name][ids] = 1

        # assemble cell data
        self.cell_data = {'GlobalElementID': np.expand_dims(np.arange(len(self.cells)) + 1, axis=1)}
        for name, ids in self.vol_dict.items():
            self.cell_data['ids_' + name] = np.zeros(len(self.cells))
            self.cell_data['ids_' + name][ids] = 1
            self.cell_data['ids_' + name] = np.expand_dims(self.cell_data['ids_' + name], axis=1)
        cells = [('hexahedron', [cell]) for cell in self.cells]

        # export mesh
        mesh = meshio.Mesh(self.points, cells, point_data=self.point_data, cell_data=self.cell_data)
        mesh.write(self.p['fname'])

    def extract_svFSI(self):
        # read volume mesh in vtk
        f_fsi = os.path.join(self.p['f_out'], self.p['fname'])
        os.makedirs(self.p['f_out'], exist_ok=True)
        shutil.move(self.p['fname'], f_fsi)
        vol = read_geo(f_fsi).GetOutput()

        surf_ids = {}
        points_inlet = []
        for f in ['solid', 'fluid']:
            # select sub-mesh
            vol_f = threshold(vol, 1, 'ids_' + f).GetOutput()

            # reset global ids
            n_array = n2v(np.arange(vol_f.GetNumberOfPoints()) + 1)
            e_array = n2v(np.arange(vol_f.GetNumberOfCells()) + 1)
            n_array.SetName('GlobalNodeID')
            e_array.SetName('GlobalElementID')
            vol_f.GetPointData().AddArray(n_array)
            vol_f.GetCellData().AddArray(e_array)

            # make output dirs
            os.makedirs(os.path.join(self.p['f_out'], f), exist_ok=True)
            os.makedirs(os.path.join(self.p['f_out'], f, 'mesh-surfaces'), exist_ok=True)

            # map point data to cell data
            p2c = vtk.vtkPointDataToCellData()
            p2c.SetInputData(vol_f)
            p2c.PassPointDataOn()
            p2c.Update()
            vol_f = p2c.GetOutput()

            # extract surfaces
            extract = vtk.vtkGeometryFilter()
            extract.SetInputData(vol_f)
            # extract.SetNonlinearSubdivisionLevel(0)
            extract.Update()
            surfaces = extract.GetOutput()

            # threshold surfaces
            for name in self.surf_dict.keys():
                # select only current surface
                thresh = vtk.vtkThreshold()
                thresh.SetInputData(surfaces)
                thresh.SetInputArrayToProcess(0, 0, 0, 0, 'ids_' + name)
                thresh.ThresholdBetween(1, 1)
                thresh.Update()
                surf = thresh.GetOutput()

                # export to file
                fout = os.path.join(self.p['f_out'], f, 'mesh-surfaces', name + '.vtp')
                write_geo(fout, extract_surface(surf))

                # get new GlobalNodeIDs of surface points
                surf_ids[f + '_' + name] = v2n(surf.GetPointData().GetArray('GlobalNodeID')).tolist()

                # store inlet points (to calculate flow profile later)
                if f == 'fluid' and name == 'start':
                    points_inlet = v2n(surf.GetPoints().GetData())

            # export volume mesh
            write_geo(os.path.join(self.p['f_out'], f, 'mesh-complete.mesh.vtu'), vol_f)

        # all nodes on inlet
        i_inlet = surf_ids['fluid_start']

        # quadratic flow profile (integrates to one, zero on the FS-interface)
        rad = np.sqrt(points_inlet[:, 0]**2 + points_inlet[:, 1]**2) / self.p['r_inner']

        profile = 'quad'
        if profile == 'quad':
            u_profile = 2 * (1 - rad ** 2)
        elif profile == 'plug':
            u_profile = (np.abs(rad - 1) > 1e-12).astype(float)
        else:
            raise ValueError('Unknown profile option: ' + profile)

        # export inflow profile: GlobalNodeID, weight
        with open(os.path.join(self.p['f_out'], 'inflow_profile.dat'), 'w') as file:
            for line, (i, v) in enumerate(zip(i_inlet, u_profile)):
                file.write(str(i) + ' ' + str(v))
                if line < len(i_inlet) - 1:
                    file.write('\n')

        # # generate quadratic mesh
        # convert_quad = False
        # if convert_quad:
        #     # read quadratic mesh
        #     f_quad = '/home/pfaller/work/repos/svFSI_examples_fork/05-struct/03-GR/mesh_tube_quad/mesh-complete.mesh.vtu'
        #     vol = read_geo(f_quad).GetOutput()

        #     # calculate cell centers
        #     centers = vtk.vtkCellCenters()
        #     centers.SetInputData(vol)
        #     centers.Update()
        #     centers.VertexCellsOn()
        #     centers.CopyArraysOn()
        #     points = v2n(centers.GetOutput().GetPoints().GetData())

        #     # radial vector
        #     rad = points
        #     rad[:, 2] = 0
        #     rad = (rad.T / np.linalg.norm(rad, axis=1)).T

        #     arr = n2v(rad)
        #     arr.SetName('FIB_DIR')
        #     vol.GetCellData().AddArray(arr)

        #     write_geo(f_quad, vol)
        #     # write_geo('test.vtu', vol)


def generate_mesh(displacement=None):
    f_params = 'in/fsg_coarse.json'
    mesh = Mesh(f_params)
    mesh.generate_points()
    mesh.generate_cells()
    mesh.extract_svFSI()
    mesh.save_params('cylinder.json')


if __name__ == '__main__':
    generate_mesh()
