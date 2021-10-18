#!/usr/bin/env python

import pdb
import numpy as np
import meshio
import sys
import vtk
import os

from collections import defaultdict
from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

# from https://github.com/StanfordCBCL/DataCuration
sys.path.append('/home/pfaller/work/osmsc/curation_scripts')
sys.path.append('/Users/pfaller/work/repos/DataCuration')
from vtk_functions import read_geo, write_geo, get_points_cells, extract_surface, threshold

# cell vertices in (cir, rad, axi)
coords = [[0, 1, 0],
        [0, 1, 1],
        [1, 1, 1],
        [1, 1, 0],
        [0, 0, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 0, 0]]
        
class Mesh():
    def __init__(self):
        # mesh parameters
        self.p = {}

        # output folder
        # self.p['f_out'] = '/home/pfaller/work/repos/svFSI_examples_fork/05-struct/03-GR/mesh_tube'
        self.p['f_out'] = '/Users/pfaller/work/repos/svFSI_examples_fork/10-FSG/mesh_tube_fsi'
        # self.p['f_out'] = 'mesh_tube'

        # cylinder size
        self.p['r_inner'] = 0.64678
        self.p['r_outer'] = 0.687
        self.p['height'] = 3.0 #0.03

        # number of cells in each dimension

        # radial g&r layer
        self.p['n_rad_gr'] = 4

        # radial transition layer
        self.p['n_rad_tran'] = 10

        # circumferential
        self.p['n_cir'] = 20

        # axial
        self.p['n_axi'] = 50

        # number of circle segments (1 = full circle, 2 = half circle, ...)
        self.p['n_seg'] = 4

        assert self.p['n_cir'] // 2 == self.p['n_cir'] / 2, 'number of elements in cir direction must be divisible by two'
        assert self.p['n_rad_tran'] >= self.p['n_cir'] // 2, 'choose number of transition elements at least half the number of cir elements'

        # initialize
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

        # initialize
        self.points = np.zeros((n_points, 3))
        self.cells = np.zeros((n_cells, 8))
        self.cosy = np.zeros((n_points, 6))
        self.fiber_dict = defaultdict(lambda: np.zeros((n_points, 3)))
        self.vol_dict = defaultdict(list)
        self.surf_dict = defaultdict(list)
        
        # file name
        self.p['fname'] = 'tube_' + str(self.p['n_rad_f']) + '+' + str(self.p['n_rad_gr']) + 'x' + str(self.p['n_cir']) + 'x' + str(self.p['n_axi']) + '.vtu'

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
                    axi = self.p['height'] * ia / self.p['n_axi']
                    rad = self.p['r_inner'] / (self.p['n_rad_f'] - 1)
                    
                    self.points[pid] = [ix * rad, iy * rad, axi]

                    self.get_surfaces_cart(pid, ia, ix, iy)
                    pid += 1

        # generate transition mesh
        for ia in range(self.p['n_axi'] + 1):
            for ir in range(self.p['n_rad_tran']):
                for ic in range(self.p['n_point_cir']):
                    # cylindrical coordinate system
                    axi = self.p['height'] * ia / self.p['n_axi']
                    cir = 2 * np.pi * ic / self.p['n_cell_cir'] / self.p['n_seg']
                    rad = self.p['r_inner'] * (ir + self.p['n_quad']) / (self.p['n_rad_f'] - 1)

                    i_trans = (ir + 1) / self.p['n_rad_tran']
                    if ic <= self.p['n_cell_cir'] // 2:
                        rad_mod = rad * ((1 - i_trans) / np.cos(cir) + i_trans)
                    else:
                        rad_mod = rad * ((1 - i_trans) / np.sin(cir) + i_trans)
                    self.points[pid] = [rad_mod * np.cos(cir), rad_mod * np.sin(cir), axi]

                    self.get_surfaces_cyl(pid, ia, ir, ic)
                    pid += 1

        # generate circular g&r mesh
            for ir in range(self.p['n_rad_gr']):
                for ic in range(self.p['n_point_cir']):
                    # cylindrical coordinate system
                    axi = self.p['height'] * ia / self.p['n_axi']
                    cir = 2 * np.pi * ic / self.p['n_cell_cir'] / self.p['n_seg']
                    rad = self.p['r_inner'] + (self.p['r_outer'] - self.p['r_inner']) * (ir + 1) / self.p['n_rad_gr']

                    self.points[pid] = [rad * np.cos(cir), rad * np.sin(cir), axi]
        
                    # store (normalized) coordinates
                    self.cosy[pid, 0] = rad # / r_outer
                    self.cosy[pid, 1] = ic / self.p['n_cell_cir'] / self.p['n_seg']
                    self.cosy[pid, 2] = ia / self.p['n_axi']
                    self.cosy[pid, 3:] = self.points[pid, :]
        
                    # store fibers
                    self.fiber_dict['axi'][pid] = [0, 0, 1]
                    self.fiber_dict['rad'][pid] = [-np.cos(cir), -np.sin(cir), 0]
                    self.fiber_dict['cir'][pid] = [-np.sin(cir), np.cos(cir), 0]

                    self.get_surfaces_cyl(pid, ia, self.p['n_rad_tran'] + ir, ic)
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
        point_data = {'GlobalNodeID': np.arange(len(self.points)) + 1,
                      'FIB_DIR': np.array(self.fiber_dict['rad']),
                      'varWallProps': self.cosy}
        for name, ids in self.surf_dict.items():
            point_data['ids_' + name] = np.zeros(len(self.points))
            point_data['ids_' + name][ids] = 1

        # assemble cell data
        cell_data = {'GlobalElementID': np.expand_dims(np.arange(len(self.cells)) + 1, axis=1)}
        for name, ids in self.vol_dict.items():
            cell_data['ids_' + name] = np.zeros(len(self.cells))
            cell_data['ids_' + name][ids] = 1
            cell_data['ids_' + name] = np.expand_dims(cell_data['ids_' + name], axis=1)
        cells = [('hexahedron', [cell]) for cell in self.cells]

        # export mesh
        mesh = meshio.Mesh(self.points, cells, point_data=point_data, cell_data=cell_data)
        mesh.write(self.p['fname'])

    def extract_svFSI(self):
        # read volume mesh in vtk
        vol = read_geo(self.p['fname']).GetOutput()

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

                # export to file
                fout = os.path.join(self.p['f_out'], f, 'mesh-surfaces', name + '.vtp')
                write_geo(fout, extract_surface(thresh.GetOutput()))

            extract_edges = vtk.vtkExtractEdges()
            extract_edges.SetInputData(vol_f)
            extract_edges.Update()

            # export volume mesh
            write_geo(os.path.join(self.p['f_out'], f, 'mesh-complete.mesh.vtu'), vol_f)

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

def main():
    mesh = Mesh()
    mesh.generate_points()
    mesh.generate_cells()
    mesh.extract_svFSI()

if __name__ == '__main__':
    main()
