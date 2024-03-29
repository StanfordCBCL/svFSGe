#!/usr/bin/env python

import pdb
import numpy as np
import meshio
import vtk
import os
import json
import shutil
import argparse
import subprocess
from collections import defaultdict

from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

from vtk_functions import (
    read_geo,
    write_geo,
    extract_surface,
    threshold,
    clean,
)

# cell vertices in (cir, rad, axi)
coords = [
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 1],
    [1, 1, 0],
    [0, 0, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 0, 0],
]


def spacing(z, p):
    zones = np.array(p["zones"])
    densities = np.array(p["density"])

    lengths = np.cumsum(zones)
    bounds = np.cumsum(densities)

    for i in range(len(zones)):
        a = 0.0
        if z <= bounds[i]:
            if i > 0:
                a += lengths[i - 1]
                z -= bounds[i - 1]
            a += z * zones[i] / densities[i]
            return a


def spacing_var(z, e=6):
    if z > 0.5:
        coeff = (z - 0.5)**e
    else:
        coeff = - (-z + 0.5)**e
    return 0.5 * (z + 0.5 + coeff / 0.5**(e - 1.0))


class Simulation:
    """
    Base class to handle parameters
    """

    def __init__(self, f_params=None):
        # mesh parameters
        self.p = {}

        # set mesh parameters (load from json file if given)
        if f_params is None:
            self.set_params()
        else:
            self.load_params(f_params)

        # set default parameters
        self.set_defaults()

        # validate parameters
        self.validate_params()

    def set_params(self):
        """
        Manually set parameters (e.g. to default values)
        """
        raise ValueError("Implement set_params in derived class")

    def set_defaults(self):
        """
        Validate parameters
        """
        # raise ValueError("Implement set_defaults in derived class")
        pass

    def validate_params(self):
        """
        Validate parameters
        """
        # raise ValueError("Implement validate_params in derived class")
        pass

    def load_params(self, file_name):
        """
        Load parameters from json file_name
        """
        # read parameters from json file
        with open(file_name, "r") as file:
            param = json.load(file)

        # set parameters
        for k, v in param.items():
            self.p[k] = v

    def save_params(self, file_name):
        """
        Save parameters to json file_name
        """
        # save parameters to json file
        file_name = os.path.join(self.p["f_out"], file_name)
        with open(file_name, "w") as file:
            json.dump(self.p, file, indent=4, sort_keys=True)


class Mesh(Simulation):
    def __init__(self, f_params=None):
        # mesh parameters
        Simulation.__init__(self, f_params)

        # size of quadratic mesh
        if self.p["n_seg"] == 4:
            # number of layers in quadratic mesh
            self.p["n_quad"] = self.p["n_cir"] // 2 + 1

            # number of layers in fluid mesh
            self.p["n_rad_f"] = self.p["n_quad"] + self.p["n_rad_tran"]
        if self.p["n_seg"] == 1:
            # number of layers in quadratic mesh
            self.p["n_quad"] = self.p["n_cir"] // 4 + 1

            # number of layers in fluid mesh
            self.p["n_rad_f"] = self.p["n_quad"] + self.p["n_rad_tran"] * 2

        # number of cells in circumferential direction (one more if the circle is closed)
        self.p["n_cell_cir"] = self.p["n_cir"]
        self.p["n_point_cir"] = self.p["n_cir"]
        self.p["n_point_eff"] = self.p["n_cir"] * self.p["n_seg"]
        if self.p["n_seg"] > 1:
            self.p["n_point_cir"] += 1

        # total number of points
        n_points = (self.p["n_axi"] + 1) * (
            self.p["n_quad"] ** 2
            + (self.p["n_rad_tran"] + self.p["n_rad_gr"]) * self.p["n_point_cir"]
        )

        # total number of cells
        n_cells = self.p["n_axi"] * (
            (self.p["n_quad"] - 1) ** 2
            + (self.p["n_rad_tran"] + self.p["n_rad_gr"]) * self.p["n_cell_cir"]
        )

        # initialize arrays
        self.points = np.zeros((n_points, 3))
        self.cells = np.zeros((n_cells, 8), dtype=int)
        self.cosy = np.zeros((n_points, 13))
        self.fiber_dict = defaultdict(lambda: np.zeros((n_points, 3)))
        self.vol_dict = defaultdict(list)
        self.surf_dict = defaultdict(list)

        self.point_data = {}
        self.cell_data = {}

        # file name
        self.p["fname"] = "tube.vtu"

    def set_defaults(self):
        # axial mesh function
        self.f = {}
        if "adapt" in self.p:
            self.f["axi"] = lambda z: spacing(z, self.p["adapt"])
        elif "exp" in self.p:
            self.f['axi'] = lambda z: spacing_var(z, self.p["exp"])
            # self.f['cir'] = lambda z: (spacing_var((z*2)%1, self.p["exp"]) + (z>=0.5)) * 0.5
            self.f["cir"] = lambda z: z
        # self.f['axi'] = lambda z: np.sqrt(z)
        # self.f['axi'] = lambda z: z**2
        # self.f['axi'] = lambda z: np.log(z + 1) / np.log(2)
        # self.f['axi'] = lambda z: np.exp(z ** 2) + 0.5
        else:
            self.f["axi"] = lambda z: z
        if "boundary" not in self.p:
            self.p["boundary"] = {"n": 0, "thickness": 0.0}

    def validate_params(self):
        if self.p["n_seg"] == 1:
            assert divisible(
                self.p["n_cir"], 8
            ), "number of elements in cir direction must be divisible by eight"
            assert (
                self.p["n_rad_tran"] - self.p["boundary"]["n"] >= self.p["n_cir"] // 4
            ), "choose number of transition elements at least a quarter the number of cir elements"
        elif self.p["n_seg"] == 4:
            assert divisible(
                self.p["n_cir"], 2
            ), "number of elements in cir direction must be divisible by two"
            assert (
                self.p["n_rad_tran"] - self.p["boundary"]["n"] >= self.p["n_cir"] // 2
            ), "choose number of transition elements at least half the number of cir elements"
        else:
            raise ValueError("FSI mesh only possible for full or quarter circles")
        assert divisible(
            self.p["n_rad_tran"], 2
        ), "number of transition elements must be divisible by two"

        # adaptive meshing
        if "adapt" in self.p:
            assert np.sum(self.p["adapt"]["zones"]) == 1.0, "zones must sum up to one"
            assert (
                np.sum(self.p["adapt"]["density"]) == 1.0
            ), "densities must sum up to one"
            assert len(self.p["adapt"]["zones"]) == len(
                self.p["adapt"]["density"]
            ), "zones and densities must have equal length"

            # todo: check if discretization size matches with densities
            # n_adapt = 0
            # for i in range(len(self.p['adapt']['zones'])):

        # boundary layer
        assert (
            self.p["n_rad_tran"] - self.p["boundary"]["n"] >= 1
        ), "boundary layer too large"
        if self.p["boundary"]["thickness"] > 0.0:
            assert self.p["boundary"]["n"] > 0, "define at least one boundary layer"

    def get_surfaces_cyl(self, pid, ia, ir, ic):
        # store surfaces
        if ir == self.p["n_rad_tran"] - 1:
            self.surf_dict["interface"] += [pid]
        if ir == self.p["n_rad_tran"] + self.p["n_rad_gr"] - 1:
            self.surf_dict["outside"] += [pid]
        if ia == 0:
            self.surf_dict["start"] += [pid]
        if ia == self.p["n_axi"]:
            self.surf_dict["end"] += [pid]

        # cut-surfaces only exist for cylinder sections, not the whole cylinder
        if self.p["n_seg"] > 1:
            if ic == 0:
                self.surf_dict["y_zero"] += [pid]
            if ic == self.p["n_point_cir"] - 1:
                self.surf_dict["x_zero"] += [pid]

        if self.p["n_seg"] == 1:
            # surfaces to apply tortuosity perturbation
            if ir == self.p["n_rad_tran"] - 1:
                if (
                    abs(ia - self.p["n_axi"] // 4) <= 1
                    and abs(ic - 3 * self.p["n_point_cir"] // 4) <= 1
                ):
                    self.surf_dict["tortuosity"] += [pid]
                if (
                    abs(ia - 3 * self.p["n_axi"] // 4) <= 1
                    and abs(ic - self.p["n_point_cir"] // 4) <= 1
                ):
                    self.surf_dict["tortuosity"] += [pid]

            # surfaces to prevent x- and y-movement
            if ia <= 1 or ia >= self.p["n_axi"] - 1:
                if (
                    ic == self.p["n_point_cir"] // 4
                    or ic == 3 * self.p["n_point_cir"] // 4
                ):
                    self.surf_dict["x_zero"] += [pid]
                if ic == self.p["n_point_cir"] // 2 or ic == 0:
                    self.surf_dict["y_zero"] += [pid]

    def get_surfaces_cart(self, pid, ia, ix, iy):
        # store surfaces
        if ia == 0:
            self.surf_dict["start"] += [pid]
        if ia == self.p["n_axi"]:
            self.surf_dict["end"] += [pid]

        # cut-surfaces only exist for cylinder sections, not the whole cylinder
        if self.p["n_seg"] > 1:
            if iy == 0:
                self.surf_dict["y_zero"] += [pid]
            if ix == 0:
                self.surf_dict["x_zero"] += [pid]

    def generate_points(self):
        pid = 0

        # generate quadratic mesh
        ri = self.p["r_inner"] - self.p["boundary"]["thickness"]
        nr = self.p["n_rad_f"] - 1 - self.p["boundary"]["n"]
        rad = ri / nr
        delta = (self.p["n_quad"] - 1) * ri / nr

        # offset from center
        if self.p["n_seg"] == 1:
            x0 = -delta
            y0 = -delta
            rad *= 2.0
        elif self.p["n_seg"] == 4:
            x0 = 0.0
            y0 = 0.0
        else:
            raise ValueError("not implemented for n_seg=" + str(self.p["n_seg"]))

        for ia in range(self.p["n_axi"] + 1):
            axi = self.p["height"] * self.f["axi"](ia / self.p["n_axi"])
            for iy in range(self.p["n_quad"]):
                for ix in range(self.p["n_quad"]):
                    self.points[pid] = [x0 + ix * rad, y0 + iy * rad, axi]
                    self.get_surfaces_cart(pid, ia, ix, iy)
                    pid += 1

        # generate transition mesh
        for ia in range(self.p["n_axi"] + 1):
            for ir in range(self.p["n_rad_tran"] - 1):
                for ic in range(self.p["n_point_cir"]):
                    # boundary index in case of boundary layer
                    ib = ir - (self.p["n_rad_tran"] - self.p["boundary"]["n"] - 1)

                    # transition between two radii
                    i_rad = (ir + 1) / (self.p["n_rad_tran"] - self.p["boundary"]["n"])
                    rad_1 = self.p["r_inner"] - self.p["boundary"]["thickness"]
                    rad_0 = (
                        rad_1
                        * (self.p["n_quad"] - 1)
                        / (self.p["n_rad_f"] - 1 - self.p["boundary"]["n"])
                    )

                    # cylindrical coordinate system
                    axi = self.p["height"] * self.f["axi"](ia / self.p["n_axi"])
                    cir = 2 * np.pi * self.f["cir"](ic / self.p["n_cell_cir"] / self.p["n_seg"])
                    rad = rad_0 + (rad_1 - rad_0) * i_rad

                    # transition from quad mesh to circular mesh
                    i_trans = (ir + 1) / (self.p["n_rad_tran"] - self.p["boundary"]["n"])

                    # in which octant is the point located?
                    oct = int((ic / self.p["n_cell_cir"] / self.p["n_seg"]) * 8.0) % 8

                    # offset so radial lines don't point to the center but the interface between quad and circular mesh
                    dx = 0.0
                    dy = 0.0

                    # check if point not on axis
                    if (ic * 4.0 / self.p["n_cell_cir"] / self.p["n_seg"]) % 1 != 0:
                        cir90 = cir % (np.pi / 2.0)
                        if self.p["n_seg"] == 1:
                            nq = (self.p["n_quad"] - 1) / 2
                        else:
                            nq = self.p["n_quad"] - 1
                        icm = (ic % nq) / nq
                        if oct % 2 == 0:
                            rad_mod = rad * ((1.0 - i_trans) ** 2 / np.cos(cir90) + 2.0 * i_trans - i_trans**2)
                            dd = (1.0 - i_trans) * (icm - np.tan(cir90))
                        else:
                            rad_mod = rad * ((1.0 - i_trans) ** 2 / np.sin(cir90) + 2.0 * i_trans - i_trans**2)
                            dd = (1.0 - i_trans) * (1.0 - icm + np.tan(cir90 - np.pi / 2.0))
                        if oct in [2, 4, 5, 7]:
                            dd *= -1
                        if oct in [1, 2, 5, 6]:
                            dx = dd
                        else:
                            dy = dd
                    else:
                        rad_mod = rad

                    # perfectly circular boundary layer
                    if ib >= 0:
                        ib_ratio = 1.0 - ib / (self.p["boundary"]["n"])
                        rad_mod = (
                            self.p["r_inner"]
                            - ib_ratio * self.p["boundary"]["thickness"]
                        )
                    self.points[pid] = [
                        rad_mod * np.cos(cir) + rad_0 * dx,
                        rad_mod * np.sin(cir) + rad_0 * dy,
                        axi,
                    ]

                    self.get_surfaces_cyl(pid, ia, ir, ic)
                    pid += 1

            # generate circular g&r mesh
            for ir in range(self.p["n_rad_gr"] + 1):
                for ic in range(self.p["n_point_cir"]):
                    # cylindrical coordinate system
                    axi = self.p["height"] * self.f["axi"](ia / self.p["n_axi"])
                    cir = 2 * np.pi * self.f["cir"](ic / self.p["n_cell_cir"] / self.p["n_seg"])
                    rad = (
                        self.p["r_inner"]
                        + (self.p["r_outer"] - self.p["r_inner"])
                        * (ir)
                        / self.p["n_rad_gr"]
                    )

                    self.points[pid] = [rad * np.cos(cir), rad * np.sin(cir), axi]

                    # store (normalized) coordinates
                    self.cosy[pid, 0] = rad  # / r_outer
                    self.cosy[pid, 1] = ic / self.p["n_cell_cir"] / self.p["n_seg"]
                    self.cosy[pid, 2] = ia / self.p["n_axi"]
                    self.cosy[pid, 3:6] = self.points[pid, :]
                    # wss
                    self.cosy[pid, 6] = (
                        self.p["n_seg"]
                        * 4.0
                        * 0.04
                        * 0.1
                        / np.pi
                        / self.p["r_inner"] ** 3
                    )
                    # time
                    self.cosy[pid, 7] = 0.0
                    # interface id
                    self.cosy[pid, 8] = ia * self.p["n_point_cir"] + ic
                    # interface node id, dwss, wss_old
                    self.cosy[pid, 9:] = 0.0
                    self.cosy[pid, 12] = 1337

                    # store fibers
                    self.fiber_dict["axi"][pid] = [0, 0, 1]
                    self.fiber_dict["rad"][pid] = [-np.cos(cir), -np.sin(cir), 0]
                    self.fiber_dict["cir"][pid] = [-np.sin(cir), np.cos(cir), 0]

                    self.get_surfaces_cyl(pid, ia, self.p["n_rad_tran"] + ir - 1, ic)
                    pid += 1

        # add curve
        if "curve" in self.p:
            curve = np.cos(self.points[:, 2] / np.max(self.points[:, 2]) * 2.0 * np.pi)
            self.points[:, 0] += self.p["curve"] * (1.0 - curve) / 2.0

    def generate_cells(self):
        cid = 0

        # generate quadratic mesh
        for ia in range(self.p["n_axi"]):
            for iy in range(self.p["n_quad"] - 1):
                for ix in range(self.p["n_quad"] - 1):
                    ids = []
                    for c in coords:
                        ids += [
                            (iy + c[0]) * self.p["n_quad"]
                            + ix
                            + c[1]
                            + (ia + c[2]) * self.p["n_quad"] ** 2
                        ]
                    self.cells[cid] = ids
                    self.vol_dict["fluid"] += [cid]
                    cid += 1

        # generate transition mesh
        for ia in range(self.p["n_axi"]):
            for ic in range(self.p["n_cell_cir"]):
                ids = []

                # number of segments per quad edge
                ns = self.p["n_cell_cir"] * self.p["n_seg"] // 4

                # quad edges (looking in negative z): 0: east, 1: north, 2: west, 3: south
                qua = (
                    int(
                        (ic + self.p["n_cell_cir"] // 8)
                        / self.p["n_cell_cir"]
                        / self.p["n_seg"]
                        * 4.0
                    )
                    % 4
                )

                # circumferential coordinate along each quad edge
                icq = (ic + ns // 2 - qua * ns) % self.p["n_cell_cir"]

                # loop element nodes
                for c in coords:
                    # quarter circle
                    if self.p["n_seg"] == 4:
                        # circular side
                        if c[1] == 1:
                            ids += [
                                ic
                                + c[0]
                                + (self.p["n_axi"] + 1) * self.p["n_quad"] ** 2
                                + (ia + c[2])
                                * (self.p["n_rad_tran"] + self.p["n_rad_gr"])
                                * self.p["n_point_cir"]
                            ]

                        # quadratic side
                        else:
                            if ic < self.p["n_cell_cir"] // 2:
                                ids += [
                                    self.p["n_quad"]
                                    - 1
                                    + (ic + c[0]) * self.p["n_quad"]
                                    + (ia + c[2]) * self.p["n_quad"] ** 2
                                ]
                            else:
                                ids += [
                                    self.p["n_quad"] ** 2
                                    - 1
                                    + self.p["n_cell_cir"] // 2
                                    - ic
                                    - c[0]
                                    + (ia + c[2]) * self.p["n_quad"] ** 2
                                ]
                    # full circle
                    elif self.p["n_seg"] == 1:
                        # circular side
                        if c[1] == 1:
                            # starting node
                            offset = (self.p["n_axi"] + 1) * self.p["n_quad"] ** 2

                            # axial step
                            fa = (self.p["n_rad_tran"] + self.p["n_rad_gr"]) * self.p[
                                "n_point_cir"
                            ]

                            # closing the circle
                            if qua == 0 and ic + c[0] == self.p["n_cell_cir"]:
                                fc = 0
                            else:
                                fc = 1
                            ids += [offset + (ic + c[0]) * fc + (ia + c[2]) * fa]
                        # quadratic side
                        else:
                            # axial step
                            fa = self.p["n_quad"] ** 2
                            # east
                            if qua == 0:
                                offset = self.p["n_quad"] - 1
                                fc = self.p["n_quad"]
                            # north
                            elif qua == 1:
                                offset = self.p["n_quad"] ** 2 - 1
                                fc = -1
                            # west
                            elif qua == 2:
                                offset = self.p["n_quad"] * (self.p["n_quad"] - 1)
                                fc = -self.p["n_quad"]
                            # south
                            elif qua == 3:
                                offset = 0
                                fc = 1
                            ids += [offset + (icq + c[0]) * fc + (ia + c[2]) * fa]
                    else:
                        raise ValueError(
                            "not implemented for n_seg=" + str(self.p["n_seg"])
                        )
                self.cells[cid] = ids
                self.vol_dict["fluid"] += [cid]
                cid += 1
        # generate circular g&r mesh
        for ia in range(self.p["n_axi"]):
            for ir in range(self.p["n_rad_tran"] + self.p["n_rad_gr"] - 1):
                for ic in range(self.p["n_cell_cir"]):
                    ids = []
                    for c in coords:
                        ids += [
                            (self.p["n_axi"] + 1) * self.p["n_quad"] ** 2
                            + (ic + c[0]) % self.p["n_point_cir"]
                            + (ir + c[1]) * self.p["n_point_cir"]
                            + (ia + c[2])
                            * (self.p["n_rad_tran"] + self.p["n_rad_gr"])
                            * self.p["n_point_cir"]
                        ]
                    self.cells[cid] = ids

                    if ir < self.p["n_rad_tran"] - 1:
                        self.vol_dict["fluid"] += [cid]
                    else:
                        self.vol_dict["solid"] += [cid]
                    cid += 1

        # label according to cell centers
        cell_cent = np.mean(self.points[self.cells], axis=1)
        for i, c in enumerate(["x", "y"]):
            self.vol_dict[c + "_seg"] = np.where(cell_cent[:, i] > 0.0)[0].tolist()

        # assemble point data
        self.point_data = {
            "GlobalNodeID": np.arange(len(self.points)) + 1,
            "FIB_DIR": np.array(self.fiber_dict["rad"]),
            "gr_properties": self.cosy,
        }
        for name, ids in self.surf_dict.items():
            self.point_data["ids_" + name] = np.zeros(len(self.points), dtype=np.int32)
            self.point_data["ids_" + name][ids] = 1

        # add insult profile
        # axi = self.points[:, 2]
        # axi -= np.max(axi)
        # axi /= np.max(axi) * 8
        f_axi = np.exp(-np.power(np.abs((self.cosy[:, 2] - 0.5) * 4.0), 2))
        f_cir = np.exp(-np.power(np.abs((self.cosy[:, 1] - 0.5) / 0.55), 6))
        self.point_data["insult"] = f_axi * f_cir

        # assemble cell data
        self.cell_data = {
            "GlobalElementID": np.expand_dims(np.arange(len(self.cells)) + 1, axis=1)
        }
        for name, ids in self.vol_dict.items():
            self.cell_data["ids_" + name] = np.zeros(len(self.cells), dtype=np.int32)
            self.cell_data["ids_" + name][ids] = 1
            self.cell_data["ids_" + name] = np.expand_dims(
                self.cell_data["ids_" + name], axis=1
            )
        cells = [("hexahedron", [cell]) for cell in self.cells]

        # export mesh
        mesh = meshio.Mesh(
            self.points, cells, point_data=self.point_data, cell_data=self.cell_data
        )
        mesh.write(self.p["fname"])

    def extract_svFSI(self):
        # read volume mesh in vtk
        f_fsi = os.path.join(self.p["f_out"], self.p["fname"])
        os.makedirs(self.p["f_out"], exist_ok=True)
        shutil.move(self.p["fname"], f_fsi)
        vol = read_geo(f_fsi).GetOutput()

        surf_ids = {}
        points_inlet = []
        for f in ["solid", "fluid"]:
            # select sub-mesh
            vol_f = threshold(vol, 1, "ids_" + f).GetOutput()

            # reset global ids
            n_array = n2v(np.arange(vol_f.GetNumberOfPoints()).astype(np.int32) + 1)
            e_array = n2v(np.arange(vol_f.GetNumberOfCells()).astype(np.int32) + 1)
            n_array.SetName("GlobalNodeID")
            e_array.SetName("GlobalElementID")
            vol_f.GetPointData().AddArray(n_array)
            vol_f.GetCellData().AddArray(e_array)

            # make output dirs
            os.makedirs(os.path.join(self.p["f_out"], f), exist_ok=True)
            os.makedirs(
                os.path.join(self.p["f_out"], f, "mesh-surfaces"), exist_ok=True
            )

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
                # interior quad elements
                if self.p["n_seg"] == 1 and "_zero" in name:
                    # threshold circle segments first
                    thresh = vtk.vtkThreshold()
                    thresh.SetInputData(vol_f)
                    thresh.SetInputArrayToProcess(0, 0, 0, 1, "ids_" + name[0] + "_seg")
                    thresh.SetUpperThreshold(1)
                    thresh.SetLowerThreshold(1)
                    thresh.Update()

                    # extract surfaces
                    extract = vtk.vtkGeometryFilter()
                    extract.SetInputData(thresh.GetOutput())
                    extract.SetNonlinearSubdivisionLevel(0)
                    extract.Update()
                    inp = extract.GetOutput()
                else:
                    inp = surfaces

                # select only current surface
                thresh = vtk.vtkThreshold()
                thresh.SetInputData(inp)
                thresh.SetInputArrayToProcess(0, 0, 0, 0, "ids_" + name)
                thresh.SetUpperThreshold(1)
                thresh.SetLowerThreshold(1)
                thresh.Update()
                surf = thresh.GetOutput()
                if surf.GetNumberOfPoints() > 0:
                    surf = clean(extract_surface(surf))

                fout = os.path.join(self.p["f_out"], f, "mesh-surfaces", name + ".vtp")
                write_geo(fout, extract_surface(surf))

                # get new GlobalNodeIDs of surface points
                surf_ids[f + "_" + name] = v2n(
                    surf.GetPointData().GetArray("GlobalNodeID")
                ).tolist()

                # store inlet points (to calculate flow profile later)
                if f == "fluid" and name == "start":
                    points_inlet = v2n(surf.GetPoints().GetData())

            # export volume mesh
            write_geo(os.path.join(self.p["f_out"], f, "mesh-complete.mesh.vtu"), vol_f)

        # all nodes on inlet
        i_inlet = surf_ids["fluid_start"]

        # quadratic flow profile (integrates to one, zero on the FS-interface)
        rad = (
            np.sqrt(points_inlet[:, 0] ** 2 + points_inlet[:, 1] ** 2)
            / self.p["r_inner"]
        )

        profile = "quad"
        if profile == "quad":
            u_profile = 2 * (1 - rad**2)
        elif profile == "plug":
            u_profile = (np.abs(rad - 1) > 1e-12).astype(float)
        else:
            raise ValueError("Unknown profile option: " + profile)

        # export inflow profile: GlobalNodeID, weight
        if True:
            with open(os.path.join(self.p["f_out"], "inflow_profile.dat"), "w") as file:
                for line, (i, v) in enumerate(zip(i_inlet, u_profile)):
                    file.write(str(i) + " " + str(-v))
                    if line < len(i_inlet) - 1:
                        file.write("\n")

        if "quad" in self.p and self.p["quad"]:
            fpath_lin = self.p["f_out"] + "_lin"
            shutil.move(self.p["f_out"], fpath_lin)
            os.makedirs(self.p["f_out"])
            
            surfaces = {"solid": ["start", "end", "interface", "outside"], "fluid": ["start", "end", "interface"]}
            for f in ["solid", "fluid"]:
                txt = "# numSpatialDim\n3\n\n# meshFilePath\n"
                f_out = os.path.join("..", fpath_lin, f, "mesh-complete.mesh.vtu")
                txt += f_out + "\n\n# faceFilesPaths\n"
                for surf in surfaces[f]:
                    f_out = os.path.join("..", fpath_lin, f, "mesh-surfaces", surf + ".vtp")
                    txt += f_out + "\n"

                fname = "convert.txt"
                with open(os.path.join(self.p["f_out"], fname), "w") as text_file:
                    text_file.write(txt)

                # call quad conversion script (select y, 2)
                exe = "/Users/pfaller/work/repos/useful_codes/gambitToVTK/bin/convertMesh.exe"
                p = subprocess.call([exe, "convert.txt"], cwd=self.p["f_out"])

                os.makedirs(os.path.join(self.p["f_out"], f))
                shutil.move(os.path.join(self.p["f_out"], "mesh-complete.mesh.vtu"), os.path.join(self.p["f_out"], f))
                shutil.move(os.path.join(self.p["f_out"], "mesh-surfaces"), os.path.join(self.p["f_out"], f))

            # add minimal mesh properties
            for f in ["solid", "fluid"]:
                props(os.path.join(self.p["f_out"], f, "mesh-complete.mesh.vtu"))
                for surf in surfaces[f]:
                    props(os.path.join(self.p["f_out"], f, "mesh-surfaces", surf + ".vtp"))

def props(fn):
    geo = read_geo(fn).GetOutput()

    # add point coordinates (rest of cosy not necessary)
    fiber = np.zeros((geo.GetNumberOfCells(), 3))
    cosy = np.zeros((geo.GetNumberOfPoints(), 13))
    cosy[:, 3:6] = v2n(geo.GetPoints().GetData())
    
    for a, n, out in zip([cosy, fiber], ["varWallProps", "FIB_DIR"], [geo.GetPointData(), geo.GetCellData()]):
        arr = n2v(a)
        arr.SetName(n)
        out.AddArray(arr)

    write_geo(fn, geo)

def generate_mesh(f_params):
    mesh = Mesh(f_params)
    mesh.generate_points()
    mesh.generate_cells()
    mesh.extract_svFSI()
    mesh.save_params("cylinder.json")
    return mesh.p


def divisible(f, i):
    return f // i == f / i


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate FSI mesh")
    parser.add_argument("geo", help="geometry parameters (.json)")
    args = parser.parse_args()
    generate_mesh(args.geo)
