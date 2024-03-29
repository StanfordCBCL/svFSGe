# coding=utf-8

import pdb
import vtk
import os
import time
import shutil
import datetime
import scipy
import scipy.stats
import subprocess
import platform
import distro
import numpy as np
from copy import deepcopy
from collections import defaultdict
from os.path import join

from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

from cylinder import Mesh, Simulation, generate_mesh
from vtk_functions import read_geo, write_geo

# names of fields in SimVascular
sv_names = {
    "disp": "Displacement",
    "press": "Pressure",
    "velo": "Velocity",
    "wss": "WSS",
    "pwss": "pWSS",
    "jac": "Jacobian",
    "cauchy": "Cauchy_stress",
    "stress": "Stress",
    "strain": "Strain",
    "gr": "GR",
}


class svFSI(Simulation):
    """
    svFSI base class (handles simulation runs)
    """

    def __init__(self, f_params=None, load=False):
        # simulation parameters
        Simulation.__init__(self, f_params)

        # time stamp
        ct = str(datetime.datetime.now()).replace(" ", "_").replace(":", "-")

        # select paths for this platform
        plat = platform.system().lower()
        if plat == "linux":
            plat += "_" + distro.name().split()[0].lower()
        self.p["paths"] = self.p["paths_" + plat]

        # output folder name
        if load:
            self.p["f_out"] = os.path.dirname(f_params)
        else:
            self.p["f_out"] = join(self.p["paths"]["root"], self.p["name"] + "_" + ct)
        self.p["f_sim"] = join(self.p["f_out"], "partitioned")
        self.p["f_conv"] = join(self.p["f_sim"], "converged")
        self.p["f_arx"] = join(self.p["f_out"], "archive")

        # generate and move files and folders
        if load:
            fm = os.path.join(
                os.path.dirname(f_params), "mesh_tube_fsi", "cylinder.json"
            )
            self.mesh_p = Mesh(fm).p
        else:
            self.setup_files()

        # intialize meshes
        self.fields = ["fluid", "solid", "mesh"]
        self.mesh = {}

        for d in ["fluid", "solid"]:
            fp = join(self.p["f_out"], "mesh_tube_fsi", d)
            self.mesh[("int", d)] = read_geo(
                fp + "/mesh-surfaces/interface.vtp"
            ).GetOutput()
            self.mesh[("vol", d)] = read_geo(fp + "/mesh-complete.mesh.vtu").GetOutput()

        fp = join(self.p["f_out"], "mesh_tube_fsi/")
        self.mesh[("vol", "tube")] = read_geo(fp + self.mesh_p["fname"]).GetOutput()

        surfaces = ["start", "end", "interface"]
        for s in surfaces:
            fp = join(
                self.p["f_out"], "mesh_tube_fsi", "fluid", "mesh-surfaces", s + ".vtp"
            )
            self.mesh[("int", s)] = read_geo(fp).GetOutput()

        if self.p["tortuosity"]:
            fp = join(self.p["f_out"], "mesh_tube_fsi", d)
            self.mesh[("int", "perturbation")] = read_geo(
                fp + "/mesh-surfaces/tortuosity.vtp"
            ).GetOutput()

        # read points
        self.points = {}
        for d in self.mesh.keys():
            self.points[d] = v2n(self.mesh[d].GetPoints().GetData())

        # stored map nodes [src][trg]
        self.maps = {}

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
        self.p_vec = np.linspace(1.0, self.p["fmax"], self.p["nmax"] + 1)

        # relaxation parameter
        self.p["coup"]["omega"] = defaultdict(list)

        # calculate reynolds number
        c1 = 2.0 * self.p["fluid"]["rho"] * self.p["fluid"]["q0"]
        c2 = self.mesh_p["r_inner"] * np.pi * self.p["fluid"]["mu"]
        self.p["re"] = c1 / c2

    def setup_files(self):
        # make folders
        for f in self.p:
            if f[:2] == "f_":
                os.makedirs(self.p[f])

        # copy configureation files
        for f in ["in_petsc", "in_svfsi"]:
            shutil.copytree(self.p["paths"][f], join(self.p["f_out"], f))

        # generate and initialize mesh
        self.mesh_p = generate_mesh(join(self.p["paths"]["in_geo"], self.p["mesh"]))
        shutil.move("mesh_tube_fsi", join(self.p["f_out"], "mesh_tube_fsi"))

    def set_defaults(self):
        pass

    def validate_params(self):
        assert self.p["coup"]["method"] in ["static", "aitken", "iqn_ils"], (
            "Unknown coupling method " + self.p["coup"]["method"]
        )
        if self.p["coup"]["method"] == "iqn_ils":
            assert (
                "iqn_ils_q" in self.p["coup"]
            ), "set parameter iqn_ils_q (maximum number of time steps used)"
            assert (
                "iqn_ils_eps" in self.p["coup"]
            ), "set parameter iqn_ils_eps (tolerane for linearly dependency)"
        if self.p["coup"]["method"] in ["static", "aitken"]:
            assert "omega0" in self.p["coup"], "set parameter omega0"
            assert 0 < self.p["coup"]["omega0"] < 1, "set 0 < omega0 < 1"

    def map(self, m):
        # if not exists, generate new map from src to trg
        if m not in self.maps:
            self.maps[m] = map_ids(self.points[m[0]], self.points[m[1]])
        return self.maps[m]

    def set_fluid(self, i, t):
        # fluid flow (scale by number of tube segments)
        q0 = deepcopy(self.p["fluid"]["q0"] / self.mesh_p["n_seg"])

        # ramp up flow over the first iterations
        if t == 0:
            q = q0 * np.min([i * self.p["fluid"]["q0_rate"] / q0, 1.0])
        else:
            if "q1" in self.p["fluid"]:
                f_time = t / self.p["nmax"]
                q1 = deepcopy(self.p["fluid"]["q1"] / self.mesh_p["n_seg"])
                q = q0 * (1.0 - f_time) + q1 * f_time
            else:
                q = q0

        # fluid pressure (scale by current pressure load step)
        p = self.p["fluid"]["p0"] * self.p_vec[t]

        # set bc pressure and flow
        for bc, val in zip(["pressure", "flow"], [p, q]):
            fn = join(self.p["f_out"], self.p["interfaces"]["bc_" + bc])
            with open(fn, "w") as f:
                f.write("2 1\n")
                f.write("0.0 " + str(val) + "\n")
                f.write("9999999.0 " + str(val) + "\n")

        # write inflow profile
        i_inlet, u_profile = self.write_profile(t)
        ids_all = v2n(
            self.mesh[("vol", "fluid")].GetPointData().GetArray("GlobalNodeID")
        )
        ids = ids_all[i_inlet]

        # define angle (in degrees)
        alpha0 = 0
        alphan = 0
        f_time = t / self.p["nmax"]
        alpha = (alpha0 * (1 - f_time) + alphan * f_time) * np.pi / 180.0

        # set bc flow vector
        direct = [0, np.sin(alpha), np.cos(alpha)]
        fn = join(self.p["f_out"], self.p["interfaces"]["inflow_vector"])
        with open(fn, "w") as f:
            # don't add time zero twice
            f.write("3 2 " + str(len(ids)) + "\n")

            # time steps of mesh displacement (subtract 1 since no mesh sim in first first iteration)
            f.write("0.0\n")
            f.write("9999999.0\n")

            # write displacements of previous and current iteration
            for n, u in zip(ids, u_profile):
                f.write(str(n) + "\n")
                for d in [direct, direct]:
                    for di in d:
                        f.write(str(di * q * u) + " ")
                    f.write("\n")

        # get displacements
        # todo: move all to dedicated folder mesh_fluid_deformed
        disp = self.curr.get(("fluid", "disp", "vol"))

        # add solution to fluid mesh
        fluid = self.mesh[("vol", "fluid")]
        add_array(fluid, disp, sv_names["disp"])

        # warp mesh by displacements
        fluid.GetPointData().SetActiveVectors(sv_names["disp"])
        warp = vtk.vtkWarpVector()
        warp.SetInputData(fluid)
        warp.Update()

        # write geometry to file
        f_out = join(self.p["f_out"], self.p["interfaces"]["geo_fluid"])
        write_geo(f_out, warp.GetOutput())

        surfaces = ["start", "end", "interface"]
        for s in surfaces:
            surf = self.mesh[("int", s)]
            map_s = self.map((("int", s), ("vol", "fluid")))
            add_array(surf, disp[map_s], sv_names["disp"])

            # warp mesh by displacements
            surf.GetPointData().SetActiveVectors(sv_names["disp"])
            warp = vtk.vtkWarpVector()
            warp.SetInputData(surf)
            warp.Update()

            # write geometry to file
            f_out = join(self.p["f_out"], s + ".vtp")
            write_geo(f_out, warp.GetOutput())

    def set_mesh(self, i):
        # write general bc file
        pre = self.prev.get(("fluid", "disp", "int"))
        sol = self.curr.get(("fluid", "disp", "int"))
        msh = self.mesh[("int", "fluid")]
        points = v2n(msh.GetPointData().GetArray("GlobalNodeID"))

        fn = join(self.p["f_out"], self.p["interfaces"]["disp"])
        with open(fn, "w") as f:
            # don't add time zero twice
            if i > 2:
                f.write("3 4 " + str(len(sol)) + "\n")
            else:
                f.write("3 3 " + str(len(sol)) + "\n")

            # time steps of mesh displacement (subtract 1 since no mesh sim in first first iteration)
            if i > 2:
                f.write("0.0\n")
            f.write(str(float(i - 2)) + "\n")
            f.write(str(float(i - 1)) + "\n")
            f.write(str(float(i)) + "\n")

            # write displacements of previous and current iteration
            for n, disp_new, disp_old in zip(points, sol, pre):
                f.write(str(n) + "\n")
                if i > 2:
                    dlist = [np.zeros(3), disp_old, disp_new, disp_new]
                else:
                    dlist = [disp_old, disp_new, disp_new]
                for d in dlist:
                    for di in d:
                        f.write(str(di) + " ")
                    f.write("\n")

        # add solution to fluid mesh
        mesh = self.mesh[("vol", "fluid")]
        disp = self.curr.get(("fluid", "disp", "vol"))
        if i == 1:
            disp = np.zeros(disp.shape)
        add_array(mesh, disp, sv_names["disp"])

        # write geometry to file
        write_geo(join(self.p["f_out"], self.p["interfaces"]["geo_mesh"]), mesh)

    def set_solid(self, n, t):
        # name of wall properties array
        name = "gr_properties"

        # read solid volume mesh
        solid = self.mesh[("vol", "solid")]

        # set wss
        props = v2n(solid.GetPointData().GetArray(name))
        props[:, 6] = self.curr.get(("solid", "wss", "vol"))

        # set time
        props[:, 7] = t + 1

        # beginning of new load step?
        props[:, 12] = n == 0

        add_array(solid, props, name)

        # write geometry to file
        fn = join(self.p["f_out"], self.p["interfaces"]["geo_solid"])
        write_geo(fn, solid)

        # write interface pressure to file
        geo = self.mesh[("int", "solid")]
        num = self.curr.get(("solid", "press", "int"))
        name = "Pressure"
        add_array(geo, num, name)
        fn = join(self.p["f_out"], self.p["interfaces"]["load_pressure"])
        write_geo(fn, geo)

        # write interface pressure perturbation to file
        if self.p["tortuosity"]:
            geo = self.mesh[("int", "perturbation")]
            if t == 0:
                perturb = 0.0
            else:
                perturb = 0.01 * self.p["fluid"]["p0"]
            num = perturb * np.ones(geo.GetNumberOfPoints())
            name = "Pressure"
            add_array(geo, num, name)
            fn = join(self.p["f_out"], self.p["interfaces"]["load_perturbation"])
            write_geo(fn, geo)

    def step(self, name, i, t, n, times):
        if name not in self.fields:
            raise ValueError("Unknown step option " + name)

        # set up input files
        if name == "fluid":
            self.set_fluid(i, t)
        elif name == "solid":
            self.set_solid(n, t)
        elif name == "mesh":
            self.set_mesh(i)

        # execute svFSI
        exe = ["mpiexec", "-np", str(self.p["n_procs"][name])]
        # exe = ["mpiexec", "--use-hwthread-cpus"]
        exe += [join(self.p["paths"]["exe"], self.p["exe"][name])]
        exe += [join("in_svfsi", self.p["inp"][name])]

        t_start = time.time()
        if self.p["debug"]:
            print(" ".join(exe))
            child = subprocess.run(exe, cwd=self.p["f_out"])
        else:
            i_str = str(i).zfill(3)
            fn = join(self.p["f_sim"], name + "_" + i_str + ".log")
            with open(fn, "w") as f:
                child = subprocess.run(exe, stdout=f, stderr=f, cwd=self.p["f_out"])
        times[name] = time.time() - t_start

        # check if simulation crashed and return error
        if child.returncode != 0:
            for f in self.curr.sol.keys():
                self.curr.sol[f] = None
            return True

        # read and store results
        return self.post(name, i)

    def post(self, domain, i):
        out = self.p["out"][domain]
        fname = join(out, out + "_")
        phys = domain
        i_str = str(i).zfill(3)
        if domain == "solid":
            # read current iteration
            fields = ["disp", "jac", "cauchy", "stress", "strain", "gr"]
            src = [fname + str(self.p["n_max"][domain] * i).zfill(3) + ".vtu"]
        elif domain == "fluid":
            # read converged steady state flow
            fields = ["velo", "wss", "press"]

            # read n_fluid last time steps
            n_fluid = 1
            src = [
                fname + str(self.p["n_max"][domain] * i - j).zfill(3) + ".vtu"
                for j in range(n_fluid)
            ]
        elif domain == "mesh":
            # read fully displaced mesh
            fields = ["disp"]
            phys = "fluid"
            src = [fname + str(self.p["n_max"][domain] * (i - 1)).zfill(3) + ".vtu"]
        else:
            raise ValueError("Unknown domain " + domain)
        src = [join(self.p["f_out"], s) for s in src]

        # check if simulation crashed
        if np.any([not os.path.exists(s) for s in src]):
            for f in fields:
                self.curr.sol[f] = None
                return True
        else:
            # archive results
            trg = join(self.p["f_sim"], domain + "_out_" + i_str + ".vtu")
            shutil.copyfile(src[0], trg)

            # read results
            res = []
            for s in src:
                res += [read_geo(s).GetOutput()]

            # extract fields
            for f in fields:
                if f == "wss":
                    sol = []
                    for r in res:
                        n_smooth = 1
                        c2p = r
                        for _ in range(n_smooth):
                            # map point data to cell data
                            p2c = vtk.vtkPointDataToCellData()
                            p2c.SetInputData(c2p)
                            p2c.Update()

                            # map cell data to point data
                            c2p = vtk.vtkCellDataToPointData()
                            c2p.SetInputData(p2c.GetOutput())
                            c2p.Update()
                            c2p = c2p.GetOutput()

                        # get element-wise wss maped to point data
                        sol += [v2n(c2p.GetPointData().GetArray("WSS"))]
                    sol = np.mean(np.array(sol), axis=0)

                    # points on fluid interface
                    map_int = self.map((("int", "fluid"), ("vol", "fluid")))

                    # only store magnitude of wss at interface (doesn't make sense elsewhere)
                    self.curr.add((phys, f, "int"), sol[map_int])

                    # # only for logging, store svFSI point-wise wss
                    # sol = v2n(res.GetPointData().GetArray(sv_names[f]))
                    # self.curr.add((phys, 'pwss', 'int'), np.linalg.norm(sol[map_int], axis=1))
                else:
                    extr = []
                    for r in res:
                        if not r.GetPointData().HasArray(sv_names[f]):
                            raise ValueError("no array in PointData: " + sv_names[f])
                        extr += [v2n(r.GetPointData().GetArray(sv_names[f]))]
                    sol = np.mean(np.array(extr), axis=0)
                    self.curr.add((phys, f, "vol"), sol)

        # archive input
        if domain in ["fluid", "solid"]:
            src = join(self.p["f_out"], self.p["interfaces"]["geo_" + domain])
            trg = join(self.p["f_sim"], domain + "_inp_" + i_str + ".vtu")
            shutil.copyfile(src, trg)
        return False

    def get_profile(self, x_norm, rad_norm, t):
        # quadratic flow profile (integrates to one, zero on the FS-interface)
        u_profile = 2.0 * (1.0 - rad_norm**2.0)

        # time factor
        f_time = t / self.p["nmax"]

        # custom flow profile
        if "profile_beta" in self.p:
            # limits
            beta_min = self.p["profile"]["beta_min"]
            beta_max = self.p["profile"]["beta_max"]

            # beta distribution for x-bias
            beta = beta_min + (beta_max - beta_min) * f_time
            bias = scipy.stats.beta.pdf(x_norm, 2, beta)
            bias0 = scipy.stats.beta.pdf(x_norm, 2, beta_min)

            # normalize with initial profile
            pos = bias0 != 0.0
            bias[pos] /= bias0[pos]

            u_profile *= bias
        elif "profile_plub" in self.p:
            plug = self.p["profile_plug"] * f_time

        return u_profile

    def write_profile(self, t):
        # GlobalNodeID of inlet within fluid mesh
        i_inlet = self.map((("int", "start"), ("vol", "fluid")))

        # inlet points in current configuration
        points = deepcopy(self.points[("vol", "fluid")])[i_inlet]

        # radial coordinate [0, 1]
        rad = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        rad_norm = rad / np.max(rad)
        area = np.max(rad) ** 2 * np.pi

        # normalized x coordinate [0, 1]
        x = points[:, 0]
        x_norm = (1.0 + x / np.max(x)) / 2.0

        # get flow profile at inlet
        u_profile = 1.0 / area * self.get_profile(x_norm, rad_norm, t)

        return i_inlet, u_profile

    def poiseuille(self, t):
        # fluid flow and pressure
        q = self.p["fluid"]["q0"]
        p = self.p["fluid"]["p0"] * self.p_vec[t]

        # fluid mesh points in reference configuration
        points_r = deepcopy(self.points[("vol", "fluid")])

        # fluid mesh points in current configuration
        points_f = deepcopy(points_r) + deepcopy(
            self.curr.get(("fluid", "disp", "vol"))
        )
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
        rmin = np.min(rad[self.map((("int", "fluid"), ("vol", "fluid")))])

        # estimate Poiseuille resistance
        res = 8.0 * self.p["fluid"]["mu"] * amax / np.pi / rmin**4

        # estimate linear pressure gradient
        press = p * np.ones(len(rad)) + res * q * (1.0 - ax)
        self.curr.add(("fluid", "press", "vol"), press)

        # get local cross-sectional area and maximum radius (assuming a structured mesh)
        z_slices = np.unique(points_r[:, 2])
        areas = np.zeros(n_points)
        rad_norm = np.zeros(n_points)
        for z in z_slices:
            i_slice = points_r[:, 2] == z
            rmax = np.max(rad[i_slice])
            areas[i_slice] = rmax**2.0 * np.pi
            rad_norm[i_slice] = rad[i_slice] / rmax
        assert not np.any(areas == 0.0), "area zero"

        # estimate flow profile
        velo = np.zeros(points_f.shape)
        velo[:, 2] = q / areas * self.get_profile(x_norm, rad_norm, t)
        self.curr.add(("fluid", "velo", "vol"), velo)

        # points on fluid interface
        map_int = self.map((("int", "fluid"), ("vol", "fluid")))

        # make sure wss is nonzero even for q=0 (only ratio is important for g&r)
        if q == 0.0:
            q = 1.0

        # calculate wss from const Poiseuille flow
        # todo: use actual profile (and local gradient??)
        wss = np.zeros((len(map_int), 3))
        wss[:, -1] = 4.0 * self.p["fluid"]["mu"] * q / np.pi / rad[map_int] ** 3.0
        self.curr.add(("fluid", "wss", "int"), wss)

    def ctrl_vol(self, t):
        # fluid flow and pressure
        q = self.p["fluid"]["q0"]
        p = self.p["fluid"]["p0"] * self.p_vec[t]

        # fluid mesh points in reference configuration
        points_r = deepcopy(self.points[("vol", "fluid")])

        # fluid mesh points in current configuration
        disp = deepcopy(self.curr.get(("fluid", "disp", "vol")))
        points_f = points_r + disp

        # radial coordinate of all points
        rad = np.sqrt(points_f[:, 0] ** 2 + points_f[:, 1] ** 2)

        pdb.set_trace()


class Solution:
    """
    Object to handle solutions
    """

    def __init__(self, sim):
        self.sim = sim
        self.sol = {}

        # physics of fields
        self.field2phys = {
            "disp": "solid",
            "press": "fluid",
            "velo": "fluid",
            "wss": "fluid",
        }

        dim_vec = self.sim.points[("vol", "tube")].shape
        dim_sca = dim_vec[0]
        dim_ten = (dim_sca, 6)

        # "zero" vectors. use nan where quantity is not defined
        self.zero = {
            "disp": np.zeros(dim_vec),
            "velo": np.zeros(dim_vec),
            "wss": np.ones(dim_sca) * np.nan,
            "pwss": np.ones(dim_sca) * np.nan,
            "press": np.zeros(dim_sca) * np.nan,
            "jac": np.zeros(dim_sca) * np.nan,
            "cauchy": np.zeros(dim_ten) * np.nan,
            "stress": np.zeros(dim_ten) * np.nan,
            "strain": np.zeros(dim_ten) * np.nan,
            "gr": np.ones((dim_sca, 50)) * np.nan,
        }
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
            if f == "disp":
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

        map_v = self.sim.map(((p, d), ("vol", "tube")))
        if f in ["disp", "velo", "press", "jac", "cauchy", "stress", "strain", "gr"]:
            self.sol[f][map_v] = deepcopy(sol)
        elif "wss" in f:
            # wss in tube volume
            self.sol[f][map_v] = deepcopy(np.linalg.norm(sol, axis=1))

            # wss at fluid interface
            sol_int = self.sol[f][self.sim.map((("int", "fluid"), ("vol", "tube")))]

            # wss in solid volume (assume wss is constant radially)
            map_src = self.sim.map((("vol", "solid"), ("int", "fluid")))
            map_trg = self.sim.map((("vol", "solid"), ("vol", "tube")))
            self.sol[f][map_trg] = deepcopy(sol_int[map_src])
        else:
            raise ValueError(f + " not in fields " + str(list(self.fields)))

    def get(self, kind):
        # fluid, solid, tube
        # disp, velo, wss, press
        # vol, int
        d, f, p = kind
        if self.sol[f] is None:
            raise ValueError("no solution " + ",".join(kind))

        map_s = self.sim.map(((p, d), ("vol", "tube")))
        return deepcopy(self.sol[f][map_s])

    def archive(self, domain, fname):
        geo = self.sim.mesh[("vol", domain)]
        for f in self.fields:
            add_array(
                geo,
                self.sol[f][self.sim.map((("vol", domain), ("vol", "tube")))],
                sv_names[f],
            )
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
