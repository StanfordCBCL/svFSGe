# coding=utf-8

import pdb
import re
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

        # number of G&R load steps. Renamed nmax -> nloads (the JSON already has
        # a coup["nmax"] for coupling iterations, so the top-level name was
        # confusing). Accept the old "nmax" for backward compatibility.
        if "nloads" not in self.p and "nmax" in self.p:
            self.p["nloads"] = self.p["nmax"]

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

        # Override n_max for fluid if pulsatile mode is enabled
        if self.p.get("pulsatile", False):
            pulsatile_config = self.p.get("pulsatile_config", {})
            n_cycles = pulsatile_config.get("n_cycles", 2)
            steps_per_cycle = pulsatile_config.get("steps_per_cycle")
            self.p["n_max"]["fluid"] = n_cycles * steps_per_cycle
            print(f"Pulsatile mode enabled: fluid solver will run {self.p['n_max']['fluid']} steps ({n_cycles} cycles × {steps_per_cycle} steps/cycle)")
        else:
            print(f"Steady flow mode: fluid solver will run {self.p['n_max']['fluid']} steps")

        # store input file path for archiving
        self._f_params = f_params

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
        self.debug_qr = {
            "V_before": [], "W_before": [], "ncols_before": [],
            "Q": [], "R": [], "V_after": [], "W_after": [], "ncols_after": [],
            "cc": [], "t": [], "n": [],
        }

        # current/previous solution vector at interface and in volume
        self.curr = Solution(self)
        self.prev = Solution(self)

        # generate load vector
        self.p_vec = np.linspace(1.0, self.p["fmax"], self.p["nloads"] + 1)

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

        # Copy the appropriate flow data file based on mode
        if self.p.get("pulsatile", False):
            flow_file = self.p.get("pulsatile_config", {}).get("flow_data_file", "pulsatile_flow.dat")
        else:
            flow_file = "steady_flow.dat"
        src = join(self.p["paths"]["in_svfsi"], flow_file)
        if os.path.exists(src):
            shutil.copy(src, self.p["f_out"])

        # generate and initialize mesh
        self.mesh_p = generate_mesh(join(self.p["paths"]["in_geo"], self.p["mesh"]))
        shutil.move("mesh_tube_fsi", join(self.p["f_out"], "mesh_tube_fsi"))

        # copy input JSON to archive
        if self._f_params and os.path.isfile(self._f_params):
            shutil.copy(self._f_params, self.p["f_arx"])

        # copy in_geo folder alongside archive
        in_geo_src = self.p["paths"]["in_geo"]
        if os.path.isdir(in_geo_src):
            shutil.copytree(in_geo_src, join(self.p["f_out"], "in_geo"))

        # inject the G&R load profile and insult profile from the JSON into the
        # solid XML
        self.set_gr_load()
        self.set_gr_insult()
        self.set_gr_growth()

    def _write_curve(self, rel_path, curve):
        """Write a 2-column tabulated curve to <f_out>/<rel_path>. Each row is
        "x y"; the curve may be a list of [x, y] pairs or a plain list of y
        values (x is then taken as 0, 1, 2, ...)."""
        with open(join(self.p["f_out"], rel_path), "w") as f:
            for i, row in enumerate(curve):
                x, y = (i, row) if np.isscalar(row) else (row[0], row[1])
                f.write(str(x) + " " + str(y) + "\n")

    def _patch_solid_xml(self, tags):
        """Set/insert <tag> elements in the GR_equilibrated material block of the
        solid solver XML (each run owns a private copy). Existing tags are
        overridden in place; new tags are inserted right after <coup_wss>,
        reusing its indentation. Same regex-on-XML approach as the pulsatile
        step-count patch."""
        xml_file = join(self.p["f_out"], "in_svfsi", self.p["inp"]["solid"])
        with open(xml_file) as f:
            xml = f.read()
        for tag, val in tags.items():
            new = "<" + tag + "> " + str(val) + " </" + tag + ">"
            pat = re.compile(r"<%s>.*?</%s>" % (tag, tag))
            if pat.search(xml):
                xml = pat.sub(new, xml)
            else:
                xml = re.sub(
                    r"(\n([ \t]*)<coup_wss>.*?</coup_wss>)",
                    r"\1\n\g<2>" + new,
                    xml,
                    count=1,
                )
        with open(xml_file, "w") as f:
            f.write(xml)

    def set_gr_growth(self):
        """Inject optional G&R growth-stabilization params into the solid XML.

        Currently exposes <tau_ratio_floor> (config key "tau_ratio_floor"): a
        lower clamp on the WSS-stimulus ratio tau/tauo in gr_equilibrated.cpp.
        A developed aneurysm bulge has collapsed luminal WSS (recirculation); an
        accurate WSS surrogate then drives tau/tauo -> 0, saturating the growth
        stimulus into runaway growth the partitioned coupling cannot integrate.
        A positive floor caps that. Omitting the key leaves the XML untouched
        (svFSI default 0 = disabled = historical behavior).
        """
        floor = self.p.get("tau_ratio_floor")
        if floor is None:
            return
        self._patch_solid_xml({"tau_ratio_floor": floor})

    def set_gr_load(self):
        """Inject the G&R load profile from the JSON into the solid solver XML.

        The load profile controls how the G&R insult is ramped over pseudo-time
        inside gr_equilibrated.cpp. Previously this was a hard-coded tanh ramp;
        now it is exposed through the GR_equilibrated material parameters
        <load_profile>, <load_steep> and <load_file>, so it can be changed from
        the JSON without editing/rebuilding svFSI. Configure via an optional
        "gr_load" section:

            "gr_load": {
                "profile": "tanh",         # linear | tanh | power | file
                "steep": 2.0,              # tanh steepness / power exponent
                "curve": [[0, 0.0],        # only for profile == "file":
                          [1, 0.33],       #   one [step, factor] per load step,
                          [2, 0.67],       #   x = step number (0 = pre-stress,
                          [3, 1.00]]       #   1..nloads = G&R loads), so the
            }                              #   curve has nloads + 1 entries.

        For profile "file" the curve gives the load factor directly per step
        (no normalization/interpolation between steps), so the number of entries
        equals the number of load steps. Omitting "gr_load" leaves the XML
        untouched, so svFSI falls back to its defaults (tanh, steep=2.0) which
        reproduce the historical ramp.
        """
        cfg = self.p.get("gr_load")
        if cfg is None:
            return

        profile = cfg.get("profile", "tanh")
        steep = cfg.get("steep", 2.0)
        load_file = cfg.get("file", "")

        # write a tabulated load curve when requested. Each row is "x y" with
        # x the integer load-step number (0 = pre-stress, 1..nloads = G&R loads)
        # and y the load factor at that step. A plain list of factors is also
        # accepted (step numbers are then taken as 0, 1, 2, ...).
        if profile == "file" and "curve" in cfg:
            load_file = join("in_svfsi", "gr_load_curve.dat")
            self._write_curve(load_file, cfg["curve"])

        # tags to write into the GR_equilibrated material block
        tags = {"load_profile": profile, "load_steep": steep}
        if load_file:
            tags["load_file"] = load_file
        self._patch_solid_xml(tags)

    def set_gr_insult(self):
        """Inject the spatial G&R insult profile from the JSON into the solid XML.

        The insult profile localizes the aneurysm: the elastin/stimulus
        knock-down at each point is scaled by an axial x azimuthal factor.
        Previously this super-Gaussian shape was hard-coded in
        gr_equilibrated.cpp; it is now exposed through the GR_equilibrated
        material parameters, so it can be set from the JSON without rebuilding
        svFSI. Configure via an optional "gr_insult" section:

            "gr_insult": {
                "profile": "gaussian",   # gaussian (default) | file
                "mag": 0.7,              # peak elastin loss fraction
                "z_loc": 0.5,            # axial center / tube length
                "z_wid": 0.25,           # axial width / tube length
                "z_exp": 2,              # axial super-Gaussian exponent
                "asym": true,            # apply circumferential localization
                "theta_wid": 0.55,       # azimuthal width / pi
                "theta_exp": 6,          # azimuthal super-Gaussian exponent
                "curve": [[0.0, 0.0],    # only for profile == "file": the axial
                          [0.5, 1.0],    #   factor f_axi vs normalized axial
                          [1.0, 0.0]]    #   position z/lo in [0, 1] (the azimuth
            }                            #   factor stays gaussian)

        For profile "file" the curve defines the axial insult shape (any
        function); the azimuthal localization still follows the gaussian
        asym/theta parameters. Omitting "gr_insult" leaves the XML untouched, so
        svFSI falls back to its defaults, reproducing the historical insult.
        """
        cfg = self.p.get("gr_insult")
        if cfg is None:
            return

        profile = cfg.get("profile", "gaussian")
        insult_file = cfg.get("file", "")

        # write the axial insult curve (z/lo, f_axi) for a custom shape
        if profile == "file" and "curve" in cfg:
            insult_file = join("in_svfsi", "gr_insult_curve.dat")
            self._write_curve(insult_file, cfg["curve"])

        # map JSON keys -> GR_equilibrated XML tags
        keys = {
            "mag": "insult_mag", "z_loc": "insult_z_loc", "z_wid": "insult_z_wid",
            "z_exp": "insult_z_exp", "asym": "insult_asym",
            "theta_wid": "insult_theta_wid", "theta_exp": "insult_theta_exp",
        }
        tags = {"insult_profile": profile}
        for jkey, tag in keys.items():
            if jkey in cfg:
                val = cfg[jkey]
                # svFSI parses booleans as true/false
                tags[tag] = str(val).lower() if isinstance(val, bool) else val
        if insult_file:
            tags["insult_file"] = insult_file
        self._patch_solid_xml(tags)

    def set_defaults(self):
        pass

    def validate_params(self):
        assert self.p["coup"]["method"] in ["static", "aitken", "iqn_ils", "weak", "linesearch", "uber_robin"], (
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
                f_time = t / self.p["nloads"]
                q1 = deepcopy(self.p["fluid"]["q1"] / self.mesh_p["n_seg"])
                q = q0 * (1.0 - f_time) + q1 * f_time
            else:
                q = q0

        # fluid pressure (scale by current pressure load step)
        p = self.p["fluid"]["p0"] * self.p_vec[t]

        # set bc pressure and flow
        # in pulsatile mode, skip bc_flow — the waveform file must not be overwritten
        bcs = ["pressure"] if self.p.get("pulsatile", False) else ["pressure", "flow"]
        for bc, val in zip(bcs, [p, q]):
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
        f_time = t / self.p["nloads"]
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

        # for pulsatile mode, update XML total step count before each run
        # steps accumulate: iteration i ends at n_max["fluid"] * i
        if self.p.get("pulsatile", False):
            total_steps = self.p["n_max"]["fluid"] * i
            xml_file = join(self.p["f_out"], "in_svfsi", self.p["inp"]["fluid"])
            with open(xml_file, 'r') as f:
                xml_content = f.read()
            xml_content = re.sub(
                r'<Number_of_time_steps>\s*\d+\s*</Number_of_time_steps>',
                f'<Number_of_time_steps> {total_steps} </Number_of_time_steps>',
                xml_content
            )
            with open(xml_file, 'w') as f:
                f.write(xml_content)

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

    def save_tube(self, idx, folder=None):
        """Write the current coupling solution as tube_<idx>.vtu (all fields).

        Single reusable VTU-save used by the load-controlled loop, the
        arc-length driver, and crash/failure handling, so what gets written (and
        how it is fixed) lives in one place. ``folder`` defaults to the
        per-iteration sim directory; pass ``self.p["f_conv"]`` for a converged
        step. ``idx`` may be an int (zero-padded to 3 digits) or a string tag.
        Returns the written path.
        """
        if folder is None:
            folder = self.p["f_sim"]
        tag = str(idx).zfill(3) if isinstance(idx, int) else str(idx)
        path = join(folder, "tube_" + tag + ".vtu")
        self.curr.archive("tube", path)
        return path

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

        # Explicit restart paths for the solid G&R solver. A continuation driver
        # (arc-length / displacement / stimulus) can set self.restart_in /
        # self.restart_out to re-solve the same load step from an exact
        # checkpoint at different load factors without the solver compounding
        # state: --restart-in is read exactly (never "_last"), --restart-out is
        # written exactly. Paths are relative to f_out (the solver's cwd).
        if name == "solid":
            if getattr(self, "restart_in", None):
                exe += ["--restart-in", self.restart_in]
            if getattr(self, "restart_out", None):
                exe += ["--restart-out", self.restart_out]

        # Optional per-domain wall-clock timeout. On an element-inversion abort the
        # solver's MPI_Abort can hang indefinitely; a timeout kills it and reports
        # failure so a continuation driver (arc-length / line search) can back off
        # instead of wedging. Configure via "solve_timeout": {"solid": <seconds>}.
        timeout = self.p.get("solve_timeout", {}).get(name)

        t_start = time.time()
        try:
            if self.p["debug"]:
                print(" ".join(exe))
                child = subprocess.run(exe, cwd=self.p["f_out"], timeout=timeout)
            else:
                i_str = str(i).zfill(3)
                fn = join(self.p["f_sim"], name + "_" + i_str + ".log")
                with open(fn, "w") as f:
                    child = subprocess.run(exe, stdout=f, stderr=f,
                                           cwd=self.p["f_out"], timeout=timeout)
        except subprocess.TimeoutExpired:
            # hung solve (e.g. MPI_Abort after a negative Jacobian) -> treat as failure
            print(f"    [{name}] solve exceeded {timeout}s -> killed, treated as failure")
            for f in self.curr.sol.keys():
                self.curr.sol[f] = None
            return True
        times[name] = time.time() - t_start

        # check if simulation crashed and return error
        if child.returncode != 0:
            for f in self.curr.sol.keys():
                self.curr.sol[f] = None
            return True

        # read and store results
        return self.post(name, i)

    def extract_pulsatile_time_average(self, fname, i, fields, verbose=False):
        """
        Extract time-averaged quantities from pulsatile flow simulation.
        Averages over the last cardiac cycle (last n_reduction_steps).

        Args:
            fname: Base filename for VTU files (e.g., "steady/steady_")
            i: Current iteration counter
            fields: List of field names to extract

        Returns:
            dict: {field_name: time-averaged_data}
            list: geometries for archiving
        """
        pulsatile_config = self.p.get("pulsatile_config", {})
        n_reduction_steps = pulsatile_config.get("n_reduction_steps")

        # Steps accumulate across coupling iterations, same as steady mode
        # iteration i ends at step n_max["fluid"] * i
        end_step = self.p["n_max"]["fluid"] * i
        start_step = end_step - n_reduction_steps + 1

        if verbose:
            print(f"    Pulsatile mode: Averaging time steps {start_step} to {end_step} (last {n_reduction_steps} steps)")

        # Read VTU files for the last cycle
        # Files are numbered: steady_001.vtu, steady_002.vtu, ..., steady_192.vtu
        # fname already includes the output directory and prefix (e.g., "steady/steady_")
        src_files = []
        for step in range(start_step, end_step + 1):
            filepath = fname + str(step).zfill(3) + ".vtu"
            fullpath = join(self.p["f_out"], filepath)
            if os.path.exists(fullpath):
                src_files.append(fullpath)
            else:
                print(f"    WARNING: File not found: {fullpath}")

        if len(src_files) == 0:
            raise ValueError(f"No VTU files found for time averaging from step {start_step} to {end_step}")

        if verbose:
            print(f"    Found {len(src_files)} VTU files for averaging")

        # Read all geometries
        geometries = [read_geo(f).GetOutput() for f in src_files]

        # Extract and average fields
        averaged_fields = {}
        for field in fields:
            field_data = []
            for geo in geometries:
                if geo.GetPointData().HasArray(sv_names[field]):
                    field_data.append(v2n(geo.GetPointData().GetArray(sv_names[field])))

            if len(field_data) > 0:
                # Time average over the cardiac cycle
                averaged_fields[field] = np.mean(np.array(field_data), axis=0)
                if verbose:
                    print(f"    {field}: averaged over {len(field_data)} time steps")
            else:
                averaged_fields[field] = None
                print(f"    WARNING: {field} not found in geometries")

        return averaged_fields, geometries

    def extract_pulsatile_amplitude(self, fname, i, fields, verbose=False):
        """
        Extract the true magnitude amplitude of a vector quantity over the
        last cardiac cycle: max_k(||v_k(p)||) - min_k(||v_k(p)||), i.e. the
        peak-to-peak range of the magnitude signal itself.

        Since downstream consumers (Solution.add, svfsi.py) expect an (N,3)
        vector and recover the stimulus by taking np.linalg.norm(sol, axis=1),
        the scalar result is packed into the z-component of a zero vector so
        that norm exactly recovers it (same convention as
        extract_pulsatile_magnitude).

        Args:
            fname: Base filename for VTU files (e.g., "steady/steady_")
            i: Current iteration counter
            fields: List of field names to extract (only these fields are processed)

        Returns:
            dict: {field_name: (N,3) array with amplitude in the z-component}
            list: geometries for archiving
        """
        pulsatile_config = self.p.get("pulsatile_config", {})
        n_reduction_steps = pulsatile_config.get("n_reduction_steps")

        end_step = self.p["n_max"]["fluid"] * i
        start_step = end_step - n_reduction_steps + 1

        if verbose:
            print(f"    Pulsatile mode: Computing magnitude amplitude over time steps {start_step} to {end_step} (last {n_reduction_steps} steps)")

        src_files = []
        for step in range(start_step, end_step + 1):
            filepath = fname + str(step).zfill(3) + ".vtu"
            fullpath = join(self.p["f_out"], filepath)
            if os.path.exists(fullpath):
                src_files.append(fullpath)
            else:
                print(f"    WARNING: File not found: {fullpath}")

        if len(src_files) == 0:
            raise ValueError(f"No VTU files found for amplitude computation from step {start_step} to {end_step}")

        if verbose:
            print(f"    Found {len(src_files)} VTU files for amplitude computation")

        geometries = [read_geo(f).GetOutput() for f in src_files]

        amplitude_fields = {}
        for field in fields:
            field_data = []
            for geo in geometries:
                if geo.GetPointData().HasArray(sv_names[field]):
                    field_data.append(v2n(geo.GetPointData().GetArray(sv_names[field])))

            if len(field_data) > 0:
                stacked = np.array(field_data)  # (n_steps, n_points, 3)
                mag = np.linalg.norm(stacked, axis=2)  # (n_steps, n_points)
                amp = np.max(mag, axis=0) - np.min(mag, axis=0)  # (n_points,)
                packed = np.zeros((amp.shape[0], 3))
                packed[:, 2] = amp
                amplitude_fields[field] = packed
                if verbose:
                    print(f"    {field}: magnitude amplitude (max-min of ||v||) computed over {len(field_data)} time steps")
            else:
                amplitude_fields[field] = None
                print(f"    WARNING: {field} not found in geometries")

        return amplitude_fields, geometries

    def extract_pulsatile_magnitude(self, fname, i, fields, verbose=False):
        """
        Extract the true time-averaged magnitude of a vector quantity over the
        last cardiac cycle: mean_k(||v_k(p)||), i.e. magnitude-then-average
        (classical TAWSS), as opposed to extract_pulsatile_time_average's
        mean-then-magnitude.

        Since downstream consumers (Solution.add, svfsi.py) expect an (N,3)
        vector and recover the stimulus by taking np.linalg.norm(sol, axis=1),
        the scalar result is packed into the z-component of a zero vector so
        that norm exactly recovers it - the same convention neural_operator.py
        uses for delivering scalar WSS predictions.

        Args:
            fname: Base filename for VTU files (e.g., "steady/steady_")
            i: Current iteration counter
            fields: List of field names to extract

        Returns:
            dict: {field_name: (N,3) array with magnitude in the z-component}
            list: geometries for archiving
        """
        pulsatile_config = self.p.get("pulsatile_config", {})
        n_reduction_steps = pulsatile_config.get("n_reduction_steps")

        end_step = self.p["n_max"]["fluid"] * i
        start_step = end_step - n_reduction_steps + 1

        if verbose:
            print(f"    Pulsatile mode: Averaging magnitude over time steps {start_step} to {end_step} (last {n_reduction_steps} steps)")

        src_files = []
        for step in range(start_step, end_step + 1):
            filepath = fname + str(step).zfill(3) + ".vtu"
            fullpath = join(self.p["f_out"], filepath)
            if os.path.exists(fullpath):
                src_files.append(fullpath)
            else:
                print(f"    WARNING: File not found: {fullpath}")

        if len(src_files) == 0:
            raise ValueError(f"No VTU files found for magnitude averaging from step {start_step} to {end_step}")

        if verbose:
            print(f"    Found {len(src_files)} VTU files for magnitude averaging")

        geometries = [read_geo(f).GetOutput() for f in src_files]

        magnitude_fields = {}
        for field in fields:
            field_data = []
            for geo in geometries:
                if geo.GetPointData().HasArray(sv_names[field]):
                    field_data.append(v2n(geo.GetPointData().GetArray(sv_names[field])))

            if len(field_data) > 0:
                stacked = np.array(field_data)  # (n_steps, n_points, 3)
                mean_mag = np.mean(np.linalg.norm(stacked, axis=2), axis=0)  # (n_points,)
                packed = np.zeros((mean_mag.shape[0], 3))
                packed[:, 2] = mean_mag
                magnitude_fields[field] = packed
                if verbose:
                    print(f"    {field}: magnitude averaged over {len(field_data)} time steps")
            else:
                magnitude_fields[field] = None
                print(f"    WARNING: {field} not found in geometries")

        return magnitude_fields, geometries

    def extract_pulsatile_data(self, fname, i, fields, verbose=False):
        """
        Dispatch per-field extraction to time_average, amplitude, or magnitude
        based on the pulsatile_config.field_reduction mapping in the JSON.

        Default reductions (overridable in JSON under pulsatile_config.field_reduction):
            wss  -> "amplitude"
            velo -> "time_average"
            press -> "time_average"

        Args:
            fname: Base filename for VTU files
            i: Current iteration counter
            fields: List of field names to extract

        Returns:
            dict: {field_name: extracted_data}
            list: geometries (from the last extraction call, for archiving)
        """
        pulsatile_config = self.p.get("pulsatile_config", {})
        defaults = {"wss": "amplitude", "velo": "time_average", "press": "time_average"}
        field_reduction = {**defaults, **pulsatile_config.get("field_reduction", {})}

        avg_fields = [f for f in fields if field_reduction.get(f, "time_average") == "time_average"]
        amp_fields = [f for f in fields if field_reduction.get(f, "time_average") == "amplitude"]
        mag_fields = [f for f in fields if field_reduction.get(f, "time_average") == "magnitude"]

        combined = {}
        geometries = []

        if avg_fields:
            avg_data, geometries = self.extract_pulsatile_time_average(fname, i, avg_fields, verbose=verbose)
            combined.update(avg_data)

        if amp_fields:
            amp_data, geometries = self.extract_pulsatile_amplitude(fname, i, amp_fields, verbose=verbose)
            combined.update(amp_data)

        if mag_fields:
            mag_data, geometries = self.extract_pulsatile_magnitude(fname, i, mag_fields, verbose=verbose)
            combined.update(mag_data)

        return combined, geometries

    def post(self, domain, i):
        out = self.p["out"][domain]
        fname = join(out, out + "_")
        phys = domain
        i_str = str(i).zfill(3)

        if domain == "solid":
            # read current iteration
            fields = ["disp", "jac", "cauchy", "stress", "strain", "gr"]
            if getattr(self, "_arc_active", False):
                # arc-length: the n=0 restart resets decouple solver cTS from i,
                # so read the most-recently-written gr_restart file instead.
                import glob as _g
                cand = _g.glob(join(self.p["f_out"], fname + "*.vtu"))
                if cand:
                    # return path RELATIVE to f_out (post re-joins f_out below)
                    src = [os.path.relpath(max(cand, key=os.path.getmtime), self.p["f_out"])]
                else:
                    src = [fname + str(self.p["n_max"][domain] * i).zfill(3) + ".vtu"]
            else:
                src = [fname + str(self.p["n_max"][domain] * i).zfill(3) + ".vtu"]
        elif domain == "fluid":
            # read converged steady state flow
            fields = ["velo", "wss", "press"]

            # Fluid solver always outputs files from 1 to n_max["fluid"]
            # (these files are overwritten each coupling iteration)
            # We want to read the LAST time step(s) from this run
            total_fluid_steps = self.p["n_max"][domain]

            if self.p.get("pulsatile", False):
                # Pulsatile mode: 192 steps per coupling iteration (not accumulating)
                # files are overwritten each iteration; averaging handled below
                src = []
            else:
                # Steady flow: steps accumulate across coupling iterations
                # iteration i ends at step n_max["fluid"] * i
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

        # check if simulation crashed (skip for pulsatile mode - checked later)
        if len(src) > 0 and np.any([not os.path.exists(s) for s in src]):
            print(f"    ERROR: Missing VTU files for domain '{domain}':")
            for s in src:
                status = "EXISTS" if os.path.exists(s) else "MISSING"
                print(f"      {status}: {s}")
            for f in fields:
                self.curr.sol[f] = None
                return True

        # archive results (if we have src files to archive)
        if len(src) > 0:
            trg = join(self.p["f_sim"], domain + "_out_" + i_str + ".vtu")
            shutil.copyfile(src[0], trg)

        # read results
        if domain == "fluid" and self.p.get("pulsatile", False):
            # Pulsatile mode: extract per-field reduction (time_average or amplitude)
            averaged_data, res = self.extract_pulsatile_data(fname, i, fields, verbose=self.p.get("debug", False))
            # Archive the last time step (accumulated index)
            src_archive = join(self.p["f_out"], fname + str(self.p["n_max"]["fluid"] * i).zfill(3) + ".vtu")
            if os.path.exists(src_archive):
                trg = join(self.p["f_sim"], domain + "_out_" + i_str + ".vtu")
                shutil.copyfile(src_archive, trg)
        else:
            # Steady mode: read from single VTU file
            res = []
            for s in src:
                res += [read_geo(s).GetOutput()]
            averaged_data = None

        # extract fields
        for f in fields:
            if f == "wss":
                if averaged_data is not None and f in averaged_data and averaged_data[f] is not None:
                    # Pulsatile mode: use time-averaged WSS (already a vector array)
                    sol = averaged_data[f]
                else:
                    # Steady mode: smooth WSS from VTK cell data to point data
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

                        # get element-wise wss mapped to point data
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
                if averaged_data is not None and f in averaged_data and averaged_data[f] is not None:
                    # Pulsatile mode: use time-averaged data
                    sol = averaged_data[f]
                else:
                    # Steady mode: extract from VTU files
                    extr = []
                    for r in res:
                        if not r.GetPointData().HasArray(sv_names[f]):
                            raise ValueError("no array in PointData: " + sv_names[f])
                        extr += [v2n(r.GetPointData().GetArray(sv_names[f]))]
                    sol = np.mean(np.array(extr), axis=0)
                    if domain == "fluid" and self.p.get("debug", False):
                        print(f"      {f}: extracted from {len(extr)} geometries")
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
        f_time = t / self.p["nloads"]

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
