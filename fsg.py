#!/usr/bin/env python
# coding=utf-8

import pdb
import numpy as np
import shutil
import os
import glob
import time
from copy import deepcopy
from collections import defaultdict
import argparse

import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt

from vtk.util.numpy_support import vtk_to_numpy as v2n

from svfsi import svFSI, sv_names
from post import main_arg

from utilities import QRfiltering_mod
# NeuralOperator (and its torch dependency) is imported lazily in __init__ only
# when a neural_operator surrogate is enabled, so the standard FSI path runs
# without torch installed.


class FSG(svFSI):
    """
    FSG-specific stuff
    """

    def __init__(self, f_params=None):
        # svFSI simulations
        svFSI.__init__(self, f_params)

        # neural operator surrogate (replaces fluid + mesh per sub-iteration)
        no_cfg = self.p.get("neural_operator", {})
        if no_cfg.get("enabled"):
            if "work_dir" not in no_cfg:
                # Default work_dir inside this run's output dir so each FSG run
                # keeps its own lddmm_N/ directories (no global /tmp path).
                no_cfg = dict(no_cfg, work_dir=os.path.join(self.p["f_out"], "lddmm_work"))
            # Lazy import: only pull in torch/NeuralOperator when actually enabled.
            from neural_operator import NeuralOperator
            self.no = NeuralOperator(no_cfg)
        else:
            self.no = None

    def run_post(self):
        # todo: read in automatically
        self.err = np.load(
            "study_lab_meeting/fsi_res_2022-11-30_18-21-39.375658/err.npy",
            allow_pickle=True,
        ).item()
        self.p["f_out"] = "."
        self.plot_convergence()

    def save_restart(self, t, i):
        """Save all coupling state after converged load step t."""
        data = {
            "t": t,
            "i": i,
            "converged_count": len(self.converged),
            # store absolute path so load_restart can copy svFSI binary files
            "f_out": self.p["f_out"],
        }

        # IQN-ILS history: mat_W, mat_V (lists of 1D arrays)
        data["mat_W_count"] = len(self.mat_W)
        for j, v in enumerate(self.mat_W):
            data[f"mat_W_{j}"] = np.asarray(v)
        data["mat_V_count"] = len(self.mat_V)
        for j, v in enumerate(self.mat_V):
            data[f"mat_V_{j}"] = np.asarray(v)

        # residual and displacement increment lists (shared by IQN-ILS and aitken)
        data["res_count"] = len(self.res)
        for j, v in enumerate(self.res):
            data[f"res_{j}"] = np.asarray(v)

        # dk: defaultdict(list) of lists of arrays — keyed by field name (e.g. "disp")
        data["dk_keys"] = np.array(list(self.dk.keys()), dtype=object)
        for k, vlist in self.dk.items():
            data[f"dk_{k}_count"] = len(vlist)
            for j, v in enumerate(vlist):
                data[f"dk_{k}_{j}"] = np.asarray(v)

        # dtk: same structure as dk (used by aitken fallback in IQN-ILS)
        data["dtk_keys"] = np.array(list(self.dtk.keys()), dtype=object)
        for k, vlist in self.dtk.items():
            data[f"dtk_{k}_count"] = len(vlist)
            for j, v in enumerate(vlist):
                data[f"dtk_{k}_{j}"] = np.asarray(v)

        # err: defaultdict of lists-of-lists (load step -> sub-iteration errors)
        data["err_keys"] = np.array(list(self.err.keys()), dtype=object)
        for k, steps in self.err.items():
            data[f"err_{k}_nsteps"] = len(steps)
            for si, sublist in enumerate(steps):
                data[f"err_{k}_{si}"] = np.asarray(sublist)

        # omega: same nested structure as err
        omega = self.p["coup"]["omega"]
        data["omega_keys"] = np.array(list(omega.keys()), dtype=object)
        for k, steps in omega.items():
            data[f"omega_{k}_nsteps"] = len(steps)
            for si, sublist in enumerate(steps):
                data[f"omega_{k}_{si}"] = np.asarray(sublist)

        # current solution (needed to reconstruct boundary conditions on resume)
        for fname, arr in self.curr.sol.items():
            if arr is not None:
                data[f"curr_{fname}"] = np.asarray(arr)

        # converged solutions (needed for predictor extrapolation)
        for ci, sol in enumerate(self.converged):
            for fname, arr in sol.sol.items():
                if arr is not None:
                    data[f"converged_{ci}_{fname}"] = np.asarray(arr)

        # debug QR data (only populated when iqn_ils_debug is True)
        if self.p["coup"].get("iqn_ils_debug", False):
            np.save(os.path.join(self.p["f_out"], "restart_debug_qr.npy"), self.debug_qr)

        # snapshot stFile_last.bin for each solver alongside restart.npz so that
        # load_restart always gets the correct binary state even if later (failed)
        # load steps overwrite stFile_last.bin after the checkpoint was saved
        f_stfiles = os.path.join(self.p["f_out"], "restart_stfiles")
        os.makedirs(f_stfiles, exist_ok=True)
        for solver_subdir in self.p["out"].values():
            src = os.path.join(self.p["f_out"], solver_subdir, "stFile_last.bin")
            if os.path.exists(src):
                shutil.copy(src, os.path.join(f_stfiles, solver_subdir.replace("/", "_") + "_stFile_last.bin"))

        f_restart = os.path.join(self.p["f_out"], "restart.npz")
        np.savez(f_restart, **data)
        print(f"  [restart] saved to {f_restart} (t={t}, i={i})")

    def load_restart(self, f_restart):
        """Restore coupling state from restart file. Returns (t_done, i_done) of last completed step."""
        data = np.load(f_restart, allow_pickle=True)

        t = int(data["t"])
        i = int(data["i"])
        n_conv = int(data["converged_count"])

        # IQN-ILS history matrices
        self.mat_W = [data[f"mat_W_{j}"] for j in range(int(data["mat_W_count"]))]
        self.mat_V = [data[f"mat_V_{j}"] for j in range(int(data["mat_V_count"]))]

        # residual list
        self.res = [data[f"res_{j}"] for j in range(int(data["res_count"]))]

        # dk
        self.dk = defaultdict(list)
        for k in data["dk_keys"]:
            self.dk[k] = [data[f"dk_{k}_{j}"] for j in range(int(data[f"dk_{k}_count"]))]

        # dtk
        self.dtk = defaultdict(list)
        for k in data["dtk_keys"]:
            self.dtk[k] = [data[f"dtk_{k}_{j}"] for j in range(int(data[f"dtk_{k}_count"]))]

        # err
        self.err = defaultdict(list)
        for k in data["err_keys"]:
            n_steps = int(data[f"err_{k}_nsteps"])
            self.err[k] = [list(data[f"err_{k}_{si}"]) for si in range(n_steps)]

        # omega
        for k in data["omega_keys"]:
            n_steps = int(data[f"omega_{k}_nsteps"])
            self.p["coup"]["omega"][k] = [list(data[f"omega_{k}_{si}"]) for si in range(n_steps)]

        # current solution
        for fname in self.curr.sol.keys():
            key = f"curr_{fname}"
            if key in data.files:
                self.curr.sol[fname] = data[key]

        # converged solutions
        self.converged = []
        for ci in range(n_conv):
            sol = self.curr.copy()
            for fname in sol.sol.keys():
                key = f"converged_{ci}_{fname}"
                if key in data.files:
                    sol.sol[fname] = data[key]
            self.converged.append(sol)

        # copy snapshotted stFile_last.bin files into the new run's solver directories.
        # We use the snapshot saved alongside restart.npz (in restart_stfiles/) rather
        # than the live stFile_last.bin, because later (failed) load steps may have
        # overwritten the live file after the checkpoint was taken.
        old_f_out = str(data["f_out"])
        f_stfiles_snap = os.path.join(os.path.dirname(f_restart), "restart_stfiles")
        for solver_subdir in self.p["out"].values():
            dst_dir = os.path.join(self.p["f_out"], solver_subdir)
            os.makedirs(dst_dir, exist_ok=True)
            snap = os.path.join(f_stfiles_snap, solver_subdir.replace("/", "_") + "_stFile_last.bin")
            live = os.path.join(old_f_out, solver_subdir, "stFile_last.bin")
            src = snap if os.path.exists(snap) else live
            if os.path.exists(src):
                shutil.copy(src, os.path.join(dst_dir, "stFile_last.bin"))
                label = "snapshot" if src == snap else "live"
                print(f"  [restart] copied {label} stFile -> {dst_dir}/")
            else:
                print(f"  [restart] WARNING: no stFile found for {solver_subdir} — solver will start without restart")

        print(f"  [restart] loaded from {f_restart} (resuming after t={t}, i={i})")
        return t, i

    def run(self, f_restart=None):
        # CLI flag takes precedence; fall back to JSON "restart" field
        if f_restart is None:
            f_restart = self.p.get("restart", None)

        # restore coupling state if restarting
        t_start, i_start = 0, 0
        if f_restart is not None:
            t_done, i_done = self.load_restart(f_restart)
            t_start = t_done + 1
            i_start = i_done  # preserve counter so svFSI file numbering stays consistent

        # run simulation
        try:
            self.main(t_start=t_start, i_start=i_start)
        except KeyboardInterrupt:
            print("interrupted")
            pass

        # archive results
        self.archive()

        # plot convergence (skipped for arc-length / line search / weak /
        # uber_robin: none populate the single global self.p["coup"]["tol"]
        # plot_convergence() assumes)
        if (not self.p.get("arc_length", {}).get("enabled")
                and self.p["coup"]["method"] not in ["linesearch", "weak", "uber_robin"]):
            self.plot_convergence()

        # uber_robin's own convergence plot -- see its docstring for why it
        # needs a dedicated one (step-varying tolerance, per-step omega reset)
        if self.p["coup"]["method"] == "uber_robin":
            self.plot_uber_robin_history()

        # post process
        main_arg([self.p["f_out"]])

    def main(self, t_start=0, i_start=0):
        # print reynolds number
        print("Re = " + str(int(self.p["re"])))

        # Crisfield spherical arc-length continuation (opt-in via JSON "arc_length")
        if self.p.get("arc_length", {}).get("enabled"):
            self._run_arclength(i_start)
            return

        # Weak coupling with intra-step stimulus homotopy (line search on alpha)
        if self.p["coup"]["method"] == "linesearch":
            self._run_linesearch(t_start, i_start)
            return

        # Weak coupling: one fluid + one solid solve per load step, with
        # inter-step Aitken relaxation of the interface displacement.
        if self.p["coup"]["method"] == "weak":
            self._run_weak(t_start, i_start)
            return

        # AlphaEvolve-discovered scheme ("prog-uber-robin"): the real load
        # curve compressed to a handful of steps, each sub-iterated to a
        # fixed point with inter-step-reset Aitken relaxation. See
        # docs/coupling_algorithms.tex Section 4 for the derivation.
        if self.p["coup"]["method"] == "uber_robin":
            self._run_uber_robin(t_start, i_start)
            return

        # loop load steps (historical load-controlled scheme)
        i = i_start
        for t in range(t_start, self.p["nloads"] + 1):
            print(
                "=" * 30 + " t " + str(t) + " ==== fp "
                + "{:.2f}".format(self.p_vec[t]) + " " + "=" * 30
            )

            # optional tighter coupling tolerance for the LAST load step only
            # (opt-in via "coup": {"tol_final": ...}; no effect otherwise).
            if t == self.p["nloads"] and "tol_final" in self.p["coup"]:
                self.p["coup"]["tol"] = self.p["coup"]["tol_final"]

            # predict solution for next load step
            if t > 0:
                self.coup_predict(i, t)

            # loop sub-iterations
            for n in range(self.p["coup"]["nmax"]):
                i += 1
                times = {}
                if self.p["coup"]["method"] in ["static", "aitken"]:
                    status = self.coup_step_relax(i, t, n, times)
                elif self.p["coup"]["method"] == "iqn_ils":
                    status = self.coup_step_iqn_ils(i, t, n, times)
                else:
                    raise ValueError(
                        "Unknown coupling method " + self.p["coup"]["method"]
                    )

                # check if simulation failed
                for name, s in self.curr.sol.items():
                    if s is None:
                        print(name + " simulation failed")
                        self._save_failure_case(t, i)
                        return

                # screen output
                out = "i " + str(i - 1) + " \tn " + str(n) + "\t"
                for name, e in self.err.items():
                    out += "{:.2e}".format(e[-1][-1]) + "\t"
                if self.p["coup"]["method"] in ["static", "aitken"]:
                    for name, e in self.p["coup"]["omega"].items():
                        out += "{:.2e}".format(e[-1][-1]) + "\t"
                for f in times.keys():
                    out += "{:.2e}".format(times[f]) + "\t"

                if n == self.p["coup"]["nmax"] - 1:
                    out += "\n\tcoupling unconverged"
                    status = True
                print(out)

                self.save_tube(i)

                if status:
                    i_conv = str(i).zfill(3)
                    t_conv = str(t).zfill(3)
                    srcs = os.path.join(self.p["f_sim"], "*_" + i_conv + ".*")
                    for src in glob.glob(srcs):
                        trg = os.path.basename(src).replace(i_conv, t_conv)
                        trg = os.path.join(self.p["f_conv"], trg)
                        shutil.copyfile(src, trg)
                    self.converged += [self.curr.copy()]
                    if self.p.get("save_restart", False):
                        self.save_restart(t, i)
                    break

    # ======================================================================
    # Crisfield spherical arc-length continuation in (interface displacement d,
    # growth-load factor lambda). Solves the partitioned NN-FSG equilibrium
    # R(d,lambda) = solid(NN(d), lambda) - d = 0 together with the spherical
    # constraint  ||d - d_n||^2 + a*(lambda - lambda_n)^2 = dl^2,  so the path can
    # traverse a growth limit point where load control diverges. Uses the solver's
    # explicit idempotent restart (--restart-in/--restart-out): every solid
    # evaluation advances from the SAME committed step-(n-1) checkpoint, so
    # re-evaluating at a different lambda never compounds the G&R history.
    # Opt-in via JSON "arc_length": {enabled, dl, a, omega, tol, max_corr,
    # lambda_max, max_steps}. Requires wss_relax=1 (no WSS damping).
    # ======================================================================
    def _run_arclength(self, i_start):
        """Single-loop Crisfield spherical arc-length. Per load step, the EXISTING
        coupling algorithm (Aitken / IQN-ILS) drives the interface displacement;
        the load factor lambda rides inside the same sub-iteration loop. Each
        sub-iteration:
          1. solid solve S(F(d), lambda) at the current lambda (NN WSS = F(d)),
             advancing from the committed step-(t-1) checkpoint (idempotent
             --restart-in), so re-evaluating at a new lambda never compounds G&R;
          2. coupling relaxation (coup_step_*) produces the new interface disp;
          3. solve the sphere (centered at the previous converged step) for lambda
                ||d_new - d_old||^2 + phi^2 (lambda - lam_old)^2 = ds^2
             and take the LARGER root: lambda = lam_old + sqrt(.)/phi;
          4. repeat until the coupling algorithm converges.
        Config "arc_length": {enabled, ds, phi, lambda_max, max_steps}.
        Requires wss_relax=1 (no WSS damping).

        mode="nested" switches to the inner-fixed-lambda / outer-continuation
        scheme (see _run_arclength_nested); default "single" is this loop."""
        if self.p["arc_length"].get("mode", "single") in ("nested", "disp"):
            return self._run_arclength_nested(i_start)
        ac        = self.p["arc_length"]
        ds        = ac.get("ds", ac.get("dl", 0.1))
        phi       = ac.get("phi", 1.0)
        lam_relax = ac.get("lam_relax", 0.3)   # under-relax lambda over sub-iters
        lam_max   = ac.get("lambda_max", 1.0)
        nstep     = ac.get("max_steps", 200)
        ds_min    = ac.get("ds_min", 0.02 * ds)  # arc step floor -> below = limit point
        ds_max    = ac.get("ds_max", ds)         # arc step cap when growing
        nmax      = self.p["coup"]["nmax"]
        method    = self.p["coup"]["method"]
        control   = ac.get("control", "sphere")  # "sphere" or "disp" (uniform |dd|)
        # arc can take more steps than nloads; pad the pressure-factor vector
        # (p_vec[t] indexed in set_fluid/set_solid) with its last value (fmax).
        self.p_vec = np.concatenate([self.p_vec, np.full(nstep + 2, self.p_vec[-1])])

        os.makedirs(os.path.join(self.p["f_out"], "arc_ckpt"), exist_ok=True)
        ckpt    = os.path.join("arc_ckpt", "ckpt.bin")        # committed step t-1
        scratch = os.path.join("arc_ckpt", "scratch.bin")     # per-sub-iter write
        self._lam_sched = {}
        # trajectory log saved to arc_data.npy (read by post.plot_arc), mirroring
        # the debug_qr.npy / plot_cc convention
        self._arc_log = {"t": [], "n": [], "lam": [], "dd": [], "res": [], "ds": [],
                         "accept_t": [], "accept_lam": [], "tol": self.p["coup"]["tol"]}

        # CRITICAL: make the solid solver READ the arc load curve. Patch the
        # GR_equilibrated XML to load_profile="file" pointing at gr_load_curve.dat
        # (which _arc_set_load rewrites each sub-iter). Without this the solver
        # ignores the curve and falls back to its default tanh ramp -> lambda
        # never reaches the solid (the load factor would be a phantom).
        self.p["gr_load"] = {"profile": "file",
                             "curve": [[x, 0.0] for x in range(self.p["nloads"] + 1)]}
        self.set_gr_load()

        # --- prestress (load step 0): establish d_old + committed checkpoint ---
        # Run prestress through the normal coupling (no restart flags -> standard
        # stFile_last.bin continuity), works for NN and real-CFD; then snapshot the
        # converged solid restart to ckpt for the arc steps to advance from.
        i0 = i_start
        i, d_old = self._arc_prestress(i0)   # i = monotonic global sub-iter counter
        lam_old = 0.0
        self.converged += [self.curr.copy()]
        self._arc_good = self.curr.copy()   # last-good solution for retry restore
        sd = self.p["out"]["solid"]
        last = os.path.join(self.p["f_out"], sd, "stFile_last.bin")
        if os.path.exists(last):
            shutil.copyfile(last, os.path.join(self.p["f_out"], ckpt))
        # post(solid) reads the latest gr_restart while arc is active (the n=0
        # restart resets decouple the solver cTS from the monotonic index i).
        self._arc_active = True
        print("[arc] prestress |d|=%.4g finite=%s lam=0"
              % (np.linalg.norm(d_old), bool(np.all(np.isfinite(d_old)))))

        for step in range(1, nstep + 1):
            t = step

            # Final step: if a full arc increment would reach/cross lambda_max,
            # switch this step to LOAD CONTROL pinned at lambda_max so the path
            # lands EXACTLY on lambda=1 (skip the sphere update).
            final = (lam_old + ds / phi >= lam_max - 1e-12)

            # adaptive arc step: retry with halved ds on failure. ds shrinking below
            # ds_min while the step keeps failing => genuine limit point (fold).
            ds_try = ds
            ok = False
            n = 0
            err = np.nan
            d_new = d_old
            while True:
                # restore the last-good solution (a failed solid solve sets the
                # whole curr.sol field to None; the retry must start from it)
                self.curr = self._arc_good.copy()
                # fresh coupling history for this independent fixed-point solve
                self.res = []
                self.dk = defaultdict(list)
                self.dtk = defaultdict(list)
                self.mat_V = []
                self.mat_W = []

                # predictor: half a pure-load increment (the sphere pulls lambda to
                # the feasible value), or the exact target on the final step
                self._arc_lam = lam_max if final else lam_old + 0.5 * ds_try / phi
                ok = False
                lam_prev, dd_prev = lam_old, 0.0   # secant tracker (disp control)
                for n in range(nmax):
                    i += 1                  # monotonic per sub-iter (fluid/mesh advance)
                    # reset solid to the committed t-1 checkpoint only at n=0 (so a
                    # re-solve at a new lambda doesn't compound G&R); within the step
                    # the solid continues normally (like the vanilla loop), so the
                    # fluid re-solves on the updated mesh and WSS actually changes.
                    self.restart_in = ckpt if n == 0 else None
                    self.restart_out = None
                    self._arc_set_load(t, self._arc_lam)
                    times = {}
                    if method in ["static", "aitken"]:
                        status = self.coup_step_relax(i, t, n, times)
                    elif method == "iqn_ils":
                        status = self.coup_step_iqn_ils(i, t, n, times)
                    else:
                        raise ValueError("Unknown coupling method " + method)

                    if any(s is None for s in self.curr.sol.values()):
                        print("[arc]   solid failed (lam=%.4f, n=%d, ds=%.4f)"
                              % (self._arc_lam, n, ds_try))
                        self._arc_logrow(t, n, self._arc_lam, np.nan, np.nan, ds_try)
                        # A None solid solution means the solver aborted (e.g.
                        # element inversion -> negative Jacobian), not mere
                        # non-convergence. Save the crash state (last-good tube +
                        # svFSI's crash-state VTU) so the fold-adjacent failure
                        # geometry is preserved before ds is cut and retried.
                        self._save_failure_case(t, i)
                        break
                    d_new = deepcopy(self.curr.get(("solid", "disp", "int"))).flatten()
                    if not np.all(np.isfinite(d_new)):
                        print("[arc]   NaN disp (lam=%.4f, n=%d, ds=%.4f)"
                              % (self._arc_lam, n, ds_try))
                        self._arc_logrow(t, n, self._arc_lam, np.nan, np.nan, ds_try)
                        break

                    # PHYSICAL displacement metric: MEAN nodal displacement increment
                    # (cm), consistent with the coupling residual norm (coup_err also
                    # uses the mean nodal |.|). Commensurate with phi [cm]. (The
                    # full-field L2 norm scales ~sqrt(672*3) and is not a length.)
                    dd = float(np.linalg.norm((d_new - d_old).reshape(-1, 3), axis=1).mean())

                    # lambda update (skipped on the final load-controlled step):
                    #   sphere: larger root of ||dd||^2+phi^2 dlam^2 = ds^2 (load-dominated
                    #           -> ~uniform dlam, accelerating dd)
                    #   disp  : secant drive dd -> ds (uniform |dd| per step; dlam VARIES,
                    #           big early/stiff, small late/soft). ds = target mean|dd|.
                    if not final:
                        if control == "disp":
                            slope = (dd - dd_prev) / (self._arc_lam - lam_prev) \
                                if abs(self._arc_lam - lam_prev) > 1e-9 else dd / max(self._arc_lam - lam_old, 1e-6)
                            slope = max(slope, 1e-4)          # dd increases with lambda
                            lam_prev, dd_prev = self._arc_lam, dd
                            lam_tgt = self._arc_lam + (ds_try - dd) / slope
                            self._arc_lam = self._arc_lam + lam_relax * (lam_tgt - self._arc_lam)
                            self._arc_lam = min(max(self._arc_lam, lam_old), lam_max)
                        else:
                            rhs = ds_try * ds_try - dd * dd
                            lam_tgt = lam_old + np.sqrt(max(rhs, 0.0)) / phi
                            self._arc_lam = self._arc_lam + lam_relax * (lam_tgt - self._arc_lam)

                    err = self.err["disp"][-1][-1]
                    self._arc_logrow(t, n, self._arc_lam, dd, err, ds_try)
                    print("  [arc] t%d n%d lam=%.4f mean|dd|=%.4f |r|=%.2e ds=%.4f"
                          % (t, n, self._arc_lam, dd, err, ds_try))
                    if status:
                        ok = True
                        break

                if ok or final:
                    break
                ds_try *= 0.5
                if ds_try < ds_min:
                    print("[arc] LIMIT POINT near lambda=%.4f: step fails for ds "
                          "down to %.4g (likely a fold)" % (lam_old, ds_try * 2))
                    break
                print("[arc]   cutting ds -> %.4f and retrying step %d" % (ds_try, t))

            if not ok:
                print("[arc] stop: step %d did not converge (lam=%.4f, |r|=%.2e)"
                      % (t, self._arc_lam, err))
                self._arc_save()
                return

            # adaptive: grow ds if the step converged easily, else keep ds_try
            ds = min(ds_try * 1.3, ds_max) if (n + 1) <= 20 else ds_try

            # accept: commit checkpoint, advance the sphere center
            lam_new = self._arc_lam
            dd_mean = float(np.linalg.norm((d_new - d_old).reshape(-1, 3), axis=1).mean())
            print("[arc] step %d: lam=%.4f (dlam=%+.4f) mean|dd|=%.4f |r|=%.2e n=%d"
                  % (t, lam_new, lam_new - lam_old, dd_mean, err, n + 1))
            self._arc_commit(t, ckpt, last)
            self.converged += [self.curr.copy()]
            self._arc_good = self.curr.copy()   # update last-good for retry restore
            self._lam_sched[t] = lam_new
            self._arc_log["accept_t"].append(t)
            self._arc_log["accept_lam"].append(lam_new)
            d_old, lam_old = d_new, lam_new

            if lam_new >= lam_max:
                print("[arc] reached lambda_max=%.3f at step %d" % (lam_max, t))
                self._arc_save()
                return
        print("[arc] reached max_steps=%d" % nstep)
        self._arc_save()

    # ==================================================================
    # NESTED arc-length: OUTER continuation in lambda + INNER fixed-lambda
    # coupling solve (IQN-ILS / Aitken). Because lambda is CONSTANT inside the
    # inner solve, the map R(d,lambda)=S(F(d),lambda)-d is stationary, so the
    # IQN-ILS quasi-Newton history is valid (unlike the single-loop scheme).
    #   d(lambda) := inner solve of R(.,lambda)=0 (implicit function).
    #   Outer: find lambda s.t. the converged d(lambda) lies on the sphere
    #     g(lambda)= ||d(lambda)-d_old||^2 + phi^2 (lambda-lam_old)^2 - ds^2 = 0
    #   solved by a bracketed secant; each g-eval = one inner solve.
    # Config "arc_length": {mode:"nested", ds, phi, lambda_max, max_steps,
    #   tol_arc (rel. sphere tol, default 0.02), max_corr (default 8), ds_min}.
    # ==================================================================
    def _run_arclength_nested(self, i_start):
        ac        = self.p["arc_length"]
        ds        = ac.get("ds", 0.06)
        phi       = ac.get("phi", 0.6)
        lam_max   = ac.get("lambda_max", 1.0)
        nstep     = ac.get("max_steps", 40)
        ds_min    = ac.get("ds_min", 0.1 * ds)
        ds_max    = ac.get("ds_max", ds)
        tol_arc   = ac.get("tol_arc", 0.02)      # accept if |sphere_dist-ds| < tol_arc*ds
        max_corr  = ac.get("max_corr", 8)
        # arc can exceed nloads steps; pad the pressure-factor vector (see single-loop)
        self.p_vec = np.concatenate([self.p_vec, np.full(nstep + 2, self.p_vec[-1])])

        os.makedirs(os.path.join(self.p["f_out"], "arc_ckpt"), exist_ok=True)
        ckpt = os.path.join("arc_ckpt", "ckpt.bin")
        self._lam_sched = {}
        self._arc_log = {"t": [], "n": [], "lam": [], "dd": [], "res": [], "ds": [],
                         "accept_t": [], "accept_lam": [], "tol": self.p["coup"]["tol"]}

        # patch solid XML so the solver READS the load curve (else default tanh)
        self.p["gr_load"] = {"profile": "file",
                             "curve": [[x, 0.0] for x in range(self.p["nloads"] + 1)]}
        self.set_gr_load()

        # prestress -> d_old, committed checkpoint
        self._arc_i, d_old = self._arc_prestress(i_start)
        lam_old = 0.0
        self.converged += [self.curr.copy()]
        self._arc_good = self.curr.copy()
        sd = self.p["out"]["solid"]
        last = os.path.join(self.p["f_out"], sd, "stFile_last.bin")
        if os.path.exists(last):
            shutil.copyfile(last, os.path.join(self.p["f_out"], ckpt))
        self._arc_active = True
        dlam_sign = 1.0
        control = "disp" if ac.get("mode") == "disp" else "sphere"
        # displacement-control: uniform bulge increment ds per step, lambda solved
        # (lambda steps then VARY: large early/stiff, small late/soft). sphere:
        # ds/phi is the lambda predictor; disp: use the previous accepted |dlam|.
        dlam_guess = ds / phi
        print("[arc-nested] control=%s prestress |d|=%.4g lam=0"
              % (control, np.linalg.norm(d_old)))

        for step in range(1, nstep + 1):
            t = step
            ds_try = ds
            accepted = False
            while ds_try >= ds_min:
                self._arc_ds_cur = ds_try
                lam, d, ok = self._arc_outer_correct(
                    t, d_old, lam_old, phi, ds_try, tol_arc, max_corr,
                    ckpt, lam_max, dlam_sign, control=control, step_guess=dlam_guess)
                if ok:
                    accepted = True
                    break
                print("[arc-nested]   step %d: no root (ds=%.4f) -> cut"
                      % (t, ds_try))
                ds_try *= 0.5
            if not accepted:
                print("[arc-nested] LIMIT POINT near lambda=%.4f (fails to ds=%.4g)"
                      % (lam_old, ds_try * 2))
                self._arc_save()
                return

            # accept
            print("[arc-nested] step %d: lam=%.4f (dlam=%+.4f)  ds=%.4f"
                  % (t, lam, lam - lam_old, ds_try))
            shutil.copyfile(last, os.path.join(self.p["f_out"], ckpt))  # commit t-1<-t
            self.save_tube(t, self.p["f_conv"])
            self.converged += [self.curr.copy()]
            self._arc_good = self.curr.copy()
            self._lam_sched[t] = lam
            self._arc_log["accept_t"].append(t)
            self._arc_log["accept_lam"].append(lam)
            dlam_sign = 1.0 if (lam - lam_old) >= 0 else -1.0
            if abs(lam - lam_old) > 1e-6:
                dlam_guess = abs(lam - lam_old)  # predictor for next disp-control step
            d_old, lam_old = d, lam
            ds = min(ds_try * 1.3, ds_max)      # adaptive grow

            if lam_old >= lam_max - 1e-9:
                print("[arc-nested] reached lambda_max=%.3f at step %d" % (lam_max, t))
                self._arc_save()
                return
        print("[arc-nested] reached max_steps=%d" % nstep)
        self._arc_save()

    def _arc_outer_correct(self, t, d_old, lam_old, phi, ds, tol_arc, max_corr,
                           ckpt, lam_max, sign, control="sphere", step_guess=None):
        """Bracketed secant on the SIGNED constraint distance c(lambda):
          control="sphere": c = sqrt(||d(lam)-d_old||^2 + phi^2(lam-lam_old)^2) - ds
          control="disp"  : c = ||d(lam)-d_old||_meannodal - ds   (uniform disp step)
        c(lam) is monotone increasing in lam (dd grows with lam), c(lam_old)~-ds.
        d(lam) from a fixed-lambda inner solve. Returns (lam, d, ok). Accept |c|<tol_arc*ds.
        step_guess = lambda-increment for the bracket predictor (defaults: ds/phi
        for sphere, previous dlam for disp)."""
        if step_guess is None or step_guess <= 0:
            step_guess = ds / phi if control == "sphere" else 0.1

        def c_of(lam):
            d, conv = self._arc_inner_solve(t, lam, ckpt)
            if not conv:
                return None, None
            # MEAN nodal displacement increment (cm) -- consistent with coup_err.
            dd = float(np.linalg.norm((d - d_old).reshape(-1, 3), axis=1).mean())
            if control == "disp":
                c = dd - ds
            else:
                c = np.sqrt(dd * dd + phi * phi * (lam - lam_old) ** 2) - ds
            return c, d

        lo_lam, lo_c = lam_old, -ds              # c(lam_old) ~ -ds (dd~0) < 0
        hi_lam = min(max(lam_old + sign * step_guess, 0.0), lam_max)
        hi_c, d_hi = c_of(hi_lam)
        if hi_c is None:
            return hi_lam, None, False           # inner failed -> caller cuts ds
        expand = 0
        while hi_c < 0.0 and expand < 6:
            lo_lam, lo_c = hi_lam, hi_c
            hi_lam = min(hi_lam + sign * step_guess, lam_max)
            hi_c, d_hi = c_of(hi_lam)
            if hi_c is None:
                return hi_lam, None, False
            expand += 1
        if abs(hi_c) < tol_arc * ds:
            return hi_lam, d_hi, True

        a_lam, a_c = lo_lam, lo_c                 # secant/regula-falsi (a: c<0, b: c>0)
        b_lam, b_c, d_b = hi_lam, hi_c, d_hi
        for _ in range(max_corr):
            c_lam = b_lam - b_c * (b_lam - a_lam) / (b_c - a_c + 1e-30)
            c_lam = min(max(c_lam, 0.0), lam_max)
            cc, d_c = c_of(c_lam)
            if cc is None:
                return c_lam, None, False
            if abs(cc) < tol_arc * ds:
                return c_lam, d_c, True
            if (cc > 0.0) == (b_c > 0.0):
                b_lam, b_c, d_b = c_lam, cc, d_c
            else:
                a_lam, a_c = c_lam, cc
        return c_lam, d_c, False                 # no root in budget -> caller cuts ds

    def _arc_inner_solve(self, t, lam, ckpt):
        """Solve R(d,lambda)=0 at FIXED lambda from the committed t-1 checkpoint,
        using the existing coupling (IQN-ILS / Aitken). Returns (d_flat, converged).
        lambda constant => stationary map => IQN-ILS history is valid."""
        method = self.p["coup"]["method"]
        nmax = self.p["coup"]["nmax"]
        self.curr = self._arc_good.copy()        # start from last-good geometry
        self.res = []
        self.dk = defaultdict(list)
        self.dtk = defaultdict(list)
        self.mat_V = []
        self.mat_W = []
        self._arc_set_load(t, lam)               # FIXED lambda for the whole inner solve
        d = self._arc_good.get(("solid", "disp", "int")).flatten()
        ok = False
        for n in range(nmax):
            self._arc_i += 1
            self.restart_in = ckpt if n == 0 else None   # reset to t-1 at n=0 only
            self.restart_out = None
            times = {}
            if method in ["static", "aitken"]:
                status = self.coup_step_relax(self._arc_i, t, n, times)
            elif method == "iqn_ils":
                status = self.coup_step_iqn_ils(self._arc_i, t, n, times)
            else:
                raise ValueError("Unknown coupling method " + method)
            if any(s is None for s in self.curr.sol.values()):
                return d, False
            d = deepcopy(self.curr.get(("solid", "disp", "int"))).flatten()
            if not np.all(np.isfinite(d)):
                return d, False
            err = self.err["disp"][-1][-1]
            dd = float(np.linalg.norm(
                (d - self._arc_good.get(("solid", "disp", "int")).flatten()
                 ).reshape(-1, 3), axis=1).mean())
            self._arc_logrow(t, n, lam, dd, err, getattr(self, "_arc_ds_cur", 0.0))
            if status:
                ok = True
                break
        print("[arc-nested]   inner lam=%.4f -> conv=%s n=%d mean|dd|=%.4f |r|=%.2e"
              % (lam, ok, n + 1, dd, err))
        return d, ok

    def _arc_logrow(self, t, n, lam, dd, res, ds):
        """Append one trajectory row (converged sub-iter OR failed attempt)."""
        L = self._arc_log
        L["t"].append(t); L["n"].append(n); L["lam"].append(lam)
        L["dd"].append(dd); L["res"].append(res); L["ds"].append(ds)

    def _arc_save(self):
        """Persist the arc-length trajectory for post.plot_arc (cf. debug_qr.npy)."""
        np.save(os.path.join(self.p["f_out"], "arc_data.npy"), self._arc_log)

    def _arc_set_load(self, t, lam):
        """gr_load ramp curve: committed factors for steps < t, lam at step >= t."""
        n = self.p["nloads"]
        sched = getattr(self, "_lam_sched", {})
        path = os.path.join(self.p["f_out"], "in_svfsi", "gr_load_curve.dat")
        with open(path, "w") as f:
            for x in range(max(n, t) + 1):
                if x == 0:
                    y = 0.0
                elif x < t:
                    y = sched.get(x, lam)
                else:
                    y = lam
                f.write("%d %.10g\n" % (x, y))

    def _arc_commit(self, t, ckpt, scratch):
        """Promote the accepted scratch checkpoint to committed; archive the tube."""
        src = os.path.join(self.p["f_out"], scratch)
        dst = os.path.join(self.p["f_out"], ckpt)
        if os.path.exists(src):
            shutil.copyfile(src, dst)
        self.save_tube(t, self.p["f_conv"])

    def _arc_prestress(self, i0):
        """Run prestress (load step 0) through the normal coupling loop (works for
        NN and real-CFD), with standard stFile_last.bin continuity (no restart
        flags). Returns (final i, prestressed interface displacement flattened)."""
        self.restart_in, self.restart_out = None, None
        self._arc_set_load(0, 0.0)
        method = self.p["coup"]["method"]
        i = i0
        for n in range(self.p["coup"]["nmax"]):
            i += 1
            times = {}
            if method in ["static", "aitken"]:
                status = self.coup_step_relax(i, 0, n, times)
            elif method == "iqn_ils":
                status = self.coup_step_iqn_ils(i, 0, n, times)
            else:
                raise ValueError("Unknown coupling method " + method)
            if any(s is None for s in self.curr.sol.values()):
                raise RuntimeError("arc-length prestress failed at sub-iter %d" % n)
            if status:
                break
        return i, deepcopy(self.curr.get(("solid", "disp", "int"))).flatten()

    # ======================================================================
    # Weak coupling with an intra-step stimulus homotopy ("line search" on alpha).
    # Per load step: solve mesh + fluid ONCE on the entry wall -> "new" stimulus;
    # then advance the solid from the committed step-(t-1) checkpoint while blending
    # the applied WSS + pressure  S(alpha) = alpha*new + (1-alpha)*old  (old = the
    # previous load step's stimulus), alpha ramped 0 -> 1 by ADAPTIVE stepping: a
    # converged solid solve commits the state and grows alpha; an inversion halves
    # alpha and retries from the last-good alpha (via the solver's --restart-in /
    # --restart-out). The fluid is NOT re-solved during the ramp, so the coupling
    # iterations are replaced by the homotopy. Opt-in: "coup": {"method":
    # "linesearch", "alpha_ds":0.25, "alpha_ds_min":0.02, "alpha_ds_grow":1.5}.
    # ======================================================================
    def _run_linesearch(self, t_start, i_start):
        ac         = self.p["coup"]
        dalpha0    = ac.get("alpha_ds", 0.25)
        dalpha_min = ac.get("alpha_ds_min", 0.02)
        grow       = ac.get("alpha_ds_grow", 1.5)

        os.makedirs(os.path.join(self.p["f_out"], "ls_ckpt"), exist_ok=True)
        ckpt = os.path.join("ls_ckpt", "ckpt.bin")
        scr  = os.path.join("ls_ckpt", "scratch.bin")
        sd   = self.p["out"]["solid"]
        last = os.path.join(self.p["f_out"], sd, "stFile_last.bin")

        i   = i_start   # solid file/log counter (monotonic)
        i_f = i_start   # mesh/fluid file counter (advances once per load step)

        for t in range(t_start, self.p["nloads"] + 1):
            print("=" * 30 + " t " + str(t) + " ==== fp "
                  + "{:.2f}".format(self.p_vec[t]) + " " + "=" * 30)

            # ---- one mesh + fluid solve on the entry wall -> the "new" stimulus ----
            self.restart_in = self.restart_out = None
            self._arc_active = False
            times = {}
            i_f += 1
            if self.no is not None:
                self._neural_operator_step(times, i_f, t, 0)
            elif self.p["fsi"] and i_f > 1:
                if self.step("mesh", i_f, t, 0, times):
                    print("mesh simulation failed"); self._save_failure_case(t, i); return
            self.prev = self.curr.copy()          # entry wall (matches coup_step order)
            if self.no is None:
                if self.p["fsi"]:
                    if self.step("fluid", i_f, t, 0, times):
                        print("fluid simulation failed"); self._save_failure_case(t, i); return
                else:
                    self.poiseuille(t)
            new_wss   = deepcopy(self.curr.sol["wss"])
            new_press = deepcopy(self.curr.sol["press"])
            old_wss, old_press = getattr(self, "_stim_old", (new_wss, new_press))

            if t == 0:
                # prestress: plain solid solve under the baseline stimulus (new==old)
                i += 1
                if self.step("solid", i, t, 0, times):
                    print("solid simulation failed"); self._save_failure_case(t, i); return
                if os.path.exists(last):
                    shutil.copyfile(last, os.path.join(self.p["f_out"], ckpt))
                print("  [ls] t0 prestress converged")
            else:
                # ---- alpha homotopy from the committed step-(t-1) checkpoint ----
                self._arc_active = True      # post(solid) reads the latest gr_restart VTU
                good   = self.curr.copy()    # last-good full solution (restore on fail)
                alpha  = 0.0
                dalpha = dalpha0
                nsolve = 0
                while alpha < 1.0 - 1e-9:
                    trial = min(alpha + dalpha, 1.0)
                    self.curr = good.copy()
                    self.curr.sol["wss"]   = trial * new_wss   + (1.0 - trial) * old_wss
                    self.curr.sol["press"] = trial * new_press + (1.0 - trial) * old_press
                    i += 1
                    nsolve += 1
                    self.restart_in  = ckpt      # advance from last-good alpha
                    self.restart_out = scr
                    times = {}
                    failed = self.step("solid", i, t, 1, times)   # n=1 -> predictor off
                    if failed or any(s is None for s in self.curr.sol.values()):
                        self.curr = good.copy()
                        if dalpha <= dalpha_min + 1e-12:
                            print("  [ls] t%d alpha=%.3f: solid failed at dalpha_min -> stop"
                                  % (t, alpha))
                            self.restart_in = self.restart_out = None
                            self._save_failure_case(t, i); return
                        dalpha = max(dalpha * 0.5, dalpha_min)
                        print("  [ls] t%d solid failed at alpha=%.3f -> dalpha=%.3f"
                              % (t, trial, dalpha))
                        continue
                    # accepted: promote scratch -> committed, advance and grow alpha
                    shutil.copyfile(os.path.join(self.p["f_out"], scr),
                                    os.path.join(self.p["f_out"], ckpt))
                    good   = self.curr.copy()
                    alpha  = trial
                    dalpha = min(dalpha * grow, dalpha0)
                dd = float(np.linalg.norm(
                    (self.curr.get(("solid", "disp", "int")).flatten()
                     - self.prev.get(("solid", "disp", "int")).flatten()
                     ).reshape(-1, 3), axis=1).mean())
                print("  [ls] t%d converged: alpha=1 in %d solid solves, mean|dd|=%.4f"
                      % (t, nsolve, dd))

            # ---- accept the converged load step ----
            self._stim_old = (deepcopy(new_wss), deepcopy(new_press))
            self.save_tube(t, self.p["f_conv"])
            self.converged += [self.curr.copy()]
            self.restart_in = self.restart_out = None

        self._arc_active = False

    # ======================================================================
    # Weak coupling with INTER-STEP Aitken relaxation of the interface
    # displacement d -- no sub-iteration loop, no alpha-ramp. Each load step
    # t>0 performs exactly ONE fluid solve and ONE solid solve (full target
    # stimulus, no blending) and is always accepted; the coupling
    # accelerator's history is built entirely from the OUTER loop, i.e. one
    # residual per load step, not per sub-iter:
    #     s      <- Fsolve(wall(d_{t-1}))         one fluid solve on prev wall
    #     d_t*   <- Ssolve(s, t)                   one solid solve, full target,
    #                                               n=0 ("beginning of new load
    #                                               step" -> restart from
    #                                               committed step-(t-1) state,
    #                                               same convention as
    #                                               coup_step_relax)
    #     r_t    <- d_t* - d_{t-1}                 step residual
    #     omega_t (Aitken Delta^2, Kuettler) built from (r_{t-1}, r_t) only
    #     d_t    <- omega_t*d_t* + (1-omega_t)*d_{t-1}
    # omega and the previous residual persist ACROSS t (they are not reset each
    # step). A failed solid solve is fatal (no back-off/retry). Config "coup":
    # {"method":"weak", "omega0":0.1, "omega_min":0.1}.
    #
    # Note: d's relaxation only reaches the interface geometry used for the
    # NEXT step's fluid mesh morph (set_mesh reads self.curr) -- the solid
    # solver's own restart continuation always resumes from its last raw
    # (unrelaxed) output, since displacement is solver-output-only (set_solid
    # writes wss/pressure, never disp). So d-relaxation here damps what wss the
    # solid sees one step later; it does not make the relaxed d authoritative
    # for the solid's own state.
    # ======================================================================
    def _fluid_solve(self, i_f, t, n=0):
        """One mesh+fluid (or NN) solve on the current wall. Returns True on
        failure.

        `n` is the caller's sub-iteration index within load step `t` (leave
        at the default 0 if the caller has no real sub-iteration concept --
        e.g. _run_weak, _run_linesearch). It's threaded through to every
        solver call, including the neural-operator surrogate, so a method
        WITH genuine sub-iteration (currently only _run_uber_robin) gets
        the NN surrogate's designed WSS ramp-up (_wss_relax_beta(n), which
        damps the NN's WSS prediction more in a step's early sub-iterations
        and relaxes it as n grows -- see _neural_operator_step) instead of
        always seeing n=0 regardless of actual progress. Previously
        duplicated, with exactly this n=0 bug, as separate closures inside
        _run_weak and _run_uber_robin.
        """
        times = {}
        if self.no is not None:
            self._neural_operator_step(times, i_f, t, n)
        elif self.p["fsi"] and i_f > 1:
            if self.step("mesh", i_f, t, n, times):
                print("mesh simulation failed"); return True, times
        if self.no is None:
            if self.p["fsi"]:
                if self.step("fluid", i_f, t, n, times):
                    print("fluid simulation failed"); return True, times
            else:
                self.poiseuille(t)
        return False, times

    def _run_weak(self, t_start, i_start):
        ac        = self.p["coup"]
        omega0    = ac.get("omega0", 0.1)
        omega_min = ac.get("omega_min", 0.1)

        i   = i_start   # solid file/log counter (monotonic)
        i_f = i_start   # mesh/fluid file counter

        omega    = omega0   # persists across load steps (outer-loop history)
        res_prev = None

        # dedicated omega/residual history (per load step t), independent of the
        # self.err/self.p["coup"]["omega"] bookkeeping used by the other coupling
        # methods (which _run_weak never populates). Written to
        # f_out/weak_omega_history.json after every step so a mid-run failure
        # still leaves a complete record up to the last successful step.
        self.weak_history = []
        hist_path = os.path.join(self.p["f_out"], "weak_omega_history.json")

        def _save_weak_history():
            import json as _json
            with open(hist_path, "w") as f:
                _json.dump(self.weak_history, f, indent=2)

        for t in range(t_start, self.p["nloads"] + 1):
            print("=" * 30 + " t " + str(t) + " ==== fp "
                  + "{:.2f}".format(self.p_vec[t]) + " " + "=" * 30)

            # ---- t == 0: prestress, one plain solid solve ----
            if t == 0:
                i_f += 1
                failed, times = self._fluid_solve(i_f, t)
                if failed:
                    self._save_failure_case(t, i); return
                self.prev = self.curr.copy()
                i += 1
                if self.step("solid", i, t, 0, times):
                    print("solid simulation failed"); self._save_failure_case(t, i); return
                print("  [weak] t0 prestress converged")
                self.err["disp"].append([1.0])  # placeholder: no prior state to compare
                self.save_tube(t, self.p["f_conv"])
                self.converged += [self.curr.copy()]
                continue

            # ---- t > 0: ONE fluid + ONE solid solve, inter-step Aitken relaxation ----
            wall     = self.curr.copy()                              # d_{t-1}
            wall_int = wall.get(("solid", "disp", "int")).flatten()

            i_f += 1
            failed, times = self._fluid_solve(i_f, t)
            if failed:
                self._save_failure_case(t, i); return

            # single solid solve; n=0 -> always "beginning of new load step"
            # (restart from committed step-(t-1) state), since there is no
            # within-step sub-iteration to continue from.
            i += 1
            times_s = {}
            failed = self.step("solid", i, t, 0, times_s)
            if failed or any(s is None for s in self.curr.sol.values()):
                print("  [weak] t%d: solid failed" % t)
                self._save_failure_case(t, i); return

            # step residual r_t = d_t* - d_{t-1}
            d_star = self.curr.get(("solid", "disp", "int")).flatten()
            r = d_star - wall_int

            # Aitken Delta^2 (Kuettler) update of omega from (r_{t-1}, r_t)
            if res_prev is not None:
                diff  = r - res_prev
                denom = float(np.dot(diff, diff))
                if denom > 0.0:
                    omega = -omega * float(np.dot(res_prev, diff)) / denom
                    omega = min(max(omega, omega_min), 1.0)
            res_prev = r

            # relaxed interface update d_t <- omega*d_t* + (1-omega)*d_{t-1}
            curr_v = self.curr.get(("solid", "disp", "vol"))    # d_t*
            prev_v = wall.get(("solid", "disp", "vol"))         # d_{t-1}
            self.curr.add(("solid", "disp", "vol"),
                          omega * curr_v + (1.0 - omega) * prev_v)

            err = float(np.mean(np.linalg.norm(r.reshape(-1, 3), axis=1)))
            print("  [weak] t%d: mean|r|=%.3e, omega=%.3f" % (t, err, omega))

            self.weak_history.append({"t": t, "mean_r": err, "omega": omega})
            _save_weak_history()

            # also feed the standard self.err bookkeeping (one singleton
            # sub-iteration list per load step) so archive()/compare_results.py
            # work unmodified, same as the other coupling methods.
            self.err["disp"].append([err])

            self.save_tube(t, self.p["f_conv"])
            self.converged += [self.curr.copy()]

    def _run_uber_robin(self, t_start, i_start):
        """AlphaEvolve-discovered coupling scheme ("prog-uber-robin"),
        ported back from an evolutionary search over `_run_weak`. The real
        load curve is resampled onto `n_steps` (default 10) compressed
        outer steps instead of the usual ~80, and -- unlike `_run_weak`,
        which advances the load on every single solve with no inner
        sub-iteration at all -- each step now sub-iterates a fluid+solid
        exchange to a fixed point AT that step's fixed load target before
        advancing. The sub-iteration budget and tolerance are graded by
        step: cheap and loose for intermediate steps, expensive and tight
        for the final step, since that is the only state later compared
        against any accuracy reference. Aitken Delta^2 relaxation is
        applied WITHIN each step's sub-iteration and reset to `omega0` at
        the start of every step (unlike `_run_weak`, where it persists
        and is applied ACROSS steps instead). See docs/coupling_algorithms.tex
        Section 4 for the full derivation and the Algorithm 4 pseudocode
        this implements line-for-line.

        Opt-in via {"coup": {"method": "uber_robin", ...}}. Config knobs
        (all optional, defaults match the discovered algorithm):
          n_steps                 compressed outer step count N (default 10)
          omega0                  initial/reset relaxation factor (default 0.1)
          omega_min               relaxation floor (default 0.1)
          n_max_mid, n_max_final  sub-iteration budget, intermediate vs
                                   final step (default 2, 6)
          tol_mid, tol_final_sub  sub-iteration residual tolerance,
                                   intermediate vs final step (default 1e-4, 2e-5)

        Recording/visualization: every sub-iteration is appended to
        self.err["disp"] / self.p["coup"]["omega"]["disp"] in the same
        list-per-step format the "static"/"aitken" methods use, so
        plot_convergence() renders the usual residual+omega figure; a
        richer per-sub-iteration record (t, n, s_t, mean_r, omega,
        converged, n_max, tol) is written incrementally to
        f_out/uber_robin_history.json (robust to a mid-run kill) and
        summarized by plot_uber_robin_history(); every sub-iteration's
        intermediate state is also saved as a VTU under
        f_out/sub_iterations/, tagged t<step>_n<sub-iter>, so the
        within-step convergence trajectory -- not just the converged
        per-step states in partitioned/converged/ -- can be visualized.
        """
        import json as _json
        import math as _math

        ac          = self.p["coup"]
        n_steps     = int(ac.get("n_steps", 10))
        omega0      = ac.get("omega0", 0.1)
        omega_min   = ac.get("omega_min", 0.1)
        n_max_mid   = int(ac.get("n_max_mid", 2))
        n_max_final = int(ac.get("n_max_final", 6))
        tol_mid     = ac.get("tol_mid", 1e-4)
        tol_final   = ac.get("tol_final_sub", 2e-5)

        i   = i_start   # solid file/log counter (monotonic)
        i_f = i_start   # mesh/fluid file counter

        self.uber_robin_history = []
        hist_path = os.path.join(self.p["f_out"], "uber_robin_history.json")
        sub_dir = os.path.join(self.p["f_out"], "sub_iterations")
        os.makedirs(sub_dir, exist_ok=True)

        def _save_history():
            with open(hist_path, "w") as f:
                _json.dump(self.uber_robin_history, f, indent=2)

        # ---- t == 0: prestress, one plain solid solve, no sub-iteration ----
        i_f += 1
        failed, times = self._fluid_solve(i_f, 0)
        if failed:
            self._save_failure_case(0, i); return
        self.prev = self.curr.copy()
        i += 1
        if self.step("solid", i, 0, 0, times):
            print("solid simulation failed"); self._save_failure_case(0, i); return
        print("  [uber_robin] t0 prestress converged")
        self.err["disp"].append([1.0])
        self.p["coup"]["omega"]["disp"].append([1.0])  # no relaxation at prestress
        self.save_tube(0, self.p["f_conv"])
        self.save_tube("t000_n0", sub_dir)
        self.converged += [self.curr.copy()]
        self.uber_robin_history.append(
            {"t": 0, "n": 0, "s_t": 0.0, "mean_r": None, "omega": 1.0,
             "converged": True, "n_max": 1, "tol": None})
        _save_history()

        for t in range(max(t_start, 1), n_steps + 1):
            s_t = _math.tanh(2 * t / n_steps) / _math.tanh(2)
            is_final = (t == n_steps)
            n_max = n_max_final if is_final else n_max_mid
            tol   = tol_final if is_final else tol_mid

            print("=" * 30 + " t " + str(t) + " ==== s " + "{:.3f}".format(s_t)
                  + " (n_max=%d, tol=%.1e) " % (n_max, tol) + "=" * 30)

            omega    = omega0   # reset every outer step -- see Section 4
            res_prev = None
            self.err["disp"].append([])
            self.p["coup"]["omega"]["disp"].append([])

            for n in range(n_max):
                d_in     = self.curr.copy()                              # state entering this sub-iteration
                d_in_int = d_in.get(("solid", "disp", "int")).flatten()

                i_f += 1
                failed, times = self._fluid_solve(i_f, t, n)
                if failed:
                    self._save_failure_case(t, i); return

                # single solid solve at the fixed target s_t; load does not
                # advance within the sub-iteration, only the interface state
                # is refined
                i += 1
                failed = self.step("solid", i, t, n, times)
                if failed or any(s is None for s in self.curr.sol.values()):
                    print("  [uber_robin] t%d n%d: solid failed" % (t, n))
                    self._save_failure_case(t, i); return

                d_star_int = self.curr.get(("solid", "disp", "int")).flatten()
                r = d_star_int - d_in_int
                err = float(np.mean(np.linalg.norm(r.reshape(-1, 3), axis=1)))

                # Aitken Delta^2 (Kuettler), across sub-iterations at fixed s_t
                if res_prev is not None:
                    diff  = r - res_prev
                    denom = float(np.dot(diff, diff))
                    if denom > 0.0:
                        omega = -omega * float(np.dot(res_prev, diff)) / denom
                        omega = min(max(omega, omega_min), 1.0)

                # relaxed update, applied whether converged or not
                curr_v = self.curr.get(("solid", "disp", "vol"))    # d*
                prev_v = d_in.get(("solid", "disp", "vol"))         # d_in
                self.curr.add(("solid", "disp", "vol"),
                              omega * curr_v + (1.0 - omega) * prev_v)

                converged = err < tol or n == n_max - 1
                print("  [uber_robin] t%d n%d: mean|r|=%.3e, omega=%.3f%s"
                      % (t, n, err, omega, "  [accept]" if converged else ""))

                self.err["disp"][-1].append(err)
                self.p["coup"]["omega"]["disp"][-1].append(omega)
                self.uber_robin_history.append(
                    {"t": t, "n": n, "s_t": s_t, "mean_r": err, "omega": omega,
                     "converged": converged, "n_max": n_max, "tol": tol})
                _save_history()
                self.save_tube("t%03d_n%d" % (t, n), sub_dir)

                if converged:
                    break
                res_prev = r

            self.save_tube(t, self.p["f_conv"])
            self.converged += [self.curr.copy()]

    def plot_uber_robin_history(self):
        """Dedicated convergence plot for _run_uber_robin. Unlike the
        generic plot_convergence(), this makes the per-step reset of omega
        and the step-varying sub-iteration tolerance explicit, since both
        are the defining feature of this scheme (see Section 4 of
        docs/coupling_algorithms.tex). Reads self.uber_robin_history if
        available, else falls back to the on-disk JSON (so it can be
        called standalone, post-hoc, on an old run's output directory)."""
        hist = getattr(self, "uber_robin_history", None)
        if not hist:
            hist_path = os.path.join(self.p["f_out"], "uber_robin_history.json")
            if not os.path.exists(hist_path):
                return
            import json as _json
            with open(hist_path) as f:
                hist = _json.load(f)
        if not hist:
            return

        fig, (ax_r, ax_o) = plt.subplots(2, 1, figsize=(14, 6), dpi=200, sharex="all")

        xs  = list(range(len(hist)))
        res = [h["mean_r"] for h in hist]
        omg = [h["omega"] for h in hist]
        tol = [h["tol"] for h in hist]
        step_starts = [(i, h["t"]) for i, h in enumerate(hist) if h["n"] == 0]

        ax_r.plot(xs, [r if r is not None else np.nan for r in res], "k.-")
        ax_r.plot(xs, [tl if tl is not None else np.nan for tl in tol],
                  "k--", alpha=0.5, label=r"tolerance $\varepsilon(t)$")
        ax_r.set_yscale("log")
        ax_r.set_ylabel("sub-iteration residual")
        ax_r.legend(loc="upper right")

        ax_o.plot(xs, omg, "r.-")
        ax_o.axhline(hist[0]["omega"], color="r", ls=":", alpha=0.4, label=r"$\omega_0$")
        ax_o.set_ylim(0.0, 1.0)
        ax_o.set_ylabel(r"$\omega$")
        ax_o.set_xlabel("sub-iteration index (resets to $\\omega_0$ at each dashed line)")
        ax_o.legend(loc="upper right")

        for i, _ in step_starts:
            for ax in (ax_r, ax_o):
                ax.axvline(i - 0.5, color="gray", ls="-", alpha=0.3)
        ax_r.set_xticks([i for i, _ in step_starts],
                         ["$t_{%d}$" % t for _, t in step_starts])

        n_steps_seen = max(h["t"] for h in hist)
        ax_r.set_title("prog-uber-robin: N=%d compressed steps, %d total sub-iterations"
                        % (n_steps_seen, len(hist)))

        fig.savefig(os.path.join(self.p["f_out"], "uber_robin_convergence.png"),
                    bbox_inches="tight")
        plt.close(fig)

    def plot_convergence(self):
        n_sol = len(self.err.keys())
        col_err = "k"
        col_omg = "r"

        n_iter = [0]
        fig, ax = plt.subplots(
            n_sol, 1, figsize=(20, 4), dpi=200, sharex="all", sharey="all"
        )
        for i, name in enumerate(self.err.keys()):
            # get_axis handle
            if n_sol == 1:
                axi = ax
            else:
                axi = ax[i]

            # get iteration counts
            if i == 0:
                n_iter += [len(res) for res in self.err[name]]
                n_iter = np.cumsum(n_iter)

            # second axis for omega
            if self.p["coup"]["method"] in ["static", "aitken"]:
                ax2 = axi.twinx()

            # collect results
            for j, res in enumerate(self.err[name]):
                # iteration numbers
                x = np.arange(n_iter[j], n_iter[j + 1])

                # plot error
                axi.plot(x, res, linestyle="-", color=col_err)

                # plot omega
                if self.p["coup"]["method"] in ["static", "aitken"]:
                    ax2.plot(x, self.p["coup"]["omega"][name][j], color=col_omg)

            # plot convergence criterion
            axi.plot([0, n_iter[-1]], self.p["coup"]["tol"] * np.ones(2), "k--")

            # axis settings
            axi.tick_params(axis="y", colors=col_err)
            axi.set_xticks(
                n_iter[1:] - 1,
                [
                    "$t_{" + str(i) + "}$, n=" + str(j)
                    for i, j in enumerate(np.diff(n_iter))
                ],
            )
            axi.set_xticks(np.arange(0, n_iter[-1]), minor=True)
            axi.set_xlim([0, n_iter[-1]])
            axi.set_ylabel("Residual " + sv_names[name], color=col_err)
            axi.set_yscale("log")
            axi.set_ylim([self.p["coup"]["tol"] * 0.1, 10])
            axi.grid(which="minor", alpha=0.2)
            axi.grid(which="major", alpha=0.9)
            if i == len(self.err.keys()) - 1:
                axi.set_xlabel("Number of iterations $n$ per time step $t$")

            if self.p["coup"]["method"] in ["static", "aitken"]:
                ax2.tick_params(axis="y", colors=col_omg)
                ax2.set_ylabel("Omega", color=col_omg)
                ax2.set_ylim([0.0, 1.0])
                ax2.set_yticks(np.linspace(0, 1, 6))

            axi.set_title("Total iterations: " + str(n_iter[-1]))

        # save to file
        fig.savefig(
            os.path.join(self.p["f_out"], "convergence.png"), bbox_inches="tight"
        )
        # plt.show()
        plt.close(fig)

    def archive(self):
        # save debug QR data if enabled
        if self.p["coup"].get("iqn_ils_debug", False):
            np.save(os.path.join(self.p["f_out"], "debug_qr.npy"), self.debug_qr)

        # save stored results
        self.p["error"] = self.err

        # save parameters
        self.save_params(self.p["name"] + ".json")

        # save input files
        for src in self.p["inp"].values():
            trg = os.path.join(self.p["f_arx"], os.path.basename(src))
            shutil.copyfile(os.path.join(self.p["paths"]["in_svfsi"], src), trg)

        # save python scripts
        sp = os.path.dirname(os.path.realpath(__file__))
        for src in ["fsg.py", "svfsi.py"]:
            trg = os.path.join(self.p["f_arx"], os.path.basename(src))
            shutil.copyfile(os.path.join(sp, src), trg)

        # save material model
        f_code = os.path.join(
            self.p["paths"]["exe"], os.path.split(self.p["exe"]["solid"])[0]
        )
        cpp_files = f_code + "/../../../Code/Source/svFSI/gr_*.*"
        for src in glob.glob(cpp_files):
            trg = os.path.join(self.p["f_arx"], os.path.basename(src))
            shutil.copyfile(src, trg)

    def _save_failure_case(self, t, i):
        """Save the crashing load step's state to this run's OWN result directory.

        Each run keeps its failures self-contained under
        <f_out>/failed_cases/<tag>/ (rather than a shared global pool), holding:
          - interface_displacement.dat, meta.json (last-good interface geometry)
          - tube_lastgood.vtu (full last-good volume: mesh + GR stimuli +
            Jacobian + WSS + gr_properties)
          - gr_restart_crash_<cTS>.vtu (svFSI's crash-state dump of the actual
            inverted configuration at the negative-Jacobian abort)
        If a shared "failed_cases_dir" is configured (the active-learning oracle
        pool), the lightweight .dat + meta are ALSO copied there.
        """
        import json as _json
        import sys as _sys

        # write_displacement_file lives in a separate, machine-local pipeline
        # (the shared active-learning oracle pool) that isn't present on every
        # machine this repo runs on. Its output is a bonus for that pipeline,
        # not something the coupling loop depends on, so its absence must not
        # abort the run: fall back to skipping just the geometry dump below.
        _sys.path.insert(0, "/home/shiyi/TAA_CFD_pipeline")
        try:
            from generate_displacement import write_displacement_file
        except ImportError:
            write_displacement_file = None

        tag      = f"t{t:03d}_i{i:03d}_{int(time.time())}"
        # Primary location: the run's own output directory.
        case_dir = os.path.join(self.p["f_out"], "failed_cases", tag)
        os.makedirs(case_dir, exist_ok=True)

        # Solid solver failed so curr["solid","disp","int"] is None.
        # Use the last successful displacement from self.prev instead.
        try:
            disp = self.curr.get(("solid", "disp", "int"))
        except (ValueError, KeyError):
            if not hasattr(self, "prev"):
                print("  [failure] no previous state available — skipping geometry save")
                return
            disp = self.prev.get(("solid", "disp", "int"))   # (N, 3)
        surf = self.mesh[("int", "solid")]                # vtkPolyData
        ids  = v2n(surf.GetPointData().GetArray("GlobalNodeID")).astype(int)

        if write_displacement_file is not None:
            write_displacement_file(
                os.path.join(case_dir, "interface_displacement.dat"), ids, disp)
        _json.dump({"t": t, "i": i, "tag": tag, "case_dir": case_dir},
                   open(os.path.join(case_dir, "meta.json"), "w"), indent=2)
        print(f"  [failure] geometry saved → {case_dir}"
              if write_displacement_file is not None else
              f"  [failure] meta saved → {case_dir} (generate_displacement unavailable, geometry dump skipped)")

        # Richer debugging state for the inversion (the crashing load step
        # otherwise has no VTU): (a) the last-good full-volume tube; (b) svFSI's
        # crash-state VTU(s) dumped at the negative-Jacobian abort. Only crash
        # VTUs not already collected by an earlier failure are copied (the solid
        # folder is written by the container solver as root, so we copy rather
        # than move). Wrapped so a snapshot hiccup never aborts the run.
        if hasattr(self, "prev"):
            try:
                self.prev.archive("tube", os.path.join(case_dir, "tube_lastgood.vtu"))
                print(f"  [failure] last-good tube → {case_dir}/tube_lastgood.vtu")
            except Exception as e:
                print(f"  [failure] could not save last-good tube: {e}")
        if not hasattr(self, "_crashes_seen"):
            self._crashes_seen = set()
        crash_dir = os.path.join(self.p["f_out"], self.p["out"]["solid"])
        for src in sorted(glob.glob(os.path.join(crash_dir, "*_crash_*.vtu"))):
            if src in self._crashes_seen:
                continue
            self._crashes_seen.add(src)
            try:
                shutil.copy(src, case_dir)
                print(f"  [failure] crash-state VTU → {case_dir}/{os.path.basename(src)}")
            except Exception as e:
                print(f"  [failure] could not copy crash VTU {os.path.basename(src)}: {e}")

        # Optionally feed the shared oracle pool the lightweight geometry only.
        pool = self.p.get("failed_cases_dir")
        if pool and os.path.abspath(pool) != os.path.abspath(
                os.path.join(self.p["f_out"], "failed_cases")):
            pool_dir = os.path.join(pool, tag)
            os.makedirs(pool_dir, exist_ok=True)
            for fn in ("interface_displacement.dat", "meta.json"):
                shutil.copy(os.path.join(case_dir, fn), pool_dir)
            print(f"  [failure] oracle-pool copy → {pool_dir}")

    def _wss_relax_beta(self, n):
        """
        WSS under-relaxation factor beta in [beta0, beta_max].

        Aneurysm G&R is positive-feedback (bulge -> lower WSS -> more growth), so
        the NN's WSS error can run the growth away into element inversion
        (gr_equilibrated "Negative Jacobian") in the early sub-iterations of a
        load step.  We damp the WSS *deviation* from the homeostatic baseline at
        the start of each step (beta = beta0) and ramp beta up toward beta_max.

        Config (under "coup"):
          wss_relax       beta0, the start-of-step floor (default 1.0 = no damping)
          wss_relax_max   beta_max, the cap (default 1.0)
          wss_ramp_mode   "residual" (ramp keyed to coupling residual) or
                          "subiter"  (ramp keyed to sub-iteration index n).
                          "subiter" avoids the residual->beta->residual feedback
                          that limit-cycles when beta_max == 1.
          wss_relax_full  (residual mode) residual at/below which beta = beta_max
                          (default 1e-2)
          wss_ramp_iters  (subiter mode) n over which beta0 -> beta_max (default 10)
          wss_ramp_profile ramp shape: "linear" | "quad" | "sqrt" | "exp"
                          (default "linear")
        """
        c = self.p["coup"]
        beta0    = c.get("wss_relax", 1.0)
        if beta0 >= 1.0:
            return 1.0
        beta_max = c.get("wss_relax_max", 1.0)
        mode     = c.get("wss_ramp_mode", "residual")

        def shape(x):
            x = float(np.clip(x, 0.0, 1.0))
            prof = c.get("wss_ramp_profile", "linear")
            if prof == "quad":   return x * x          # stay damped longer
            if prof == "sqrt":   return np.sqrt(x)     # rise fast
            if prof == "exp":    return 1.0 - np.exp(-3.0 * x)
            return x                                   # linear

        if mode == "subiter":
            # beta depends only on the sub-iteration index -> no residual feedback
            N = max(int(c.get("wss_ramp_iters", 10)), 1)
            return beta0 + (beta_max - beta0) * shape(n / N)

        # residual mode: ramp as the coupling residual falls 1 -> wss_relax_full
        res = None
        if n > 0:
            cands = [e[-1][-1] for e in self.err.values() if e and e[-1]]
            if cands:
                res = max(cands)            # slowest field governs damping
        if res is None or res >= 1.0:
            return beta0
        res_full = c.get("wss_relax_full", 1e-2)
        if res <= res_full:
            return beta_max
        frac = np.log10(res) / np.log10(res_full)   # res=1 -> 0, res=res_full -> 1
        return beta0 + (beta_max - beta0) * shape(frac)

    def _neural_operator_step(self, times, i, t, n=0):
        """Replace mesh + fluid: LDDMM registration → pull-back → NN → WSS + pressure."""
        disp       = self.curr.get(("solid", "disp", "int"))  # (N_solid, 3)
        solid_xyz  = self.points[("int", "solid")]             # (N_solid, 3) reference
        solid_mesh = self.mesh[("int", "solid")]               # vtkPolyData connectivity
        wss, pressure = self.no.predict_wss_and_pressure(disp, solid_xyz, solid_mesh, call_id=i)

        # Residual-ramped growth under-relaxation (see _wss_relax_beta).
        beta = self._wss_relax_beta(n)
        if beta < 1.0:
            if not hasattr(self, "_wss_homeo"):
                zero = np.zeros_like(disp)
                self._wss_homeo, _ = self.no.predict_wss_and_pressure(
                    zero, solid_xyz, solid_mesh, call_id=-1)
            wss = self._wss_homeo + beta * (wss - self._wss_homeo)

        # NN predicts gauge pressure (CFD with zero outlet BC).
        # Add the absolute outlet pressure baseline p0 * p_vec[t] to match Poiseuille convention.
        p0 = self.p["fluid"]["p0"] * self.p_vec[t]
        pressure = pressure + p0
        self.curr.add(("fluid", "wss", "int"), wss)
        self.curr.add(("solid", "press", "int"), pressure)
        times["mesh"] = 0.0
        times["fluid"] = 0.0

    def coup_step_iqn_ils(self, i, t, n, times):
        # step 0+1: get WSS either from NN surrogate or from mesh+fluid solvers
        if self.no is not None:
            self._neural_operator_step(times, i, t, n)
        elif self.p["fsi"] and i > 1:
            if self.step("mesh", i, t, n, times):
                return False
        else:
            times["mesh"] = 0.0

        # store previous solutions
        self.prev = self.curr.copy()

        # step 1: fluid update (skipped when neural operator is active)
        if self.no is None:
            if self.p["fsi"]:
                if self.step("fluid", i, t, n, times):
                    return False
            else:
                self.poiseuille(t)

        # step 2: solid update
        if self.step("solid", i, t, n, times):
            return False

        # log interface solution
        dtk = deepcopy(self.curr.get(("solid", "disp", "int"))).flatten()
        dk = deepcopy(self.prev.get(("solid", "disp", "int"))).flatten()

        # store increments
        # todo: save memory by only storing necessary information
        self.dk["disp"] += [dtk]
        self.res += [dtk - dk]

        # append difference vectors after preloading (must not span different time levels)
        if t > 0 and n > 0:
            self.mat_W += [self.dk["disp"][-1] - self.dk["disp"][-2]]
            self.mat_V += [self.res[-1] - self.res[-2]]

        # get error
        self.coup_err("solid", "disp", i, t, n)

        # relax solid update
        self.coup_omega("disp", i, t, n)
        if not self.coup_converged(n):
            # no IQN-ILS update during preloading or first time step
            if ((t == 0) or (t == 1 and n < 5)):# or n == 0:
                self.coup_relax("solid", "disp", i, t, n)
            else:
                # reset history vectors at start of each new load step if requested
                if n == 0 and self.p["coup"].get("iqn_ils_reset", False):
                    self.mat_V = []
                    self.mat_W = []
                    # also clear dk and res so difference vectors don't span load steps
                    self.dk["disp"] = [self.dk["disp"][-1]]
                    self.res = [self.res[-1]]

                # fall back to relaxation if no history vectors available yet
                if not self.mat_V:
                    self.coup_relax("solid", "disp", i, t, n)
                    return

                # maximum number of time steps used in IQN-ILS
                nq = self.p["coup"]["iqn_ils_q"]

                # trim to max number of considered vectors
                self.mat_V = self.mat_V[-nq:]
                self.mat_W = self.mat_W[-nq:]

                # remove linearly dependent vectors
                tmp_V = np.array(self.mat_V).T
                tmp_W = np.array(self.mat_W).T
                eps = self.p["coup"]["iqn_ils_eps"]

                # debug: save V and W before filtering
                if self.p["coup"].get("iqn_ils_debug", False):
                    self.debug_qr["V_before"] += [tmp_V.copy()]
                    self.debug_qr["W_before"] += [tmp_W.copy()]
                    self.debug_qr["ncols_before"] += [tmp_V.shape[1]]

                qq, rr, tmp_V, tmp_W = QRfiltering_mod(tmp_V, tmp_W, eps)

                # debug: save Q, R and V, W after filtering
                if self.p["coup"].get("iqn_ils_debug", False):
                    self.debug_qr["Q"] += [qq.copy()]
                    self.debug_qr["R"] += [rr.copy()]
                    self.debug_qr["V_after"] += [tmp_V.copy()]
                    self.debug_qr["W_after"] += [tmp_W.copy()]
                    self.debug_qr["ncols_after"] += [tmp_V.shape[1]]

                # solve for coefficients
                ss = np.dot(np.transpose(qq), -self.res[-1])
                cc = np.linalg.solve(rr, ss)

                # debug: save coefficients and load step/sub-iter indices
                if self.p["coup"].get("iqn_ils_debug", False):
                    self.debug_qr["cc"] += [cc.copy()]
                    self.debug_qr["t"] += [t]
                    self.debug_qr["n"] += [n]

                # update
                vec_new = dtk + np.dot(tmp_W, cc)
                self.curr.add(("solid", "disp", "int"), vec_new.reshape((-1, 3)))

                # store matrices
                self.mat_V = tmp_V.T.tolist()
                self.mat_W = tmp_W.T.tolist()
        else:
            return True

    def coup_step_relax(self, i, t, n, times):
        # step 0+1: get WSS either from NN surrogate or from mesh+fluid solvers
        if self.no is not None:
            self._neural_operator_step(times, i, t, n)
        elif self.p["fsi"] and i > 1:
            if self.step("mesh", i, t, n, times):
                return False

        # store previous solutions
        self.prev = self.curr.copy()

        # step 1: fluid update (skipped when neural operator is active)
        if self.no is None:
            if self.p["fsi"]:
                if self.step("fluid", i, t, n, times):
                    return False
            else:
                self.poiseuille(t)

        # step 2: solid update
        if self.step("solid", i, t, n, times):
            return False

        # log interface solution for aitken relaxation
        dtk = deepcopy(self.curr.get(("solid", "disp", "int"))).flatten()
        dk = deepcopy(self.prev.get(("solid", "disp", "int"))).flatten()
        self.dk["disp"] += [dtk]
        self.res += [dtk - dk]

        # calculate new relaxation factor
        self.coup_omega("disp", i, t, n)

        # get error
        self.coup_err("solid", "disp", i, t, n)

        # relax solid update
        if not self.coup_converged(n):
            self.coup_relax("solid", "disp", i, t, n)
        else:
            return True

    def coup_predict(self, i, t):
        # predict displacements
        kind = ("solid", "disp", "vol")

        if t == 0 or not self.p["predict_file"]:
            # extrapolate from previous time step(s)
            sol = self.predictor(kind, t)
        else:
            # predict from file
            sol = self.predictor_tube(kind, t)
        self.curr.add(kind, sol)

    def predictor(self, kind, t):
        # fluid, solid, tube
        # disp, velo, wss, press
        # vol, int
        d, f, p = kind

        # number of old solutions
        n_sol = len(self.converged)

        if n_sol == 0:
            if f == "disp":
                # zero displacements
                return np.zeros(self.points[(p, d)].shape)
            elif f == "wss":
                # wss from poiseuille flow through reference configuration
                self.poiseuille(t)
                return self.curr.get(kind)
            else:
                raise ValueError("No predictor for field " + f)

        # previous solution
        vec_m0 = self.converged[-1].get(kind)
        if n_sol == 1:
            return vec_m0

        # extrapolate from previous load increment, damped by predictor_relax.
        # alpha=1 -> standard linear extrapolation (2*m0 - m1); alpha=0 -> constant
        # (m0, no overshoot). Aggressive extrapolation overshoots into element
        # inversion (gr_equilibrated "Negative Jacobian") for the nonlinear,
        # NN-driven aneurysm growth at higher load steps, so allow damping.
        vec_m1 = self.converged[-2].get(kind)
        alpha = self.p["coup"].get("predictor_relax", 1.0)
        return vec_m0 + alpha * (vec_m0 - vec_m1)

        # quadratically extrapolate from previous two load increments
        # vec_m2 = self.converged[-3].get(kind)
        # return 3.0 * vec_m0 - 3.0 * vec_m1 + vec_m2

    def predictor_tube(self, kind, t):
        d, f, p = kind
        fname = "gr_partitioned/tube_" + str(t).zfill(3) + ".vtu"
        # fname = 'gr/gr_' + str(t + 1).zfill(3) + '.vtu'
        if not os.path.exists(fname):
            return None
        geo = read_geo(fname).GetOutput()
        if f == "disp":
            return v2n(geo.GetPointData().GetArray("Displacement"))[
                self.map(((p, d), ("vol", "tube")))
            ]
        elif f == "wss":
            if geo.GetPointData().HasArray("WSS"):
                return v2n(geo.GetPointData().GetArray("WSS"))[
                    self.map(((p, d), ("vol", "tube")))
                ]
            else:
                disp = v2n(geo.GetPointData().GetArray("Displacement"))[
                    self.map(((p, d), ("vol", "solid")))
                ]
                self.curr.add((d, "disp", p), disp)
                self.poiseuille(t)
                return self.curr.get(kind)

    def coup_relax(self, domain, name, i, t, n):
        # volume increment
        curr_v = deepcopy(self.curr.get((domain, name, "vol")))
        prev_v = deepcopy(self.prev.get((domain, name, "vol")))

        # relax update
        if i == 1:
            vec_relax = curr_v
        else:
            omega = self.p["coup"]["omega"][name][-1][-1]
            vec_relax = omega * curr_v + (1.0 - omega) * prev_v

        # update solution
        self.curr.add((domain, name, "vol"), vec_relax)

        # log interface solution for aitken relaxation
        dk = deepcopy(self.curr.get((domain, name, "int"))).flatten()
        self.dtk[name] += [dk]

    def coup_err(self, domain, name, i, t, n):
        if i == 1:
            # first step: no old solution
            err = 1.0
        else:
            # inf-norm on residual displacement L2-norm
            err = np.mean(np.linalg.norm(self.res[-1].reshape((-1, 3)), axis=1))

        # start a new sub-list for new load step
        if n == 0:
            self.err[name].append([])

        # append error norm
        self.err[name][-1].append(err)

    def coup_converged(self, n):
        # check if coupling converged
        check_tol = np.all(
            np.array([e[-1][-1] for e in self.err.values()]) < self.p["coup"]["tol"]
        )
        check_n = n >= self.p["coup"]["nmin"]
        return check_tol and check_n

    def coup_omega(self, name, i, t, n):
        # no relaxation necessary during prestressing (prestress does not depend on wss)
        if t == 0:
            omega = 1.0
        else:
            # static relaxation or first step of new load step
            omega = self.p["coup"]["omega0"]

            # dynamic relaxation
            if self.p["coup"]["method"] == "aitken" and n > 0:
                kuettler = True
                if kuettler:
                    rki = self.res[-1]
                    rkm = self.res[-2]
                    diff = rki - rkm
                    omega = (
                        -self.p["coup"]["omega"][name][-1][-1]
                        * np.dot(rkm, diff)
                        / np.dot(diff, diff)
                    )
                else:
                    # get old relaxed solutions
                    dp = self.dk[name][-1]
                    dk = self.dk[name][-2]

                    # get old unrelaxed solutions
                    dtk = self.dtk[name][-1]
                    dtm = self.dtk[name][-2]

                    # aitken update
                    diff = dtk - dp - dtm + dk
                    omega = np.dot(dtk - dtm, diff) / np.dot(diff, diff)

                # lower bound
                omega = np.max([omega, 0.1])

                # upper bound
                omega = np.min([omega, 1.0])

        # start a new sub-list for new load step
        if n == 0:
            self.p["coup"]["omega"][name].append([])

        # append
        self.p["coup"]["omega"][name][-1].append(omega)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run an equilibrated Fluid-Solid-Growth interaction simulation (FSGe)"
    )
    parser.add_argument("sim", help="simulation parameters (.json)")
    parser.add_argument("-post", action="store_true", help="post-process only")
    parser.add_argument("-restart", default=None, help="path to restart.npz checkpoint file")
    args = parser.parse_args()

    fsg = FSG(args.sim)
    if args.post:
        fsg.run_post()
    else:
        fsg.run(f_restart=args.restart)
