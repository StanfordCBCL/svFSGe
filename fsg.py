#!/usr/bin/env python
# coding=utf-8

import pdb
import numpy as np
import shutil
import os
import glob
import time
from copy import deepcopy
import argparse

import matplotlib.pyplot as plt

from vtk.util.numpy_support import vtk_to_numpy as v2n

from svfsi import svFSI, sv_names


class FSG(svFSI):
    """
    FSG-specific stuff
    """

    def __init__(self, f_params=None):
        # svFSI simulations
        svFSI.__init__(self, f_params)

    def run_post(self):
        # todo: read in automatically
        self.err = np.load(
            "study_lab_meeting/fsi_res_2022-11-30_18-21-39.375658/err.npy",
            allow_pickle=True,
        ).item()
        self.p["f_out"] = "."
        self.plot_convergence()

    def run(self):
        # run simulation
        try:
            self.main()
        except KeyboardInterrupt:
            print("interrupted")
            pass

        # archive results
        self.archive()

        # plot convergence
        self.plot_convergence()

    def main(self):
        # print reynolds number
        print("Re = " + str(int(self.p["re"])))

        # loop load steps
        i = 0
        for t in range(self.p["nmax"] + 1):
            print(
                "=" * 20
                + " t "
                + str(t)
                + " ==== fp "
                + "{:.2f}".format(self.p_vec[t])
                + " "
                + "=" * 20
            )

            # predict solution for next load step
            if t > 0:
                self.coup_predict(i, t)

            # loop sub-iterations
            for n in range(self.p["coup"]["nmax"]):
                # count total iterations (load + sub-iterations)
                i += 1

                # perform coupling step
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
                        return

                # screen output
                out = "i " + str(i) + " \tn " + str(n) + "\t"
                for name, e in self.err.items():
                    out += "{:.2e}".format(e[-1][-1]) + "\t"
                if self.p["coup"]["method"] in ["static", "aitken"]:
                    for name, e in self.p["coup"]["omega"].items():
                        out += "{:.2e}".format(e[-1][-1]) + "\t"
                for f in times.keys():
                    out += "{:.2e}".format(times[f]) + "\t"

                # check if coupling unconverged (screen and file output)
                if n == self.p["coup"]["nmax"] - 1:
                    out += "\n\tcoupling unconverged"
                    status = True
                print(out)

                # archive solution
                dst = os.path.join(self.p["f_sim"], "tube_" + str(i).zfill(3) + ".vtu")
                self.curr.archive("tube", dst)

                # check if coupling converged
                if status:
                    # save converged steps
                    i_conv = str(i).zfill(3)
                    t_conv = str(t).zfill(3)

                    srcs = os.path.join(self.p["f_sim"], "*_" + i_conv + ".*")
                    for src in glob.glob(srcs):
                        trg = os.path.basename(src).replace(i_conv, t_conv)
                        trg = os.path.join(self.p["f_conv"], trg)
                        shutil.copyfile(src, trg)

                    # archive
                    self.converged += [self.curr.copy()]

                    # terminate coupling
                    break

    def plot_convergence(self):
        n_sol = len(self.err.keys())
        col_err = "k"
        col_omg = "r"

        n_iter = [0]
        fig, ax = plt.subplots(
            n_sol, 1, figsize=(20, 8), dpi=200, sharex="all", sharey="all"
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
        plt.show()
        plt.close(fig)

    def archive(self):
        # save stored results
        np.save(os.path.join(self.p["f_out"], "err.npy"), self.err)

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
        src = f_code + "/../../../Code/Source/svFSI/FEMbeCmm.cpp"
        trg = os.path.join(self.p["f_arx"], "FEMbeCmm.cpp")
        shutil.copyfile(src, trg)

    def coup_step_iqn_ils(self, i, t, n, times):
        # step 0: mesh movement (not in first first iteration)
        if self.p["fsi"] and i > 1:
            if self.step("mesh", i, t, times):
                return False
        else:
            times["mesh"] = 0.0

        # store previous solutions
        self.prev = self.curr.copy()

        # step 1: fluid update
        if self.p["fsi"]:
            if self.step("fluid", i, t, times):
                return False
        else:
            self.poiseuille(t)

        # step 2: solid update
        if self.step("solid", i, t, times):
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
            # no IQN-ILS update during preloading or first iteration step
            if t == 0 or n == 0:
                self.coup_relax("solid", "disp", i, t, n)
            else:
                # maximum number of time steps used in IQN-ILS
                nq = self.p["coup"]["iqn_ils_q"]

                # trim to max number of considered vectors
                self.mat_V = self.mat_V[-nq:]
                self.mat_W = self.mat_W[-nq:]

                # remove linearly dependent vectors
                while True:
                    # QR decomposition
                    _, rr = np.linalg.qr(np.array(self.mat_V[:nq]).T)

                    # tolerance for redundant vectors
                    i_eps = np.where(
                        np.abs(np.diag(rr)) < self.p["coup"]["iqn_ils_eps"]
                    )[0]
                    if not np.any(i_eps):
                        break

                    print("filtering " + str(len(i_eps)) + " time steps")
                    for i in reversed(i_eps):
                        self.mat_V.pop(i)
                        self.mat_W.pop(i)

                # solve for coefficients
                bb = np.linalg.solve(rr.T, -np.dot(np.array(self.mat_V), self.res[-1]))
                cc = np.linalg.solve(rr, bb)

                # update
                vec_new = dtk + np.dot(np.array(self.mat_W).T, cc)
                self.curr.add(("solid", "disp", "int"), vec_new.reshape((-1, 3)))
        else:
            return True

    def coup_step_relax(self, i, t, n, times):
        # step 0: mesh movement (not in very first iteration)
        if self.p["fsi"] and i > 1:
            if self.step("mesh", i, t, times):
                return False

        # store previous solutions
        self.prev = self.curr.copy()

        # step 1: fluid update
        if self.p["fsi"]:
            if self.step("fluid", i, t, times):
                return False
        else:
            self.poiseuille(t)

        # step 2: solid update
        if self.step("solid", i, t, times):
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

        # linearly extrapolate from previous load increment
        vec_m1 = self.converged[-2].get(kind)
        if n_sol == 2:
            return 2.0 * vec_m0 - vec_m1

        # quadratically extrapolate from previous two load increments
        vec_m2 = self.converged[-3].get(kind)
        return 3.0 * vec_m0 - 3.0 * vec_m1 + vec_m2

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
            err = np.max(np.linalg.norm(self.res[-1].reshape((-1, 3)), axis=1))

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
    args = parser.parse_args()

    fsg = FSG(args.sim)
    if args.post:
        fsg.run_post()
    else:
        fsg.run()
