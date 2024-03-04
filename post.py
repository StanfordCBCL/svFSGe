#!/usr/bin/env python
# coding=utf-8

import pdb
import argparse
import os
import glob
import json
import json
import xmltodict
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import StrMethodFormatter
from collections import defaultdict, OrderedDict
from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

from svfsi import svFSI
from vtk_functions import read_geo, threshold

# use LaTeX in text
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Computer Modern Roman",
        "font.size": 21,
    }
)
# plt.style.use("fivethirtyeight")

# field descriptions
f_labels = {
    "disp": [
        "Cir. displacement $\Delta\\theta$ [°]",
        "Rad. displacement $\Delta r$ [mm]",
        "Axi. displacement $\Delta z$ [mm]",
    ],
    "thick": ["Thickness [$\mu$m]"],
    "stim": [
        "WSS stimulus $\Delta\\tau_w$ [-]",
        "Intramular stimulus $\Delta\sigma_I$ [-]",
        "Stimulus ratio $K_{\\tau\sigma}$ [-]",
    ],
    "jac": ["Jacobian [-]"],
    "pk2": [
        "Cir. 2PK Stress [kPa]",
        "Rad. 2PK Stress [kPa]",
        "Axi. 2PK Stress [kPa]",
    ],
    "lagrange": ["Lagrange multiplier $p$ [kPa]"],
    "phic": ["Collagen mass fraction $\phi^c_h$ [-]"],
}
f_comp = {key: len(value) for key, value in f_labels.items()}
f_scales = {"disp": np.array([180.0 / np.pi, 1.0, 1.0]), "thick": [1e3]}
titles = {"gr": "G\&R", "partitioned": "FSGe"}


def read_xml_file(file_path):
    with open(file_path) as fd:
        return xmltodict.parse(fd.read())["svFSIFile"]


def read_json_file(file_path):
    if not file_path:
        return {}
    with open(file_path, "r") as file:
        return json.load(file)


def cra2xyz(cra):
    return np.array([cra[1] * np.sin(cra[0]), cra[1] * np.cos(cra[0]), cra[2]])


def xyz2cra(xyz):
    return np.array(
        [
            (np.arctan2(xyz[0], xyz[1]) + 2.0 * np.pi) % (2.0 * np.pi),
            np.sqrt(xyz[0] ** 2.0 + xyz[1] ** 2.0),
            xyz[2],
        ]
    )


def ten_xyz2cra(xyz, ten_xyz):
    # vtk stores symmetric 3x3 tensors in this form: XX, YY, ZZ, XY, YZ, XZ
    indices = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]

    # cir coordinate
    cir = xyz2cra(xyz.T)[0]

    # unit vectors in cir, rad, axi directions
    unit = [
        np.array([np.cos(cir), -np.sin(cir), np.zeros(len(xyz))]).T,
        np.array([np.sin(cir), np.cos(cir), np.zeros(len(xyz))]).T,
        np.array([[0, 0, 1]] * len(xyz)),
    ]

    # transform tensor to cir, rad, axi
    ten_cra = np.zeros(xyz.shape)
    mat_c = np.zeros((3, 3))
    for i in range(len(xyz)):
        for j, (row, col) in enumerate(indices):
            mat_c[row, col] = mat_c[col, row] = ten_xyz[i][j]
        for j in range(3):
            ten_cra[i, j] = np.dot(unit[j][i], np.dot(mat_c, unit[j][i]))
    return ten_cra


def read_config(json):
    # read simulation config
    if not os.path.exists(json):
        raise RuntimeError("No json file found: " + json)
    return svFSI(f_params=json, load=True)


def read_res(fname, fsge):
    # read all simulation results at all time steps
    res = []
    for fn in sorted(glob.glob(fname)):
        # read vtu from file
        geo = read_geo(fn).GetOutput()

        # extract solid domain
        if fsge:
            solid = threshold(geo, 1.0, "ids_solid").GetOutput()
        else:
            solid = geo
        res += [solid]
    return res


def get_ids(pts_xyz):
    # coordinates of all points (in reference configuration)
    pts_cra = xyz2cra(pts_xyz.T).T

    # cylinder dimensions
    ro = np.max(pts_cra[:, 1])
    ri = np.min(pts_cra[:, 1])
    h = np.max(pts_cra[:, 2])

    # circumferential coordinates of points to export: 0, 3, 6, 9 o'clock
    p_cir = {0: 0.0, 3: 0.5 * np.pi, 6: np.pi, 9: 1.5 * np.pi}

    # radial coordinates of points to export: inside, outside
    p_rad = {"out": ro, "in": ri}

    # axial coordinates of points to export: inlet, mid-point, outlet
    p_axi = {"start": 0.0, "mid": h / 2, "end": h}

    # collect all point coordinates
    locations = {}
    for cn, cp in p_cir.items():
        for rn, rp in p_rad.items():
            for an, ap in p_axi.items():
                identifiers = [
                    (cn, rn, an),
                    (":", rn, an),
                    (cn, ":", an),
                    (cn, rn, ":"),
                ]
                for i in identifiers:
                    locations[i] = [cp, rp, ap]

    # collect all point ids
    ids = {}
    coords = {}
    for loc, pt in locations.items():
        chk = [np.isclose(pts_cra[:, i], pt[i]) for i in range(3) if loc[i] != ":"]
        ids[loc] = np.where(np.logical_and.reduce(np.array(chk)))[0]
        if len(ids[loc]) == 0:
            print("no points found: " + str(loc))
            continue

        # sort according to coordinate
        if ":" in loc:
            dim = list(loc).index(":")
            crd = pts_cra[ids[loc], dim]
            sort = np.argsort(crd)
            ids[loc] = ids[loc][sort]
            coords[loc] = crd[sort]
            assert len(np.unique(crd)) == len(crd), "coordinates not unique: " + str(
                crd
            )

            # # limit angle to [-pi, pi)
            # if dim == 0:
            #     coords[loc] += np.pi
            #     coords[loc] %= np.pi * 2.0
            #     coords[loc] -= np.pi

    return ids, coords


def get_results(results, pts, ids):
    # get post-processed quantities at all extracted locations
    post = {}
    for loc in ids.keys():
        post[loc] = defaultdict(list)

    # get results at all time steps
    for res in results:
        extract_results(post, res, pts, ids)

    # convert to numpy arrays
    for loc in post.keys():
        for f in post[loc].keys():
            post[loc][f] = np.squeeze(np.array(post[loc][f]))

    return post


def extract_results(post, res, pts, ids):
    # get nodal displacements
    d = v2n(res.GetPointData().GetArray("Displacement"))

    # get G&R output
    if res.GetPointData().HasArray("GR"):
        gr = v2n(res.GetPointData().GetArray("GR"))
    else:
        gr = np.zeros((pts.shape[0], 50))

    # jacobian
    jac = v2n(res.GetPointData().GetArray("Jacobian"))

    # 2PK stress
    pk2_xyz = v2n(res.GetPointData().GetArray("Stress"))
    pk2_cra = ten_xyz2cra(pts, pk2_xyz)

    # get stimuli
    stim = gr[:, 31:34]

    for loc, pt in ids.items():
        # displacement in polar coordinates
        diff = xyz2cra((pts[pt] + d[pt]).T) - xyz2cra(pts[pt].T)

        # limit angle to [-pi, pi)
        diff[0] += np.pi
        diff[0] %= np.pi * 2.0
        diff[0] -= np.pi

        # store values
        post[loc]["disp"] += [diff]
        post[loc]["jac"] += [jac[pt]]
        post[loc]["pk2"] += [pk2_cra[pt].T]
        post[loc]["lagrange"] += [gr[pt, 30]]
        post[loc]["phic"] += [gr[pt, 37]]

        # extract stimuli
        post[loc]["stim"] += [stim[pt].T]

        # extract thickness and wss (only on interface)
        if loc[1] == "in":
            dloc = (loc[0], "out", loc[2])

            d1 = pts[ids[loc]] + d[ids[loc]]
            d2 = pts[ids[dloc]] + d[ids[dloc]]

            post[loc]["thick"] += [np.linalg.norm((d1 - d2).T, axis=0)]


def extract_scalar(scalar, res, pts, ids, mode):
    d = v2n(res.GetPointData().GetArray("Displacement"))
    for m in ids.keys():
        if "out" in m:
            m_in = m.replace("out", "in")
            m_thick = m.replace("out", "thick")
            d_out = pts[ids[m]] + d[ids[m]]
            d_in = pts[ids[m_in]] + d[ids[m_in]]
            scalar["p_" + m_thick] += [np.linalg.norm(d_out - d_in)]


def post_process(f_out):
    # check if FSGe or conventional G&R results
    if "gr" in f_out:
        fsge = False
        fname = os.path.join(f_out, "gr_*.vtu")
    else:
        fsge = True
        fname = os.path.join(f_out, "tube_*.vtu")

    # read results from file
    res = read_res(fname, fsge)
    if not len(res):
        raise RuntimeError("No results found in " + f_out)

    # double up results if there's only one time step
    if len(res) == 1:
        res *= 2

    # extract points
    pts = v2n(res[0].GetPoints().GetData())

    # get point and line ids
    ids, coords = get_ids(pts)

    # extract displacements
    return get_results(res, pts, ids), coords, len(res)


def plot_res(data, coords, times, param, out, study):
    # cir locations: o' clocks
    loc_cir = range(0, 12, 3)

    # rad locations: ["in", "out"]
    loc_rad = ["in", "out"]

    # axi locations: ["start", "mid", "end"]
    loc_axi = ["mid"]

    # loop time steps
    t_max = min(times.values())
    # for t in reversed(range(t_max)):
    for t in [-1]:
        # loop fields and plot
        fields = ["disp", "thick", "stim", "jac", "pk2", "lagrange", "phic"]
        for f in sorted(fields):
            # plot single points
            for lr in loc_rad:
                for la in loc_axi:
                    plot_points = [(lc, lr, la) for lc in loc_cir]
                    for dim in range(f_comp[f]):
                        plot_single(
                            data, coords, param, out, study, f, plot_points, t, dim
                        )

            if study == "single":
                # plot circumferential ring
                for lr in loc_rad:
                    for la in loc_axi:
                        plot_cir = [(":", lr, la)]
                        for dim in range(f_comp[f]):
                            plot_single(
                                data, coords, param, out, study, f, plot_cir, t, dim
                            )

                # plot along radius
                for la in loc_axi:
                    plot_rad = [(lc, ":", la) for lc in loc_cir]
                    for dim in range(f_comp[f]):
                        plot_single(
                            data, coords, param, out, study, f, plot_rad, t, dim
                        )

                # plot along axial lines
                for lr in loc_rad:
                    plot_axi = [(lc, lr, ":") for lc in loc_cir]
                    for dim in range(f_comp[f]):
                        plot_single(
                            data, coords, param, out, study, f, plot_axi, t, dim
                        )


def plot_single(data, coords, param, out, study, quant, locations, time=-1, comp=0):
    # determine plot dimensions
    ny = len(np.unique([k.split("_")[0] for k in data.keys()]))
    nx = len(data) // ny

    fig, ax = plt.subplots(
        ny, nx, figsize=(nx * 10, ny * 4), dpi=300, sharex=True, sharey=True
    )
    if nx == 1 and ny == 1:
        ax = [ax]
    for i_data, (n, res) in enumerate(data.items()):
        title = titles[n.split("_")[0]]
        title += ", $K_{\\tau\sigma} = " + param[n]["KsKi"] + "$"

        if ny > 1:
            pos = np.unravel_index(i_data, (ny, nx))
        else:
            pos = (i_data,)

        # loop mesh positions
        for lc in locations:
            # skip if no data available
            if quant not in res[lc]:
                return

            # get data for y-axis
            if f_comp[quant] > 1:
                ydata = res[lc][quant][:, comp].copy()
            else:
                ydata = res[lc][quant].copy()
            time_str = ""
            if ":" in lc:
                if time == -1:
                    time = len(ydata) - 1
                time_str = "_t" + str(time)
                if time > len(ydata) - 1:
                    continue
                ydata = ydata[time]
            if quant in f_scales:
                ydata *= f_scales[quant][comp]

            # get data for x-axis
            fname = quant
            loc = list(lc)
            if study == "single":
                if ":" in lc:
                    # plotting along a coordinate axis
                    xdata = coords[n][lc].copy()
                    dim = loc.index(":")
                    loc.remove(":")
                    if dim == 0:
                        xlabel = "Vessel circumference [°]"
                        xdata *= 180 / np.pi
                        xdata = np.append(xdata, 360)
                        ydata = np.append(ydata, [ydata[0]], axis=0)
                        dphi = 45
                        xticks = np.arange(0, 360 + dphi, dphi).astype(int)
                    elif dim == 1:
                        xlabel = "Vessel radius [mm]"
                        xticks = xdata
                    elif dim == 2:
                        xlabel = "Vessel length [mm]"
                        xticks = [0, 2, 4, 6, 7.5, 9, 11, 13, 15]
                    dim_names = ["cir", "rad", "axi"]
                    fname += "_" + dim_names[dim]
                else:
                    # plotting a single point over all load steps
                    nd = len(ydata)
                    n10 = int(np.log(nd) / np.log(10))
                    xdata = np.arange(0, nd)
                    xlabel = "Load step [-]"
                    if nd <= 10:
                        xticks = np.arange(0, nd, 1)
                    else:
                        xticks = np.arange(0, nd, nd // 10**n10 * 10 ** (n10 - 1))
                    fname += "_load"
            elif study == "kski":
                # get data for x-axis
                xdata = np.linspace(0.0, 1.0, 5)
                xlabel = "$K_{\\tau\sigma}$"
                xticks = [0, 0.25, 0.5, 0.75, 1]
            else:
                raise ValueError("unknown study: " + study)

            # assemble filename
            loc = np.array(loc).astype(str).tolist()
            if len(locations) == 1:
                fname += "_".join([""] + loc)
                stl = "-"
                col = "k"
            else:
                # assume all locations provided are circumferential
                fname += "_".join([""] + loc[1:])
                colors = {
                    l: plt.cm.tab10(k) for k, l in zip([0, 1, 3, 2], range(0, 12, 3))
                }
                if quant == "disp" and comp == 0:
                    styles = {0: "-", 3: "-", 6: ":", 9: "-"}
                else:
                    styles = {0: "-", 3: "-", 6: "-", 9: ":"}
                stl = styles[lc[0]]
                col = colors[lc[0]]

            # plot!
            try:
                ax[pos].plot(xdata, ydata, stl, color=col, linewidth=2)
            except:
                pdb.set_trace()

        # plot lines
        ax[pos].grid(True)
        ax[pos].set_xticks(xticks)
        ax[pos].set_xticklabels([str(x) for x in xticks])
        ax[pos].set_xlim([np.min(xdata), np.max(xdata)])
        ax[pos].set_title(title)
        if ny == 0 or pos[0] == ny - 1:
            ax[pos].set_xlabel(xlabel)
        if pos[-1] == 0:
            ax[pos].set_ylabel(f_labels[quant][comp])
    plt.tight_layout()
    if f_comp[quant] > 1:
        fname += "_d" + str(comp)
    fname += time_str + ".png"
    fig.savefig(os.path.join(out, fname), bbox_inches="tight")
    plt.cla()
    print(fname)


def main():
    # set study
    # for kski in np.linspace(0.0, 1.0, 5).astype(str):
    # print("\n\nplotting kski " + str(kski) + "\n")

    # folder = "/Users/pfaller/work/repos/FSGe/study_aneurysm"
    # geo = "coarse"
    # phi = "0.7"

    # define paths
    # fpath = os.path.join(folder, geo,  "phi_" + phi, "kski_" + kski)
    fpath = "/Users/pfaller/work/repos/FSG/new_study_aneurysm/coarse_kski_1.0"
    inp = {
        "G&R": os.path.join(fpath, "gr"),
        "FSGe": os.path.join(fpath, "partitioned", "partitioned.json"),
    }
    out = os.path.join(fpath, "comparison")

    # create output folder
    os.makedirs(out, exist_ok=True)

    # collect all results
    data = {}
    coords = {}
    times = {}
    for n, o in inp.items():
        data[n], coords[n], times[n] = post_process(o)

    plot_res(data, coords, times, out, "single")


def main_param():
    print("\n\nplotting all kski\n")
    # set study
    folder = "/Users/pfaller/work/repos/FSGe/study_aneurysm"
    geo = "coarse"
    phi = "0.7"

    kski = np.linspace(0.0, 1.0, 5)

    # define paths
    out = os.path.join(folder, geo, "phi_" + phi, "comparison")
    os.makedirs(out, exist_ok=True)

    # plot elastin insult (axi and cir)
    plot_insult(out)

    data = {}
    for k in kski:
        fpath = os.path.join(folder, geo, "phi_" + phi, "kski_" + str(k))
        inp = {
            "G&R": os.path.join(fpath, "gr"),
            "FSGe": os.path.join(fpath, "partitioned", "partitioned.json"),
        }
        data[k] = {}
        for n, o in inp.items():
            data[k][n], coords = post_process(o)

    # collect data for all parameter variations
    data_sorted = {}
    params = sorted(data.keys())
    sims = data[params[0]].keys()
    for nam in sims:
        data_sorted[nam] = {}
        for loc in data[params[0]][nam].keys():
            data_sorted[nam][loc] = {}
            for quant in data[params[0]][nam][loc].keys():
                data_sorted[nam][loc][quant] = []

                # append all parametric evaluations
                for k in params:
                    # extract last time step
                    data_sorted[nam][loc][quant] += [data[k][nam][loc][quant][-1]]
                data_sorted[nam][loc][quant] = np.array(data_sorted[nam][loc][quant])

    plot_res(data_sorted, coords, out, "kski")


def collect_simulations(folder):
    # define paths
    if len(folder) == 1:
        folder += [os.path.join(folder[0], "post")]
    folders = folder[:-1]
    out = folder[-1]
    os.makedirs(out, exist_ok=True)

    # post-process simulation (converged and unconverged)
    inp = {}
    p_xml = {}
    p_json = {}
    for f in folders:
        fname = os.path.split(f)[-1]
        if "gr" in f:
            dir_name = f
            f_p_json = ""
            f_p_xml = os.path.join(f, "gr_full.xml")
        elif "partitioned" in f:
            dir_name = os.path.join(f, "partitioned", "converged")
            f_p_json = os.path.join(f, "partitioned.json")
            f_p_xml = os.path.join(f, "in_svfsi", "gr_full_restart.xml")
        else:
            raise ValueError("unknown input folder: " + f)
        inp[fname] = dir_name
        p_xml[fname] = read_xml_file(f_p_xml)
        p_json[fname] = read_json_file(f_p_json)

    # extract only relevant parameters
    param = {}
    for f in inp.keys():
        param[f] = {}
        param[f]["KsKi"] = p_xml[f]["Add_equation"]["Constitutive_model"]["KsKi"]
        if p_json[f]:
            param[f]["error"] = p_json[f]["error"]["disp"]

    return out, inp, param


def main_arg(folder):
    # collect simulations
    out, inp, param = collect_simulations(folder)

    # collect all results
    data = OrderedDict()
    coords = {}
    times = {}
    for n, o in inp.items():
        data[n], coords[n], times[n] = post_process(o)

    plot_res(data, coords, times, param, out, "single")


def main_convergence(folder):
    # collect simulations
    out, inp, data = collect_simulations(folder)

    ydata = []
    labels = []
    for f in inp.keys():
        kski = data[f]["KsKi"]
        labels += ["$K_{\\tau\sigma}$ = " + kski]
        n_it = []
        for err in data[f]["error"]:
            n_it += [len(err)]
        print(kski, "{:.1f}".format(np.mean(n_it[2:])))
        ydata += [np.cumsum(n_it)]

    ydata = np.array(ydata).T
    xdata = np.arange(0, ydata.shape[0])

    fig, ax = plt.subplots(figsize=(20, 8), dpi=300)
    ax.plot(xdata, ydata, linewidth=2)
    ax.grid(True)
    ax.set_xticks(xdata)
    ax.set_xlim([np.min(xdata), np.max(xdata)])
    ax.set_ylim([0, np.max(ydata)])
    ax.set_xlabel("Load step [-]")
    ax.set_ylabel("Cumulative number of coupling iterations [-]")

    ax2 = ax.twinx()
    ax2.set_ylim([ax.get_ylim()[0], ax.get_ylim()[1]])
    ax2.set_yticks(ydata[-1])
    ax2.set_yticklabels(labels)

    plt.tight_layout()
    fig.savefig(os.path.join(out, "convergence.png"), bbox_inches="tight")
    plt.cla()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process FSGe simulation")
    parser.add_argument("out", nargs="+", default="None", help="svFSI output folder")
    parser.add_argument("-c", action="store_true", help="Plot convergence")
    args = parser.parse_args()

    if not args.out:
        # main_param()
        main()
    else:
        if args.c:
            main_convergence(args.out)
        else:
            main_arg(args.out)
