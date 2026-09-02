#!/usr/bin/env python
# coding=utf-8

import pdb
import argparse
import os
import glob
import json
import scipy
import xmltodict
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from matplotlib.ticker import StrMethodFormatter
from collections import defaultdict, OrderedDict
from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

from svfsi import svFSI
from vtk_functions import read_geo, threshold

# use LaTeX in text (disabled to avoid LaTeX dependency)
plt.rcParams.update(
    {
        "text.usetex": False,
        #"font.family": "serif",
        #"font.serif": "Computer Modern Roman",
        #"font.size": 21,
    }
)
# plt.style.use("fivethirtyeight")

# field descriptions
f_labels = {
    "disp": [
        r"Disp. $d_\theta$ [°]",
        "Disp. $d_r$ [mm]",
        "Disp. $d_z$ [mm]",
    ],
    "disp_r": ["Disp. $d_r$ [mm]"],
    "thick": [r"Thickness [$\mu$m]"],
    "stim": [
        r"Gain ratio $K_{\tau\sigma,h}$ [-]",
        r"Stim. $\Delta\sigma_I$ [-]",
        r"Stim. $\Delta\tau_w$ [-]",
    ],
    "jac": ["Jacobian [-]"],
    "pk2": [
        r"2PK $S_{\theta\theta}$ [kPa]",
        "2PK $S_{rr}$ [kPa]",
        "2PK $S_{zz}$ [kPa]",
    ],
    "lagrange": ["Lagrange $p$ [kPa]"],
    "phic": [r"Collagen $\phi^c_h$ [-]"],
    "phic_curr": [r"Collagen $\phi^c_h J_h$ [-]"],
    "pressure": ["Pressure [mmHg]"],
    "velocity": ["Velocity $u$ [mm/s]"],
}
s_labels = {r"KsKi": r"Gain ratio $K_{\tau\sigma,o}$ [-]"}
f_comp = {key: len(value) for key, value in f_labels.items()}
f_scales = {"disp": np.array([180.0 / np.pi, 1.0, 1.0]), "thick": [1e3], "pressure": [1.0/0.1333]}
titles = {"gr": r"G\&R", "partitioned": "FSGe"}
fields = {"fluid": ["pressure", "velocity"],
    "solid": ["disp_r", "disp", "thick", "stim", "jac", "pk2", "lagrange", "phic", "phic_curr"]}


def get_colormap(param):
    # continuous color map
    cstart = 0.3
    cmap = (param + cstart) / (np.max(param) + cstart)
    return plt.colormaps["Reds"](cmap)


def rec_dict():
    return defaultdict(rec_dict)


def read_xml_file(file_path):
    with open(file_path) as fd:
        return xmltodict.parse(fd.read())["svFSIFile"]


def read_json_file(file_path):
    if not file_path or not os.path.exists(file_path):
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


def read_res(fname, fsge, n_domain="solid"):
    # read all simulation results at all time steps
    res = []
    for fn in sorted(glob.glob(fname)):
        # read vtu from file
        geo = read_geo(fn).GetOutput()

        # extract solid domain
        if fsge:
            domain = threshold(geo, 1.0, "ids_" + n_domain).GetOutput()
        else:
            if n_domain != "solid":
                raise ValueError("Unknown G&R domain: " + n_domain)
            domain = geo
        res += [domain]
    return res


def get_ids(pts_xyz, domain="solid"):
    # coordinates of all points (in reference configuration)
    pts_cra = xyz2cra(pts_xyz.T).T

    # cylinder dimensions
    ro = np.max(pts_cra[:, 1])
    ri = np.min(pts_cra[:, 1])
    h = np.max(pts_cra[:, 2])

    if domain == "solid":
        # circumferential coordinates of points to export: 0, 3, 6, 9 o'clock
        p_cir = {0: 0.0, 3: 0.5 * np.pi, 6: np.pi, 9: 1.5 * np.pi}

        # radial coordinates of points to export: inside, outside
        p_rad = {"out": ro, "in": ri}

        # axial coordinates of points to export: inlet, mid-point, outlet
        p_axi = {"start": 0.0, "mid": h / 2, "end": h}
    elif domain == "fluid":
        # plot along centerline
        p_cir = {0: 0.0}
        p_rad = {"center": 0.0}
        p_axi = {"start": 0.0, "mid": h / 2, "end": h}
    else:
        raise ValueError("Unknown domain: " + domain)

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

    return ids, coords


def get_results(results, pts, ids, domain="solid"):
    # get post-processed quantities at all extracted locations
    post = {}
    for loc in ids.keys():
        post[loc] = defaultdict(list)

    # get results at all time steps
    for res in results:
        if domain == "solid":
            extract_results_solid(post, res, pts, ids)
        elif domain == "fluid":
            extract_results_fluid(post, res, pts, ids)

    # convert to numpy arrays
    for loc in post.keys():
        for f in post[loc].keys():
            post[loc][f] = np.squeeze(np.array(post[loc][f]))

    return post


def extract_results_fluid(post, res, pts, ids):
    pressure = v2n(res.GetPointData().GetArray("Pressure"))
    velocity = v2n(res.GetPointData().GetArray("Velocity"))
    # velocity = xyz2cra(velocity.T)
    velocity = np.linalg.norm(velocity, axis=1)

    for loc, pt in ids.items():
        post[loc]["pressure"] += [pressure[pt]]
        post[loc]["velocity"] += [velocity[pt]]


def extract_results_solid(post, res, pts, ids):
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
    stim = gr[:, 33:30:-1]

    for loc, pt in ids.items():
        # displacement in polar coordinates
        diff = xyz2cra((pts[pt] + d[pt]).T) - xyz2cra(pts[pt].T)

        # limit angle to [-pi, pi)
        diff[0] += np.pi
        diff[0] %= np.pi * 2.0
        diff[0] -= np.pi

        # store values
        post[loc]["disp"] += [diff]
        post[loc]["disp_r"] += [diff[1]]
        post[loc]["jac"] += [jac[pt]]
        post[loc]["pk2"] += [pk2_cra[pt].T]
        post[loc]["lagrange"] += [gr[pt, 30]]
        post[loc]["phic"] += [gr[pt, 37]]
        post[loc]["phic_curr"] += [gr[pt, 37] * jac[pt]]

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


def post_process(f_out, domain="solid"):
    # check if FSGe or conventional G&R results
    if "gr" in f_out:
        fsge = False
        fname = os.path.join(f_out, "gr_*.vtu")
    else:
        fsge = True
        fname = os.path.join(f_out, "tube_*.vtu")

    # read results from file
    res = read_res(fname, fsge, domain)
    if not len(res):
        raise RuntimeError("No results found in " + f_out)

    # double up results if there's only one time step
    if len(res) == 1:
        res *= 2

    # extract points
    pts = v2n(res[0].GetPoints().GetData())

    # get point and line ids
    ids, coords = get_ids(pts, domain)

    # extract displacements
    return get_results(res, pts, ids, domain), coords, len(res)


def plot_res(data, coords, times, param, out, domain, study, load_step=None):
    if domain == "solid":
        # cir locations: times on the clock
        loc_cir = range(0, 12, 3)

        # rad locations: ["in", "out"]
        loc_rad = ["in", "out"]

        # axi locations: ["start", "mid", "end"]
        loc_axi = ["mid"]
    elif domain == "fluid":
        loc_cir = [0]
        loc_rad = ["center"]

        # axi locations: ["start", "mid", "end"]
        loc_axi = ["start", "mid", "end"]
    else:
        raise ValueError("Unknown domain: " + domain)

    # loop time steps
    t_max = min(times.values())
    if load_step is None:
        load_step = [-1]
    # for t in reversed(range(t_max)):
    for t in load_step:
        # loop fields and plot
        for f in sorted(fields[domain]):
            # plot single points
            for lr in loc_rad:
                for la in loc_axi:
                    plot_points = [(lc, lr, la) for lc in loc_cir]
                    plot_single(data, coords, param, out, study, f, plot_points, t)

            if study == "single":
                # plot circumferential ring
                for lr in loc_rad:
                    for la in loc_axi:
                        plot_cir = [(":", lr, la)]
                        plot_single(data, coords, param, out, study, f, plot_cir, t)

                # plot along radius
                for la in loc_axi:
                    plot_rad = [(lc, ":", la) for lc in loc_cir]
                    plot_single(data, coords, param, out, study, f, plot_rad, t)

                # plot along axial lines
                for lr in loc_rad:
                    plot_axi = [(lc, lr, ":") for lc in loc_cir]
                    plot_single(data, coords, param, out, study, f, plot_axi, t)
            if study == "KsKi":
                # plot along axial lines
                plot_axi = [(lc, lr, ":") for lc in loc_cir]
                plot_single(data, coords, param, out, study, f, plot_axi, t)



def plot_single(data, coords, param, out, study, quant, locations, time=-1):
    # determine plot dimensions
    n_sim = len(np.unique([k.split("_")[0] for k in data.keys()]))
    if n_sim == 1:
        nx = len(data)
        ny = f_comp[quant]
    else:
        nx = n_sim
        ny = len(data) // nx * f_comp[quant]

    if f_comp[quant] > 1:
        h = 2.5
    else:
        h = 3
    if ny == 1:
        h = 3.5
    fs = (nx * 10, ny * h)
    fig, ax = plt.subplots(ny, nx, figsize=fs, dpi=300, sharex=True, sharey="row")
    if nx == 1 and ny == 1:
        ax = [ax]
    for i_data, (n, res) in enumerate(data.items()):
        for j_data in range(f_comp[quant]):
            title = next((v for k, v in titles.items() if k in n), n)
            if study == "single":
                title += r", $K_{\tau\sigma,o} = " + param[n]["KsKi"] + "$"

            if ny == 1 and nx == 1:
                pos = i_data
            elif ny == 1:
                pos = (i_data,)
            elif nx == 1:
                pos = (j_data,)
            else:
                pos = np.unravel_index(i_data * f_comp[quant] + j_data, (ny, nx), "F")

            # loop mesh positions
            for lc in locations:
                # skip if no data available
                if quant not in res[lc]:
                    return

                # get data for y-axis
                data_cp = np.array(res[lc][quant]).copy()
                if f_comp[quant] > 1:
                    ydata = data_cp[:, j_data]
                else:
                    ydata = data_cp
                time_str = ""
                if ":" in lc:
                    if study == "single":
                        if time == -1:
                            time = len(ydata) - 1
                        time_str = "_t" + str(time)
                        if time > len(ydata) - 1:
                            continue
                        ydata = ydata[time]
                    else:
                        ydata = ydata.T
                if quant in f_scales:
                    ydata *= f_scales[quant][j_data]
                if np.isscalar(ydata):
                    return

                # get data for x-axis
                fname = quant
                loc = list(lc)
                if ":" in lc:
                    # plotting along a coordinate axis
                    if n in coords:
                        xdata = coords[n][lc].copy()
                    else:
                        xdata = next(iter(coords.values()))[lc].copy()
                    dim = loc.index(":")
                    loc.remove(":")
                    if dim == 0:
                        xlabel = "Vessel circumference $\\varphi$ [°]"
                        xdata *= 180 / np.pi
                        xdata = np.append(xdata, 360)
                        ydata = np.append(ydata, [ydata[0]], axis=0)
                        dphi = 45
                        xticks = np.arange(0, 360 + dphi, dphi).astype(int)
                    elif dim == 1:
                        xlabel = "Vessel radius $r$ [mm]"
                        xticks = [xdata[0], xdata[-1]]
                    elif dim == 2:
                        xlabel = "Vessel axial $z$ [mm]"
                        xticks = [0, 2, 4, 6, 7.5, 9, 11, 13, 15]
                    dim_names = ["cir", "rad", "axi"]
                    fname += "_" + dim_names[dim]
                elif study == "single":
                    # plotting a single point over all load steps
                    nd = len(ydata)
                    n10 = int(np.log(nd) / np.log(10))
                    xdata = np.arange(0, nd)
                    xlabel = "Load step $t$ [-]"
                    if nd <= 10:
                        xticks = np.arange(0, nd, 1)
                    else:
                        xticks = np.arange(0, nd, nd // 10**n10 * 10 ** (n10 - 1))
                    fname += "_load"
                else:
                    if study not in s_labels:
                        raise ValueError("unknown study: " + study)
                    xlabel = s_labels[study]
                    xdata = param[n][study]
                    xticks = xdata
                    xref = [xdata[0], xdata[-1]]
                    yref = None
                    if "phic" in quant:
                        # add reference collagen mass fraction
                        yref = 0.33
                    if (quant == "disp" and j_data == 1) or quant == "disp_r":
                        yref = 0.0
                    if yref is not None and lc == locations[0]:
                        ax[pos].plot(xref, [yref] * 2, "k-", linewidth=2)

                # assemble filename
                loc = np.array(loc).astype(str).tolist()
                if len(locations) == 1:
                    fname += "_".join([""] + loc)
                    stl = "-"
                    col = "k"
                    if len(ydata.shape) == 2 and study == "KsKi":
                        col = get_colormap(param[n]["KsKi"])
                else:
                    # assume all locations provided are circumferential
                    fname += "_".join([""] + loc[1:])
                    colors = {
                        l: plt.cm.tab10(k)
                        for k, l in zip([0, 1, 3, 2], range(0, 12, 3))
                    }
                    if quant == "disp" and j_data == 0:
                        styles = {0: "-", 3: "-", 6: ":", 9: "-"}
                    else:
                        styles = {0: "-", 3: "-", 6: "-", 9: ":"}
                    stl = styles[lc[0]]
                    col = colors[lc[0]]

                # plot!
                try:
                    if isinstance(col, np.ndarray): 
                        for yd, cl in zip(ydata.T, col):
                            ax[pos].plot(xdata, yd, stl, color=cl, linewidth=2)
                    else:
                        ax[pos].plot(xdata, ydata, stl, color=col, linewidth=2)
                except Exception as e:
                    print(e)
                    pdb.set_trace()

            # plot lines
            ax[pos].grid(True)
            ax[pos].set_xticks(xticks)
            ax[pos].set_xticklabels([str(x) for x in xticks])
            ax[pos].set_xlim([np.min(xdata), np.max(xdata)])
            if ny == 1 or pos[0] == 0 or pos[0] % f_comp[quant] == 0:
                ax[pos].set_title(title)
            if ny == 1 or pos[0] == ny - 1:
                ax[pos].set_xlabel(xlabel)
            if nx == 1 or pos[-1] == 0:
                ax[pos].set_ylabel(f_labels[quant][j_data])
    
    # share y-axes
    if quant == "stim":
        sharey = [1, 2]
        ymin = []
        ymax = []
        for iy in sharey:
            if isinstance(ax[iy], (np.ndarray, list)):
                for a in ax[iy]:
                    ymin += [a.get_ylim()]
                    ymax += [a.get_ylim()]
            else:
                ymin += [ax[iy].get_ylim()]
                ymax += [ax[iy].get_ylim()]
        for iy in sharey:
            if isinstance(ax[iy], (np.ndarray, list)):
                for a in ax[iy]:
                    a.set_ylim(np.min(ymin), np.max(ymax))
            else:
                ax[iy].set_ylim(np.min(ymin), np.max(ymax))
    plt.tight_layout()
    fname += time_str + ".pdf"
    fig.savefig(os.path.join(out, fname), bbox_inches="tight")
    plt.close(fig)
    print(fname)


def main_param(folder, p_name, domain="solid", load_step=-1):
    # collect simulations
    out, inp, param = collect_simulations(folder)

    # collect all results
    data = OrderedDict()
    coords = {}
    times = {}
    for n, o in inp.items():
        data[n], coords[n], times[n] = post_process(o, domain)

    # collect all parameters
    study_params = np.unique([param[n][p_name] for n in data.keys()]).tolist()

    # get simulations names
    study_names = np.unique([n.split("_")[0] for n in data.keys()]).tolist()
    assert len(study_params) * len(study_names) == len(data), "Inconsistent data"

    # collect data for all parameter variations
    data_sorted = rec_dict()
    param_sorted = {}
    for n in sorted(data.keys()):
        i_s = n.split("_")[0]
        for loc in data[n].keys():
            for f in data[n][loc].keys():
                if f not in data_sorted[i_s][loc]:
                    data_sorted[i_s][loc][f] = []
                data_sorted[i_s][loc][f] += [data[n][loc][f][-1]]
        param_sorted[i_s] = {p_name: np.array(study_params, dtype=float)}

    plot_res(data_sorted, coords, times, param_sorted, out, domain, "KsKi", load_step=[load_step])


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
        elif "partitioned" in f or os.path.isdir(os.path.join(f, "partitioned")):
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


def plot_cc(f_out, out):
    # load debug QR data
    f_npy = os.path.join(f_out, "debug_qr.npy")
    if not os.path.exists(f_npy):
        return
    d = np.load(f_npy, allow_pickle=True).item()
    cc_list = d["cc"]
    ncols_after = d["ncols_after"]
    n_iter = len(cc_list)
    q_max = max(ncols_after)

    # t and n per IQN call
    has_tn = "t" in d and len(d["t"]) == n_iter
    t_arr = np.array(d["t"]) if has_tn else None
    n_arr = np.array(d["n"]) if has_tn else None

    # load-step transition boundaries (first call index of each new t)
    transitions = []
    if has_tn:
        for idx in range(1, n_iter):
            if t_arr[idx] != t_arr[idx - 1]:
                transitions.append(idx)

    # match residuals to IQN calls: use residual of the *next* sub-iter after
    # each IQN call to show the effect of the update (the call's own n is before update)
    residuals_before = None
    residuals_after = None
    f_json = os.path.join(f_out, "partitioned.json")
    if has_tn and os.path.exists(f_json):
        with open(f_json) as fh:
            p = json.load(fh)
        err = p.get("error", {})
        if err:
            first_key = next(iter(err))
            err_steps = err[first_key]
            res_b, res_a = [], []
            for i in range(n_iter):
                t, n = t_arr[i], n_arr[i]
                if t < len(err_steps) and n < len(err_steps[t]):
                    res_b.append(err_steps[t][n])
                    # next sub-iter residual (after IQN update was applied)
                    if n + 1 < len(err_steps[t]):
                        res_a.append(err_steps[t][n + 1])
                    else:
                        res_a.append(np.nan)
            if len(res_b) == n_iter:
                residuals_before = np.array(res_b)
                residuals_after = np.array(res_a)

    # build heatmap matrix, y from 1..q_max
    mat = np.full((n_iter, q_max), np.nan)
    for i, cc in enumerate(cc_list):
        mat[i, :len(cc)] = cc

    n_plots = 3 if residuals_before is not None else 2
    from matplotlib.gridspec import GridSpec
    # two columns: wide data column + narrow colorbar column (same for all rows)
    cbar_width = 0.03
    fig = plt.figure(figsize=(12, 3.5 * n_plots), dpi=150)
    gs = GridSpec(n_plots, 2, figure=fig,
                  width_ratios=[1 - cbar_width, cbar_width],
                  hspace=0.35, wspace=0.02)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(n_plots)]
    # add invisible placeholder subplots in colorbar column for all rows
    # so all data axes in column 0 have identical width
    for i in range(n_plots):
        ax_dummy = fig.add_subplot(gs[i, 1])
        ax_dummy.set_visible(False)
    # share x across all data axes
    for ax in axes[1:]:
        ax.sharex(axes[0])

    x = np.arange(n_iter)
    xlim = (-0.5, n_iter - 0.5)

    def add_vlines(ax, color="k", with_label=False):
        for xv in transitions:
            t_from = t_arr[xv - 1]
            t_to = t_arr[xv]
            lbl = f"t={t_from}$\\to${t_to}" if with_label else None
            ax.axvline(xv - 0.5, color=color, linestyle="--", linewidth=1,
                       label=lbl)

    # --- subplot 0: heatmap |cc|, log scale, integer y-ticks 1..q_max ---
    ax = axes[0]
    valid = mat[~np.isnan(mat)]
    vmax = np.percentile(np.abs(valid), 95)
    vmin = -vmax
    im = ax.pcolormesh(np.arange(n_iter + 1) - 0.5,
                       np.arange(q_max + 1) - 0.5,
                       mat.T,
                       cmap="RdBu_r", vmin=vmin, vmax=vmax,
                       shading="flat")
    ax.step(x, np.array(ncols_after) - 0.5, where="mid", color="k", linewidth=1.5)
    add_vlines(ax)
    ax.set_ylabel("Coefficient index")
    ytick_step = max(1, q_max // 10)
    yticks = np.arange(ytick_step - 1, q_max, ytick_step)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks + 1)
    ax.set_ylim(-0.5, q_max - 0.5)
    ax.set_xlim(xlim)
    ax.set_title(r"$c$ (95th percentile color scale)")
    # colorbar: use the row-0 placeholder (make it visible again)
    cax = fig.axes[n_plots]  # first dummy axes = row 0 col 1
    cax.set_visible(True)
    fig.colorbar(im, cax=cax)

    # --- subplot 1: norm of cc, log scale ---
    ax = axes[1]
    norms = [np.linalg.norm(cc) for cc in cc_list]
    ax.plot(x, norms, "ko-", markersize=3, linewidth=1)
    add_vlines(ax, with_label=True)
    ax.set_ylabel(r"$\|c\|$")
    ax.set_yscale("log")
    ax.set_xlim(xlim)
    ax.grid(True, which="both", alpha=0.4)
    ax.set_title(r"Norm of IQN-ILS coefficients $c$")

    # --- subplot 2: residual before and after each IQN update ---
    if n_plots == 3:
        ax = axes[2]
        ax.plot(x, residuals_before, "bo-", markersize=3, linewidth=1, label="before update")
        ax.plot(x, residuals_after,  "rs-", markersize=3, linewidth=1, label="after update")
        ax.axhline(p["coup"]["tol"], color="k", linestyle="--", linewidth=1, label="tol")
        add_vlines(ax, with_label=False)
        ax.set_ylabel("Residual")
        ax.set_yscale("log")
        ax.set_xlim(xlim)
        ax.grid(True, which="both", alpha=0.4)
        ax.set_title("Coupling residual before/after IQN-ILS update")
        ax.set_xlabel("IQN-ILS call index")
        ax.legend(fontsize=7)
    else:
        axes[-1].set_xlabel("IQN-ILS call index")

    from matplotlib.ticker import MaxNLocator
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    fname = os.path.join(out, "cc_coefficients.pdf")
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print("cc_coefficients.pdf")


def plot_arc(f_run, out):
    # load arc-length trajectory (saved by fsg._run_arclength), cf. debug_qr.npy
    f_npy = os.path.join(f_run, "arc_data.npy")
    if not os.path.exists(f_npy):
        return
    d = np.load(f_npy, allow_pickle=True).item()
    lam = np.array(d["lam"]); dd = np.array(d["dd"]); res = np.array(d["res"])
    t_arr = np.array(d["t"])
    ds = np.array(d["ds"]) if "ds" in d and len(d["ds"]) == len(lam) else None
    tol = d.get("tol", 1e-4)
    acc_t = np.array(d["accept_t"]); acc_lam = np.array(d["accept_lam"])
    n_iter = len(lam)
    if n_iter == 0:
        return

    x = np.arange(n_iter)
    xlim = (-0.5, n_iter - 0.5)

    # load-step transition boundaries (first sub-iter index of each new t)
    transitions = [i for i in range(1, n_iter) if t_arr[i] != t_arr[i - 1]]

    def add_vlines(ax, with_label=False):
        for xv in transitions:
            lbl = f"t={t_arr[xv-1]}$\\to${t_arr[xv]}" if with_label else None
            ax.axvline(xv - 0.5, color="k", linestyle="--", linewidth=1, label=lbl)

    # per-step load increment: lambda - lambda_old, where lambda_old = accepted
    # lambda of the PREVIOUS step (0 for step 1). Resets each load step, like dd.
    acc = {int(at): al for at, al in zip(acc_t, acc_lam)}
    dlam = np.array([lam[k] - acc.get(int(t_arr[k]) - 1, 0.0) for k in range(n_iter)])

    nrows = 5 if ds is not None else 4
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 2.6 * nrows), dpi=150, sharex=True)

    # --- subplot 0: load factor lambda ---
    ax = axes[0]
    ax.plot(x, lam, "b-", linewidth=1)
    for st, lv in zip(acc_t, acc_lam):
        idx = np.where(t_arr == st)[0]
        if len(idx):
            ax.plot(idx[-1], lv, "go", markersize=5)
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1, label=r"$\lambda=1$")
    add_vlines(ax)
    ax.set_ylabel(r"load factor $\lambda$")
    ax.set_xlim(xlim)
    ax.grid(True, alpha=0.4)
    ax.set_title("Arc-length continuation (green = accepted step)")
    ax.legend(fontsize=7)

    # --- subplot 1: load increment per step (lambda - lambda_old, resets each step) ---
    ax = axes[1]
    ax.plot(x, dlam, color="teal", linewidth=1)
    add_vlines(ax)
    ax.set_ylabel(r"load incr. $\Delta\lambda$")
    ax.set_xlim(xlim)
    ax.grid(True, alpha=0.4)

    # --- subplot 2: bulge increment (mean nodal |dd|, consistent w/ residual) ---
    ax = axes[2]
    ax.plot(x, dd, color="purple", linewidth=1)
    add_vlines(ax)
    ax.set_ylabel(r"bulge incr. mean$|\Delta d|$ (cm)")
    ax.set_xlim(xlim)
    ax.grid(True, alpha=0.4)

    # --- subplot 3: adaptive arc step ds (halves on a failed step) ---
    if ds is not None:
        ax = axes[3]
        ax.step(x, ds, where="post", color="darkorange", linewidth=1.2)
        add_vlines(ax)
        ax.set_ylabel(r"arc step $\Delta s$ (cm)")
        ax.set_yscale("log")
        ax.set_xlim(xlim)
        ax.grid(True, which="both", alpha=0.4)

    # --- last subplot: coupling residual, log scale ---
    ax = axes[-1]
    ax.plot(x, res, "k-", linewidth=1)
    ax.axhline(tol, color="k", linestyle="--", linewidth=1, label="tol")
    add_vlines(ax, with_label=True)
    ax.set_ylabel("Coupling residual")
    ax.set_yscale("log")
    ax.set_xlim(xlim)
    ax.grid(True, which="both", alpha=0.4)
    ax.set_xlabel("cumulative coupling sub-iteration")
    ax.legend(fontsize=7)

    from matplotlib.ticker import MaxNLocator
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    fname = os.path.join(out, "arc_lambda.pdf")
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print("arc_lambda.pdf")


def main_arg(folder, domain="solid", load_step=-1):
    # collect simulations
    out, inp, param = collect_simulations(folder)

    # collect all results
    data = OrderedDict()
    coords = {}
    times = {}
    for n, o in inp.items():
        data[n], coords[n], times[n] = post_process(o, domain)

    plot_res(data, coords, times, param, out, domain, "single", load_step=[load_step])

    # plot IQN-ILS cc coefficients + arc-length trajectory if their data exist
    for n, f_out in inp.items():
        f_run = os.path.dirname(os.path.dirname(f_out.rstrip("/")))  # go up from partitioned/converged to run root
        plot_cc(f_run, out)
        plot_arc(f_run, out)


def main_convergence(folder):
    # collect simulations
    out, inp, data = collect_simulations(folder)

    ydata = []
    labels = []
    param = []
    for f in inp.keys():
        kski = data[f]["KsKi"]
        labels += [r"$K_{\tau\sigma,o}$ = " + kski]
        n_it = []
        for err in data[f]["error"]:
            n_it += [len(err)]
        print(kski, "{:.1f}".format(np.mean(n_it[2:])), np.sum(n_it))
        ydata += [np.cumsum(n_it)]
        param += [float(kski)]

    ydata = np.array(ydata).T
    xdata = np.arange(0, ydata.shape[0])
    param = np.array(param)

    fig, ax = plt.subplots(figsize=(12.5, 5), dpi=300)

    colors = get_colormap(param)
    for y, c in zip(ydata.T, colors):
        ax.plot(xdata, y, color=c, linewidth=2)
    ax.grid(True)
    ax.set_xticks(xdata)
    ax.set_xlim([np.min(xdata), np.max(xdata)])
    ax.set_ylim([0, np.max(ydata)])
    ax.set_xlabel("Load step $t$ [-]")
    ax.set_ylabel("Coupling iterations [-]")

    ax2 = ax.twinx()
    ax2.set_ylim([ax.get_ylim()[0], ax.get_ylim()[1]])
    ax2.set_yticks(ydata[-1])
    ax2.set_yticklabels(labels)

    plt.tight_layout()
    fig.savefig(os.path.join(out, "convergence.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process FSGe simulation")
    parser.add_argument("out", nargs="+", default="None", help="svFSI output folder")
    parser.add_argument("-c", action="store_true", help="Plot convergence")
    parser.add_argument("-p", type=str, help="Plot parametric study")
    parser.add_argument("-f", action="store_true", help="Plot fluid (instead of solid)")
    parser.add_argument("-t", type=int, default=-1, help="Plot specific load step (default: last)")
    args = parser.parse_args()

    domain = "fluid" if args.f else "solid"
    if args.c:
        main_convergence(args.out)
    elif args.p:
        main_param(args.out, args.p, domain, load_step=args.t)
    else:
        main_arg(args.out, domain, load_step=args.t)
