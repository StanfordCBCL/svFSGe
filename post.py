#!/usr/bin/env python
# coding=utf-8

import pdb
import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import StrMethodFormatter
from collections import defaultdict
from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

from svfsi import svFSI
from vtk_functions import read_geo, threshold

# use LaTeX in text
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.serif': 'Computer Modern Roman', 'font.size': 24})

def cra2xyz(cra):
    return np.array([cra[1] * np.sin(cra[0]), cra[1] * np.cos(cra[0]), cra[2]])

def xyz2cra(xyz):
    return np.array([(np.arctan2(xyz[0], xyz[1]) + 2.0 * np.pi) % (2.0 * np.pi), np.sqrt(xyz[0]**2.0 + xyz[1]**2.0), xyz[2]])

def read_config(json):
    # read simulation config
    if not os.path.exists(json):
        raise RuntimeError('No json file found: ' + json)
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
    p_cir = {0: 0.0,
             3: 0.5 * np.pi,
             6: np.pi,
             9: 1.5 * np.pi}
    
    # radial coordinates of points to export: inside, outside
    p_rad = {"out": ro,
             "in": ri}

    # axial coordinates of points to export: inlet, mid-point, outlet
    p_axi = {"start": 0.0,
             "mid": h / 2,
             "end": h}

    # collect all point coordinates
    locations = {}
    for cn, cp in p_cir.items():
        for rn, rp in p_rad.items():
            for an, ap in p_axi.items():
                identifiers = [(cn, rn, an), (":", rn, an), (cn, ":", an), (cn, rn, ":")]
                for i in identifiers:
                    locations[i] = [cp, rp, ap]

    # collect all point ids
    ids = {}
    coords = {}
    for loc, pt in locations.items():
        chk = [np.isclose(pts_cra[:, i], pt[i]) for i in range(3) if loc[i] != ":"]
        ids[loc] = np.where(np.logical_and.reduce(np.array(chk)))[0]
        assert len(ids[loc]) > 0, "no points found: " + str(loc)

        # sort according to coordinate
        if ":" in loc:
            crd = pts_cra[ids[loc], list(loc).index(":")]
            sort = np.argsort(crd)
            ids[loc] = ids[loc][sort]
            coords[loc] = crd[sort]
            assert len(np.unique(crd)) == len(crd), "coordinates not unique: " + str(crd)

    return ids, coords

def get_results(results, pts, ids):
    # get post-processed quantities at all extracted locations
    post = {}
    for loc in ids.keys():
        post[loc] = defaultdict(list)

    # get results at all time steps
    for res in results:
        extract_results(post, results[0], res, pts, ids)

    # convert to numpy arrays
    for loc in post.keys():
        for f in post[loc].keys():
            post[loc][f] = np.squeeze(np.array(post[loc][f]))

    return post

def extract_results(post, res0, res, pts, ids):
    # get nodal displacements
    d = v2n(res.GetPointData().GetArray("Displacement"))

    # Cauchy stress
    sig = v2n(res.GetPointData().GetArray("Cauchy_stress"))
    sig0 = v2n(res0.GetPointData().GetArray("Cauchy_stress"))

    # stress invariant
    trace = np.sum(sig[:,:3], axis=1) / 3
    trace0 = np.sum(sig0[:,:3], axis=1) / 3

    # get nodal wall shear stress
    if res.GetPointData().HasArray("WSS"):
        wss = v2n(res.GetPointData().GetArray("WSS"))
        wss0 = v2n(res0.GetPointData().GetArray("WSS"))
    else:
        # enter constants manually based on G&R simulation
        q0 = 1000
        mu = 4.0e-06

        # radial coordinate
        rad = xyz2cra((pts + d).T)[1]
        rad0 = xyz2cra((pts).T)[1]

        # Poiseuille estimated wss (constants cancel out in stimulus function)
        wss = 4.0 * mu * q0 / np.pi / rad ** 3.0
        wss0 = 4.0 * mu * q0 / np.pi / rad0 ** 3.0
    
    # get stimuli
    np.seterr(divide='ignore', invalid='ignore')
    stim_wss = wss / wss0 - 1.0
    stim_sig = trace / trace0 - 1.0
    stim_all = stim_sig / stim_wss

    for loc, pt in ids.items():
        # displacement in polar coordinates
        diff = xyz2cra((pts[pt] + d[pt]).T) - xyz2cra(pts[pt].T)

        # limit angle to [-pi, pi)
        diff[0] += np.pi
        diff[0] %= np.pi * 2.0
        diff[0] -= np.pi

        # store displacements
        post[loc]["disp"] += [diff]
        
        # extract thickness and wss
        if loc[1] == "in":
            dloc = (loc[0], "out", loc[2])

            d1 = pts[ids[loc]] + d[ids[loc]]
            d2 = pts[ids[dloc]] + d[ids[dloc]]

            post[loc]["thick"] += [np.linalg.norm((d1 - d2).T, axis=0)]
            post[loc]["wss"] += [wss[pt]]
        
        # extract stimuli
        post[loc]["stim_all"] += [stim_all[pt]]
        post[loc]["stim"] += [np.array([stim_sig[pt], stim_wss[pt]])]

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
    if ".json" in f_out:
        fsge = True
        sim = read_config(f_out)
        fname = os.path.join(sim.p['f_out'], "partitioned", "converged", "tube_*.vtu")
    elif "partitioned" in f_out:
        fsge = True
        fname = os.path.join(f_out, "tube_*.vtu")
    else:
        fsge = False
        fname = os.path.join(f_out, "gr_*.vtu")
    
    # read results from fike
    res = read_res(fname, fsge)

    # extract points
    pts = v2n(res[0].GetPoints().GetData())

    # get point and line ids
    ids, coords = get_ids(pts)

    # extract displacements
    return get_results(res, pts, ids), coords

def plot_disp(data, coords, out, study):
    # plot locations
    loc_cir = range(0, 12, 3)
    loc_rad = ["in"] # "out"
    loc_axi = ["mid"] # "start", "end"

    # plot all points
    fields = ["disp", "thick", "wss", "stim", "stim_all"]
    for lr in loc_rad:
        for la in loc_axi:
            for f in fields:
                loc_points = []
                loc_lines = []

                # loop circumferential locations
                for lc in loc_cir:
                    # plot single points
                    loc_points += [(lc, lr, la)]

                    # plot along axial lines
                    loc_lines += [(lc, lr, ":")]
                if study == "single":
                    # plot circumferential ring
                    plot_single(data, coords, out, study, f, [(":", lr, la)])
                    plot_single(data, coords, out, study, f, loc_lines)
                plot_single(data, coords, out, study, f, loc_points)

def plot_single(data, coords, out, study, quant, locations):
    # plot text
    if quant == "disp":
        ylabel = ["Cir. displacement $\Delta\\theta$ [°]",
                  "Rad. displacement $\Delta r$ [mm]",
                  "Axi. displacement $\Delta z$ [mm]"]
    elif quant == "thick":
        ylabel =[ "Thickness [mm]"]
    elif quant == "wss":
        ylabel = ["WSS [?]"]
    elif quant == "stim_all":
        ylabel = ["$K_{\\tau\sigma}$ [-]"]
    elif quant == "stim":
        ylabel = ["Intramular stimulus $\Delta\sigma_I$ [-]", "WSS stimulus $\Delta\\tau_w$ [-]"]
    else:
        raise RuntimeError("Unknown quantity: " + quant)

    # determine plot dimensions
    nx = len(data)
    ny = len(ylabel)

    fig, ax = plt.subplots(ny, nx, figsize=(nx * 10, ny * 5), dpi=300, sharex="col", sharey="row")
    for j, (n, res) in enumerate(data.items()):
        title = n.replace("&", "\&")
        for i in range(ny):
            # select plot position
            if nx == 1 or ny == 1:
                pos = j
            else:
                pos = (i, j)

            # plot lines
            ax[pos].grid(True)
            if quant != "thick":
                ax[pos].axhline(0, color='black', zorder=2)

            # loop mesh positions
            for ic, lc in enumerate(locations):
                loc = list(lc)

                # get data for y-axis
                ydata = res[lc][quant].copy()
                if not len(ydata):
                    return
                if ":" in lc:
                    ydata = ydata[-1].T
                if ny > 1:
                    ydata = ydata.T[i]
                    
                # get data for x-axis
                if study == "single":
                    if ":" in loc:
                        xdata = coords[tuple(loc)].copy()
                        dim = loc.index(":")
                        if dim == 0:
                            xlabel = "Vessel circumference [°]"
                            xdata *= 180 / np.pi
                            xticks = xdata[0::4].astype(int)
                        elif dim == 1:
                            xlabel = "Vessel radius [°]"
                            xticks = xdata.astype(str)
                        elif dim == 2:
                            xlabel = "Vessel length [mm]"
                            xticks = [0, 2, 4, 6, 7.5, 9, 11, 13, 15]
                        loc.remove(":")
                    else:
                        xdata = np.arange(0, len(ydata))
                        xlabel = "Load step [-]"
                        xticks = xdata[::2]

                    # plot properties
                    if len(locations) > 1:
                        colors = [plt.cm.tab10(i) for i in [0, 1, 3, 2]]
                        styles = ["-", "-", "-", ":"]
                        styles_cir = ["-", "-", ":", "-"]

                        # file name
                        fname = "_".join([quant] + [str(l) for l in loc[1:]]) + ".png"
                    else:
                        # plot properties
                        styles = ["-"]
                        styles_cir = styles
                        colors = ["#8C1515"]

                        # file name
                        dim_names = ["cir", "rad", "axi"]
                        fname = "_".join([quant, dim_names[dim]] + [str(l) for l in loc]) + ".png"
                elif study == "kski":
                    # get data for x-axis
                    xdata = np.linspace(0.0, 1.0, 5)
                    xlabel = "$K_{\\tau\sigma}$"
                    xticks = [0, 0.25, 0.5, 0.75, 1]

                    # plot properties
                    colors = [plt.cm.tab10(i) for i in [0, 1, 3, 2]]
                    styles = ["-", "-", "-", ":"]
                    styles_cir = ["-", "-", ":", "-"]
                
                    # file name
                    fname = "_".join(["kski", quant] + [str(l) for l in loc[1:]]) + ".png"
                else:
                    raise ValueError("unknown study: " + study)

                # convert to degrees
                if quant == "disp" and i == 0 and len(locations) > 1:
                    stl = styles_cir[ic]
                else:
                    stl = styles[ic]

                # plot!
                ax[pos].plot(xdata, ydata, stl, color=colors[ic], linewidth=2)
            ax[pos].set_xticks(xticks)
            ax[pos].set_xticklabels([str(x) for x in xticks])
            ax[pos].set_xlim([np.min(xdata), np.max(xdata)])
            if i == 0:
                ax[pos].set_title(title)
            if i == ny - 1:
                ax[pos].set_xlabel(xlabel)
            if j == 0:
                ax[pos].set_ylabel(ylabel[i])
    plt.tight_layout()
    fig.savefig(os.path.join(out, fname), bbox_inches='tight')
    # plt.close(fig)
    plt.cla()

def plot_insult(out):
    lo = 15.0
    z_om = 0.5 * lo
    z_od = 0.25 * lo
    theta_od = 0.55
    vza = 2
    vzc = 6

    z = np.linspace(0, lo, 101)
    azimuth = np.linspace(-np.pi, np.pi, 101)

    f_axi = np.exp(-np.power(np.abs((z - z_om) / z_od), vza))
    f_cir = np.exp(-np.power(np.abs((azimuth) / (np.pi * theta_od)), vzc))

    xdata = [azimuth * 180 / np.pi, z]
    ydata = [f_cir, f_axi]
    xlabel = ["Vessel circumference [°]", "Vessel length [mm]"]
    ylabel = ["Elastin insult [-]"] * 2
    xticks = [np.arange(-180, 270, 90), [0, 7.5, 15]]
    yticks = [0, 1]
    title = ["Circumferential insult", "Axial insult"]

    nx = 2
    ny = 1
    fig, ax = plt.subplots(ny, nx, figsize=(nx * 10, ny * 5), dpi=300, sharex="col", sharey="row")

    for i in range(nx):
        ax[i].grid(True)
        ax[i].axhline(0, color='black', zorder=2)
        ax[i].plot(xdata[i], ydata[i])
        ax[i].set_title(title[i])
        ax[i].set_xlim([np.min(xdata[i]), np.max(xdata[i])])
        ax[i].set_ylim(yticks)
        ax[i].set_xticks(xticks[i])
        if i == 0:
            ax[i].set_ylabel(ylabel[i])
        ax[i].set_xticklabels([str(x) for x in xticks[i]])
        ax[i].set_yticks(yticks)
        ax[i].set_xlabel(xlabel[i])
    plt.tight_layout()
    fig.savefig(os.path.join(out, "insult.png"), bbox_inches='tight')
    plt.clf()

def main():
    # set study
    for kski in np.linspace(0.0, 1.0, 5).astype(str):
        folder = "/Users/pfaller/work/repos/FSGe/study_aneurysm"
        geo = "coarse"
        phi = "0.7"

        # define paths
        fpath = os.path.join(folder, geo,  "phi_" + phi, "kski_" + kski)
        inp = {"G&R": os.path.join(fpath, "gr"),
            "FSGe": os.path.join(fpath, "partitioned", "partitioned.json")}
        out = os.path.join(fpath, "comparison")

        # create output folder
        os.makedirs(out, exist_ok=True)

        # collect all results
        data = {}
        for n, o in inp.items():
            data[n], coords = post_process(o)
        
        plot_disp(data, coords, out, "single")

def main_param():
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
        fpath = os.path.join(folder, geo,  "phi_" + phi, "kski_" + str(k))
        inp = {"G&R": os.path.join(fpath, "gr"),
               "FSGe": os.path.join(fpath, "partitioned", "partitioned.json")}
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

    plot_disp(data_sorted, coords, out, "kski")

def main_arg(folder):
    # define paths
    out = os.path.join(folder, "post")
    os.makedirs(out, exist_ok=True)

    # post-process simulation (converged and unconverged)
    data = {}
    inp = {"FSGe": os.path.join(folder, "partitioned.json"),
           "FSGe unvconverged": os.path.join(folder, "partitioned")}
    
    # collect all results
    data = {}
    for n, o in inp.items():
        data[n], coords = post_process(o)
    
    plot_disp(data, coords, out, "single")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process FSGe simulation")
    parser.add_argument("out", nargs='?', default=None, help="svFSI output folder")
    args = parser.parse_args()
    if not args.out:
        main_param()
        main()
    else:
        main_arg(args.out)

