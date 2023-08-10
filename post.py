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
    return np.array([np.arctan2(xyz[1], xyz[0]), np.sqrt(xyz[0]**2.0 + xyz[1]**2.0), xyz[2]])

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
    
    # radial coordinates of points to export:
    p_rad = {"out": ro,
             "in": ri}

    # axial coordinates of points to export: inlet, mid-point, outlet
    p_axi = {"start": 0.0,
             "mid": h / 2,
             "end": h}
    
    # collect all point coordinates
    points = {}
    for an, ap in p_axi.items():
        for cn, cp in p_cir.items():
            for rn, rp in p_rad.items():
                points[str(cn) + "_" + rn + "_" + an] = cra2xyz([cp, rp, ap])
    
    # collect all line coordinates
    lines = {}
    for cn, cp in p_cir.items():
        for rn, rp in p_rad.items():
            lines[str(cn) + "_" + rn] = cra2xyz([cp, rp, 0.0])
    
    # collect all point ids
    pids = {}
    for n, pt in points.items():
        # get id
        chk = [np.isclose(pts_xyz[:, i], pt[i]) for i in range(3)]
        id = np.where(np.logical_and.reduce(np.array(chk)))[0]
        assert len(id) == 1, "point definition not unique: " + str(pt)
        pids[n] = id[0]

    # collect all line ids
    lids = {}
    for n, ln in lines.items():
        # get ids
        chk = [np.isclose(pts_xyz[:, i], ln[i]) for i in range(2)]
        id = np.where(np.logical_and.reduce(np.array(chk)))[0]
        assert len(id) > 1, "line definition empty: " + str(ln)

        # sort according to z-coordinate
        zs = pts_xyz[id, 2]
        lids[n] = id[np.argsort(zs)]
    return pids, lids

def get_results(results, pts, pids, lids):
    # get displacements in radial coordinates at all extracted points
    disp = defaultdict(list)
    for res in results:
        extract_results(disp, results[0], res, pts, pids, "p")
    extract_results(disp, results[0], results[-1], pts, lids, "l")
    for loc in disp.keys():
        disp[loc] = np.array(disp[loc])
    return disp

def extract_results(disp, res0, res, pts, ids, mode):
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

        # Poiseuille estimated wss
        wss = 4.0 * mu * q0 / np.pi / rad ** 3.0
        wss0 = 4.0 * mu * q0 / np.pi / rad0 ** 3.0
    
    # get stimuli
    np.seterr(divide='ignore', invalid='ignore')
    stim_wss = wss / wss0 - 1.0
    stim_sig = trace / trace0 - 1.0
    stim_all = stim_sig / stim_wss

    for n, pt in ids.items():
        # displacement in polar coordinates
        diff = xyz2cra((pts[pt] + d[pt]).T) - xyz2cra(pts[pt].T)

        # limit angle to [-pi, pi)
        diff[0] += np.pi
        diff[0] %= np.pi * 2.0
        diff[0] -= np.pi

        # store displacements
        disp[mode + "_disp_" + n] += [diff.T]
        
        # extract thickness and wss
        if "out" in n:
            n_in = n.replace("out", "in")
            n_scalar = n.replace("_out", "")
            d_out = pts[ids[n]] + d[ids[n]]
            d_in = pts[ids[n_in]] + d[ids[n_in]]
            if mode == "l":
                disp["l_thick_" + n_scalar] = np.linalg.norm((d_out - d_in).T, axis=0)
                disp["l_wss_" + n_scalar] = wss[pt]
                disp["l_z_" + n_scalar] = pts[pt, 2]
            else:
                disp["p_thick_" + n_scalar] += [np.linalg.norm((d_out - d_in).T, axis=0)]
                disp["p_wss_" + n_scalar] += [wss[pt]]
        
        # extract stimuli
        if mode == "l":
            disp[mode + "_stim_all_" + n] = stim_all[pt]
            disp[mode + "_stim_" + n] = np.array([stim_sig[pt], stim_wss[pt]]).T
        else:
            disp[mode + "_stim_all_" + n] += [stim_all[pt]]
            disp[mode + "_stim_" + n] += [np.array([stim_sig[pt], stim_wss[pt]])]

        # store z-coordinate for lines
        if mode == "l":
            disp["l_z_" + n] = pts[pt, 2]

def extract_scalar(scalar, res, pts, ids, mode):
    d = v2n(res.GetPointData().GetArray("Displacement"))
    for m in ids.keys():
        if "out" in m:
            m_in = m.replace("out", "in")
            m_thick = m.replace("out", "thick")
            d_out = pts[ids[m]] + d[ids[m]]
            d_in = pts[ids[m_in]] + d[ids[m_in]]
            scalar["p_" + m_thick] += [np.linalg.norm(d_out - d_in)]

def post(f_out):
    # check if FSGe or conventional G&R results
    if ".json" in f_out:
        fsge = True
        sim = read_config(f_out)
        fname = os.path.join(sim.p['f_out'], "partitioned", "converged", "tube_*.vtu")
    else:
        fsge = False
        fname = os.path.join(f_out, "gr_*.vtu")
    
    # read results from fike
    res = read_res(fname, fsge)

    # extract points
    pts = v2n(res[0].GetPoints().GetData())

    # get point and line ids
    pids, lids = get_ids(pts)

    # extract displacements
    return get_results(res, pts, pids, lids)

def plot_disp(data, out, study):
    # assemble all plots (quantity and location)
    fields = ["thick", "wss"]
    # stimuli = ["wss", "sig", "all"]
    stimuli = ["", "_all"]
    locations = ["start", "mid", "end"]

    for loc in locations:
        plot_single(data, os.path.join(out, "disp_points_" + loc + ".png"), study, "p", "disp", "in_" + loc)
        for s in stimuli:
            plot_single(data, os.path.join(out, "stim" + s + "_points_" + loc + ".png"), study, "p", "stim" + s, "in_" + loc)
        for res in fields:
            plot_single(data, os.path.join(out, res + "_points_" + loc + ".png"), study, "p", res, loc)
        
    if study == "single":
        plot_single(data, os.path.join(out, "disp_lines.png"), study, "l", "disp", "in")
        for s in stimuli:
            plot_single(data, os.path.join(out, "stim" + s + "_lines.png"), study, "l", "stim" + s, "in")
        for res in fields:
            plot_single(data, os.path.join(out, res + "_lines.png"), study, "l", res)

def plot_single(data, out, study, mode, quant, loc=""):
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
    
    xstudies = {"kski": "$K_{\\tau\sigma}$"}

    # collect data for all parameter variations
    if study != "single":
        data_sorted = {}
        params = sorted(data.keys())
        sims = data[params[0]].keys()
        for nam in sims:
            data_sorted[nam] = {}
            for j in range(0, 12, 3):
                yres = "_".join(filter(None, [mode, quant, str(j), loc]))
                data_sorted[nam][yres] = []
                for k in params:
                    data_sorted[nam][yres] += [data[k][nam][yres][-1]]
                data_sorted[nam][yres] = np.array(data_sorted[nam][yres])
        data = data_sorted

    # determine plot dimensions
    nx = len(data)
    ny = len(ylabel)

    # plot properties
    oclocks = range(0, 12, 3)
    colors = [plt.cm.tab10(i) for i in [0, 1, 3, 2]]
    # from matplotlib.colors import to_hex
    # print([to_hex(c) for c in colors])
    styles = ["-", "-", "-", ":"]
    styles_cir = ["-", "-", ":", "-"]

    fig, ax = plt.subplots(ny, nx, figsize=(nx * 10, ny * 5), dpi=300, sharex="col", sharey="row")
    for j, (n, res) in enumerate(data.items()):
        for i in range(ny):
            # select plot position
            if ny == 1:
                pos = j
            else:
                pos = (i, j)

            # plot lines
            ax[pos].grid(True)
            if quant != "thick":
                ax[pos].axhline(0, color='black', zorder=2)

            # loop mesh positions
            for ik, k in enumerate(oclocks):
                # get data for y-axis
                xres = "l_z_" +  "_".join(filter(None, [str(k), loc]))
                yres =  "_".join(filter(None, [mode, quant, str(k), loc]))
                title = n.replace("&", "\&")
                ydata = res[yres]
                assert len(ydata) > 0, "no data found: " + yres
                if ny > 1:
                    ydata = ydata.T[i]

                # get data for x-axis
                if study == "single":
                    if mode == "p":
                        xdata = np.arange(0, len(ydata))
                        xlabel = "Load step [-]"
                        xticks = np.arange(0, len(ydata), 2)
                    elif mode == "l":
                        xdata = res[xres]
                        xlabel = "Vessel length [mm]"
                        xticks = [0, 2, 4, 6, 7.5, 9, 11, 13, 15]
                    else:
                        raise RuntimeError("Unknown mode: " + mode)
                else:
                    xdata = params
                    xlabel = xstudies[study]
                    xticks = [0, 0.25, 0.5, 0.75, 1]
                assert len(xdata) > 0, "no data found: " + xres

                # convert to degrees
                if quant == "disp" and i == 0:
                    ydata *= - 180 / np.pi
                    stl = styles_cir[ik]
                else:
                    stl = styles[ik]

                # plot!
                ax[pos].plot(xdata, ydata, stl, color=colors[ik], linewidth=2)
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
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)

def plot_insult(out):
    lo = 15.0
    z_om = 0.5 * lo
    z_od = 0.25 * lo
    theta_od = 0.55
    vza = 2
    vzc = 6

    z = np.linspace(0, lo, 101)
    # azimuth = np.linspace(-np.pi, np.pi, 101)
    theta = np.linspace(0.0, 2.0 * np.pi, 101)

    f_axi = np.exp(-np.power(np.abs((z - z_om) / z_od), vza))
    azimuth = theta + np.pi
    azimuth %= 2.0 * np.pi
    azimuth -= np.pi
    f_cir = np.exp(-np.power(np.abs(azimuth / (np.pi * theta_od)), vzc))

    xdata = [theta, z]
    ydata = [f_cir, f_axi]
    xlabel = ["Vessel circumference", "Vessel length [mm]"]
    ylabel = ["Elastin insult [-]"] * 2
    xticks = [np.arange(0.0, 360.0, 45), [0, 7.5, 15]]
    yticks = [0, 1]
    title = ["Circumferential insult", "Axial insult"]

    nx = 2
    ny = 1
    fig = plt.figure(figsize=(nx * 10, ny * 5), dpi=300)
    ax = [plt.subplot2grid((ny, nx), (0, 0), projection='polar'),
          plt.subplot2grid((ny, nx), (0, 1))]

    for i in range(nx):
        ax[i].grid(True)
        ax[i].axhline(0, color='black', zorder=2)
        ax[i].plot(xdata[i], ydata[i])
        ax[i].set_title(title[i])
        if i == 0:
            ax[i].set_theta_zero_location("N")
            ax[i].set_theta_direction(-1)
            ax[i].set_rlim(yticks)
            ax[i].set_rticks(yticks)
            ax[i].set_rlabel_position(20)
            ax[i].set_thetagrids(xticks[i])
        else:
            ax[i].set_xlim([np.min(xdata[i]), np.max(xdata[i])])
            ax[i].set_ylim(yticks)
            ax[i].set_xticks(xticks[i])
            ax[i].set_xticklabels([str(x) for x in xticks[i]])
            ax[i].set_yticks(yticks)
            ax[i].set_xlabel(xlabel[i])
            ax[i].set_ylabel(ylabel[i])

    # plt.tight_layout()
    fig.savefig(os.path.join(out, "insult.png"), bbox_inches='tight')
    plt.close(fig)

    import sys
    sys.exit(0)
    # pdb.set_trace()

def main():
    # set study
    # for kski in ["0.5"]:  
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
            data[n] = post(o)
        
        plot_disp(data, out, "single")

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
            data[k][n] = post(o)
    
    plot_disp(data, out, "kski")

if __name__ == "__main__":
    main_param()
    main()
