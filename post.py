#!/usr/bin/env python
# coding=utf-8

import pdb
import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

from svfsi import svFSI
from vtk_functions import read_geo, threshold

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

def get_disp(results, pts, pids, lids):
    # get displacements in radial coordinates at all extracted points
    disp = defaultdict(list)
    for res in results:
        extract_disp(disp, res, pts, pids, "p")
    extract_disp(disp, results[-1], pts, lids, "l")
    for loc in disp.keys():
        disp[loc] = np.array(disp[loc])
    return disp

def extract_disp(disp, res, pts, ids, mode):
    d = v2n(res.GetPointData().GetArray("Displacement"))
    for n, pt in ids.items():
        # displacement in polar coordinates
        diff = xyz2cra((pts[pt] + d[pt]).T) - xyz2cra(pts[pt].T)

        # limit angle to [-pi, pi)
        diff[0] += np.pi
        diff[0] %= 2.0 * np.pi
        diff[0] -= np.pi
        disp[mode + "_" + n] += [diff.T]

        # store z-coordinate for lines
        if mode == "l":
            disp["lr_" + n] = pts[pt, 2]

def get_scalar(results, pts, pids, lids):
    scalar = defaultdict(list)

    ids = {"p": pids, "l": lids}
    # extract thickness
    for res in results:
        d = v2n(res.GetPointData().GetArray("Displacement"))
        for m in pids.keys():
            if "out" in m:
                m_in = m.replace("out", "in")
                m_thick = m.replace("out", "thick")
                d_out = pts[pids[m]] + d[pids[m]]
                d_in = pts[pids[m_in]] + d[pids[m_in]]
                scalar["p_" + m_thick] += [np.linalg.norm(d_out - d_in)]
    else:
        d = v2n(res.GetPointData().GetArray("Displacement"))
        for m in lids.keys():
            if "out" in m:
                m_in = m.replace("out", "in")
                m_thick = m.replace("out", "thick")
                d_out = pts[lids[m]] + d[lids[m]]
                d_in = pts[lids[m_in]] + d[lids[m_in]]
                scalar["l_" + m_thick] += [np.linalg.norm((d_out - d_in).T, axis=0)]
    return scalar

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
    return get_disp(res, pts, pids, lids)#, get_scalar(res, pts, pids, lids)

def plot_disp(data, out):
    plot_single(data, os.path.join(out, "displacement_points.png"), "pt", "out_mid")
    plot_single(data, os.path.join(out, "displacement_lines.png"), "ln", "out")
    # plot_single(data, os.path.join(out, "thickness_points.png"), "pt", "thick_mid")
    # plot_single(data, os.path.join(out, "thickness_lines.png"), "ln", "thick")

def plot_disp_param(data, out):
    plot_single_param(data, os.path.join(out, "displacement_kski.png"), "pt", "out_mid")
    # plot_single_param(data, os.path.join(out, "thickness_kski.png"), "pt", "thick_mid")

def plot_single(data, out, mode, loc):
    coords = ["cir", "rad", "axi"]
    units = ["°", "mm", "mm"]

    nx = len(data)
    ny = 3

    fig, ax = plt.subplots(ny, nx, figsize=(ny * 5, nx * 5), sharex="col", sharey="row")
    for j, (n, d) in enumerate(data.items()):
        for i in range(ny):
            for k in range(0, 12, 3):
                label = str(k) + "_" + loc
                if mode == "pt":
                    ydata = d["p_" + label].T[i]
                    xdata = np.arange(0, len(ydata))
                    xlabel = "Load step [-]"
                elif mode == "ln":
                    ydata = d["l_" + label].T[i]
                    xdata = d["lr_" + label]
                    xlabel = "Vessel length [mm]"
                else:
                    raise RuntimeError("Unknown mode: " + mode)
                if i == 0:
                    ydata *= 180 / np.pi
                ax[i, j].plot(xdata, ydata)
            ax[i, j].grid(True)
            ax[i, j].set_title(n + " " + coords[i] + " " + loc)
            ax[i, j].set_xlabel(xlabel)
            ax[i, j].set_ylabel("Displacement " + coords[i] + " [" +units[i] + "]")
            ax[i, j].set_xlim([np.min(xdata), np.max(xdata)])
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)

def plot_single_param(data, out, mode, loc):
    kski = sorted(data.keys())
    sims = data[kski[0]].keys()

    coords = ["cir", "rad", "axi"]
    units = ["°", "mm", "mm"]

    # collect data for all parameter variations
    data_sorted = {}
    for j in range(0, 12, 3):
        label = "p_" + str(j) + "_" + loc
        data_sorted[label] = {}
        for nam in sims:
            data_sorted[label][nam] = []
            for k in kski:
                data_sorted[label][nam] += [data[k][nam][label][-1]]
            data_sorted[label][nam] = np.array(data_sorted[label][nam])

    # plot
    fig, ax = plt.subplots(3, len(data_sorted), figsize=(15, len(data_sorted) * 5), sharex="col", sharey="row")
    for j, (lc, dat) in enumerate(data_sorted.items()):
        for mod, dat_k in dat.items():
            for i in range(3):
                xdata = kski
                ydata = dat_k[:, i]
                if i == 0:
                    ydata *= 180 / np.pi
                ax[i, j].plot(xdata, ydata)
                ax[i, j].grid(True)
                ax[i, j].set_title(lc + " " + coords[i])
                ax[i, j].set_xlabel("KsKi")
                ax[i, j].set_ylabel("Displacement " + coords[i] + " [" +units[i] + "]")
                ax[i, j].set_xlim([np.min(xdata), np.max(xdata)])
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)

def main():
    # set study
    # for kski in ["0.5"]:  
    for kski in np.linspace(0.0, 1.0, 5).astype(str):
        folder = "study_aneurysm"
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
        
        plot_disp(data, out)

def main_param():
    # set study
    folder = "study_aneurysm"
    geo = "coarse"
    phi = "0.7"
    
    kski = np.linspace(0.0, 1.0, 5)

    # define paths
    out = os.path.join(folder, geo, "phi_" + phi, "comparison")
    os.makedirs(out, exist_ok=True)

    data = {}
    for k in kski:
        fpath = os.path.join(folder, geo,  "phi_" + phi, "kski_" + str(k))
        inp = {"G&R": os.path.join(fpath, "gr"),
               "FSGe": os.path.join(fpath, "partitioned", "partitioned.json")}
        data[k] = {}
        for n, o in inp.items():
            data[k][n] = post(o)
    
    plot_disp_param(data, out)

if __name__ == "__main__":
    main()
    main_param()

# todo: plot thickness and wss