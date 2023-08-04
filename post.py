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

def get_results(results, pts, pids, lids):
    # get displacements in radial coordinates at all extracted points
    disp = defaultdict(list)
    for res in results:
        extract_results(disp, res, pts, pids, "p")
    extract_results(disp, results[-1], pts, lids, "l")
    for loc in disp.keys():
        disp[loc] = np.array(disp[loc])
    return disp

def extract_results(disp, res, pts, ids, mode):
    # get nodal displacements
    d = v2n(res.GetPointData().GetArray("Displacement"))

    # get nodal wall shear stress
    if res.GetPointData().HasArray("WSS"):
        wss = v2n(res.GetPointData().GetArray("WSS"))
    else:
        q0 = 1000
        mu = 4.0e-06
        rad = xyz2cra((pts + d).T)[1]
        wss = 4.0 * mu * q0 / np.pi / rad ** 3.0
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
    plot_single(data, os.path.join(out, "disp_points.png"), study, "p", "disp", "out_mid")
    for res in ["thick", "wss"]:
        plot_single(data, os.path.join(out, res + "_points.png"), study, "p", res, "mid")
    if study == "single":
        plot_single(data, os.path.join(out, "disp_lines.png"), study, "l", "disp", "out")
        for res in ["thick", "wss"]:
            plot_single(data, os.path.join(out, res + "_lines.png"), study, "l", res)

def plot_single(data, out, study, mode, quant, loc=""):
    # plot text
    if quant == "disp":
        coords = ["cir", "rad", "axi"]
        units = ["°", "mm", "mm"]
        ylabel = "Displacement"
    elif quant == "thick":
        coords = [""]
        units = ["mm"]
        ylabel = "Thickness"
    elif quant == "wss":
        coords = [""]
        units = ["?"]
        ylabel = "WSS"
    else:
        raise RuntimeError("Unknown quantity: " + quant)

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
    ny = len(coords)

    # plot properties
    oclocks = range(0, 12, 3)
    colors = [ '#4477AA', '#EE6677', '#CCBB44', '#228833', '#66CCEE', '#AA3377', '#BBBBBB']
    styles = ["-", "-", "-", ":"]
    styles_cir = ["-", "-", ":", "-"]

    fig, ax = plt.subplots(ny, nx, figsize=(nx * 10, ny * 5), dpi=300, sharex="col", sharey="row")
    for j, (n, res) in enumerate(data.items()):
        for i in range(ny):
            if ny == 1:
                pos = j
            else:
                pos = (i, j)
            for ik, k in enumerate(oclocks):
                # get data for y-axis
                xres = "l_z_" +  "_".join(filter(None, [str(k), loc]))
                yres =  "_".join(filter(None, [mode, quant, str(k), loc]))
                title = " ".join(filter(None, [n, ylabel, loc, coords[i]]))
                ydata = res[yres]
                assert len(ydata) > 0, "no data found: " + yres
                if quant == "disp":
                    ydata = ydata.T[i]

                # get data for x-axis
                if study == "single":
                    if mode == "p":
                        xdata = np.arange(0, len(ydata))
                        xlabel = "Load step [-]"
                    elif mode == "l":
                        xdata = res[xres]
                        xlabel = "Vessel length [mm]"
                    else:
                        raise RuntimeError("Unknown mode: " + mode)
                else:
                    xdata = params
                    xlabel = study
                assert len(xdata) > 0, "no data found: " + xres

                # convert to degrees
                if i == 0:
                    ydata *= 180 / np.pi

                # plot!
                if quant == "disp" and i == 0:
                    stl = styles_cir[ik]
                else:
                    stl = styles[ik]
                ax[pos].plot(xdata, ydata, stl, color=colors[ik], linewidth=2)
            ax[pos].grid(True)
            ax[pos].set_title(title)
            ax[pos].set_xlabel(xlabel)
            ax[pos].set_ylabel(ylabel + " " + coords[i] + " [" +units[i] + "]")
            ax[pos].set_xlim([np.min(xdata), np.max(xdata)])
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
        
        plot_disp(data, out, "single")

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
    
    plot_disp(data, out, "kski")

if __name__ == "__main__":
    main()
    main_param()
