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

        # sort according to z-coordinate
        zs = pts_xyz[id, 2]
        lids[n] = id[np.argsort(zs)]
    return pids, lids

def get_disp(results, pts, pids, lids):
    # get displacements in radial coordinates at all extracted points
    disp = defaultdict(list)
    for res in results:
        d = v2n(res.GetPointData().GetArray("Displacement"))
        for n, pt in pids.items():
            diff = xyz2cra(pts[pt] + d[pt]) - xyz2cra(pts[pt])
            if diff[0] < -np.pi:
                diff[0] += 2.0 * np.pi
            disp["p_" + n] += [diff]
    else:
        d = v2n(res.GetPointData().GetArray("Displacement"))
        for n, ln in lids.items():
            ref = []
            ds = []
            for l in ln:
                diff = xyz2cra(pts[l] + d[l]) - xyz2cra(pts[l])
                if diff[0] < -np.pi:
                    diff[0] += 2.0 * np.pi
                ref += [pts[l][2]]
                ds += [diff]
            disp["l_" + n] = np.array(ds)
            disp["lr_" + n] = np.array(ref)
    
    # convert to numpy arrays
    for n in disp.keys():
        disp[n] = np.array(disp[n])
    return disp

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
    return get_disp(res, pts, pids, lids)

def plot_disp(data, out):
    plot(data, os.path.join(out, "displacement_points.png"), "pt", "out_mid")
    plot(data, os.path.join(out, "displacement_lines.png"), "ln", "out")

def plot(data, out, mode, loc):
    coords = ["cir", "rad", "axi"]
    units = ["°", "mm", "mm"]

    fig, ax = plt.subplots(3, len(data), figsize=(15, len(data) * 5), sharex="col", sharey="row")
    for j, (n, d) in enumerate(data.items()):
        for i in range(3):
            for k in range(0, 12, 3):
                label = str(k) + "_" + loc
                if mode == "pt":
                    ydata = d["p_" + label][:, i]
                    xdata = np.arange(0, len(ydata))
                    xlabel = "Load step [-]"
                elif mode == "ln":
                    ydata = d["l_" + label][:, i]
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

def main():
    # define paths
    inp = {"G&R": "study_aneurysm/kski_1.0_phi_0.7/coarse/gr/",
           "FSGe": "study_aneurysm/kski_1.0_phi_0.7/coarse/partitioned/partitioned.json"}
    out = "study_aneurysm/kski_1.0_phi_0.7/coarse/comparison"

    # create output folder
    os.makedirs(out, exist_ok=True)

    # collect all results
    data = {}
    for n, o in inp.items():
        data[n] = post(o)
    
    plot_disp(data, out)

if __name__ == "__main__":
    main()
