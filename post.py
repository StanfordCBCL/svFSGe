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

def get_points(pts_xyz):
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
    
    # collect all point ids
    ids = {}
    for n, pt in points.items():
        chk = [np.isclose(pts_xyz[:, i], pt[i]) for i in range(3)]
        id = np.where(np.logical_and.reduce(np.array(chk)))[0]
        assert len(id) == 1, "point definition not unique: " + str(pt)
        ids[n] = id[0]
    return ids

# def get_lines(pts_xyz):
    # # collect all line coordinates
    # lines = {}
    # for cn, cp in p_cir.items():
    #     for rn, rp in p_rad.items():

    # pdb.set_trace()

def get_disp(results, pts, ids):
    # get displacements in radial coordinates at all extracted points
    disp = defaultdict(list)
    for res in results:
        d = v2n(res.GetPointData().GetArray("Displacement"))
        for n, pt in ids.items():
            disp[n] += [xyz2cra(pts[pt] + d[pt]) - xyz2cra(pts[pt])]
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

    # get point ids
    ids = get_points(pts)

    # extract displacements
    return get_disp(res, pts, ids)

def plot_disp(data, out):
    coords = ["cir", "rad", "axi"]
    fig, ax = plt.subplots(3, 2, figsize=(20, 10), sharex="col", sharey="row")
    for j, (n, d) in enumerate(data.items()):
        for i in range(3):
            ax[i, j].plot(d["0_out_mid"][:, i])
            ax[i, j].plot(d["3_out_mid"][:, i])
            ax[i, j].plot(d["6_out_mid"][:, i])
            ax[i, j].plot(d["9_out_mid"][:, i])
            ax[i, j].grid(True)
            ax[i, j].set_title(n + " " + coords[i])
    fig.savefig(os.path.join(out, "displacement.png"))
    plt.close(fig)

# def main_arg():
#     parser = argparse.ArgumentParser(description="Post-process FSGe simulation")
#     parser.add_argument("out1", help="svFSI output folder")
#     parser.add_argument("out2", help="fsg.py output folder")
#     args = parser.parse_args()
#     plot_disp(args.out1, args.out2)

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
