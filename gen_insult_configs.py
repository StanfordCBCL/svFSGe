#!/usr/bin/env python3
"""Generate diverse FSG insult configs for a SLURM array job.

Each trajectory is a full real-CFD FSG run that differs only in the spatial G&R
insult shape, injected through the `gr_insult` JSON section (consumed by
svfsi.py:set_gr_insult -> GR_equilibrated insult_* XML params -> gr_equilibrated.cpp
eval_insult_profile).

The insult stays centered "around (theta, z) = (0, 0)": the azimuth center is
fixed at pi (theta=0) by the solver, and the axial center z_loc only jitters
slightly around the tube mid-plane (0.5). Diversity comes from varying ALL the
shape levers around the vanilla defaults:

    lever        vanilla   range            meaning
    mag          0.70      [0.45, 0.85]     peak elastin loss fraction (severity)
    z_loc        0.50      [0.45, 0.55]     axial center / length (small jitter)
    z_wid        0.25      [0.15, 0.40]     axial width / length
    z_exp        2.0       [2.0, 4.0]       axial super-Gaussian exponent
    theta_wid    0.55      [0.40, 0.90]     azimuthal width / pi
    theta_exp    6.0       [4.0, 8.0]       azimuthal super-Gaussian exponent

Sample 0 is the exact vanilla insult (reference). Samples 1..N-1 are a
Latin-Hypercube fill of the ranges (reproducible via --seed).
"""
import argparse
import copy
import json
import os

import numpy as np

VANILLA = {
    "profile": "gaussian",
    "mag": 0.70,
    "z_loc": 0.50,
    "z_wid": 0.25,
    "z_exp": 2.0,
    "asym": True,
    "theta_wid": 0.55,
    "theta_exp": 6.0,
}

# (lo, hi) ranges for the LHS-sampled levers
RANGES = {
    "mag": (0.45, 0.85),
    "z_loc": (0.45, 0.55),
    "z_wid": (0.15, 0.40),
    "z_exp": (2.0, 4.0),
    "theta_wid": (0.40, 0.90),
    "theta_exp": (4.0, 8.0),
}


def latin_hypercube(n, d, rng):
    """Centered Latin-Hypercube sample in the unit cube, shape (n, d)."""
    cut = np.linspace(0, 1, n + 1)
    u = (cut[:-1] + cut[1:]) / 2.0
    out = np.empty((n, d))
    for j in range(d):
        out[:, j] = u[rng.permutation(n)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24, help="number of trajectories")
    ap.add_argument("--nloads", type=int, default=10)
    ap.add_argument("--base", default="in_sim/partitioned_full.json")
    ap.add_argument("--out_dir", default="in_sim/insult_array")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    base = json.load(open(os.path.join(here, args.base)))
    out_dir = os.path.join(here, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    keys = list(RANGES.keys())
    # sample 0 = vanilla; remaining n-1 via LHS over the ranges
    lhs = latin_hypercube(args.n - 1, len(keys), rng)

    manifest = []
    for i in range(args.n):
        ins = copy.deepcopy(VANILLA)
        if i > 0:
            for j, k in enumerate(keys):
                lo, hi = RANGES[k]
                ins[k] = float(lo + lhs[i - 1, j] * (hi - lo))
        # round for readable XML
        for k in ("mag", "z_loc", "z_wid", "z_exp", "theta_wid", "theta_exp"):
            ins[k] = round(ins[k], 4)

        cfg = copy.deepcopy(base)
        cfg["nloads"] = args.nloads
        cfg["name"] = "traj%02d" % i
        cfg["gr_insult"] = ins
        cfg.pop("neural_operator", None)  # ensure real CFD

        fn = os.path.join(out_dir, "traj_%02d.json" % i)
        json.dump(cfg, open(fn, "w"), indent=4)
        row = {"traj": "traj%02d" % i, **{k: ins[k] for k in keys}, "mag": ins["mag"]}
        manifest.append(row)

    # manifest for harvest-time mapping run -> insult params
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=2)
    print("wrote %d configs to %s" % (args.n, out_dir))
    hdr = "traj      " + "  ".join("%9s" % k for k in keys)
    print(hdr)
    for r in manifest:
        print("%-8s " % r["traj"] + "  ".join("%9.4f" % r[k] for k in keys))


if __name__ == "__main__":
    main()
