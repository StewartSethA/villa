#!/usr/bin/env python3
"""Do two overlapping spiral-fit windows agree where they overlap?

A whole scroll is fit as z-windows (a single 13,000-slice fit ran out of memory on a 32 GB V100) with a
shared overlap band. Inside that band both fits place a surface for every winding index.
If the windows are one consistent unwrapping, winding w of window A lies on winding w of
window B. If a window's numbering drifted, A's w matches B's w+k for some k != 0.

Method: for each winding, points of its lattice with z in the overlap are binned by angle
about the umbilicus (at their own z) into 72 bins; the per-bin median radius is the
winding's radial profile. For every offset k in [-5, 5], the median over windings and bins
of |r_A(w) - r_B(w+k)| is reported, in L0 voxels and in lamina pitches (20.5 voxels,
the default is the value measured on PHerc0211 at 9.362 um; pass --pitch for another scroll). The best k and its margin over the
runner-up are the verdict: k = 0 with a residual well under one pitch = consistent.

usage: window_seams.py MESH_A MESH_B UMBILICUS.json Z0 Z1 OUT.json [--pitch 20.5]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import tifffile

NBIN = 72


def profiles(meshdir: Path, umb, z0, z1):
    cps = sorted(json.loads(Path(umb).read_text())["control_points"], key=lambda p: p["z"])
    uz = np.array([p["z"] for p in cps], float)
    ux = np.array([p["x"] for p in cps], float)
    uy = np.array([p["y"] for p in cps], float)
    out = {}
    for d in sorted(meshdir.glob("w*_spliced_*")):
        m = re.match(r"w(\d{3})", d.name)
        X, Y, Z = (tifffile.imread(d / f"{c}.tif").astype(np.float64) for c in "xyz")
        V = (X != -1) & (Z >= z0) & (Z < z1)
        if V.sum() < 50:
            continue
        x, y, z = X[V], Y[V], Z[V]
        cx, cy = np.interp(z, uz, ux), np.interp(z, uz, uy)
        th = np.arctan2(y - cy, x - cx)
        r = np.hypot(x - cx, y - cy)
        b = ((th + np.pi) / (2 * np.pi) * NBIN).astype(int) % NBIN
        prof = np.full(NBIN, np.nan)
        for i in range(NBIN):
            s = b == i
            if s.sum() >= 3:
                prof[i] = np.median(r[s])
        out[int(m.group(1))] = prof
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("umb")
    ap.add_argument("z0", type=float)
    ap.add_argument("z1", type=float)
    ap.add_argument("out", type=Path)
    ap.add_argument("--pitch", type=float, default=20.5)
    a = ap.parse_args(argv)
    A = profiles(a.a, a.umb, a.z0, a.z1)
    B = profiles(a.b, a.umb, a.z0, a.z1)
    res = {}
    for k in range(-5, 6):
        diffs, n = [], 0
        for w, pa in A.items():
            pb = B.get(w + k)
            if pb is None:
                continue
            dd = np.abs(pa - pb)
            dd = dd[np.isfinite(dd)]
            if dd.size:
                diffs.append(dd)
                n += 1
        if diffs:
            allv = np.concatenate(diffs)
            res[k] = {"n_windings": n, "n_bins": int(allv.size), "med_vox": float(np.median(allv)),
                      "p90_vox": float(np.percentile(allv, 90)),
                      "med_pitch": float(np.median(allv) / a.pitch)}
    best = min(res, key=lambda k: res[k]["med_vox"])
    others = sorted(v["med_vox"] for k, v in res.items() if k != best)
    # per-winding residual at the best offset: where do the windows disagree?
    perw = {}
    for w, pa in A.items():
        pb = B.get(w + best)
        if pb is not None:
            dd = np.abs(pa - pb)
            dd = dd[np.isfinite(dd)]
            if dd.size:
                perw[w] = float(np.median(dd))
    rep = {"a": str(a.a), "b": str(a.b), "overlap_z": [a.z0, a.z1], "pitch_vox": a.pitch,
           "windings_a": len(A), "windings_b": len(B), "by_offset": res, "best_offset": best,
           "best_med_vox": res[best]["med_vox"], "runner_up_med_vox": others[0] if others else None,
           "per_winding_med_vox_at_best": perw}
    a.out.write_text(json.dumps(rep, indent=1))
    print(f"overlap z[{a.z0:.0f},{a.z1:.0f}): windings A {len(A)} B {len(B)}; best offset k={best} "
          f"median |dr| {res[best]['med_vox']:.1f} vox ({res[best]['med_pitch']:.2f} pitch), p90 "
          f"{res[best]['p90_vox']:.1f}; runner-up {others[0] if others else float('nan'):.1f} vox; "
          f"k=0: {res.get(0, {}).get('med_vox', float('nan')):.1f} vox")


if __name__ == "__main__":
    main()
