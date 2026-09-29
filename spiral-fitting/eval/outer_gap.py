#!/usr/bin/env python3
"""Does a spiral-fit window reach the OUTSIDE of the scroll, or is it truncated by
`shell_outer_winding_idx`?

At one z slice: for every angle bin about the umbilicus, the outermost radius of
(a) any fitted winding lattice point within +-dz of the slice and (b) any surface-prediction
positive voxel on that slice (level L, upsampled to L0 units). If (b) - (a) is many winding
pitches on most bins, the fit stops short of the papyrus and windings are MISSING.

Also counts prediction-positive "sheet crossings" outside the fit's envelope along each ray
(runs of positives separated by >= 2 L-voxels of background), i.e. how many windings the fit
would need to add there.

usage: outer_gap.py MESHDIR PRED_ZARR UMBILICUS_JSON Z --voxel-um UM [--level 2] [--dz 20] [--bins 72] [--json out]

MESHDIR holds fit_spiral.py's per-winding tifxyz directories (`w*_spliced_*`); PRED_ZARR is an
OME-Zarr group whose array '<level>' is the surface prediction; UMBILICUS_JSON has
`control_points` [{z, x, y}] in level-0 voxels.
Read-only. No set -e analogue needed: one slice, one answer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
import zarr


def umb_at(umb: dict, z: float):
    pts = sorted(umb["control_points"], key=lambda p: p["z"])
    zz = np.array([p["z"] for p in pts], float)
    return float(np.interp(z, zz, [p["x"] for p in pts])), float(np.interp(z, zz, [p["y"] for p in pts]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("meshdir", type=Path)
    ap.add_argument("pred", type=Path)
    ap.add_argument("umb", type=Path)
    ap.add_argument("z", type=int)
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--dz", type=float, default=20.0)
    ap.add_argument("--bins", type=int, default=72)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--voxel-um", type=float, required=True, help="level-0 voxel size of the volume (um), for the mm columns")
    ap.add_argument("--ct", type=Path, help="CT zarr: use CT > --ct-th as the papyrus body instead of the "
                    "prediction (the prediction carries a blocky chunk-edge border artifact outside the scroll)")
    ap.add_argument("--ct-th", type=float, default=0.0, help="CT threshold; 0 = Otsu-free default: 0.5 x p99 of the slice")
    a = ap.parse_args(argv)
    ux, uy = umb_at(json.loads(a.umb.read_text()), a.z)
    s = 2 ** a.level
    arr = zarr.open(str(a.pred), mode="r")[str(a.level)]
    sl = np.asarray(arr[a.z // s]) > 0                     # (y, x) at level L
    if a.ct:
        c = np.asarray(zarr.open(str(a.ct), mode="r")[str(a.level)][a.z // s]).astype(np.float32)
        th = a.ct_th or 0.5 * float(np.percentile(c[c > 0], 99)) if (c > 0).any() else 1.0
        sl = c > th
    yy, xx = np.nonzero(sl)
    px, py = xx * s + s / 2.0, yy * s + s / 2.0            # L0 units
    ang_p = np.arctan2(py - uy, px - ux)
    rad_p = np.hypot(px - ux, py - uy)
    B = a.bins
    bp = ((ang_p + np.pi) / (2 * np.pi) * B).astype(int) % B
    # fitted winding lattice points near the slice
    rmax_b = np.full(B, np.nan)
    nwind = 0
    wmax = {}
    for d in sorted(a.meshdir.glob("w*_spliced_*")):
        try:
            z = tifffile.imread(d / "z.tif")
        except Exception as e:                               # noqa: BLE001
            print(f"skip {d.name}: {e}")
            continue
        m = (np.abs(z - a.z) <= a.dz) & (z > 0)
        if not m.any():
            continue
        x = tifffile.imread(d / "x.tif")[m]
        y = tifffile.imread(d / "y.tif")[m]
        nwind += 1
        ang = np.arctan2(y - uy, x - ux)
        rad = np.hypot(x - ux, y - uy)
        b = ((ang + np.pi) / (2 * np.pi) * B).astype(int) % B
        for k in range(B):
            sel = b == k
            if sel.any():
                r = float(rad[sel].max())
                if np.isnan(rmax_b[k]) or r > rmax_b[k]:
                    rmax_b[k] = r
                    wmax[k] = d.name.split("_")[0]
    rows = []
    for k in range(B):
        sel = bp == k
        rp = float(rad_p[sel].max()) if sel.any() else float("nan")
        # sheet crossings outside the fit envelope along this bin: count runs in a radial profile
        n_out = 0
        if sel.any() and not np.isnan(rmax_b[k]):
            r_out = np.sort(rad_p[sel & (rad_p > rmax_b[k] + 2 * s)])
            if r_out.size:
                n_out = int(1 + np.sum(np.diff(r_out) > 2 * s))
        rows.append({"bin": k, "r_pred_max_L0": rp, "r_fit_max_L0": None if np.isnan(rmax_b[k]) else float(rmax_b[k]),
                     "outer_winding": wmax.get(k), "pred_runs_outside_fit": n_out})
    gap = np.array([r["r_pred_max_L0"] - r["r_fit_max_L0"] for r in rows
                    if r["r_fit_max_L0"] is not None and not np.isnan(r["r_pred_max_L0"])])
    runs = np.array([r["pred_runs_outside_fit"] for r in rows if r["r_fit_max_L0"] is not None])
    out = {"meshdir": str(a.meshdir), "pred": str(a.pred), "umbilicus": str(a.umb), "z": a.z, "level": a.level,
           "umb_xy_L0": [ux, uy], "windings_on_slice": nwind, "bins": B,
           "gap_L0_p10_p50_p90": [float(np.percentile(gap, q)) for q in (10, 50, 90)] if gap.size else None,
           "gap_mm_p10_p50_p90": [float(np.percentile(gap, q)) * a.voxel_um * 1e-3 for q in (10, 50, 90)] if gap.size else None,
           "pred_runs_outside_fit_p10_p50_p90": [float(np.percentile(runs, q)) for q in (10, 50, 90)] if runs.size else None,
           "bins_with_gap_gt_5mm": int(np.sum(gap * a.voxel_um * 1e-3 > 5.0)) if gap.size else None,
           "rows": rows}
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
