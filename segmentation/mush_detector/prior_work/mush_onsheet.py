#!/usr/bin/env python3
"""Task 3(a): is on-sheet lift lower INSIDE mush than OUTSIDE, for the one Route B
window whose z-range (4500-7300) overlaps the mush-labelled z-range (2000-10000)?

Reuses onsheet.py's on_sheet/null-shift definition and the SAME prediction zarr,
but groups sampled points by mush-mask containment instead of by winding, over
ALL points of ALL 115 windings in the production a0125_z4500_acwup fit (not a
3000-point-per-winding subsample -- each winding's own grid is only ~8000 points,
so reading everything is cheap).
"""
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
import zarr

sys.path.insert(0, "/home/seth/ScrollPrizeTutorial/scripts/routeB")
from onsheet import normals  # noqa: E402

MESH_DIR = Path(
    "/mnt/raid7/experiments/routeB_full/PHerc0125/a0125_z4500_acwup/fit/meshes/fitted_a0125_z4500_acwup"
)
PRED = Path("/mnt/raid7/surface_predictions/PHerc0125/20250821151825-surface-20260413222639-surface-m7-L0-th0.2.zarr")
MASK = Path("/mnt/raid7/experiments/mush_labels/PHerc0125/mush_mask.zarr")
SHIFT = 10.0
LEVEL = 1
TOL = 1


def main():
    pred = zarr.open(str(PRED / str(LEVEL)), mode="r")
    F = 2 ** LEVEL
    cz, cy, cx = pred.chunks

    mush = zarr.open(str(MASK), mode="r")
    blob = mush["blob"]
    mush_level = int(mush.attrs["mush_level"])
    mf = 2 ** mush_level
    z_lo, z_hi = mush.attrs["z_valid_range_level0"]

    windings = sorted(d for d in MESH_DIR.glob("w*_spliced_*") if (d / "x.tif").exists())
    print(f"{len(windings)} windings", flush=True)

    all_xyz = []
    for d in windings:
        X, Y, Z = (tifffile.imread(d / f"{c}.tif").astype(np.float32) for c in "xyz")
        V = (X != -1) & (Z > 0) & np.isfinite(X) & (Z >= z_lo) & (Z <= z_hi)
        if not V.any():
            continue
        N = normals(X, Y, Z)
        P = np.stack([X, Y, Z], -1).astype(np.float64)
        all_xyz.append((P[V], N[V], d.name))

    if not all_xyz:
        print("NO POINTS in mush z-range for this window"); return

    P = np.concatenate([a for a, _, _ in all_xyz], axis=0)
    N = np.concatenate([a for _, a, _ in all_xyz], axis=0)
    wind_id = np.concatenate([np.full(len(a), i) for i, (a, _, _) in enumerate(all_xyz)])
    names = [n for _, _, n in all_xyz]
    print(f"{len(P)} total points across {len(all_xyz)} windings with z in [{z_lo},{z_hi}]", flush=True)

    # mush containment per point (vectorised: mask coords = round(xyz / mf))
    mz = np.round(P[:, 2] / mf).astype(np.int64)
    my = np.round(P[:, 1] / mf).astype(np.int64)
    mx = np.round(P[:, 0] / mf).astype(np.int64)
    Zb, Yb, Xb = blob.shape
    mok = (mz >= 0) & (mz < Zb) & (my >= 0) & (my < Yb) & (mx >= 0) & (mx < Xb)
    blobval = np.zeros(len(P), np.uint8)
    # group by z-plane to minimise chunk reads
    order = np.argsort(mz[mok])
    idx_ok = np.flatnonzero(mok)[order]
    for z0 in np.unique(mz[idx_ok]):
        sel = idx_ok[mz[idx_ok] == z0]
        plane = np.asarray(blob[z0])
        blobval[sel] = plane[my[sel], mx[sel]]
    inside = mok & (blobval >= 128)
    outside = mok & (blobval < 128)
    print(f"inside-mush points: {inside.sum()}  outside-mush points: {outside.sum()}  "
          f"windings touching mush: {len(np.unique(wind_id[inside]))}", flush=True)

    def sample_onsheet(sel):
        Ps, Ns = P[sel], N[sel]

        def hit(Q):
            zyx = np.round(Q[:, ::-1] / F).astype(np.int64)
            offs = np.array([(dz, dy, dx) for dz in range(-TOL, TOL + 1)
                              for dy in range(-TOL, TOL + 1) for dx in range(-TOL, TOL + 1)])
            q = (zyx[:, None, :] + offs[None]).reshape(-1, 3)
            shp = np.array(pred.shape)
            ok = np.all((q >= 0) & (q < shp), axis=1)
            vals = np.zeros(len(q), np.uint8)
            ck = (q[:, 0] // cz) * 1_000_000 + (q[:, 1] // cy) * 1000 + q[:, 2] // cx
            o = np.argsort(ck[ok], kind="stable")
            qi = np.flatnonzero(ok)[o]
            cks = ck[qi]
            bounds = np.flatnonzero(np.diff(cks)) + 1
            for grp in np.split(qi, bounds):
                if grp.size == 0:
                    continue
                z0, y0, x0 = (q[grp[0]] // (cz, cy, cx)) * (cz, cy, cx)
                blk = np.asarray(pred[z0:z0 + cz, y0:y0 + cy, x0:x0 + cx])
                loc = q[grp] - (z0, y0, x0)
                vals[grp] = blk[loc[:, 0], loc[:, 1], loc[:, 2]]
            return (vals.reshape(len(Q), len(offs)) > 0).any(1)

        on = hit(Ps).astype(np.float64)
        null_out = hit(Ps + SHIFT * Ns).astype(np.float64)
        null_in = hit(Ps - SHIFT * Ns).astype(np.float64)
        null = 0.5 * (null_out + null_in)
        return on, null

    def boot(on, null, nboot=2000, rng=None):
        rng = rng or np.random.default_rng(0)
        n = len(on)
        lifts = []
        for _ in range(nboot):
            idx = rng.integers(0, n, n)
            lifts.append(on[idx].mean() - null[idx].mean())
        lifts = np.array(lifts)
        return float(np.percentile(lifts, 2.5)), float(np.percentile(lifts, 97.5))

    out = {"window": "a0125_z4500_acwup", "z_overlap_level0": [int(min(P[inside | outside, 2])) if (inside | outside).any() else None,
                                                                 int(max(P[inside | outside, 2])) if (inside | outside).any() else None]}
    for label, sel in (("inside_mush", inside), ("outside_mush", outside)):
        n = int(sel.sum())
        if n == 0:
            out[label] = {"n": 0}
            continue
        on, null = sample_onsheet(sel)
        lift = on.mean() - null.mean()
        lo, hi = boot(on, null)
        out[label] = {"n": n, "on_sheet": float(on.mean()), "null_shift": float(null.mean()),
                      "lift": float(lift), "lift_ci95": [lo, hi],
                      "n_windings": int(len(np.unique(wind_id[sel])))}
        print(label, out[label], flush=True)

    Path("/home/seth/ScrollPrizeTutorial/docs/experiments/mush_labels/PHerc0125_onsheet_mush_vs_outside.json").write_text(
        json.dumps(out, indent=1))
    print("DONE")


if __name__ == "__main__":
    main()
