#!/usr/bin/env python3
"""Snap a grown tifxyz onto the papyrus sheet it lies nearest to.

Measured 2026-09-05 against published Scroll 5 meshes: our grown lattices sit a median
2.8-12 voxels from the labelled sheet, and the ink model tolerates only about +-4. At the
worst patch the CT under our lattice reads a median of 50 (air) with 39 % of points on
material. Each lattice point gets the CT profile along its normal (-R..+R voxels); the
nearest run of material (above an Otsu threshold of the local block) is found and the
point moves to that run's centre. Offsets are median-filtered over the lattice so the
surface stays coherent, and capped at R. Points with no material within R stay put.

    sheet_snap.py <tifxyz_in> <volume.zarr> <tifxyz_out> [--radius 10] [--min-run 2]
Prints the offset statistics; writes meta.json with a `sheet_snap` record.
"""
import argparse, json, os, shutil
import numpy as np


def lattice_normals(P, valid):
    dc = P[:, 1:] - P[:, :-1]; dr = P[1:] - P[:-1]
    n = np.zeros_like(P)
    nn = np.cross(dc[:-1], dr[:, :-1]); nn /= np.linalg.norm(nn, axis=-1, keepdims=True) + 1e-9
    n[:-1, :-1] = nn; n[-1] = n[-2]; n[:, -1] = n[:, -2]
    # smooth the normals over 3x3 so a noisy lattice does not scatter the profiles
    from scipy.ndimage import uniform_filter
    for k in range(3):
        n[..., k] = uniform_filter(n[..., k], size=3, mode="nearest")
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    n[~valid] = 0
    return n


def material_threshold(v):
    """Midpoint between the two intensity modes (air/gap ~20-50, papyrus ~85-135 on Scroll 5)
    by two-means on a sample; Otsu on the raw histogram was dragged to 249 by the bright
    inclusions in the volume."""
    v = v.astype(np.float64); lo, hi = np.percentile(v, 25), np.percentile(v, 85)
    for _ in range(20):
        mid = (lo + hi) / 2
        a = v[v < mid]; b = v[v >= mid]
        if not len(a) or not len(b):
            break
        lo, hi = a.mean(), b.mean()
    return float((lo + hi) / 2)


def snap(P, valid, vol, radius=10, min_run=2, thresh=None, target="centre", axis_xy=None):
    """target: 'centre' of the nearest material run, or 'recto' = its face toward the scroll
    axis (the published meshes sit on the recto face, not the sheet centre)."""
    import zarr
    n = lattice_normals(P, valid)
    pts = P[valid]; nrm = n[valid]
    arr = zarr.open_array(os.path.join(vol, "0"), mode="r")
    shape_xyz = np.array(arr.shape)[::-1]
    offs = np.arange(-radius, radius + 1)
    # sample only the voxels on the profiles (points x offsets), chunk-wise through zarr's
    # point selection: loading the lattice's whole bounding box took gigabytes and minutes
    # on a curved 30x30 mm patch (26 GB RSS, 2026-09-05)
    Q = pts[:, None, :] + offs[None, :, None] * nrm[:, None, :]
    Q = np.clip(np.rint(Q).astype(int), 0, shape_xyz - 1).reshape(-1, 3)
    vals = arr.vindex[Q[:, 2], Q[:, 1], Q[:, 0]]
    prof = np.asarray(vals, np.float32).reshape(len(pts), len(offs))
    # the material threshold from the profile samples themselves (a fair sample of sheet and gap)
    sub = prof
    if thresh is None:
        thresh = material_threshold(sub[sub > 0].ravel()) if (sub > 0).any() else 60.0
    mat = prof > thresh
    move = np.zeros(len(pts)); found = np.zeros(len(pts), bool)
    c0 = radius                                   # index of offset 0
    # sign of the normal toward the axis, per point (recto face selection)
    if axis_xy is not None:
        to_axis = np.stack([axis_xy[0] - pts[:, 0], axis_xy[1] - pts[:, 1], np.zeros(len(pts))], -1)
        inward = np.sign((nrm * to_axis).sum(1))          # +1: +normal points toward the axis
    else:
        inward = np.ones(len(pts))
    for k in range(len(pts)):
        row = mat[k]
        if not row.any():
            continue
        # runs of material along the profile
        d = np.diff(np.concatenate([[0], row.astype(int), [0]])); starts = np.flatnonzero(d == 1); ends = np.flatnonzero(d == -1)
        best = None
        for s0, e0 in zip(starts, ends):
            if e0 - s0 < min_run:
                continue
            centre = (s0 + e0 - 1) / 2.0
            dist = 0.0 if s0 <= c0 < e0 else min(abs(s0 - c0), abs(e0 - 1 - c0))
            if best is None or dist < best[0]:
                best = (dist, centre)
        if best is not None:
            if target == "recto":
                # the run's face on the axis side: its end along +normal if +normal is inward, else its start
                s0, e0 = [(s_, e_) for s_, e_ in zip(starts, ends) if (s_ + e_ - 1) / 2.0 == best[1]][0]
                face = (e0 - 1) if inward[k] > 0 else s0
                move[k] = face - c0
            else:
                move[k] = best[1] - c0
            found[k] = True
    # coherent surface: median-filter the offsets over the lattice, then cap
    from scipy.ndimage import median_filter
    M = np.zeros(valid.shape); M[valid] = move
    F = np.zeros(valid.shape, bool); F[valid] = found
    Mf = median_filter(M, size=5, mode="nearest")
    Mf = np.clip(Mf, -radius, radius)
    Q = P.copy(); Q[valid] += Mf[valid][:, None] * n[valid]
    stats = {"threshold": float(thresh), "points": int(valid.sum()), "found_frac": float(found.mean()),
             "raw_offset_median_abs": float(np.median(np.abs(move[found]))) if found.any() else None,
             "applied_offset_median": float(np.median(Mf[valid])), "applied_offset_median_abs": float(np.median(np.abs(Mf[valid]))),
             "applied_offset_p90_abs": float(np.percentile(np.abs(Mf[valid]), 90)),
             "on_material_before": float((prof[:, c0] > thresh).mean()), "radius": radius}
    return Q, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("vol"); ap.add_argument("out")
    ap.add_argument("--radius", type=int, default=10); ap.add_argument("--min-run", type=int, default=2); ap.add_argument("--thresh", type=float, default=None)
    ap.add_argument("--target", choices=["centre", "recto"], default="centre")
    a = ap.parse_args()
    axis = None
    try:
        m = json.load(open(os.path.join(a.vol, "meta.json"))); axis = (float(m["width"]) / 2, float(m["height"]) / 2)
    except (OSError, ValueError, KeyError):
        pass
    import tifffile
    P = np.stack([tifffile.imread(os.path.join(a.src, f"{c}.tif")).astype(np.float64) for c in "xyz"], -1)
    valid = P[..., 0] > 0
    Q, st = snap(P, valid, a.vol, a.radius, a.min_run, a.thresh, target=a.target, axis_xy=axis)
    st["target"] = a.target
    os.makedirs(a.out, exist_ok=True)
    for k, c in enumerate("xyz"):
        A = Q[..., k].astype(np.float32); A[~valid] = -1
        tifffile.imwrite(os.path.join(a.out, f"{c}.tif"), A)
    meta = {}
    try:
        meta = json.load(open(os.path.join(a.src, "meta.json")))
    except (OSError, ValueError):
        pass
    meta["sheet_snap"] = st; meta["snapped_from"] = os.path.abspath(a.src)
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(json.dumps(st))


if __name__ == "__main__":
    main()
