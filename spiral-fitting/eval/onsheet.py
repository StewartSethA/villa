#!/usr/bin/env python3
"""Per-winding ON-SHEET fraction of a spiral fit (fit_spiral.py meshes), with a displaced-surface null.

WHY: `material_frac` is saturated on spiral-fit windings (measured 0.9997 against a
bbox_occupancy control of 0.987 on PHerc0211) because a winding inside the scroll body is on CT > 5
whether it follows a sheet or the gap between two. This asks the sharper question --
is the fitted surface ON a predicted sheet -- and carries its own negative control:

  on_sheet    fraction of sampled lattice points with surface-prediction > 0 within
              +-TOL level-1 voxels (a 3x3x3 max at L1 = +-2 L0 voxels)
  null_shift  the SAME points displaced along the local surface normal by half the
              lamina pitch (default 10 L0 voxels: half the ~20.5-voxel pitch measured on PHerc0211 at 9.362 um;
              set --shift to half the pitch of your scroll), i.e. into
              the inter-sheet gap. A fit that tracks sheets scores on_sheet >> null_shift;
              a fit sitting at random radii scores them equal.
  lift        on_sheet - null_shift (the evidence), reported per winding.

Both are computed from the same chunk reads, so the pair fails the same way if the
prediction store is wrong. Seeded; `--n` points per winding.

usage: onsheet.py MESHDIR PRED_ZARR OUT.jsonl [--n 3000] [--shift 10] [--glob 'w*_spliced_*']
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile
import zarr


def normals(X, Y, Z):
    P = np.stack([X, Y, Z], -1).astype(np.float64)
    tr = np.gradient(P, axis=0)
    tc = np.gradient(P, axis=1)
    n = np.cross(tr, tc)
    ln = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / np.maximum(ln, 1e-9)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("meshdir", type=Path)
    ap.add_argument("pred", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--shift", type=float, default=10.0, help="null displacement, L0 voxels")
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--tol", type=int, default=1, help="+- voxels at --level")
    ap.add_argument("--glob", default="w*_spliced_*")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    arr = zarr.open(str(a.pred / str(a.level)), mode="r")
    F = 2 ** a.level
    cz, cy, cx = arr.chunks
    rng = np.random.default_rng(a.seed)
    samples = []            # (winding, kind, zyx L-level int array)
    names = []
    for d in sorted(a.meshdir.glob(a.glob)):
        if not (d / "x.tif").exists():
            continue
        X, Y, Z = (tifffile.imread(d / f"{c}.tif").astype(np.float32) for c in "xyz")
        V = (X != -1) & (Z > 0) & np.isfinite(X)
        idx = np.flatnonzero(V.ravel())
        if idx.size == 0:
            continue
        pick = rng.choice(idx, size=min(a.n, idx.size), replace=False)
        N = normals(X, Y, Z).reshape(-1, 3)[pick]
        P = np.stack([X.ravel()[pick], Y.ravel()[pick], Z.ravel()[pick]], -1).astype(np.float64)
        for kind, Q in (("on", P), ("null_out", P + a.shift * N), ("null_in", P - a.shift * N)):
            zyx = np.round(Q[:, ::-1] / F).astype(np.int64)
            samples.append((len(names), kind, zyx))
        names.append(d.name)
    # every (point, offset) query, grouped by chunk so each chunk is read once
    offs = np.array([(dz, dy, dx) for dz in range(-a.tol, a.tol + 1)
                     for dy in range(-a.tol, a.tol + 1) for dx in range(-a.tol, a.tol + 1)])
    allq = np.concatenate([s[2] for s in samples])
    q = (allq[:, None, :] + offs[None]).reshape(-1, 3)
    shp = np.array(arr.shape)
    ok = np.all((q >= 0) & (q < shp), axis=1)
    vals = np.zeros(len(q), np.uint8)
    ck = (q[:, 0] // cz) * 1_000_000 + (q[:, 1] // cy) * 1000 + q[:, 2] // cx
    order = np.argsort(ck[ok], kind="stable")
    qi = np.flatnonzero(ok)[order]
    cks = ck[qi]
    bounds = np.flatnonzero(np.diff(cks)) + 1
    for grp in np.split(qi, bounds):
        if grp.size == 0:
            continue
        z0, y0, x0 = (q[grp[0]] // (cz, cy, cx)) * (cz, cy, cx)
        blk = np.asarray(arr[z0:z0 + cz, y0:y0 + cy, x0:x0 + cx])
        loc = q[grp] - (z0, y0, x0)
        vals[grp] = blk[loc[:, 0], loc[:, 1], loc[:, 2]]
    hit = (vals.reshape(len(allq), len(offs)) > 0).any(1)
    res = {}
    k = 0
    for wi, kind, zyx in samples:
        res.setdefault(wi, {})[kind] = float(hit[k:k + len(zyx)].mean())
        res[wi]["n"] = int(len(zyx))
        k += len(zyx)
    code = hashlib.md5(Path(__file__).read_bytes()).hexdigest()
    with open(a.out, "w") as fh:
        for wi, r in sorted(res.items()):
            null = 0.5 * (r["null_out"] + r["null_in"])
            row = {"winding_dir": names[wi], "n": r["n"], "on_sheet": r["on"], "null_out": r["null_out"],
                   "null_in": r["null_in"], "null_shift": null, "lift": r["on"] - null,
                   "shift_l0_vox": a.shift, "level": a.level, "tol": a.tol, "pred": str(a.pred),
                   "seed": a.seed, "code_md5": code}
            fh.write(json.dumps(row) + "\n")
    on = np.array([r["on"] for r in res.values()])
    nl = np.array([0.5 * (r["null_out"] + r["null_in"]) for r in res.values()])
    print(f"{len(res)} windings: on_sheet p10/50/90 {np.percentile(on, [10, 50, 90]).round(3)} | "
          f"null p10/50/90 {np.percentile(nl, [10, 50, 90]).round(3)} | lift>0 in {(on > nl).sum()}/{len(on)}")


if __name__ == "__main__":
    main()
