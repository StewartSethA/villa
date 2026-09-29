"""Build the per-sample review inputs for ridge_hit decisions, on the machine that holds the data.

For each JOB (one guard round of one segment), this:
  1. reads the round's pre-guard checkpoint (what the tracer grew) and its guarded checkpoint (what
     the guard kept);
  2. recomputes ridge_hit's per-cell mask on the pre-guard lattice, exactly as the guard does
     (`growth_guard.ridge_hit_mask`, `ridge_hit_vox` from the policy);
  3. draws cells, seeded:
       REMOVED  flagged by ridge_hit AND absent from the guarded checkpoint
       KEPT     not flagged by ridge_hit AND present in the guarded checkpoint;
  4. writes one compressed .npz per cell:
       ct_stack, pred_stack   (2*D+1, 2*H+1, 2*H+1) uint8: planes parallel to the surface at normal
                              offsets -D..+D voxels, each (2H+1)^2 at 1 voxel/px, centred on the cell
                              (nearest-voxel reads at pyramid level 0; the prediction is binary 0/255)
       ct_xy, pred_xy         (2C, 2C) the axis-aligned CT / prediction slice at the cell's z
       lattice_xyz, lattice_valid   the (2L+1)^2 lattice patch around the cell (pre-guard)
       frame                  centre (x, y, z), normal n, tangents u, v (level-0 voxels)
     The keep/remove label is NOT stored in the npz; it goes to the job's key file only, so that
     a reviewer who opens the npz is still blind.

Usage (on the host with the CT, prediction and checkpoints):
  python build_inputs.py jobs.json OUTDIR [--seed 20260929]
jobs.json: [{"round_id", "scroll", "segment", "round", "pre", "guarded", "ct", "pred",
             "voxel_um", "ridge_hit_vox", "n_removed", "n_kept", "stratum", "source"}]
Writes OUTDIR/samples/<sample_id>.npz, OUTDIR/built.json (metadata, no labels), OUTDIR/key.json.
Needs numpy, scipy, tifffile, zarr and tifxyz_tools (this repository).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import tifffile
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tifxyz_tools import growth_guard as G  # noqa: E402

D, H, C, L = 16, 24, 64, 10


class Sampler:
    """Nearest-voxel reads of one OME-Zarr level, grouped by chunk. Points are level-0 (x, y, z)."""

    def __init__(self, path: str, level: int = 0):
        g = zarr.open(path, mode="r")
        self.a = g[str(level)] if str(level) in g else g
        self.f = 2 ** level
        self.shp = np.array(self.a.shape[-3:])
        self.ch = np.array(self.a.chunks[-3:])

    def __call__(self, xyz):
        xyz = np.asarray(xyz, dtype=np.float64)
        idx = np.rint(xyz / self.f).astype(np.int64)[:, ::-1]
        out = np.zeros(len(idx), np.float32)
        inb = np.all((idx >= 0) & (idx < self.shp), axis=1)
        ii, pos = idx[inb], np.nonzero(inb)[0]
        if not len(ii):
            return out
        cid = ii // self.ch
        key = (cid[:, 0] * 100000 + cid[:, 1]) * 100000 + cid[:, 2]
        order = np.argsort(key, kind="stable")
        ks = key[order]
        starts = np.r_[0, np.nonzero(np.diff(ks))[0] + 1]
        ends = np.r_[starts[1:], len(ks)]
        for s, e in zip(starts, ends, strict=True):
            sel = order[s:e]
            lo = cid[sel[0]] * self.ch
            hi = np.minimum(lo + self.ch, self.shp)
            blk = np.asarray(self.a[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
            loc = ii[sel] - lo
            out[pos[sel]] = blk[loc[:, 0], loc[:, 1], loc[:, 2]]
        return out

    def slab_xy(self, x, y, z, half):
        z = int(round(z))
        x0, y0 = int(round(x)) - half, int(round(y)) - half
        out = np.zeros((2 * half, 2 * half), np.uint8)
        if not (0 <= z < self.shp[0]):
            return out
        ys, xs = max(0, y0), max(0, x0)
        ye, xe = min(self.shp[1], y0 + 2 * half), min(self.shp[2], x0 + 2 * half)
        if ye > ys and xe > xs:
            out[ys - y0:ye - y0, xs - x0:xe - x0] = np.asarray(self.a[z, ys:ye, xs:xe])
        return out


def read_lattice(d):
    X, Y, Z = (tifffile.imread(os.path.join(d, f"{a}.tif")).astype(np.float64) for a in "xyz")
    return X, Y, Z


def frame_at(P, V, n, i, j):
    """Unit normal n, tangent u (along the lattice's column direction, made orthogonal to n), v = n x u."""
    nn = n[i, j]
    jj = j + 1 if j + 1 < V.shape[1] and V[i, j + 1] else j - 1
    t = P[i, jj] - P[i, j] if 0 <= jj < V.shape[1] and V[i, jj] else np.array([1.0, 0, 0])
    t = t - np.dot(t, nn) * nn
    if np.linalg.norm(t) < 1e-6:
        t = np.cross(nn, [0, 0, 1.0]) if abs(nn[2]) < 0.9 else np.cross(nn, [1.0, 0, 0])
    u = t / np.linalg.norm(t)
    v = np.cross(nn, u)
    return nn, u, v


def oriented_stack(sampler, c, n, u, v):
    o = np.arange(-D, D + 1, dtype=np.float64)
    a = np.arange(-H, H + 1, dtype=np.float64)
    pts = (c[None, None, None, :] + o[:, None, None, None] * n + a[None, :, None, None] * v
           + a[None, None, :, None] * u).reshape(-1, 3)
    return np.clip(sampler(pts), 0, 255).astype(np.uint8).reshape(len(o), len(a), len(a))


def sample_id(round_id, cls, i, j):
    return hashlib.sha1(f"{round_id}:{cls}:{i}:{j}".encode()).hexdigest()[:12]


def build(jobs, out, seed):
    out = Path(out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cache, built, key = {}, [], []
    for job in jobs:
        for p in ("ct", "pred"):
            if job[p] not in cache:
                cache[job[p]] = Sampler(job[p], 0)
        ct, pr = cache[job["ct"]], cache[job["pred"]]
        X, Y, Z = read_lattice(job["pre"])
        P, V = G.lattice_frame(X, Y, Z)
        Xg, Yg, Zg = read_lattice(job["guarded"]) if job.get("guarded") else (X, Y, Z)
        Vg = (Xg > 0) & (Yg > 0) & (Zg > 0)
        n, ok = G.grid_normals(P, V)
        pol = G.GuardPolicy(ridge_hit_vox=float(job.get("ridge_hit_vox", 3.0)))
        m = G.ridge_hit_mask(P, V, pr, pol)
        pools = {"removed": np.argwhere(V & ~Vg & m & ok), "kept": np.argwhere(V & Vg & ~m & ok)}
        stats = {"cells": int(V.sum()), "flagged": int((V & m).sum()), "removed_flagged": int((V & ~Vg & m).sum()),
                 "kept_unflagged": int((V & Vg & ~m).sum())}
        for cls, want in (("removed", int(job.get("n_removed", 1))), ("kept", int(job.get("n_kept", 1)))):
            pool = pools[cls]
            if not len(pool) or want <= 0:
                continue
            for i, j in pool[rng.choice(len(pool), min(want, len(pool)), replace=False)]:
                c = P[i, j]
                nn, u, v = frame_at(P, V, n, i, j)
                sid = sample_id(job["round_id"], cls, i, j)
                i0, i1, j0, j1 = max(0, i - L), min(V.shape[0], i + L + 1), max(0, j - L), min(V.shape[1], j + L + 1)
                patch = np.full((2 * L + 1, 2 * L + 1, 3), -1.0, np.float32)
                pv = np.zeros((2 * L + 1, 2 * L + 1), bool)
                patch[i0 - i + L:i1 - i + L, j0 - j + L:j1 - j + L] = P[i0:i1, j0:j1]
                pv[i0 - i + L:i1 - i + L, j0 - j + L:j1 - j + L] = V[i0:i1, j0:j1]
                np.savez_compressed(out / "samples" / f"{sid}.npz",
                                    ct_stack=oriented_stack(ct, c, nn, u, v), pred_stack=oriented_stack(pr, c, nn, u, v),
                                    ct_xy=ct.slab_xy(c[0], c[1], c[2], C), pred_xy=pr.slab_xy(c[0], c[1], c[2], C),
                                    lattice_xyz=patch, lattice_valid=pv,
                                    frame=np.stack([c, nn, u, v]).astype(np.float32))
                built.append({"sample_id": sid, "scroll": job["scroll"], "segment": job["segment"], "round": job.get("round"),
                              "round_id": job["round_id"], "source": job.get("source"), "cell": [int(i), int(j)],
                              "center_xyz": [round(float(x), 2) for x in c], "voxel_um": job["voxel_um"],
                              "ridge_hit_vox": pol.ridge_hit_vox, "ct_volume": os.path.basename(job["ct"].rstrip("/")),
                              "prediction": os.path.basename(job["pred"].rstrip("/")),
                              "geometry": {"normal_offsets_vox": [-D, D], "inplane_half_px": H, "xy_half_px": C,
                                           "lattice_half_cells": L, "px_um": job["voxel_um"]}})
                key.append({"sample_id": sid, "class": cls, "stratum": job["stratum"], "scroll": job["scroll"],
                            "round_id": job["round_id"], "round_stats": stats})
        print(job["round_id"], job["scroll"], job["stratum"], stats, flush=True)
    json.dump(built, open(out / "built.json", "w"), indent=0)
    json.dump(key, open(out / "key.json", "w"), indent=0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=20260929)
    a = ap.parse_args(argv)
    build(json.load(open(a.jobs)), a.out, a.seed)


if __name__ == "__main__":
    main()
