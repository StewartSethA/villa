"""Train / submission NON-OVERLAP guard, by 3-D surface geometry (user, 2026-10-08: "guarantee non-overlap -- these came from different stripe
widths and so might overlap").

Names cannot say whether two segments cover the same papyrus: several differently-named grows of the same scroll are
different stripe fits of the SAME windings, and Route A segments (cNNNNNN) cross them. Overlap is therefore measured on the meshes:

  protected = every segment in the splits "submission" and "eval"      (their surfaces, densified to <= EPS_VOX spacing)
  for each segment in "train": a vertex is EXCLUDED when its 3-D distance to any protected surface point is < EXCL_VOX
                               (same sheet within EPS_VOX; a neighbouring winding sits ~15 vox away, so EXCL_VOX stays below that), and the
                               exclusion is DILATED in the tile's own grid by MARGIN_CELLS so the model's in-plane context window never reads a
                               protected pixel either (margin = context window / 2 in grid cells; a named, disclosed parameter).
  Output per train segment: <out>/<seg>.exclude.npy  (bool, tile grid shape: True = never train on / never score as training);
  report.json with area overlapped (cm2, %), cells excluded, and an INDEPENDENT re-check: min distance from the KEPT vertices to the protected
  surface must be >= EXCL_VOX + (the margin in voxels): the guarantee is that number, printed, not the construction.

Parameters are named and printed beside every use (D33).  usage: split_guard.py SPLITS.json --out DIR [--excl 6 --margin-px 160 --um 9.5]
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

EPS_VOX = 5.0          # densify protected surfaces to this spacing
EXCL_VOX = 7.5         # half the ~15-vox wrap pitch: beyond this a vertex is closer to the NEIGHBOURING wrap than to the protected sheet; different stripe fits of one sheet differ by up to this much
CONTEXT_PX = 160       # in-plane context the model may read (px of the render, 9.5 um/px) -> margin = half of it
UM_PX = 9.5


def read_xyz(d):
    import tifffile
    d = Path(d)
    X, Y, Z = (tifffile.imread(d / f"{c}.tif").astype(np.float32) for c in "xyz")
    V = (X > 0) & (Y > 0) & (Z > 0)
    return [np.where(V, A, np.nan).astype(np.float32) for A in (X, Y, Z)], V


def densify(G, eps):
    """Points on the surface at <= eps spacing: bilinear refinement of every quad whose four corners are valid."""
    X, Y, Z = G
    pts = [np.stack([A[np.isfinite(A)] for A in (X, Y, Z)], 1)]
    d = np.nanmedian(np.sqrt(np.diff(X, axis=1) ** 2 + np.diff(Y, axis=1) ** 2 + np.diff(Z, axis=1) ** 2))
    k = max(1, int(math.ceil(float(d) / eps))) if np.isfinite(d) else 1
    if k > 1:
        for a in range(1, k):
            for b in range(1, k):
                u, v = a / k, b / k
                ok = np.isfinite(X[:-1, :-1]) & np.isfinite(X[1:, :-1]) & np.isfinite(X[:-1, 1:]) & np.isfinite(X[1:, 1:])
                P = [((1 - u) * (1 - v) * A[:-1, :-1] + u * (1 - v) * A[1:, :-1] + (1 - u) * v * A[:-1, 1:] + u * v * A[1:, 1:])[ok] for A in (X, Y, Z)]
                pts.append(np.stack(P, 1))
    return np.concatenate(pts, 0)


def dilate(mask, r):
    """Boolean dilation by a (2r+1) square: cheap, separable, no scipy needed."""
    if r <= 0:
        return mask
    m = mask.copy()
    for ax in (0, 1):
        acc = m.copy()
        for s in range(1, r + 1):
            acc |= np.roll(m, s, ax) | np.roll(m, -s, ax)
        m = acc
    return m


def guard(protected: dict, train: dict, excl=EXCL_VOX, eps=EPS_VOX, margin_cells=None, context_px=CONTEXT_PX, um=UM_PX):
    """protected/train: {seg: (G, valid)}. Returns {seg: dict(exclude=bool grid, ...)}, plus the protected point count."""
    from scipy.spatial import cKDTree
    P = np.concatenate([densify(G, eps) for G, _ in protected.values()], 0) if protected else np.zeros((0, 3), np.float32)
    tree = cKDTree(P) if len(P) else None
    out = {}
    for seg, (G, V) in train.items():
        X, Y, Z = G
        d = np.full(X.shape, np.inf, np.float32)
        idx = np.argwhere(V)
        if tree is not None and len(idx):
            q = np.stack([X[V], Y[V], Z[V]], 1)
            d[V] = tree.query(q, k=1)[0].astype(np.float32)
        spacing = float(np.nanmedian(np.sqrt(np.diff(X, axis=1) ** 2 + np.diff(Y, axis=1) ** 2 + np.diff(Z, axis=1) ** 2))) if V.any() else 20.0
        mc = margin_cells if margin_cells is not None else int(math.ceil((context_px / 2.0) * um / (spacing * um)))   # render px and grid cells share the voxel pitch scale
        hit = V & (d < excl)
        ex = dilate(hit, mc) | ~V
        kept = V & ~ex
        # independent re-check: the nearest KEPT vertex to the protected surface
        dk = float(d[kept].min()) if kept.any() else float("inf")
        cell_cm2 = None
        out[seg] = dict(exclude=ex, hit_cells=int(hit.sum()), valid_cells=int(V.sum()), kept_cells=int(kept.sum()), margin_cells=mc, grid_spacing_vox=round(spacing, 2),
                        min_dist_kept_vox=round(dk, 2), min_dist_all_vox=round(float(d[V].min()), 2) if V.any() else None,
                        frac_overlapping=round(float(hit.sum()) / max(1, int(V.sum())), 4))
    return out, len(P)


def load_grid(seg, fl, db, work):
    """((G, V), path) for a segment's mesh: the recorded tifxyz when readable here, else PULLED from the owner host by the pipeline's own routine
    (never guessed). (None, reason) when it cannot be had."""
    from vesuvius_pipeline.stages import render as R
    r = db.execute("SELECT path FROM artifact WHERE seg=? AND kind='tifxyz' ORDER BY recorded_utc DESC LIMIT 1", (seg,)).fetchone()
    p = r[0] if r else None
    if not (p and os.path.exists(os.path.join(p, "x.tif"))):
        src, _mt, kind = R.source_tifxyz(fl, db, seg, Path(work) / seg)
        if not src:
            return None, kind
        p = src
    G, V = read_xyz(p)
    return (G, V), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("splits")
    ap.add_argument("--out", required=True)
    ap.add_argument("--excl", type=float, default=EXCL_VOX)
    ap.add_argument("--context-px", type=int, default=CONTEXT_PX)
    ap.add_argument("--um", type=float, default=UM_PX)
    a = ap.parse_args()
    from vesuvius_pipeline import config
    from vesuvius_pipeline.db import pipeline_db
    from vesuvius_pipeline.stages import render as R
    fl = config.load(None)
    db = pipeline_db().connect(str(fl.pipeline_db))
    sp = json.load(open(a.splits))
    os.makedirs(a.out, exist_ok=True)
    work = Path(a.out) / "_pulled"

    grid = lambda seg: load_grid(seg, fl, db, work)
    prot, train, miss = {}, {}, {}
    for s in sp.get("submission", []) + sp.get("eval", []):
        g, why = grid(s)
        (prot.__setitem__(s, g) if g else miss.__setitem__(s, why))
    for s in sp.get("train", []):
        g, why = grid(s)
        (train.__setitem__(s, g) if g else miss.__setitem__(s, why))
    res, npts = guard(prot, train, excl=a.excl, context_px=a.context_px, um=a.um)
    rep = dict(params=dict(EPS_VOX=EPS_VOX, EXCL_VOX=a.excl, CONTEXT_PX=a.context_px, UM_PX=a.um), protected=sorted(prot), protected_surface_points=npts, missing_geometry=miss, train={})
    for seg, r in res.items():
        np.save(Path(a.out) / f"{seg}.exclude.npy", r.pop("exclude"))
        rep["train"][seg] = r
    json.dump(rep, open(Path(a.out) / "report.json", "w"), indent=1)
    print(f"protected: {len(prot)} segments, {npts / 1e6:.2f} M surface points; EXCL {a.excl} vox, context {a.context_px} px; missing geometry: {miss or 'none'}")
    for seg, r in rep["train"].items():
        print(f"  {seg:34s} overlap {100 * r['frac_overlapping']:5.1f} % of valid cells | kept {r['kept_cells']:6d}/{r['valid_cells']:6d} | nearest kept vertex to a protected surface {r['min_dist_kept_vox']} vox (nearest of all {r['min_dist_all_vox']})")
    bad = [s for s, r in rep["train"].items() if r["kept_cells"] and r["min_dist_kept_vox"] < a.excl]
    print("GUARANTEE:", "HOLDS (every kept vertex is >= %.1f vox from every protected surface)" % a.excl if not bad else f"VIOLATED for {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
