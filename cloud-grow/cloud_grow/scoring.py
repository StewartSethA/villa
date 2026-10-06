"""Per-checkpoint yield scoring without the fleet's `scripts/segment_yield.py`.

Ported from `segment_yield.measure` (material_frac, bbox_occupancy, topology/onesheet) and
`stages.grow.score_checkpoint` (verdict thresholds). DIFFERENCE, stated: the fleet reads CT level 0 from a raw
memmap; a cloud box holds CT LEVEL 1 only (32-36 GB; level 0 is 252 GB for PHerc0211), so material is sampled
at the nearest level-1 voxel (`ZarrSampler(level=1)`). The hub re-scores every imported checkpoint (importer),
so the number written here is the box's own claim, never the verified one.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

MATERIAL_TH = 5
GEO_MIN = 600.0
EUC_MAX = 15.0
N_SOURCES = 128
VACUUM_TH, WATCH_TH, FRAME_TH = 0.75, 0.90, 0.02     # stages/grow.py


@dataclass
class Score:
    area_cm2: float
    material_frac: float
    onesheet_frac: float | None
    bbox_occupancy: float | None
    verdict: str
    n_valid: int = 0
    n_sampled: int = 0

    @property
    def verified_cm2(self) -> float:
        return self.area_cm2 * self.material_frac * (self.onesheet_frac if self.onesheet_frac is not None else 1.0)


def valid_mask(X, Y, Z):
    return np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z) & (X > 0) & (Y > 0) & (Z > 0)


def read_xyz(d):
    import tifffile
    return tuple(tifffile.imread(os.path.join(d, f"{a}.tif")).astype(np.float32) for a in "xyz")


def topology(X, Y, Z, V, seed=0):
    """Shortcut-pair rate (segment_yield.topology, verbatim logic): far along the surface, near in 3D."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    H, W = X.shape
    idx = -np.ones((H, W), np.int64)
    n = int(V.sum())
    idx[V] = np.arange(n)
    P = np.stack([X[V], Y[V], Z[V]], 1).astype(np.float64)
    if n < 4:
        return None, None
    r, c, w = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        if dx >= 0:
            a = idx[:H - dy, :W - dx]; b = idx[dy:, dx:]
        else:
            a = idx[:H - dy, -dx:]; b = idx[dy:, :W + dx]
        m = (a >= 0) & (b >= 0)
        ai, bi = a[m], b[m]
        if not len(ai):
            continue
        d = np.linalg.norm(P[ai] - P[bi], axis=1)
        r += [ai, bi]; c += [bi, ai]; w += [d, d]
    if not r:
        return None, None
    g = coo_matrix((np.concatenate(w), (np.concatenate(r), np.concatenate(c))), shape=(n, n)).tocsr()
    rng = np.random.default_rng(seed)
    src = rng.choice(n, min(N_SOURCES, n), replace=False)
    D = dijkstra(g, directed=False, indices=src)
    Eu = np.linalg.norm(P[src][:, None, :] - P[None, :, :], axis=2)
    far = D > GEO_MIN
    bad = far & (Eu < EUC_MAX)
    return float(bad.any(1).mean()), (float(bad.sum() / far.sum()) if far.sum() else 0.0)


def score_checkpoint(ckpt: str, sampler, sample: int = 5000, seed: int = 0, do_topology: bool = True) -> Score:
    """sampler(xyz level-0 voxels (N,3)) -> CT values (N,). Uses meta.json's area_cm2 like the fleet scorer."""
    X, Y, Z = read_xyz(ckpt)
    meta = {}
    try:
        with open(os.path.join(ckpt, "meta.json")) as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        pass
    area = float(meta.get("area_cm2") or 0.0)
    V = valid_mask(X, Y, Z)
    n_valid = int(V.sum())
    if n_valid == 0:
        return Score(area, 0.0, None, None, "VACUUM", 0, 0)
    P = np.stack([X[V], Y[V], Z[V]], 1)
    rng = np.random.default_rng(seed)
    k = min(sample, len(P))
    sel = rng.choice(len(P), k, replace=False) if k < len(P) else np.arange(len(P))
    vals = np.asarray(sampler(P[sel]))
    mat = float((vals > MATERIAL_TH).mean()) if len(vals) else 0.0
    lo, hi = P.min(0), P.max(0)
    bb = rng.uniform(lo, lo + np.maximum(hi - lo, 1.0), size=(min(20000, max(2000, k)), 3))
    occ = float((np.asarray(sampler(bb)) > MATERIAL_TH).mean())
    one = None
    if do_topology:
        try:
            pt, _pair = topology(X, Y, Z, V, seed=seed)
            one = None if pt is None else 1.0 - pt
        except Exception:                                   # noqa: BLE001 - a topology failure leaves onesheet unknown
            one = None
    if occ < FRAME_TH:
        v = "FRAME_SUSPECT"
    elif mat < VACUUM_TH:
        v = "VACUUM"
    elif mat < WATCH_TH:
        v = "WATCH"
    else:
        v = "OK"
    return Score(area, mat, one, occ, v, n_valid, int(len(vals)))
