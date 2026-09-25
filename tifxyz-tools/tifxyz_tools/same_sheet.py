"""Same-sheet estimator: how much of one tifxyz surface another surface already holds.

Answers "is point p of surface P on the same sheet as surface T?" from the two surfaces alone,
no volume needed. A query point p of P is COVERED by T when T's surface passes within
`SAME_SHEET_VOX` voxels of p ALONG T's NORMAL, T's normal there is within `ALIGN_DEG` of P's
(orientation-free: |cos|), and the lateral offset to the T point used is within
1.5 x T's median lattice edge (so a point beyond T's boundary is not "covered" by T's rim).

Why 4 voxels: adjacent wraps of a scroll are one sheet pitch apart (25-40 voxels at 7.9-9.4 um),
while duplicates of one sheet grown twice sit within a few voxels (medians 1.4-3.2 voxels over the
pairs measured, VALIDATION.md). The threshold is a real sensitivity, not a valley in the
distribution: read fractions as +-35 %.

Pure numpy/scipy. Lattices are tifxyz (x.tif / y.tif / z.tif, invalid <= 0).
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

SAME_SHEET_VOX = 4.0        # sheet-normal distance that counts as "the same sheet"
ALIGN_DEG = 20.0            # normals must agree this well (orientation-free)
R_QUERY = 60.0              # KD neighbourhood, voxels
K_NEIGH = 6
JUMP_FACTOR = 3.0           # an edge > 3x the median edge is a hole, not surface


def read_lattice(kind: str, path: str):
    """(X, Y, Z float32 (H, W)); invalid points are <= 0.

    kind: "tifxyz" | "flat" (a directory holding x.tif, y.tif, z.tif) | "pushed" (a gzipped
    float32 H*W*3 blob whose sibling latest.json holds {"grid": [H, W]}: the fleet's own
    push format, kept because the validation lattices were stored that way).
    """
    import numpy as np
    if kind in ("tifxyz", "flat"):
        import tifffile
        d = Path(path)
        return tuple(tifffile.imread(d / f"{a}.tif").astype(np.float32) for a in "xyz")
    if kind == "pushed":
        d = Path(path).parent
        meta = json.loads((d / "latest.json").read_text())
        h, w = meta["grid"]
        raw = gzip.decompress(Path(path).read_bytes())
        xyz = np.frombuffer(raw[: h * w * 12], dtype=np.float32).reshape(h, w, 3)
        return xyz[..., 0].copy(), xyz[..., 1].copy(), xyz[..., 2].copy()
    raise ValueError(kind)


def surface_points(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> dict:
    """Valid points, unit normals (central/one-sided differences, jump edges excluded), the
    median lattice edge in voxels and the bbox. Normals invalid where undefined."""
    import numpy as np
    v = (X > 0) & (Y > 0) & (Z > 0)
    P = np.stack([X, Y, Z], axis=-1).astype(np.float64)
    H, W = v.shape
    empty = {"xyz": np.zeros((0, 3), np.float32), "nrm": np.zeros((0, 3), np.float32),
             "nok": np.zeros(0, bool), "edge_med": float("nan"), "bbox": None}
    if v.sum() < 4 or H < 2 or W < 2:
        return empty
    ei = np.linalg.norm(P[1:, :] - P[:-1, :], axis=-1)
    ei_ok = v[1:, :] & v[:-1, :]
    ej = np.linalg.norm(P[:, 1:] - P[:, :-1], axis=-1)
    ej_ok = v[:, 1:] & v[:, :-1]
    edges = np.concatenate([ei[ei_ok], ej[ej_ok]])
    if not edges.size:
        return empty
    med = float(np.median(edges))
    ei_good = ei_ok & (ei <= med * JUMP_FACTOR)
    ej_good = ej_ok & (ej <= med * JUMP_FACTOR)

    def deriv(good, axis):
        D = np.zeros_like(P)
        fwd = np.zeros((H, W), bool)
        bwd = np.zeros((H, W), bool)
        if axis == 0:
            fwd[:-1, :] = good
            bwd[1:, :] = good
        else:
            fwd[:, :-1] = good
            bwd[:, 1:] = good
        nxt = np.roll(P, -1, axis)
        prv = np.roll(P, 1, axis)
        both = fwd & bwd
        D[both] = (nxt - prv)[both]
        of = fwd & ~bwd
        D[of] = (nxt - P)[of]
        ob = bwd & ~fwd
        D[ob] = (P - prv)[ob]
        return D, fwd | bwd

    di, di_ok = deriv(ei_good, 0)
    dj, dj_ok = deriv(ej_good, 1)
    n = np.cross(di, dj)
    nn = np.linalg.norm(n, axis=-1)
    nok = v & di_ok & dj_ok & (nn > 1e-9)
    n = n / np.maximum(nn, 1e-12)[..., None]
    ii, jj = np.nonzero(v)
    xyz = P[ii, jj].astype(np.float32)
    return {"xyz": xyz, "nrm": n[ii, jj].astype(np.float32), "nok": nok[ii, jj], "edge_med": med,
            "bbox": [xyz.min(axis=0).tolist(), xyz.max(axis=0).tolist()]}


def sample_idx(key: str, n: int, size: int) -> np.ndarray:
    """A deterministic per-segment sample of point indices (same seed every run)."""
    import numpy as np
    import zlib
    if n <= size:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    return np.sort(rng.choice(n, size=size, replace=False))


def normal_distance(p: np.ndarray, pn: np.ndarray, pok: np.ndarray, T: dict, tree=None,
                    align_deg: float = ALIGN_DEG) -> np.ndarray:
    """Per query point: the smallest sheet-normal distance (voxels) to T among T's K nearest
    points whose normal is aligned with the query's within `align_deg` and whose lateral offset
    is within 1.5 x T's median edge. inf where no such T point exists within R_QUERY."""
    import numpy as np
    out = np.full(len(p), np.inf)
    if not len(p) or not len(T["xyz"]):
        return out
    lo = np.asarray(T["bbox"][0]) - R_QUERY
    hi = np.asarray(T["bbox"][1]) + R_QUERY
    inb = np.all((p >= lo) & (p <= hi), axis=1) & pok
    if not inb.any():
        return out
    if tree is None:
        from scipy.spatial import cKDTree
        tree = cKDTree(T["xyz"], leafsize=32, balanced_tree=False, compact_nodes=False)
    q = p[inb]
    qn = pn[inb]
    k = min(K_NEIGH, len(T["xyz"]))
    d, j = tree.query(q, k=k, distance_upper_bound=R_QUERY, workers=1)
    if k == 1:
        d = d[:, None]
        j = j[:, None]
    ok = np.isfinite(d)
    jc = np.where(ok, j, 0)
    tp = T["xyz"][jc]
    tn = T["nrm"][jc]
    tok = T["nok"][jc] & ok
    t = np.abs(np.einsum("ijk,ijk->ij", q[:, None, :] - tp, tn))
    lat = np.sqrt(np.maximum(d * d - t * t, 0.0))
    lat_max = max(25.0, 1.5 * float(T["edge_med"]))
    cos = np.abs(np.einsum("ik,ijk->ij", qn, tn))
    acc = tok & (lat <= lat_max) & (cos >= np.cos(np.radians(align_deg)))
    out[inb] = np.where(acc, t, np.inf).min(axis=1)
    return out


def covered_fraction(p, pn, pok, T, tree=None, vox: float = SAME_SHEET_VOX,
                     align_deg: float = ALIGN_DEG) -> float:
    """Fraction of P's points (those with a defined normal) lying on the same sheet as surface T.
    `T` is a `surface_points()` dict. Point-weighted, not area-weighted. nan if P has no normals."""
    import numpy as np
    if not np.asarray(pok).any():
        return float("nan")
    d = normal_distance(p, pn, pok, T, tree=tree, align_deg=align_deg)
    return float((d[pok] <= vox).mean())
