# VENDORED from the source fleet repo (src/vesuvius_pipeline/coverage.py), see ../VENDORED.json for the exact source md5, release and the patches applied.
# Do not edit here: re-vendor, so the guard stays the SAME code the production fleet runs.
"""Same-sheet coverage: how much of a segment's surface another segment already holds.

The costliest waste in this pipeline (user, 2026-09-25) is segments that re-grow the same
sheet and are then flattened, rendered and ink-detected again. This module is the estimator
and the gate's decision rule; the census/publisher that fills the metric lives in
`scripts/coverage/` (it needs every lattice of a scroll and runs off the hub).

ESTIMATOR (the overlap study's, FINDINGS 49 / scripts/stitch/plan.py `measure`, per point):
a query point p of segment P is COVERED by segment T when T's surface passes within
`SAME_SHEET_VOX` voxels of p ALONG T's NORMAL, T's normal there is within `ALIGN_DEG` of P's
(orientation-free: |cos|), and the lateral offset to the T point used is within
1.5 x T's median lattice edge (so a point beyond T's boundary is not "covered" by T's rim).
The adjacent wrap is one sheet pitch away (~25-35 voxels at these scans' 7.9-9.4 um), so 4 voxels
separates "the same sheet grown twice" from "the next wrap" -- measured in the census
(FINDINGS, 2026-09-25 coverage section), not assumed.

STORED vs DERIVED (CLAUDE.md "there is no such thing as permanent damage"): the census
stores a MEASUREMENT per segment (metric `covered_frac`, text_value = JSON naming the
covering segments). Whether that skips the segment is decided at READ time against the
pipeline_setting `finish.max_covered_frac`, so raising the threshold (or setting it to 1.01)
undoes every skip at once, retroactively, with nothing deleted.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:                 # numpy only where a function needs it: the gate's rule does not
    import numpy as np

SAME_SHEET_VOX = 4.0        # sheet-normal distance that counts as "the same sheet" (point-wise)
ALIGN_DEG = 20.0            # normals must agree this well (orientation-free)
R_QUERY = 60.0              # KD neighbourhood; also the histogram range
K_NEIGH = 6
JUMP_FACTOR = 3.0           # an edge > 3x the median edge is a hole, not surface

# 🔴 FOUND 2026-09-30 (c2322ce, n = 42,436 query points; docs/experiments/ink_fusion/STATE.md):
# the point-wise gate ALONE admits adjacent-winding material as "same sheet". Genuine-duplicate
# pairs have a per-PAIR MEDIAN matched normal distance of 1.09-1.35 vox; likely adjacent-winding
# pairs 5.84-13.56 vox -- yet the point-wise rule alone still "covers" 22-36% of the smaller
# segment for the latter, because a few points near a fold/crossing land within SAME_SHEET_VOX
# by chance even though the pair as a whole sits a wrap apart. There is no valley in the POOLED
# POINT distance histogram to threshold on (that population is a continuum), but the two
# populations' PAIR MEDIANS are separated by > 4x (1.35 vs 5.84 vox) with a wide, unoccupied gap
# between them. SAME_SHEET_MEDIAN_VOX sits in that gap: well clear of the highest genuine-
# duplicate median seen (1.35) and well below the lowest adjacent-winding median seen (5.84).
# Checked (this fix): PHerc0211 and a second scroll both show the same two-mode separation with
# this cut; see the fix's own STATE.md section for the per-scroll numbers, n and a sanity check
# that it is not a tuned/one-scroll artifact (CLAUDE.md "a sweep must span the population").
SAME_SHEET_MEDIAN_VOX = 2.0    # pair-level: median matched normal distance that counts as one sheet
                               # (a pair, not a point -- see `pair_same_sheet()` / `gated_hit()`)

# the metric the census writes and the gate reads
METRIC = "covered_frac"
SETTING = "finish.max_covered_frac"
# Default threshold: chosen from the 2026-09-25 census (FINDINGS). A segment at least this
# fraction covered, same-sheet, by segments already flattened/inked is not finished again.
DEFAULT_MAX_COVERED = 0.8


def read_lattice(kind: str, path: str):
    """(X, Y, Z float32 (H, W)); invalid points are <= 0. kind: tifxyz | flat | pushed."""
    import numpy as np
    if kind in ("tifxyz", "flat", "remote"):
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


def normal_distance(p: np.ndarray, pn: np.ndarray, pok: np.ndarray, T: dict, tree=None) -> np.ndarray:
    """Per query point: the smallest sheet-normal distance (voxels) to T among T's K nearest
    points whose normal is aligned with the query's within ALIGN_DEG and whose lateral offset
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
    acc = tok & (lat <= lat_max) & (cos >= np.cos(np.radians(ALIGN_DEG)))
    out[inb] = np.where(acc, t, np.inf).min(axis=1)
    return out


def pair_match_stats(d: np.ndarray) -> tuple[float, int]:
    """Median and count of MATCHED points in a (query segment, T) pair: points with a finite
    `normal_distance` (an aligned, in-range T point was found), over ALL matched points -- not
    only the ones the point-wise SAME_SHEET_VOX rule accepts. (nan, 0) if nothing matched."""
    import numpy as np
    m = d[np.isfinite(d)]
    if not len(m):
        return float("nan"), 0
    return float(np.median(m)), int(len(m))


def pair_same_sheet(d: np.ndarray, median_vox: float = SAME_SHEET_MEDIAN_VOX) -> tuple[bool, float, int]:
    """A (query segment, T) PAIR counts as same-sheet only if it has >= 1 matched point and the
    pair's median matched normal distance is <= median_vox. This is the fix for the finding
    above `SAME_SHEET_MEDIAN_VOX`: rejects an adjacent-winding pair the point-wise gate alone
    still admits. Returns (pair_ok, median_vox_of_pair, n_matched)."""
    med, n = pair_match_stats(d)
    return (n > 0 and med <= median_vox), med, n


def gated_hit(d: np.ndarray, vox: float = SAME_SHEET_VOX,
             median_vox: float = SAME_SHEET_MEDIAN_VOX) -> tuple[np.ndarray, bool, float]:
    """The production accept rule for one (query segment, T) pair: the point-wise gate
    (d <= vox) AND the pair-level median gate (`pair_same_sheet`). If the pair fails the median
    gate every point is returned un-hit (all False) -- the pair is adjacent-winding, not a
    same-sheet duplicate, however many individual points the point-wise rule alone would admit.
    Returns (hit: bool array same shape as d, pair_ok, pair_median_vox). Every production caller
    that used to write `d <= SAME_SHEET_VOX` goes through this function now."""
    import numpy as np
    ok, med, _n = pair_same_sheet(d, median_vox)
    hit = (d <= vox) if ok else np.zeros(len(d), dtype=bool)
    return hit, ok, med


# ---------------------------------------------------------------------------------------
# the gate's decision rule (pure; scheduler.finish_candidates calls it)

def max_covered(st: dict | None) -> float:
    """The operator's threshold from pipeline_setting (a fraction 0..1; > 1 disables)."""
    try:
        return float((st or {}).get(SETTING, DEFAULT_MAX_COVERED))
    except (TypeError, ValueError):
        return DEFAULT_MAX_COVERED


def load_coverage(db) -> dict:
    """seg -> (covered_frac, detail dict) from the NEWEST `covered_frac` row per segment."""
    out = {}
    try:
        rows = db.execute(f"SELECT seg, value, text_value FROM metric WHERE name='{METRIC}' AND id IN "
                          f"(SELECT MAX(id) FROM metric WHERE name='{METRIC}' GROUP BY seg)").fetchall()
    except Exception:     # noqa: BLE001 - no metric table in a bare test DB: nothing is covered
        return out
    for seg, val, tv in rows:
        try:
            det = json.loads(tv) if tv else {}
        except ValueError:
            det = {}
        out[seg] = (float(val) if val is not None else 0.0, det)
    return out


def covered_skip(seg: str, cov: dict, thr: float, done: set) -> tuple[bool, str]:
    """(skip?, reason). Skip only when the measured same-sheet coverage BY SEGMENTS THAT ARE
    DONE (flattened or inked) reaches `thr`, and at least one named coverer is still done
    now -- a coverer that was since purged/moved aside no longer covers anything, so the
    measurement is not trusted past it."""
    c = cov.get(seg)
    if not c or thr > 1.0:
        return False, ""
    frac, det = c
    if frac < thr:
        return False, ""
    by = [b for b in (det.get("by") or []) if (b[0] if isinstance(b, (list, tuple)) else b) in done]
    if not by:
        return False, ""
    top = by[0][0] if isinstance(by[0], (list, tuple)) else by[0]
    return True, f"covered {frac:.0%} same-sheet by {top} (+{len(by) - 1} more); threshold {thr:.0%}"
