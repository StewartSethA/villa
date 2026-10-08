"""Modular DEGENERACY checks for a tifxyz lattice (user 2026-10-07: "no degenerate surfaces, ever"; precursors BEFORE they become crossings).

Every check: pure numpy on (X, Y, Z[, V]); returns a bool mask on lattice CELLS (H, W) (quad-based checks mark the 4 corners) plus, for the
continuous ones, the measured quantity in its unit. Every threshold is a NAMED TUNABLE in `DEFAULTS` with a disclosure line (`disclosure(name)`).

ALREADY IN growth_guard.py (wrapped, not duplicated): fold_mask (hairpin radius), plan_mask (window-mean normal deviation), quad_flip_mask
(cell normal > 90 deg from window mean), stretch_mask (edge > stretch_ratio_th x median), overlap/wrap_spacing/curvature (need CT / umbilicus).
ALREADY IN selfcontact.py (wrapped): coincident/close non-neighbour samples (self-proximity < sheet pitch), long-edge quads.
NEW HERE: turning_angle (deg), dihedral (deg, between adjacent quad normals), fold_over (negative Jacobian vs window-mean normal),
normal_reversal, zero_area, edge_ratio (stretch AND compression), duplicate_vertices, orphan_vertices (a lattice cannot be non-manifold except
through coincident vertices or vertices in no complete quad), radial_jump (offset of a cell from the mean of its 8 neighbours along the local
normal, in median edges: the sheet-skip precursor). Units: degrees, voxels, median-edge multiples. Not validated against human annotation.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

# NAMED TUNABLES: name -> (default, unit, basis). Provisional defaults are overwritten by the population measurement (see STATE.md of
# docs/experiments/selfx_population_2026-10-07/degeneracy); tests pass explicit values.
DEFAULTS = {
    "TURN_DEG": (120.0, "deg", "tangent direction change across one cell along a row/col; = P99.9 (122.7 deg) of the in-solve-guarded crossing-free baseline, n=39 segments / 2.29 M cells / 12 scrolls (degeneracy/STATE.md)"),
    "CREASE_DEG": (165.0, "deg", "dihedral between adjacent quad normals; FP 0.51 % on the guarded baseline (its P99 is 151 deg, P99.9 175.8 deg: reflections persist without crossing) (degeneracy/STATE.md)"),
    "NORMAL_REV_DEG": (90.0, "deg", "adjacent quad normals pointing apart by more than a right angle = locally inside-out; physical, not fitted"),
    "ZERO_AREA_FRAC": (0.05, "x median quad area", "quad area below this fraction of the segment's median quad area"),
    "EDGE_HI": (3.0, "x local median edge", "edge stretch (5x5 local median of that axis)"),
    "EDGE_LO": (0.2, "x local median edge", "edge compression"),
    "JUMP_REL": (0.5, "x median edge", "|offset of a cell from the mean of its 8 neighbours along the local normal|"),
    "DUP_VOX": (0.5, "vox", "duplicate vertex distance (selfcontact.COINCIDENT_VOX)"),
}


def tun(name: str, override: dict | None = None) -> float:
    return float((override or {}).get(name, DEFAULTS[name][0]))


def disclosure(name: str, override: dict | None = None, n: int | None = None, cells: int | None = None) -> str:
    d, unit, basis = DEFAULTS[name]
    s = f"degeneracy.{name}={tun(name, override):g} {unit} (named tunable; {basis})"
    if n is not None:
        s += f"; scored n={n} segments" + (f", {cells} cells" if cells is not None else "")
    return s


def _valid(X, Y, Z, V=None):
    X, Y, Z = (np.asarray(a, np.float64) for a in (X, Y, Z))
    if V is None:
        V = (X > 0) & (Y > 0) & (Z > 0)
    return np.stack([X, Y, Z], -1), V


def _ang(a, b):
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.einsum("...c,...c->...", a, b) / (na * nb)
    return np.degrees(np.arccos(np.clip(c, -1, 1))), (na > 1e-9) & (nb > 1e-9)


def turning_angle(P, V):
    """(H,W) deg: at each cell the larger of the row and column tangent direction changes (incoming vs outgoing edge). NaN where undefined."""
    H, W = V.shape
    out = np.full((H, W), np.nan)
    for ax in (0, 1):
        Q, U = (P, V) if ax == 0 else (P.transpose(1, 0, 2), V.T)
        t_in, t_out = Q[1:-1] - Q[:-2], Q[2:] - Q[1:-1]
        ok = U[:-2] & U[1:-1] & U[2:]
        a, good = _ang(t_in, t_out)
        a = np.where(ok & good, a, np.nan)
        full = np.full(U.shape, np.nan); full[1:-1] = a
        full = full if ax == 0 else full.T
        out = np.fmax(out, full)
    return out


def quad_normals(P, V):
    """((H-1,W-1,3) unit quad normals from the two diagonals-free mean tangents, (H-1,W-1) valid)."""
    q = V[:-1, :-1] & V[1:, :-1] & V[:-1, 1:] & V[1:, 1:]
    ti = (P[1:, :-1] - P[:-1, :-1]) + (P[1:, 1:] - P[:-1, 1:])
    tj = (P[:-1, 1:] - P[:-1, :-1]) + (P[1:, 1:] - P[1:, :-1])
    n = np.cross(ti, tj)
    nn = np.linalg.norm(n, axis=-1)
    q = q & (nn > 1e-9)
    return n / np.maximum(nn, 1e-12)[..., None], q


def dihedral(P, V):
    """(H-1,W-1) deg: per quad the largest angle between its normal and its 4 edge-adjacent quads' normals. NaN where undefined."""
    n, q = quad_normals(P, V)
    out = np.full(q.shape, np.nan)
    for dr, dc in ((0, 1), (1, 0)):
        a = n[:q.shape[0] - dr, :q.shape[1] - dc]; b = n[dr:, dc:]
        ok = q[:q.shape[0] - dr, :q.shape[1] - dc] & q[dr:, dc:]
        ang, _ = _ang(a, b)
        ang = np.where(ok, ang, np.nan)
        out[:q.shape[0] - dr, :q.shape[1] - dc] = np.fmax(out[:q.shape[0] - dr, :q.shape[1] - dc], ang)
        out[dr:, dc:] = np.fmax(out[dr:, dc:], ang)
    return out


def quad_to_cells(qmask, shape):
    out = np.zeros(shape, bool)
    out[:-1, :-1] |= qmask; out[1:, :-1] |= qmask; out[:-1, 1:] |= qmask; out[1:, 1:] |= qmask
    return out


def fold_over(P, V, win: int = 1):
    """Negative-Jacobian quads: quad normal points >90 deg from the mean normal of its (2*win+1)^2 quad window (the lattice orientation is
    consistent, so a sign flip = the surface folded over). Quad mask (H-1,W-1). Related to growth_guard.quad_flip_mask (cell normals)."""
    n, q = quad_normals(P, V)
    w = 2 * win + 1
    wt = q.astype(np.float64)
    m = np.stack([ndi.uniform_filter(n[..., c] * wt, size=w, mode="constant") for c in range(3)], -1)
    return q & (np.einsum("...c,...c->...", n, m) < 0.0)


def normal_reversal(P, V, override=None):
    d = dihedral(P, V)
    return np.nan_to_num(d, nan=0.0) > tun("NORMAL_REV_DEG", override)


def zero_area(P, V, override=None):
    n, q = quad_normals(P, V)
    a = 0.5 * (np.linalg.norm(np.cross(P[1:, :-1] - P[:-1, :-1], P[:-1, 1:] - P[:-1, :-1]), axis=-1)
               + np.linalg.norm(np.cross(P[1:, 1:] - P[1:, :-1], P[1:, 1:] - P[:-1, 1:]), axis=-1))
    qq = V[:-1, :-1] & V[1:, :-1] & V[:-1, 1:] & V[1:, 1:]
    if not qq.any():
        return qq
    return qq & (a < tun("ZERO_AREA_FRAC", override) * float(np.median(a[qq])))


def edge_ratio(P, V):
    """Per-edge ratio to the 5x5 local median edge of the same axis -> (ratio_rows (H,W-1), ratio_cols (H-1,W)), NaN on invalid edges."""
    res = []
    for ax in (0, 1):
        a, b = (P[1:], P[:-1]) if ax == 0 else (P[:, 1:], P[:, :-1])
        ok = (V[1:] & V[:-1]) if ax == 0 else (V[:, 1:] & V[:, :-1])
        L = np.linalg.norm(a - b, axis=-1)
        if not ok.any():
            res.append(np.full(L.shape, np.nan)); continue
        fill = np.where(ok, L, np.median(L[ok]))
        med = ndi.median_filter(fill, size=5, mode="nearest")
        res.append(np.where(ok, L / np.maximum(med, 1e-9), np.nan))
    return res[1], res[0]


def edge_ratio_mask(P, V, override=None):
    hi, lo = tun("EDGE_HI", override), tun("EDGE_LO", override)
    out = np.zeros(V.shape, bool)
    rr, rc = edge_ratio(P, V)                                   # rr: along columns j (H,W-1); rc: along rows i (H-1,W)
    bad = np.nan_to_num(rr, nan=1.0); bad = (bad > hi) | (bad < lo)
    out[:, :-1] |= bad; out[:, 1:] |= bad
    bad = np.nan_to_num(rc, nan=1.0); bad = (bad > hi) | (bad < lo)
    out[:-1] |= bad; out[1:] |= bad
    return out & V


def duplicate_vertices(P, V, override=None):
    """Two valid vertices of DIFFERENT lattice cells closer than DUP_VOX (lattice neighbours included: a collapsed edge is a duplicate)."""
    from scipy.spatial import cKDTree
    out = np.zeros(V.shape, bool)
    idx = np.argwhere(V)
    if len(idx) < 2:
        return out
    pr = cKDTree(P[V]).query_pairs(r=tun("DUP_VOX", override), output_type="ndarray")
    if len(pr):
        for c in (0, 1):
            out[idx[pr[:, c], 0], idx[pr[:, c], 1]] = True
    return out


def orphan_vertices(P, V):
    """Valid vertices belonging to no complete valid quad (isolated / diagonal-only pinch points)."""
    q = V[:-1, :-1] & V[1:, :-1] & V[:-1, 1:] & V[1:, 1:]
    return V & ~quad_to_cells(q, V.shape)


def radial_jump(P, V):
    """(H,W) |offset of the cell from the mean of its 8 neighbours along the local normal| / median edge. NaN where the 8-neighbourhood is incomplete."""
    from .growth_guard import grid_normals, median_edge
    med = median_edge(P, V)
    nrm, ok = grid_normals(P, V)
    k = np.ones((3, 3)); k[1, 1] = 0
    cnt = ndi.convolve(V.astype(np.float64), k, mode="constant")
    s = np.stack([ndi.convolve(P[..., c] * V, k, mode="constant") for c in range(3)], -1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s / cnt[..., None]
    off = np.abs(np.einsum("...c,...c->...", P - mean, nrm)) / med
    return np.where(V & ok & (cnt == 8), off, np.nan)


def run_checks(X, Y, Z, V=None, override: dict | None = None, voxel_um: float = 9.362, sheet_pitch_sep: float | None = None) -> dict:
    """All checks -> {name: {'mask': bool (H,W), 'n_cells': int}, ...} with the continuous quantities under 'values'."""
    from . import growth_guard as GG, selfcontact as SC
    P, V = _valid(X, Y, Z, V)
    H, W = V.shape
    t = turning_angle(P, V); d = dihedral(P, V); j = radial_jump(P, V)
    out = {"values": {"turning_deg": t, "dihedral_deg": d, "radial_jump_rel": j}}
    m = {
        "turning": np.nan_to_num(t, nan=0.0) > tun("TURN_DEG", override),
        "crease": quad_to_cells(np.nan_to_num(d, nan=0.0) > tun("CREASE_DEG", override), (H, W)),
        "fold_over": quad_to_cells(fold_over(P, V), (H, W)),
        "normal_reversal": quad_to_cells(normal_reversal(P, V, override), (H, W)),
        "zero_area": quad_to_cells(zero_area(P, V, override), (H, W)),
        "edge_ratio": edge_ratio_mask(P, V, override),
        "duplicate_vertices": duplicate_vertices(P, V, override),
        "orphan_vertices": orphan_vertices(P, V),
        "radial_jump": np.nan_to_num(j, nan=0.0) > tun("JUMP_REL", override),
        "hairpin": GG.fold_mask(P, V, voxel_um, GG.GuardPolicy()),                       # wraps the existing criterion
        "self_proximity": SC.contact_report(X, Y, Z, V, min_sep=sheet_pitch_sep)["close_mask"],   # wraps selfcontact
    }
    for k, v in m.items():
        out[k] = {"mask": v & V, "n_cells": int((v & V).sum())}
    return out
