"""Self-CONTACT detector: the part of 'no degenerate surface' that the transverse-crossing census (vc_tifxyz_selfcross) does NOT see.

Principle (user, 2026-10-07): adjacent sheets are always separable, so collisions at any resolution are physically impossible. Beyond
transverse crossings this module flags, on a tifxyz lattice:
  * COINCIDENT   - two lattice samples of non-neighbouring cells closer than COINCIDENT_VOX (shared / duplicated vertices);
  * CLOSE        - non-neighbouring samples closer than MIN_SEP_VOX (grazing, coplanar and near-parallel overlapping patches all show up as
                   many such pairs); lattice-neighbourhood exclusion LATTICE_EXCL cells (Chebyshev);
  * LONG_EDGE    - quads with an edge longer than MAXEDGE_VOX: the transverse census drops these ("quads_dropped_for_edge_length"), so they
                   are NEVER checked there; here they are reported (not treated as contacts: they are an unchecked-region report).
Samples = vertices + edge midpoints + quad centres (half-cell sampling), so the distance to a facing sheet is measured to within ~step/4.
This is an APPROXIMATION of point-to-triangle distance, disclosed as such; it is not validated against human annotation (none exist).

MIN_SEP_VOX is a NAMED TUNABLE (default fitted/justified in docs/experiments/selfx_population_2026-10-07/STATE.md from the measured nearest
non-neighbour distance on crossing-free surfaces). `disclosure()` prints the value, units and basis; call it beside every use.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

MIN_SEP_VOX = 5.0          # voxels; sheets / patches closer than this (non-neighbouring cells) are a contact. Fit: see STATE.md (measured clean-surface nearest-neighbour distance)
COINCIDENT_VOX = 0.5       # voxels; shared / duplicated vertex
LATTICE_EXCL = 3           # cells (Chebyshev): samples this close on the lattice are the same sheet patch and never a contact
PARALLEL_DEG = 20.0        # degrees; patches whose normals differ by less than this (mod 180) are 'near-parallel' = coplanar-type contact
LOCAL_CELLS = 12           # cells; contacting samples this close on the lattice belong to one region of one sheet (fold / doubling-back / re-entry); farther = another wrap or distant part
MAXEDGE_VOX = 60.0         # voxels; the transverse detector's edge filter (vc_tifxyz_selfcross --maxedge default)


def disclosure(min_sep: float = MIN_SEP_VOX, n: int | None = None, area_cm2: float | None = None, basis: str = "") -> str:
    s = (f"selfcontact: min_sep={min_sep:g} vox (named tunable MIN_SEP_VOX), coincident<{COINCIDENT_VOX:g} vox, lattice_excl={LATTICE_EXCL} cells, "
         f"maxedge={MAXEDGE_VOX:g} vox, half-cell point sampling (approximate point-to-surface distance)")
    if n is not None:
        s += f"; scored n={n} segments" + (f", {area_cm2:.1f} cm2" if area_cm2 is not None else "")
    return s + (f"; basis: {basis}" if basis else "")


def _samples(P: np.ndarray, V: np.ndarray):
    """Sample points (N,3) with half-cell lattice coordinates (N,2) in units of cells*2, and the owning cell indices (N,2) of the corner cell."""
    H, W = V.shape
    pts, lat, own = [], [], []

    r, c = np.nonzero(V)
    pts.append(P[V]); lat.append(np.stack([2 * r, 2 * c], 1)); own.append(np.stack([r, c], 1))
    mh = V[:, :-1] & V[:, 1:]                               # horizontal edge midpoints
    r, c = np.nonzero(mh); pts.append((P[:, :-1][mh] + P[:, 1:][mh]) / 2); lat.append(np.stack([2 * r, 2 * c + 1], 1)); own.append(np.stack([r, c], 1))
    mv = V[:-1, :] & V[1:, :]
    r, c = np.nonzero(mv); pts.append((P[:-1][mv] + P[1:][mv]) / 2); lat.append(np.stack([2 * r + 1, 2 * c], 1)); own.append(np.stack([r, c], 1))
    mq = V[:-1, :-1] & V[1:, :-1] & V[:-1, 1:] & V[1:, 1:]
    r, c = np.nonzero(mq); pts.append((P[:-1, :-1][mq] + P[1:, :-1][mq] + P[:-1, 1:][mq] + P[1:, 1:][mq]) / 4); lat.append(np.stack([2 * r + 1, 2 * c + 1], 1)); own.append(np.stack([r, c], 1))
    return np.concatenate(pts), np.concatenate(lat), np.concatenate(own)


def _close_pairs(pts, lat, min_sep: float, ex: int):
    """Sample pairs within max(min_sep, COINCIDENT_VOX) whose lattice (Chebyshev, half-cell units) distance exceeds 2*ex. -> (pairs (n,2), dist (n,), lattice_dist_in_cells (n,))."""
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=max(min_sep, COINCIDENT_VOX), output_type="ndarray")
    if not len(pairs):
        return pairs, np.zeros(0), np.zeros(0)
    d = np.max(np.abs(lat[pairs[:, 0]] - lat[pairs[:, 1]]), axis=1)
    pairs = pairs[d > 2 * ex]
    d = d[d > 2 * ex]
    dist = np.linalg.norm(pts[pairs[:, 0]] - pts[pairs[:, 1]], axis=1) if len(pairs) else np.zeros(0)
    return pairs, dist, d / 2.0


def contact_pairs(X, Y, Z, V=None, min_sep: float | None = None, lattice_excl: int | None = None, parallel_deg: float = PARALLEL_DEG) -> dict:
    """Typed contact pairs for analysis. Types (distance d between the two samples of non-neighbouring cells, normals n1,n2 from the lattice):
      coincident : d < COINCIDENT_VOX (shared / duplicated vertex or edge point)
      grazing    : COINCIDENT_VOX <= d < min_sep and the surfaces are NOT near-parallel (angle(n1,n2) > parallel_deg and < 180-parallel_deg): a tangential touch
      coplanar   : d < min_sep and near-parallel (angle <= parallel_deg or >= 180-parallel_deg): overlapping patches pressed together
    Returned per pair: owner cells (ra,ca,rb,cb), d, lattice distance in cells, generation-free type code 0/1/2. `local` = lattice distance <= LOCAL_CELLS (same region of one sheet:
    a fold / doubling-back / re-entry of the grown surface) vs distant (another wrap or a far part of the sheet)."""
    from . import growth_guard as GG
    min_sep = MIN_SEP_VOX if min_sep is None else float(min_sep)
    ex = LATTICE_EXCL if lattice_excl is None else int(lattice_excl)
    X, Y, Z = (np.asarray(a, np.float64) for a in (X, Y, Z))
    if V is None:
        V = (X > 0) & (Y > 0) & (Z > 0)
    P = np.stack([X, Y, Z], -1)
    pts, lat, own = _samples(P, V)
    pairs, dist, latd = _close_pairs(pts, lat, min_sep, ex)
    keep = dist < min_sep
    pairs, dist, latd = pairs[keep], dist[keep], latd[keep]
    N, ok = GG.grid_normals(P, V)
    na = N[own[pairs[:, 0], 0], own[pairs[:, 0], 1]] if len(pairs) else np.zeros((0, 3))
    nb = N[own[pairs[:, 1], 0], own[pairs[:, 1], 1]] if len(pairs) else np.zeros((0, 3))
    oka = ok[own[pairs[:, 0], 0], own[pairs[:, 0], 1]] if len(pairs) else np.zeros(0, bool)
    okb = ok[own[pairs[:, 1], 0], own[pairs[:, 1], 1]] if len(pairs) else np.zeros(0, bool)
    cos = np.abs(np.einsum("ij,ij->i", na, nb)) if len(pairs) else np.zeros(0)
    par = cos >= np.cos(np.radians(parallel_deg))
    typ = np.where(dist < COINCIDENT_VOX, 0, np.where(par, 2, 1))
    return {"ra": own[pairs[:, 0], 0], "ca": own[pairs[:, 0], 1], "rb": own[pairs[:, 1], 0], "cb": own[pairs[:, 1], 1], "d": dist, "lat_cells": latd,
            "type": typ, "normals_ok": oka & okb, "local": latd <= LOCAL_CELLS, "min_sep": min_sep}


def contact_report(X, Y, Z, V=None, min_sep: float | None = None, lattice_excl: int | None = None, maxedge: float | None = None) -> dict:
    """-> {'close_mask','coincident_mask' (bool H,W: cells owning a contacting sample), 'n_close_pairs','n_coincident_pairs','long_edge_quads',
           'long_edge_mask','nn_dist' (per-sample nearest non-neighbour distance, for calibration), 'min_sep'...}."""
    min_sep = MIN_SEP_VOX if min_sep is None else float(min_sep)
    ex = LATTICE_EXCL if lattice_excl is None else int(lattice_excl)
    maxedge = MAXEDGE_VOX if maxedge is None else float(maxedge)
    X, Y, Z = (np.asarray(a, np.float64) for a in (X, Y, Z))
    if V is None:
        V = (X > 0) & (Y > 0) & (Z > 0)
    H, W = V.shape
    P = np.stack([X, Y, Z], -1)
    out = {"min_sep": min_sep, "close_mask": np.zeros((H, W), bool), "coincident_mask": np.zeros((H, W), bool), "long_edge_mask": np.zeros((H, W), bool)}
    # long-edge quads (never checked by the transverse census)
    q = V[:-1, :-1] & V[1:, :-1] & V[:-1, 1:] & V[1:, 1:]
    if q.any():
        e = np.stack([np.linalg.norm(P[:-1, :-1] - P[:-1, 1:], axis=-1), np.linalg.norm(P[1:, :-1] - P[1:, 1:], axis=-1),
                      np.linalg.norm(P[:-1, :-1] - P[1:, :-1], axis=-1), np.linalg.norm(P[:-1, 1:] - P[1:, 1:], axis=-1)]).max(0)
        le = q & (e > maxedge)
        out["long_edge_quads"] = int(le.sum()); out["long_edge_mask"][:-1, :-1] |= le
    else:
        out["long_edge_quads"] = 0
    if not V.any():
        out.update(n_close_pairs=0, n_coincident_pairs=0, nn_dist=np.zeros(0)); return out
    pts, lat, own = _samples(P, V)
    pairs, dist, _lat_d = _close_pairs(pts, lat, min_sep, ex)
    close = pairs[dist < min_sep]; coin = pairs[dist < COINCIDENT_VOX]
    for pp, key in ((close, "close_mask"), (coin, "coincident_mask")):
        if len(pp):
            for col in (0, 1):
                o = own[pp[:, col]]
                out[key][o[:, 0], o[:, 1]] = True
                # the owning cell's far corner too, so a flagged sample removes its whole local patch when culled
                out[key][np.minimum(o[:, 0] + 1, H - 1), o[:, 1]] = True; out[key][o[:, 0], np.minimum(o[:, 1] + 1, W - 1)] = True
    out["close_mask"] &= V; out["coincident_mask"] &= V
    out["n_close_pairs"] = int(len(close)); out["n_coincident_pairs"] = int(len(coin))
    # calibration: nearest non-neighbour sample distance per sample (searching to 4*min_sep)
    return out


def nn_distance(X, Y, Z, V=None, rmax: float = 60.0, lattice_excl: int | None = None) -> np.ndarray:
    """Per-sample distance to the nearest NON-neighbouring sample (<= rmax; rmax if none) -- the calibration quantity for MIN_SEP_VOX."""
    ex = LATTICE_EXCL if lattice_excl is None else int(lattice_excl)
    X, Y, Z = (np.asarray(a, np.float64) for a in (X, Y, Z))
    if V is None:
        V = (X > 0) & (Y > 0) & (Z > 0)
    pts, lat, _ = _samples(np.stack([X, Y, Z], -1), V)
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=rmax, output_type="ndarray")
    best = np.full(len(pts), rmax)
    if len(pairs):
        d = np.max(np.abs(lat[pairs[:, 0]] - lat[pairs[:, 1]]), axis=1)
        pairs = pairs[d > 2 * ex]
        dd = np.linalg.norm(pts[pairs[:, 0]] - pts[pairs[:, 1]], axis=1)
        np.minimum.at(best, pairs[:, 0], dd); np.minimum.at(best, pairs[:, 1], dd)
    return best
