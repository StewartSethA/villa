"""Lattice geometry used by the growth guard: turn-radius (hairpin) statistics and PCA planarity.

Ported unchanged in behaviour from the source project's geometry-map module (the functions the
guard calls); pure numpy.
"""
from __future__ import annotations

import numpy as np

HAIRPIN_ARC_UM = 1000.0     # the arc over which a turn is measured
HAIRPIN_RADIUS_UM = 500.0   # below this the crease is implausible


def fold_metrics(P: np.ndarray, V: np.ndarray, voxel_um: float,
                 arc_um: float = HAIRPIN_ARC_UM, radius_um: float = HAIRPIN_RADIUS_UM,
                 max_lines: int = 256, seed: int = 0) -> dict:
    """Turn-radius statistics of a grown lattice, in micrometres.

    Returns {"geo_turn_radius_p05_um", "geo_fold_frac", "geo_hairpin_lines", "geo_lines_read"}:
    the 5th-percentile turn radius (small = creased), the fraction of measured arc sitting
    inside a turn tighter than `radius_um`, and how many lattice lines carried a hairpin.
    NaN when the lattice is too small or too holey to measure, which callers must treat as
    "unknown", never as "bad" -- absence is neutral (see scheduler.shape_score).

    Rows AND columns are read, because a fold can run along either lattice direction. At
    most `max_lines` of each, evenly spaced: this is a diagnostic over a lattice that can be
    10^6 points and it runs every grow round.
    """
    if voxel_um is None or not np.isfinite(voxel_um) or voxel_um <= 0:
        voxel_um = 1.0
    arc_vox = max(2.0, float(arc_um) / float(voxel_um))
    radii: list[float] = []
    arc_total = 0.0
    arc_folded = 0.0
    hairpin_lines = 0
    lines_read = 0
    for axis in (0, 1):
        n = P.shape[axis]
        if n < 3:
            continue
        idx = np.linspace(0, n - 1, min(n, max_lines)).astype(int)
        for k in np.unique(idx):
            line = P[k] if axis == 0 else P[:, k]
            vmask = V[k] if axis == 0 else V[:, k]
            pts = line[vmask]
            if pts.shape[0] < 5:
                continue
            d = np.diff(pts, axis=0)
            seglen = np.linalg.norm(d, axis=1)
            good = seglen > 1e-9
            if good.sum() < 4:
                continue
            d = d[good]; seglen = seglen[good]
            t = d / seglen[:, None]                      # unit tangents
            cum = np.concatenate([[0.0], np.cumsum(seglen)])
            lines_read += 1
            line_hairpin = False
            # compare the tangent at each step with the tangent one `arc_vox` further on
            j = np.searchsorted(cum, cum[:-1] + arc_vox)
            for i0, i1 in enumerate(j):
                if i1 >= t.shape[0]:
                    break
                arc = float(cum[i1] - cum[i0])
                if arc <= 1e-9:
                    continue
                cosang = float(np.clip(np.dot(t[i0], t[i1]), -1.0, 1.0))
                theta = float(np.arccos(cosang))
                r_um = (arc * voxel_um / theta) if theta > 1e-6 else float("inf")
                radii.append(r_um)
                arc_total += arc * voxel_um
                if r_um < radius_um:
                    arc_folded += arc * voxel_um
                    line_hairpin = True
            if line_hairpin:
                hairpin_lines += 1
    if not radii or arc_total <= 0:
        return {"geo_turn_radius_p05_um": float("nan"), "geo_fold_frac": float("nan"),
                "geo_hairpin_lines": 0, "geo_lines_read": lines_read}
    finite = [r for r in radii if np.isfinite(r)]
    p05 = float(np.percentile(finite, 5)) if finite else float("inf")
    return {"geo_turn_radius_p05_um": p05,
            "geo_fold_frac": float(arc_folded / arc_total),
            "geo_hairpin_lines": int(hairpin_lines),
            "geo_lines_read": int(lines_read)}


def planarity_score(P: np.ndarray, V: np.ndarray, max_pts: int = 200000, seed: int = 0) -> float:
    """PCA planarity of the valid 3-D points: (lambda2 - lambda3) / lambda1 of the
    point cloud's covariance eigenvalues (lambda1 >= lambda2 >= lambda3 >= 0). 1.0 is
    a perfect plane, 0.0 is not planar in any direction.

    This is the standard point-cloud "dimensionality feature" (Weinmann et al. 2015;
    also `linearity = (l1-l2)/l1`, `sphericity = l3/l1`), not something bespoke here.
    It is unlike the per-CELL `tilt` map above: tilt is local (one cell vs. its
    neighbours) and reads high on any small fold, while this is GLOBAL, so a sheet
    rolled into a tube -- large spread along the axis, roughly EQUAL spread in the two
    directions across it (l2 ~= l3) -- correctly scores low rather than being
    mistaken for planar because each small patch of it is locally flat.

    Subsampled at `max_pts` with a fixed seed: this is an O(n) reduction pass over a
    lattice that can be 10^6+ points, and the eigenvalues of a random 200k-point
    subsample of a smooth surface match the full set to far better precision than
    this score's own reporting resolution."""
    pts = P[V]
    if pts.shape[0] < 3:
        return float("nan")
    if pts.shape[0] > max_pts:
        rng = np.random.default_rng(seed)
        pts = pts[rng.choice(pts.shape[0], max_pts, replace=False)]
    c = pts - pts.mean(axis=0)
    cov = (c.T @ c) / max(pts.shape[0] - 1, 1)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]   # ascending -> descending
    l1, l2, l3 = float(ev[0]), float(ev[1]), float(ev[2])
    if l1 <= 0:
        return float("nan")
    return max(0.0, min(1.0, (l2 - l3) / l1))
