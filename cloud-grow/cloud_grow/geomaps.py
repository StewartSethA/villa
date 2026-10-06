# VENDORED from the source fleet repo (src/vesuvius_pipeline/stages/geomaps.py), see ../VENDORED.json for the exact source md5, release and the patches applied.
# Do not edit here: re-vendor, so the guard stays the SAME code the production fleet runs.
"""Geometry maps: what the flattening did to the surface, and what the surface is.

One multi-panel PNG per segment (family "geo", published beside the predictions so
every gallery card, the inked page and the prospect pane pick it up), plus each
panel as its own PNG (families "geo_<panel>") for the click-to-open view.

Everything is computed on the FLAT lattice at lattice resolution: the flattened
tifxyz stores the 3-D position of every UV cell, so the map UV -> 3-D is the
lattice itself and its Jacobian is a central difference. Cheap (numpy only, plus
scipy for the shortcut-pair markers) -- seconds per segment, not minutes.

Scales. A lattice step is `step_size` in UV and `edge_med_vox` voxels in 3-D, and
those differ by 5.8-20.4x across one scroll (CLAUDE.md). So an "area ratio" here
is always RELATIVE to the segment's own median cell, never absolute, and every
absolute length is printed in voxels or micrometres from the Frame.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

FAMILY = "geo"
PANELS = ("stretch", "aniso", "normal", "tilt", "radius", "z", "curvature", "edge", "valid")


# ------------------------------------------------------------------ lattice


# ---- collapsed flattens ----------------------------------------------------------------
# 🔴 Four shipped flattens are collapses and one of them, PHerc0211_p6, is a 2x2 grid with
# ZERO valid points whose meta.json still reads 683.8 cm2 -- `vc_flatten` copies the
# grower's `area_cm2` verbatim, so a total collapse reports full area (CLAUDE.md). Nine
# panels of NaN over that header is the worst possible output: it looks like a measurement
# of a 683.8 cm2 surface. One panel that says what happened is the honest one.


def collapsed(flat_dir: str) -> dict | None:
    """The flatten's own verdict from `artifact_checks.check_flatten`, or None when it is
    healthy. Never re-implemented here: the checker is where the four collapse signals and
    their thresholds live, and a second copy would drift from it."""
    import sys as _sys
    # cloud-grow: the fleet's artifact_checks.py is not shipped; `collapsed` is a hub-side (flatten) check
    raise RuntimeError("geomaps.collapsed is not available in cloud-grow (needs the fleet's artifact_checks)")
    root = ""  # unreachable
    if root not in _sys.path:
        _sys.path.insert(0, root)
    try:
        import artifact_checks as AC
    except Exception:                          # noqa: BLE001 - no checker, no verdict
        return None
    try:
        res = AC.check_flatten(flat_dir)
    except Exception as e:                     # noqa: BLE001
        return {"why": f"check_flatten raised {type(e).__name__}: {e}", "grid": None, "valid": None}
    if not res.failures:
        return None
    grid = valid = None
    try:
        from PIL import Image as _I
        with _I.open(os.path.join(flat_dir, "x.tif")) as im:
            grid = f"{im.width}x{im.height}"
    except Exception:                          # noqa: BLE001
        pass
    for n in list(res.notes) + list(res.failures):
        if "valid points" in n.lower():
            valid = n.strip()
    return {"why": "; ".join(res.failures), "grid": grid, "valid": valid}


def write_collapsed_png(seg: str, out_dir: str, verdict: dict, area_cm2=None,
                        panels=PANELS) -> dict:
    """ONE panel saying the flatten collapsed, written to every name the composite would
    have used -- a card that fetches `geo_stretch.png` must not still get a blank plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    head = "FLATTEN COLLAPSED: %s grid, %s" % (verdict.get("grid") or "unknown",
                                               verdict.get("valid") or "0 valid points")
    body = (("area shown is INHERITED from the grower, not measured: %.1f cm2 (inherited)"
             % area_cm2) if area_cm2 else "no area was measured")
    out = {}
    for name in ("geo",) + tuple(panels):
        fig, ax = plt.subplots(1, 1, figsize=(7.4, 4.6), dpi=110)
        ax.axis("off")
        ax.text(0.5, 0.62, head, ha="center", va="center", fontsize=13, color="#a11",
                fontweight="bold", wrap=True)
        ax.text(0.5, 0.44, body, ha="center", va="center", fontsize=9, wrap=True)
        ax.text(0.5, 0.30, str(verdict.get("why") or "")[:300], ha="center", va="center",
                fontsize=7, color="#444", wrap=True)
        ax.text(0.5, 0.12, seg + "  ·  no geometry can be computed from a collapsed lattice",
                ha="center", va="center", fontsize=8, color="#666")
        pp = os.path.join(out_dir, ("geo.png" if name == "geo" else "geo_%s.png" % name))
        fig.savefig(pp); plt.close(fig)
        out[name] = pp
    return out


COLLAPSED_SUMMARY = {"flatten_collapsed": 1}


def read_lattice(tifxyz: str):
    """(P, V): P is (H,W,3) voxel coordinates, V the validity mask.

    The loader's own rule: a point with any of x/y/z <= 0 is invalid
    (`QuadSurface.cpp` invalidates on z <= 0; the -1 sentinel is written on all three)."""
    import tifffile
    X = tifffile.imread(os.path.join(tifxyz, "x.tif")).astype(np.float64)
    Y = tifffile.imread(os.path.join(tifxyz, "y.tif")).astype(np.float64)
    Z = tifffile.imread(os.path.join(tifxyz, "z.tif")).astype(np.float64)
    V = (X > 0) & (Y > 0) & (Z > 0)
    return np.stack([X, Y, Z], axis=-1), V


def _central(P: np.ndarray, V: np.ndarray, axis: int):
    """Central difference along `axis` with its own validity: both neighbours must
    be valid, so a derivative is never taken across a hole."""
    a = np.roll(P, -1, axis=axis)
    b = np.roll(P, 1, axis=axis)
    va = np.roll(V, -1, axis=axis)
    vb = np.roll(V, 1, axis=axis)
    ok = va & vb & V
    sl = [slice(None)] * V.ndim
    sl[axis] = 0
    ok[tuple(sl)] = False
    sl[axis] = -1
    ok[tuple(sl)] = False
    return (a - b) / 2.0, ok


def jacobian(P: np.ndarray, V: np.ndarray):
    """Per-cell first fundamental form of the UV -> 3-D map.

    Returns (area, s1, s2, theta, ok):
      area   3-D area of one UV cell, in voxels^2 (sqrt(det G))
      s1,s2  singular values of the Jacobian, s1 >= s2 (stretch along the two
             principal directions, in voxels per UV step)
      theta  UV-plane angle of the PRINCIPAL (largest) stretch direction, radians
      ok     cells where both derivatives exist
    """
    Pu, ou = _central(P, V, 0)
    Pv, ov = _central(P, V, 1)
    ok = ou & ov
    E = (Pu * Pu).sum(-1); F = (Pu * Pv).sum(-1); G = (Pv * Pv).sum(-1)
    det = E * G - F * F
    area = np.sqrt(np.maximum(det, 0.0))
    # eigenvalues of the symmetric 2x2 [[E,F],[F,G]] are the SQUARED singular values
    tr = E + G
    disc = np.sqrt(np.maximum(tr * tr / 4.0 - det, 0.0))
    l1 = np.maximum(tr / 2.0 + disc, 0.0)
    l2 = np.maximum(tr / 2.0 - disc, 0.0)
    s1, s2 = np.sqrt(l1), np.sqrt(l2)
    # eigenvector for l1: (F, l1 - E), degenerating to (1,0) when F == 0 and E >= G
    vx = np.where(np.abs(F) > 1e-12, F, np.where(E >= G, 1.0, 0.0))
    vy = np.where(np.abs(F) > 1e-12, l1 - E, np.where(E >= G, 0.0, 1.0))
    theta = np.arctan2(vy, vx)
    return area, s1, s2, theta, ok & (area > 0)


def anisotropy(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """s1/s2, >= 1. A cell with a collapsed direction reads inf; clip at the caller."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(s2 > 0, s1 / np.maximum(s2, 1e-12), np.inf)
    return r


# ---- fold-back / hairpin geometry -------------------------------------------
# A grown lattice sometimes DOUBLES BACK on itself: the tracer leaves the sheet it was
# following, turns through a crease far tighter than papyrus can take, and comes back along
# the neighbouring wrap. The resulting segment looks large and can even score well on
# `planarity_score` (each half of the fold is locally flat) and on material_frac (it is
# sitting on material -- the wrong material), but a fold like that is not one surface, and
# ink read across it is read across two different sheets.
#
# `planarity_score` cannot see this and neither can `tilt`: the first is global and the
# second is one cell against its neighbours. What distinguishes a hairpin is its RADIUS.
# Walk each lattice row in 3-D, measure the direction before and after a fixed arc length,
# and the turn radius is R = L / theta for a turn of theta radians over arc L. A real scroll
# wrap has a radius of millimetres even at the core; a crease of a few hundred micrometres
# is not something a papyrus sheet does, so it is the tracer having jumped.
# User, 2026-09-20: "some segment growth jobs ended up doubling back on themselves. Too
# tight creases including hairpin turns are usually implausible."
HAIRPIN_ARC_UM = float(os.environ.get("VPIPE_HAIRPIN_ARC_UM", "1000"))     # the arc over which a turn is measured
HAIRPIN_RADIUS_UM = float(os.environ.get("VPIPE_HAIRPIN_RADIUS_UM", "500"))  # below this the crease is implausible


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


def squareness_score(V: np.ndarray) -> float:
    """How close the segment's UV footprint is to a square, in (0, 1].

    Two OpenCV measures, multiplied: `extent` (contour area / minAreaRect area --
    1.0 means the footprint fills its own bounding rectangle, low means an
    irregular or concave outline) times the rectangle's own aspect ratio
    (short side / long side -- 1.0 means that rectangle is a square, low means a
    long strip). Multiplying them means a long, clean rectangle and a ragged
    square-ish blob both score below either single-cause defect."""
    import cv2
    mask = np.ascontiguousarray(V.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return float("nan")
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area <= 0:
        return float("nan")
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    if w <= 0 or h <= 0:
        return float("nan")
    extent = area / (w * h)
    aspect = min(w, h) / max(w, h)
    return max(0.0, min(1.0, extent * aspect))


def normals(P: np.ndarray, V: np.ndarray):
    """Unit surface normal per cell (Pu x Pv) and the angle to the MEAN normal, in
    degrees: folds and creases are exactly where that angle jumps."""
    Pu, ou = _central(P, V, 0)
    Pv, ov = _central(P, V, 1)
    ok = ou & ov
    N = np.cross(Pu, Pv)
    n = np.linalg.norm(N, axis=-1)
    ok = ok & (n > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        N = N / np.where(n[..., None] > 0, n[..., None], 1.0)
    mean = N[ok].mean(axis=0) if ok.any() else np.array([0.0, 0.0, 1.0])
    m = np.linalg.norm(mean)
    mean = mean / m if m > 0 else np.array([0.0, 0.0, 1.0])
    tilt = np.degrees(np.arccos(np.clip((N * mean).sum(-1), -1.0, 1.0)))
    return N, tilt, ok, mean


def mean_curvature(P: np.ndarray, V: np.ndarray):
    """H = (EN - 2FM + GL) / (2(EG - F^2)), in 1/voxel. Second derivatives are the
    plain 5-point stencils; a cell touching a hole is dropped rather than one-sided."""
    Pu, ou = _central(P, V, 0)
    Pv, ov = _central(P, V, 1)
    Puu, ouu = _central(Pu, ou, 0)
    Pvv, ovv = _central(Pv, ov, 1)
    Puv, ouv = _central(Pu, ou, 1)
    ok = ou & ov & ouu & ovv & ouv
    E = (Pu * Pu).sum(-1); F = (Pu * Pv).sum(-1); G = (Pv * Pv).sum(-1)
    N = np.cross(Pu, Pv)
    nn = np.linalg.norm(N, axis=-1)
    ok = ok & (nn > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = N / np.where(nn[..., None] > 0, nn[..., None], 1.0)
        L = (Puu * n).sum(-1); M = (Puv * n).sum(-1); Nc = (Pvv * n).sum(-1)
        den = 2.0 * (E * G - F * F)
        H = np.where(den != 0, (E * Nc - 2 * F * M + G * L) / np.where(den != 0, den, 1.0), 0.0)
    return H, ok


def edge_lengths(P: np.ndarray, V: np.ndarray):
    """Per-cell mean 3-D length of its valid lattice edges, in voxels, and the
    median over the segment. An edge over 3x the median JUMPS A GAP -- that is the
    wrap-jump detector, and leaving those in makes a hole read as distortion."""
    acc = np.zeros(V.shape); cnt = np.zeros(V.shape)
    lens = []
    for axis in (0, 1):
        d = np.linalg.norm(np.roll(P, -1, axis=axis) - P, axis=-1)
        ok = V & np.roll(V, -1, axis=axis)
        sl = [slice(None)] * V.ndim
        sl[axis] = -1
        ok[tuple(sl)] = False
        acc += np.where(ok, d, 0.0); cnt += ok
        back = np.roll(np.where(ok, d, 0.0), 1, axis=axis)
        bok = np.roll(ok, 1, axis=axis)
        acc += np.where(bok, back, 0.0); cnt += bok
        lens.append(d[ok])
    e = np.concatenate(lens) if lens else np.array([])
    med = float(np.median(e)) if e.size else float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        cell = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    return cell, med, e


def edge_spread(e: np.ndarray, med: float, factor: float = 3.0) -> dict:
    """p10/p90 of edge length relative to the median, and the jump rate at `factor`."""
    if not e.size or not (med > 0):
        return {"p10": float("nan"), "p90": float("nan"), "jump_frac": float("nan"), "med_vox": med}
    r = e / med
    return {"p10": float(np.percentile(r, 10)), "p90": float(np.percentile(r, 90)),
            "jump_frac": float(np.mean(r > factor)), "med_vox": float(med)}


# ------------------------------------------------------------------ scroll frame

def umbilicus_xy(root, scroll: str):
    """(z, x, y) control points of the scroll axis, or None. `umbilicus.json` is the
    ray-crossing detector's output (tools/umbilicus_auto/output/<scroll>/)."""
    d = os.path.join(str(root), "tools", "umbilicus_auto", "output", scroll or "")
    for name in ("umbilicus.json", "umbilicus_dense.json"):
        p = os.path.join(d, name)
        if not os.path.isfile(p):
            continue
        try:
            pts = json.load(open(p)).get("control_points") or []
            a = np.array([[float(q["z"]), float(q["x"]), float(q["y"])] for q in pts])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if len(a) >= 2:
            a = a[np.argsort(a[:, 0])]
            return a
    return None


def radius_map(P: np.ndarray, V: np.ndarray, axis: np.ndarray | None):
    """Distance from the scroll axis in the xy plane, in voxels, plus how the axis
    was obtained. With no umbilicus the fallback is the segment's own per-z centroid,
    which is a LOCAL centre and not the scroll axis -- labelled as such on the panel."""
    Z = P[..., 2]
    if axis is not None:
        cx = np.interp(Z, axis[:, 0], axis[:, 1])
        cy = np.interp(Z, axis[:, 0], axis[:, 2])
        src = "umbilicus"
    else:
        cx = np.full(Z.shape, P[..., 0][V].mean() if V.any() else 0.0)
        cy = np.full(Z.shape, P[..., 1][V].mean() if V.any() else 0.0)
        src = "lattice centroid (no umbilicus)"
    r = np.hypot(P[..., 0] - cx, P[..., 1] - cy)
    return np.where(V, r, np.nan), src


# ------------------------------------------------------------------ topology

def shortcut_points(P: np.ndarray, V: np.ndarray, max_pts: int = 20000, n_sources: int = 48,
                    geo_min: float = 600.0, euc_max: float = 15.0, seed: int = 0):
    """Lattice cells that participate in a SHORTCUT PAIR -- far along the surface,
    near in 3-D -- i.e. where the sheet folds back onto itself. Same criterion as
    scripts/segment_yield.py (8-connected graph, Dijkstra, reduce over POINTS)."""
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra
    except ImportError:
        return None
    H, W = V.shape
    step = max(1, int(math.ceil(math.sqrt(max(1, int(V.sum())) / max_pts))))
    S = np.zeros_like(V)
    S[::step, ::step] = True
    M = V & S
    n = int(M.sum())
    if n < 64:
        return None
    idx = -np.ones(V.shape, np.int64)
    idx[M] = np.arange(n)
    Q = P[M]
    r, c, w = [], [], []
    for dy, dx in ((0, step), (step, 0), (step, step), (step, -step)):
        if dx >= 0:
            a = idx[:H - dy, :W - dx]; b = idx[dy:, dx:]
        else:
            a = idx[:H - dy, -dx:];    b = idx[dy:, :W + dx]
        m = (a >= 0) & (b >= 0)
        ai, bi = a[m], b[m]
        if not len(ai):
            continue
        d = np.linalg.norm(Q[ai] - Q[bi], axis=1)
        r += [ai, bi]; c += [bi, ai]; w += [d, d]
    if not r:
        return None
    g = coo_matrix((np.concatenate(w), (np.concatenate(r), np.concatenate(c))), shape=(n, n)).tocsr()
    rng = np.random.default_rng(seed)
    src = rng.choice(n, min(n_sources, n), replace=False)
    D = dijkstra(g, directed=False, indices=src)
    Eu = np.linalg.norm(Q[src][:, None, :] - Q[None, :, :], axis=2)
    bad = (D > geo_min) & (Eu < euc_max)
    hit = bad.any(0)
    out = np.zeros_like(V)
    out[M] = hit
    return out


# ------------------------------------------------------------------ figure

def _finite(a: np.ndarray):
    f = a[np.isfinite(a)]
    return f


def compute(flat_dir: str, root=None, scroll: str = "", voxel_um: float = 9.362,
            render_frame=None, with_topology: bool = True) -> dict:
    """Every map plus the summary numbers. Numpy (cv2 for the squareness only); no plotting.

    `render_frame` is the segment's render Frame; the caption quotes ITS pitch and
    nothing here ever computes a micrometre pitch by hand (that division was a 20x
    error once). Voxel lengths come from `voxel_um`, the scan's own pitch."""
    P, V = read_lattice(flat_dir)
    area, s1, s2, theta, jok = jacobian(P, V)
    med_area = float(np.median(area[jok])) if jok.any() else float("nan")
    logdet = np.where(jok & (area > 0) & (med_area > 0), np.log2(np.maximum(area, 1e-12) / max(med_area, 1e-12)), np.nan)
    aniso = np.where(jok, anisotropy(s1, s2), np.nan)
    N, tilt, nok = normals(P, V)[:3]
    H, cok = mean_curvature(P, V)
    cell_edge, med_edge, e = edge_lengths(P, V)
    spread = edge_spread(e, med_edge)
    rad, rad_src = radius_map(P, V, umbilicus_xy(root or ".", scroll))
    short = shortcut_points(P, V) if with_topology else None
    ar = _finite(area[jok])
    dist_cv = float(ar.std() / ar.mean()) if ar.size and ar.mean() > 0 else float("nan")
    an = _finite(aniso[jok])
    try:
        sq = squareness_score(V)
    except ImportError:
        sq = float("nan")   # the fleet venvs carry no cv2 (grow.py guards the same call); the score is absent, not fatal
    return {
        "P": P, "V": V, "ok": jok,
        "logdet": logdet, "aniso": aniso, "theta": theta,
        "normal": N, "tilt": np.where(nok, tilt, np.nan),
        "curv": np.where(cok, H, np.nan),
        "edge": cell_edge, "edge_med_vox": med_edge, "edge_spread": spread,
        "radius": rad, "radius_source": rad_src, "z": np.where(V, P[..., 2], np.nan),
        "shortcut": short,
        "voxel_um": voxel_um, "render_frame": render_frame,
        "summary": {
            "geo_distortion_cv": dist_cv,
            "geo_aniso_median": float(np.median(an)) if an.size else float("nan"),
            "geo_edge_p10": spread["p10"], "geo_edge_p90": spread["p90"],
            "geo_edge_jump_frac": spread["jump_frac"],
            "geo_tilt_p90_deg": float(np.percentile(_finite(tilt[nok]), 90)) if nok.any() else float("nan"),
            "geo_valid_frac": float(V.mean()),
            "geo_shortcut_frac": (float(short[V].mean()) if short is not None and V.any() else float("nan")),
            "geo_planarity": planarity_score(P, V),
            "geo_squareness": sq,
        },
    }


def _panel(ax, img, title, cbar_label, cmap="viridis", vmin=None, vmax=None, rgb=False):
    import matplotlib.pyplot as plt
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    if rgb:
        ax.imshow(img, interpolation="nearest")
        return
    im = ax.imshow(np.ma.masked_invalid(img), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=6)
    cb.set_label(cbar_label, fontsize=6)


def draw_panel(ax, g: dict, name: str) -> None:
    """One named panel onto one axis. Every panel carries its units."""
    import numpy as _np
    vox_um = g["voxel_um"]
    if name == "stretch":
        v = _np.nanpercentile(_np.abs(g["logdet"]), 98) if _np.isfinite(g["logdet"]).any() else 1.0
        v = float(max(v, 0.1))
        _panel(ax, g["logdet"], "area distortion", "log2 (cell area / median)", cmap="RdBu_r", vmin=-v, vmax=v)
    elif name == "aniso":
        a = _np.clip(g["aniso"], 1.0, _np.nanpercentile(g["aniso"], 98) if _np.isfinite(g["aniso"]).any() else 3.0)
        _panel(ax, a, "anisotropy + principal stretch", "s1 / s2 (ratio, 1 = isotropic)", cmap="magma")
        H, W = g["V"].shape
        st = max(1, int(max(H, W) / 24))
        ys, xs = _np.mgrid[0:H:st, 0:W:st]
        th = g["theta"][::st, ::st]; ok = _np.isfinite(th) & g["ok"][::st, ::st]
        L = 0.45 * st
        ax.quiver(xs[ok], ys[ok], (L * _np.cos(th))[ok], (L * _np.sin(th))[ok],
                  angles="xy", scale_units="xy", scale=1, headwidth=0, headlength=0,
                  color="w", linewidth=0.4, pivot="mid", alpha=0.7)
    elif name == "normal":
        N = g["normal"]
        rgb = _np.clip((N + 1.0) / 2.0, 0, 1)
        rgb[~g["V"]] = 0.0
        _panel(ax, rgb, "normal in the scroll frame", "", rgb=True)
    elif name == "tilt":
        _panel(ax, g["tilt"], "normal tilt vs mean (folds/creases)", "degrees", cmap="inferno", vmin=0,
               vmax=float(_np.nanpercentile(g["tilt"], 99)) if _np.isfinite(g["tilt"]).any() else 90.0)
    elif name == "radius":
        _panel(ax, g["radius"] * vox_um / 1000.0, "radius from the axis (%s)" % g["radius_source"], "mm", cmap="cividis")
    elif name == "z":
        _panel(ax, g["z"] * vox_um / 1000.0, "z (sheet climb)", "mm", cmap="cool")
    elif name == "curvature":
        c = g["curv"] * 1000.0 / vox_um          # 1/voxel -> 1/mm
        v = float(_np.nanpercentile(_np.abs(c), 98)) if _np.isfinite(c).any() else 1.0
        _panel(ax, c, "mean curvature", "1/mm", cmap="PuOr", vmin=-v, vmax=v)
    elif name == "edge":
        med = g["edge_med_vox"]
        _panel(ax, g["edge"], "lattice edge (median %.1f vox = %.1f um/step)" % (med, med * vox_um),
               "voxels per step", cmap="viridis", vmin=0,
               vmax=float(3.0 * med) if med == med else None)
        jump = _np.isfinite(g["edge"]) & (g["edge"] > 3.0 * med)
        ys, xs = _np.nonzero(jump)
        if xs.size:
            ax.plot(xs, ys, ",", color="red")
    elif name == "valid":
        base = _np.where(g["V"], 1.0, 0.0)
        _panel(ax, base, "validity + shortcut pairs", "1 = surface, 0 = hole", cmap="Greys_r", vmin=0, vmax=1)
        s = g.get("shortcut")
        if s is not None:
            ys, xs = _np.nonzero(s)
            if xs.size:
                ax.plot(xs, ys, ".", color="#ff3355", markersize=1.2, alpha=0.8)
    else:
        raise ValueError("unknown panel " + name)


def caption(seg: str, g: dict, area_cm2: float | None) -> str:
    s = g["summary"]
    bits = [seg]
    if area_cm2:
        bits.append("%.1f cm2" % area_cm2)
    fr = g.get("render_frame")
    if fr is not None:
        bits.append("%.2f um/render px" % fr.um_per_px)
    bits.append("%.3f um/voxel" % g["voxel_um"])
    bits.append("distortion CV %.3f" % s["geo_distortion_cv"])
    bits.append("anisotropy median %.2f" % s["geo_aniso_median"])
    bits.append("edge spread p10/p90 %.2f/%.2f" % (s["geo_edge_p10"], s["geo_edge_p90"]))
    bits.append("planarity %.2f · squareness %.2f" % (s["geo_planarity"], s["geo_squareness"]))
    return "  ·  ".join(bits)


def write_pngs(g: dict, seg: str, out_dir: str, area_cm2: float | None = None,
               panels=PANELS) -> dict:
    """The composite (out_dir/geo.png) and one PNG per panel. Returns {name: path}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    out = {}
    H, W = g["V"].shape
    ar = max(0.35, min(2.5, H / max(W, 1)))
    ncol = 3
    nrow = int(math.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.2 * ar * nrow + 0.6), dpi=110)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(panels):]:
        ax.axis("off")
    for ax, name in zip(axes, panels):
        try:
            draw_panel(ax, g, name)
        except Exception as e:                      # noqa: BLE001 - one bad panel must not lose the sheet
            ax.axis("off"); ax.set_title("%s failed: %s" % (name, type(e).__name__), fontsize=7)
    fig.suptitle(caption(seg, g, area_cm2), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = os.path.join(out_dir, "geo.png")
    fig.savefig(p); plt.close(fig)
    out["geo"] = p
    for name in panels:
        f2, a2 = plt.subplots(1, 1, figsize=(6.0, 6.0 * ar + 0.4), dpi=120)
        try:
            draw_panel(a2, g, name)
        except Exception as e:                      # noqa: BLE001
            a2.axis("off"); a2.set_title("%s failed: %s" % (name, type(e).__name__), fontsize=8)
        f2.suptitle(caption(seg, g, area_cm2), fontsize=7)
        f2.tight_layout(rect=(0, 0, 1, 0.95))
        pp = os.path.join(out_dir, "geo_%s.png" % name)
        f2.savefig(pp); plt.close(f2)
        out[name] = pp
    return out
