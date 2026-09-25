"""3D-first fusion of overlapping tifxyz lattices of ONE sheet into one deduplicated lattice (PROTOTYPE).

Why. Merging two lattices through their *parameter* (UV) frames fails when the parameter overlap is not
the surfaces' 3-D overlap (in the source project 91 % of 6,086 candidate pairs on one scroll were rejected
for that reason). This module never uses a member's UV frame. It works in a
FRAME derived from geometry, in which one sheet is a height field h(u, v):

    cyl    cylinder about the scroll's umbilicus: u = arc length (vox) at the cluster's mean radius,
           v = z, h = r - r0.            A wrap is a graph r(theta, z); the next wrap is ~25-35 vox
           further out at the same (theta, z), so adjacent wraps are separable by h.
    plane  the cluster's PCA plane: u, v in-plane, h along the plane normal. No umbilicus needed.

Every member point becomes (u, v, h). A fresh regular grid is laid over (u, v); each node takes the
CONSENSUS h of the member points around it. Points at one (u, v) whose h splits into two modes more
than `jump_tol` apart are two sheets (an adjacent wrap) or a sheet jump: the mode continuous with the
neighbouring nodes wins, the other is DROPPED AND COUNTED (never averaged in), so a wrong wrap can
lower coverage but cannot bend the surface. Points whose normal is far from the frame's h axis
(hairpins, steep folds: not a graph) are dropped and counted too.

Output is a tifxyz (x/y/z.tif, -1 invalid) plus a sidecar with the members, frame, parameters and every
count above. Sources are never modified. Validation lives in `validate()`: it re-measures the product
with code that did not build it (same_sheet's estimator, growth_guard's fold/plan masks and a CT sampler),
see the docstring there.

Other entry points: `graft` (keep the best member's lattice intact and grow it over the others), `verify`
(anchor a member to the surface prediction), `setcover` (which members can be retired without losing surface),
`sheet_groups`, `edge_ridge_runs` (does a lattice edge step from one wrap onto the next?). VALIDATION.md states
what has and has not been measured, including what did not work.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from . import growth_guard as G
from . import same_sheet as C

TILT_COS = float(np.cos(np.radians(50.0)))     # |n . h_axis| below this: not a graph over the frame


# ------------------------------------------------------------------------------------ members
def load_member(seg: str, path: str, kind: str = "tifxyz") -> dict:
    """A member from a tifxyz directory (kind "tifxyz") or a pushed blob (kind "pushed"; see same_sheet.read_lattice)."""
    X, Y, Z = C.read_lattice(kind, path)
    sp = C.surface_points(X, Y, Z)
    return {"seg": seg, "X": X, "Y": Y, "Z": Z, "sp": sp, "edge": sp["edge_med"]}


def member_from_arrays(seg, X, Y, Z) -> dict:
    X, Y, Z = (np.asarray(a, np.float32) for a in (X, Y, Z))
    sp = C.surface_points(X, Y, Z)
    return {"seg": seg, "X": X, "Y": Y, "Z": Z, "sp": sp, "edge": sp["edge_med"]}


# ------------------------------------------------------------------------------------ frames
class CylFrame:
    """Cylinder about the umbilicus polyline (x(z), y(z))."""
    name = "cyl"

    def __init__(self, umbilicus_points: list[dict], xyz: np.ndarray):
        cp = sorted(umbilicus_points, key=lambda d: d["z"])
        self.cz = np.array([d["z"] for d in cp], float)
        self.cx = np.array([d["x"] for d in cp], float)
        self.cy = np.array([d["y"] for d in cp], float)
        r, th = self._polar(xyz)
        self.r0 = float(np.median(r))
        self.th0 = float(np.arctan2(np.sin(th).mean(), np.cos(th).mean()))
        dth = self._wrap(th - self.th0)
        self.span_deg = float(np.degrees(dth.max() - dth.min()))

    def _centre(self, z):
        return np.interp(z, self.cz, self.cx), np.interp(z, self.cz, self.cy)

    def _polar(self, xyz):
        cx, cy = self._centre(xyz[:, 2])
        dx, dy = xyz[:, 0] - cx, xyz[:, 1] - cy
        return np.hypot(dx, dy), np.arctan2(dy, dx)

    @staticmethod
    def _wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def fwd(self, xyz, nrm=None):
        r, th = self._polar(xyz)
        u = self._wrap(th - self.th0) * self.r0
        h = r - self.r0
        out = np.c_[u, xyz[:, 2], h]
        if nrm is None:
            return out
        cx, cy = self._centre(xyz[:, 2])
        rh = np.c_[xyz[:, 0] - cx, xyz[:, 1] - cy, np.zeros(len(xyz))]
        rh /= np.maximum(np.linalg.norm(rh, axis=1, keepdims=True), 1e-9)
        return out, np.einsum("ij,ij->i", nrm, rh)             # signed n . r_hat

    def inv(self, u, v, h):
        th = self.th0 + u / self.r0
        r = self.r0 + h
        cx, cy = self._centre(v)
        return np.c_[cx + r * np.cos(th), cy + r * np.sin(th), v]

    def meta(self):
        return {"frame": "cyl", "r0_vox": round(self.r0, 1), "theta0_deg": round(float(np.degrees(self.th0)), 2),
                "span_deg": round(self.span_deg, 1)}

    def usable(self):
        return self.span_deg < 150.0 and self.r0 > 300.0


class PlaneFrame:
    name = "plane"

    def __init__(self, xyz: np.ndarray, nrm: np.ndarray | None = None):
        self.c = xyz.mean(axis=0)
        _, _, vt = np.linalg.svd(xyz - self.c, full_matrices=False)
        self.e = vt                                              # rows: e1, e2, e3(normal)
        self.span_deg = 0.0
        q = (xyz - self.c) @ self.e.T
        self.extent = (q.max(axis=0) - q.min(axis=0)).tolist()

    def fwd(self, xyz, nrm=None):
        q = (xyz - self.c) @ self.e.T
        if nrm is None:
            return q
        return q, nrm @ self.e[2]

    def inv(self, u, v, h):
        return self.c + np.c_[u, v, h] @ self.e

    def meta(self):
        return {"frame": "plane", "extent_vox": [round(x, 0) for x in self.extent]}

    def usable(self):
        return True


# ------------------------------------------------------------------------------------ fusion
def _modes(h: np.ndarray, jump_tol: float):
    """h (N, K) with NaN for missing. Returns mode id (N, K) (-1 where NaN): sorted-gap clustering."""
    order = np.argsort(np.where(np.isnan(h), np.inf, h), axis=1)
    hs = np.take_along_axis(h, order, axis=1)
    gap = np.zeros_like(hs, bool)
    d = np.diff(hs, axis=1)
    gap[:, 1:] = np.nan_to_num(d, nan=0.0) > jump_tol
    mid_sorted = np.cumsum(gap, axis=1)
    mid_sorted[np.isnan(hs)] = -1
    mid = np.empty_like(mid_sorted)
    np.put_along_axis(mid, order, mid_sorted, axis=1)
    return mid


def fuse(members: list[dict], frame, spacing: float | None = None, rho_f: float = 0.8, jump_tol: float = 6.0,
         K: int = 24, despike: float = 3.0, seed_seg: str | None = None) -> dict:
    """Consensus fusion. Returns {"P": (H,W,3) float32 (-1 invalid), "diag": {...}, "support": (H,W) members-per-node}."""
    pts, mids, tilt_drop, tot = [], [], 0, 0
    sign_votes = 0.0
    for k, m in enumerate(members):
        sp = m["sp"]
        if not len(sp["xyz"]):
            continue
        ok = sp["nok"]
        f, ndot = frame.fwd(sp["xyz"][ok].astype(np.float64), sp["nrm"][ok].astype(np.float64))
        good = np.abs(ndot) >= TILT_COS
        tot += len(good)
        tilt_drop += int((~good).sum())
        sign_votes += float(np.sign(ndot[good]).sum())
        pts.append(f[good])
        mids.append(np.full(int(good.sum()), k))
    if not pts or sum(len(p) for p in pts) == 0:
        return {"P": None, "diag": {"error": "no usable points (all dropped: normals not along the frame's h axis)",
                                    "member_points": int(tot), "dropped_tilt_frac": round(tilt_drop / max(1, tot), 4)}}
    F = np.concatenate(pts)
    M = np.concatenate(mids)
    edges = [m["edge"] for m in members if np.isfinite(m["edge"])]
    s = float(spacing or np.clip(np.median(edges), 8.0, 20.0))
    u0, v0 = F[:, 0].min(), F[:, 1].min()
    H = int((F[:, 1].max() - v0) / s) + 1
    W = int((F[:, 0].max() - u0) / s) + 1
    if H * W > 4_000_000:
        return {"P": None, "diag": {"error": f"grid {H}x{W} too large"}}
    gi, gj = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    nodes = np.c_[u0 + gj.ravel() * s, v0 + gi.ravel() * s]
    tree = cKDTree(F[:, :2])
    rho = rho_f * s
    d, idx = tree.query(nodes, k=K, distance_upper_bound=rho, workers=1)
    have = np.isfinite(d)
    hh = np.where(have, F[np.where(have, idx, 0), 2], np.nan)
    mm = np.where(have, M[np.where(have, idx, 0)], -1)
    n_any = have.any(axis=1)
    mid = _modes(hh, jump_tol)
    nmodes = mid.max(axis=1) + 1
    nmodes[~n_any] = 0

    Hf = np.full(H * W, np.nan)
    single = nmodes == 1
    with np.errstate(all="ignore"):
        med0 = np.nanmedian(np.where((mid == 0) & have, hh, np.nan), axis=1)      # a single-mode node: all its points
    Hf[single] = med0[single]
    multi = np.nonzero(nmodes >= 2)[0]
    conflict = np.zeros(H * W, bool)
    conflict[multi] = True
    # resolve conflicts by continuity with already-assigned neighbours, a few passes
    for _ in range(8):
        todo = [n for n in multi if np.isnan(Hf[n])]
        if not todo:
            break
        gridp = np.pad(Hf.reshape(H, W), 1, constant_values=np.nan)
        progressed = 0
        for n in todo:
            i, j = divmod(n, W)
            nb = gridp[i:i + 3, j:j + 3].ravel()
            nb = nb[~np.isnan(nb)]
            if len(nb) < 2:
                continue
            ref = float(np.median(nb))
            best, bd = None, 1e9
            for mo in range(int(nmodes[n])):
                sel = (mid[n] == mo) & have[n]
                if not sel.any():
                    continue
                mh = float(np.median(hh[n][sel]))
                if abs(mh - ref) < bd:
                    best, bd = mh, abs(mh - ref)
            if best is not None and bd <= jump_tol:
                Hf[n] = best
                progressed += 1
        if not progressed:
            break
    valid = ~np.isnan(Hf.reshape(H, W))
    # despike, then keep the largest connected component
    g = Hf.reshape(H, W).copy()
    nbmed = ndi.generic_filter(np.where(valid, g, np.nan), np.nanmedian, size=3, mode="constant", cval=np.nan) \
        if valid.sum() < 200_000 else None
    n_spike = 0
    if nbmed is not None:
        sp_ = valid & (np.abs(g - nbmed) > despike * 1.0) & np.isfinite(nbmed)
        # a single spike node with >= 5 valid neighbours is an outlier; isolated edge nodes are kept
        cnt = ndi.convolve(valid.astype(int), np.ones((3, 3), int), mode="constant") - 1
        sp_ &= cnt >= 5
        n_spike = int(sp_.sum())
        valid &= ~sp_
    lab, nl = ndi.label(valid, structure=np.ones((3, 3), int))
    n_comp = int(nl)
    if nl > 1:
        sizes = ndi.sum(valid, lab, index=np.arange(1, nl + 1))
        valid = lab == (1 + int(np.argmax(sizes)))
    g = np.where(valid, g, np.nan)
    # support = distinct members contributing to the chosen node (approx: members in the node's neighbourhood)
    support = np.zeros(H * W, int)
    for n in np.nonzero(valid.ravel())[0]:
        sel = have[n] & (np.abs(hh[n] - g.ravel()[n]) <= jump_tol)
        support[n] = len(set(mm[n][sel]))
    support = support.reshape(H, W)
    uu = (u0 + gj * s).astype(float)
    vv = (v0 + gi * s).astype(float)
    xyz = frame.inv(uu.ravel(), vv.ravel(), np.nan_to_num(g.ravel(), nan=0.0)).reshape(H, W, 3)
    # orientation: the product's normal (du x dv) should agree with the members' majority (n . h_axis sign)
    flip = sign_votes < 0
    P = np.where(valid[..., None], xyz, -1.0).astype(np.float32)
    if flip:
        P = P[:, ::-1].copy()
        valid = valid[:, ::-1].copy()
        support = support[:, ::-1].copy()
    # drop empty border rows/cols
    rr, cc = np.nonzero(valid)
    P = P[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
    support = support[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
    n_valid = int((P[..., 0] > 0).sum())
    diag = {"frame": frame.meta(), "spacing_vox": round(s, 2), "grid": list(P.shape[:2]), "nodes_valid": n_valid,
            "member_points": int(tot), "dropped_tilt_frac": round(tilt_drop / max(1, tot), 4),
            "conflict_nodes": int(conflict.sum()), "conflict_frac": round(float(conflict.sum()) / max(1, int(n_any.sum())), 4),
            "conflict_unresolved": int((conflict & np.isnan(Hf)).sum()), "despiked_nodes": n_spike,
            "components_before_largest": n_comp, "orientation_flipped": bool(flip), "jump_tol_vox": jump_tol,
            "support_median": float(np.median(support[support > 0])) if (support > 0).any() else 0.0,
            "support_ge2_frac": round(float((support >= 2).sum() / max(1, (support > 0).sum())), 4)}
    return {"P": P, "diag": diag, "support": support}


# ------------------------------------------------------------------------------------ output
def write_tifxyz(P: np.ndarray, out: Path, meta: dict) -> None:
    import tifffile
    out.mkdir(parents=True, exist_ok=True)
    for k, a in zip("xyz", np.moveaxis(P, -1, 0), strict=True):
        tifffile.imwrite(out / f"{k}.tif", a.astype(np.float32))
    (out / "fuse3d.json").write_text(json.dumps(meta, indent=1))


# ------------------------------------------------------------------------------------ validation
def coverage_of(sp: dict, tree, members: list[dict]) -> dict:
    """Per member: the fraction of its points (<= 4000 sampled) that the surface `sp` covers on the same sheet."""
    cov = {}
    for m in members:
        s = m["sp"]
        if not len(s["xyz"]):
            continue
        idx = C.sample_idx(m["seg"], len(s["xyz"]), 4000)
        d = C.normal_distance(s["xyz"][idx].astype(np.float64), s["nrm"][idx].astype(np.float64), s["nok"][idx], sp, tree)
        cov[m["seg"]] = round(float((d <= C.SAME_SHEET_VOX).mean()), 4)
    return cov


def validate(P: np.ndarray, members: list[dict], voxel_um: float, sampler=None, n_prof: int = 2500, seed: int = 0) -> dict:
    """Re-measure a fused lattice with code that did not build it.

    coverage_of_members   per member: fraction of its points that the product covers on the SAME sheet
                          (same_sheet estimator: <= 4 vox along the normal, normals within 20 deg).
                          >= ~0.98 is what lets a member be retired.
    support               fraction of the product's points that some member covers (invented surface if low).
    guard                 growth_guard's per-cell fold / planarity flags on the product (fraction of cells).
    edges                 lattice edge length p10/p50/p90 and the fraction of jump edges (> 3x median).
    ct                    sampler(): material fraction (CT > ct_th) and the CT sheet-centre offset along the
                          normal: |offset| p50 / p90 and the fraction > 8 vox (a surface off the sheet, or on
                          the wrong wrap, shows up here and nowhere in the geometry above).
    """
    X, Y, Z = P[..., 0], P[..., 1], P[..., 2]
    sp = C.surface_points(X, Y, Z)
    out = {"n_points": int(len(sp["xyz"])), "n_normals": int(sp["nok"].sum())}
    if not len(sp["xyz"]):
        return {"error": "empty product"}
    from scipy.spatial import cKDTree
    tree = cKDTree(sp["xyz"], leafsize=32, balanced_tree=False, compact_nodes=False)
    out["coverage_of_members"] = coverage_of(sp, tree, members)
    # support of the product by the union of members
    idx = C.sample_idx("prod", len(sp["xyz"]), 6000)
    best = np.full(len(idx), np.inf)
    for m in members:
        s = m["sp"]
        if len(s["xyz"]):
            best = np.minimum(best, C.normal_distance(sp["xyz"][idx].astype(np.float64), sp["nrm"][idx].astype(np.float64), sp["nok"][idx], s))
    out["support"] = round(float((best <= C.SAME_SHEET_VOX).mean()), 4)
    pol = G.GuardPolicy(enabled=True, vacuum=False, overlap=False)
    r = G.evaluate(X, Y, Z, pol, voxel_um)
    Pg, V = G.lattice_frame(X, Y, Z)
    out["guard"] = {k: round(float(mk[V].mean()), 4) for k, mk in r.masks.items() if mk is not None}
    out["area_cm2"] = round(float(G.lattice_area_cm2(Pg, V, voxel_um)), 3)
    out["cells"] = int(V.sum())
    e = np.concatenate([np.linalg.norm(Pg[1:, :] - Pg[:-1, :], axis=-1)[V[1:, :] & V[:-1, :]],
                        np.linalg.norm(Pg[:, 1:] - Pg[:, :-1], axis=-1)[V[:, 1:] & V[:, :-1]]])
    med = float(np.median(e)) if e.size else float("nan")
    out["edges"] = {"p10": round(float(np.percentile(e, 10)), 1), "p50": round(med, 1), "p90": round(float(np.percentile(e, 90)), 1),
                    "jump_frac": round(float((e > 3 * med).mean()), 4)}
    if sampler is not None:
        rng = np.random.default_rng(seed)
        ok = np.nonzero(sp["nok"])[0]
        pick = rng.choice(ok, size=min(n_prof, len(ok)), replace=False)
        out["ct"] = ct_profile(sampler, sp["xyz"][pick].astype(np.float64), sp["nrm"][pick].astype(np.float64))
    return out


def ct_profile(sampler, xyz, nrm, half: float = 7.0, step: float = 1.0, sigma: float = 1.5, th: float = 5.0) -> dict:
    """CT brightest-layer offset within +-half vox of the surface. `half` MUST stay below half the sheet pitch (PHerc0211: ~15 vox,
    so 7): a wider window just picks the brightest of several neighbouring wraps and reads ~pitch/2 for any surface (source project)."""
    offs = np.arange(-half, half + 1e-6, step)
    P = xyz[:, None, :] + nrm[:, None, :] * offs[None, :, None]
    v = sampler(P.reshape(-1, 3)).reshape(len(xyz), len(offs))
    vs = ndi.gaussian_filter1d(v, sigma / 1.0, axis=1, mode="nearest")
    peak = offs[np.argmax(vs, axis=1)]
    ap = np.abs(peak)
    centre = v[:, len(offs) // 2]
    return {"n": int(len(xyz)), "material_frac": round(float((centre > th).mean()), 4),
            "offset_abs_p50": round(float(np.percentile(ap, 50)), 1), "offset_abs_p90": round(float(np.percentile(ap, 90)), 1),
            "offset_gt3_frac": round(float((ap > 3).mean()), 4), "offset_gt8_frac": round(float((ap > 8).mean()), 4)}


# ------------------------------------------------------------------------------------ graft (no global frame)
def prepare(m: dict, voxel_um: float, sampler=None, pol: G.GuardPolicy | None = None) -> dict:
    """TRIM a member before fusion with the growth guard's own criteria (fold/hairpin, planarity, and CT vacuum
    when a sampler is given): cells the guard would cut are set invalid, only the largest piece is kept."""
    pol = pol or G.GuardPolicy(enabled=True, overlap=False, vacuum=sampler is not None)
    res = G.evaluate(m["X"], m["Y"], m["Z"], pol, voxel_um, sampler=sampler)
    keep = res.keep
    X, Y, Z = (np.where(keep, a, -1.0).astype(np.float32) for a in (m["X"], m["Y"], m["Z"]))
    out = member_from_arrays(m["seg"], X, Y, Z)
    out["trim"] = {"cells_before": int(res.cells_before), "cells_after": int(res.cells_after), "pruned_by": res.pruned_by, "stop": res.stop}
    for k in ("value",):
        if k in m:
            out[k] = m[k]
    return out


def _shift(a, di, dj):
    """b[i, j] = a[i - di, j - dj] (zero/False outside)."""
    b = np.zeros_like(a)
    H, W = a.shape[:2]
    si, sj = slice(max(di, 0), H + min(di, 0)), slice(max(dj, 0), W + min(dj, 0))
    ti, tj = slice(max(-di, 0), H + min(-di, 0)), slice(max(-dj, 0), W + min(-dj, 0))
    b[si, sj] = a[ti, tj]
    return b


def graft(members: list[dict], primary: int | None = None, margin: int = 90, snap_tol: float = 8.0, rho_f: float = 1.0,
          min_support: int = 2, align_deg: float = 25.0, max_iter: int = 400, max_gap: int = 1) -> dict:
    """Keep the best member's lattice intact and GROW it over the other members' surface.

    The primary lattice P is the parameterisation (a valid grid, already trimmed). The frontier is marched outward
    one ring at a time: a new node is PREDICTED by linear extrapolation of the two valid nodes behind it, then
    SNAPPED along the local normal to the median of the member points that are within `rho` of the prediction,
    within `snap_tol` vox of it along the normal and aligned with it. No support -> the node is not created, so a
    hairpin, a sheet jump (support sits >= ~25 vox off the extrapolation) or a gap stops the growth there instead of
    being bridged. Nothing inside the primary is moved, so the primary is always fully covered.
    """
    if primary is None:
        primary = int(np.argmax([m.get("value", 0.0) or int((m["X"] > 0).sum()) for m in members]))
    Pm = members[primary]
    X, Y, Z = Pm["X"], Pm["Y"], Pm["Z"]
    V0 = (X > 0) & (Y > 0) & (Z > 0)
    if V0.sum() < 20:
        return {"P": None, "diag": {"error": "primary too small"}}
    s = float(Pm["edge"])
    H0, W0 = V0.shape
    Hc, Wc = H0 + 2 * margin, W0 + 2 * margin
    P = np.zeros((Hc, Wc, 3))
    V = np.zeros((Hc, Wc), bool)
    P[margin:margin + H0, margin:margin + W0] = np.stack([X, Y, Z], axis=-1)
    V[margin:margin + H0, margin:margin + W0] = V0
    pts, nrm, mem = [], [], []
    for k, m in enumerate(members):
        sp = m["sp"]
        ok = sp["nok"]
        pts.append(sp["xyz"][ok])
        nrm.append(sp["nrm"][ok])
        mem.append(np.full(int(ok.sum()), k))
    Q = np.concatenate(pts).astype(np.float64)
    QN = np.concatenate(nrm).astype(np.float64)
    QM = np.concatenate(mem)
    tree = cKDTree(Q)
    rho = rho_f * s
    cos_a = float(np.cos(np.radians(align_deg)))
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    added_total, iters = 0, 0
    src = np.zeros((Hc, Wc), np.int16)              # iteration at which each node appeared (0 = primary)
    nsup = np.zeros((Hc, Wc), np.int16)             # distinct members supporting a grown node
    for it in range(1, max_iter + 1):
        pred = np.zeros((Hc, Wc, 3))
        dsum = np.zeros((Hc, Wc, 3))
        npred = np.zeros((Hc, Wc))
        for di, dj in dirs:
            for k in range(1, max_gap + 1):
                # the two valid nodes k and k+1 steps behind: extrapolate k steps across a hole of k-1 missing nodes
                n1 = _shift(V, k * di, k * dj)
                n2 = _shift(V, (k + 1) * di, (k + 1) * dj)
                a = _shift(P, k * di, k * dj)
                b = _shift(P, (k + 1) * di, (k + 1) * dj)
                ok = ~V & n1 & n2
                pred[ok] += (a + k * (a - b))[ok]
                dsum[ok] += (a - b)[ok]
                npred[ok] += 1
        cand = npred > 0
        if not cand.any():
            break
        pred[cand] /= npred[cand][:, None]
        Nrm, nok = G.grid_normals(P, V)
        has_n = np.zeros((Hc, Wc), bool)
        for di, dj in dirs:
            for k in range(1, max_gap + 1):
                has_n |= _shift(nok, k * di, k * dj)
        ci, cj = np.nonzero(cand & has_n)
        if not len(ci):
            break
        # orientation-consistent mean normal (flip each neighbour normal to agree with the first)
        nrm_c = []
        for i, j in zip(ci, cj, strict=True):
            vs = [Nrm[i - k * di, j - k * dj] for di, dj in dirs for k in range(1, max_gap + 1)
                  if 0 <= i - k * di < Hc and 0 <= j - k * dj < Wc and nok[i - k * di, j - k * dj]]
            ref = vs[0]
            nrm_c.append(sum(v if v @ ref >= 0 else -v for v in vs))
        nrm_c = np.array(nrm_c)
        nrm_c /= np.maximum(np.linalg.norm(nrm_c, axis=1, keepdims=True), 1e-9)
        pc = pred[ci, cj]
        lists = tree.query_ball_point(pc, r=rho)
        new = 0
        upd = []
        for n_, idxs in enumerate(lists):
            if len(idxs) < min_support:
                continue
            idxs = np.asarray(idxs)
            off = (Q[idxs] - pc[n_]) @ nrm_c[n_]
            al = np.abs(QN[idxs] @ nrm_c[n_]) >= cos_a
            sel = (np.abs(off) <= snap_tol) & al
            if sel.sum() < min_support:
                continue
            # the prediction must lie INSIDE the support, not beyond its edge: some supporting point at or past it
            # along the growth direction (within half a spacing behind). Without this the ball overlap lets the
            # frontier creep off the end of the data one ring per iteration, indefinitely.
            ev = dsum[ci[n_], cj[n_]]
            ev = ev / max(float(np.linalg.norm(ev)), 1e-9)
            if not ((Q[idxs][sel] - pc[n_]) @ ev >= -0.5 * s).any():
                continue
            upd.append((ci[n_], cj[n_], pc[n_] + nrm_c[n_] * float(np.median(off[sel])), len(set(QM[idxs[sel]].tolist()))))
        for i, j, p, ns in upd:
            # REGULARITY GATE (sequential, so two new neighbours are checked against each other too): every valid
            # lattice neighbour must sit at its expected distance (s for edge, 1.41 s for diagonal). A node that
            # fails is not created; this is what stops two fronts that meet out of register from tearing the grid.
            bad = False
            for di, dj in dirs:                       # a bridged node: its nearest valid anchor k steps away, at ~k * s
                for k in range(2, max_gap + 1):
                    if V[i - k * di, j - k * dj] and not V[i - (k - 1) * di, j - (k - 1) * dj]:
                        d = float(np.linalg.norm(p - P[i - k * di, j - k * dj]))
                        if not (0.6 * k * s <= d <= 1.6 * k * s):
                            bad = True
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if (di or dj) and V[i + di, j + dj]:
                        d = float(np.linalg.norm(p - P[i + di, j + dj]))
                        e = s * (1.4142 if di and dj else 1.0)
                        if not (0.6 * e <= d <= 1.6 * e):
                            bad = True
            if bad:
                continue
            P[i, j] = p
            V[i, j] = True
            src[i, j] = it
            nsup[i, j] = ns
            new += 1
        added_total += new
        iters = it
        if new == 0:
            break
    rr, cc = np.nonzero(V)
    P = np.where(V[..., None], P, -1.0).astype(np.float32)[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
    src = src[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
    nsup = nsup[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
    Vf = P[..., 0] > 0
    lab, nl = ndi.label(Vf, structure=np.ones((3, 3), int))
    diag = {"method": "graft", "primary": Pm["seg"], "primary_cells": int(V0.sum()), "grown_cells": int(added_total),
            "grid": list(P.shape[:2]), "iterations": iters, "spacing_vox": round(s, 2), "snap_tol_vox": snap_tol,
            "min_support": min_support, "components": int(nl),
            "grown_supported_by_ge2_members_frac": round(float((nsup[src > 0] >= 2).mean()), 4) if (src > 0).any() else None}
    return {"P": P, "diag": diag, "src_iter": src}


# ------------------------------------------------------------------------------------ ridge verification
def verify(m: dict, pred_sampler, span: float = 8.0, win: float = 3.0, step: float = 1.0, thr: float = 127.0) -> dict:
    """Anchor a member to the SURFACE PREDICTION. On the source project's densest scroll the sheets are only ~15 vox apart
    (median spacing of prediction ridges along a normal) and only ~64 % of a grown segment's points sit within 4 vox
    of one (both recorded there, not re-measured here), so a geometric same-sheet test alone cannot tell "same sheet, wobbling" from "the next wrap". A point is
    VERIFIED when the prediction has a ridge within +-`win` vox along its normal; it is then moved to that ridge's
    centre. Everything else is invalid in the returned member (a sheet jump / mush cell is dropped, and counted).
    The lattice keeps its own grid, so `graft` can use the verified member as a primary."""
    X, Y, Z = m["X"], m["Y"], m["Z"]
    V = (X > 0) & (Y > 0) & (Z > 0)
    ii, jj = np.nonzero(V)
    sp = m["sp"]
    if len(ii) != len(sp["xyz"]):
        raise ValueError("lattice/point mismatch")
    offs = np.arange(-span, span + 1e-6, step)
    xyz = sp["xyz"].astype(np.float64)
    n = sp["nrm"].astype(np.float64)
    ok = sp["nok"]
    hitc = np.zeros(len(xyz), bool)
    shift = np.zeros(len(xyz))
    sel = np.nonzero(ok)[0]
    B = 20000
    for a in range(0, len(sel), B):
        s_ = sel[a:a + B]
        Pq = xyz[s_, None, :] + n[s_, None, :] * offs[None, :, None]
        v = pred_sampler(Pq.reshape(-1, 3)).reshape(len(s_), len(offs)) > thr
        near = np.abs(offs)[None, :] <= win
        h = v & near
        any_h = h.any(axis=1)
        cen = np.where(any_h, (h * offs[None, :]).sum(1) / np.maximum(h.sum(1), 1), 0.0)
        hitc[s_] = any_h
        shift[s_] = cen
    Xn, Yn, Zn = np.full_like(X, -1.0), np.full_like(Y, -1.0), np.full_like(Z, -1.0)
    good = hitc
    p2 = xyz + n * shift[:, None]
    Xn[ii[good], jj[good]] = p2[good, 0]
    Yn[ii[good], jj[good]] = p2[good, 1]
    Zn[ii[good], jj[good]] = p2[good, 2]
    out = member_from_arrays(m["seg"], Xn, Yn, Zn)
    out["verify"] = {"points": int(len(xyz)), "with_normal": int(ok.sum()), "verified": int(good.sum()),
                     "verified_frac": round(float(good.sum() / max(1, len(xyz))), 4),
                     "shift_abs_p50_vox": round(float(np.median(np.abs(shift[good]))), 2) if good.any() else None}
    if "value" in m:
        out["value"] = m["value"] * out["verify"]["verified_frac"]
    return out


# ------------------------------------------------------------------------------------ set cover (no fusion)
def setcover(members: list[dict], target: float = 0.95, n: int = 1500) -> dict:
    """Which members can be RETIRED without losing surface? Pool a sample of every member's points; a pooled point counts
    1/(number of members that hold it same-sheet), so a location held by five members is one location, not five.
    Members are then chosen greedily (most uncovered unique weight per member) until `target` of the unique weight is
    held. The rest are covered by the chosen ones and can be retired. Pure 3D, no lattice, no frame: nothing can jump."""
    from scipy.spatial import cKDTree
    segs = [m["seg"] for m in members]
    trees = [cKDTree(m["sp"]["xyz"]) if len(m["sp"]["xyz"]) else None for m in members]
    P, Nn, Ok, own, wt = [], [], [], [], []
    for k, m in enumerate(members):
        s = m["sp"]
        if not len(s["xyz"]):
            continue
        idx = C.sample_idx(m["seg"], len(s["xyz"]), n)
        P.append(s["xyz"][idx].astype(np.float64))
        Nn.append(s["nrm"][idx].astype(np.float64))
        Ok.append(s["nok"][idx])
        own.append(np.full(len(idx), k))
        wt.append(np.full(len(idx), len(s["xyz"]) / len(idx)))          # each sample stands for this many points
    P, Nn, Ok, own, wt = (np.concatenate(a) for a in (P, Nn, Ok, own, wt))
    cov = np.zeros((len(P), len(members)), bool)
    for j, m in enumerate(members):
        if trees[j] is None:
            continue
        d = C.normal_distance(P, Nn, Ok, m["sp"], trees[j])
        cov[:, j] = d <= C.SAME_SHEET_VOX
        cov[own == j, j] = True
    w = wt / np.maximum(cov.sum(axis=1), 1)                      # unique-location weight
    total = float(w.sum())
    held = np.zeros(len(P), bool)
    chosen: list[int] = []
    while held @ w < target * total and len(chosen) < len(members):
        gain = np.array([(w * (cov[:, j] & ~held)).sum() if j not in chosen else -1.0 for j in range(len(members))])
        j = int(np.argmax(gain))
        if gain[j] <= 0:
            break
        chosen.append(j)
        held |= cov[:, j]
    retired = [segs[j] for j in range(len(members)) if j not in chosen]
    return {"chosen": [segs[j] for j in chosen], "retirable": retired, "held_unique_frac": round(float(held @ w / total), 4),
            "sum_over_unique": round(float((wt.sum()) / total), 2)}


def sheet_groups(members: list[dict], thr: float = 0.35, n: int = 1500) -> list[list[int]]:
    """Members that hold the SAME ridge: connected components of "one covers >= thr of the other" (same-sheet, 4 vox).
    Run on RIDGE-VERIFIED members this separates the parallel wraps that share one bounding box (PHerc0211: ~15 vox apart)
    into their own groups, which is what a fusion has to be given -- a cluster that mixes wraps is not one sheet."""
    from scipy.spatial import cKDTree
    k = len(members)
    trees = [cKDTree(m["sp"]["xyz"]) if len(m["sp"]["xyz"]) else None for m in members]
    M = np.zeros((k, k))
    for i in range(k):
        si = members[i]["sp"]
        if not len(si["xyz"]):
            continue
        idx = C.sample_idx(members[i]["seg"], len(si["xyz"]), n)
        for j in range(k):
            if i != j and trees[j] is not None:
                d = C.normal_distance(si["xyz"][idx].astype(np.float64), si["nrm"][idx].astype(np.float64), si["nok"][idx], members[j]["sp"], trees[j])
                M[i, j] = float((d <= C.SAME_SHEET_VOX).mean())
    A = np.maximum(M, M.T) >= thr
    seen, groups = set(), []
    for i in range(k):
        if i in seen:
            continue
        comp, stack = [], [i]
        while stack:
            a = stack.pop()
            if a in seen:
                continue
            seen.add(a)
            comp.append(a)
            stack += [b for b in range(k) if A[a, b] and b not in seen]
        groups.append(sorted(comp))
    return groups


# ------------------------------------------------------------------------------------ ridge-crossing (sheet-jump) edges
def edge_ridge_runs(X, Y, Z, pred_sampler, step: float = 1.5, thr: float = 127.0, max_edge_mult: float = 3.0) -> dict:
    """For every lattice edge (both ends valid, length <= max_edge_mult x median: longer ones are holes), count the separate
    prediction ridge bodies the straight edge passes through. A lattice that stays on one sheet runs along a ridge (1 body) or
    through the gap beside it (0); an edge that steps from one wrap onto the next crosses >= 2. Returns {"dir0": runs (H-1,W),
    "dir1": runs (H,W-1), "ok0", "ok1"} with runs = -1 where the edge is not evaluated."""
    P, V = G.lattice_frame(X, Y, Z)
    med = G.median_edge(P, V)
    out = {}
    for name, (a, b, ok) in {"dir0": (P[:-1], P[1:], V[:-1] & V[1:]), "dir1": (P[:, :-1], P[:, 1:], V[:, :-1] & V[:, 1:])}.items():
        ln = np.linalg.norm(b - a, axis=-1)
        ok = ok & (ln <= max_edge_mult * med)
        idx = np.argwhere(ok)
        runs = np.full(ok.shape, -1, np.int16)
        if len(idx):
            k = int(np.ceil(np.nanmax(ln[ok]) / step)) + 1
            t = np.linspace(0.0, 1.0, k)
            A, B = a[ok], b[ok]
            pts = A[:, None, :] + (B - A)[:, None, :] * t[None, :, None]
            v = pred_sampler(pts.reshape(-1, 3)).reshape(len(A), k) > thr
            # only the samples that lie on this edge (edges are shorter than the longest): mask by length
            tt = t[None, :] * ln[ok][:, None]
            live = tt <= ln[ok][:, None] + 1e-6
            v = v & live
            rise = (v[:, 1:] & ~v[:, :-1]).sum(axis=1) + v[:, 0]
            runs[ok] = rise.astype(np.int16)
        out[name] = runs
    return out
