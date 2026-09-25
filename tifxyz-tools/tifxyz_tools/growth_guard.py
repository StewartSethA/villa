"""Per-cell growth guard: prune the FRONTIER of a grown tifxyz sheet where it has run into
vacuum, a hairpin, a crease / crumple swamp, or a sheet we already hold.

Whole-segment verdicts that land after a round ("material fraction < 0.75 -> vacuum") cannot say
WHERE the sheet went wrong and cannot remove anything. This works per CELL, on the lattice, with
numpy/scipy only (no change to the C++ tracer):

  1. per-cell bad masks on the lattice grid
       vacuum   CT at the cell's 3-D position <= ct_th, smoothed over a (2*vacuum_win+1)^2 window:
                >= vacuum_frac of the window is air.  (needs a CT sampler; see ZarrSampler)
       fold     turn radius R = arc / theta < fold_radius_um over fold_arc_um of lattice line, read
                along rows AND columns: a hairpin.
       plan     the cell's normal deviates > plan_deg from the mean normal of its window: a crease /
                crumple that a segment-level planarity number cannot see.
       overlap  the cell lies within overlap_vox voxels ALONG a neighbouring segment's normal, normals
                within overlap_deg (same_sheet.normal_distance): that sheet is already held.
  2. only bad regions that TOUCH THE FRONTIER (within reach_rings of an invalid cell) are cut, plus
     margin_rings. A bad region wholly enclosed by good surface is left alone: a frontier WRAPPING
     AROUND a vacuum leaves a hole, which is fine; what must not happen is growing THROUGH it.
     overlap cells keep an overlap_keep_rings seam next to the good interior so a stitcher still has
     overlap to register on.
  3. keep the largest connected piece (or the piece the tracer started in), and report why each cell
     went.
  4. regrowth guard: a tracer resumed from the cropped lattice will happily grow into the same bad
     ground again. `regrown_fraction` measures how much of a round's NEW surface lies on previously
     pruned ground, so a caller can end the segment instead of looping.

PROTOTYPE. Every threshold is a `GuardPolicy` field. Nothing here writes into a source tifxyz:
`guard_tifxyz` writes a new directory. VALIDATION.md says what has and has not been measured.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

_S3 = np.ones((3, 3), bool)

# reasons in ATTRIBUTION order: a pruned cell is charged to the first that flags it
REASONS = ("overlap", "vacuum", "fold", "plan")


@dataclass(frozen=True)
class GuardPolicy:
    enabled: bool = False
    # -- vacuum
    vacuum: bool = True
    ct_th: float = 5.0            # material criterion: CT > 5 (insensitive 5..32 in the source project)
    ct_level: int = 1             # CT pyramid level the sampler reads (1 = 2x coarser than level 0)
    vacuum_win: int = 2           # half-window (cells)
    vacuum_frac: float = 0.75     # window fraction that must be air (source-project sweep, VALIDATION.md)
    # -- hairpin: turn radius R = arc / theta over fold_arc_um of lattice line
    fold: bool = True
    fold_arc_um: float = 1000.0
    fold_radius_um: float = 300.0     # a 500 um radius over-pruned in the source project's sweep; see VALIDATION.md
    fold_half: int = 1                # cells flagged either side of a tight turn's centre (large = the whole window, the over-pruning first draft)
    # -- local planarity / crumple
    plan: bool = True
    plan_win: int = 3             # half-window (cells)
    plan_deg: float = 70.0
    plan_frac: float = 0.4        # a cell is a swamp cell when >= this fraction of its window is crumpled (speckle is noise)
    # -- same-sheet overlap (same_sheet.SAME_SHEET_VOX / ALIGN_DEG)
    overlap: bool = True
    overlap_vox: float = 4.0
    overlap_deg: float = 20.0
    overlap_keep_rings: int = 2
    # -- frontier prune
    min_bad_cells: int = 12       # a bad speck smaller than this is noise, not a swamp
    reach_rings: int = 2          # "touches the frontier" = within this many cells of an invalid cell
    margin_rings: int = 0         # source-project sweep: extra margin cost area for little gain (VALIDATION.md)
    min_keep_cells: int = 64
    keep_islands: bool = False
    # -- regrowth
    regrow_block_frac: float = 0.3
    regrow_tol_cells: float = 0.75   # x median edge: "on pruned ground"


# ----------------------------------------------------------------------------- lattice helpers
def lattice_frame(X, Y, Z):
    """P (H, W, 3) float64 and V (H, W) valid. Invalid cells are <= 0 (tifxyz writes -1)."""
    V = (X > 0) & (Y > 0) & (Z > 0)
    P = np.stack([X, Y, Z], axis=-1).astype(np.float64)
    return P, V


def median_edge(P, V) -> float:
    e = []
    for ax in (0, 1):
        a, b = (P[1:], P[:-1]) if ax == 0 else (P[:, 1:], P[:, :-1])
        ok = (V[1:] & V[:-1]) if ax == 0 else (V[:, 1:] & V[:, :-1])
        if ok.any():
            e.append(np.linalg.norm(a - b, axis=-1)[ok])
    return float(np.median(np.concatenate(e))) if e else float("nan")


def grid_normals(P, V):
    """Unit normals (cross of the two lattice tangents, central differences where both
    neighbours are valid, one-sided otherwise) and the mask where they are defined. Orientation
    follows the lattice, so a hairpin FLIPS the normal -- which is what `plan` reads."""
    def d(axis):
        nxt = np.roll(P, -1, axis)
        prv = np.roll(P, 1, axis)
        vn = np.roll(V, -1, axis)
        vp = np.roll(V, 1, axis)
        if axis == 0:
            vn[-1] = False
            vp[0] = False
        else:
            vn[:, -1] = False
            vp[:, 0] = False
        D = np.zeros_like(P)
        both = V & vn & vp
        D[both] = (nxt - prv)[both]
        f = V & vn & ~vp
        D[f] = (nxt - P)[f]
        b = V & vp & ~vn
        D[b] = (P - prv)[b]
        return D, V & (vn | vp)
    di, oi = d(0)
    dj, oj = d(1)
    n = np.cross(di, dj)
    nn = np.linalg.norm(n, axis=-1)
    ok = V & oi & oj & (nn > 1e-9)
    n = n / np.maximum(nn, 1e-12)[..., None]
    return n, ok


def lattice_area_cm2(P, V, voxel_um: float) -> float:
    """Sum of the areas of the lattice quads whose four corners are valid."""
    q = V[:-1, :-1] & V[1:, :-1] & V[:-1, 1:] & V[1:, 1:]
    a = 0.5 * (np.linalg.norm(np.cross(P[1:, :-1] - P[:-1, :-1], P[:-1, 1:] - P[:-1, :-1]), axis=-1)
               + np.linalg.norm(np.cross(P[1:, 1:] - P[1:, :-1], P[1:, 1:] - P[:-1, 1:]), axis=-1))
    return float(a[q].sum() * (voxel_um * 1e-4) ** 2)


# ----------------------------------------------------------------------------- per-cell criteria
def fold_mask(P, V, voxel_um: float, pol: GuardPolicy):
    """Cells inside a turn tighter than pol.fold_radius_um, read along rows and columns
    (a hairpin criterion, per cell). A window needs every cell valid."""
    H, W = V.shape
    med = median_edge(P, V)
    out = np.zeros((H, W), bool)
    if not np.isfinite(med) or med <= 0:
        return out
    k = int(max(2, round((pol.fold_arc_um / voxel_um) / med)))
    for ax in (0, 1):
        Q = P if ax == 0 else P.transpose(1, 0, 2)
        U = V if ax == 0 else V.T
        n = Q.shape[0]
        if n <= k + 1:
            continue
        T = Q[1:] - Q[:-1]                                   # (n-1, m, 3)
        L = np.linalg.norm(T, axis=-1)
        good = U[1:] & U[:-1] & (L > 1e-9)
        Tn = T / np.maximum(L, 1e-12)[..., None]
        cs = np.concatenate([np.zeros((1,) + L.shape[1:]), np.cumsum(np.where(good, L, 0.0), axis=0)], axis=0)
        gc = np.concatenate([np.zeros((1,) + good.shape[1:], int), np.cumsum(good, axis=0)], axis=0)
        i = np.arange(0, n - 1 - k)
        arc = cs[i + k] - cs[i]                              # arc over k steps starting at i
        full = (gc[i + k] - gc[i]) == k
        ok0 = good[i]
        ok1 = good[i + k - 1] if k >= 1 else good[i]
        cosang = np.clip(np.einsum("imc,imc->im", Tn[i], Tn[i + k - 1]), -1.0, 1.0)
        theta = np.arccos(cosang)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_um = np.where(theta > 1e-6, arc * voxel_um / theta, np.inf)
        hit = full & ok0 & ok1 & (r_um < pol.fold_radius_um)          # (n-1-k, m)
        # flag the cells around the turn's centre (the whole window when fold_half is large)
        m = np.zeros((n, hit.shape[1]), bool)
        c0 = k // 2
        for off in range(max(0, c0 - pol.fold_half), min(k, c0 + pol.fold_half) + 1):
            m[off:off + hit.shape[0]] |= hit
        out |= m if ax == 0 else m.T
    return out & V


def plan_mask(P, V, pol: GuardPolicy):
    """Cells whose normal deviates > pol.plan_deg from the window-mean normal."""
    n, ok = grid_normals(P, V)
    w = (2 * pol.plan_win + 1)
    wt = ok.astype(np.float64)
    mean = np.stack([ndi.uniform_filter(n[..., c] * wt, size=w, mode="constant") for c in range(3)], axis=-1)
    norm = np.linalg.norm(mean, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = np.einsum("hwc,hwc->hw", n, mean) / np.maximum(norm, 1e-9)
    # a window whose normals cancel (norm ~ 0) is itself a crumple
    raw = ok & ((cosang < np.cos(np.radians(pol.plan_deg))) | (norm < 0.2 * ndi.uniform_filter(wt, size=w, mode="constant")))
    frac = ndi.uniform_filter(raw.astype(np.float64), size=w, mode="constant")
    den = ndi.uniform_filter(ok.astype(np.float64), size=w, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        return ok & (np.where(den > 0, frac / den, 0.0) >= pol.plan_frac)


def vacuum_mask(P, V, sampler, pol: GuardPolicy):
    """Cells lying in air: CT(x, y, z) <= ct_th, smoothed to a (2*win+1)^2 window fraction."""
    out = np.zeros(V.shape, bool)
    if sampler is None or not V.any():
        return out
    ct = np.asarray(sampler(P[V]), dtype=np.float64)
    air = np.zeros(V.shape, np.float64)
    air[V] = (ct <= pol.ct_th).astype(np.float64)
    w = 2 * pol.vacuum_win + 1
    num = ndi.uniform_filter(air, size=w, mode="constant")
    den = ndi.uniform_filter(V.astype(np.float64), size=w, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(den > 0, num / den, 0.0)
    return V & (frac >= pol.vacuum_frac)


def overlap_mask(P, V, cover, pol: GuardPolicy, own: str | None = None):
    """Cells already held by another segment. `cover(points, normals, ok, own)` -> bool per
    point; see SegmentIndex."""
    out = np.zeros(V.shape, bool)
    if cover is None or not V.any():
        return out
    n, ok = grid_normals(P, V)
    idx = np.nonzero(V)
    hit = cover(P[idx], n[idx], ok[idx], own)
    out[idx] = hit
    return out


class SegmentIndex:
    """Same-sheet coverage by a set of other segments: a lazy KD-tree per segment and a bbox
    prefilter, using same_sheet.normal_distance. `add(seg, X, Y, Z)` thins nothing; pass a decimated lattice if it is huge."""

    def __init__(self, vox: float = 4.0, deg: float = 20.0):
        from . import same_sheet as C
        self.C, self.vox, self.deg = C, vox, deg
        self.T: dict[str, dict] = {}
        self.tree: dict[str, object] = {}

    def add(self, seg: str, X, Y, Z):
        T = self.C.surface_points(X, Y, Z)
        if len(T["xyz"]) and T["bbox"] is not None:
            self.T[seg] = T

    def attribute(self, pts, nrm, ok, own=None):
        """(hit mask, name per point or '') -- the covering segment with the SMALLEST along-normal
        distance wins."""
        from scipy.spatial import cKDTree
        C = self.C
        hit = np.zeros(len(pts), bool)
        name = np.full(len(pts), "", dtype=object)
        if not len(pts) or not self.T:
            return hit, name
        p = pts.astype(np.float64)
        best = np.full(len(pts), np.inf)
        lo, hi = p.min(axis=0), p.max(axis=0)
        for s, T in self.T.items():
            if s == own:
                continue
            b0 = np.asarray(T["bbox"][0]) - C.R_QUERY
            b1 = np.asarray(T["bbox"][1]) + C.R_QUERY
            if np.any(b0 > hi) or np.any(b1 < lo):
                continue
            if s not in self.tree:
                self.tree[s] = cKDTree(T["xyz"], leafsize=32, balanced_tree=False, compact_nodes=False)
            d = C.normal_distance(p, nrm.astype(np.float64), ok, T, tree=self.tree[s], align_deg=self.deg)
            better = (d <= self.vox) & (d < best)
            best[better] = d[better]
            name[better] = s
            hit |= d <= self.vox
        return hit, name

    def __call__(self, pts, nrm, ok, own=None):
        return self.attribute(pts, nrm, ok, own)[0]


# ----------------------------------------------------------------------------- pruning
@dataclass
class GuardResult:
    keep: np.ndarray
    masks: dict
    pruned_by: dict
    cells_before: int
    cells_after: int
    components_cut: int
    interior_bad_kept: int
    frontier_bad_frac: float
    stop: str | None = None

    def summary(self) -> dict:
        return {"cells_before": self.cells_before, "cells_after": self.cells_after,
                "pruned_by": self.pruned_by, "components_cut": self.components_cut,
                "interior_bad_components_kept": self.interior_bad_kept,
                "frontier_bad_frac": round(self.frontier_bad_frac, 4), "stop": self.stop}


def frontier_band(V, rings: int):
    """Valid cells within `rings` of an invalid cell (the grid edge counts as invalid)."""
    if rings <= 0:
        return V & ~ndi.binary_erosion(V, _S3, border_value=0)
    return V & ~ndi.binary_erosion(V, _S3, iterations=rings, border_value=0)


def prune(V, masks: dict, pol: GuardPolicy, anchor=None) -> GuardResult:
    """Cut the bad regions that touch the frontier; keep enclosed ones; keep the largest piece."""
    V = V.astype(bool)
    band = frontier_band(V, pol.reach_rings)
    rm_by: dict[str, np.ndarray] = {}
    cut = 0
    interior = 0
    any_bad = np.zeros_like(V)
    for r in REASONS:
        m = masks.get(r)
        if m is None:
            continue
        m = m & V
        lab, n = ndi.label(m, structure=_S3)
        if n == 0:
            rm_by[r] = np.zeros_like(V)
            continue
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        touch = np.zeros(n + 1, bool)
        touch[np.unique(lab[band & (lab > 0)])] = True
        big = sizes >= pol.min_bad_cells
        # A bad region that does not touch the frontier is still CUT when it is a BARRIER: a
        # crease/hairpin that runs across the sheet, so that removing it leaves a second piece
        # (the tail the tracer followed round the fold). An enclosed hole splits nothing and stays.
        barrier = np.zeros(n + 1, bool)
        for lb in np.nonzero(big & ~touch)[0]:
            if lb == 0:
                continue
            cm = ndi.binary_dilation(lab == lb, _S3, iterations=max(0, pol.margin_rings)) if pol.margin_rings > 0 else (lab == lb)
            l2, n2 = ndi.label(V & ~cm, structure=_S3)
            if n2 >= 2 and (np.bincount(l2.ravel())[1:] >= pol.min_keep_cells).sum() >= 2:
                barrier[lb] = True
        sel = (touch | barrier) & big
        sel[0] = False
        interior += int(((~touch) & ~barrier & big)[1:].sum())
        cut += int(sel.sum())
        rm = sel[lab]
        any_bad |= m & big[lab]
        if r == "overlap" and pol.overlap_keep_rings > 0 and rm.any():
            good = V & ~m
            seam = ndi.binary_dilation(good, _S3, iterations=pol.overlap_keep_rings) & rm
            rm = rm & ~seam
        rm_by[r] = rm
    rm_all = np.zeros_like(V)
    for r in REASONS:
        if r in rm_by:
            rm_all |= rm_by[r]
    if pol.margin_rings > 0 and rm_all.any():
        rm_all = ndi.binary_dilation(rm_all, _S3, iterations=pol.margin_rings) & V
    keep = V & ~rm_all
    if not pol.keep_islands and keep.any():
        lab, n = ndi.label(keep, structure=_S3)
        if n > 1:
            main = 1 + int(np.argmax(np.bincount(lab.ravel())[1:]))
            if anchor is not None and lab[anchor] > 0:      # the piece the tracer STARTED in, when it survived
                main = int(lab[anchor])
            keep = lab == main
    stop = None
    # "nothing_left" means the GUARD emptied the sheet. A young lattice that is simply small (9 or 28 cells after
    # round 1) has had nothing cut, and ending it here killed two real seeds on the first canary (2026-09-25).
    if (V & ~keep).any() and keep.sum() < pol.min_keep_cells:
        stop = "nothing_left"
    pruned_by, taken = {}, np.zeros_like(V)
    dropped = V & ~keep
    for r in REASONS:
        m = masks.get(r)
        if m is None:
            continue
        got = dropped & m & ~taken
        pruned_by[r] = int(got.sum())
        taken |= got
    pruned_by["margin_or_island"] = int((dropped & ~taken).sum())
    fb = float((any_bad & band).sum() / max(1, band.sum()))
    return GuardResult(keep, masks, pruned_by, int(V.sum()), int(keep.sum()), cut, interior, fb, stop)


def evaluate(X, Y, Z, pol: GuardPolicy, voxel_um: float, sampler=None, cover=None, own=None, anchor=None) -> GuardResult:
    """All enabled criteria on one lattice, then `prune`."""
    P, V = lattice_frame(X, Y, Z)
    masks = {}
    if pol.overlap and cover is not None:
        masks["overlap"] = overlap_mask(P, V, cover, pol, own)
    if pol.vacuum and sampler is not None:
        masks["vacuum"] = vacuum_mask(P, V, sampler, pol)
    if pol.fold:
        masks["fold"] = fold_mask(P, V, voxel_um, pol)
    if pol.plan:
        masks["plan"] = plan_mask(P, V, pol)
    return prune(V, masks, pol, anchor=anchor)


# ----------------------------------------------------------------------------- regrowth guard
def regrown_fraction(prev_keep_xyz: np.ndarray, pruned_xyz: np.ndarray, new_xyz: np.ndarray, tol: float) -> float:
    """Of the cells a round ADDED (further than `tol` from the previous kept surface), the
    fraction lying within `tol` of ground that was pruned. 0 when the round added nothing."""
    from scipy.spatial import cKDTree
    if not len(new_xyz) or not len(pruned_xyz):
        return 0.0
    if len(prev_keep_xyz):
        d_old, _ = cKDTree(prev_keep_xyz).query(new_xyz, k=1)
        added = new_xyz[d_old > tol]
    else:
        added = new_xyz
    if not len(added):
        return 0.0
    d_pr, _ = cKDTree(pruned_xyz).query(added, k=1)
    return float((d_pr <= tol).mean())


# ----------------------------------------------------------------------------- tifxyz I/O
def _read_xyz(d):
    import tifffile
    return tuple(tifffile.imread(Path(d) / f"{a}.tif").astype(np.float32) for a in "xyz")


def _read_gen(d):
    import tifffile
    p = Path(d) / "generations.tif"
    return tifffile.imread(p) if p.exists() else None


def guard_tifxyz(src: str, dst: str, pol: GuardPolicy, voxel_um: float, sampler=None, cover=None,
                 own: str | None = None) -> GuardResult:
    """Evaluate `src` and write the cropped surface to `dst` (a new tifxyz dir: x/y/z -- and
    generations.tif when the tracer wrote one -- with -1 / 0 where pruned, meta.json with updated
    bbox/area, guard.json with the numbers). `src` is never modified. The kept piece is the one holding the tracer's START (the smallest
    generation) when that survives, else the largest."""
    X, Y, Z = _read_xyz(src)
    gen = _read_gen(src)
    anchor = None
    if gen is not None and gen.shape == X.shape:
        V = (X > 0) & (Y > 0) & (Z > 0)
        if V.any():
            anchor = np.unravel_index(int(np.argmin(np.where(V, gen, np.iinfo(gen.dtype).max))), V.shape)
    res = evaluate(X, Y, Z, pol, voxel_um, sampler=sampler, cover=cover, own=own, anchor=anchor)
    write_cropped(src, dst, res.keep, voxel_um, res.summary())
    return res


def write_cropped(src: str, dst: str, keep: np.ndarray, voxel_um: float, summary: dict | None = None) -> None:
    """Write `src`'s lattice restricted to `keep` as a new tifxyz dir (x/y/z = -1 and
    generations = 0 where dropped; meta.json with bbox / area / max_gen refreshed). Never
    touches `src`."""
    import tifffile
    X, Y, Z = _read_xyz(src)
    gen = _read_gen(src)
    dstp = Path(dst)
    dstp.mkdir(parents=True, exist_ok=True)
    for a, A in zip("xyz", (X, Y, Z), strict=True):
        B = A.copy()
        B[~keep] = -1.0
        tifffile.imwrite(dstp / f"{a}.tif", B)
    g2 = None
    if gen is not None and gen.shape == keep.shape:
        g2 = gen.copy()
        g2[~keep] = 0
        tifffile.imwrite(dstp / "generations.tif", g2)
    meta = {}
    mp = Path(src) / "meta.json"
    if mp.exists():
        meta = json.loads(mp.read_text())
    P, V = lattice_frame(np.where(keep, X, -1), np.where(keep, Y, -1), np.where(keep, Z, -1))
    if V.any():
        meta["bbox"] = [P[V].min(axis=0).tolist(), P[V].max(axis=0).tolist()]
    meta["area_cm2"] = lattice_area_cm2(P, V, voxel_um)
    if g2 is not None and V.any():
        meta["max_gen"] = int(g2[V].max())
    if summary is not None:
        meta["guard"] = summary
        (dstp / "guard.json").write_text(json.dumps(summary, indent=1))
    (dstp / "meta.json").write_text(json.dumps(meta, indent=1))


# ----------------------------------------------------------------------------- the grow-loop step
@dataclass
class GuardState:
    """What the loop carries between rounds: where we cut, and what the surface was after the cut."""
    pruned_xyz: np.ndarray | None = None
    keep_xyz: np.ndarray | None = None
    rounds_cut: int = 0


def guard_round(cur: str, pol: GuardPolicy, voxel_um: float, state: GuardState, sampler=None, cover=None,
                own: str | None = None, tag: str = "g") -> tuple[str, dict]:
    """One guard step on the checkpoint a round just produced.

    Returns (checkpoint to resume from, info). `info["stop"]` is None, "frontier_blocked" (the round
    mostly re-grew into ground we had already pruned) or "nothing_left". When nothing is cut the
    checkpoint is returned unchanged and nothing is written."""
    X, Y, Z = _read_xyz(cur)
    P, V = lattice_frame(X, Y, Z)
    info = {"regrown_frac": 0.0, "stop": None}
    if state.pruned_xyz is not None and len(state.pruned_xyz) and V.any():
        tol = pol.regrow_tol_cells * max(1.0, median_edge(P, V))
        info["regrown_frac"] = round(regrown_fraction(state.keep_xyz if state.keep_xyz is not None else np.zeros((0, 3)),
                                                      state.pruned_xyz, P[V], tol), 4)
    dst = str(Path(cur).parent / f"guarded_{tag}_{Path(cur).name}")
    res = guard_tifxyz(cur, dst, pol, voxel_um, sampler=sampler, cover=cover, own=own)
    info.update(res.summary())
    cut = V & ~res.keep
    if info["regrown_frac"] >= pol.regrow_block_frac and state.rounds_cut > 0 and info["stop"] is None:
        info["stop"] = "frontier_blocked"
    if res.stop:
        info["stop"] = info["stop"] or res.stop
    if not cut.any():
        shutil.rmtree(dst, ignore_errors=True)
        state.keep_xyz = P[V]
        return cur, info
    state.rounds_cut += 1
    state.pruned_xyz = P[cut] if state.pruned_xyz is None else np.concatenate([state.pruned_xyz, P[cut]])
    state.keep_xyz = P[res.keep]
    return dst, info

# ----------------------------------------------------------------------------- CT sampler
class ZarrSampler:
    """CT(x, y, z) at pyramid level `level` from an OME-Zarr volume (array order z, y, x;
    lattice coordinates are LEVEL-0 voxels). Nearest voxel, chunk-grouped gather, so 20-50 k
    scattered points read each touched chunk once. Points outside the array read 0 (= air)."""

    def __init__(self, zarr_path: str, level: int = 1):
        import zarr
        g = zarr.open(zarr_path, mode="r")
        self.a = g[str(level)] if str(level) in g else g
        self.f = 2 ** level

    def __call__(self, xyz):
        a = self.a
        idx = np.rint(np.asarray(xyz) / self.f).astype(np.int64)[:, ::-1]        # (z, y, x)
        out = np.zeros(len(idx), dtype=np.float32)
        shp = np.array(a.shape[-3:])
        inb = np.all((idx >= 0) & (idx < shp), axis=1)
        ch = np.array(a.chunks[-3:])
        cid = idx // ch
        keys = {}
        for n in np.nonzero(inb)[0]:
            keys.setdefault(tuple(cid[n]), []).append(n)
        for c, ns in keys.items():
            lo = np.array(c) * ch
            blk = np.asarray(a[tuple(slice(int(l0), int(min(l0 + s, m))) for l0, s, m in zip(lo, ch, shp, strict=True))])
            ii = idx[ns] - lo
            out[ns] = blk[ii[:, 0], ii[:, 1], ii[:, 2]]
        return out
