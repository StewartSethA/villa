"""Per-cell growth guard for tifxyz lattices grown by VC3D's `vc_grow_seg_from_seed`.

Between growth rounds, keep growing a sheet while it stays sane and prune the FRONTIER where it
runs into vacuum, a hairpin, a crease/crumple, a self-intersection, or a sheet another segment
already holds. Pure numpy/scipy on the lattice (plus upstream's `vc_tifxyz_selfcross` when
`selfcross` is on); no change to the C++ tracer. `guarded_grow.py` is the round-by-round driver.

THE ALGORITHM

  1. per-CELL bad masks on the lattice grid (every criterion is a `GuardPolicy` flag + threshold):
       selfx    cells at the corners of quads `vc_tifxyz_selfcross` reports as crossing another quad
                of the same surface (checked on this round's new cells + a halo).
       vacuum   CT at the cell's 3-D position <= ct_th, over a (2*vacuum_win+1)^2 window.
       fold     turn radius < fold_radius_um over fold_arc_um of lattice line (rows and columns).
       plan     the cell's normal deviates > plan_deg from its window's mean normal (crumple).
       overlap  the cell lies within overlap_vox voxels along a neighbouring segment's normal
                (same-sheet estimator, `same_sheet.py`): the sheet is already held.
       ...plus the "shadow" criteria (quad_flip, stretch, normal_dev, ridge_hit, seam,
       wrap_spacing, curvature, empty_space), each always MEASURED and only ENFORCED when its
       `<name>_enforce` flag is set, and two segment-level stops (roughness, flatten_feedback).
  2. only bad regions that TOUCH THE FRONTIER are cut (plus margin_rings); a bad region wholly
       enclosed by good surface becomes a hole, not a cut.
  3. keep the largest connected piece; report why each cell went (`pruned_by`).
  4. regrowth guard: if a round's new surface mostly lies on previously pruned ground the
       frontier is BLOCKED and the segment ends (`frontier_blocked`) instead of looping.
  5. after the crop, re-run `vc_tifxyz_selfcross`; any remaining crossing stops the segment
       (`selfcross_nonzero`), and a hairpin ratio >= selfcross_hairpin_abort_ratio aborts early.

Comments below keep the source project's measurement notes (dates, sample sizes, the names of
its lab-notebook entries such as FINDINGS / growth_degeneracy). The numbers they cite are
summarised with their n in GUARDED_GROW.md and VALIDATION.md. `enabled` defaults to False.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

PREFIX = "grow.guard."
_S3 = np.ones((3, 3), bool)

# reasons in ATTRIBUTION order: a pruned cell is charged to the first that flags it
REASONS = ("selfx", "overlap", "vacuum", "fold", "plan",
          "quad_flip", "stretch", "normal_dev", "ridge_hit", "seam", "wrap_spacing", "curvature",
          "empty_space")
# roughness and flatten_feedback are SEGMENT-level scalars (frontier_roughness / flatten_feedback_check),
# not per-cell masks, so they are never in REASONS -- they can only ever be a segment stop, never a cut.


@dataclass(frozen=True)
class GuardPolicy:
    enabled: bool = False
    hosts: str = ""               # comma list of hosts the guard runs on ('' = every host): the CANARY switch
    # -- vacuum
    vacuum: bool = True
    ct_th: float = 5.0            # segment_yield's material criterion (`preds & CT>5`; insensitive 5..32)
    ct_level: int = 1             # CT pyramid level the sampler reads (1 = 2x coarser than level 0)
    vacuum_win: int = 2           # half-window (cells)
    vacuum_frac: float = 0.75     # window fraction that must be air (sweep: 0.6 -> 0.75 loses no verified area on clean/watch)
    # -- hairpin (geomaps.HAIRPIN_ARC_UM / HAIRPIN_RADIUS_UM: user-verified 2026-09-20)
    fold: bool = True
    fold_arc_um: float = 1000.0
    fold_radius_um: float = 300.0     # SWEEP: 500 um (the metric's radius) over-prunes; 300 keeps fleet-verified area (scripts/growguard/sweep.py)
    fold_half: int = 1                # cells flagged either side of a tight turn's centre (large = the whole window, the over-pruning first draft)
    # -- local planarity / crumple
    plan: bool = True
    plan_win: int = 3             # half-window (cells)
    plan_deg: float = 70.0
    plan_frac: float = 0.4        # a cell is a swamp cell when >= this fraction of its window is crumpled (speckle is noise)
    # -- same-sheet overlap (coverage.SAME_SHEET_VOX / ALIGN_DEG)
    overlap: bool = True
    overlap_vox: float = 4.0
    overlap_deg: float = 20.0
    overlap_keep_rings: int = 2
    # -- frontier prune
    min_bad_cells: int = 12       # a bad speck smaller than this is noise, not a swamp
    reach_rings: int = 2          # "touches the frontier" = within this many cells of an invalid cell
    margin_rings: int = 0         # sweep: any extra margin costs verified area for little gain
    min_keep_cells: int = 64
    keep_islands: bool = False
    # -- regrowth
    regrow_block_frac: float = 0.3
    regrow_tol_cells: float = 0.75   # x median edge: "on pruned ground"
    # -- neighbour-aware pause (user, 2026-09-25: pause frontier elements that overlap a real, mergeable
    #    neighbour, and queue the merge once the whole segment is mergeable with a friendly, clean one)
    merge_pause: bool = False
    merge_frontier_frac: float = 0.5      # share of the frontier band lying on ONE neighbour that triggers the pause
    merge_neigh_min_planarity: float = 0.5
    merge_neigh_max_fold: float = 0.05
    # -- large sheets (user, 2026-09-25: "I want the largest sheets we can")
    grow_to_sane: bool = False    # ignore the letters-sized target (`TARGET_MM`) while the frontier stays sane
    max_area_cm2: float = 60.0    # hard ceiling either way (RAM/time; finish.FINISH_MAX_CM2 must be raised in step)
    # -- ARM B (growth_degeneracy_2026-09-28.md): a post-round GEOMETRIC self-intersection check,
    #    distinct from `fold`/`plan` (which are per-cell proxies that, measured directly against
    #    `vc_tifxyz_selfcross`, barely reduce true crossing density -- 0.81->0.59 round1 but only
    #    0.85->0.80 round3 on the segment that motivated this arm). Default OFF; composes with the
    #    existing vacuum/fold/plan/overlap criteria (same `prune()` machinery, same frontier-touch /
    #    interior-kept / regrowth-block behaviour -- this is just another REASON).
    selfcross: bool = False
    selfcross_bin: str = ""              # explicit path to vc_tifxyz_selfcross; "" resolves via PATH/caller
    selfcross_env: dict | None = None    # subprocess env for the binary (LD_LIBRARY_PATH etc); None = inherit
    # process env, which is WRONG for a portable vc_kit (2026-09-29 production incident: the guard's
    # subprocess call did not carry LD_LIBRARY_PATH, so `vc_tifxyz_selfcross` resolved fine (fix 1)
    # but failed to LOAD ("libvc_core.so: cannot open shared object file") on every phi host the
    # first time selfcross was enabled fleet-wide -- caught loudly (mark_broken fired correctly),
    # reverted, fixed here. `stages.grow.resolve_selfcross_bin` fills this from `tool_env(fleet)`,
    # the SAME environment every other VC3D subprocess call in that stage already uses -- never
    # set directly in code that does not have a `fleet` to resolve it from.
    selfcross_timeout_s: float = 120.0
    # (3), coordinator 2026-09-29: "every self-intersection is disallowed; there is no threshold
    # to calibrate" -- a segment above this many triangles no longer SKIPS the check (silently
    # unverified, contradicting that principle). It instead triggers `selfcross_check_incremental`:
    # crop to this round's NEW cells + a spatial halo (`selfcross_crop_halo_vox`) and check that.
    # This field is now the crop's OWN size ceiling (still a genuine cost safety valve for a
    # pathological single-round growth blob): if even the incremental crop exceeds it, that is
    # treated the same as a broken binary (mark_broken(), coordinator item 2) -- loud, not quiet.
    selfcross_max_triangles: int = 400_000
    # spatial halo (voxels) added around this round's new-cell bounding box before cropping for
    # the incremental check: must exceed the binary's own broad-phase `--cell` (default 40 vox,
    # "affects speed, never contact verdicts or counts") by a comfortable margin, since the
    # binary can only ever find a contact between triangles within roughly one such neighbourhood
    # of each other -- a wider halo can only ADD old cells the crop didn't strictly need, never
    # exclude a contact the full-lattice check would have found. 200 vox = 5x the broad-phase
    # cell; not yet calibrated against a real pathologically-large segment (measured only on the
    # sizes in the n=400 sample, max 155,040 triangles -- see FINDINGS 2026-09-29).
    selfcross_crop_halo_vox: float = 200.0
    # STOP if the density (mean transverse contacts / triangle, both diagonals) measured on the
    # CROPPED surface is still above this. Target is exactly 0 (user directive): a value > 0 means
    # cutting the flagged quads did not remove every crossing (a hairpin can re-cross outside the
    # cells the census flagged as its own corners).
    selfcross_density_stop: float = 0.0
    # early-abort: geo_hairpin_lines / geo_lines_read (geomaps.fold_metrics), calibrated on a
    # held-out HALF of a 40-segment sample (growth_degeneracy_2026-09-28.md data; seed 20260929,
    # threshold chosen on the train half only): TPR 1.00 / FPR 0.20 (train, n=20) and TPR 1.00 /
    # FPR 0.29 (test, n=20) for the positive class "self-intersection density > 0.3". 0 disables.
    selfcross_hairpin_abort_ratio: float = 0.66
    selfcross_hairpin_min_lines: int = 20     # do not trust the ratio on a near-empty sample

    # -- SHADOW CRITERIA (2026-09-29, coordinator/user directive): nine more structural
    #    invariants, each with a COMPUTE flag (default True: measure and record every round,
    #    cheap) and a separate ENFORCE flag (default False for every one: measure only, never
    #    prune or stop, until a human-reviewed calibration says otherwise). This is how "added,
    #    and demonstrably firing" is delivered without a guessed threshold touching production --
    #    see SHADOW_CRITERIA_2026-09-29.md for what each measures, its unit, and why its default
    #    threshold is conservative (rarely-firing) rather than tuned: none of the nine has the
    #    n=40 held-out calibration selfx/hairpin got in the original report.
    quad_flip: bool = True
    quad_flip_enforce: bool = False           # threshold-free (cosang < 0 is unambiguous)
    stretch: bool = True
    stretch_ratio_th: float = 3.0             # edge length / lattice median edge, unitless
    stretch_enforce: bool = False
    roughness: bool = True                    # SEGMENT-level: frontier perimeter_um / sqrt(area_um2)
    roughness_th: float = 6.0                 # conservative (rarely-firing); no held-out data yet
    roughness_enforce: bool = False
    normal_dev: bool = True
    normal_dev_deg_th: float = 45.0           # degrees vs the normal-grid; conservative, no data yet
    normal_dev_enforce: bool = False
    ridge_hit: bool = True
    ridge_hit_vox: float = 3.0                # window, LEVEL-0 voxels, along the lattice normal
    ridge_hit_frac_th: float = 0.5            # SEGMENT-level would_stop: fraction of cells with no nearby ridge
    ridge_hit_enforce: bool = False
    seam: bool = True
    seam_step_vox: float = 12.0               # LEVEL-0 voxels; matches fuse3d_review's own planted-step test
    seam_search_vox: float = 30.0             # ridge search window MUST exceed seam_step_vox or a real jump can never be resolved on both sides
    seam_enforce: bool = False
    wrap_spacing: bool = True
    wrap_spacing_pitch_um: float = 700.0      # conservative mid-range default (shape_metrics' own fallback)
    wrap_spacing_enforce: bool = False
    curvature: bool = True
    curvature_frac_th: float = 0.30           # turn radius < this * distance-from-umbilicus is implausible
    curvature_enforce: bool = False
    flatten_feedback: bool = True             # SEGMENT-level: recent flatten(s) collapsed
    flatten_feedback_valid_th: float = 0.20   # stages/render.py's OWN "collapsed flatten" bar, reused not guessed
    flatten_feedback_streak: int = 2
    flatten_feedback_enforce: bool = False
    # -- empty_space (coordinator, 2026-09-29 ARM E addition): "many PHerc0211 segments wander
    #    into empty space undetected" -- a CHEAP ridge_hit: single-point (no +-window search)
    #    surface-prediction support at each FRONTIER cell's own position, not the whole lattice.
    #    O(frontier cells), one pred_sampler call/round -- meant as a per-round early-warning that
    #    is far cheaper than ridge_hit's windowed search, at the cost of being less tolerant of a
    #    slightly-off-surface frontier (no search radius).
    empty_space: bool = True
    empty_space_frac_th: float = 0.5          # SEGMENT-level would_stop: frontier fraction with no support AT ALL
    empty_space_enforce: bool = False


def policy_from_dict(d: dict | None) -> GuardPolicy:
    """Defaults, overridden by a flat {field: value} dict (a preset JSON file, or CLI overrides).
    Unknown keys raise: a misspelt policy field must not silently run the default."""
    by = {f.name: f for f in fields(GuardPolicy)}
    kw = {}
    for k, v in (d or {}).items():
        k = k.removeprefix(PREFIX)
        if k not in by:
            raise KeyError(f"unknown GuardPolicy field {k!r}")
        t = by[k].type
        if t in (bool, "bool"):
            kw[k] = v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on")
        elif t in (int, "int"):
            kw[k] = int(float(v))
        elif t in (str, "str"):
            kw[k] = str(v)
        elif t in (float, "float"):
            kw[k] = float(v)
        else:
            kw[k] = v
    return GuardPolicy(**kw)


def load_policy(path: str) -> GuardPolicy:
    """A preset file (presets/*.json): {"policy": {...}} or a flat dict."""
    d = json.loads(Path(path).read_text())
    return policy_from_dict(d.get("policy", d))


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
    (geomaps.fold_metrics' criterion, per cell). A window needs every cell valid."""
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


# ----------------------------------------------------------------------------- shadow criteria
# 2026-09-29: nine more, all SHADOW MODE (compute + record every round, never prune/stop unless
# `<crit>_enforce` is set -- see GuardPolicy). Each takes exactly the inputs it needs; the ones
# that need an external resource (a surface-prediction sampler, an umbilicus, a normal-grid
# sampler, the pipeline DB) accept it as a plain callable/None so growth_guard.py itself never
# has to import volumes/scrolls -- the caller (stages/grow.py) resolves those the same way it
# already resolves the CT sampler, and passes None where nothing is available THIS ROUND (a
# criterion the caller cannot feed is SKIPPED, never scored as "clean").

def quad_flip_mask(P, V, pol: GuardPolicy):
    """Cells whose own normal points MORE than 90 deg from its window-mean neighbourhood normal
    -- the surface has locally turned inside out (a negative-signed-area quad), the sharpest and
    only threshold-free criterion here. A strict subset of what `plan_mask` can eventually catch
    at a very permissive `plan_deg`, kept separate because it needs no threshold to justify."""
    n, ok = grid_normals(P, V)
    w = 2 * pol.plan_win + 1
    wt = ok.astype(np.float64)
    mean = np.stack([ndi.uniform_filter(n[..., c] * wt, size=w, mode="constant") for c in range(3)], axis=-1)
    norm = np.linalg.norm(mean, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = np.einsum("hwc,hwc->hw", n, mean) / np.maximum(norm, 1e-9)
    return ok & (norm > 1e-9) & (cosang < 0.0)


def stretch_ratio_p90(P, V) -> float | None:
    """p90 of (3-D edge length / lattice median edge) over every valid edge -- the SCALAR the
    growth-ab schema and guard_review's EXTRA_THRESHOLD_CRITERIA call `stretch_p90`."""
    med = median_edge(P, V)
    if not np.isfinite(med) or med <= 0:
        return None
    ratios = []
    for ax in (0, 1):
        a, b = (P[1:], P[:-1]) if ax == 0 else (P[:, 1:], P[:, :-1])
        ok = (V[1:] & V[:-1]) if ax == 0 else (V[:, 1:] & V[:, :-1])
        if ok.any():
            ratios.append(np.linalg.norm(a - b, axis=-1)[ok] / med)
    if not ratios:
        return None
    return round(float(np.percentile(np.concatenate(ratios), 90)), 4)


def stretch_mask(P, V, pol: GuardPolicy):
    """A cell touching a 3-D edge longer than `stretch_ratio_th` x the lattice's own median edge
    -- local over-stretching, a precursor to a bad flatten (`edge_med_vox`, stages/render.py,
    measures the same quantity as a single segment-level number; this is the per-cell version)."""
    med = median_edge(P, V)
    out = np.zeros(V.shape, bool)
    if not np.isfinite(med) or med <= 0:
        return out
    th = pol.stretch_ratio_th * med
    for ax in (0, 1):
        a, b = (P[1:], P[:-1]) if ax == 0 else (P[:, 1:], P[:, :-1])
        ok = (V[1:] & V[:-1]) if ax == 0 else (V[:, 1:] & V[:, :-1])
        d = np.linalg.norm(a - b, axis=-1)
        bad = ok & (d > th)
        if ax == 0:
            out[1:][bad] = True
            out[:-1][bad] = True
        else:
            out[:, 1:][bad] = True
            out[:, :-1][bad] = True
    return out & V


def frontier_roughness(V, pol: GuardPolicy, voxel_um: float, step_size: float = 20.0) -> dict:
    """SEGMENT-level: frontier perimeter_um / sqrt(area_um2) -- compactness of the growing edge.
    A smooth, round frontier reads low; a fractal, many-fingered one (often preceding a crumple)
    reads high. `step_size` is the tracer's lattice pitch (voxels/cell) -- 20 is the fleet
    default (DEFAULT_STEP_SIZE, stages/grow.py); pass the segment's own recorded one when known."""
    band = frontier_band(V, 0)
    cell_um = step_size * voxel_um
    perim_um = float(band.sum()) * cell_um
    area_um2 = float(V.sum()) * cell_um ** 2
    r = perim_um / np.sqrt(area_um2) if area_um2 > 0 else float("nan")
    return {"cells": int(band.sum()), "value": round(r, 4) if np.isfinite(r) else None,
           "would_stop": bool(np.isfinite(r) and r > pol.roughness_th)}


def _normal_dev_degrees(P, V, sampler):
    """(idx, dev_degrees) or (None, None) when `sampler` is None. Shared by `normal_dev_mask`
    (threshold) and `normal_dev_p90` (the scalar growth-ab/guard_review call `normal_dev_p90_deg`)."""
    if sampler is None:
        return None, None
    n, ok = grid_normals(P, V)
    idx = np.nonzero(ok)
    if not len(idx[0]):
        return idx, np.zeros(0)
    ref = np.asarray(sampler(P[idx]), dtype=np.float64)
    refn = ref / np.maximum(np.linalg.norm(ref, axis=-1, keepdims=True), 1e-9)
    cosang = np.clip(np.abs(np.einsum("nc,nc->n", n[idx], refn)), -1.0, 1.0)   # abs: orientation sign is a convention, not a defect
    return idx, np.degrees(np.arccos(cosang))


def normal_dev_p90(P, V, sampler) -> float | None:
    _, dev = _normal_dev_degrees(P, V, sampler)
    return round(float(np.percentile(dev, 90)), 3) if dev is not None and len(dev) else None


def normal_dev_mask(P, V, sampler, pol: GuardPolicy):
    """Angle between the lattice's OWN normal (`grid_normals`) and a normal-grid's published
    normal at the same point, in degrees. `sampler(xyz) -> (nx, ny, nz)` per point, or None where
    no normal-grid sampler is wired for this scroll/host this round (SKIPPED -- returns None, not
    an empty/clean mask: absence of the reference is not evidence of agreement)."""
    idx, dev = _normal_dev_degrees(P, V, sampler)
    if idx is None:
        return None
    out = np.zeros(V.shape, bool)
    if len(dev):
        out[idx] = dev > pol.normal_dev_deg_th
    return out


def _ridge_offset_slow(pts: np.ndarray, nv: np.ndarray, pred_sampler, window_vox: float, step: float = 1.0) -> np.ndarray:
    """Reference implementation, one point at a time -- kept ONLY as the ground truth
    `tests/test_growth_guard_ridge_offset_perf.py` checks the vectorised `_ridge_offset` against
    (measured 39.6-74.2 s/segment, growth_degeneracy_2026-09-28.md's 2026-09-29 follow-up
    replay -- do not call this from production code)."""
    offs = np.arange(-window_vox, window_vox + 1e-6, step)
    vals = np.zeros((len(pts), len(offs)), bool)
    for k, o in enumerate(offs):
        vals[:, k] = np.asarray(pred_sampler(pts + nv * o)) > 0
    r = np.full(len(pts), np.nan)
    for i in range(len(pts)):
        row = vals[i]
        if not row.any():
            continue
        d = np.diff(np.r_[0, row.astype(int), 0])
        st = np.nonzero(d == 1)[0]
        en = np.nonzero(d == -1)[0]
        centres = offs[0] + (st + en - 1) / 2.0 * step
        r[i] = centres[np.argmin(np.abs(centres))]
    return r


def _ridge_offset(pts: np.ndarray, nv: np.ndarray, pred_sampler, window_vox: float, step: float = 1.0) -> np.ndarray:
    """Nearest predicted-surface crossing along the normal, within +-window_vox (LEVEL-0 voxels),
    or NaN. Simplified from docs/experiments/fuse3d_review/code/s1_ct.py / s3_jump.py's sub-voxel
    edge fit to a fixed-offset sample, appropriate for a per-round guard cost, not a research
    measurement -- ridge_hit_mask/seam_mask both build on this.

    VECTORISED (2026-09-29 follow-up): the original per-point Python loop measured 39.6-74.2 s
    per real segment (growth_degeneracy_2026-09-28.md's shadow-criteria replay); this version
    finds every point's run of True samples in ONE `ndi.label` call (row-only connectivity, so a
    run never crosses between points) and picks each point's run nearest offset 0 with a single
    sort + `np.unique(..., return_index=True)` -- no per-point Python loop. Verified to return
    IDENTICAL values to `_ridge_offset_slow` on real segment data (`tests/
    test_growth_guard_ridge_offset_perf.py`), same run-centre semantics, not an approximation.

    Measured separately (FINDINGS.md 2026-09-29 follow-up): on real data, `pred_sampler` I/O
    (chunk fetch + decompress from the real prediction zarr) is the DOMINANT cost -- 35.9 s of a
    39.6 s total for one real segment at window_vox=30 -- not the post-processing this function
    vectorises. So ALL `window_vox` offsets are queried in ONE combined call (every offset's
    points concatenated) rather than one call per offset: at a 192-voxel chunk size and a <=60-
    voxel window, most of a point's offsets land in the SAME zarr chunk, and `ZarrSampler`
    already reads each touched chunk once per call regardless of how many of its points hit it
    -- so one big call avoids re-fetching that chunk once per offset. `pred_sampler` must accept
    an arbitrary-length point array (every sampler in this module already does)."""
    vals, offs = _sample_window(pts, nv, pred_sampler, window_vox, step)
    return _reduce_to_offset(vals, offs, step)


def _sample_window(pts: np.ndarray, nv: np.ndarray, pred_sampler, window_vox: float, step: float = 1.0):
    """(vals bool (N, W), offs (W,)): `pred_sampler` queried ONCE for every point x every offset
    combined -- see `_ridge_offset`'s docstring for why this beats one call per offset. Factored
    out of `_ridge_offset` so `ridge_hit_and_seam_masks` can sample ONCE at the wider window and
    derive both criteria (2026-09-29, coordinator item (c): "share one wide prediction sample
    between ridge_hit and seam")."""
    offs = np.arange(-window_vox, window_vox + 1e-6, step)
    N, W = len(pts), len(offs)
    if N == 0:
        return np.zeros((0, W), bool), offs
    combined = (pts[:, None, :] + nv[:, None, :] * offs[None, :, None]).reshape(N * W, 3)
    vals = (np.asarray(pred_sampler(combined)) > 0).reshape(N, W)
    return vals, offs


def _reduce_to_offset(vals: np.ndarray, offs: np.ndarray, step: float = 1.0) -> np.ndarray:
    """Given a (N, W) boolean sample and its (W,) offsets, the per-point nearest-to-zero run
    centre (or NaN) -- the reduction half of `_ridge_offset`, factored out so it can run on a
    COLUMN SLICE of an already-sampled wider array (see `ridge_hit_and_seam_masks`)."""
    N, W = vals.shape
    r = np.full(N, np.nan)
    if N == 0 or W == 0 or not vals.any():
        return r
    idx0 = int(np.argmin(np.abs(offs)))
    struct = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], bool)   # horizontal (within-row) connectivity ONLY
    lab, n = ndi.label(vals, structure=struct)
    if n == 0:
        return r
    rows, cols = np.nonzero(lab)
    labels = lab[rows, cols]
    # per-run: mean column (its centre) and which point (row) it belongs to -- every pixel of a
    # given label shares one row by construction (no cross-row connectivity), so any is fine.
    counts = np.bincount(labels, minlength=n + 1)
    col_sums = np.bincount(labels, weights=cols.astype(np.float64), minlength=n + 1)
    centres = col_sums[1:] / counts[1:]                          # index 0 = label 1
    label_row = np.zeros(n + 1, dtype=np.int64)
    label_row[labels] = rows                                     # last write per label is fine (same row throughout)
    dist = np.abs(centres - idx0)
    order = np.lexsort((dist, label_row[1:]))                    # sort by row, then by distance to idx0
    sorted_rows = label_row[1:][order]
    _, first = np.unique(sorted_rows, return_index=True)         # first (= closest) run per row
    best_rows = sorted_rows[first]
    best_centres = centres[order[first]]
    r[best_rows] = offs[0] + best_centres * step
    return r


def _seam_edges_from_r(V: np.ndarray, idx, pts: np.ndarray, nv: np.ndarray, n_full: np.ndarray,
                       r: np.ndarray, pol: GuardPolicy) -> np.ndarray:
    """The edge-comparison half of the seam criterion, factored out of `seam_mask` so
    `ridge_hit_and_seam_masks` can reuse it on a shared sample's `r`. `n_full` is the FULL
    (H, W, 3) normal field (`grid_normals`'s first return); `r` is per-point (same order as
    `idx`)."""
    R = np.full(V.shape + (3,), np.nan)
    R[idx] = pts + nv * np.where(np.isfinite(r), r, 0.0)[:, None]
    known = np.zeros(V.shape, bool)
    known[idx] = np.isfinite(r)
    out = np.zeros(V.shape, bool)
    for ax in (0, 1):
        a = R[1:] if ax == 0 else R[:, 1:]
        b = R[:-1] if ax == 0 else R[:, :-1]
        ka = known[1:] if ax == 0 else known[:, 1:]
        kb = known[:-1] if ax == 0 else known[:, :-1]
        na = n_full[1:] if ax == 0 else n_full[:, 1:]
        nb = n_full[:-1] if ax == 0 else n_full[:, :-1]
        va = V[1:] if ax == 0 else V[:, 1:]
        vb = V[:-1] if ax == 0 else V[:, :-1]
        good = va & vb & ka & kb
        nbar = na + nb
        nbar = nbar / np.maximum(np.linalg.norm(nbar, axis=-1, keepdims=True), 1e-9)
        cut = np.abs(np.einsum("...c,...c->...", b - a, nbar))
        bad = good & (cut > pol.seam_step_vox)
        if ax == 0:
            out[1:][bad] = True
            out[:-1][bad] = True
        else:
            out[:, 1:][bad] = True
            out[:, :-1][bad] = True
    return out & V


def ridge_hit_and_seam_masks(P, V, pred_sampler, pol: GuardPolicy):
    """Shared-sample version of `ridge_hit_mask` + `seam_mask` (2026-09-29, coordinator item
    (c)): samples ONCE at the wider `seam_search_vox` window, then derives ridge_hit's narrower
    `ridge_hit_vox` answer by slicing the SAME sampled array's central columns and reducing that
    slice separately -- avoiding a second `pred_sampler` call (and its I/O) entirely, since
    `ridge_hit_vox <= seam_search_vox` always. Returns (ridge_hit_mask, seam_mask), either None
    if `pred_sampler` is None. `shadow_round` uses this instead of calling `ridge_hit_mask`/
    `seam_mask` separately whenever both are enabled; the two standalone functions are kept
    (and still tested) for callers that want only one."""
    if pred_sampler is None:
        return None, None
    n_full, ok = grid_normals(P, V)
    idx = np.nonzero(ok)
    if not len(idx[0]):
        empty = np.zeros(V.shape, bool)
        return empty, empty
    pts, nv = P[idx], n_full[idx]
    wide_vals, wide_offs = _sample_window(pts, nv, pred_sampler, pol.seam_search_vox)

    center = int(np.argmin(np.abs(wide_offs)))
    narrow = np.abs(wide_offs) <= pol.ridge_hit_vox + 1e-9
    r_ridge = _reduce_to_offset(wide_vals[:, narrow], wide_offs[narrow])
    ridge_out = np.zeros(V.shape, bool)
    ridge_out[idx] = ~np.isfinite(r_ridge)

    r_seam = _reduce_to_offset(wide_vals, wide_offs)
    seam_out = _seam_edges_from_r(V, idx, pts, nv, n_full, r_seam, pol)
    del center   # only used to document that wide_offs is symmetric about it; not otherwise needed
    return ridge_out & V, seam_out


def empty_space_mask(P, V, pred_sampler, pol: GuardPolicy):
    """(ARM E addition, coordinator 2026-09-29: "many PHerc0211 segments wander into empty space
    undetected".) A CHEAP `ridge_hit`: single-point surface-prediction support (no +-window
    search) at each FRONTIER cell's own lattice position, not the whole lattice's interior --
    the frontier is exactly where "wandering into empty space" would first show up, and this is
    meant as a much lighter per-round check than `ridge_hit_mask`'s windowed search (one
    `pred_sampler` call over `frontier_band` cells only, instead of `window_vox`-many offsets
    over every valid cell). `pred_sampler(xyz) -> value` (>0 = predicted surface), or None where
    no prediction is local to this host/scroll this round (SKIPPED, never scored clean)."""
    if pred_sampler is None:
        return None
    band = frontier_band(V, pol.reach_rings)
    idx = np.nonzero(band)
    out = np.zeros(V.shape, bool)
    if not len(idx[0]):
        return out
    supported = np.asarray(pred_sampler(P[idx])) > 0
    out[idx] = ~supported
    return out


def ridge_hit_mask(P, V, pred_sampler, pol: GuardPolicy):
    """Cells with NO surface-prediction hit within +-ridge_hit_vox of the lattice along their own
    normal -- unsupported by the independent surface-prediction model. `pred_sampler(xyz) ->
    value` (>0 = predicted surface, e.g. a ZarrSampler over the thresholded m7 prediction), or
    None where no prediction is local to this host this round (SKIPPED). Standalone version --
    `shadow_round` uses `ridge_hit_and_seam_masks` instead when both criteria are enabled, to
    share one sample."""
    if pred_sampler is None:
        return None
    n, ok = grid_normals(P, V)
    idx = np.nonzero(ok)
    if not len(idx[0]):
        return np.zeros(V.shape, bool)
    r = _ridge_offset(P[idx], n[idx], pred_sampler, pol.ridge_hit_vox)
    out = np.zeros(V.shape, bool)
    out[idx] = ~np.isfinite(r)
    return out


def seam_mask(P, V, pred_sampler, pol: GuardPolicy):
    """Wrap-jump seam detector (fuse3d_review's "ridge-coordinate seam" idea, code/s3_jump.py):
    for each edge between adjacent lattice cells, correct BOTH endpoints to their nearest
    predicted-ridge crossing along the normal (`_ridge_offset`, wider window than ridge_hit's own
    so a genuine jump is still findable); an edge whose corrected endpoints differ by more than
    `seam_step_vox` along their averaged normal is a coordinate discontinuity in the SAME
    predicted surface -- the lattice jumped across a scan/wrap gap, not a smooth bend. Edges with
    an unresolved endpoint on either side are left alone (unknown, not clean, not cut).
    `pred_sampler` as `ridge_hit_mask`; None SKIPS this criterion entirely. Standalone version --
    `shadow_round` uses `ridge_hit_and_seam_masks` instead when both criteria are enabled."""
    if pred_sampler is None:
        return None
    n, ok = grid_normals(P, V)
    idx = np.nonzero(ok)
    if not len(idx[0]):
        return np.zeros(V.shape, bool)
    r = _ridge_offset(P[idx], n[idx], pred_sampler, pol.seam_search_vox)
    return _seam_edges_from_r(V, idx, P[idx], n[idx], n, r, pol)


def wrap_spacing_mask(P, V, umbilicus_of_z, voxel_um: float, pol: GuardPolicy):
    """A lattice edge whose distance-from-umbilicus (radius) jumps by more than one winding
    pitch (`shape_metrics.winding_span_and_jump_frac`'s `radial_jump_frac` criterion, made
    per-cell): the edge crossed to a different wrap of the spiral. `umbilicus_of_z(z) -> (ux,
    uy)`, or None where no umbilicus is registered/trusted for this scroll (SKIPPED)."""
    if umbilicus_of_z is None or not V.any():
        return None
    idx = np.nonzero(V)
    z = P[idx][:, 2]
    ux, uy = umbilicus_of_z(z)
    r = np.hypot(P[idx][:, 0] - ux, P[idx][:, 1] - uy)
    R = np.full(V.shape, np.nan)
    R[idx] = r
    pitch_vox = pol.wrap_spacing_pitch_um / voxel_um if voxel_um else pol.wrap_spacing_pitch_um
    out = np.zeros(V.shape, bool)
    for ax in (0, 1):
        a = R[1:] if ax == 0 else R[:, 1:]
        b = R[:-1] if ax == 0 else R[:, :-1]
        ok = (V[1:] & V[:-1]) if ax == 0 else (V[:, 1:] & V[:, :-1])
        d = np.abs(a - b)
        bad = ok & (d > pitch_vox)
        if ax == 0:
            out[1:][bad] = True
            out[:-1][bad] = True
        else:
            out[:, 1:][bad] = True
            out[:, :-1][bad] = True
    return out & V


def curvature_mask(P, V, voxel_um: float, umbilicus_of_z, pol: GuardPolicy):
    """A local in-plane turn radius much tighter than the sheet's own distance from the
    umbilicus is geometrically implausible for a spiral wrap (curvature ~ 1/r) and is a fold, not
    a wrap: reuses `fold_mask`'s turn-radius construction verbatim, with the threshold scaled to
    THIS segment's own median umbilicus distance (`curvature_frac_th * r_med`) instead of
    `fold_mask`'s fixed `fold_radius_um` -- the same shape check, a radius-relative threshold.
    None (no umbilicus) SKIPS this criterion."""
    if umbilicus_of_z is None or not V.any():
        return None
    idx = np.nonzero(V)
    z = P[idx][:, 2]
    ux, uy = umbilicus_of_z(z)
    r = np.hypot(P[idx][:, 0] - ux, P[idx][:, 1] - uy)
    r_med = float(np.median(r)) if len(r) else float("nan")
    if not np.isfinite(r_med) or r_med <= 0:
        return np.zeros(V.shape, bool)
    from dataclasses import replace
    pol2 = replace(pol, fold_radius_um=pol.curvature_frac_th * r_med, fold_arc_um=pol.fold_arc_um, fold_half=pol.fold_half)
    return fold_mask(P, V, voxel_um, pol2)


def flatten_feedback_check(recent_flatten_reasons, seg: str, pol: GuardPolicy) -> dict:
    """Have this segment's most recent flatten(s) collapsed `flatten_feedback_streak` times in a
    row? `recent_flatten_reasons(seg, n)` is the CALLER's hook returning the newest n flatten
    outcome strings (whatever records flattens in your pipeline); a reason containing "collapsed"
    counts. SEGMENT-level (no mask). Without a hook this criterion is not computed."""
    rows = list(recent_flatten_reasons(seg, pol.flatten_feedback_streak) or [])
    collapsed = sum(1 for r in rows if r and "collapsed" in str(r).lower())
    return {"cells": None, "checked": len(rows), "collapsed_recent": collapsed,
            "would_stop": bool(collapsed >= pol.flatten_feedback_streak and len(rows) >= pol.flatten_feedback_streak)}


@dataclass
class ShadowContext:
    """External resources the nine shadow criteria need but growth_guard.py never resolves
    itself (no volumes/scrolls import here): the caller (guarded_grow.py) builds whichever of
    these it can for this round and leaves the rest None, which SKIPS that criterion rather than
    scoring it as clean. All optional; an all-None context still runs quad_flip/stretch/roughness
    (purely geometric, no external data needed)."""
    pred_sampler: object = None          # (xyz) -> value; >0 = predicted surface
    normal_sampler: object = None        # (xyz) -> (nx, ny, nz); the normal-grid's own normal
    umbilicus_of_z: object = None        # (z) -> (ux, uy)
    db: object = None                    # callable (seg, n) -> newest n flatten outcome strings, for flatten_feedback_check
    seg: str | None = None
    step_size: float = 20.0


# per-cell criteria that live in REASONS; roughness/flatten_feedback are segment-level (see below)
SHADOW_CELL_CRITERIA = ("quad_flip", "stretch", "normal_dev", "ridge_hit", "seam", "wrap_spacing", "curvature")


def shadow_round(P, V, pol: GuardPolicy, voxel_um: float, ctx: ShadowContext | None = None) -> dict:
    """Compute EVERY shadow criterion this round (cheap ones always; the ones needing external
    data whenever `ctx` supplies it), regardless of any `_enforce` flag -- this is the "shadow"
    half of shadow mode. Returns {name: {"mask": ndarray|None, **info}} for the seven per-cell
    criteria, plus "roughness" and "flatten_feedback" with `"mask": None` (segment-level).
    A None mask means SKIPPED this round (not computed), not "no cells flagged" -- callers must
    keep that distinction when recording `guard_<crit>_cells` (None, not 0)."""
    ctx = ctx or ShadowContext()
    out: dict[str, dict] = {}

    def rec(name, mask, extra=None):
        info = {"mask": mask, "cells": (None if mask is None else int(mask.sum()))}
        if extra:
            info.update(extra)
        out[name] = info

    if pol.quad_flip:
        rec("quad_flip", quad_flip_mask(P, V, pol))
    if pol.stretch:
        # stretch_p90: the growth-ab SCHEMA.md / guard_review.EXTRA_THRESHOLD_CRITERIA column name
        rec("stretch", stretch_mask(P, V, pol), {"stretch_p90": stretch_ratio_p90(P, V)})
    if pol.normal_dev:
        # normal_dev_p90_deg: same cross-module contract
        rec("normal_dev", normal_dev_mask(P, V, ctx.normal_sampler, pol),
            {"normal_dev_p90_deg": normal_dev_p90(P, V, ctx.normal_sampler)})
    if pol.ridge_hit and pol.seam:
        # share ONE wide pred_sampler call between the two (2026-09-29, coordinator item (c)):
        # ridge_hit_vox <= seam_search_vox always, so seam's sample already covers ridge_hit's.
        m_ridge, m_seam = ridge_hit_and_seam_masks(P, V, ctx.pred_sampler, pol)
        n_valid = int(V.sum())
        frac = (m_ridge.sum() / n_valid) if (m_ridge is not None and n_valid) else None
        rec("ridge_hit", m_ridge, {"would_stop": bool(frac is not None and frac > pol.ridge_hit_frac_th),
                                   "ridge_hit": bool(frac is not None and frac > pol.ridge_hit_frac_th),
                                   "frac_unsupported": (round(frac, 4) if frac is not None else None)})
        rec("seam", m_seam, {"seam_steps": (None if m_seam is None else int(m_seam.sum()))})
    else:
        if pol.ridge_hit:
            m = ridge_hit_mask(P, V, ctx.pred_sampler, pol)
            n_valid = int(V.sum())
            frac = (m.sum() / n_valid) if (m is not None and n_valid) else None
            # ridge_hit: guard_review reads this as a plain boolean ("did this round fail the
            # ridge-support check"), not a cell count -- would_stop IS that boolean.
            rec("ridge_hit", m, {"would_stop": bool(frac is not None and frac > pol.ridge_hit_frac_th),
                                 "ridge_hit": bool(frac is not None and frac > pol.ridge_hit_frac_th),
                                 "frac_unsupported": (round(frac, 4) if frac is not None else None)})
        if pol.seam:
            m = seam_mask(P, V, ctx.pred_sampler, pol)
            # seam_steps: guard_review reads this as a count ("(v or 0) > 0"); the flagged-edge count
            rec("seam", m, {"seam_steps": (None if m is None else int(m.sum()))})
    if pol.empty_space:
        m = empty_space_mask(P, V, ctx.pred_sampler, pol)
        n_front = int(frontier_band(V, pol.reach_rings).sum())
        frac = (m.sum() / n_front) if (m is not None and n_front) else None
        # naming mirrors ridge_hit's: guard_review-style boolean AND the fraction, so a UI/replay
        # reader that already knows ridge_hit's contract needs no new convention to learn.
        rec("empty_space", m, {"would_stop": bool(frac is not None and frac > pol.empty_space_frac_th),
                               "empty_space": bool(frac is not None and frac > pol.empty_space_frac_th),
                               "frac_unsupported": (round(frac, 4) if frac is not None else None)})
    if pol.wrap_spacing:
        rec("wrap_spacing", wrap_spacing_mask(P, V, ctx.umbilicus_of_z, voxel_um, pol))
    if pol.curvature:
        rec("curvature", curvature_mask(P, V, voxel_um, ctx.umbilicus_of_z, pol))
    if pol.roughness:
        info = frontier_roughness(V, pol, voxel_um, ctx.step_size)
        out["roughness"] = {"mask": None, **info}
    if pol.flatten_feedback and ctx.db is not None and ctx.seg is not None:
        info = flatten_feedback_check(ctx.db, ctx.seg, pol)
        out["flatten_feedback"] = {"mask": None, **info}
    return out


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
    prefilter, using vesuvius_pipeline.coverage.normal_distance (the production gate's
    estimator). `add(seg, X, Y, Z)` thins nothing; pass a decimated lattice if it is huge."""

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
        old = getattr(C, "ALIGN_DEG", None)
        C.ALIGN_DEG = self.deg
        try:
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
        finally:
            if old is not None:
                C.ALIGN_DEG = old
        return hit, name

    def __call__(self, pts, nrm, ok, own=None):
        return self.attribute(pts, nrm, ok, own)[0]


def merge_readiness(X, Y, Z, pol: GuardPolicy, cover, neighbour_ok, voxel_um: float, own: str | None = None) -> dict:
    """Is this segment's FRONTIER now mostly lying on one real, clean neighbour, and is the segment
    itself clean? Then growing on only adds a duplicate: pause it and hand the pair to the merger.
    `neighbour_ok(name) -> bool` is the caller's quality gate (planarity >= merge_neigh_min_planarity,
    fold < merge_neigh_max_fold, not mush/degenerate), read from the segment inventory."""
    P, V = lattice_frame(X, Y, Z)
    out = {"mergeable": False, "with": None, "band_share": 0.0, "reason": None}
    if cover is None or not V.any():
        out["reason"] = "no neighbour index"
        return out
    band = V & frontier_band(V, pol.reach_rings)
    n, ok = grid_normals(P, V)
    idx = np.nonzero(band)
    if not len(idx[0]):
        out["reason"] = "no frontier"
        return out
    hit, names = cover.attribute(P[idx], n[idx], ok[idx], own)
    if not hit.any():
        out["reason"] = "frontier overlaps nothing"
        return out
    vals, counts = np.unique(names[hit], return_counts=True)
    k = int(np.argmax(counts))
    out["with"] = str(vals[k])
    out["band_share"] = round(float(counts[k] / len(hit)), 4)
    if out["band_share"] < pol.merge_frontier_frac:
        out["reason"] = "frontier share on the best neighbour below threshold"
        return out
    if not neighbour_ok(out["with"]):
        out["reason"] = "neighbour fails quality gate (mush / hairpin / non-planar)"
        return out
    from . import geometry as GM
    pl = GM.planarity_score(P, V)
    fold = float(fold_mask(P, V, voxel_um, pol).mean())
    out["own_planarity"], out["own_fold_cells"] = round(float(pl), 3), round(fold, 4)
    if not (np.isfinite(pl) and pl >= pol.merge_neigh_min_planarity and fold <= pol.merge_neigh_max_fold):
        out["reason"] = "segment itself fails the quality gate"
        return out
    out["mergeable"] = True
    return out


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
    extra: dict = field(default_factory=dict)   # e.g. {"selfcross_pre": {...}} -- ARM B diagnostics

    def summary(self) -> dict:
        out = {"cells_before": self.cells_before, "cells_after": self.cells_after,
               "pruned_by": self.pruned_by, "components_cut": self.components_cut,
               "interior_bad_components_kept": self.interior_bad_kept,
               "frontier_bad_frac": round(self.frontier_bad_frac, 4), "stop": self.stop}
        out.update(self.extra)
        return out


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


def evaluate(X, Y, Z, pol: GuardPolicy, voxel_um: float, sampler=None, cover=None, own=None, anchor=None,
            selfx_mask: np.ndarray | None = None, shadow: dict | None = None) -> GuardResult:
    """All enabled criteria on one lattice, then `prune`. `selfx_mask` (ARM B) is computed by the
    caller (`guard_tifxyz`, which has the on-disk `src` a subprocess census needs) rather than
    here. `shadow` (the nine 2026-09-29 criteria, from `shadow_round`) is measured regardless of
    enforcement, but a criterion's mask is only added to `masks` -- i.e. only actually PRUNED --
    when its own `pol.<name>_enforce` is True. This is the shadow/enforce split: measurement and
    action are two different questions, answered by two different flags."""
    P, V = lattice_frame(X, Y, Z)
    masks = {}
    if pol.selfcross and selfx_mask is not None:
        masks["selfx"] = selfx_mask
    if pol.overlap and cover is not None:
        masks["overlap"] = overlap_mask(P, V, cover, pol, own)
    if pol.vacuum and sampler is not None:
        masks["vacuum"] = vacuum_mask(P, V, sampler, pol)
    if pol.fold:
        masks["fold"] = fold_mask(P, V, voxel_um, pol)
    if pol.plan:
        masks["plan"] = plan_mask(P, V, pol)
    for name in SHADOW_CELL_CRITERIA:
        if shadow and getattr(pol, f"{name}_enforce", False):
            m = shadow.get(name, {}).get("mask")
            if m is not None:
                masks[name] = m
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


def selfcross_check(src: str, shape: tuple[int, int], pol: GuardPolicy) -> tuple[np.ndarray | None, dict]:
    """ARM B (growth_degeneracy_2026-09-28.md): run upstream's `vc_tifxyz_selfcross` (report-only,
    never modifies `src`) and mark every lattice cell that is a corner of a flagged quad -- fed into
    `evaluate()` as reason "selfx", so it gets the SAME frontier-touch / interior-kept / largest-
    component / regrowth-block treatment as vacuum/fold/plan, for free.

    Returns (mask or None, info). `info['ran']` is False, and mask is None, whenever the check did
    NOT produce a trustworthy answer (binary missing, subprocess failure, timeout, lattice above
    `selfcross_max_triangles`) -- that is recorded in `info['error']`/`info['skipped']` and treated
    as "no check this round", never as "clean": a broken tool must not silently mask real crossings,
    and it must not be able to stop growth on its own failure either.

    Cost, measured (growth_degeneracy_2026-09-28.md, n=40 real segments up to ~50k triangles /
    ~9 cm2): wall time p50 0.050 s, p90 0.078 s, max 0.085 s -- negligible against a grow round's
    ~700-1800 CPU-s. Re-measure `info['wall_s']` in production before relying on this number at
    much larger sizes than tested here."""
    info: dict = {"ran": False}
    if not pol.selfcross:
        info["skipped"] = "policy_off"
        return None, info
    binp = pol.selfcross_bin or shutil.which("vc_tifxyz_selfcross")
    if not binp or not os.path.exists(binp):
        info["skipped"] = "binary_not_found"
        return None, info
    out = Path(src) / f".selfcross_guard_{os.getpid()}.json"
    t0 = time.time()
    try:
        r = subprocess.run([binp, "--surface", str(src), "-o", str(out)],
                           capture_output=True, text=True, timeout=pol.selfcross_timeout_s,
                           env=pol.selfcross_env)
        info["wall_s"] = round(time.time() - t0, 3)
        if r.returncode != 0 or not out.exists():
            info["error"] = ((r.stderr or r.stdout) or "")[-200:]
            return None, info
        rep = json.loads(out.read_text())
    except (OSError, subprocess.TimeoutExpired, ValueError) as e:
        info["error"] = f"{type(e).__name__}: {e}"
        return None, info
    finally:
        try:
            out.unlink()
        except OSError:
            pass
    census = rep.get("census") or []
    tri = census[0].get("triangles") if census else 0
    if not census or not tri:
        info["ran"] = True
        info["density"] = 0.0
        info["triangles"] = int(tri or 0)
        return np.zeros(shape, bool), info
    if tri > pol.selfcross_max_triangles:
        info["skipped"] = f"triangles {tri} > selfcross_max_triangles {pol.selfcross_max_triangles}"
        return None, info
    trans = [c.get("transverse", 0) for c in census]
    info["ran"] = True
    info["triangles"] = int(tri)
    info["density"] = round((sum(trans) / len(trans)) / tri, 4)
    mask = np.zeros(shape, bool)
    h, w = shape
    for c in census:
        for hit in (c.get("transverse_contacts") or []):
            for q in (hit.get("quad1"), hit.get("quad2")):
                if not q:
                    continue
                r0, c0 = int(q[0]), int(q[1])
                for dr in (0, 1):
                    for dc in (0, 1):
                        rr, cc = r0 + dr, c0 + dc
                        if 0 <= rr < h and 0 <= cc < w:
                            mask[rr, cc] = True
    return mask, info


def new_cell_mask(V: np.ndarray, prev_V: np.ndarray | None) -> np.ndarray:
    """Cells valid NOW that were not valid at the end of the previous guard round. `prev_V` is
    None on the first round, or when the tracer's checkpoint grid changed shape between rounds
    (it can grow when the bounding array is extended) -- both cases fall back to "every valid
    cell is new", which is exactly the existing full-lattice behaviour, never a false negative."""
    if prev_V is None or prev_V.shape != V.shape:
        return V.copy()
    return V & ~prev_V


def selfcross_check_incremental(src: str, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                                V: np.ndarray, new_mask: np.ndarray,
                                pol: GuardPolicy) -> tuple[np.ndarray | None, dict]:
    """(3), coordinator 2026-09-29: replaces the `selfcross_max_triangles` SKIP with an
    incremental check, so a large segment is never silently left unverified. Crops `src` to this
    round's NEW cells (`new_mask`) plus every OLD cell within `selfcross_crop_halo_vox` of their
    axis-aligned bounding box, writes that as a TEMPORARY tifxyz (invalid cells set to -1; `src`
    itself is never touched -- CLAUDE.md), and runs the real `selfcross_check` on the crop
    instead of the whole lattice.

    Why an axis-aligned box, not a per-point radius: a bounding box expanded by the halo is a
    conservative SUPERSET of "every old cell within halo of some new cell" -- it can only keep
    MORE old cells than strictly needed (extra cost, never a missed contact), so it cannot hide a
    real self-intersection the full-lattice check would have found. When growth is scattered
    (the box would barely shrink anything) or the new region is not genuinely local, this falls
    back to the full-lattice `selfcross_check` -- the incremental path is a cost optimisation,
    never the sole means of ever checking a cell.

    Returns exactly `selfcross_check`'s (mask, info) contract, `info` additionally carrying
    `incremental`, `crop_cells`, `full_cells` when the crop path was actually used."""
    if not pol.selfcross:
        return None, {"skipped": "policy_off"}
    if not new_mask.any() or not V.any():
        return selfcross_check(src, X.shape, pol)
    P = np.stack([X, Y, Z], axis=-1).astype(np.float64)
    new_pts = P[new_mask & V]
    if len(new_pts) == 0:
        return selfcross_check(src, X.shape, pol)
    halo = pol.selfcross_crop_halo_vox
    lo = new_pts.min(axis=0) - halo
    hi = new_pts.max(axis=0) + halo
    keep = V & np.all((P >= lo) & (P <= hi), axis=-1)
    n_keep, n_full = int(keep.sum()), int(V.sum())
    if n_keep >= n_full * 0.9:          # the crop barely shrank anything -- just run the real thing
        return selfcross_check(src, X.shape, pol)
    import tempfile

    import tifffile
    with tempfile.TemporaryDirectory(prefix="selfx_incr_") as td:
        Xc, Yc, Zc = X.copy(), Y.copy(), Z.copy()
        Xc[~keep] = -1.0
        Yc[~keep] = -1.0
        Zc[~keep] = -1.0
        tifffile.imwrite(os.path.join(td, "x.tif"), Xc)
        tifffile.imwrite(os.path.join(td, "y.tif"), Yc)
        tifffile.imwrite(os.path.join(td, "z.tif"), Zc)
        # start from SRC's own meta.json (whatever fields the binary needs -- grid_offset,
        # vc_gsfs_params, etc -- travel with it unchanged) and only refresh bbox/area, exactly
        # like write_cropped() does for a real guard crop; a hand-built minimal meta.json was
        # missing a field the binary requires and failed to load ("type must be number, but is
        # null") -- reusing the real one is the same fix write_cropped already relies on.
        meta = {}
        srcmeta = Path(src) / "meta.json"
        if srcmeta.exists():
            try:
                meta = json.loads(srcmeta.read_text())
            except (ValueError, OSError):
                meta = {}
        Pc = np.stack([Xc, Yc, Zc], axis=-1).astype(np.float64)
        if keep.any():
            meta["bbox"] = [Pc[keep].min(axis=0).tolist(), Pc[keep].max(axis=0).tolist()]
        meta["uuid"] = "selfx_incremental_crop"
        meta["source"] = "growth_guard.selfcross_check_incremental"
        Path(td, "meta.json").write_text(json.dumps(meta))
        mask, info = selfcross_check(td, X.shape, pol)
    info["incremental"] = True
    info["crop_cells"] = n_keep
    info["full_cells"] = n_full
    return mask, info


def guard_tifxyz(src: str, dst: str, pol: GuardPolicy, voxel_um: float, sampler=None, cover=None,
                 own: str | None = None, shadow_ctx: ShadowContext | None = None,
                 new_mask: np.ndarray | None = None) -> GuardResult:
    """Evaluate `src` and write the cropped surface to `dst` (a new tifxyz dir: x/y/z -- and
    generations.tif when the tracer wrote one -- with -1 / 0 where pruned, meta.json with updated
    bbox/area, guard.json with the numbers). `src` is never modified (CLAUDE.md: never modify an
    existing meta.json). The kept piece is the one holding the tracer's START (the smallest
    generation) when that survives, else the largest.

    `new_mask` (3), 2026-09-29: this round's new cells, from `guard_round`'s `GuardState.prev_V`
    bookkeeping. When given, the selfcross check runs `selfcross_check_incremental` (crops to the
    new region + a halo, falling back to the full lattice on its own when that would not help);
    when None (self_test's direct calls, or no prior state), it runs the full-lattice
    `selfcross_check` exactly as before -- never a regression for a caller that does not track
    round-to-round state."""
    X, Y, Z = _read_xyz(src)
    gen = _read_gen(src)
    anchor = None
    V = (X > 0) & (Y > 0) & (Z > 0)
    if gen is not None and gen.shape == X.shape:
        if V.any():
            anchor = np.unravel_index(int(np.argmin(np.where(V, gen, np.iinfo(gen.dtype).max))), V.shape)
    if not pol.selfcross:
        selfx_mask, selfx_info = None, {}
    elif new_mask is not None:
        selfx_mask, selfx_info = selfcross_check_incremental(src, X, Y, Z, V, new_mask, pol)
    else:
        selfx_mask, selfx_info = selfcross_check(src, X.shape, pol)
    if pol.selfcross and not selfx_info.get("ran"):
        # (2) selfcross is ENABLED but the pre-round check could not run at all (binary missing,
        # subprocess error, timeout) -- coordinator, 2026-09-29: this is not a skip, it is a
        # guard that has gone dark while believed to be protecting production. Force the
        # self-test cache to FAILED (agent.run_grow refuses this host's next grow claim) and
        # surface the reason as its own metric (stages/grow.py records `guard_selfx_error`).
        reason = selfx_info.get("error") or selfx_info.get("skipped") or "unknown"
        mark_broken(f"selfcross enabled but pre-round check failed: {reason}")
        selfx_info["guard_selfx_error"] = f"pre: {reason}"
    P, _ = lattice_frame(X, Y, Z)
    shadow = shadow_round(P, V, pol, voxel_um, shadow_ctx)
    res = evaluate(X, Y, Z, pol, voxel_um, sampler=sampler, cover=cover, own=own, anchor=anchor,
                  selfx_mask=selfx_mask, shadow=shadow)
    if selfx_info:
        res.extra["selfcross_pre"] = selfx_info
        if "guard_selfx_error" in selfx_info:
            res.extra["guard_selfx_error"] = selfx_info["guard_selfx_error"]
    if shadow:
        # masks are large arrays and never belong in a JSON summary; keep everything else,
        # nested (for debugging/completeness)...
        res.extra["shadow"] = {k: {kk: vv for kk, vv in v.items() if kk != "mask"} for k, v in shadow.items()}
        # ...AND flattened to the TOP LEVEL under the exact names guard_review.py's
        # EXTRA_THRESHOLD_CRITERIA / growth_ab.py's SUMMARY_METRICS look for directly in
        # guard_summary (its primary lookup path; stages/grow.py also records these as their own
        # `metric` rows, guard_review's documented fallback path -- both are kept in sync here).
        for name, key in (("stretch", "stretch_p90"), ("normal_dev", "normal_dev_p90_deg"),
                          ("seam", "seam_steps"), ("ridge_hit", "ridge_hit"),
                          ("empty_space", "empty_space")):
            v = shadow.get(name, {}).get(key)
            if v is not None:
                res.extra[key] = v
        if "roughness" in shadow and shadow["roughness"].get("value") is not None:
            res.extra["frontier_roughness"] = shadow["roughness"]["value"]
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
    # (3), 2026-09-29: the valid-cell mask as of the END of the last guard_round call, for
    # `selfcross_check_incremental`'s "new cells this round" -- None on the first round (every
    # valid cell counts as new then, same as a full-lattice check).
    prev_V: np.ndarray | None = None


def guard_round(cur: str, pol: GuardPolicy, voxel_um: float, state: GuardState, sampler=None, cover=None,
                own: str | None = None, tag: str = "g", neighbour_ok=None,
                shadow_ctx: ShadowContext | None = None) -> tuple[str, dict]:
    """One guard step on the checkpoint a round just produced.

    Returns (checkpoint to resume from, info). `info["stop"]` is None, "frontier_blocked" (the round
    mostly re-grew into ground we had already pruned) or "nothing_left". When nothing is cut the
    checkpoint is returned unchanged and nothing is written."""
    X, Y, Z = _read_xyz(cur)
    P, V = lattice_frame(X, Y, Z)
    # (3), 2026-09-29: this round's new cells, relative to the valid mask as of the end of the
    # LAST guard_round call -- everything on round 1 (state.prev_V is None), same as before.
    nm = new_cell_mask(V, state.prev_V)
    info = {"regrown_frac": 0.0, "stop": None}
    if state.pruned_xyz is not None and len(state.pruned_xyz) and V.any():
        tol = pol.regrow_tol_cells * max(1.0, median_edge(P, V))
        info["regrown_frac"] = round(regrown_fraction(state.keep_xyz if state.keep_xyz is not None else np.zeros((0, 3)),
                                                      state.pruned_xyz, P[V], tol), 4)
    dst = str(Path(cur).parent / f"guarded_{tag}_{Path(cur).name}")
    ready = None
    if pol.merge_pause and cover is not None and neighbour_ok is not None:
        ready = merge_readiness(X, Y, Z, pol, cover, neighbour_ok, voxel_um, own)
        info["merge"] = ready
    # a mergeable frontier keeps its overlap (the merger registers on it); everything else is still cut
    res = guard_tifxyz(cur, dst, pol, voxel_um, sampler=sampler,
                       cover=None if (ready and ready["mergeable"]) else cover, own=own,
                       shadow_ctx=shadow_ctx, new_mask=nm)
    info.update(res.summary())
    # SHADOW segment-level stops (roughness / flatten_feedback): measured every round regardless
    # (info["shadow"] always carries them when their compute flag is on), but only ALLOWED to end
    # the segment when their own _enforce flag is set -- default False for both.
    sh = info.get("shadow") or {}
    if pol.roughness_enforce and (sh.get("roughness") or {}).get("would_stop") and info["stop"] is None:
        info["stop"] = "roughness"
    if pol.flatten_feedback_enforce and (sh.get("flatten_feedback") or {}).get("would_stop") and info["stop"] is None:
        info["stop"] = "flatten_feedback"
    if ready and ready["mergeable"]:
        info["stop"] = "mergeable"
    cut = V & ~res.keep
    if info["regrown_frac"] >= pol.regrow_block_frac and state.rounds_cut > 0 and info["stop"] is None:
        info["stop"] = "frontier_blocked"
    if res.stop:
        info["stop"] = info["stop"] or res.stop
    # ARM B verification + early-abort (growth_degeneracy_2026-09-28.md), on `dst` -- the surface
    # that will actually be resumed from if this round's cut survives. `dst` always exists here
    # (guard_tifxyz/write_cropped wrote it unconditionally); the "nothing cut" cleanup below runs
    # AFTER this block. Target is EXACTLY ZERO self-intersections: any density > 0 after our own
    # pruning means the flagged quads' corners did not cover every crossing, and the segment stops
    # rather than shipping a still-degenerate surface (this is the failure mode measured directly
    # on PHerc0191_cea9032 -- fold/plan cropping cut density only 6-28%, never to 0).
    if pol.selfcross and info["stop"] is None:
        if cut.any():
            Xd2, Yd2, Zd2 = _read_xyz(dst)
            Vd2 = (Xd2 > 0) & (Yd2 > 0) & (Zd2 > 0)
            _, post = selfcross_check_incremental(dst, Xd2, Yd2, Zd2, Vd2, nm & Vd2, pol)
        else:
            post = res.extra.get("selfcross_pre", {})   # dst == src, no need to re-run
        if post.get("ran"):
            info["selfcross_post"] = post
            if (post.get("density") or 0.0) > pol.selfcross_density_stop:
                info["stop"] = "selfcross_nonzero"
        else:
            # (2) same as the pre-round case above: enabled but could not verify the crop. A
            # round that cannot be shown clean must not be silently allowed to look clean.
            reason = post.get("error") or post.get("skipped") or "unknown"
            mark_broken(f"selfcross enabled but post-round check failed: {reason}")
            info["guard_selfx_error"] = f"post: {reason}"
    if pol.selfcross_hairpin_abort_ratio > 0 and info["stop"] is None:
        try:
            from . import geometry as GM
            Xd, Yd, Zd = _read_xyz(dst)
            Pd, Vd = lattice_frame(Xd, Yd, Zd)
            fm = GM.fold_metrics(Pd, Vd, voxel_um)
            lr = int(fm.get("geo_lines_read") or 0)
            if lr >= pol.selfcross_hairpin_min_lines:
                ratio = float(fm.get("geo_hairpin_lines") or 0) / lr
                info["hairpin_ratio"] = round(ratio, 4)
                if ratio >= pol.selfcross_hairpin_abort_ratio:
                    info["stop"] = "hairpin_abort"
        except (OSError, ValueError, KeyError) as e:
            info["hairpin_error"] = f"{type(e).__name__}: {e}"
    if not cut.any():
        shutil.rmtree(dst, ignore_errors=True)
        state.keep_xyz = P[V]
        state.prev_V = V
        return cur, info
    state.rounds_cut += 1
    state.pruned_xyz = P[cut] if state.pruned_xyz is None else np.concatenate([state.pruned_xyz, P[cut]])
    state.keep_xyz = P[res.keep]
    state.prev_V = res.keep
    return dst, info


# ----------------------------------------------------------------------------- lattice bbox
def lattice_bbox(ckpt: str):
    X, Y, Z = _read_xyz(ckpt)
    P, V = lattice_frame(X, Y, Z)
    return (P[V].min(axis=0), P[V].max(axis=0)) if V.any() else None


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


# ----------------------------------------------------------------------------- self-test and loud failure
# A planted defect per criterion, run before growth. A guard that cannot run raises GuardBroken.


class GuardBroken(RuntimeError):
    """The guard was asked to verify a surface and could not (binary missing, failed to load,
    timed out, crop too large). A guard that cannot run must not look like a clean pass, so this
    is raised, never swallowed: the driver stops the seed and reports it."""


def mark_broken(reason: str) -> None:
    raise GuardBroken(reason)


def self_test(vc_bin: str | None = None, selfcross_required: bool = False,
              env: dict | None = None) -> tuple[bool, str]:
    """Plant one synthetic defect per criterion and require that each fires on its own defect and
    leaves a clean region of the same lattice alone. When `vc_tifxyz_selfcross` resolves, run it
    once through `env` (a binary that resolves but cannot load its libraries must fail here, not
    in round 1). `guarded-grow` runs this before the first round. Returns (ok, detail)."""
    t0 = time.time()
    try:
        ok, detail = _run_self_test(vc_bin, selfcross_required, env)
    except Exception as e:                       # noqa: BLE001 - a self-test that raises FAILED
        ok, detail = False, f"self-test raised {type(e).__name__}: {e}"
    return ok, f"{detail} ({(time.time() - t0) * 1000:.0f} ms)"


def _plane_xyz(h=30, w=30, pitch=20.0, z=5000.0, x0=1000.0, y0=1000.0):
    j, i = np.meshgrid(np.arange(w), np.arange(h))
    X = (x0 + j * pitch).astype(np.float32)
    Y = (y0 + i * pitch).astype(np.float32)
    Z = np.full_like(X, z)
    return X, Y, Z


def _selfx_fixture_check(fixture: Path, binp: str, env: dict | None = None) -> tuple[np.ndarray | None, dict]:
    """Run the real `vc_tifxyz_selfcross` (at `binp`) against the packed, known-degenerate
    cea9032 round-1 fixture. Factored out of `_run_self_test` so both the required and the
    optional selfx branches share exactly one call."""
    import tempfile

    import tifffile
    with tempfile.TemporaryDirectory() as td, np.load(fixture) as z:
        for a in "xyz":
            tifffile.imwrite(os.path.join(td, f"{a}.tif"), z[a])
        tifffile.imwrite(os.path.join(td, "generations.tif"), z["generations"])
        open(os.path.join(td, "meta.json"), "w").write(str(z["meta_json"].item()))
        return selfcross_check(td, z["x"].shape,
                               GuardPolicy(selfcross=True, selfcross_bin=binp, selfcross_env=env))


def _selfx_smoke_check(binp: str, env: dict | None = None) -> dict:
    """(Production incident, 2026-09-29.) Run the real binary ONCE, on a trivial synthetic clean
    plane -- NOT the packed fixture, which is a source-tree-only path
    (`tests/fixtures/growth_guard_selfcross_cea9032_r1.npz`) that does not ship in a production
    deploy. That is exactly why the OLD self-test's optional branch (`elif binp and ... and
    fixture.exists()`) silently never ran on any real phi host: `fixture.exists()` was always
    False there, so the check was SKIPPED, not merely non-fatal -- self-test reported clean while
    the exact same missing-`LD_LIBRARY_PATH` bug it should have caught fired on every real
    selfcross call minutes later. This function exists so `_run_self_test` can require "the
    binary actually executes through this environment" UNCONDITIONALLY, independent of whether
    the fixture happens to be present. Returns just the `info` dict (`ran` is the load-bearing
    key); the mask is irrelevant here (a clean plane has none)."""
    import tempfile

    import tifffile
    X, Y, Z = _plane_xyz(5, 5)
    with tempfile.TemporaryDirectory() as td:
        for a, A in zip("xyz", (X, Y, Z), strict=True):
            tifffile.imwrite(os.path.join(td, f"{a}.tif"), A)
        Path(td, "meta.json").write_text(json.dumps({
            "area_cm2": 0.0, "max_gen": 0, "scale": [0.05, 0.05], "format": "tifxyz",
            "type": "seg", "uuid": "selfx_smoke", "source": "growth_guard._selfx_smoke_check"}))
        _, info = selfcross_check(td, X.shape, GuardPolicy(selfcross=True, selfcross_bin=binp, selfcross_env=env))
        return info


def _run_self_test(vc_bin: str | None, selfcross_required: bool = False,
                   env: dict | None = None) -> tuple[bool, str]:
    problems: list[str] = []
    checked: list[str] = []
    VOX = 9.362

    # ---- 1. vacuum: right third of the lattice is air ----
    X, Y, Z = _plane_xyz(30, 30)
    P, Vv = lattice_frame(X, Y, Z)
    air = lambda p: np.where(np.asarray(p)[:, 0] > 1000 + 20 * 20, 0.0, 100.0)
    m = vacuum_mask(P, Vv, air, GuardPolicy())
    checked.append("vacuum")
    if not m[:, 22:].any():
        problems.append("vacuum: did not fire on the planted air region")
    if m[:, :15].any():
        problems.append("vacuum: fired on the clean (material) region")

    # ---- 2. fold/hairpin: a 180-degree turn, exact construction as tests/test_growth_guard.py ----
    h, w, bend = 60, 100, 60
    j = np.arange(w)
    fold = np.where(j < bend, j, bend - (j - bend))
    X2 = (1000.0 + fold[None, :] * 20.0 + np.zeros((h, 1))).astype(np.float32)
    Y2 = _plane_xyz(h, w)[1]
    Z2 = (5000.0 + np.where(j < bend, 0.0, 6.0)[None, :] + np.zeros((h, 1))).astype(np.float32)
    P2, V2 = lattice_frame(X2, Y2, Z2)
    m = fold_mask(P2, V2, VOX, GuardPolicy(fold_radius_um=500.0))
    checked.append("fold")
    if not m[:, bend - 3:bend + 8].any():
        problems.append("fold: did not fire near the planted hairpin")
    if m[:, :40].any():
        problems.append("fold: fired on the clean body")

    # ---- 3. plan/crumple: gaussian noise on the right third ----
    X3, Y3, Z3 = _plane_xyz(80, 80)
    rng = np.random.default_rng(3)
    Z3 = Z3.copy()
    Z3[:, 62:] += rng.normal(0, 80.0, size=Z3[:, 62:].shape).astype(np.float32)
    P3, V3 = lattice_frame(X3, Y3, Z3)
    m = plan_mask(P3, V3, GuardPolicy())
    checked.append("plan")
    if not m[:, 62:].any():
        problems.append("plan: did not fire on the planted crumple")
    if m[:, :50].any():
        problems.append("plan: fired on the clean region")

    # ---- 4. overlap: a neighbour covers the right part of the lattice ----
    X4, Y4, Z4 = _plane_xyz(70, 90)
    Xn, Yn, Zn = _plane_xyz(70, 40, x0=1000.0 + 60 * 20.0)
    idx = SegmentIndex(vox=4.0, deg=20.0)
    idx.add("neighbour", Xn, Yn, Zn)
    P4, V4 = lattice_frame(X4, Y4, Z4)
    m = overlap_mask(P4, V4, idx, GuardPolicy())
    checked.append("overlap")
    if not m[:, 65:].any():
        problems.append("overlap: did not fire on the covered region")
    if m[:, :50].any():
        problems.append("overlap: fired on the uncovered region")

    # ---- 5. quad_flip: reuse the fold construction (a real orientation reversal at the hinge) ----
    m = quad_flip_mask(P2, V2, GuardPolicy())
    checked.append("quad_flip")
    if m[:, :40].any() or m[:, 70:].any():
        problems.append("quad_flip: fired away from the hinge transition")
    # (a flip firing INSIDE the transition band is not asserted: the exact band width depends on
    # plan_win and is not this test's job to pin down -- only that it stays off the clean wings)

    # ---- 6. stretch: one column pulled far away ----
    X6, Y6, Z6 = _plane_xyz(30, 30)
    X6 = X6.copy()
    X6[:, 20] += 500.0
    P6, V6 = lattice_frame(X6, Y6, Z6)
    m = stretch_mask(P6, V6, GuardPolicy())
    checked.append("stretch")
    if not m[:, 19:22].any():
        problems.append("stretch: did not fire near the planted stretch")
    if m[:, :15].any():
        problems.append("stretch: fired on the clean region")

    # ---- 7. wrap_spacing / curvature: umbilicus-relative checks, fake umbilicus_of_z ----
    def umb0(z):
        return np.zeros_like(z), np.zeros_like(z)
    X7, Y7, Z7 = _plane_xyz(30, 30)
    X7 = X7.copy()
    X7[:, 15:] += 2000.0            # a radial jump at column 15
    P7, V7 = lattice_frame(X7, Y7, Z7)
    m = wrap_spacing_mask(P7, V7, umb0, VOX, GuardPolicy())
    checked.append("wrap_spacing")
    if m is None or not m[:, 13:17].any():
        problems.append("wrap_spacing: did not fire on the planted radial jump")
    if m is not None and m[:, :10].any():
        problems.append("wrap_spacing: fired on the clean region")
    m = curvature_mask(P2, V2, VOX, umb0, GuardPolicy(curvature_frac_th=0.9))
    checked.append("curvature")
    if m is None or not m.any():
        problems.append("curvature: did not fire on the planted tight hairpin at a generous threshold")

    # ---- 8. ridge_hit / seam: fake prediction sampler ----
    X8, Y8, Z8 = _plane_xyz(30, 30)
    P8, V8 = lattice_frame(X8, Y8, Z8)
    pred_never = lambda pts: np.zeros(len(pts))
    m = ridge_hit_mask(P8, V8, pred_never, GuardPolicy())
    checked.append("ridge_hit")
    if m is None or not m.all():
        problems.append("ridge_hit: did not fire everywhere against a prediction that never matches")
    pred_shift = lambda pts: (np.abs(np.asarray(pts)[:, 2] - np.where(
        np.asarray(pts)[:, 0] > 1000 + 15 * 20.0, 5020.0, 5000.0)) < 0.6).astype(float)
    m = seam_mask(P8, V8, pred_shift, GuardPolicy())
    checked.append("seam")
    if m is None or not m[:, 13:17].any():
        problems.append("seam: did not fire on the planted coordinate jump")
    if m is not None and m[:, :10].any():
        problems.append("seam: fired on the clean region")

    # ---- 8b. empty_space: same never-supported sampler, but must fire ONLY on the frontier ----
    m = empty_space_mask(P8, V8, pred_never, GuardPolicy())
    checked.append("empty_space")
    if m is None or not m.any():
        problems.append("empty_space: did not fire against a prediction that never matches")
    band = frontier_band(V8, GuardPolicy().reach_rings)
    if m is not None and not np.array_equal(m, m & band):
        problems.append("empty_space: fired outside the frontier band it is defined over")

    # ---- 9. roughness / flatten_feedback: segment-level, sanity only (no external threshold to plant against yet) ----
    info = frontier_roughness(V8, GuardPolicy(), VOX)
    checked.append("roughness")
    if info.get("value") is None:
        problems.append("roughness: produced no value on a valid lattice")

    # ---- 10. selfx ----
    # (production incident, 2026-09-29): the SMOKE check below is UNCONDITIONAL whenever the
    # binary resolves AT ALL -- never gated on selfcross_required, never on the packed test
    # fixture. That gate is exactly what let self-test pass on every phi host while production
    # selfcross calls failed to load their shared library minutes later: `fixture.exists()` was
    # always False in a real deploy (tests/fixtures/ is source-tree-only), so the OLD optional
    # branch below never even ran there. "Make self_test ALWAYS execute the real binary once
    # through the SAME env path production uses, whenever the binary is configured."
    binp = vc_bin or shutil.which("vc_tifxyz_selfcross")
    fixture = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "growth_guard_selfcross_cea9032_r1.npz"
    if selfcross_required and not (binp and os.path.exists(binp)):
        checked.append("selfx")
        problems.append(f"selfx: selfcross is ENABLED in production but no binary resolved "
                        f"(vc_bin={vc_bin!r}, PATH which={shutil.which('vc_tifxyz_selfcross')!r})")
    elif binp and os.path.exists(binp):
        checked.append("selfx")
        smoke_info = _selfx_smoke_check(binp, env)
        if not smoke_info.get("ran"):
            problems.append(f"selfx: binary resolved but could not actually run through this "
                            f"process's environment (env carries LD_LIBRARY_PATH? {bool(env)}) -- {smoke_info}")
        # bonus accuracy check against the known-degenerate fixture, when it happens to ship
        # (dev/CI boxes only -- never required for self-test to pass, so selfcross can still be
        # enabled in a real deploy that never carries tests/fixtures/)
        if fixture.exists():
            sc_mask, sc_info = _selfx_fixture_check(fixture, binp, env)
            if not sc_info.get("ran") or not (sc_info.get("density") or 0) > 0:
                problems.append(f"selfx: real binary did not detect the known-degenerate fixture ({sc_info})")
            if sc_mask is None or not sc_mask.any():
                problems.append("selfx: mask empty on the known-degenerate fixture")

    ok = not problems
    detail = ("all clear: " if ok else "FAILED: ") + ", ".join(problems) if problems else f"{len(checked)} criteria checked: {', '.join(checked)}"
    return ok, detail
