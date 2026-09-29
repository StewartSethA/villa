"""growth_guard._ridge_offset vectorisation (2026-09-29, coordinator item 1): the original
per-point Python loop measured 39.6-74.2 s/segment on real fleet data
(growth_degeneracy_2026-09-28.md's shadow-criteria replay); this checks the vectorised
`_ridge_offset` against the kept-as-reference `_ridge_offset_slow` for IDENTICAL outputs on
edge cases and randomised data, plus a real-segment speedup measurement gated on real
prediction data being available (skips, does not fail, when it is not).
"""
import os
import time

import numpy as np
import pytest

from tifxyz_tools import growth_guard as G

pytestmark = pytest.mark.unit


def test_edge_cases_all_false_all_true_single_ties_alternating():
    """Every case that could break a run-detection rewrite: no run, the whole row is one run,
    a single-cell run, a run touching only one side, a genuine TIE between two equidistant
    runs (must break the tie the SAME way as the reference), and a maximally fragmented row."""
    W = 7  # offsets -3..3, step 1
    cases = [
        [0, 0, 0, 0, 0, 0, 0],   # all false -> NaN
        [1, 1, 1, 1, 1, 1, 1],   # all true -> centre = middle = offset 0
        [0, 0, 0, 1, 0, 0, 0],   # single true exactly at offset 0
        [1, 1, 0, 0, 0, 0, 0],   # run only on the left
        [0, 0, 0, 0, 0, 1, 1],   # run only on the right
        [1, 1, 0, 0, 0, 1, 1],   # TIE: two runs equidistant from offset 0
        [1, 0, 1, 0, 1, 0, 1],   # alternating -- many single-cell runs
    ]
    vals2d = np.array(cases, dtype=bool)
    N = len(cases)
    pts = np.zeros((N, 3))
    pts[:, 0] = np.arange(N)
    nv = np.zeros((N, 3))
    nv[:, 1] = 1.0

    def pred(p):
        rows = np.round(p[:, 0]).astype(int)
        cols = np.clip(np.round(p[:, 1]).astype(int) + 3, 0, W - 1)
        return vals2d[rows, cols].astype(float)

    slow = G._ridge_offset_slow(pts, nv, pred, 3.0, 1.0)
    fast = G._ridge_offset(pts, nv, pred, 3.0, 1.0)
    assert np.array_equal(np.isnan(slow), np.isnan(fast))
    ok = np.isnan(slow) | np.isclose(slow, fast)
    assert np.all(ok), list(zip(cases, slow, fast, strict=True))


def test_identical_on_randomised_data():
    rng = np.random.default_rng(0)
    N = 3000
    pts = rng.uniform(0, 1000, size=(N, 3))
    nv = rng.normal(size=(N, 3))
    nv /= np.linalg.norm(nv, axis=1, keepdims=True)

    def fake_pred(p):
        h = (np.round(p[:, 0] * 3).astype(np.int64) * 911
            + np.round(p[:, 1] * 7).astype(np.int64) * 7919
            + np.round(p[:, 2] * 13).astype(np.int64) * 104729)
        return (np.abs(np.sin(h.astype(np.float64))) > 0.7).astype(float)

    slow = G._ridge_offset_slow(pts, nv, fake_pred, 10.0, 1.0)
    fast = G._ridge_offset(pts, nv, fake_pred, 10.0, 1.0)
    close = np.isclose(slow, fast, equal_nan=True)
    assert np.all(close), f"{np.sum(~close)}/{N} mismatches"


def test_empty_input():
    empty = np.zeros((0, 3))
    r = G._ridge_offset(empty, empty, lambda p: np.zeros(len(p)), 3.0)
    assert len(r) == 0


def test_no_prediction_anywhere_is_all_nan():
    pts = np.zeros((10, 3))
    nv = np.zeros((10, 3))
    nv[:, 2] = 1.0
    r = G._ridge_offset(pts, nv, lambda p: np.zeros(len(p)), 3.0)
    assert np.all(np.isnan(r))


# ---------------------------------------------------------------------- real-data speedup
SAMPLE_DIR = os.environ.get(
    "SHADOW_REPLAY_SAMPLE_DIR",
    "/tmp/claude-1000/-home-seth-ScrollPrizeTutorial/2009ddb2-9fa9-4865-b0ef-fcf612867acf/scratchpad/sample2")
REAL_SEG = "0_PHerc0191_c007bac"


@pytest.mark.skipif(not os.path.isdir(os.path.join(SAMPLE_DIR, REAL_SEG)),
                    reason="real segment sample not present on this box")
def test_speedup_on_a_real_segment_using_a_synthetic_stand_in_sampler():
    """Uses a REAL degenerate lattice (same one growth_degeneracy_2026-09-28.md measured) but
    a synthetic (fast, deterministic) prediction sampler, so this test does not need the real
    9+ GB prediction zarr to run in CI -- it isolates the vectorisation's effect on the
    POST-PROCESSING step specifically. The real-zarr end-to-end number (sampling cost included)
    is reported separately in FINDINGS.md's 2026-09-29 entry, measured directly against the
    real store, since sampling cost cannot be faked honestly."""
    import tifffile
    d = os.path.join(SAMPLE_DIR, REAL_SEG)
    X = tifffile.imread(os.path.join(d, "x.tif"))
    Y = tifffile.imread(os.path.join(d, "y.tif"))
    Z = tifffile.imread(os.path.join(d, "z.tif"))
    P, V = G.lattice_frame(X, Y, Z)
    n, ok = G.grid_normals(P, V)
    idx = np.nonzero(ok)
    pts, nv = P[idx], n[idx]

    def fake_pred(p):
        h = (np.round(p[:, 0]).astype(np.int64) * 911
            + np.round(p[:, 1]).astype(np.int64) * 7919
            + np.round(p[:, 2]).astype(np.int64) * 104729)
        return (np.abs(np.sin(h.astype(np.float64))) > 0.6).astype(float)

    t0 = time.time()
    slow = G._ridge_offset_slow(pts, nv, fake_pred, 30.0)
    t_slow = time.time() - t0
    t0 = time.time()
    fast = G._ridge_offset(pts, nv, fake_pred, 30.0)
    t_fast = time.time() - t0

    close = np.isclose(slow, fast, equal_nan=True)
    assert np.all(close), f"{np.sum(~close)}/{len(pts)} mismatches on {len(pts)} real points"
    assert t_fast < t_slow, f"vectorised ({t_fast:.3f}s) must be faster than the reference ({t_slow:.3f}s)"


def test_ridge_hit_and_seam_masks_matches_the_separate_standalone_calls():
    """Item (c), 2026-09-29: shares ONE wide pred_sampler sample between ridge_hit and seam.
    Must produce IDENTICAL masks to calling ridge_hit_mask/seam_mask separately."""
    PITCH = 20.0

    def plane(h=30, w=30, z=5000.0):
        j, i = np.meshgrid(np.arange(w), np.arange(h))
        X = (1000.0 + j * PITCH).astype(np.float32)
        Y = (1000.0 + i * PITCH).astype(np.float32)
        Z = np.full_like(X, z)
        return X, Y, Z

    X, Y, Z = plane()
    P, V = G.lattice_frame(X, Y, Z)
    pol = G.GuardPolicy()

    def pred_shift(pts):
        pts = np.asarray(pts)
        shifted_target = np.where(pts[:, 0] > 1000 + 15 * PITCH, 5020.0, 5000.0)
        return (np.abs(pts[:, 2] - shifted_target) < 0.6).astype(float)

    sep_ridge = G.ridge_hit_mask(P, V, pred_shift, pol)
    sep_seam = G.seam_mask(P, V, pred_shift, pol)
    comb_ridge, comb_seam = G.ridge_hit_and_seam_masks(P, V, pred_shift, pol)
    assert np.array_equal(sep_ridge, comb_ridge)
    assert np.array_equal(sep_seam, comb_seam)
    assert comb_ridge.sum() > 0 and comb_seam.sum() > 0, "setup: both criteria must actually fire here"


def test_ridge_hit_and_seam_masks_on_the_real_degenerate_fixture():
    with np.load(str(__import__("pathlib").Path(__file__).parent / "fixtures" /
                    "growth_guard_selfcross_cea9032_r1.npz")) as z:
        X, Y, Z = z["x"], z["y"], z["z"]
    P, V = G.lattice_frame(X, Y, Z)
    pol = G.GuardPolicy()
    pred_never = lambda pts: np.zeros(len(pts))  # noqa: E731
    sep_ridge = G.ridge_hit_mask(P, V, pred_never, pol)
    sep_seam = G.seam_mask(P, V, pred_never, pol)
    comb_ridge, comb_seam = G.ridge_hit_and_seam_masks(P, V, pred_never, pol)
    assert np.array_equal(sep_ridge, comb_ridge)
    assert np.array_equal(sep_seam, comb_seam)


def test_ridge_hit_and_seam_masks_none_sampler_returns_none_none():
    X, Y, Z = np.zeros((5, 5)), np.zeros((5, 5)), np.zeros((5, 5))
    P, V = G.lattice_frame(X, Y, Z)
    r, s = G.ridge_hit_and_seam_masks(P, V, None, G.GuardPolicy())
    assert r is None and s is None
