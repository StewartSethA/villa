"""Synthetic tests for umbilicus_raycross: a known spiral must give back its centre.

Everything runs on CPU on small synthetic slices. The claims under test are the
ones the tool relies on, each checked so it can fail:

* the crossing count is maximal at the true centre (the criterion itself),
* the two-stage and single-stage searches agree,
* the estimate follows a moving centre through a whole build,
* the written file loads through the fitter's own umbilicus reader,
* a machine estimate can never carry the hand-placed score of 100.
"""

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

torch = pytest.importorskip("torch")
scipy = pytest.importorskip("scipy")

import umbilicus_raycross as ur  # noqa: E402
from umbilicus import json_umbilicus_z_to_yx  # noqa: E402

DEV = "cpu"
N = 160          # slice is N x N pixels
PITCH = 8.0      # pixels between successive windings
RADIUS = 68.0   # scroll body radius, pixels


def spiral_slice(cy, cx, squash=1.0, phase=0.0):
    """Archimedean spiral sheet r = PITCH * theta / 2pi, two pixels thick, inside
    a disc body. ``squash`` flattens it into an ellipse (a deformed roll)."""
    yy, xx = np.mgrid[0:N, 0:N].astype(float)
    dy, dx = (yy - cy) / squash, xx - cx
    r = np.hypot(dy, dx)
    theta = np.mod(np.arctan2(dy, dx) + phase, 2 * np.pi)
    # distance from the sheet along the radius, folded to the local pitch
    frac = np.mod(r - PITCH * theta / (2 * np.pi), PITCH)
    sheet = ((frac < 2.0) | (frac > PITCH - 0.0)) & (r > 6.0) & (r < RADIUS)
    body = r < RADIUS + 3
    return sheet, body


def test_crossing_count_peaks_at_the_true_centre():
    sheet, body = spiral_slice(80, 80)
    windings = int((RADIUS - 6) / PITCH)
    cands = np.array([[80, 80], [80, 105], [105, 80], [60, 68], [80, 40]], float)
    r, _ = ur.ray_counts(sheet, body, cands, n_rays=32, device=DEV)
    mean = r.mean(1)
    assert int(np.argmax(mean)) == 0
    # from the centre every ray crosses about every winding once
    assert windings - 3 <= mean[0] <= windings + 3
    assert mean[0] > mean[1:].max() + 0.5   # at least half a crossing better than any offset centre


@pytest.mark.parametrize("centre", [(80, 80), (66, 92), (95, 70)])
def test_slice_estimate_recovers_the_centre(centre):
    sheet, body = spiral_slice(*centre)
    r = ur.slice_stats(sheet, body, stride=3, n_rays=32, erode=4, device=DEV)
    assert r is not None
    err = np.hypot(r["y_soft"] - centre[0], r["x_soft"] - centre[1])
    assert err < 0.5 * PITCH, f"soft centre off by {err:.1f} px (pitch {PITCH})"


def test_estimate_survives_an_elliptical_deformation():
    centre = (78, 82)
    sheet, body = spiral_slice(*centre, squash=0.8)
    r = ur.slice_stats(sheet, body, stride=3, n_rays=32, erode=4, device=DEV)
    err = np.hypot(r["y_soft"] - centre[0], r["x_soft"] - centre[1])
    assert err < 1.0 * PITCH


def test_two_stage_search_agrees_with_single_stage():
    sheet, body = spiral_slice(74, 86)
    one = ur.slice_stats(sheet, body, stride=3, n_rays=32, erode=4, device=DEV)
    two = ur.slice_stats(sheet, body, stride=3, n_rays=32, erode=4, device=DEV,
                         coarse=9, half=30)
    assert np.hypot(one["y_soft"] - two["y_soft"], one["x_soft"] - two["x_soft"]) < 3.0


def test_too_little_scroll_is_refused_not_guessed():
    sheet = np.zeros((N, N), bool)
    body = np.zeros((N, N), bool)
    body[100:110, 100:110] = True
    assert ur.slice_stats(sheet, body, device=DEV) is None


def test_robust_smooth_ignores_an_outlier():
    z = np.arange(0, 400, 10, dtype=float)
    # Small noise: with an exactly-linear signal the residual MAD is 0 and the
    # IRLS loop (correctly) has no scale to reject against.
    y = 100 + 0.05 * z + np.random.default_rng(0).normal(0, 0.5, len(z))
    bad = y.copy()
    bad[20] += 300.0
    fit, w = ur.robust_smooth(z, bad, np.ones_like(z), frac=0.2)
    plain, _ = ur.robust_smooth(z, bad, np.ones_like(z), frac=0.2, iters=0)
    assert abs(fit[20] - y[20]) < 3.0             # rejected ...
    assert abs(plain[20] - y[20]) > 10.0          # ... where a plain smooth is not
    assert w[20] < 0.2 * np.median(w)


def _detections_along(path, centres_by_z):
    pts = []
    for z, (cy, cx) in centres_by_z.items():
        sheet, body = spiral_slice(cy, cx)
        r = ur.slice_stats(sheet, body, stride=3, n_rays=32, erode=4, device=DEV)
        r["z"], r["zl"] = int(z), int(z)
        pts.append(r)
    doc = {"level": 0, "scale": 1, "shape_level": [1000, N, N],
           "shape_level0": [1000, N, N], "params": {}, "pred": "synthetic",
           "ct": None, "body": "synthetic", "points": pts}
    with open(path, "w") as f:
        json.dump(doc, f)


def test_build_follows_a_moving_centre_and_round_trips(tmp_path):
    truth = {z: (72 + 0.02 * z, 90 - 0.02 * z) for z in range(0, 500, 50)}
    det = tmp_path / "det.json"
    out = tmp_path / "umbilicus.json"
    _detections_along(det, truth)
    assert ur.main(["build", str(det), "--out", str(out), "--z-step", "25"]) == 0

    doc = json.loads(out.read_text())
    pts = doc["control_points"]
    assert set(pts[0]) == {"x", "y", "z", "score"}
    assert all(isinstance(p[k], int) for p in pts for k in "xyz")
    zs = [p["z"] for p in pts]
    assert zs == sorted(set(zs))                       # strictly increasing
    assert max(p["score"] for p in pts) <= 99          # never the human 100
    assert min(p["score"] for p in pts) >= 1
    assert doc["_provenance"]["detections_md5"]

    fn = json_umbilicus_z_to_yx(str(out), coordinate_scale=1.0)   # the fitter's reader
    zq = np.array(sorted(truth), float)
    got = fn(zq)
    want = np.array([truth[int(z)] for z in zq])
    err = np.hypot(*(got - want).T)
    assert np.median(err) < 0.5 * PITCH, f"median error {np.median(err):.1f} px"
