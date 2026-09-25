"""umbilicus_checks on a synthetic scroll whose umbilicus is known, and the corrupted-control requirement.

The scroll is an Archimedean spiral in every slice, its centre drifting slowly with z. A check is only a
check if it FAILS on a wrong curve, so each volume/curve check is run on the true curve and on
shifted / jittered / stepped copies of it.
"""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

zarr = pytest.importorskip("zarr")
import umbilicus_checks as UC  # noqa: E402

N, PITCH, RADIUS = 160, 8.0, 68.0
VOX_UM = 10.0            # so 1 mm = 100 voxels at level 0
WINDOW_MM, STRIDE = 0.6, 3
NZ = 40


def centre(z):
    return 80.0 + 0.10 * z, 76.0 - 0.10 * z          # (y, x)


def spiral(cy, cx):
    yy, xx = np.mgrid[0:N, 0:N].astype(float)
    dy, dx = yy - cy, xx - cx
    r = np.hypot(dy, dx)
    theta = np.mod(np.arctan2(dy, dx), 2 * np.pi)
    frac = np.mod(r - PITCH * theta / (2 * np.pi), PITCH)
    sheet = ((frac < 2.0) | (frac > PITCH)) & (r > 6.0) & (r < RADIUS)
    return sheet, r < RADIUS + 3


@pytest.fixture(scope="module")
def scroll(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    ct = zarr.open(str(d / "ct.zarr" / "0"), mode="w", shape=(NZ, N, N), chunks=(4, N, N), dtype="uint8")
    pr = zarr.open(str(d / "pred.zarr" / "0"), mode="w", shape=(NZ, N, N), chunks=(4, N, N), dtype="uint8")
    for z in range(NZ):
        sh, bd = spiral(*centre(z))
        ct[z] = (bd * 200).astype("uint8")
        pr[z] = (sh * 255).astype("uint8")
    return d / "ct.zarr", d / "pred.zarr"


def true_curve():
    z = np.arange(2.0, NZ, 3.0)
    y, x = centre(z)
    return {"z": z, "x": x.copy(), "y": y.copy(), "score": np.full(len(z), 50.0), "fmt": "control_points", "prov": None}


def vol(scroll, curve):
    return UC.volume_checks(scroll[0], scroll[1], curve, VOX_UM, level=0, n_slices=8,
                            window_mm=WINDOW_MM, stride=STRIDE)


def test_true_curve_passes_and_reads_every_slice(scroll):
    v = vol(scroll, true_curve())
    s = v["summary"]
    assert s["n_slices_used"] == 8
    assert s["offset_mm"]["p50"] < 0.1               # 0.1 mm = 10 voxels; pitch is 8 voxels
    assert UC.flags(UC.curve_checks(true_curve(), VOX_UM), v, None) == []


@pytest.mark.parametrize("shift_vox", [25, 40])
def test_a_shifted_curve_is_flagged_by_the_criterion_offset(scroll, shift_vox):
    bad = UC.corrupt(true_curve(), "shift", shift_vox, 0)
    v = vol(scroll, bad)
    assert v["summary"]["offset_mm"]["p50"] > 0.15   # >> the clean 0.1 mm
    assert v["summary"]["offset_mm"]["p50"] > vol(scroll, true_curve())["summary"]["offset_mm"]["p50"] * 2


def test_jitter_is_caught_by_the_curve_check_and_a_pure_shift_is_not():
    c = true_curve()
    c = {**c, "z": np.arange(0.0, 400.0, 4.0)}
    y, x = centre(c["z"] * 0.0)
    c["x"], c["y"] = np.full(len(c["z"]), 80.0) + 0.05 * c["z"], np.full(len(c["z"]), 70.0)
    c["score"] = np.full(len(c["z"]), 50.0)
    clean = UC.jitter_mm(c, VOX_UM)
    assert clean["p90"] < 0.01
    jit = UC.jitter_mm(UC.corrupt(c, "jitter", 5.0), VOX_UM)          # sigma 5 voxels = 0.05 mm per axis
    assert jit["p90"] > 10 * max(clean["p90"], 1e-3)
    shifted = UC.jitter_mm(UC.corrupt(c, "shift", 300.0, 0), VOX_UM)
    assert shifted["p90"] == pytest.approx(clean["p90"], abs=1e-6)     # a pure shift is invisible to it, by construction


def test_agreement_and_reference_error_are_in_mm():
    a = true_curve()
    b = UC.corrupt(a, "shift", 100.0, 0)              # 100 voxels x 10 um = 1 mm
    assert UC.agreement_mm(a, b, VOX_UM)["p50"] == pytest.approx(1.0, rel=1e-6)
    assert UC.reference_error_mm(b, a, VOX_UM)["p50"] == pytest.approx(1.0, rel=1e-6)


def test_a_local_step_is_seen_by_the_p90_but_not_the_p50(scroll):
    """The reason `offset_p90_mm` exists: a wrong stretch over the middle quarter of z leaves the median clean."""
    c = true_curve()
    stepped = UC.corrupt(c, "step", 40.0, 0)
    s = vol(scroll, stepped)["summary"]["offset_mm"]
    clean = vol(scroll, c)["summary"]["offset_mm"]
    assert s["p90"] > 2 * clean["p90"] + 0.05
    assert s["p50"] <= clean["p50"] + 0.05


def test_unpairable_levels_are_reported_not_guessed(scroll, tmp_path):
    other = zarr.open(str(tmp_path / "p.zarr" / "0"), mode="w", shape=(NZ, N + 8, N), chunks=(4, N, N), dtype="uint8")
    other[:] = 1
    with pytest.raises(FileNotFoundError):
        UC.volume_checks(scroll[0], tmp_path / "p.zarr", true_curve(), VOX_UM, level=0, n_slices=4)


def test_load_curve_file_formats_and_cli(tmp_path, scroll):
    c = true_curve()
    p = tmp_path / "umbilicus.json"
    p.write_text(json.dumps({"control_points": [{"x": int(x), "y": int(y), "z": int(z), "score": 50}
                                                for z, x, y in zip(c["z"], c["x"], c["y"])]}))
    assert UC.load_curve_file(p)["fmt"] == "control_points"
    out = tmp_path / "r.json"
    assert UC.main([str(p), "--voxel-um", str(VOX_UM), "--ct", str(scroll[0]), "--pred", str(scroll[1]),
                    "--level", "0", "--out", str(out)]) == 0
    r = json.loads(out.read_text())
    assert r["md5"] and r["curve"]["unit"] == "mm" and "volume" in r and "flags" in r
    with pytest.raises(SystemExit):
        UC.main([str(p)])                              # --voxel-um is required: never guessed
