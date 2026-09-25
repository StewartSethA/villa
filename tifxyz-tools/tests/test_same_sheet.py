"""Same-sheet estimator on synthetic sheets whose right answer is known.

Each positive case has a paired negative one (the NEXT WRAP, a crossing sheet), so an estimator
that calls everything covered, or nothing covered, cannot pass.
"""
import json

import numpy as np
import pytest

from tifxyz_tools import same_sheet as C


def plane(z0, n=40, step=10.0, off=0.0, tilt=0.0):
    i, j = np.mgrid[0:n, 0:n].astype(np.float32)
    X = 1000 + off + i * step
    Y = 1000 + j * step
    Z = (z0 + tilt * i * step + 0 * j).astype(np.float32)
    return C.surface_points(X.astype(np.float32), Y.astype(np.float32), Z)


def covered(P, T, **kw):
    return C.covered_fraction(P["xyz"], P["nrm"], P["nok"], T, **kw)


def test_the_same_sheet_is_covered_and_the_next_wrap_is_not():
    A = plane(500.0)
    assert covered(A, plane(502.0)) == 1.0        # same sheet, grown twice
    assert covered(A, plane(530.0)) == 0.0        # the adjacent wrap, ~30 voxels on
    assert covered(A, plane(504.5)) == 0.0        # just past the 4-voxel band
    f = covered(A, plane(502.0, off=200.0))       # half-overlapping copy
    assert 0.45 <= f <= 0.6, f
    assert covered(A, plane(500.0, tilt=1.0)) < 0.1   # crossing at 45 degrees: not the same sheet


def test_the_alignment_argument_is_honoured_and_global_state_is_not_touched():
    A = plane(500.0)
    tilted = plane(500.0, tilt=0.25)               # ~14 degrees off
    assert covered(A, tilted, align_deg=20.0) > covered(A, tilted, align_deg=5.0)
    assert C.ALIGN_DEG == 20.0


def test_a_surface_with_no_normals_is_nan_not_zero():
    T = plane(500.0)
    assert np.isnan(C.covered_fraction(np.zeros((3, 3)), np.zeros((3, 3)), np.zeros(3, bool), T))


def test_holes_are_not_surface():
    i, j = np.mgrid[0:30, 0:30].astype(np.float32)
    X, Y, Z = 1000 + i * 10, 1000 + j * 10, np.full_like(i, 500.0)
    X[10:14, :] = -1                               # invalid rows, tifxyz style
    S = C.surface_points(X, Y, Z)
    assert len(S["xyz"]) == 30 * 30 - 4 * 30
    assert S["edge_med"] == pytest.approx(10.0)


def test_read_lattice_tifxyz_and_pushed_round_trip(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    i, j = np.mgrid[0:8, 0:6].astype(np.float32)
    X, Y, Z = 100 + i, 200 + j, np.full_like(i, 50.0)
    d = tmp_path / "seg"
    d.mkdir()
    for n, A in zip("xyz", (X, Y, Z)):
        tifffile.imwrite(d / f"{n}.tif", A)
    x, y, z = C.read_lattice("tifxyz", str(d))
    assert np.array_equal(x, X) and np.array_equal(z, Z)

    import gzip
    p = tmp_path / "push"
    p.mkdir()
    (p / "latest.json").write_text(json.dumps({"grid": [8, 6]}))
    (p / "latest.bin").write_bytes(gzip.compress(np.stack([X, Y, Z], -1).astype(np.float32).tobytes()))
    x2, y2, z2 = C.read_lattice("pushed", str(p / "latest.bin"))
    assert np.array_equal(y2, Y)
    with pytest.raises(ValueError):
        C.read_lattice("nonsense", str(d))


def test_sample_idx_is_deterministic_per_key():
    a, b = C.sample_idx("segA", 10000, 500), C.sample_idx("segA", 10000, 500)
    assert np.array_equal(a, b) and len(a) == 500
    assert not np.array_equal(a, C.sample_idx("segB", 10000, 500))
    assert np.array_equal(C.sample_idx("x", 5, 500), np.arange(5))
