import numpy as np
import pytest

from tifxyz_tools import seed_dedup as SD



def plane(z=5000.0, h=60, w=60):
    j, i = np.meshgrid(np.arange(w), np.arange(h))
    return ((1000.0 + j * 20.0).astype(np.float32), (1000.0 + i * 20.0).astype(np.float32),
            np.full((h, w), z, np.float32))


def idx():
    ix = SD.SurfaceIndex()
    ix.add("PHercX_c1", 100.0, *plane(), stride=3)
    return ix.finalize()


POL = SD.SeedPolicy(True, 5.0, 50.0)


def test_seed_on_an_existing_sheet_is_refused_and_names_the_coverer():
    ok, why = idx().check([1500.0, 1500.0, 5001.0], POL)
    assert ok and why["by"] == "PHercX_c1" and why["along_normal"] <= 5.0


def test_the_NEXT_WRAP_is_a_legitimate_seed():
    """25-35 voxels along the normal is the neighbouring wrap: the production dilated mask
    forbids it, the sheet-aware check must not."""
    assert idx().check([1500.0, 1500.0, 5030.0], POL)[0] is False


def test_a_seed_far_along_the_sheet_is_novel():
    assert idx().check([1500.0 + 3000.0, 1500.0, 5000.0], POL)[0] is False


def test_only_EARLIER_segments_cover_a_seed():
    ix = idx()
    assert ix.check([1500.0, 1500.0, 5000.0], POL, before=50.0)[0] is False     # c1 was created at t=100: not earlier
    assert ix.check([1500.0, 1500.0, 5000.0], POL, before=150.0)[0] is True


def test_a_segment_never_covers_its_own_seed():
    assert idx().check([1500.0, 1500.0, 5000.0], POL, own="PHercX_c1")[0] is False


def test_points_without_a_defined_normal_never_cover():
    ix = SD.SurfaceIndex()
    X, Y, Z = plane()
    ix.add("PHercX_c2", 100.0, X, Y, Z)
    ix.finalize()
    ix.nok[:] = False                  # pretend every normal is undefined
    assert ix.check([1500.0, 1500.0, 5000.0], POL)[0] is False


def test_a_seed_beyond_the_lateral_reach_is_not_covered_by_the_rim():
    # 50 voxels lateral reach: a seed 200 voxels off the plane edge lies on the same plane but
    # on no surface point we hold.
    ix = idx()
    edge = 1000.0 + 59 * 20.0
    assert ix.check([edge + 200.0, 1500.0, 5000.0], POL)[0] is False
