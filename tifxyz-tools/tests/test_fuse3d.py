"""fuse3d on synthetic sheets whose right answer is known: a cylinder-wrap r(theta, z) about a vertical umbilicus.

Positive controls (two overlapping re-parameterised, noisy crops fuse back to the truth) each have a
NEGATIVE control (an adjacent wrap, a sheet jump) that the same code must refuse to average in.
"""
import numpy as np
import pytest

from tifxyz_tools import fuse3d as F
CX = CY = 5000.0
UMB = [{"x": CX, "y": CY, "z": z} for z in (0.0, 10000.0, 20000.0)]


def r_true(th, z):
    return 2000.0 + 12.0 * np.sin(3 * th) + 0.002 * (z - 6000.0)


def sheet(th0, th1, z0, z1, step, phase=0.0, dr=0.0, noise=0.6, seed=0, jump_from=None, jump=0.0):
    rng = np.random.default_rng(seed)
    nth = int((th1 - th0) * 2000.0 / step)
    nz = int((z1 - z0) / step)
    th = th0 + (np.arange(nth) + phase) * step / 2000.0
    z = z0 + (np.arange(nz) + phase) * step
    T, Zg = np.meshgrid(th, z)
    r = r_true(T, Zg) + dr
    if jump_from is not None:
        r = r + np.where(T > jump_from, jump, 0.0)
    r = r + rng.normal(0, noise, r.shape)
    return (CX + r * np.cos(T)).astype(np.float32), (CY + r * np.sin(T)).astype(np.float32), Zg.astype(np.float32)


def members(*specs):
    return [F.member_from_arrays(f"m{i}", *sheet(**s)) for i, s in enumerate(specs)]


def frame_for(ms):
    xyz = np.concatenate([m["sp"]["xyz"] for m in ms]).astype(float)
    return F.CylFrame(UMB, xyz)


A = dict(th0=0.0, th1=0.8, z0=5000.0, z1=6600.0, step=15.0, seed=1)
B = dict(th0=0.5, th1=1.2, z0=5400.0, z1=7000.0, step=13.0, phase=0.4, seed=2)


def rms_to_truth(P):
    v = P[..., 0] > 0
    xyz = P[v].astype(float)
    dx, dy = xyz[:, 0] - CX, xyz[:, 1] - CY
    return np.abs(np.hypot(dx, dy) - r_true(np.arctan2(dy, dx), xyz[:, 2]))


def test_two_overlapping_reparameterised_crops_fuse_back_to_the_truth():
    ms = members(A, B)
    fr = frame_for(ms)
    assert fr.usable()
    res = F.fuse(ms, fr)
    P = res["P"]
    err = rms_to_truth(P)
    assert np.median(err) < 1.0 and np.percentile(err, 95) < 2.5, (np.median(err), np.percentile(err, 95))
    v = F.validate(P, ms, 8.64)
    assert min(v["coverage_of_members"].values()) >= 0.95, v["coverage_of_members"]
    assert v["support"] >= 0.95
    assert res["diag"]["conflict_frac"] == 0.0
    assert res["diag"]["support_ge2_frac"] > 0.05                     # the overlap really was fused (not just abutted)


def test_an_adjacent_wrap_is_never_averaged_into_the_sheet():
    ms = members(A, B, dict(A, dr=30.0, seed=3))                        # the next wrap, 30 vox out, same (theta, z)
    res = F.fuse(ms, frame_for(ms[:2]))
    err = rms_to_truth(res["P"])
    assert np.percentile(err, 95) < 3.0, "the surface was pulled toward the other wrap"
    assert res["diag"]["conflict_frac"] > 0.1
    v = F.validate(res["P"], ms, 8.64)
    assert v["coverage_of_members"]["m2"] < 0.2                          # the other wrap is NOT subsumed -> never retired
    # control: without the second wrap the same call reports no conflicts
    assert F.fuse(ms[:2], frame_for(ms[:2]))["diag"]["conflict_frac"] == 0.0


def test_a_sheet_jump_inside_a_member_is_dropped_not_bent_around():
    jumped = dict(A, jump_from=0.4, jump=25.0, seed=4)
    ms = members(A, jumped)
    res = F.fuse(ms, frame_for(ms))
    err = rms_to_truth(res["P"])
    assert np.percentile(err, 95) < 3.0
    v = F.validate(res["P"], ms, 8.64)
    assert v["coverage_of_members"]["m1"] < 0.75, v["coverage_of_members"]     # the jumped half is not covered => member not retirable
    assert res["diag"]["conflict_nodes"] > 0


def test_steep_normals_are_dropped_and_counted():
    """A wall in the plane x = CX seen at theta = 90 deg has a TANGENTIAL normal: not a graph over the cylinder."""
    jj, zz = np.meshgrid(np.arange(40, dtype=np.float32), np.arange(40, dtype=np.float32))
    wall = (np.full((40, 40), CX, np.float32), CY + 2000.0 + jj * 15.0, 5000.0 + zz * 15.0)
    m = F.member_from_arrays("wall", *wall)
    fr = F.CylFrame(UMB, m["sp"]["xyz"].astype(float))
    res = F.fuse([m], fr)
    assert res["P"] is None and res["diag"]["dropped_tilt_frac"] > 0.9 and "error" in res["diag"]
    ms = members(A)
    assert F.fuse(ms, frame_for(ms))["diag"]["dropped_tilt_frac"] == 0.0        # control: a radial sheet loses nothing


# ---- graft: keep the primary lattice, grow it over the others' points
def _prep(ms):
    for m in ms:
        m["value"] = float((m["X"] > 0).sum())
    return ms


def test_graft_extends_the_primary_over_an_overlapping_member_and_keeps_the_primary_intact():
    ms = _prep(members(A, B))
    res = F.graft(ms, primary=0)
    P = res["P"]
    assert res["diag"]["grown_cells"] > 50
    err = rms_to_truth(P)
    assert np.percentile(err, 95) < 2.5, np.percentile(err, 95)
    v = F.validate(P, ms, 8.64)
    assert min(v["coverage_of_members"].values()) >= 0.9, v["coverage_of_members"]
    # control: a single member grows nothing (no other support)
    assert F.graft(ms[:1])["diag"]["grown_cells"] == 0


def test_graft_will_not_jump_to_the_adjacent_wrap_or_across_a_sheet_jump():
    wrap = dict(B, dr=30.0, seed=5)                 # the NEXT wrap, overlapping the primary's edge: must not be grafted
    ms = _prep(members(A, wrap))
    res = F.graft(ms, primary=0)
    assert res["diag"]["grown_cells"] < 30
    v = F.validate(res["P"], ms, 8.64)
    assert v["coverage_of_members"]["m1"] < 0.2
    jumped = _prep(members(A, dict(B, jump_from=0.9, jump=25.0, seed=6)))    # B has a 25 vox jump at theta = 0.9
    res2 = F.graft(jumped, primary=0)
    err = rms_to_truth(res2["P"])
    assert np.percentile(err, 99) < 3.5, "the growth crossed the sheet jump"
    assert F.validate(res2["P"], jumped, 8.64)["coverage_of_members"]["m1"] < 0.95     # the far side of the jump is not subsumed


def test_setcover_retires_true_copies_and_never_a_distinct_sheet():
    copies = _prep(members(dict(A, seed=11), dict(A, seed=12, phase=0.5), dict(A, seed=13, phase=0.25)))   # the same sheet, 3 noisy re-samplings
    other = _prep(members(dict(A, dr=30.0, seed=14)))                                                       # the next wrap: a different sheet
    r = F.setcover(copies + [dict(other[0], seg="wrap")])
    assert "wrap" in r["chosen"], "a distinct sheet must never be retired"
    assert len(r["retirable"]) == 2 and 1.8 < r["sum_over_unique"] < 2.2
    r2 = F.setcover(copies[:1] + [dict(other[0], seg="wrap")])          # control: nothing is a copy of anything
    assert r2["retirable"] == []


def test_graft_bridges_a_short_hole_only_when_asked_and_never_a_long_one():
    left = dict(th0=0.0, th1=0.40, z0=5000.0, z1=6000.0, step=15.0, seed=21)
    near = dict(th0=0.405, th1=0.80, z0=5000.0, z1=6000.0, step=15.0, seed=22)       # ~1.7 nodes (25 vox) of no data between
    far = dict(th0=0.42, th1=0.80, z0=5000.0, z1=6000.0, step=15.0, seed=23)         # ~3.7 nodes (55 vox) of no data
    ms = _prep(members(left, near))
    assert F.graft(ms, primary=0, max_gap=1)["diag"]["grown_cells"] < 20             # default: the hole stops the march
    assert F.graft(ms, primary=0, max_gap=3)["diag"]["grown_cells"] > 800            # asked: it crosses and covers the far side
    ms_far = _prep(members(left, far))
    assert F.graft(ms_far, primary=0, max_gap=3)["diag"]["grown_cells"] < 20         # a hole longer than max_gap is never bridged


def test_sheet_groups_separate_parallel_wraps_that_share_a_bounding_box():
    ms = _prep(members(dict(A, seed=31), dict(A, seed=32, phase=0.5), dict(A, dr=15.0, seed=33), dict(A, dr=15.0, seed=34, phase=0.5)))
    assert F.sheet_groups(ms) == [[0, 1], [2, 3]]                    # two wraps 15 vox apart (PHerc0211's pitch), two copies each
    assert F.sheet_groups(ms[:2] + ms[2:3]) == [[0, 1], [2]]
    assert F.sheet_groups(ms[:2], thr=0.35) == [[0, 1]]              # control: the same-wrap copies stay together


# ---- ridge-verified trim and the planted-jump test: a surface prediction with sheets every 15 voxels
PITCH = 15.0


def pred(xyz):
    """Prediction ridges (255) every PITCH voxels along the radius, +-1.5 voxels thick, about r_true(theta, z)."""
    xyz = np.asarray(xyz, float)
    r = np.hypot(xyz[:, 0] - CX, xyz[:, 1] - CY)
    th = np.arctan2(xyz[:, 1] - CY, xyz[:, 0] - CX)
    d = (r - r_true(th, xyz[:, 2]) + PITCH / 2) % PITCH - PITCH / 2
    return np.where(np.abs(d) <= 1.5, 255.0, 0.0)


def test_verify_keeps_points_on_a_ridge_and_drops_a_sheet_between_ridges():
    on = members(A)[0]
    v = F.verify(on, pred)
    assert v["verify"]["verified_frac"] > 0.95
    between = members(dict(A, dr=PITCH / 2, seed=7))[0]           # halfway between two prediction sheets
    assert F.verify(between, pred)["verify"]["verified_frac"] < 0.1


def test_planted_sheet_jump_is_flagged_on_the_boundary_and_not_elsewhere():
    """PLANT a jump of one sheet pitch at one column and require the detector to find exactly that column:
    a detector that flags edges everywhere, or misses the step, fails."""
    clean = members(A)[0]
    jumped = members(dict(A, jump_from=0.4, jump=PITCH, seed=1))[0]
    rc = F.edge_ridge_runs(clean["X"], clean["Y"], clean["Z"], pred)
    rj = F.edge_ridge_runs(jumped["X"], jumped["Y"], jumped["Z"], pred)

    def flagged(r):
        return (r["dir1"] >= 2)

    # the clean sheet: essentially no edge crosses two ridge bodies
    assert flagged(rc).mean() < 0.01
    # the jumped sheet: flagged edges concentrate in the column pair that straddles theta = 0.4
    fj = flagged(rj)
    cols = np.nonzero(fj.any(axis=0))[0]
    assert len(cols) >= 1
    th_of_col = 0.0 + (cols + 0.5) * 15.0 / 2000.0
    assert np.all(np.abs(th_of_col - 0.4) < 0.02), th_of_col
    assert fj.mean() < 0.05                                       # and nowhere else
