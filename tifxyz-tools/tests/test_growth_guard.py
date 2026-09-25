"""growth_guard: per-cell frontier pruning on SYNTHETIC lattices whose right answer is known.

Every positive test has a paired 'switch it off and the cell stays' control, so a test that
passes because the guard prunes everything (or nothing) cannot pass.
"""
import json

import numpy as np
import pytest

from tifxyz_tools import growth_guard as G

VOX = 9.362           # um per voxel
PITCH = 20.0          # voxels per lattice cell (tracer step_size)


def plane(h=90, w=90, z=5000.0):
    j, i = np.meshgrid(np.arange(w), np.arange(h))
    X = 1000.0 + j * PITCH
    Y = 1000.0 + i * PITCH
    Z = np.full_like(X, z)
    return X.astype(np.float32), Y.astype(np.float32), Z.astype(np.float32)


def pol(**kw):
    base = dict(enabled=True, vacuum=False, fold=False, plan=False, overlap=False)
    base.update(kw)
    return G.GuardPolicy(**base)


def test_clean_plane_is_untouched_by_every_criterion():
    X, Y, Z = plane()
    r = G.evaluate(X, Y, Z, G.GuardPolicy(enabled=True), VOX, sampler=lambda p: np.full(len(p), 100.0))
    assert r.cells_after == r.cells_before == 90 * 90
    assert r.stop is None and r.components_cut == 0


def test_vacuum_at_the_frontier_is_cut_and_a_control_keeps_it():
    X, Y, Z = plane()
    air = lambda p: np.where(p[:, 0] > 1000 + 76 * PITCH, 0.0, 100.0)      # right 13 columns are air
    r = G.evaluate(X, Y, Z, pol(vacuum=True), VOX, sampler=air)
    assert r.pruned_by["vacuum"] >= 90 * 12
    assert r.cells_after <= 90 * 78 and r.cells_after > 90 * 70      # air starts at column 76; margin_rings=0
    off = G.evaluate(X, Y, Z, pol(vacuum=False), VOX, sampler=air)
    assert off.cells_after == off.cells_before                             # control: switched off, nothing cut


def test_enclosed_vacuum_is_a_hole_not_a_cut():
    """The frontier may wrap AROUND a vacuum and leave a hole; only growing THROUGH it is limited."""
    X, Y, Z = plane()
    bite = lambda p: np.where((np.abs(p[:, 0] - 1900) < 130) & (np.abs(p[:, 1] - 1900) < 130), 0.0, 100.0)
    r = G.evaluate(X, Y, Z, pol(vacuum=True), VOX, sampler=bite)
    assert r.components_cut == 0 and r.interior_bad_kept >= 1
    assert r.cells_after == r.cells_before


def test_hairpin_tail_is_dropped():
    """Columns >= 60 fold back on themselves (a 180-degree turn over a few cells): the crease is
    a BARRIER, so the folded-back tail goes with it."""
    X, Y, Z = plane(60, 100)
    X = X.copy()
    j = np.arange(100)
    fold = np.where(j < 60, j, 60 - (j - 60))            # 0..59, then 60, 59, 58, ...
    X = 1000.0 + fold[None, :] * PITCH + np.zeros((60, 1))
    Z = (5000.0 + np.where(j < 60, 0.0, 6.0)[None, :] + np.zeros((60, 1))).astype(np.float32)   # tail on the next wrap, ~6 vox above
    Y = plane(60, 100)[1]
    r = G.evaluate(X.astype(np.float32), Y, Z, pol(fold=True, fold_radius_um=500.0), VOX)     # this synthetic crease is a 500 um radius one
    assert r.pruned_by["fold"] > 0 and r.stop is None
    assert r.cells_after < 60 * 62                        # the tail beyond the crease is gone
    assert r.cells_after > 60 * 50                        # the body before it is kept
    off = G.evaluate(X.astype(np.float32), Y, Z, pol(fold=False), VOX)
    assert off.cells_after == off.cells_before


def test_smooth_curvature_is_not_a_hairpin_and_not_a_crumple():
    X, Y, Z = plane(80, 80)
    z = 5000.0 + 2000.0 * (1 - np.cos(np.arange(80) / 80.0 * 0.6))       # gentle bend, radius ~ mm
    Z = (z[None, :] + np.zeros((80, 1))).astype(np.float32)
    r = G.evaluate(X, Y, Z, pol(fold=True, plan=True), VOX)
    assert r.cells_after == r.cells_before


def test_crumpled_frontier_is_cut_by_the_planarity_criterion():
    X, Y, Z = plane(80, 80)
    rng = np.random.default_rng(3)
    Z = Z.copy()
    Z[:, 62:] += rng.normal(0, 80.0, size=Z[:, 62:].shape).astype(np.float32)   # mush swamp on the right
    r = G.evaluate(X, Y, Z, pol(plan=True), VOX)
    assert r.pruned_by["plan"] > 80 * 10
    assert r.cells_after < 80 * 66
    off = G.evaluate(X, Y, Z, pol(plan=False), VOX)
    assert off.cells_after == off.cells_before


def test_overlap_with_a_neighbour_is_cut_but_a_seam_is_kept():
    X, Y, Z = plane(70, 90)
    idx = G.SegmentIndex(vox=4.0, deg=20.0)
    Xo, Yo, Zo = plane(70, 90)
    Xo, Yo, Zo = Xo[:, 45:], Yo[:, 45:], Zo[:, 45:] + 2.0        # a neighbour holding the right half, 2 vox off
    idx.add("PHercX_other", Xo, Yo, Zo)
    r = G.evaluate(X, Y, Z, pol(overlap=True, overlap_keep_rings=2, margin_rings=0), VOX, cover=idx)
    kept_cols = np.nonzero(r.keep.any(axis=0))[0]
    assert r.pruned_by["overlap"] > 70 * 30
    assert 44 <= kept_cols.max() <= 50               # body + the 2-ring seam for the stitcher, not the whole overlap
    r0 = G.evaluate(X, Y, Z, pol(overlap=True, overlap_keep_rings=0, margin_rings=0), VOX, cover=idx)
    assert r0.keep.any(axis=0).nonzero()[0].max() < kept_cols.max()   # the seam is the only difference
    off = G.evaluate(X, Y, Z, pol(overlap=False), VOX, cover=idx)
    assert off.cells_after == off.cells_before


def test_a_neighbouring_WRAP_is_not_overlap():
    """The next wrap sits ~25-35 voxels along the normal (same_sheet.py): it must NOT read as the same sheet."""
    X, Y, Z = plane(70, 90)
    idx = G.SegmentIndex(vox=4.0, deg=20.0)
    idx.add("PHercX_next_wrap", *(a if k < 2 else a + 30.0 for k, a in enumerate(plane(70, 90))))
    r = G.evaluate(X, Y, Z, pol(overlap=True), VOX, cover=idx)
    assert r.cells_after == r.cells_before


def test_only_the_largest_piece_survives():
    X, Y, Z = plane(60, 90)
    X, Y, Z = X.copy(), Y.copy(), Z.copy()
    X[:, 50:60] = -1; Y[:, 50:60] = -1; Z[:, 50:60] = -1              # a gap splits an island of 30 columns off
    r = G.evaluate(X, Y, Z, pol(min_keep_cells=10), VOX)
    assert r.cells_after == 60 * 50
    r2 = G.evaluate(X, Y, Z, pol(min_keep_cells=10, keep_islands=True), VOX)
    assert r2.cells_after == 60 * 80


def test_regrowth_into_pruned_ground_is_measured():
    prev = np.array([[i * 20.0, 0, 0] for i in range(10)])
    pruned = np.array([[i * 20.0, 0, 0] for i in range(10, 20)])
    new_on_pruned = np.array([[i * 20.0, 0, 0] for i in range(10, 16)])
    new_elsewhere = np.array([[i * 20.0, 400, 0] for i in range(10, 16)])
    assert G.regrown_fraction(prev, pruned, new_on_pruned, 10.0) == 1.0
    assert G.regrown_fraction(prev, pruned, new_elsewhere, 10.0) == 0.0
    assert G.regrown_fraction(prev, pruned, prev, 10.0) == 0.0          # nothing added


def test_guard_tifxyz_writes_a_new_dir_and_never_touches_the_source(tmp_path):
    import tifffile
    X, Y, Z = plane(60, 60)
    src = tmp_path / "src"; src.mkdir()
    for a, A in zip("xyz", (X, Y, Z)):
        tifffile.imwrite(src / f"{a}.tif", A)
    (src / "meta.json").write_text(json.dumps({"uuid": "s"}))
    before = {p.name: p.read_bytes() for p in src.iterdir()}
    air = lambda p: np.where(p[:, 0] > 1000 + 45 * PITCH, 0.0, 100.0)
    res = G.guard_tifxyz(str(src), str(tmp_path / "dst"), pol(vacuum=True), VOX, sampler=air)
    assert {p.name: p.read_bytes() for p in src.iterdir()} == before
    x2 = tifffile.imread(tmp_path / "dst" / "x.tif")
    assert (x2 == -1).sum() == 60 * 60 - res.cells_after > 0
    meta = json.loads((tmp_path / "dst" / "meta.json").read_text())
    assert meta["guard"]["cells_after"] == res.cells_after and meta["area_cm2"] > 0


def test_guard_round_crops_carries_state_and_blocks_a_frontier_that_regrows(tmp_path):
    import tifffile
    X, Y, Z = plane(60, 60)
    src = tmp_path / "s"; src.mkdir()
    for a, A in zip("xyz", (X, Y, Z)):
        tifffile.imwrite(src / f"{a}.tif", A)
    air = lambda p: np.where(p[:, 0] > 1000 + 45 * PITCH, 0.0, 100.0)
    st = G.GuardState()
    out, info = G.guard_round(str(src), pol(vacuum=True), VOX, st, sampler=air)
    assert out != str(src) and info["stop"] is None and st.rounds_cut == 1 and len(st.pruned_xyz) > 0
    # the "tracer" now regrows exactly the ground we pruned: the next round must say the frontier is blocked
    out2, info2 = G.guard_round(str(src), pol(vacuum=True), VOX, st, sampler=air)
    assert info2["regrown_frac"] >= 0.9 and info2["stop"] == "frontier_blocked"
    # control: a round that adds nothing on pruned ground is not blocked
    st3 = G.GuardState(pruned_xyz=np.array([[9e5, 9e5, 9e5]]), keep_xyz=np.zeros((0, 3)), rounds_cut=1)
    _, info3 = G.guard_round(str(src), pol(vacuum=False), VOX, st3)
    assert info3["stop"] is None


def test_zarr_sampler_reads_level_and_treats_outside_as_air(tmp_path):
    zarr = pytest.importorskip("zarr")
    g = zarr.open_group(str(tmp_path / "ct.zarr"), mode="w")
    a = g.create_dataset("1", shape=(20, 20, 20), chunks=(8, 8, 8), dtype="uint8")
    a[:] = 77
    a[0:5] = 0                                       # air below z = 10 (level-1 index 5 -> level-0 z 10)
    s = G.ZarrSampler(str(tmp_path / "ct.zarr"), level=1)
    got = s(np.array([[20.0, 20.0, 30.0], [20.0, 20.0, 4.0], [1e6, 0.0, 0.0]]))   # x, y, z in level-0 voxels
    assert got.tolist() == [77.0, 0.0, 0.0]


def test_a_young_small_lattice_that_lost_nothing_is_never_declared_nothing_left():
    """A tiny lattice below min_keep_cells that lost nothing must not be reported as emptied by the guard
    (a real canary run ended two young 9- and 28-cell lattices this way)."""
    X, Y, Z = plane(4, 4)
    r = G.evaluate(X, Y, Z, G.GuardPolicy(enabled=True), VOX, sampler=lambda p: np.full(len(p), 100.0))
    assert r.cells_after == r.cells_before == 16 and r.stop is None
    air = lambda p: np.where(p[:, 0] > 1000 + 1 * PITCH, 0.0, 100.0)      # but a guard that empties the sheet still says so
    r2 = G.evaluate(*plane(12, 12), pol(vacuum=True, min_keep_cells=64), VOX, sampler=air)
    assert r2.stop == "nothing_left"
