"""ARM B (growth_degeneracy_2026-09-28.md): a post-round GEOMETRIC self-intersection check,
against the REAL round-1 checkpoint of the motivating example (PHerc0191_cea9032, packed into
`tests/fixtures/growth_guard_selfcross_cea9032_r1.npz`, 79.5 KB, since the repo's `.gitignore`
blanket-excludes `*.tif`/`*.npy` but not `.npz`) and the REAL upstream `vc_tifxyz_selfcross`
binary -- not a mock or a synthetic fold, because the whole point of this arm is that the
existing per-cell fold/plan proxies do not agree with the tool that actually knows what a
triangle crossing is, and a synthetic "hairpin" is exactly the shape those proxies ARE tuned for
(confirmed while building this file: several hand-built folded/crumpled synthetic lattices
registered 0 transverse crossings with the real tool -- a fold that merely stacks two flat layers
does not necessarily self-intersect; this real, badly-grown segment reliably does).

Fixture provenance: segment `PHerc0191_cea9032`, round 1 checkpoint `auto_grown_20260928193716670`, on
one CPU host, pulled via rsync for `docs/experiments/growth_degeneracy_2026-09-28.md`; density there
(11958+12784)/2/15214 = 0.813, matching this file's own `test_the_fixture_is_authenticated`.

Skips (not fails) when the binary is not present on this box, so the suite stays green on a peer
that never carries vc3d_appimage; set SELFCROSS_BIN or run on the hub to exercise it.
"""
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import tifffile

from tifxyz_tools import growth_guard as G

FIXTURE_NPZ = Path(__file__).parent / "fixtures" / "growth_guard_selfcross_cea9032_r1.npz"
SELFCROSS_BIN = os.environ.get("SELFCROSS_BIN") or shutil.which("vc_tifxyz_selfcross") or ""
pytestmark = [pytest.mark.unit,
             pytest.mark.skipif(not os.path.exists(SELFCROSS_BIN),
                                reason=f"vc_tifxyz_selfcross not found at {SELFCROSS_BIN}")]
VOX = 9.362


def _shape():
    with np.load(FIXTURE_NPZ) as z:
        return z["x"].shape


def fresh_cur(tmp_path) -> Path:
    """Unpack the real fixture into its own private tifxyz dir -- guard_tifxyz/guard_round never
    modify `src`, but each test gets its own directory regardless, to avoid any cross-test state
    via the `.selfcross_guard_*.json` scratch file guard writes next to `src` and deletes."""
    d = tmp_path / "r1"
    d.mkdir(parents=True, exist_ok=True)
    with np.load(FIXTURE_NPZ) as z:
        for a in "xyz":
            tifffile.imwrite(d / f"{a}.tif", z[a])
        tifffile.imwrite(d / "generations.tif", z["generations"])
        (d / "meta.json").write_text(str(z["meta_json"].item()))
    return d


def test_the_fixture_is_authenticated_a_real_badly_grown_round_not_a_synthetic_shape(tmp_path):
    """CLAUDE.md: never trust a label, always check the artifact. Confirms the copied fixture
    still reads exactly as it did when pulled for the report, and is a genuine planted positive."""
    cur = fresh_cur(tmp_path)
    pol = G.GuardPolicy(selfcross=True, selfcross_bin=SELFCROSS_BIN)
    mask, info = G.selfcross_check(str(cur), _shape(), pol)
    assert info["ran"] and info["triangles"] == 15214
    assert info["density"] == pytest.approx(0.813, abs=0.001), info
    assert mask is not None and mask.any(), "the selfx mask must mark at least one lattice cell"


def test_RED_without_arm_B_the_existing_fold_plan_criteria_leave_self_intersections_in_the_shipped_checkpoint(tmp_path):
    """Current production default (grow.guard.enabled=1 fleet-wide, selfcross OFF): fold+plan DO
    prune some cells, but nothing verifies the result is actually free of self-intersection.
    Matches the report's finding on the real fleet run of this exact segment (density 0.813 ->
    guarded 0.589 -- reduced, never zeroed)."""
    cur = fresh_cur(tmp_path)
    state = G.GuardState()
    pol = G.GuardPolicy(enabled=True, fold=True, plan=True, vacuum=False, overlap=False,
                        selfcross=False, selfcross_hairpin_abort_ratio=0.0)
    dst, info = G.guard_round(str(cur), pol, VOX, state)
    assert dst != str(cur), "setup: fold/plan must have cut something this round"
    _, post = G.selfcross_check(dst, _shape(), G.GuardPolicy(selfcross=True, selfcross_bin=SELFCROSS_BIN))
    assert post["ran"] and post["density"] > 0.0, (
        "the checkpoint fold/plan actually SHIPPED (no stop, no verification) still has "
        f"self-intersections: {post}")
    assert info["stop"] != "selfcross_nonzero", "selfcross is OFF; it must not be what (didn't) stop this"


def test_GREEN_arm_B_never_lets_a_still_crossing_checkpoint_through_without_stopping(tmp_path):
    """The invariant this arm exists to guarantee: whatever guard_round returns, EITHER its
    self-intersection density is exactly 0, OR info['stop'] says why growth ended instead of
    shipping it. Same fixture and fold/plan settings as the RED test, selfcross ON."""
    cur = fresh_cur(tmp_path)
    state = G.GuardState()
    pol = G.GuardPolicy(enabled=True, fold=True, plan=True, vacuum=False, overlap=False,
                        selfcross=True, selfcross_bin=SELFCROSS_BIN, selfcross_hairpin_abort_ratio=0.0)
    dst, info = G.guard_round(str(cur), pol, VOX, state)
    assert "selfcross_post" in info, info
    density = info["selfcross_post"]["density"]
    if density > 0.0:
        assert info["stop"] == "selfcross_nonzero", (
            f"density is {density} (not 0) but nothing stopped the segment -- it would ship "
            f"a still-crossing surface to flatten/render. info={info}")
    assert density == 0.0 or info["stop"] is not None


def test_switch_it_off_control_selfcross_false_changes_nothing_about_selfx_pruning(tmp_path):
    on_dst, on_info = G.guard_round(str(fresh_cur(tmp_path / "on")), G.GuardPolicy(
        enabled=True, selfcross=True, selfcross_bin=SELFCROSS_BIN, selfcross_hairpin_abort_ratio=0.0),
        VOX, G.GuardState())
    off_dst, off_info = G.guard_round(str(fresh_cur(tmp_path / "off")), G.GuardPolicy(
        enabled=True, selfcross=False, selfcross_hairpin_abort_ratio=0.0), VOX, G.GuardState())
    assert "selfcross_post" in on_info and "selfcross_post" not in off_info


def test_the_hairpin_ratio_early_abort_fires_on_the_real_degenerate_round(tmp_path):
    """geo_hairpin_lines/geo_lines_read, calibrated on a held-out half of the 40-segment fleet
    sample (see GuardPolicy.selfcross_hairpin_abort_ratio's docstring: threshold 0.66, TPR 1.00 on
    both halves). This segment reads 177/180 = 0.983 at round 1 in the report -- confirm the guard
    reproduces that and aborts BEFORE even running the (here, unnecessary) selfcross subprocess."""
    cur = fresh_cur(tmp_path)
    pol = G.GuardPolicy(enabled=True, selfcross=False, selfcross_hairpin_abort_ratio=0.66,
                        selfcross_hairpin_min_lines=5)
    dst, info = G.guard_round(str(cur), pol, VOX, G.GuardState())
    assert info["stop"] == "hairpin_abort", info
    assert info["hairpin_ratio"] == pytest.approx(0.983, abs=0.01), info


def test_a_clean_plane_never_aborts_on_either_signal(tmp_path):
    """Switch-it-off-style negative control using growth_guard's OWN synthetic clean-plane
    fixture (test_growth_guard.py's `plane()`), so this file's positive (a real degenerate
    checkpoint) is checked against a genuine negative, not just against itself."""
    import numpy as np
    import tifffile
    d = tmp_path / "clean"
    d.mkdir()
    h, w, pitch = 90, 90, 20.0
    j, i = np.meshgrid(np.arange(w), np.arange(h))
    X = (1000.0 + j * pitch).astype("float32")
    Y = (1000.0 + i * pitch).astype("float32")
    Z = np.full_like(X, 5000.0)
    for a, A in zip("xyz", (X, Y, Z), strict=True):
        tifffile.imwrite(d / f"{a}.tif", A)
    (d / "meta.json").write_text(json.dumps({
        "area_cm2": 1.0, "max_gen": 1, "scale": [0.05, 0.05],
        "format": "tifxyz", "type": "seg", "uuid": "clean", "source": "test"}))
    pol = G.GuardPolicy(enabled=True, selfcross=True, selfcross_bin=SELFCROSS_BIN,
                        selfcross_hairpin_abort_ratio=0.66, selfcross_hairpin_min_lines=5)
    dst, info = G.guard_round(str(d), pol, VOX, G.GuardState())
    assert info["stop"] is None, info
    assert info["selfcross_post"]["density"] == 0.0
    assert info.get("hairpin_ratio", 0.0) == 0.0
