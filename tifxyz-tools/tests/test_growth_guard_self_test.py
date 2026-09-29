"""growth_guard.self_test(): a planted defect per criterion must fire on its defect and stay off a
clean region of the same lattice; a guard that cannot verify raises GuardBroken.

The selfx sub-check needs `vc_tifxyz_selfcross`: set SELFCROSS_BIN to run it, else it is skipped
(self_test reports which criteria it checked)."""
import os

import numpy as np
import pytest

from tifxyz_tools import growth_guard as G

SELFCROSS_BIN = os.environ.get("SELFCROSS_BIN", "")


def test_self_test_passes_clean_and_reports_every_criterion_it_checked():
    ok, detail = G.self_test(vc_bin=SELFCROSS_BIN or None)
    assert ok, detail
    for name in ("vacuum", "fold", "plan", "overlap", "quad_flip", "stretch",
                 "wrap_spacing", "curvature", "ridge_hit", "seam", "empty_space", "roughness"):
        assert name in detail, f"{name} missing from self_test detail: {detail}"


def test_self_test_budget_under_one_second():
    import time
    t0 = time.time()
    G.self_test()
    assert (time.time() - t0) < 1.0


def test_RED_a_criterion_that_never_fires_is_caught(monkeypatch):
    monkeypatch.setattr(G, "vacuum_mask", lambda P, V, sampler, pol: np.zeros(V.shape, bool))
    ok, detail = G.self_test()
    assert not ok and "vacuum" in detail


def test_RED_a_criterion_that_fires_everywhere_is_caught(monkeypatch):
    monkeypatch.setattr(G, "stretch_mask", lambda P, V, pol: V.copy())
    ok, detail = G.self_test()
    assert not ok and "stretch" in detail


def test_selfcross_required_without_a_binary_fails_loudly(monkeypatch):
    monkeypatch.setattr(G.shutil, "which", lambda name: None)
    ok, detail = G.self_test(vc_bin="/nonexistent/vc_tifxyz_selfcross", selfcross_required=True)
    assert not ok and "selfx" in detail


def test_mark_broken_raises():
    with pytest.raises(G.GuardBroken):
        G.mark_broken("planted")
