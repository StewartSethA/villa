import numpy as np
from contrib.render_direction_check.direction_check import block_correlations, verdict
from contrib.render_direction_check.layer_order_check import plane_corr, order_verdict


def test_direction_picks_the_map_that_matches_the_reference():
    rng = np.random.default_rng(0); ref = rng.integers(1, 255, (512, 512)).astype(np.uint8)
    fwd = np.clip(ref.astype(np.int16) + rng.integers(-20, 20, ref.shape), 1, 255).astype(np.uint8); rev = rng.integers(1, 255, ref.shape).astype(np.uint8)
    assert verdict(block_correlations(fwd, rev, ref))["verdict"] == "fwd"
    assert verdict(block_correlations(rev, fwd, ref))["verdict"] == "rev"


def test_direction_is_ambiguous_for_two_unrelated_maps():
    rng = np.random.default_rng(1); r = lambda: rng.integers(1, 255, (512, 512)).astype(np.uint8)
    assert verdict(block_correlations(r(), r(), r()))["verdict"] == "ambiguous"


def test_layer_order_detects_a_reversed_stack():
    rng = np.random.default_rng(2); base = rng.integers(1, 255, (1, 64, 64)).astype(np.float32); A = np.concatenate([base * (1 + 0.2 * k) + rng.normal(0, 3, base.shape) for k in range(9)]).clip(1, 255)
    assert order_verdict(plane_corr(A, A))["verdict"] != "REVERSED"
    A2 = np.stack([(np.sin(np.arange(64 * 64).reshape(64, 64) * (k + 1) / 50.0) + 2) * 50 + rng.normal(0, 1, (64, 64)) for k in range(9)])
    assert order_verdict(plane_corr(A2, A2[::-1]))["verdict"] == "REVERSED"
    assert order_verdict(plane_corr(A2, A2))["verdict"] == "same order"
