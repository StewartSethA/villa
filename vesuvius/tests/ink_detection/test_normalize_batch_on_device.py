"""normalize_batch_on_device must agree with the per-patch CPU reference.

The device path exists to move the median/percentile work of
``normalize_robust`` off the dataloader workers. It is only acceptable if it
computes the same thing, including the degenerate-spread fallback chain, so
each case here is compared against ``normalize_flat_patch`` directly. The
tests run the device function on CPU tensors: they check the arithmetic, not
CUDA.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vesuvius.ink_detection.inference.infer import (
    normalize_batch_on_device,
    normalize_flat_patch,
)

SHAPE = (26, 64, 64)  # Z, Y, X: an even element count (106496), like production


def _patches() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1234)
    texture = rng.normal(120.0, 30.0, SHAPE).clip(0, 255).round()
    outliers = np.full(SHAPE, 50.0)
    outliers.flat[[3, 4000, 90000]] = [255.0, 0.0, 200.0]  # MAD == 0, std > 0
    sparse = np.zeros(SHAPE)
    sparse[:, :8, :8] = rng.integers(1, 255, (SHAPE[0], 8, 8))  # >50 % zeros
    return {
        "texture": texture,
        "outliers_mad_zero": outliers,
        "constant": np.full(SHAPE, 77.0),
        "all_zero": np.zeros(SHAPE),
        "mostly_zero": sparse,
    }


@pytest.mark.parametrize("name", list(_patches()))
def test_tifxyz_robust_matches_cpu_reference(name: str) -> None:
    patch = _patches()[name].astype(np.float32)
    expected = normalize_flat_patch(patch, "tifxyz_robust")
    got = normalize_batch_on_device(
        torch.from_numpy(patch)[None, None], "tifxyz_robust"
    )[0, 0].numpy()
    # float32 rounding only; a wrong fallback tier differs by whole units.
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=2e-3, err_msg=name)


def test_statistics_are_per_patch_not_per_batch() -> None:
    patches = _patches()
    batch = np.stack([patches["texture"], patches["mostly_zero"]]).astype(np.float32)
    got = normalize_batch_on_device(torch.from_numpy(batch)[:, None], "tifxyz_robust")
    for i in range(2):
        expected = normalize_flat_patch(batch[i], "tifxyz_robust")
        np.testing.assert_allclose(got[i, 0].numpy(), expected, rtol=0.0, atol=2e-3)


def test_divide_255_matches_cpu_reference() -> None:
    patch = _patches()["texture"].astype(np.float32)
    # normalize_flat_patch("divide_255") scales a contiguous float32 input in
    # place, so hand it a copy.
    expected = normalize_flat_patch(patch.copy(), "divide_255")
    got = normalize_batch_on_device(torch.from_numpy(patch)[None, None], "divide_255")
    np.testing.assert_allclose(got[0, 0].numpy(), expected, rtol=0.0, atol=1e-7)


def test_unknown_preprocessing_is_rejected() -> None:
    with pytest.raises(ValueError):
        normalize_batch_on_device(torch.zeros(1, 1, 2, 2, 2), "nope")
