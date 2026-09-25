"""Regression test for the occupancy-scan memory blowup in infer.py.

Feeding infer.py's occupancy scanner a segment whose Zarr group has no
genuine downsampled pyramid level (only a native-resolution "0", as with
older single-level surface-volume Zarrs) used to force a dense
``array[:]`` read of the full native array. On a real ~32k x 51k x 65
segment that hit ~187GB RSS. This test checks correctness against a naive
reference on small arrays, and checks that the chunked implementation
never materializes more than one chunk at a time on a large array.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from vesuvius.ink_detection.inference.infer import (
    compute_nonempty_mask_from_lowres_array,
)


def _naive_reference(values: np.ndarray) -> np.ndarray:
    if values.ndim == 2:
        return values != 0
    depth_axis = 0 if int(np.argmin(values.shape)) == 0 else 2
    return np.any(values != 0, axis=depth_axis)


@pytest.mark.parametrize("shape,chunks", [
    ((5, 17, 23), (5, 4, 4)),
    ((17, 23), (4, 4)),
    ((23, 17, 5), (4, 4, 5)),
])
def test_matches_naive_reference_on_small_arrays(shape, chunks):
    rng = np.random.default_rng(0)
    data = (rng.random(shape) * 3).astype(np.uint8)
    array = zarr.array(data, chunks=chunks)
    result = compute_nonempty_mask_from_lowres_array(array)
    np.testing.assert_array_equal(result, _naive_reference(data))


def test_all_zero_array_yields_all_false():
    data = np.zeros((6, 10, 10), dtype=np.uint8)
    array = zarr.array(data, chunks=(6, 3, 3))
    result = compute_nonempty_mask_from_lowres_array(array)
    assert result.shape == (10, 10)
    assert not result.any()


def test_never_reads_more_than_one_chunk_at_a_time():
    """Reproduces the fallback-to-native-level case: no low-res pyramid
    level exists, so the caller hands this function a large native-
    resolution array. The fix must stream it chunk-by-chunk rather than
    loading it whole."""

    shape = (65, 4096, 4096)
    chunks = (65, 128, 128)
    store = zarr.storage.MemoryStore()
    array = zarr.open_array(
        store=store, mode="w", shape=shape, chunks=chunks, dtype=np.uint8,
        fill_value=0,
    )
    array[:, 0:128, 0:128] = 1  # one occupied chunk; rest stays zero-filled

    max_bytes_read = 0
    chunk_bytes = int(np.prod(chunks))

    class _TrackingArray:
        """Wraps a zarr array, recording the size of every slice read."""

        def __init__(self, inner):
            self._inner = inner
            self.shape = inner.shape
            self.chunks = inner.chunks

        def __getitem__(self, key):
            nonlocal max_bytes_read
            block = self._inner[key]
            max_bytes_read = max(max_bytes_read, block.nbytes)
            return block

    tracked = _TrackingArray(array)
    result = compute_nonempty_mask_from_lowres_array(tracked)

    assert max_bytes_read <= chunk_bytes, (
        f"read {max_bytes_read} bytes at once; expected <= one chunk "
        f"({chunk_bytes} bytes) -- the fix regressed to a whole-array read"
    )
    assert result.shape == (4096, 4096)
    assert result[0:128, 0:128].all()
    assert not result[128:, 128:].any()
