import zarr
import numpy as np

from vesuvius.image_proc.run.zarr_tasks.tasks.recompress import (
    RecompressConfig,
    RecompressTask,
)


def test_run_inplace_recompresses_v3_source_BASELINE_reproduces_issue_1670(tmp_path):
    """Exact reproducer from https://github.com/ScrollPrize/villa/issues/1670,
    run against unmodified main to confirm the reported failure is real."""
    data = np.arange(64, dtype="uint8").reshape(4, 4, 4) + 1
    path = tmp_path / "vol.zarr"
    arr = zarr.open(str(path), mode="w", shape=(4, 4, 4), chunks=(2, 2, 2), dtype="uint8")
    arr[:] = data

    task = RecompressTask(
        RecompressConfig(input_zarr=str(path), output_zarr=None, num_workers=1, inplace=True)
    )
    task.prepare()
    task._run_inplace()

    out = zarr.open(str(path), mode="r")
    assert np.array_equal(out[:], data)
