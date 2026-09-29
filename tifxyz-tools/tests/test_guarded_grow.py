"""guarded_grow driver, end to end, with a FAKE tracer (a script standing in for
`vc_grow_seg_from_seed`) so the loop runs without a VC3D build: rounds, resume from the guard's
trimmed checkpoint, stop reasons, the newest-checkpoint rule, and result.json provenance."""
import json
import os
import stat
import sys
import time

import numpy as np
import pytest
import zarr

from tifxyz_tools import guarded_grow as GGR

FAKE_TRACER = r'''#!{py}
import json, os, sys, time
import numpy as np, tifffile
a = sys.argv[1:]
tgt = a[a.index("-t") + 1]
params = json.load(open(a[a.index("--params") + 1]))
g = int(params["generations"])
n = 2 * g + 4                               # the sheet grows with the generation count
j, i = np.meshgrid(np.arange(n), np.arange(n))
X = (100.0 + j * 10.0).astype(np.float32)
Y = (100.0 + i * 10.0).astype(np.float32)
Z = np.full((n, n), 60.0, np.float32)
os.makedirs(tgt, exist_ok=True)
d = os.path.join(tgt, "auto_grown_%d" % time.time_ns())
os.makedirs(d)
for k, A in zip("xyz", (X, Y, Z)):
    tifffile.imwrite(os.path.join(d, k + ".tif"), A)
vox = float(params["voxelsize"])
area = (n - 1) ** 2 * (10 * vox * 1e-4) ** 2
json.dump({{"area_cm2": area, "max_gen": g, "scale": [0.05, 0.05], "format": "tifxyz", "type": "seg",
           "uuid": os.path.basename(d)}}, open(os.path.join(d, "meta.json"), "w"))
'''


@pytest.fixture
def world(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    t = bin_dir / "vc_grow_seg_from_seed"
    t.write_text(FAKE_TRACER.format(py=sys.executable))
    t.chmod(t.stat().st_mode | stat.S_IEXEC)
    vol = tmp_path / "vol.zarr"
    g = zarr.open_group(str(vol), mode="w")
    ct = np.full((64, 512, 512), 100, np.uint8)       # (z, y, x) at level 1 = 2 level-0 voxels per index
    ct[:, :, 500 // 2:] = 0                           # air beyond x = 500 (level-0 voxels)
    g.create_dataset("1", data=ct, chunks=(32, 64, 64))
    (vol / "meta.json").write_text(json.dumps({"voxelsize": 9.362}))
    grids = tmp_path / "grids"
    grids.mkdir()
    return tmp_path, str(bin_dir), str(vol), str(grids)


def _args(tmp, bin_dir, vol, grids, *extra):
    return ["--volume", vol, "--normal-grids", grids, "--seed", "110", "110", "60",
            "--out", str(tmp / "out"), "--vc-bin", bin_dir, *extra]


def test_unguarded_rounds_resume_and_stop_when_exhausted(world):
    tmp, b, vol, grids = world
    assert GGR.main(_args(tmp, b, vol, grids, "--max-rounds", "3")) == 0
    res = json.loads((tmp / "out" / "result.json").read_text())
    # round 2 adds 1,320 cells of 93.6 um = 11.6 mm2 < 50 mm2: exhausted
    assert res["rounds"] == 2 and res["stop_reason"].startswith("exhausted")
    assert res["provenance"]["vc_grow_seg_from_seed_md5"]            # the binary is identified
    lines = [json.loads(x) for x in (tmp / "out" / "guard_report.jsonl").read_text().splitlines()]
    assert [x["gen_to"] for x in lines] == [10, 20]              # --seed, then --resume +10
    assert "--resume" in (tmp / "out" / "round2.log").read_text()


def test_guard_trims_vacuum_and_resumes_from_the_trimmed_checkpoint(world, tmp_path):
    tmp, b, vol, grids = world
    pol = tmp_path / "vac.json"
    pol.write_text(json.dumps({"policy": {"enabled": True, "selfcross": False, "fold": False, "plan": False,
                                          "overlap": False, "selfcross_hairpin_abort_ratio": 0,
                                          "regrow_block_frac": 2.0}}))
    GGR.main(_args(tmp, b, vol, grids, "--policy", str(pol), "--max-rounds", "3", "--ct", vol, "--exhausted-mm2", "0"))
    lines = [json.loads(x) for x in (tmp / "out" / "guard_report.jsonl").read_text().splitlines()]
    cut = [x for x in lines if x["guard"]["cells_after"] < x["guard"]["cells_before"]]
    assert cut, "the planted air region was never trimmed"
    assert all(x["guard"]["pruned_by"].get("vacuum", 0) > 0 for x in cut)
    # the round after a trim resumes FROM the trimmed checkpoint
    k = lines.index(cut[0])
    if k + 1 < len(lines):
        log = (tmp / "out" / f"round{k + 2}.log").read_text()
        assert "guarded_g_" in log
    res = json.loads((tmp / "out" / "result.json").read_text())
    assert res["policy"]["enabled"] is True and res["guard_s"] > 0


def test_nothing_left_reports_zero_area_and_failed(world, tmp_path):
    tmp, b, vol, grids = world
    root = zarr.open_group(vol, mode="r+")
    root["1"][:] = 0                                   # everything is air: the guard cuts it all
    pol = tmp_path / "all.json"
    pol.write_text(json.dumps({"policy": {"enabled": True, "selfcross": False, "selfcross_hairpin_abort_ratio": 0}}))
    rc = GGR.main(_args(tmp, b, vol, grids, "--policy", str(pol), "--max-rounds", "3", "--ct", vol))
    res = json.loads((tmp / "out" / "result.json").read_text())
    assert rc == 1 and res["status"] == "failed" and res["stop_reason"] == "guard_nothing_left"
    assert res["area_cm2"] == 0.0                      # never the pre-trim area (a real past bug)


def test_newest_checkpoint_is_by_time_not_name(tmp_path):
    for name in ("guarded_g_auto_grown_1", "auto_grown_2"):
        d = tmp_path / name
        d.mkdir()
        (d / "meta.json").write_text("{}")
        time.sleep(0.02)
    assert os.path.basename(GGR.newest_checkpoint(str(tmp_path))) == "auto_grown_2"


def test_missing_voxel_size_is_an_error_not_a_default(tmp_path):
    with pytest.raises(ValueError):
        GGR.voxel_um_of(str(tmp_path), None)
