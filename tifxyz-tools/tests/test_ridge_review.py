"""ridge_review package: the input builder, the renderer and the scorer on synthetic data with a known
answer, plus a blindness check on the shipped review page data."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr

RR = Path(__file__).resolve().parents[1] / "ridge_review"
sys.path.insert(0, str(RR))


def _lattice(d, X, Y, Z):
    d.mkdir(parents=True)
    for a, A in zip("xyz", (X, Y, Z), strict=True):
        tifffile.imwrite(d / f"{a}.tif", A.astype(np.float32))
    (d / "meta.json").write_text("{}")


@pytest.fixture
def world(tmp_path):
    # a flat bright sheet at z = 40 (CT 200) and the prediction on it; left half of the lattice on the
    # sheet, right half 8 voxels above it (off-sheet: ridge_hit must flag it)
    ct = np.full((80, 128, 128), 40, np.uint8)
    ct[39:42] = 200
    pred = np.zeros_like(ct)
    pred[40] = 255
    for name, arr in (("ct.zarr", ct), ("pred.zarr", pred)):
        g = zarr.open_group(str(tmp_path / name), mode="w")
        g.create_dataset("0", data=arr, chunks=(40, 64, 64))
    j, i = np.meshgrid(np.arange(30), np.arange(30))
    X = 20.0 + j * 3.0
    Y = 20.0 + i * 3.0
    Z = np.where(j < 15, 40.0, 48.0)
    _lattice(tmp_path / "pre", X, Y, Z)
    keep = j < 15                                             # the guard kept the on-sheet half
    _lattice(tmp_path / "guarded", np.where(keep, X, -1), np.where(keep, Y, -1), np.where(keep, Z, -1))
    jobs = [{"round_id": "r1", "scroll": "SYN", "segment": "SYN_c1", "round": "r1", "pre": str(tmp_path / "pre"),
             "guarded": str(tmp_path / "guarded"), "ct": str(tmp_path / "ct.zarr"), "pred": str(tmp_path / "pred.zarr"),
             "voxel_um": 9.362, "ridge_hit_vox": 3.0, "n_removed": 4, "n_kept": 4, "stratum": "10to50", "source": "synthetic"}]
    (tmp_path / "jobs.json").write_text(json.dumps(jobs))
    return tmp_path


def test_build_render_score_end_to_end(world):
    import build_inputs
    out = world / "out"
    build_inputs.main([str(world / "jobs.json"), str(out)])
    built = json.load(open(out / "built.json"))
    key = json.load(open(out / "key.json"))
    assert len(built) == 8 and {k["class"] for k in key} == {"removed", "kept"}
    assert all("class" not in b and "stratum" not in b for b in built)          # blind: labels only in key.json
    z = np.load(out / "samples" / f"{built[0]['sample_id']}.npz")
    assert z["ct_stack"].shape == (33, 49, 49) and z["pred_stack"].shape == (33, 49, 49)
    cls = {k["sample_id"]: k["class"] for k in key}
    for b in built:                                        # kept cells sit on the sheet, removed ones 8 voxels above
        assert (b["center_xyz"][2] == 40.0) == (cls[b["sample_id"]] == "kept")
        zz = np.load(out / "samples" / f"{b['sample_id']}.npz")
        D, H = 16, 24
        on_sheet_at_cell = zz["ct_stack"][D - 1:D + 2, H, H].max() >= 200
        assert on_sheet_at_cell == (cls[b["sample_id"]] == "kept")
    # render
    import render_pages
    render_pages.main([str(out / "samples"), str(out / "built.json"), "--out", str(world / "site")])
    js = (world / "site" / "samples.js").read_text()
    assert "removed" not in js and "10to50" not in js                          # nothing in the page reveals the answer
    assert (world / "site" / "img" / built[0]["sample_id"] / "ct_u.png").exists()
    # score: a reviewer who calls kept = sheet, removed = not sheet (ridge_hit perfect)
    data = world / "data"
    (data / "samples").mkdir(parents=True)
    for f in (out / "samples").iterdir():
        (data / "samples" / f.name).write_bytes(f.read_bytes())
    (data / "key.json").write_text(json.dumps(key))
    (data / "prevalence.json").write_text(json.dumps([{"source": "production_policy_D", "scroll": "SYN", "stratum": "10to50",
                                                       "rounds": 1, "cells_grown": 900, "cells_kept": 450, "cells_flagged": 450,
                                                       "cells_removed_by_ridge": 450}]))
    labels = {k["sample_id"]: {"label": "sheet" if k["class"] == "kept" else "not_sheet"} for k in key}
    (world / "labels.json").write_text(json.dumps({"reviewer": "t", "labels": labels}))
    res = subprocess.run([sys.executable, str(RR / "score.py"), str(world / "labels.json"), "--data", str(data), "--boot", "200"],
                         capture_output=True, text=True, check=True).stdout
    rep = json.loads(res)
    s = next(x for x in rep["strata"] if x["stratum"] == "10to50")
    assert s["false_removal_share"] == 0.0 and s["miss_share"] == 0.0 and s["FPR"] == 0.0 and s["FNR"] == 0.0
    assert rep["ct_check_vs_human"]["agreement"] == 1.0                       # the CT check sees the sheet too
    # the opposite reviewer: every removed cell was sheet
    labels = {k["sample_id"]: {"label": "sheet"} for k in key}
    (world / "labels.json").write_text(json.dumps({"reviewer": "t", "labels": labels}))
    rep = json.loads(subprocess.run([sys.executable, str(RR / "score.py"), str(world / "labels.json"), "--data", str(data), "--boot", "200"],
                                    capture_output=True, text=True, check=True).stdout)
    s = next(x for x in rep["strata"] if x["stratum"] == "10to50")
    assert s["false_removal_share"] == 1.0 and s["FPR"] == 0.5          # 450 removed + 450 kept, all sheet


def test_shipped_package_is_blind_and_complete():
    js = (RR / "site" / "samples.js").read_text()
    items = json.loads(js[len("window.SAMPLES = "):].rstrip().rstrip(";"))
    key = json.load(open(RR / "data" / "key.json"))
    assert len(items) == len(key) > 0
    assert not any(("class" in it or "stratum" in it) for it in items)
    for it in items[:5]:
        assert (RR / "data" / "samples" / f"{it['sample_id']}.npz").exists()
        assert (RR / "site" / "img" / it["sample_id"] / "ct_xy.png").exists()
