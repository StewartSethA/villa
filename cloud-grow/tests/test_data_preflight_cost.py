import hashlib
import json
import os

import pytest

from cloud_grow import cost as COST
from cloud_grow import data_fetch as DF
from cloud_grow import preflight as PF
from cloud_grow import cli


@pytest.fixture
def bucket(tmp_path):
    root = tmp_path / "bucket"
    z = root / "PHercT" / "volumes" / "111-9.362um-1.2m-113keV-masked.zarr"
    for rel, n in ((".zattrs", 20), (".zgroup", 10), ("0/.zarray", 30), ("0/0/0/0", 5000), ("1/.zarray", 30), ("1/0/0/0", 3000), ("4/.zarray", 30), ("4/0/0/0", 700)):
        p = z / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(hashlib.md5(rel.encode()).digest() * (n // 16 + 1))
    g = root / "PHercT" / "representations" / "predictions" / "surfaces"
    (g / "111-surface-x-L0-th0.2.normal-grids" / "xy").mkdir(parents=True)
    (g / "111-surface-x-L0-th0.2.normal-grids" / "xy" / "a.grid").write_bytes(b"g" * 100)
    (g / "111-surface-x-L0-th0.2.zarr" / "0").mkdir(parents=True)
    (g / "111-surface-x-L0-th0.2.zarr" / "0" / ".zarray").write_bytes(b"{}")
    return DF.DirSource(str(root))


def test_plan_skips_level0_and_keeps_root_metadata(bucket):
    objs = DF.plan_zarr(bucket, "PHercT/volumes/111-9.362um-1.2m-113keV-masked.zarr", {1, 4})
    rel = sorted(o.key.split(".zarr/")[1] for o in objs)
    assert rel == [".zattrs", ".zgroup", "1/.zarray", "1/0/0/0", "4/.zarray", "4/0/0/0"]      # level 0 (252 GB for 0211) never planned


def test_preview_reports_size_before_any_download(bucket, tmp_path):
    objs = DF.plan_zarr(bucket, "PHercT/volumes/111-9.362um-1.2m-113keV-masked.zarr", {1})
    pv = DF.preview(objs, str(tmp_path / "dest"), "ct")
    assert pv["n_files"] == len(objs) and pv["total_gb"] >= 0 and "fits" in pv and not (tmp_path / "dest").exists()


def test_fetch_markers_resume_and_skip(bucket, tmp_path):
    pre = "PHercT/volumes/111-9.362um-1.2m-113keV-masked.zarr"
    objs = DF.plan_zarr(bucket, pre, {1, 4})
    dest = str(tmp_path / "ct")
    big = next(o for o in objs if o.key.endswith("1/0/0/0"))
    os.makedirs(os.path.dirname(os.path.join(dest, "1/0/0/0")), exist_ok=True)
    with open(os.path.join(dest, "1/0/0/0.part"), "wb") as fh:                 # an interrupted download: first 1000 bytes
        fh.write(open(os.path.join(bucket.root, big.key), "rb").read()[:1000])
    r = DF.fetch_objects(bucket, objs, pre + "/", dest, workers=4)
    assert r["complete"] and r["resumed"] == 1 and r["downloaded"] == len(objs) - 1
    assert os.path.exists(os.path.join(dest, ".cache_complete")) and not os.path.exists(os.path.join(dest, ".fetching"))
    assert open(os.path.join(dest, "1/0/0/0"), "rb").read() == open(os.path.join(bucket.root, big.key), "rb").read()
    r2 = DF.fetch_objects(bucket, objs, pre + "/", dest)
    assert r2["skipped"] == len(objs) and r2["downloaded"] == 0


def test_corrupt_remote_is_detected_not_marked_complete(bucket, tmp_path, monkeypatch):
    pre = "PHercT/volumes/111-9.362um-1.2m-113keV-masked.zarr"
    objs = DF.plan_zarr(bucket, pre, {4})
    bad = [DF.Obj(o.key, o.size, "0" * 32) if o.key.endswith("4/0/0/0") else o for o in objs]     # listing md5 does not match bytes
    monkeypatch.setattr(DF.time, "sleep", lambda s: None)
    r = DF.fetch_objects(bucket, bad, pre + "/", str(tmp_path / "ct"), workers=2)
    assert not r["complete"] and r["failed"] and "verification failed" in r["failed"][0]
    assert not os.path.exists(tmp_path / "ct" / ".cache_complete") and os.path.exists(tmp_path / "ct" / ".fetching")
    assert not os.path.exists(tmp_path / "ct" / "4/0/0/0")                      # a bad file is never kept


def test_truncated_remote_size_mismatch_detected(bucket, tmp_path, monkeypatch):
    pre = "PHercT/volumes/111-9.362um-1.2m-113keV-masked.zarr"
    objs = [DF.Obj(o.key, o.size + 7, o.etag) for o in DF.plan_zarr(bucket, pre, {4})]
    monkeypatch.setattr(DF.time, "sleep", lambda s: None)
    assert not DF.fetch_objects(bucket, objs, pre + "/", str(tmp_path / "ct"))["complete"]


def test_grids_found_by_scan_and_meta_written(bucket, tmp_path):
    assert DF.find_grids_prefix(bucket, "PHercT", "111").endswith(".normal-grids/")
    d = tmp_path / "111-9.362um-1.2m-113keV-masked.zarr"; (d / "0").mkdir(parents=True)
    (d / "0" / ".zarray").write_text(json.dumps({"shape": [10, 20, 30]}))
    assert DF.write_meta(str(d), "PHercT") and json.load(open(d / "meta.json"))["voxelsize"] == 9.362
    assert not DF.write_meta(str(d), "PHercT")                                     # never overwrites


def test_cli_fetch_preview_only_downloads_nothing(bucket, tmp_path, capsys):
    rc = cli.main(["fetch", "--scroll", "PHercT", "--dest", str(tmp_path / "d"), "--from-dir", str(bucket.root), "--what", "ct",
                   "--volume-name", "111-9.362um-1.2m-113keV-masked.zarr", "--scrolls-json", str(_scrolls(tmp_path))])
    assert rc == 0 and not (tmp_path / "d").exists() and "preview" in capsys.readouterr().out


def _scrolls(tmp_path):
    p = tmp_path / "scrolls.json"
    p.write_text(json.dumps({"scrolls": {"PHercT": {"voxel_um": 9.362, "ct_zarr": None, "prediction_zarr": None}}}))
    return p


def test_shipped_scrolls_json_matches_registry_conventions():
    sc = DF.load_scrolls()
    assert sc["PHerc0358"]["volume_id"] == "20250821151737" and sc["PHerc0358"]["voxel_um"] == 9.362
    assert sc["PHerc0358"]["prediction_bytes"] == 18885945395
    assert sc["PHercParis4"]["ct_zarr"] is None and sc["PHercParis4"]["needs_explicit_names"]      # null = unknown, never a guess
    with pytest.raises(DF.FetchError):
        DF.scroll_info("PHercNOPE")


# ---------------------------------------------------------------- preflight / cost
def _ready_cfg(tmp_path, kit, ct_zarr, grids):
    pz = tmp_path / "pred.zarr"; (pz / "0").mkdir(parents=True); (pz / "meta.json").write_text("{}")
    return {"workdir": str(tmp_path), "kit_bin": kit, "ct_zarr": ct_zarr, "prediction_zarr": str(pz), "normal_grids": grids, "voxel_um": 9.362}


def test_preflight_passes_on_a_ready_box(tmp_path, kit, pins, ct_zarr, grids):
    import zarr
    zarr.open(ct_zarr, mode="a")  # levels 1 and 4 exist from the fixture; .zarray present
    r = PF.run(_ready_cfg(tmp_path, kit, ct_zarr, grids), grows=4, cores=8, ram_gb=128, free_gb=500, pins=pins, passmark_st=2696, usd_per_hour=1.28)
    assert r["ok"], [c for c in r["checks"] if c["status"] == "FAIL"]
    assert r["estimate"]["prod_equiv_verified_cm2_h"]["mid"] > 0 and r["estimate"]["usd_per_100_cm2"]["mid"] > 0


def test_preflight_flags_each_failure(tmp_path, kit, pins, ct_zarr, grids):
    cfg = _ready_cfg(tmp_path, kit, ct_zarr, grids)
    st = lambda r, n: next(c["status"] for c in r["checks"] if c["check"] == n)
    r = PF.run(cfg, grows=64, cores=64, ram_gb=100, free_gb=500, pins=pins)             # needs max(64, 3*64+40)=232 GB
    assert st(r, "ram") == "FAIL" and not r["ok"]
    assert st(PF.run(cfg, grows=2, cores=2, ram_gb=64, free_gb=5, pins=pins), "disk") == "FAIL"
    assert st(PF.run(cfg, grows=2, cores=2, ram_gb=64, free_gb=50, pins={"vc_grow_seg_from_seed": {"md5": "0" * 32}, "vc_tifxyz_selfcross": pins["vc_tifxyz_selfcross"]}), "tool:vc_grow_seg_from_seed") == "FAIL"
    bad = dict(cfg, ct_zarr=str(tmp_path / "nope"))
    r = PF.run(bad, grows=2, cores=2, ram_gb=64, free_gb=50, pins=pins)
    assert st(r, "ct_level_1") == "FAIL" and st(r, "ct_level_4") == "FAIL"
    assert st(PF.run(dict(cfg, normal_grids=""), grows=2, cores=2, ram_gb=64, free_gb=50, pins=pins), "normal_grids") == "FAIL"
    assert st(PF.run(dict(cfg, voxel_um=0), grows=2, cores=2, ram_gb=64, free_gb=50, pins=pins), "voxel_um") == "FAIL"


def test_cost_model_reproduces_benchmark_arithmetic():
    e = COST.estimate(2696, 12, 100.0, 1.0)                 # TR PRO 3945WX: ST 2696, 12 physical cores
    assert abs(e["slot_cm2_h_benchmark_workload"] - 68.4) < 0.2                         # k x ST (measured slot 76-82: model is -15 %)
    assert abs(e["machine_cm2_h_benchmark_workload"] - 68.4 * 12 * 1.2) < 1.0
    assert e["prod_equiv_verified_cm2_h"]["low"] < e["prod_equiv_verified_cm2_h"]["mid"] < e["prod_equiv_verified_cm2_h"]["high"]
    assert e["label"].startswith("EXTRAPOLATED")
    assert COST.ram_needed_gb(8) == 64.0 and COST.ram_needed_gb(100) == 340.0
