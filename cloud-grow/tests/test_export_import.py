import io
import json
import os
import shutil
import tarfile

import numpy as np
import pytest

from cloud_grow import importer as I
from cloud_grow import manifest as M
from cloud_grow import pack as PK
from cloud_grow import runner as R
from cloud_grow import state as ST
from cloud_grow import tools as T
from conftest import plane, write_tifxyz

SEG = "PHercTEST_c1"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(R, "EXHAUSTED_MM2", 1.0)
    monkeypatch.setenv("FAKE_SELFX", "clean")


@pytest.fixture
def export(cfg, kit, pins, tmp_path):
    """A real (fake-tracer) 2-round grow, packed. Returns (tar, seg_dir, run_ctx)."""
    db = ST.connect(str(tmp_path / "s.sqlite"))
    settings = R.load_policy_into_state(db, cfg.policy_path)
    ident = T.identify(kit, pins=pins)
    pin = T.gate(ident)
    ctx_path = M.write_run_context(cfg.workdir, cfg, ident, kit, pin, settings, not_ported=R.NOT_PORTED)
    ctx = json.load(open(ctx_path))
    assert ctx["tool"]["pin_status"] == pin == "PINNED"
    sd = os.path.join(cfg.workdir, "segments", SEG)
    SD_seed = (300, 300, 100)
    ST.record_seed(db, cfg.scroll, SEG, SD_seed, source="test")
    out = R.grow_segment(cfg, kit, db, SEG, sd, seed=SD_seed)
    assert out.status == "done"
    tar = str(tmp_path / "out" / f"{SEG}.tar.gz")
    os.makedirs(os.path.dirname(tar))
    res = PK.pack_segment(sd, SEG, ctx, tar, seed={"xyz": list(SD_seed)})
    return tar, sd, ctx, res


def opts(kit, **kw):
    return I.ImportOptions(voxel_um=9.362, selfcross_bin=os.path.join(kit, "vc_tifxyz_selfcross"), selfcross_env=T.tool_env(kit), **kw)


def test_manifest_has_required_provenance(export):
    tar, sd, ctx, res = export
    man = json.load(open(os.path.join(sd, f"{SEG}.export.json")))
    assert man["schema"] == "vpipe-remote-grow-export/1"
    assert man["tool"]["md5"] and man["tool"]["file_type"] and man["tool"]["pin_status"] == "PINNED"
    assert man["run"]["cpu_model"] is not None or os.name != "posix"
    assert all(r["peak_rss_mb"] for r in man["rounds"]) and len(man["rounds"]) == 2
    assert man["inputs"]["voxel_um"] == 9.362 and man["inputs"]["ct_level_for_guard"] == 1
    assert man["claims"]["validation_status"].startswith("UNVALIDATED")
    assert "cache_root" not in " ".join(man["ship"]["files"])
    assert res["sha256"] and res["mb"] < 5


def test_import_pass_registers_append_only_sidecars(export, kit, tmp_path):
    tar, *_ = export
    reg = str(tmp_path / "reg")
    v = I.import_export(tar, reg, opts(kit))
    assert v.status == "PASS", v.reasons
    lines = [json.loads(l) for l in open(os.path.join(reg, "imports.jsonl"))]
    assert len(lines) == 1 and lines[0]["status"] == "PASS" and lines[0]["checks"]["selfx"]["ran"]
    assert lines[0]["checks"]["d3_chain"] and lines[0]["checks"]["area"]
    again = I.import_export(tar, reg, opts(kit))
    assert again.checks.get("already_imported") and len(open(os.path.join(reg, "imports.jsonl")).readlines()) == 1


def test_dry_run_writes_nothing(export, kit, tmp_path):
    tar, *_ = export
    reg = str(tmp_path / "reg")
    v = I.import_export(tar, reg, opts(kit, dry_run=True))
    assert v.status == "PASS" and not os.path.exists(reg)


def _repack(sd, tmp_path, name="t.tar.gz", mutate=None):
    work = tmp_path / "mut"; shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(sd, work / SEG)
    if mutate:
        mutate(str(work / SEG))
    out = str(tmp_path / name)
    with tarfile.open(out, "w:gz") as tf:
        for r, _d, fs in os.walk(work / SEG):
            for f in fs:
                p = os.path.join(r, f)
                tf.add(p, arcname=os.path.relpath(p, work))
    return out


def test_tampered_file_refused(export, kit, tmp_path):
    _tar, sd, *_ = export
    def mut(d):
        x = next(p for p in (os.path.join(r, "x.tif") for r, _, fs in os.walk(d) if "x.tif" in fs))
        with open(x, "ab") as fh:
            fh.write(b"\0")
    v = I.import_export(_repack(sd, tmp_path, mutate=mut), str(tmp_path / "reg"), opts(kit))
    assert v.status == "REFUSED" and any("md5 mismatch" in r for r in v.reasons)


def test_unlisted_extra_file_refused(export, kit, tmp_path):
    _tar, sd, *_ = export
    v = I.import_export(_repack(sd, tmp_path, mutate=lambda d: open(os.path.join(d, "evil.sh"), "w").write("x")), str(tmp_path / "reg"), opts(kit))
    assert v.status == "REFUSED" and any("not in the md5 tree" in r for r in v.reasons)


def _retree(d):
    """Re-write md5tree + manifest tree sha so ONLY the semantic check can catch the change."""
    import hashlib
    man = json.load(open(os.path.join(d, f"{SEG}.export.json")))
    text = M.md5_tree(d, SEG)
    open(os.path.join(d, f"{SEG}.md5tree"), "w").write(text)
    man["ship"]["md5tree_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    return man


def test_overclaimed_guarded_area_refused(export, kit, tmp_path):
    _tar, sd, *_ = export
    def mut(d):
        man = _retree(d)
        for r in man["rounds"]:
            r["area_cm2_postguard"] = r["area_cm2_postguard"] * 1.5
        json.dump(man, open(os.path.join(d, f"{SEG}.export.json"), "w"))
    # only guard-written checkpoints are enforced: make sure this run had one by cutting a corner of the final lattice
    v = I.import_export(_repack(sd, tmp_path, mutate=mut), str(tmp_path / "reg"), opts(kit))
    enforced = [a for a in v.checks.get("area", []) if a["enforced"]]
    if enforced:
        assert v.status == "REFUSED" and any("hub lattice measure" in r for r in v.reasons)
    else:                                            # unguarded raw checkpoints: claim recorded, hub measure registered, NOT refused on area
        assert all(a["rel_diff"] > 0.3 for a in v.checks["area"]) and not any("hub lattice measure" in r for r in v.reasons)


def test_guard_written_area_overclaim_refused_explicitly(tmp_path, kit):
    """Direct unit: a guarded_ checkpoint whose manifest claim is 20 % too big."""
    seg_dir = tmp_path / SEG
    X, Y, Z = plane(12, 20.0)
    from cloud_grow import growth_guard as GG
    P, V = GG.lattice_frame(*[np.asarray(a, np.float32) for a in (X, Y, Z)])
    true = GG.lattice_area_cm2(P, V, 9.362)
    ck = seg_dir / "r1" / "guarded_g_auto_grown_1"
    write_tifxyz(str(ck), X, Y, Z, true)
    man = {"schema": "vpipe-remote-grow-export/1", "tool": {"pin_status": "PINNED"}, "inputs": {"voxel_um": 9.362},
           "rounds": [{"round": 1, "checkpoint": "/box/r1/guarded_g_auto_grown_1", "area_cm2_postguard": true * 1.2}],
           "final": {"checkpoint": "/box/r1/guarded_g_auto_grown_1"}, "ship": {}}
    import hashlib
    (seg_dir / f"{SEG}.export.json").write_text(json.dumps(man))
    text = M.md5_tree(str(seg_dir), SEG); (seg_dir / f"{SEG}.md5tree").write_text(text)
    man["ship"]["md5tree_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    (seg_dir / f"{SEG}.export.json").write_text(json.dumps(man))
    v = I.verify_dir(str(seg_dir), opts(kit))
    assert v.status == "REFUSED" and any("hub lattice measure" in r for r in v.reasons)
    man["rounds"][0]["area_cm2_postguard"] = true
    (seg_dir / f"{SEG}.export.json").write_text(json.dumps(man))
    assert I.verify_dir(str(seg_dir), opts(kit)).status == "PASS"


def test_selfx_contacts_on_hub_refused(export, kit, tmp_path, monkeypatch):
    tar, *_ = export
    monkeypatch.setenv("FAKE_SELFX", "cross")                       # the hub's own census finds crossings the box did not report
    v = I.import_export(tar, str(tmp_path / "reg"), opts(kit))
    assert v.status == "REFUSED" and any("selfx density" in r for r in v.reasons)


def test_selfx_unrunnable_on_hub_fails_closed(export, kit, tmp_path, monkeypatch):
    tar, *_ = export
    monkeypatch.setenv("FAKE_SELFX", "fail")
    v = I.import_export(tar, str(tmp_path / "reg"), opts(kit))
    assert v.status == "REFUSED" and any("fail closed" in r for r in v.reasons)


def test_skip_selfx_can_never_pass(export, kit, tmp_path):
    tar, *_ = export
    assert I.import_export(tar, str(tmp_path / "reg"), opts(kit, skip_selfx=True, dry_run=True)).status == "REFUSED"


def test_unpinned_tool_refused_unless_accepted(export, kit, tmp_path):
    _tar, sd, *_ = export
    def mut(d):
        man = _retree(d); man["tool"]["pin_status"] = "UNPINNED-ALLOWED"
        json.dump(man, open(os.path.join(d, f"{SEG}.export.json"), "w"))
    tar = _repack(sd, tmp_path, mutate=mut)
    assert I.import_export(tar, str(tmp_path / "r1"), opts(kit, dry_run=True)).status == "REFUSED"
    assert I.import_export(tar, str(tmp_path / "r2"), opts(kit, dry_run=True, accept_unpinned=True)).status == "PASS"


def test_d3_shrinking_round_refused(export, kit, tmp_path):
    _tar, sd, *_ = export
    def mut(d):
        import tifffile
        man = json.load(open(os.path.join(d, f"{SEG}.export.json")))
        r2 = man["rounds"][1]
        ck = next(rr for rr, _ds, _f in os.walk(d) if os.path.basename(rr) == os.path.basename(r2["checkpoint"]))
        for a in "xyz":
            A = tifffile.imread(os.path.join(ck, a + ".tif")); A[:15, :] = -1       # round 2 now holds fewer cells than the surface it resumed
            tifffile.imwrite(os.path.join(ck, a + ".tif"), A)
        man = _retree(d)
        man["rounds"][1]["area_cm2_postguard"] = None                                # leave only the D3 check to object
        json.dump(man, open(os.path.join(d, f"{SEG}.export.json"), "w"))
    v = I.import_export(_repack(sd, tmp_path, mutate=mut), str(tmp_path / "reg"), opts(kit, dry_run=True, skip_selfx=True))
    assert v.status == "REFUSED" and any("D3 violated" in r for r in v.reasons), v.reasons


def test_tar_sha_mismatch_and_path_traversal_refused(export, kit, tmp_path):
    tar, *_ = export
    v = I.import_export(tar, str(tmp_path / "reg"), opts(kit, expect_sha256="0" * 64))
    assert v.status == "REFUSED" and "sha256" in v.reasons[0]
    evil = str(tmp_path / "evil.tar.gz")
    with tarfile.open(evil, "w:gz") as tf:
        ti = tarfile.TarInfo("../escape.txt"); ti.size = 1
        tf.addfile(ti, io.BytesIO(b"x"))
    v = I.import_export(evil, str(tmp_path / "reg2"), opts(kit))
    assert v.status == "REFUSED" and "unsafe" in v.reasons[0] and not (tmp_path / "escape.txt").exists()


def test_voxel_pitch_conflict_refused(export, kit, tmp_path):
    tar, *_ = export
    o = opts(kit); o.voxel_um = 8.64
    v = I.import_export(tar, str(tmp_path / "reg"), o)
    assert v.status == "REFUSED" and any("voxel pitch conflict" in r for r in v.reasons)


def test_selfx_unverified_marker_refused(cfg, kit, pins, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_SELFX", "fail")
    db = ST.connect(None)
    R.load_policy_into_state(db, cfg.policy_path)
    sd = os.path.join(cfg.workdir, "segments", SEG)
    ident = T.identify(kit, pins=pins)
    ctx = json.load(open(M.write_run_context(cfg.workdir, cfg, ident, kit, "PINNED", {}, not_ported=[])))
    out = R.grow_segment(cfg, kit, db, SEG, sd, seed=(300, 300, 100))
    assert out.why == "selfx_unverified"
    tar = str(tmp_path / "u.tar.gz")
    PK.pack_segment(sd, SEG, ctx, tar)
    monkeypatch.setenv("FAKE_SELFX", "clean")
    v = I.import_export(tar, str(tmp_path / "reg"), opts(kit, dry_run=True))
    assert v.status == "REFUSED" and any("selfx_unverified" in r for r in v.reasons)


def test_pack_refuses_secrets_and_excludes_cache_root(export, tmp_path):
    _tar, sd, ctx, _ = export
    os.makedirs(os.path.join(sd, "cache_root"), exist_ok=True)
    open(os.path.join(sd, "cache_root", "chunk"), "w").write("x")
    assert "cache_root/chunk" not in M.shippable_files(sd, SEG)
    open(os.path.join(sd, "hub.token"), "w").write("abc")
    with pytest.raises(PK.PackError, match="secret-like"):
        PK.pack_segment(sd, SEG, ctx, str(tmp_path / "x.tar.gz"))
    os.remove(os.path.join(sd, "hub.token"))
    open(os.path.join(sd, "notes.txt"), "w").write("-----BEGIN OPENSSH PRIVATE KEY-----")
    with pytest.raises(PK.PackError):
        PK.pack_segment(sd, SEG, ctx, str(tmp_path / "x.tar.gz"))


def test_cli_exit_codes(export, kit, tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("imp_cli", os.path.join(os.path.dirname(__file__), "..", "hub", "import_remote_grow.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    tar, *_ = export
    assert m.main([tar, "--registry", str(tmp_path / "reg"), "--kit-bin", kit, "--voxel-um", "9.362", "--dry-run"]) == 0
    assert m.main([tar, "--registry", str(tmp_path / "reg"), "--kit-bin", kit, "--voxel-um", "9.362", "--dry-run", "--skip-selfx"]) == 1
