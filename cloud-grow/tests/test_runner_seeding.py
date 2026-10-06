import json
import os

import numpy as np
import pytest

from cloud_grow import growth_guard as GG
from cloud_grow import runner as R
from cloud_grow import seeding as SD
from cloud_grow import state as ST
from conftest import plane, write_tifxyz

SEED = (300, 300, 100)


@pytest.fixture(autouse=True)
def _small_fake_growth(monkeypatch):
    monkeypatch.setattr(R, "EXHAUSTED_MM2", 1.0)       # the fake tracer's rings gain ~9 mm2 each; production's floor is 50


def test_exhaustion_rule_at_production_floor(cfg, kit, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "EXHAUSTED_MM2", 50.0)
    monkeypatch.setenv("FAKE_SELFX", "clean")
    _db, out = _grow(cfg, kit, tmp_path)
    assert out.status == "exhausted" and "mm2" in out.why


def _grow(cfg, kit, tmp_path, seg="PHercTEST_c1", **kw):
    db = ST.connect(str(tmp_path / "s.sqlite"))
    R.load_policy_into_state(db, cfg.policy_path)
    out = R.grow_segment(cfg, kit, db, seg, str(tmp_path / "segs" / seg), seed=SEED, **kw)
    return db, out


def test_grow_two_rounds_with_guard_records_rss_and_state(cfg, kit, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_SELFX", "clean")
    db, out = _grow(cfg, kit, tmp_path)
    assert out.status == "done" and out.why.startswith("round_target"), (out.status, out.why)
    rj = [json.loads(l) for l in open(tmp_path / "segs" / "PHercTEST_c1" / "rounds.jsonl")]
    assert [r["round"] for r in rj] == [1, 2]
    assert rj[0]["gen_to"] == 6 and rj[1]["gen_from"] == 6 and rj[1]["gen_to"] == 10          # resume continues the generations
    assert rj[1]["resume_from"] == rj[0]["checkpoint"]                                          # D3: round 2 resumes round 1's surface
    assert all(r["peak_rss_mb"] and r["peak_rss_mb"] > 0 for r in rj)                          # NULL in every production attempt row
    assert rj[0]["guard_summary"]["cells_before"] > 0 and rj[0]["guard_inputs_available"]["ct_sampler"]
    assert rj[0]["guard_inputs_available"]["neighbour_cover"] is False                         # announced, not scored clean
    assert rj[1]["area_cm2_postguard"] >= rj[0]["area_cm2_postguard"]
    assert ST.latest_metric(db, "PHercTEST_c1", "verified_cm2") > 0
    assert db.execute("SELECT count(*) FROM artifact").fetchone()[0] >= 2


def test_d3_rerun_resumes_and_never_regrows(cfg, kit, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_SELFX", "clean")
    _db, first = _grow(cfg, kit, tmp_path)
    db, second = _grow(cfg, kit, tmp_path)
    assert second.area_cm2 >= first.area_cm2                                                    # output >= input
    rj = [json.loads(l) for l in open(tmp_path / "segs" / "PHercTEST_c1" / "rounds.jsonl")]
    assert [r["round"] for r in rj] == [1, 2, 3, 4] and rj[2]["resume_from"] is not None


def test_resume_from_non_checkpoint_refused(cfg, kit, tmp_path):
    db = ST.connect(None)
    with pytest.raises(RuntimeError, match="not a grown tifxyz"):
        R.grow_segment(cfg, kit, db, "s", str(tmp_path / "o"), resume=str(tmp_path / "nothing"))


def test_selfx_cannot_run_fails_closed_and_pauses(cfg, kit, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_SELFX", "fail")
    monkeypatch.setenv("CLOUD_GROW_ALERT_LOG", str(tmp_path / "alerts.jsonl"))
    db, out = _grow(cfg, kit, tmp_path)
    assert out.status == "paused" and out.why == "selfx_unverified"
    assert ST.is_paused(db, "PHercTEST_c1", "grow")
    assert (tmp_path / "alerts.jsonl").exists()
    # round 1 of a fresh seed has no verified surface: the raw one is marked so the importer refuses it
    assert os.path.exists(os.path.join(out.checkpoint, "selfx_unverified.json"))


def test_guard_exception_fails_closed_not_open(cfg, kit, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_SELFX", "clean")
    def boom(*a, **k):
        raise OSError("zarr EIO")
    monkeypatch.setattr(R.GG, "guard_round", boom)                 # 92 guard_error rows in 7 days of production
    db, out = _grow(cfg, kit, tmp_path)
    assert out.status == "paused" and out.why == "selfx_unverified"
    assert ST.latest_metric(db, "PHercTEST_c1", "guard_error").startswith("OSError")
    rj = [json.loads(l) for l in open(tmp_path / "segs" / "PHercTEST_c1" / "rounds.jsonl")]
    assert rj[-1]["checkpoint"] and "guard_error" in rj[-1]


def test_tracer_failure_reported_not_swallowed(cfg, kit, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_TRACER", "fail")
    _db, out = _grow(cfg, kit, tmp_path)
    assert out.status == "failed" and out.why == "exit_3"


def test_missing_inputs_fail_loudly(cfg, kit, tmp_path):
    cfg.normal_grids = str(tmp_path / "absent")
    _db, out = _grow(cfg, kit, tmp_path)
    assert out.status == "failed" and "missing local input" in out.why


def test_newest_checkpoint_orders_by_mtime_not_name(tmp_path):
    t = tmp_path / "r1"
    for name, mt in (("guarded_g_old", 1000), ("auto_grown_new", 2000)):
        d = t / name; d.mkdir(parents=True); (d / "meta.json").write_text("{}")
        os.utime(d / "meta.json", (mt, mt))
    assert os.path.basename(R.newest_checkpoint(str(t))) == "auto_grown_new"   # 'g' > 'a' by name: the 2026-09-29 bug


def test_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "c.json"; p.write_text(json.dumps({"scroll": "X", "workdir": "w", "typo_key": 1}))
    with pytest.raises(ValueError, match="unknown config key"):
        R.RunConfig.load(str(p))


# ---------------------------------------------------------------- seeding
def _field(shape=(40, 40, 40)):
    sup = np.zeros(shape, bool); sup[5:35, 5:35, 5:35] = True
    return sup


def test_seeds_avoid_coverage_and_edges_and_respect_separation():
    sup = _field()
    cov = np.zeros_like(sup); cov[:, :, :20] = True
    shape0 = tuple(s * SD.FACTOR for s in sup.shape)
    seeds = SD.propose(sup, cov, shape0, count=6, min_sep=100.0, rng_seed=3)
    assert len(seeds) == 6
    for s in seeds:
        assert s.x // SD.FACTOR >= 20 + SD.RADIUS_L4 - 1                                        # never inside coverage (dist >= 1)
        assert all(SD.EDGE_MARGIN_VOX <= v < n - SD.EDGE_MARGIN_VOX for v, n in zip((s.z, s.y, s.x), shape0))
    for i, a in enumerate(seeds):
        for b in seeds[i + 1:]:
            assert (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2 >= 100 ** 2


def test_seeding_returns_nothing_when_everything_covered_or_unsupported():
    sup = _field()
    assert SD.propose(sup, np.ones_like(sup), (640,) * 3, count=3, rng_seed=1) == []
    assert SD.propose(np.zeros_like(sup), np.zeros_like(sup), (640,) * 3, count=3, rng_seed=1) == []


def test_seed_verification_rejects_gap_seeds():
    sup = _field(); cov = np.zeros_like(sup)
    assert SD.propose(sup, cov, (640,) * 3, count=2, rng_seed=1, verify=lambda x, y, z: (0.0, 100.0)) == []     # centre in a gap (the n1615 case)
    assert len(SD.propose(sup, cov, (640,) * 3, count=2, rng_seed=1, verify=lambda x, y, z: (200.0, 200.0))) == 2


def test_coverage_mask_marks_grown_surface(tmp_path):
    X, Y, Z = plane(20, pitch=20, x0=320, y0=320, z=320)
    write_tifxyz(str(tmp_path / "ck"), X, Y, Z, 1.0)
    cov = SD.coverage_mask((60, 60, 60), [str(tmp_path / "ck")])
    assert cov[20, 20:25, 20:25].any() and not cov[2, 2, 2]
    assert SD.distance(cov)[20, 22, 22] == 0.0
    assert cov[20, 22, 44] and not cov[20, 22, 46]               # last marked cell is 42; dilated by RADIUS_L4 (2) to 44


def test_reserve_records_provenance(tmp_path):
    db = ST.connect(None)
    seg = SD.reserve(db, "PHercTEST", SD.Seed(1, 2, 3, 0.5, 4.0, 200.0, 190.0))
    assert ST.seed_xyz_of(db, seg) == (1, 2, 3)
    prov = json.loads(db.execute("SELECT provenance_json FROM seed").fetchone()[0])
    assert prov["verify"] == "ct_level1_window7" and prov["pred_th"] == 180
