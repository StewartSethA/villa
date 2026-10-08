import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from routeB import box8 as B8   # noqa: E402
from routeB import ladder as L  # noqa: E402

CFG = L.load_config(ladder="all")


def job(scroll="PHerc0191", rung=0, z0=4500, z1=17500):
    return L.make_job(CFG, scroll, 146, rung, z0, z1)


# ---------------------------------------------------------------- pure ladder
def test_classify():
    assert L.classify("RuntimeError: number of categories cannot exceed 2^24", 1) == "multinomial"
    assert L.classify("torch.OutOfMemoryError: CUDA out of memory", 1) == "oom"
    assert L.classify("ModuleNotFoundError: x", 1) == "env"
    assert L.classify("whatever", -9) == "host_oom"
    assert L.classify("whatever", 1, stalled=True) == "stall"
    assert L.classify("weird", 1) == "unknown"


def test_multinomial_jumps_to_rung_under_limit_not_next_rung():
    j = job()
    d = L.decide(CFG, j, "multinomial", 146)
    assert d["action"] == "descend" and d["to_rung"] == "q4500"           # 22M tracks: full and w13000 both over 16.78M; 4500 -> 7.6M
    assert [(k["z0"], k["z1"]) for k in d["jobs"]] == [(4500, 9000), (7500, 12000), (10500, 15000), (13500, 17500)]
    assert d["jobs"][0]["tag"] == "q4500" and d["jobs"][0]["id"] == "PHerc0191/q4500"
    assert j["oom_n"] == 0                                                   # no memory retry wasted


def test_multinomial_without_known_tracks_uses_gb_proxy_then_next_rung():
    j = job("PHerc0211")
    d = L.decide(CFG, j, "multinomial", 115, dbm_bytes=7.485e9)             # proxy: 16.6M tracks < limit on 'full' span -> w13000 is not skipped by the proxy
    assert d["action"] == "descend" and d["to_rung"] == "w13000"


def test_oom_retries_then_descends():
    j = job("PHerc0211")
    d1 = L.decide(CFG, j, "oom", 115)
    d2 = L.decide(CFG, j, "oom", 115)
    assert (d1["action"], d1["extra"]) == ("retry", {"sample_count_tracks_per_step": 16000})
    assert d1["fresh"] is False                                             # tracks-only change: resume from the checkpoint
    assert d2["action"] == "descend" and d2["to_rung"] == "w13000"


def test_flow_grid_change_is_a_fresh_start():
    cfg = L.load_config(ladder="all")
    cfg["attempts"]["oom_retries_in_rung"] = 2
    j = L.make_job(cfg, "PHerc0211", 115, 0, 4500, 17500)
    L.decide(cfg, j, "oom", 115)
    d = L.decide(cfg, j, "oom", 115)
    assert d["action"] == "retry" and d["fresh"] is True and d["extra"]["model_flow_voxel_resolution"] == 48


def test_ladder_walks_to_2800_and_exhausts():
    j = job("PHerc0211", rung=2, z0=4500, z1=9000)
    L.decide(CFG, j, "oom", 115)
    d = L.decide(CFG, j, "oom", 115)
    assert d["to_rung"] == "sw2800"
    assert [(k["z0"], k["z1"]) for k in d["jobs"]] == [(4500, 7300), (7100, 9000)]
    k = d["jobs"][0]
    L.decide(CFG, k, "oom", 115)
    assert L.decide(CFG, k, "oom", 115)["action"] == "fail"                 # nothing narrower than 2800


def test_env_and_tiles_are_terminal_and_unknown_retries_once():
    assert L.decide(CFG, job(), "env", 146)["action"] == "fail"
    j = job()
    assert L.decide(CFG, j, "unknown", 146)["action"] == "retry"
    assert L.decide(CFG, j, "unknown", 146)["action"] == "fail"


def test_stall_one_resume_then_descend():
    j = job("PHerc0211")
    assert L.decide(CFG, j, "stall", 115)["action"] == "retry"
    assert L.decide(CFG, j, "stall", 115)["action"] == "descend"


def test_first_rung_preskip_from_known_tracks_only():
    assert L.first_rung(CFG, "PHerc0191", 4500, 17500)[0] == 2              # known 22M -> q4500
    assert L.first_rung(CFG, "PHerc0125", 4500, 17500)[0] == 0              # unknown count -> never pre-skipped by the GB proxy (0125 completed full)


def test_parse_progress():
    assert B8.parse_progress("x\nPROGRESS Optimizing — 1,489/1,500 iterations (99.3%) — 1.5 it/s") == (1489, 1500)
    assert B8.parse_progress("PROGRESS Loading tracks — 1/2 DB keys") is None


def test_skip_reasons():
    assert "no published tracks" in B8.skip_reason("PHerc0846B")
    assert "lasagna" in B8.skip_reason("PHerc0175A")
    assert "sense" in B8.skip_reason("PHerc0846A") or "lasagna" in B8.skip_reason("PHerc0846A")
    assert B8.skip_reason("PHerc0211") is None


# ---------------------------------------------------------------- scheduler with stub jobs
def make_sched(tmp_path, monkeypatch, scrolls, gpus, plan, soft=45.0, hard=49.0, clock_scale=None, sleep="1.0", extra=()):
    monkeypatch.setenv("ROUTEB_HOME", str(tmp_path))
    monkeypatch.setenv("STUB_PLAN", json.dumps(plan))
    monkeypatch.setenv("STUB_SLEEP", sleep)
    a = B8.build_parser().parse_args(["--scrolls", ",".join(scrolls), "--fake-gpus", str(gpus), "--poll-s", "0.2", "--steps", "10", *extra])
    a.stall_minutes = 0.05 if "stall" in json.dumps(plan) else 30.0
    B = B8._budget()
    clk = B.ScaledClock(clock_scale) if clock_scale else time.time
    gov = B.Governor(tmp_path / "box8" / "budget", B.Rates(soft_usd=soft, hard_usd=hard), clock=clk, box_start=clk(), say=lambda *_: None)
    s = B8.Scheduler(a, CFG, gov, fetch_fn=lambda sc: 0.1, job_cmd=[sys.executable, str(HERE / "stub_job.py")], box_home=tmp_path)
    for sc in scrolls:
        s.add_scroll(sc, 4500, 17500, None)
    s.gpus = B8.discover_gpus(None, gpus)
    return s


def test_work_stealing_keeps_all_gpus_busy_and_ladder_records_rung(tmp_path, monkeypatch):
    # 0191 is pre-skipped to q4500 (4 stripes); 0211 full fails with multinomial (proxy) -> w13000 fails OOM x3 -> q4500 ...; 0125 ok on full
    plan = {"full": []}
    s = make_sched(tmp_path, monkeypatch, ["PHerc0191", "PHerc0125", "PHerc0211"], 3, {"full13k": ["oom", "oom", "oom"]})
    s.jobs["PHerc0211/full"]["extra_overrides"] = {}
    # make 0211's full job fail deterministically: edit the plan after the fact
    monkeypatch.setenv("STUB_PLAN", json.dumps({"full": ["multinomial"], "full13k": ["oom", "oom", "oom"]}))
    # 0125 'full' would also hit the plan: give it a distinct tag by running it with the same plan -> it fails once too; that is fine and tested below
    t0 = time.time()
    s.run()
    ev = [json.loads(x) for x in (tmp_path / "box8" / "events.jsonl").read_text().splitlines()]
    st191 = json.loads((tmp_path / "box8" / "state" / "PHerc0191.json").read_text())
    assert st191["rung_succeeded"] == {"q4500": "q4500", "q7500": "q4500", "q10500": "q4500", "q13500": "q4500"}
    st211 = json.loads((tmp_path / "box8" / "state" / "PHerc0211.json").read_text())
    assert set(st211["rung_succeeded"].values()) == {"q4500"}               # full -> (multinomial; proxy: w13000 not skipped) -> oom x3 -> q4500
    for sc in ("PHerc0191", "PHerc0211", "PHerc0125"):
        pd = tmp_path / "box8" / "payload" / sc
        assert (pd / "DONE").exists(), sc
        meta = json.loads((pd / "PAYLOAD.json").read_text())
        assert meta["status"] == "complete" and not any(e["path"].endswith(".ckpt") for e in meta["files"])
        from routeB.box8 import md5_file
        assert all(md5_file(pd / e["path"]) == e["md5"] for e in meta["files"])
        assert all(sg["tile_name_stem"].startswith(f"{sc}_") for sg in meta["segments"])
    # concurrency: the stub log must show 3 jobs overlapping at some instant
    lines = [x.split() for x in (tmp_path / "stub_events.log").read_text().splitlines()]
    starts = sorted(float(x[0]) for x in lines)
    assert max(sum(1 for t2 in starts if t <= t2 < t + 0.9) for t in starts) >= 3


def test_budget_refusal_and_hard_stop(tmp_path, monkeypatch):
    # 1 s real = 1 simulated hour; soft cap $6 -> after ~2 h no new launch
    s = make_sched(tmp_path, monkeypatch, ["PHerc0211", "PHerc0125", "PHerc0191", "PHerc0257"], 1, {}, soft=6.0, hard=49.0, clock_scale=3600.0, sleep="0.5")
    s.run()
    sm = json.loads((tmp_path / "box8" / "ALLDONE.json").read_text())
    assert sm["jobs"].get("deferred", 0) >= 1 and sm["jobs"].get("done", 0) >= 0
    ev = [json.loads(x) for x in (tmp_path / "box8" / "events.jsonl").read_text().splitlines()]
    assert any(e["kind"] == "defer" for e in ev)


def test_hard_stop_kills_running(tmp_path, monkeypatch):
    s = make_sched(tmp_path, monkeypatch, ["PHerc0211"], 1, {"full": ["stall"]}, soft=1000.0, hard=3.0, clock_scale=3600.0, sleep="0.2")
    t0 = time.time()
    s.run()
    assert time.time() - t0 < 60 and s.stop_reason and "HARD BUDGET STOP" in s.stop_reason
    assert s.jobs["PHerc0211/full"]["status"] == "hard_stopped"


def test_watchdog_stall_then_resume(tmp_path, monkeypatch):
    s = make_sched(tmp_path, monkeypatch, ["PHerc0211"], 1, {"full": ["stall"]})
    s.run()
    j = s.jobs["PHerc0211/full"]
    assert j["status"] == "done" and [a["class"] for a in j["attempts"]] == ["stall", "ok"]


def test_resume_from_state(tmp_path, monkeypatch):
    s = make_sched(tmp_path, monkeypatch, ["PHerc0211"], 1, {"full": ["env"]})
    s.run()
    assert s.jobs["PHerc0211/full"]["status"] == "failed"                   # env fault is terminal: loud, no descent
    # rerun with a healthy plan: the failed job stays failed (state is truth), a pending one would continue
    s2 = make_sched(tmp_path, monkeypatch, ["PHerc0211"], 1, {})
    assert s2.jobs["PHerc0211/full"]["status"] == "failed"


def test_order_spt_and_lpt(tmp_path, monkeypatch, capsys):
    import re
    for order in ("spt", "lpt"):
        monkeypatch.setenv("ROUTEB_HOME", str(tmp_path / order))
        assert B8.main(["--scrolls", "PHerc0125,PHerc0358,PHerc0268,PHerc0826", "--fake-gpus", "2", "--dry-run", "--order", order]) == 0
        tot, seq = {}, []
        for x in capsys.readouterr().out.splitlines():
            m = re.search(r"plan:\s+(PHerc\d+\w?)/\S+.*expected ([\d.]+) GPU-h", x)
            if m:
                tot[m.group(1)] = tot.get(m.group(1), 0) + float(m.group(2))
                if m.group(1) not in seq:
                    seq.append(m.group(1))
        vals = [tot[s] for s in seq]
        assert len(seq) == 4 and len(set(vals)) > 1
        assert vals == sorted(vals, reverse=(order == "lpt")), (order, seq, vals)


def test_smoke_and_short_runs_are_not_budgeted_as_full_fits(tmp_path, monkeypatch):
    s = make_sched(tmp_path, monkeypatch, ["PHerc0211"], 1, {}, extra=("--smoke",))
    assert s.jobs["PHerc0211/full"]["expected_h"] == 0.9
    s2 = make_sched(tmp_path / "b", monkeypatch, ["PHerc0211"], 1, {})           # --steps 10 from make_sched
    assert s2.jobs["PHerc0211/full"]["expected_h"] < 0.3


def test_dry_run_writes_no_state_and_resume_rederives_expected_h(tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTEB_HOME", str(tmp_path))
    assert B8.main(["--scrolls", "PHerc0211", "--fake-gpus", "1", "--dry-run", "--smoke"]) == 0
    assert not list((tmp_path / "box8" / "state").glob("*.json")) and not (tmp_path / "box8" / "events.jsonl").exists()
    # a state file written under full-fit pricing must not keep that price when resumed as a smoke
    s = make_sched(tmp_path / "r", monkeypatch, ["PHerc0211"], 1, {}, extra=())
    s.jobs["PHerc0211/full"]["expected_h"] = 5.5
    s.save("PHerc0211")
    s2 = make_sched(tmp_path / "r", monkeypatch, ["PHerc0211"], 1, {}, extra=("--smoke",))
    assert s2.jobs["PHerc0211/full"]["expected_h"] == 0.9


def test_default_ladder_is_full_then_2800_and_0191_goes_straight_to_2800():
    cfg = L.load_config()
    assert [r["name"] for r in cfg["rungs"]] == ["full", "sw2800"]
    i, why = L.first_rung(cfg, "PHerc0191", 4500, 17500)
    assert cfg["rungs"][i]["name"] == "sw2800"
    jobs = L.plan(cfg, "PHerc0191", 146, 4500, 17500, start=i)
    assert len(jobs) == 5 and jobs[0]["tag"] == "sw4500" and abs(jobs[0]["expected_h"] - 9.43 / 5) < 1e-6      # plan-agent p50 / n_stripes


def test_detect_early_kills_on_too_many_tracks_and_descends(tmp_path, monkeypatch):
    # a stub whose fit.log says 20M tracks loaded: the scheduler must kill it (the stub would sleep 600 s) and descend to sw2800 on the loaded-tracks evidence
    s = make_sched(tmp_path, monkeypatch, ["PHerc0211"], 1, {"full": ["loaded20m"]}, extra=("--ladder", "full,sw2800"))
    s.cfg = L.load_config(ladder="full,sw2800")
    t0 = time.time()
    s.run()
    assert time.time() - t0 < 60
    j = s.jobs["PHerc0211/full"]
    assert j["status"] == "descended" and j["attempts"][0]["class"] == "multinomial" and j["n_loaded"] == 20_000_000
    assert any(k.startswith("PHerc0211/sw") and v["status"] == "done" for k, v in s.jobs.items())


# ---------------------------------------------------------------- computed stripe height
def test_computed_height_matches_measurements(tmp_path):
    h16 = L.computed_height(15.58)                      # 4060 Ti: 13,000 must be refused, 2,800 must be allowed
    assert 2800 <= h16 < 13000 and h16 % 100 == 0
    assert L.computed_height(32.0) >= 13000 or L.computed_height(32.0, span=13000) == 13000 or L.computed_height(32.0) > 11000
    assert L.computed_height(4.0) == 1000               # floor
    assert L.computed_height(80.0, span=13000) == 13000  # capped at the span


def test_shrink_and_memory(tmp_path):
    assert L.shrink_height(4500) == 3300
    assert L.shrink_height(1200) is None
    p = tmp_path / "h.json"
    assert L.start_height(p, "PHerc0191", 15.58, 13000) == L.computed_height(15.58, span=13000)
    L.record_height(p, "PHerc0191", 2800, 15.58)
    assert L.start_height(p, "PHerc0191", 15.58, 13000) == 2800
    assert L.start_height(p, "PHerc0125", 15.58, 13000) == 2800     # next scroll starts from the last success
    assert L.start_height(p, "PHerc0125", 32.0, 13000) == L.computed_height(32.0, span=13000)   # different card size: model, not memory
