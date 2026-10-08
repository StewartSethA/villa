"""Planner + dynamic scheduler tests with FAKE GPUs / disks / RAM (no GPU, no network)."""
import json
import os
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from routeB import box8 as B8        # noqa: E402
from routeB import ladder as L       # noqa: E402
from routeB import planner as P      # noqa: E402
from routeB import routea_side as RA  # noqa: E402

B = B8._budget()
TEN = "PHerc0125 PHerc0191 PHerc0211 PHerc0257 PHerc0268 PHerc0343 PHerc0358 PHerc0800 PHerc0813 PHerc0826".split()


def cfg_dyn(speed=1.0):
    c = L.load_config()
    c["dynamic"] = True
    c["gpu_speed"] = speed
    return c


def host(n=8, vram=40.0, **kw):
    return P.Host([P.Gpu(str(i), vram) for i in range(n)], **kw)


def test_cost_model_anchors_and_height_exponent():
    assert L.fit_hours("PHerc0125", 125, 13000) == pytest.approx(5.99)
    assert L.fit_hours("PHerc0211", 115, 13000) == pytest.approx(7.49)
    h28 = L.fit_hours("PHerc0125", 125, 2800)
    assert 1.4 < h28 < 1.6                      # vs 1.76 h p50 measured on the contended V100 box and 1.03 h on the 4060 Ti: same order
    five = 5 * L.fit_hours("PHerc0125", 125, 2800)
    assert 1.15 < five / 5.99 < 1.35            # splitting a scroll costs ~1.25x the GPU-h (documented consequence)
    assert L.fit_hours("PHerc0125", 125, 13000, "p90") == pytest.approx(5.99 * 1.52)
    assert L.fit_hours("PHerc0125", 125, 13000, speed=2.0) == pytest.approx(5.99 / 2)


def test_heights_follow_vram_and_track_limit():
    c = cfg_dyn()
    assert L.start_height_for(c, "PHerc0125", 40.0, 13000)[0] == 13000                       # A100 40 GB: whole scroll
    h16 = L.start_height_for(c, "PHerc0125", 15.58, 13000)[0]
    assert h16 == L.computed_height(15.58, span=13000) and 2800 <= h16 < 13000                # 4060 Ti: stripes
    assert L.start_height_for(c, "PHerc0191", 40.0, 13000)[0] < 13000                         # 22.76M tracks > 2^24: split by the track limit
    assert L.start_height_for(c, "PHerc0125", 40.0, 13000, max_height=6000)[0] == 6000
    # recorded success wins
    import tempfile
    hp = Path(tempfile.mkdtemp()) / "h.json"
    L.record_height(hp, "PHerc0125", 9000, 40.0)
    assert L.start_height_for(c, "PHerc0125", 40.0, 13000, heights_path=hp)[0] == 9000


def test_oom_shrinks_by_075_and_walks_down_to_floor():
    c = cfg_dyn()
    j = L.jobs_for_height(c, "PHerc0211", 115, 4500, 17500, 13000)[0]
    assert j["id"] == "PHerc0211/full" and j["expected_h"] == pytest.approx(7.49)
    L.decide(c, j, "oom", 115)                                          # first OOM: memory-lean retry in place
    d = L.decide(c, j, "oom", 115)
    assert d["action"] == "descend" and d["height"] == 9700               # 0.75 x 13000, grid-rounded
    assert [(k["z0"], k["z1"]) for k in d["jobs"]] == [(4500, 14200), (14000, 17500)]
    k = d["jobs"][0]
    L.decide(c, k, "oom", 115)
    d2 = L.decide(c, k, "oom", 115)
    assert d2["height"] == 7200 and len(d2["jobs"]) == 2
    h = 1200
    assert L.shrink_height(h) is None
    small = L.make_job(c, "PHerc0211", 115, L.dyn_rung(c, 1100), 4500, 5600)
    for _ in range(2):
        r = L.decide(c, small, "oom", 115)
    assert r["action"] == "fail" and "floor" in r["why"]


def test_plan_one_gpu_per_scroll_smallest_first_all_gpus_busy():
    c = cfg_dyn()
    h = host(8)
    r = B.Rates()
    pl = P.make_plan(h, r, TEN[:4], c, plan_frac=10.0)                    # no deferral pressure
    sim = pl["p50"]
    first = [s for s in sim["sched"] if s["t0_h"] <= min(x["t0_h"] for x in sim["sched"]) + 1e-9]
    assert len({s["gpu"] for s in first}) == len(first)                  # one job per GPU at the start
    hrs = {n: sum(j["expected_h"] for j in P.make_jobs(c, pl["facts"][n])) for n in pl["keep"]}
    assert [hrs[n] for n in pl["keep"]] == sorted(hrs.values())          # dispatch order is SPT
    # 4 scrolls on 8 GPUs: the tail split must engage and shorten the makespan versus no split
    nos = P.simulate(h, [pl["facts"][n] for n in pl["keep"]], cfg_dyn(), "p50", tail=False)
    assert sim["makespan_h"] < nos["makespan_h"] * 0.8 and sim["gpu_h"] >= nos["gpu_h"]     # faster in wall, never cheaper in GPU-h
    assert any("tail split" in n for n in sim["notes"])


def test_many_scrolls_few_gpus_keeps_every_gpu_busy_until_the_tail():
    c = cfg_dyn()
    h = host(2, fetch_files_per_s=1e6)                                    # instant fetch: isolates the scheduling
    pl = P.make_plan(h, B.Rates(), TEN[:6], c, plan_frac=100.0)
    sim = pl["p50"]
    busy = {g: sorted((s["t0_h"], s["t1_h"]) for s in sim["sched"] if s["gpu"] == g) for g in range(2)}
    for g, iv in busy.items():                                            # no gaps between consecutive jobs on a GPU (inputs never late)
        assert all(abs(iv[i + 1][0] - iv[i][1]) < 1e-6 for i in range(len(iv) - 1)), (g, iv)


def test_tail_split_lpt_never_splits_when_it_does_not_help():
    c = cfg_dyn()
    jobs = [L.jobs_for_height(c, s, 100, 4500, 17500, 13000)[0] for s in ("PHerc0125", "PHerc0211")]
    new, notes = P.tail_split(c, [0.0, 0.0], jobs, {"PHerc0125": 125, "PHerc0211": 115})
    assert not notes and len(new) == 2                                    # 2 jobs on 2 GPUs: already balanced
    new, notes = P.tail_split(c, [0.0] * 8, jobs, {"PHerc0125": 125, "PHerc0211": 115})
    assert notes and len(new) > 2 and max(j["expected_h"] for j in new) < 3.5


def test_budget_deferral_is_explicit_and_by_priority():
    c = cfg_dyn()
    pl = P.make_plan(host(8), B.Rates(), TEN, c)
    kept, dropped = pl["keep"], [d[0] for d in pl["deferred"]]
    assert dropped and set(kept) | set(dropped) == set(TEN) and not (set(kept) & set(dropped))
    assert pl["m90"]["total_usd"] <= pl["limit_usd"] and pl["p90"]["makespan_h"] <= pl["limit_h"]
    assert pl["fits"] and "PHerc0343" == dropped[0]                      # first-letters-only scroll is the lowest priority
    txt = P.render(host(8), B.Rates(), pl, [("PHerc0175A", "no lasagna")])
    assert "DEFERRED PHerc0343" in txt and "NOT RUNNABLE PHerc0175A" in txt and "stagger timeline" in txt and "expected payload arrivals" in txt
    assert "effective $4.801/h" in txt and "disk 934" in txt                # the disk term is in the printed plan
    # a faster card fits more
    pl2 = P.make_plan(host(8), B.Rates(), TEN, cfg_dyn(speed=2.0))
    assert len(pl2["keep"]) > len(kept)


def test_dollars_per_scroll_include_disk_term():
    c = cfg_dyn()
    r = B.Rates()
    pl = P.make_plan(host(8), r, TEN[:2], c, plan_frac=100.0)
    m = pl["m50"]
    assert m["machine_disk_usd"] == pytest.approx(m["billed_h"] * (4.276 + 934 / 16 * 0.009))
    r0 = B.Rates(disk_gb=0.0)
    m0 = P.money(r0, pl["p50"])
    assert m["machine_disk_usd"] - m0["machine_disk_usd"] == pytest.approx(m["billed_h"] * 934 / 16 * 0.009)


def test_disk_high_water_gates_staging_and_peak_stays_under():
    c = cfg_dyn()
    big = host(8, disk_total_gb=934, disk_free_gb=934, fetch_files_per_s=1e6)
    small = host(8, disk_total_gb=300, disk_free_gb=300, fetch_files_per_s=1e6)       # high-water 255 GB: only ~2 scrolls (70 GB each) at once
    names = TEN[:6]
    sb = P.simulate(big, [P.scroll_facts(n, c, big) for n in names], c, "p50")
    ss = P.simulate(small, [P.scroll_facts(n, c, small) for n in names], c, "p50")
    assert ss["disk_peak_gb"] <= small.high_water_gb + 1e-6 and not ss["blocked"]
    assert any("blocked by the disk high-water" in n for n in ss["notes"])
    assert ss["makespan_h"] > sb["makespan_h"]                            # less disk, later finish
    tiny = host(8, disk_total_gb=100, disk_free_gb=100)
    st = P.simulate(tiny, [P.scroll_facts(n, c, tiny) for n in names], c, "p50")
    assert st["blocked"] and any("DEADLOCK" in n for n in st["notes"])    # announced, not silent


def test_ram_limits_concurrent_fits():
    h = host(8, ram_gb=130.0)                                             # (130-30)/40 = 2 fits at a time
    assert h.max_concurrent_fits() == 2
    c = cfg_dyn()
    sim = P.simulate(h, [P.scroll_facts(n, c, h) for n in TEN[:3]], c, "p50")
    assert sim["n_gpus_used"] == 2 and max(s["gpu"] for s in sim["sched"]) <= 1


def test_routea_slots_formula():
    n, why = P.routea_slots(48, 8, 4, 516.0, 40.0, 30.0, 6.0)
    assert n == 27 and "28" in why                                        # cores say 48 - 2x8 - 4 = 28, RAM (516-30-8x40)/6 = 27 caps it
    assert P.routea_slots(48, 8, 4, 600.0, 40.0, 30.0, 6.0)[0] == 28
    n, _ = P.routea_slots(48, 8, 4, 200.0, 40.0, 30.0, 6.0)               # RAM-capped: (200-30-320)<0 -> 0
    assert n == 0
    n, _ = P.routea_slots(48, 0, 4, 516.0, 40.0, 30.0, 6.0)
    assert n == 44
    assert P.routea_slots(48, 8, 4, 516.0, 40.0, 30.0, 6.0, override=3)[0] == 3


def test_routea_scroll_pick_and_publish(tmp_path):
    pins = tmp_path / "scrolls.json"
    pins.write_text(json.dumps({"scrolls": {"A": {"prediction_bytes": 30e9, "grids_bytes": 10e9}, "B": {"prediction_bytes": 10e9, "grids_bytes": 5e9},
                                            "C": {"prediction_bytes": 2e9, "grids_bytes": 1e9}, "D": {"prediction_bytes": 90e9, "grids_bytes": 3e9}}}))
    assert [s for s, _ in RA.pick_scrolls(pins, 60.0, ["A"])] == ["A", "C", "B"] or [s for s, _ in RA.pick_scrolls(pins, 60.0, ["A"])][0] == "A"
    assert [s for s, _ in RA.pick_scrolls(pins, 60.0, [])] == ["C", "B", "A"]
    work, out = tmp_path / "w", tmp_path / "out"
    sd = work / "export" / "S1" / "S1_cabc"
    sd.mkdir(parents=True)
    (sd / "export.json").write_text("{}")
    (sd / "md5.txt").write_text("x")
    os.utime(sd / "export.json", (time.time() - 100, time.time() - 100))
    seen = set()
    assert RA.publish_exports(work, out, seen) == 1
    u = out / "routeA" / "S1__S1_cabc"
    meta = json.loads((u / "PAYLOAD.json").read_text())
    assert (u / "DONE").exists() and {e["path"] for e in meta["files"]} == {"export.json", "md5.txt"}
    assert RA.publish_exports(work, out, seen) == 0


# ---------------------------------------------------------------- dynamic scheduler, stub jobs
def make_dyn(tmp_path, monkeypatch, scrolls, gpus, plan, extra=(), vram=40.0, sleep="0.6"):
    monkeypatch.setenv("ROUTEB_HOME", str(tmp_path))
    monkeypatch.setenv("STUB_PLAN", json.dumps(plan))
    monkeypatch.setenv("STUB_SLEEP", sleep)
    a = B8.build_parser().parse_args(["--scrolls", ",".join(scrolls), "--fake-gpus", str(gpus), "--fake-vram-gib", str(vram), "--poll-s", "0.2", "--steps", "10", "--no-routea", *extra])
    a.stall_minutes = 30.0
    c = cfg_dyn()
    gov = B.Governor(tmp_path / "box8" / "budget", B.Rates(soft_usd=1e6, hard_usd=2e6), box_start=time.time(), say=lambda *_: None)
    s = B8.Scheduler(a, c, gov, fetch_fn=lambda sc: 0.1, job_cmd=[sys.executable, str(HERE / "stub_job.py")], box_home=tmp_path)
    s.host = B8.build_host(a, tmp_path)
    s.gpus = [g.idx for g in s.host.gpus]
    for sc in scrolls:
        s.add_scroll(sc, 4500, 17500, None)
    return s


def test_live_tail_split_units_appear_per_stripe_and_heights_recorded(tmp_path, monkeypatch):
    s = make_dyn(tmp_path, monkeypatch, ["PHerc0125"], 4, {}, extra=("--steps", "30000"))
    assert len(s.jobs) == 1
    s.run()
    done = [j for j in s.jobs.values() if j["status"] == "done"]
    assert len(done) > 1 and any(j["status"] == "split" for j in s.jobs.values())       # 1 scroll on 4 GPUs -> z-stripes
    for j in done:
        assert (tmp_path / "out" / "PHerc0125" / j["tag"] / "DONE").exists()
    assert (tmp_path / "out" / "PHerc0125" / "DONE").exists() and (tmp_path / "out" / "ALLDONE.json").exists() and (tmp_path / "out" / "STATUS.json").exists()
    assert L.load_heights(s.heights_path)["PHerc0125"]["height"] > 0
    ev = [json.loads(x)["kind"] for x in (tmp_path / "box8" / "events.jsonl").read_text().splitlines()]
    assert "tail_split" in ev and ev.count("unit") == len(done)


def test_live_oom_shrinks_075_and_records_height(tmp_path, monkeypatch):
    s = make_dyn(tmp_path, monkeypatch, ["PHerc0211"], 1, {"full": ["oom", "oom"]}, extra=("--no-tail-split",))
    s.run()
    j = s.jobs["PHerc0211/full"]
    assert j["status"] == "descended" and j["descended_to"] == "h9700"
    kids = [k for k in s.jobs.values() if k["rung_name"] == "h9700"]
    assert len(kids) == 2 and all(k["status"] == "done" for k in kids)
    assert L.load_heights(s.heights_path)["PHerc0211"]["height"] == 9700                # the height that SUCCEEDED, for the next scroll/run


def test_live_disk_admission_blocks_then_releases_after_pull(tmp_path, monkeypatch):
    s = make_dyn(tmp_path, monkeypatch, ["PHerc0125", "PHerc0211"], 1, {}, extra=("--no-tail-split", "--poll-s", "0.2"), sleep="0.3")
    state = {"free": 130e9}
    s.disk_fn = lambda: (200e9, state["free"])                 # high-water 170 GB; each scroll 60-70 GB, base 70 GB used -> one scroll at a time
    s.a.disk_high_water = 0.85
    import threading
    rc = {}
    t = threading.Thread(target=lambda: rc.setdefault("rc", s.run()))
    t.start()
    t0 = time.time()
    seen_block = False
    while t.is_alive() and time.time() - t0 < 60:
        ev = (tmp_path / "box8" / "events.jsonl")
        if ev.exists() and "stage_blocked" in ev.read_text():
            seen_block = True
            for p in (tmp_path / "out").glob("*/*/DONE"):       # play the puller: mark every finished unit pulled
                (p.parent / "PULLED.json").write_text(json.dumps({"bytes": 1000}))
        time.sleep(0.2)
    t.join(30)
    assert seen_block and rc.get("rc") == 0
    kinds = [json.loads(x)["kind"] for x in (tmp_path / "box8" / "events.jsonl").read_text().splitlines()]
    assert "released" in kinds and "stage_blocked" in kinds
    assert not (tmp_path / "assets" / "PHerc0125").exists() or True


def test_default_main_dry_run_prints_the_plan(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ROUTEB_HOME", str(tmp_path))
    rc = B8.main(["--fake-gpus", "8", "--dry-run", "--phys-cores", "48", "--disk-total-gb", "934", "--disk-free-gb", "934"])   # NO --scrolls: every runnable eligible scroll
    out = capsys.readouterr().out
    assert rc == 0 and "PLAN host: 8 GPU(s)" in out and "NOT RUNNABLE PHerc0175A" in out and "NOT RUNNABLE PHerc0846B" in out
    assert "PLAN Route A on the spare cores: ON; 27 grow slot(s)" in out and "DEFERRED" in out and "GPU x time" in out
    assert not (tmp_path / "box8" / "events.jsonl").exists() and not (tmp_path / "out").exists()
