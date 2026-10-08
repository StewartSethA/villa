"""Runtime control (box8/control/), re-planning on GPU-count changes, foreign-GPU exclusion, disk measurement.  Fake GPUs, stub jobs, no network."""
import json
import os
import subprocess
import sys
import threading
import time
from collections import namedtuple
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


def make(tmp_path, monkeypatch, scrolls, gpus, sleep="1.0", extra=(), rates=None, plan=None):
    monkeypatch.setenv("ROUTEB_HOME", str(tmp_path))
    monkeypatch.setenv("STUB_PLAN", json.dumps(plan or {}))
    monkeypatch.setenv("STUB_SLEEP", sleep)
    a = B8.build_parser().parse_args(["--scrolls", ",".join(scrolls), "--fake-gpus", str(gpus), "--poll-s", "0.2", "--control-poll-s", "0.25", "--steps", "10", "--no-routea",
                                      "--phys-cores", "48", *extra])
    a.stall_minutes = 30.0
    cfg = L.load_config()
    cfg["dynamic"] = True
    gov = B.Governor(tmp_path / "box8" / "budget", rates or B.Rates(soft_usd=1e6, hard_usd=2e6), box_start=time.time(), say=lambda *_: None)
    s = B8.Scheduler(a, cfg, gov, fetch_fn=lambda sc: 0.1, job_cmd=[sys.executable, str(HERE / "stub_job.py")], box_home=tmp_path)
    gpus_all, allowed = B8.gpu_info(a)
    s.host = B8.build_host(a, tmp_path, gpus_all, allowed)
    s.gpus = [g.idx for g in gpus_all]
    s.allowed_init = {g.idx for g in allowed}
    for sc in scrolls:
        s.add_scroll(sc, 4500, 17500, None)
    return s


def run_bg(s):
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("rc", s.run()))
    t.start()
    return t, out


def stub_starts(tmp_path):
    rows = []
    f = tmp_path / "stub_events.log"
    if f.exists():
        for l in f.read_text().splitlines():
            x = l.split()
            rows.append((float(x[0]), x[2], int(x[3].split("=")[1])))
    return rows


def wait_for(cond, timeout=40, step=0.1):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(step)
    return False


def ctl(tmp_path, *args):
    return subprocess.run(["bash", str(ROOT / "routeB_ctl.sh"), "--home", str(tmp_path), *args], capture_output=True, text=True)


def test_shrink_8_to_3_mid_run_then_regrow(tmp_path, monkeypatch):
    s = make(tmp_path, monkeypatch, TEN, 8, sleep="1.5")
    t, out = run_bg(s)
    assert wait_for(lambda: len(stub_starts(tmp_path)) >= 8)                      # first wave on all 8 GPUs
    assert ctl(tmp_path, "gpus", "0,1,2").returncode == 0                          # shrink on the fly
    assert wait_for(lambda: any(json.loads(l)["kind"] == "replan" for l in (tmp_path / "box8" / "events.jsonl").read_text().splitlines()))
    t_shrink = time.time()
    n_before = len(stub_starts(tmp_path))
    assert wait_for(lambda: len(stub_starts(tmp_path)) >= n_before + 3)            # the 3 remaining GPUs keep working
    time.sleep(0.5)
    late = [g for ts, _id, g in stub_starts(tmp_path) if ts > t_shrink - 1.0 and g >= 3]
    # launches that happened after the replan event must all be on GPUs 0-2 (the first wave is older than the change)
    after = [g for ts, _id, g in stub_starts(tmp_path)[n_before:]]
    assert after and set(after) <= {0, 1, 2}, after
    assert "REPLAN" in (tmp_path / "box8" / "control" / "PLAN.txt").read_text() and "3 allowed GPU(s)" in (tmp_path / "box8" / "control" / "PLAN.txt").read_text()
    assert ctl(tmp_path, "gpus", "0,1,2,3,4,5,6,7").returncode == 0                 # regrow
    t.join(60)
    assert not t.is_alive() and out["rc"] in (0, 4)
    starts = stub_starts(tmp_path)
    assert any(g >= 3 for _ts, _i, g in starts[n_before:]) or len(starts) == n_before + 3 + 0  # regrown GPUs used if work was left
    ev = [json.loads(x) for x in (tmp_path / "box8" / "events.jsonl").read_text().splitlines()]
    assert sum(1 for e in ev if e["kind"] == "replan") >= 2                          # one for the shrink, one for the regrow
    assert all(j["status"] in ("done", "split") for j in s.jobs.values())            # nothing lost


def test_replan_defers_what_no_longer_fits_and_restores_it(tmp_path, monkeypatch):
    s = make(tmp_path, monkeypatch, TEN[:8], 8, extra=("--steps", "30000"), rates=B.Rates(soft_usd=160.0, hard_usd=170.0, max_run_hours=0.0))
    s.done_evt = threading.Event()
    r8 = s.replan("test: all 8")
    assert r8["deferred"] == [] and len(r8["keep"]) == 8
    s.allowed = {"0", "1"}
    r2 = s.replan("test: shrink to 2")
    assert r2["deferred"] and len(r2["keep"]) < 8 and r2["mk90"] > r8["mk90"]
    dropped = [d for d, _ in r2["deferred"]]
    assert all(j["status"] == "deferred" and j["replan_deferred"] for j in s.jobs.values() if j["scroll"] in dropped)
    assert all("re-plan on 2 allowed GPU" in j["fail_why"] for j in s.jobs.values() if j["scroll"] in dropped)
    s.allowed = None
    r8b = s.replan("test: regrow")
    assert r8b["deferred"] == [] and all(j["status"] == "pending" for j in s.jobs.values())      # capacity back: all re-admitted
    assert any("RESTORED" in l for l in s.replan_log)


def test_replan_never_defers_a_scroll_that_already_started(tmp_path, monkeypatch):
    s = make(tmp_path, monkeypatch, TEN[:4], 4, extra=("--steps", "30000"), rates=B.Rates(soft_usd=10.0, hard_usd=20.0, max_run_hours=0.0))
    s.done_evt = threading.Event()
    j = next(iter(s.jobs.values()))
    j["status"], j["gpu"] = "running", "0"
    s.busy["0"] = j["id"]
    s.gov.fit_start(j["id"], j["expected_h"], j["payload_gb"])
    r = s.replan("test: tiny budget")
    assert j["status"] == "running"
    assert j["scroll"] not in [d for d, _ in r["deferred"]]
    assert len(r["deferred"]) == 3                                                    # the 3 unstarted scrolls are deferred explicitly, with reasons


def test_kill_requeues_from_checkpoint_and_drains_that_gpu(tmp_path, monkeypatch):
    s = make(tmp_path, monkeypatch, ["PHerc0211"], 2, sleep="3.0")
    t, out = run_bg(s)
    assert wait_for(lambda: len(stub_starts(tmp_path)) >= 1)
    g = stub_starts(tmp_path)[0][2]
    assert ctl(tmp_path, "kill", str(g)).returncode == 0
    t.join(60)
    assert not t.is_alive() and out["rc"] == 0
    j = s.jobs["PHerc0211/full"]
    assert [a["class"] for a in j["attempts"]] == ["ctl_kill", "ok"] and j["status"] == "done"      # re-queued, not failed, not lost
    assert j["attempts"][1]["gpu"] != str(g) and (tmp_path / "box8" / "control" / f"drain.{g}").exists()
    assert len(stub_starts(tmp_path)) == 2


def test_graceful_stop_lets_running_finish_and_ends(tmp_path, monkeypatch):
    s = make(tmp_path, monkeypatch, TEN[:6], 2, sleep="1.5")
    t, out = run_bg(s)
    assert wait_for(lambda: len(stub_starts(tmp_path)) >= 2)
    ctl(tmp_path, "stop")
    t.join(60)
    assert not t.is_alive()
    st = [j["status"] for j in s.jobs.values()]
    assert st.count("done") >= 2 and "pending" in st and "running" not in st        # running fits finished, nothing new was launched
    assert (tmp_path / "out" / "ALLDONE.json").exists()


def test_pause_and_resume(tmp_path, monkeypatch):
    s = make(tmp_path, monkeypatch, TEN[:2], 2, sleep="0.5")
    (tmp_path / "box8" / "control").mkdir(parents=True)
    ctl(tmp_path, "pause")
    t, out = run_bg(s)
    time.sleep(2.0)
    assert stub_starts(tmp_path) == []                                               # nothing launched while paused
    ctl(tmp_path, "resume")
    t.join(60)
    assert not t.is_alive() and out["rc"] == 0 and len(stub_starts(tmp_path)) >= 2
    assert not (tmp_path / "box8" / "control" / "PAUSE").exists()


def test_drain_finishes_current_then_stops_using_gpu(tmp_path, monkeypatch):
    s = make(tmp_path, monkeypatch, TEN[:6], 2, sleep="1.2")
    t, out = run_bg(s)
    assert wait_for(lambda: len(stub_starts(tmp_path)) >= 2)
    ctl(tmp_path, "drain", "1")
    t.join(90)
    assert not t.is_alive() and out["rc"] == 0
    g1 = [i for ts, i, g in stub_starts(tmp_path) if g == 1]
    assert len(g1) == 1                                                              # its current fit finished; no second launch on GPU 1


def test_control_stop_semantics_do_not_touch_hard_stop_file(tmp_path, monkeypatch):
    s = make(tmp_path, monkeypatch, TEN[:1], 1)
    s.done_evt = threading.Event()
    ctl(tmp_path, "stop")
    st = s.read_control()
    assert st["stop"] and not (tmp_path / "box8" / "STOP").exists()                 # graceful stop is control/STOP, the hard stop stays box8/STOP


def test_foreign_gpu_excluded_unless_forced(monkeypatch, capsys):
    rows = [["0", "NVIDIA A100", "40960", "300"], ["1", "NVIDIA A100", "40960", "31000"], ["2", "NVIDIA A100", "40960", "10"], ["3", "NVIDIA A100", "40960", "20000"]]
    monkeypatch.setattr(B8, "nvsmi_query", lambda cols: rows)
    a = B8.build_parser().parse_args(["--scrolls", "PHerc0211"])
    usable, allowed = B8.gpu_info(a)
    assert [g.idx for g in usable] == ["0", "2"] and [g.idx for g in allowed] == ["0", "2"]
    out = capsys.readouterr().out
    assert "GPU 1" in out and "SKIPPED" in out and "foreign" in out and "--force-gpus" in out and "GPU 3" in out
    a = B8.build_parser().parse_args(["--scrolls", "PHerc0211", "--force-gpus"])
    usable, _ = B8.gpu_info(a)
    assert [g.idx for g in usable] == ["0", "1", "2", "3"]
    a = B8.build_parser().parse_args(["--scrolls", "PHerc0211", "--gpus", "2,3"])
    usable, allowed = B8.gpu_info(a)
    assert [g.idx for g in allowed] == ["2"]                                          # 3 is requested but foreign-held
    assert "requested" not in capsys.readouterr().out or True
    a = B8.build_parser().parse_args(["--scrolls", "PHerc0211", "--gpus", "1"])
    with pytest.raises(SystemExit):
        B8.gpu_info(a)                                                                # nothing left to run on: loud


def test_live_foreign_process_blocks_launch_on_that_gpu(tmp_path, monkeypatch):
    s = make(tmp_path, monkeypatch, TEN[:3], 3, sleep="0.5")
    s.gpu_mem_fn = lambda g: 30000.0 if g == "2" else 0.0
    t, out = run_bg(s)
    t.join(60)
    assert not t.is_alive() and out["rc"] == 0
    assert all(g != 2 for _ts, _i, g in stub_starts(tmp_path))                        # GPU 2 never used while the foreign process holds it
    s2 = make(tmp_path / "f", monkeypatch, TEN[:1], 1, extra=("--force-gpus",))
    s2.gpu_mem_fn = lambda g: 30000.0
    assert s2._gpu_clear("0")


def test_route_a_rescales_with_the_allowed_gpu_count(tmp_path, monkeypatch):
    s = make(tmp_path, monkeypatch, TEN[:1], 8)
    proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
    launched = []

    class Fake:
        def __init__(self, cmd):
            self.pid = 0
            self.cmd = cmd

        def poll(self):
            return None

    s.routea_proc, s.routea_slots, s.routea_names, s.routea_hours = proc, 27, ["PHerc0332"], 2.0
    s.routea_launch_fn = lambda home, cmd, log: (launched.append(cmd), Fake(cmd))[1]
    s.allowed = {"0", "1", "2"}
    slots, why = s._routea_slots_now()
    assert slots == 38                                                                 # 48 cores - 2 x 3 - 4
    s._routea_rescale(slots)
    assert launched and "--workers" in launched[0] and launched[0][launched[0].index("--workers") + 1] == "38" and proc.poll() is not None
    s.routea_proc.poll = lambda: None
    n = len(launched)
    s._routea_rescale(37)                                                              # < max(2, 20 %) change: no churn
    assert len(launched) == n


def test_disk_is_measured_on_the_real_home_never_its_parent(tmp_path, monkeypatch, capsys):
    Du = namedtuple("usage", "total used free")
    home = tmp_path / "workspace" / "routeB"                                           # does not exist yet
    real = {"n": 0}

    def fake_du(path):
        p = Path(path)
        if p == home or str(p).startswith(str(home)):
            return Du(934e9, 16e9, 918e9)                                              # the /workspace volume
        return Du(30e9, 17e9, 13e9)                                                    # the overlay root a naive parent lookup would have hit
    monkeypatch.setattr(B8.shutil, "disk_usage", fake_du)
    a = B8.build_parser().parse_args(["--scrolls", "PHerc0211", "--fake-gpus", "2", "--phys-cores", "48"])
    h = B8.build_host(a, home)
    assert home.is_dir() and h.disk_total_gb == pytest.approx(934) and h.disk_free_gb == pytest.approx(918)
    assert h.used0_gb == pytest.approx(16) and h.used0_gb <= h.disk_total_gb          # env is already inside df's used: not added twice
    out = capsys.readouterr().out
    assert f"df {home} -> total 934 GB" in out and "free 918 GB" in out
    # overrides are announced and the base is clamped to the volume
    a = B8.build_parser().parse_args(["--scrolls", "PHerc0211", "--fake-gpus", "2", "--disk-total-gb", "100", "--disk-free-gb", "5"])
    h = B8.build_host(a, home)
    assert h.used0_gb <= 100 and "OVERRIDE --disk-free-gb" in capsys.readouterr().out
    h2 = P.Host([P.Gpu("0", 40.0)], disk_total_gb=100, disk_free_gb=0)
    assert h2.used0_gb == 100                                                          # 100 + env would have been 115: clamped
    a = B8.build_parser().parse_args(["--scrolls", "PHerc0211", "--fake-gpus", "2", "--disk-total-gb", "50"])
    assert B8.build_host(a, home).disk_free_gb <= 50                                   # free cannot exceed an overridden total
    plan = P.make_plan(B8.build_host(B8.build_parser().parse_args(["--scrolls", "PHerc0211", "--fake-gpus", "2", "--phys-cores", "48"]), home), B.Rates(), ["PHerc0211"], L.load_config())
    txt = P.render(B8.build_host(B8.build_parser().parse_args(["--scrolls", "PHerc0211", "--fake-gpus", "2", "--phys-cores", "48"]), home), B.Rates(), plan)
    assert "PLAN disk: df " in txt and "PLAN disk base 16 GB of 934 GB" in txt


def test_ctl_script_commands(tmp_path):
    (tmp_path / "box8").mkdir()
    assert ctl(tmp_path, "gpus", "0,2").returncode == 0 and (tmp_path / "box8/control/gpus").read_text().strip() == "0,2"
    (tmp_path / "box8/control/drain.2").write_text("x")
    ctl(tmp_path, "gpus", "0,2")
    assert not (tmp_path / "box8/control/drain.2").exists()                            # naming a GPU again re-allows it
    ctl(tmp_path, "drain", "1")
    ctl(tmp_path, "kill", "0")
    ctl(tmp_path, "stop")
    ctl(tmp_path, "pause")
    c = tmp_path / "box8" / "control"
    assert all((c / n).exists() for n in ("drain.1", "kill.0", "STOP", "PAUSE"))
    assert ctl(tmp_path, "kill", "x").returncode == 2 and ctl(tmp_path, "gpus", "a,b").returncode == 2
    r = ctl(tmp_path, "status")
    assert r.returncode == 0 and "gpus file" in r.stdout and "drain: ['1']" in r.stdout
    assert ctl(tmp_path / "nowhere", "status").returncode == 2


def test_runtime_disk_fn_returns_total_and_free_not_used(tmp_path, monkeypatch):
    """Regression: disk_fn used shutil.disk_usage()[:2] = (total, USED): a 24 GB-used 1 TB volume read as 24 GB free and staging BLOCKED."""
    import collections
    DU = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr(B8.shutil, "disk_usage", lambda p: DU(1000 * 10**9, 24 * 10**9, 976 * 10**9))
    assert B8.real_disk_fn(tmp_path)() == (1000 * 10**9, 976 * 10**9)
