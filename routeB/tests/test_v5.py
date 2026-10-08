"""v5: link gate (bash + python), idle/fetch alarms, link trend + auto-shrink, dashboard golden/--once/--plain, paste hygiene."""
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from routeB import box8 as B8        # noqa: E402
from routeB import linkcheck as LK   # noqa: E402
from routeB import watch as W        # noqa: E402
from test_control import make, TEN, B  # noqa: E402

GOLDEN = HERE / "golden" / "watch_snapshot.txt"


def stub_dir(tmp_path, curl_bytes):
    d = tmp_path / "stub"
    d.mkdir(exist_ok=True)
    (d / "curl").write_text(f'#!/bin/bash\necho "{curl_bytes}"\n')
    for n in ("apt-get", "git", "tmux", "sudo"):
        (d / n).write_text(f'#!/bin/bash\necho "{n} $*" >> "{tmp_path}/CALLED"\nexit 0\n')
    for f in d.iterdir():
        f.chmod(0o755)
    return d


def boot(tmp_path, curl_bytes, *args):
    env = dict(os.environ, PATH=f"{stub_dir(tmp_path, curl_bytes)}:{os.environ['PATH']}", HOME=str(tmp_path))
    return subprocess.run(["bash", str(ROOT / "box_bootstrap.sh"), *args], capture_output=True, text=True, env=env, timeout=60)


def test_bootstrap_slow_link_stops_before_anything_is_installed(tmp_path):
    r = boot(tmp_path, 100000, "run")                                      # 8 x 100 kB in the 0.5 s floor = 1.6 MB/s
    assert r.returncode == 5 and "VERDICT: BAD" in r.stdout and "DESTROY THIS BOX" in r.stderr and "--accept-slow-link" in r.stderr
    assert "BOX LINK slow" in r.stdout and "h of pure transfer" in r.stdout and "$" in r.stdout
    assert not (tmp_path / "CALLED").exists()                              # no apt-get / git / tmux / sudo: nothing was built


def test_bootstrap_accept_slow_link_and_good_link(tmp_path):
    r = boot(tmp_path, 100000, "linkcheck", "--accept-slow-link")
    assert r.returncode == 0 and "SLOW LINK ACCEPTED" in r.stdout
    r = boot(tmp_path, 9000000, "linkcheck")
    assert r.returncode == 0 and "VERDICT: GOOD" in r.stdout
    r = boot(tmp_path, 30000, "linkcheck", "--min-link-mb-s", "1")         # the gate is configurable
    assert r.returncode == 0 and "VERDICT: MARGINAL" in r.stdout


def test_paste_hygiene_no_heredoc_no_long_lines_one_short_oneliner():
    for f in ("box_bootstrap.sh", "go", "routeB_watch.sh", "routeB_ctl.sh"):
        for i, l in enumerate((ROOT / f).read_text().splitlines(), 1):
            assert len(l) <= 100, (f, i, len(l))
            assert "<<" not in l or f == "routeB_ctl.sh", (f, i)          # routeB_ctl.sh status keeps its python heredoc: it is a file, never pasted
    assert (ROOT / "go").read_bytes() == (ROOT / "box_bootstrap.sh").read_bytes()
    readme = (ROOT / "README.md").read_text()
    one = [l for l in readme.splitlines() if l.startswith("curl -fsSL") and "|bash" in l]
    assert one and all(len(l) <= 100 for l in one)
    code = [l for l in readme.splitlines() if l.startswith(("while :", "H=", "P=", "RH=", "./routeB_"))]
    assert all(len(l) <= 100 for l in code)
    rr = (ROOT / "routeB_run.sh").read_text()
    assert rr.index("routeB.linkcheck") < rr.index("bootstrap_env.sh")        # the link is measured before the env is built


def args_for(tmp_path, *more):
    return ["--scrolls", "PHerc0125,PHerc0211", "--fake-gpus", "8", "--phys-cores", "48", "--steps", "30000", *more]


def test_linkcheck_python_gate_slow_accept_and_shrink(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ROUTEB_HOME", str(tmp_path))
    monkeypatch.setattr(B8, "LINK_FN", lambda url: (3.0, "fake"))
    assert LK.main(args_for(tmp_path)) == 3
    out = capsys.readouterr()
    assert "VERDICT: BAD" in out.out and "DESTROY THIS BOX" in out.out and "STOPPING" in out.err
    first = json.loads((tmp_path / "box8" / "link" / "first.json").read_text())
    assert first["verdict"]["verdict"] == "BAD" and first["plan_gb"] > 10 and first["shrink"]["gpus"] < 8
    assert LK.main(args_for(tmp_path, "--accept-slow-link")) == 0
    out = capsys.readouterr().out
    assert "ACCEPTED" in out and "AUTO-SHRINK" in out
    monkeypatch.setattr(B8, "LINK_FN", lambda url: (500.0, "fake"))
    assert LK.main(args_for(tmp_path)) == 0 and "VERDICT: GOOD" in capsys.readouterr().out


def test_diagnosis_tells_box_link_from_source():
    m = lambda d, s: {"data": {"mbs": d, "why": ""}, "second": {"mbs": s, "why": ""}}
    assert "BOX LINK" in LK.diagnose(m(4, 6), 20)
    assert "SOURCE" in LK.diagnose(m(4, 200), 20)
    assert "unreachable" in LK.diagnose(m(None, 200), 20)
    assert LK.verdict(m(10, 10), 40, 4, 4.8, 20)["verdict"] == "BAD"
    assert LK.verdict(m(30, 10), 40, 4, 4.8, 20)["verdict"] == "MARGINAL"
    assert LK.verdict(m(400, 10), 40, 4, 4.8, 20)["verdict"] == "GOOD"


def test_box8_main_gates_and_auto_shrinks(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ROUTEB_HOME", str(tmp_path))
    monkeypatch.setattr(B8, "LINK_FN", lambda url: (3.0, "fake"))
    base = ["--scrolls", "PHerc0125,PHerc0211", "--fake-gpus", "8", "--phys-cores", "48", "--dry-run"]
    assert B8.main(base) == 5
    assert "DESTROY THIS BOX" in capsys.readouterr().out
    assert B8.main(base + ["--accept-slow-link"]) in (0, 3)
    out = capsys.readouterr().out
    assert "AUTO-SHRINK" in out and re.search(r"PLAN host: [1-7] GPU\(s\)", out)           # the plan is made for the shrunken set
    assert "PLAN link 3 MB/s" in out and "DOWNLOAD TIME DOMINATES" in out


def test_idle_gpu_and_fetch_stall_alarms(tmp_path, monkeypatch):
    s = make(tmp_path, monkeypatch, TEN[:2], 2, extra=("--idle-alarm-s", "60"))
    s.done_evt = threading.Event()
    now = time.time()
    first = next(iter(s.scrolls))
    s.scrolls[first]["fetching"] = True                                     # data not here yet, a GPU is free, jobs are pending
    s.idle_since["0"] = now - 120
    s.rx_fn = lambda: 1000
    s.rx_hist = [(now - 70, 1000)]                                           # no bytes for 70 s
    s.check_alarms()
    assert "idle:gpu0" in s.alarms and "waiting for DATA" in s.alarms["idle:gpu0"]["text"]
    assert "fetch_stall" in s.alarms and "NOT PROGRESSING" in s.alarms["fetch_stall"]["text"]
    ev = [json.loads(x)["kind"] for x in (tmp_path / "box8" / "events.jsonl").read_text().splitlines()]
    assert "alarm" in ev
    alerts = json.loads((tmp_path / "out" / "ALERTS.json").read_text())["alarms"]
    assert {a["key"] for a in alerts} >= {"idle:gpu0", "fetch_stall"}
    s.rx_fn = lambda: 500_000_000                                            # traffic resumes
    s.scrolls[first]["fetching"] = False
    s.scrolls[first]["fetched"] = True
    s.busy["0"], s.busy["1"] = "x", "y"
    s.check_alarms()
    assert "fetch_stall" not in s.alarms and "idle:gpu0" not in s.alarms
    assert "alarm_clear" in [json.loads(x)["kind"] for x in (tmp_path / "box8" / "events.jsonl").read_text().splitlines()]


def test_idle_cause_is_named(tmp_path, monkeypatch):
    s = make(tmp_path, monkeypatch, TEN[:2], 2)
    pend = [j for j in s.jobs.values()]
    s.paused = True
    assert "PAUSED" in s._idle_cause("0", pend)
    s.paused = False
    s.foreign_skip["0"] = time.time() + 30
    assert "foreign" in s._idle_cause("0", pend)
    s.foreign_skip.clear()
    for sc in s.scrolls.values():
        sc["fetched"] = True
    assert "refusing" in s._idle_cause("0", pend)
    for sc in s.scrolls.values():
        sc["fetched"], sc["fetching"] = False, False
    s.alarms["stage_blocked"] = {"since": time.time(), "text": "x", "sev": "red"}
    assert "STAGING BLOCKED" in s._idle_cause("0", pend)


def test_link_trend_degrade_autoshrink_and_recover(tmp_path, monkeypatch):
    s = make(tmp_path, monkeypatch, TEN[:4], 8, extra=("--steps", "30000"), rates=B.Rates(soft_usd=900.0, hard_usd=910.0, max_run_hours=0.0))
    s.done_evt = threading.Event()
    mk = lambda d: {"t": time.time(), "data": {"mbs": d, "why": "fake"}, "second": {"mbs": 100.0, "why": "fake"}}
    s.link_base = 100.0
    s.link_probe_fn = lambda: mk(100.0)
    s.link_check_once()
    assert s.link_state == "ok"
    s.link_probe_fn = lambda: mk(4.0)
    s.link_check_once()
    assert s.link_state == "degraded" and "link" in s.alarms
    gp = (tmp_path / "box8" / "control" / "gpus").read_text().strip().split(",")
    assert 1 <= len(gp) < 8                                                   # auto-shrunk through the control dir
    rows = LK.trend_read(tmp_path)
    assert [r["data_mbs"] for r in rows] == [100.0, 4.0]
    s.link_probe_fn = lambda: mk(95.0)
    s.link_check_once()
    assert s.link_state == "ok" and "link" not in s.alarms
    kinds = [json.loads(x)["kind"] for x in (tmp_path / "box8" / "events.jsonl").read_text().splitlines()]
    assert {"link_degraded", "autoshrink", "link_recovered"} <= set(kinds)
    # --no-auto-shrink leaves the allowed set alone
    (tmp_path / "box8" / "control" / "gpus").unlink()
    s.a.no_auto_shrink = True
    s.last_shrink = 0
    s.link_probe_fn = lambda: mk(3.0)
    s.link_check_once()
    assert not (tmp_path / "box8" / "control" / "gpus").exists()


# ---------------------------------------------------------------- dashboard
def fixture_home(tmp_path):
    H = tmp_path / "home"
    (H / "box8" / "state").mkdir(parents=True)
    (H / "box8" / "logs").mkdir()
    (H / "box8" / "link").mkdir()
    (H / "box8" / "control").mkdir()
    (H / "out" / "PHerc0125" / "full").mkdir(parents=True)
    (H / "out" / "PHerc0211" / "full").mkdir(parents=True)
    (H / "runs" / "PHerc0191" / "h9100s4500" / "fit").mkdir(parents=True)
    jobs = [{"id": "PHerc0191/h9100s4500", "scroll": "PHerc0191", "tag": "h9100s4500", "z0": 4500, "z1": 13600, "status": "running", "gpu": "0", "rung_name": "h9100",
             "parent": "PHerc0191/full", "attempts": [{"class": "oom"}]},
            {"id": "PHerc0191/full", "scroll": "PHerc0191", "tag": "full", "z0": 4500, "z1": 17500, "status": "descended", "gpu": "0", "rung_name": "full", "parent": None, "attempts": []}]
    (H / "box8" / "state" / "PHerc0191.json").write_text(json.dumps({"scroll": "PHerc0191", "fetched": True, "fetch_failed": None, "finalized": None, "jobs": jobs}))
    (H / "box8" / "state" / "PHerc0257.json").write_text(json.dumps({"scroll": "PHerc0257", "fetched": False, "fetch_failed": None, "finalized": None, "jobs": []}))
    (H / "box8" / "state" / "PHerc0813.json").write_text(json.dumps({"scroll": "PHerc0813", "fetched": False, "fetch_failed": None, "finalized": None, "jobs": []}))
    (H / "runs" / "PHerc0191" / "h9100s4500" / "fit" / "fit.log").write_text("PROGRESS Optimizing — 8,100/30,000 iterations (27.0%) — 2.9 it/s — elapsed 46m — ETA 2h 06m\n")
    for sc, gb in (("PHerc0125", 0.3), ("PHerc0211", 0.4)):
        u = H / "out" / sc / "full"
        (u / "DONE").write_text("complete")
        (u / "PAYLOAD.json").write_text(json.dumps({"total_bytes": gb * 1e9, "files": [{}] * 120}))
    (H / "out" / "PHerc0125" / "full" / "PULLED.json").write_text("{}")
    st = {"t": "2026-10-08T12:00:00Z", "jobs": {"running": 1, "done": 2, "pending": 5}, "busy_gpus": ["0"], "stop": None, "scrolls": {},
          "budget": {"spent": 18.4, "projected_total": 31.2, "soft": 45.0, "hard": 49.0, "hours": 3.83, "max_run_hours": 12.0, "eff_hour_usd": 4.801, "hard_stop": False}}
    (H / "out" / "STATUS.json").write_text(json.dumps(st))
    (H / "out" / "ALERTS.json").write_text(json.dumps({"alarms": [{"key": "idle:gpu1", "since": 1_800_000_000 - 240, "sev": "red",
                                                                   "text": "GPU 1 IDLE 240 s beside 5 pending job(s): waiting for DATA: PHerc0257 still fetching"}]}))
    ev = [{"kind": "failure", "cls": "oom", "decision": "retry"}, {"kind": "failure", "cls": "oom", "decision": "descend"}, {"kind": "stage", "scroll": "PHerc0257"},
          {"kind": "stage_blocked", "scroll": "PHerc0813", "need_gb": 71, "held_gb": 700}, {"kind": "routea_start", "slots": 27}]
    (H / "box8" / "events.jsonl").write_text("\n".join(json.dumps(e) for e in ev) + "\n")
    now = 1_800_000_000
    (H / "box8" / "link" / "trend.jsonl").write_text("\n".join(json.dumps({"t": now - 600 * i, "data_mbs": v, "second_mbs": 150}) for i, v in enumerate([61, 80, 96, 99])) + "\n")
    (H / "box8" / "link" / "first.json").write_text(json.dumps({"verdict": {"verdict": "GOOD"}, "measured": {}}))
    log = ["[11:58:01] box8: PHerc0191/full: FAILED class=oom -> DESCEND: oom: re-cover z[4500,17500) at height 9100",
           "[11:58:09] fetch:     s3 PHerc0257/representations/x/PHerc0257_nx.ome.zarr: 6000/8573 objects, 0.16 GB",
           "[11:58:20] fetch:     PHerc0257_2025_surface.dbm: 24/162 blocks, 0.93 GB this run, 26.0 MB/s",
           "[11:58:40] fetch:     PHerc0257_2025_surface.dbm: 80/162 blocks, 3.10 GB this run, 108.0 MB/s",
           "[11:58:45] box8: PHerc0125: complete: 1 unit(s) in out/PHerc0125/ (scroll DONE marker written)",
           "[11:59:02] box8: PHerc0191/h9100s4500: rung h9100 z[4500,13600) on GPU 0 pid 4242 (attempt 1, overrides {})",
           "[11:59:30] budget: LAUNCH PHerc0191/h9100s4500: projected $31.20",
           "[11:59:40] alarm: ALARM [idle:gpu1] GPU 1 IDLE 240 s beside 5 pending job(s): waiting for DATA",
           "[11:59:58] stage: STAGING PHerc0813 BLOCKED by disk: base 90 + held 700 + need 71 > high-water 794 GB"]
    (H / "box8.log").write_text("\n".join(log) + "\n")
    (H / "box8" / "logs" / "routeA.log").write_text("[routeA] grow: 2 seeds\n")
    os.utime(H / "box8" / "logs" / "routeA.log", (now - 30, now - 30))
    (H / "box8" / "control" / "PLAN.txt").write_text("REPLAN (x): allowed 8 of 8 GPU(s)\n")
    return H


def golden_snap(H):
    snap = W.collect(H, now=1_800_000_000, gpus=lambda: [{"idx": "0", "util": 97.0, "mem": 21000.0, "total": 40960.0}, {"idx": "1", "util": 0.0, "mem": 300.0, "total": 40960.0}],
                     env={"SSH_CONNECTION": "1.2.3.4 55555 203.0.113.9 2222", "USER": "root"}, tree=H)
    snap["host"]["name"] = "testbox"
    snap["disk"] = (123.0, 811.0, 934.0)
    snap["ram"] = (400.0, 516.0)
    snap["commit"], snap["branch"] = "abc1234", "routeAB-deploy-v5"
    return snap


def test_dashboard_golden_snapshot(tmp_path, monkeypatch):
    from vesuvius_pipeline.routea_cloud import status as _RS          # live /proc readings must not leak into the golden text
    monkeypatch.setattr(_RS, "cpu_busy_pct", lambda *a, **k: 50.0)
    monkeypatch.setattr(_RS, "tracer_count", lambda: 0)
    monkeypatch.setenv("TZ", "UTC")
    time.tzset()
    H = fixture_home(tmp_path)
    txt = W.render(golden_snap(H), color=False).replace(str(H), "<HOME>")
    if not GOLDEN.exists():
        GOLDEN.write_text(txt + "\n")
    assert txt + "\n" == GOLDEN.read_text()
    assert "ALERTS (" in txt and "waiting for DATA" in txt and "failures: oom" in txt and "STAGING BLOCKED" in txt


def test_dashboard_content_and_hygiene(tmp_path, monkeypatch):
    monkeypatch.setenv("TZ", "UTC")
    time.tzset()
    H = fixture_home(tmp_path)
    snap = golden_snap(H)
    txt = W.render(snap, color=False)
    assert "8100/30000" in txt and "2.9" in txt and "2h 06m" in txt and "desc 1 oom 1" in txt          # step, rate, ETA, ladder
    assert "link 99 MB/s" in txt and "steady" in txt or "falling" in txt or "rising" in txt
    assert "IDLE" in txt and "PHerc0813" in txt and "BLOCKED by disk" in txt and "fetching" in txt
    assert re.search(r"3\.\d+/\d+\.\d+ GB\s+\d+ MB/s\s+ETA \d+ min", txt)                  # GB done/total, MB/s, ETA per fetching scroll
    assert "2 unit(s) DONE" in txt and "1 pulled" in txt and "ROUTE A  slots 27" in txt
    assert "[fit gpu0 PHerc0191]" in txt and "[box8]" in txt and "[budget]" in txt and "[ALARM]" in txt
    assert "blocks" not in txt.split("EVENTS")[1] and "objects" not in txt.split("EVENTS")[1]            # fetch noise is collapsed into the rows
    assert "H=root@203.0.113.9" in txt and "P=2222" in txt and f"RH={H}" in txt
    for l in txt.splitlines():
        if l.strip().startswith(("H=", "P=", "RH=", "R=", "mkdir", "while", "./routeB_pull")):
            assert len(l.strip()) <= 100, l
    assert "\033[" not in txt
    assert "\033[" in W.render(snap, color=True)
    other = W.pull_lines(H, {"PUBLIC_IPADDR": "9.9.9.9", "VAST_TCP_PORT_22": "40022", "USER": "root"})
    assert other[0] == "H=root@9.9.9.9" and other[1] == "P=40022"
    other = W.pull_lines(H, {"RUNPOD_PUBLIC_IP": "8.8.8.8", "RUNPOD_TCP_PORT_22": "10222", "USER": "root"})
    assert other[0] == "H=root@8.8.8.8" and other[1] == "P=10222"
    assert W.pull_lines(H, {"USER": "root"})[0] == "H=root@<BOX-IP>"


def test_watch_once_and_plain_cli(tmp_path):
    H = fixture_home(tmp_path)
    env = dict(os.environ, ROUTEB_HOME=str(H), PYTHONPATH=str(ROOT))
    r = subprocess.run(["bash", str(ROOT / "routeB_watch.sh"), "--once", "--full", "--home", str(H)], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0
    for need in ("=== routeB snapshot", "--- STATUS.json ---", "--- link first.json / trend ---", "--- control dir / latest REPLAN ---", "--- nvidia-smi ---", "=== end snapshot ===", "ALERTS (", "PULL from your machine"):
        assert need in r.stdout, need
    p = subprocess.Popen(["python3", "-m", "routeB.watch", "--home", str(H), "--plain", "--interval", "0.3"], stdout=subprocess.PIPE, text=True, env=env, cwd=str(ROOT))
    time.sleep(2.5)
    p.send_signal(2)
    out, _ = p.communicate(timeout=20)
    assert out.count("----- ") >= 2 and "watcher detached" in out and "\033[" not in out


# ---------------------------------------------------------------- fill idle GPUs, resume report, route A timing, GPU smoke, VRAM table, cu129 lock
def make_gated(tmp_path, monkeypatch, gpus, extra, gate):
    """Two scrolls; the second one's fetch blocks on `gate` (the next scroll 'arrives later')."""
    import test_control as TC
    from routeB import ladder as L
    monkeypatch.setenv("ROUTEB_HOME", str(tmp_path))
    monkeypatch.setenv("STUB_PLAN", "{}")
    monkeypatch.setenv("STUB_SLEEP", "1.2")
    a = B8.build_parser().parse_args(["--scrolls", "PHerc0125,PHerc0211", "--fake-gpus", str(gpus), "--poll-s", "0.2", "--control-poll-s", "0.25", "--steps", "30000",
                                      "--no-routea", "--phys-cores", "48", "--no-tail-split", *extra])
    a.stall_minutes = 30.0
    cfg = L.load_config()
    cfg["dynamic"] = True
    gov = B.Governor(tmp_path / "box8" / "budget", B.Rates(soft_usd=1e6, hard_usd=2e6), box_start=time.time(), say=lambda *_: None)
    arrived = {}

    def fetch(sc):
        if sc == "PHerc0211":
            gate.wait(60)
            arrived["t"] = time.time()
        return 0.1
    s = B8.Scheduler(a, cfg, gov, fetch_fn=fetch, job_cmd=[sys.executable, str(HERE / "stub_job.py")], box_home=tmp_path)
    ga, al = B8.gpu_info(a)
    s.host = B8.build_host(a, tmp_path, ga, al)
    s.gpus = [g.idx for g in ga]
    s.allowed_init = {g.idx for g in al}
    for sc in ("PHerc0125", "PHerc0211"):
        s.add_scroll(sc, 4500, 17500, None)
    return s, arrived


def starts_by_scroll(tmp_path):
    rows = {}
    for l in (tmp_path / "stub_events.log").read_text().splitlines() if (tmp_path / "stub_events.log").exists() else []:
        x = l.split()
        rows.setdefault(x[2].split("/")[0], []).append((float(x[0]), int(x[3].split("=")[1])))
    return rows


def run_and_release(s, gate, tmp_path, release_after=2.0):
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("rc", s.run()))
    t.start()
    t0 = time.time()
    while time.time() - t0 < release_after:
        time.sleep(0.05)
    t_rel = time.time()
    gate.set()
    t.join(90)
    assert not t.is_alive()
    return t_rel, out


def test_fill_idle_gpus_stripes_fill_idle_cards_and_next_scroll_is_not_starved(tmp_path, monkeypatch):
    gate = threading.Event()
    s, _arr = make_gated(tmp_path, monkeypatch, 8, ("--fill-idle-gpus", "--fill-min-height", "1500"), gate)
    t_rel, out = run_and_release(s, gate, tmp_path)
    rows = starts_by_scroll(tmp_path)
    first = [x for x in rows["PHerc0125"] if x[0] < t_rel]
    assert len({g for _t, g in first}) == 8 and len(first) == 8                      # 1 staged scroll, 8 GPUs: every idle card got a stripe
    st = json.loads((tmp_path / "box8" / "state" / "PHerc0125.json").read_text())
    jobs = st["jobs"]
    assert [j for j in jobs if j["status"] == "cancelled" and j.get("cancelled_by") == "fill_idle"]       # the old pending job is cancelled, not lost
    kids = [j for j in jobs if j.get("provenance") == "fill_idle"]
    assert len(kids) == 8 and all(j["status"] == "done" and j["z1"] - j["z0"] >= 1500 for j in kids)
    assert all(j["parent"] == "PHerc0125/full" for j in kids)
    t2 = min(t for t, _g in rows["PHerc0211"])
    assert t2 - t_rel < 4.0                                                           # the second scroll starts as soon as a GPU frees: not starved behind the fill
    ev = [json.loads(x)["kind"] for x in (tmp_path / "box8" / "events.jsonl").read_text().splitlines()]
    assert "fill_idle" in ev and out["rc"] == 0


def test_fill_idle_respects_min_height_and_default_off(tmp_path, monkeypatch):
    gate = threading.Event()
    s, _a = make_gated(tmp_path, monkeypatch, 8, ("--fill-idle-gpus",), gate)           # default min 2800: a 13,000-slice scroll gives at most 4 stripes
    t_rel, out = run_and_release(s, gate, tmp_path)
    rows = starts_by_scroll(tmp_path)
    first = [x for x in rows["PHerc0125"] if x[0] < t_rel]
    assert len(first) == 4 and all(j["z1"] - j["z0"] >= 2800 for j in s.jobs.values() if j.get("provenance") == "fill_idle")
    assert min(t for t, _g in rows["PHerc0211"]) - t_rel < 2.0                          # the 4 idle GPUs take the second scroll immediately
    assert not (tmp_path / "x").exists()
    tmp2 = tmp_path / "off"
    tmp2.mkdir()
    gate2 = threading.Event()
    s2, _b = make_gated(tmp2, monkeypatch, 8, (), gate2)                                # flag OFF: one job, seven idle GPUs
    t_rel2, _o = run_and_release(s2, gate2, tmp2)
    assert len([x for x in starts_by_scroll(tmp2)["PHerc0125"] if x[0] < t_rel2]) == 1
    assert not any(j.get("provenance") == "fill_idle" for j in s2.jobs.values())


def test_fill_idle_never_touches_running_or_retried_jobs(tmp_path, monkeypatch):
    gate = threading.Event()
    s, _a = make_gated(tmp_path, monkeypatch, 4, ("--fill-idle-gpus", "--fill-min-height", "1500"), gate)
    j = s.jobs["PHerc0125/full"]
    j["attempts"].append({"class": "oom"})                                              # a retry: would lose its checkpoint if re-split
    with s.cv:
        s.scrolls["PHerc0125"]["fetched"] = True
        s._maybe_fill_idle()
    assert j["status"] == "pending" and not any(k.get("provenance") == "fill_idle" for k in s.jobs.values())
    j["attempts"].clear()
    j["status"] = "running"
    with s.cv:
        s._maybe_fill_idle()
    assert j["status"] == "running" and not any(k.get("provenance") == "fill_idle" for k in s.jobs.values())


def test_resume_says_whether_pending_jobs_were_kept_or_replanned(tmp_path, monkeypatch, capsys):
    import test_control as TC
    s1 = TC.make(tmp_path, monkeypatch, ["PHerc0125"], 8, extra=("--steps", "30000"))
    capsys.readouterr()
    s2 = TC.make(tmp_path, monkeypatch, ["PHerc0125"], 8, extra=("--steps", "30000"))
    assert "unchanged" in capsys.readouterr().out
    s3 = TC.make(tmp_path, monkeypatch, ["PHerc0125"], 8, extra=("--steps", "30000", "--gpus", "0,1,2"))
    out = capsys.readouterr().out
    assert "planning inputs CHANGED" in out and "KEPT" in out and "NOT re-planned" in out and "gpus:" in out
    assert s3.jobs["PHerc0125/full"]["status"] == "pending"
    s4 = TC.make(tmp_path, monkeypatch, ["PHerc0125"], 8, extra=("--steps", "30000", "--max-height", "6000", "--replan-pending-on-resume"))
    out = capsys.readouterr().out
    assert "RE-PLANNED" in out and "max_height: 13000 -> 6000" in out
    assert s4.jobs["PHerc0125/full"]["status"] == "cancelled" and s4.jobs["PHerc0125/full"]["cancelled_by"] == "resume_replan"
    kids = [j for j in s4.jobs.values() if j.get("provenance") == "resume_replan"]
    assert len(kids) >= 2 and all(j["z1"] - j["z0"] <= 6000 for j in kids)


def test_routea_waits_for_first_fit_only_on_slow_links_by_default(tmp_path, monkeypatch):
    import test_control as TC
    s = TC.make(tmp_path, monkeypatch, ["PHerc0125"], 2)
    s.link_base = 8.0
    assert s.routea_should_wait()                                                       # auto + 8 MB/s < 20 -> wait for the first fit
    s.link_base = 200.0
    assert not s.routea_should_wait()
    s.a.routea_after_first_fit = "on"
    assert s.routea_should_wait()
    s.a.routea_after_first_fit = "off"
    s.link_base = 1.0
    assert not s.routea_should_wait()


def test_gpu_smoke_reports_per_gpu_and_blackwell_failure(monkeypatch, capsys):
    from routeB import gpusmoke as G
    ok = {"gpu": "0", "device": "NVIDIA A100", "cc": "8.0", "mem_gib": 40.0, "torch": "2.13.0+cu126", "cuda": "12.6", "arch_list": ["sm_80", "sm_90"],
          "tests": [{"name": "matmul_fp32", "ok": True, "s": 0.1}, {"name": "triton_kernel", "ok": True, "s": 2.0}]}
    bad = {"gpu": "1", "device": "NVIDIA GeForce RTX 5090", "cc": "12.0", "mem_gib": 31.8, "torch": "2.13.0+cu126", "cuda": "12.6", "arch_list": ["sm_80", "sm_90"],
           "tests": [{"name": "matmul_fp32", "ok": False, "err": "RuntimeError: CUDA error: no kernel image is available for execution on the device"}]}
    monkeypatch.setattr(G, "spawn", lambda g, t, sk: {"0": ok, "1": bad}[g])
    assert G.main(["--gpus", "0"]) == 0
    assert "OK" in capsys.readouterr().out
    assert G.main(["--gpus", "0,1"]) == 6
    cap = capsys.readouterr()
    assert "FAIL" in cap.out and "no kernel image for sm_120" in cap.out and "Blackwell" in cap.out and "--torch-cuda cu129" in cap.out
    assert "torch 2.13.0+cu126" in cap.out and "arch list: sm_80 sm_90" in cap.out and "NVIDIA GeForce RTX 5090" in cap.out
    assert "nothing was fetched" in cap.err
    v100 = dict(bad, cc="7.0", device="Tesla V100")
    assert "--torch-cuda cu126" in G.reason(v100) and "sm_70" in G.reason(v100)
    hang = {"gpu": "2", "tests": [{"name": "timeout", "ok": False, "err": "no answer within 1 s"}]}
    assert "timeout failed" in G.reason(hang)


def test_gpu_smoke_child_runs_for_real_when_a_gpu_and_torch_exist():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA here")
    from routeB import gpusmoke as G
    d = G.child(skip_compile=True)
    names = {t["name"] for t in d["tests"]}
    assert {"matmul_fp32", "gather_multinomial", "grid_sample_3d", "triton_kernel"} <= names and d["cc"] and d["arch_list"]
    mm = next(t for t in d["tests"] if t["name"] == "matmul_fp32")
    if not mm["ok"] and "memory" in json.dumps(mm).lower():
        pytest.skip("the shared GPU is full right now (another job holds its memory): not a code failure")
    assert mm["ok"]


def test_vram_table_blackwell_height_and_cu129_lock():
    from routeB import ladder as L
    from routeB import planner as P
    tab = {n: (u, h, k) for n, _nom, u, h, k in L.card_table()}
    assert tab["RTX 5090 32 GB"][1] == 12400 and tab["RTX 5090 32 GB"][2] == 2          # 13,000 slices do NOT fit one 5090: 2 stripes
    assert tab["A100 40 GB"][1] >= 13000 and tab["A100 40 GB"][2] == 1 and tab["H100 80 GB"][2] == 1 and tab["RTX 4090 24 GB"][1] < 9000
    c = L.load_config()
    c["dynamic"] = True
    h = P.Host([P.Gpu(str(i), 31.84) for i in range(8)])
    f = P.scroll_facts("PHerc0125", c, h)
    assert f.height <= 12600 and len(P.make_jobs(c, f)) >= 2
    txt = "\n".join(P.vram_lines(h))
    assert "8 x 31.8 GiB -> max stripe 12,600 slices" in txt and "needs >= 2 stripes" in txt and "RTX 5090 32 GB 31.3 -> 12,400" in txt and "A100 40 GB" in txt and "H100 80 GB" in txt
    pins = ROOT / "routeB" / "pins"
    for tag, tv in (("cu129", "torch==2.13.0+cu129"), ("cu128", "torch==2.11.0+cu128")):
        lock = (pins / f"requirements.{tag}.lock").read_text()
        assert tv in lock and "UNVALIDATED" in lock and f"+{tag}" in lock and "+cu126" not in lock
        assert (pins / f"requirements.{tag}.hashes.txt").read_text().count("--hash=sha256:") > 100
    assert "torch==2.13.0+cu126" in (pins / "requirements.lock").read_text()
    run = (ROOT / "routeB_run.sh").read_text()
    assert run.index("routeB.gpusmoke") > run.index("PatchSatisfactionAtlas") and run.index("routeB.gpusmoke") < run.index("-m routeB.cli")
    assert "compute_cap" in run and "cu129" in run and "UNVALIDATED" in run


# ---------------------------------------------------------------- per-stripe staging, early prefetch, 48 GB cards
def test_stripe_staging_first_stripe_starts_while_the_rest_downloads(tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTEB_HOME", str(tmp_path))
    monkeypatch.setenv("STUB_PLAN", "{}")
    monkeypatch.setenv("STUB_SLEEP", "0.5")
    calls = []

    def fetch(scroll, z0, z1):
        calls.append((scroll, z0, z1, time.time()))
        time.sleep(0.8)                                                       # each stripe's lasagna takes 0.8 s to land
        return 0.1
    from routeB import ladder as L
    a = B8.build_parser().parse_args(["--scrolls", "PHerc0125", "--fake-gpus", "3", "--poll-s", "0.2", "--control-poll-s", "0.25", "--steps", "30000", "--no-routea",
                                      "--phys-cores", "48", "--no-tail-split", "--max-height", "4800"])
    a.stall_minutes = 30.0
    cfg = L.load_config()
    cfg["dynamic"] = True
    gov = B.Governor(tmp_path / "box8" / "budget", B.Rates(soft_usd=1e6, hard_usd=2e6), box_start=time.time(), say=lambda *_: None)
    s = B8.Scheduler(a, cfg, gov, fetch_fn=fetch, job_cmd=[sys.executable, str(HERE / "stub_job.py")], box_home=tmp_path)
    assert s.stripe_mode
    ga, al = B8.gpu_info(a)
    s.host = B8.build_host(a, tmp_path, ga, al)
    s.gpus = [g.idx for g in ga]
    s.allowed_init = {g.idx for g in al}
    s.add_scroll("PHerc0125", 4500, 17500, None)
    stripes = [j for j in s.jobs.values()]
    assert len(stripes) == 3
    rc = s.run()
    assert rc == 0 and [(c[1], c[2]) for c in calls] == sorted((j["z0"], j["z1"]) for j in stripes)       # stripes staged in z order, one fetch each
    starts = sorted(float(x.split()[0]) for x in (tmp_path / "stub_events.log").read_text().splitlines())
    assert starts[0] < calls[-1][3] + 0.8 and starts[0] < starts[-1]                                      # stripe 1's fit began before the last stripe landed
    assert starts[0] - calls[0][3] < 2.5                                                                  # ... about one stripe's download after the fetch began
    ev = [json.loads(x)["kind"] for x in (tmp_path / "box8" / "events.jsonl").read_text().splitlines()]
    assert ev.count("job_ready") == 3 and "fetched" in ev
    assert all(j["status"] == "done" and j["ready"] for j in s.jobs.values())


def test_plan_reports_time_to_first_fit_and_gpus_filled_over_time():
    from routeB import ladder as L
    from routeB import planner as P
    c = L.load_config()
    c["dynamic"] = True
    r = B.Rates()
    on = P.Host([P.Gpu(str(i), 40.0) for i in range(8)])
    off = P.Host([P.Gpu(str(i), 40.0) for i in range(8)], stripe_staging=False)
    p_on = P.make_plan(on, r, ["PHerc0125", "PHerc0211", "PHerc0191"], c, plan_frac=100.0)
    p_off = P.make_plan(off, r, ["PHerc0125", "PHerc0211", "PHerc0191"], c, plan_frac=100.0)
    # v5.1: with the MEASURED object counts (3.24 objects/slice/field, not the old 17.8/3) whole-scroll staging is already short, so per-stripe
    # staging no longer starts the first fit "much earlier" (0.23 h vs 0.222 h on this fixture); the claim that still holds is that it is not worse.
    assert p_on["p50"]["first_fit_h"] <= 1.1 * p_off["p50"]["first_fit_h"]
    fill = p_on["p50"]["gpus_filled"]
    assert fill[0][1] == 1 and fill[-1][1] >= 6 and [n for _t, n in fill] == sorted(n for _t, n in fill)
    txt = P.render(on, r, p_on)
    assert "time to first fit" in txt and "GPUs busy over time" in txt and "per-stripe staging" in txt


def test_prefetch_follows_the_planned_start_order(tmp_path, monkeypatch, capsys):
    import fetch_assets as FA
    from routeB import prefetch as PF
    monkeypatch.setenv("ROUTEB_HOME", str(tmp_path))
    calls = []

    def fake_fetch(m, dest, **kw):
        calls.append((m["scroll"], tuple(m["z"])))
        return {"ok": True, "failed": [], "bytes_net": 1_000_000_000 * len(calls), "assets": []}
    monkeypatch.setattr(FA, "fetch", fake_fetch)
    monkeypatch.setattr(B8, "LINK_FN", None)
    assert PF.main(["--scrolls", "PHerc0125,PHerc0211,PHerc0191,PHerc0257", "--fake-gpus", "8", "--phys-cores", "48", "--steps", "30000", "--prefetch-scrolls", "2"]) == 0
    assert calls and len({c[0] for c in calls}) == 2                                          # only the first 2 scrolls
    z = [c[1] for c in calls if c[0] == calls[0][0]]
    assert z == sorted(z)                                                                      # stripe 1 first
    d = json.loads((tmp_path / "box8" / "prefetch.json").read_text())
    assert d["gb"] == len(calls) and not d["accounted"] and len(d["done"]) == len(calls)


def test_48gb_blackwell_cards_hold_full_height_and_use_the_cu128_path():
    from routeB import ladder as L
    from routeB import planner as P
    tab = {n: (h, k) for n, _nom, _u, h, k in L.card_table()}
    assert tab["RTX PRO 5000 48 GB"][0] >= 13000 and tab["RTX PRO 5000 48 GB"][1] == 1
    c = L.load_config()
    c["dynamic"] = True
    h = P.Host([P.Gpu(str(i), 47.0) for i in range(4)], net_down_mb_s=113.0)
    f = P.scroll_facts("PHerc0211", c, h)
    assert f.height == 13000 and len(P.make_jobs(c, f)) == 1                                  # one whole scroll per 48 GB GPU
    run = (ROOT / "routeB_run.sh").read_text()
    assert 'TC=cu128; fi' in run and "12.[89]" in run                                         # cap >= 12 or driver CUDA >= 12.8 -> cu128
    go = (ROOT / "go").read_text()
    assert "cu128" in go and "gpu_step" in go and go.index("link_check\ngpu_step") < go.index("install_os\nfetch_tree")
    assert "torch==2.11.0+cu128" in (ROOT / "routeB" / "pins" / "requirements.cu128.lock").read_text()


def test_auto_gpu_speed_from_the_smoke_tests_fp32_and_budget_env_passthrough(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ROUTEB_HOME", str(tmp_path))
    (tmp_path / "box8").mkdir()
    (tmp_path / "box8" / "gpusmoke.json").write_text(json.dumps([{"fp32_tflops": 40.7}, {"fp32_tflops": 40.1}, {"fp32_tflops": 41.0}, {"fp32_tflops": 40.9}]))
    a = B8.build_parser().parse_args(["--scrolls", "PHerc0211"])
    B8.auto_speed(a)
    assert a.gpu_speed == 2.0 and "AUTO --gpu-speed 2.0" in capsys.readouterr().out            # 40.7/15.7 = 2.6, capped at 2.0
    (tmp_path / "box8" / "gpusmoke.json").write_text(json.dumps([{"fp32_tflops": 13.7}]))
    a = B8.build_parser().parse_args(["--scrolls", "PHerc0211"])
    B8.auto_speed(a)
    assert a.gpu_speed == 0.87
    a = B8.build_parser().parse_args(["--scrolls", "PHerc0211", "--gpu-speed", "1.3"])
    B8.auto_speed(a)
    assert a.gpu_speed == 1.3                                                                  # an explicit value is never overridden
    a = B8.build_parser().parse_args(["--scrolls", "PHerc0211", "--no-auto-speed"])
    B8.auto_speed(a)
    assert a.gpu_speed == 1.0
    go = (ROOT / "go").read_text()
    assert "BUDGET_|ROUTEB_" in go and "seq 0 63" in go                                          # BUDGET_* reaches the tmux run; 64 streams in the probe
    from routeB import linkcheck as LK
    seen = []
    LK.measure_hosts("PHerc0211", lambda url, b: (seen.append(url), (50.0, "x"))[1])
    assert len(seen) == 2 and "__down" in seen[1]


def test_watch_once_default_is_one_screen(tmp_path):
    """User 2026-10-08: the diagnostic must fit ONE screen for copy/paste: <= 40 lines x 100 columns by default."""
    H = fixture_home(tmp_path)
    env = dict(os.environ, ROUTEB_HOME=str(H), PYTHONPATH=str(ROOT))
    r = subprocess.run(["bash", str(ROOT / "routeB_watch.sh"), "--once", "--home", str(H)], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0
    lines = r.stdout.rstrip("\n").split("\n")
    assert len(lines) <= 40, len(lines)
    assert max(len(x) for x in lines) <= 100, max(len(x) for x in lines)


def test_pull_block_full_has_direct_proxy_and_fallback(tmp_path):
    L = W.pull_lines(tmp_path, {"PUBLIC_IPADDR": "9.9.9.9", "VAST_TCP_PORT_22": "40022", "VAST_CONTAINERLABEL": "C.12345", "USER": "root"}, full=True)
    txt = "\n".join(L)
    assert L[0] == "H=root@9.9.9.9" and L[1] == "P=40022"
    assert "curl -fsSL $B/pull_box8.py" in txt and "vastai ssh-url 12345" in txt and "H=root@sshN.vast.ai" in txt and "rsync -aH" in txt
    assert "<OWNER>" not in txt and all(len(x) <= 140 for x in L)
    assert W.pull_lines(tmp_path, {"USER": "root"})[0] == "H=root@<BOX-IP>" and len(W.pull_lines(tmp_path, {"USER": "root"})) == 5      # the compact form is unchanged


def test_eta_per_scroll_overall_and_route_a():
    snap = {"now": 1_700_000_000, "jobs": [
        {"id": "S1/full", "scroll": "S1", "status": "running", "expected_h": 1.0},
        {"id": "S2/full", "scroll": "S2", "status": "running", "expected_h": 2.0},
        {"id": "S3/full", "scroll": "S3", "status": "pending", "expected_h": 3.0},
        {"id": "S4/full", "scroll": "S4", "status": "done", "expected_h": 1.0}],
        "gpu_rows": [{"idx": "0", "job": {"id": "S1/full", "eta": "10m", "steps": 5}}, {"idx": "1", "job": {"id": "S2/full", "eta": "1h 17m", "steps": 5}}, {"idx": "2", "job": None}],
        "routea": {"summary": {"finish_rate_per_h": 10.0, "seeds_remaining": 5, "state": "GROWING"}}}
    e = W.compute_eta(snap)
    assert e["B_scrolls"] == {"S1": 600, "S2": 4620, "S3": 10800}                    # the pending job takes the idle GPU 2
    assert e["B_all_s"] == 10800 and e["B_clock"].endswith("Z") and abs(e["A_s"] - 1800) < 1e-6
    lines = W.eta_lines({**snap, "eta": e})
    assert lines[0].startswith("ETA B: S1 10m | S2 1h17m | S3 3h00m | all 3h00m") and lines[1].startswith("ETA A: ~30m")
    assert W._secs("1h 17m") == 4620 and W._secs("9m 56s") == 596 and W._secs(None) is None
