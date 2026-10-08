"""box8 mode: fits-and-segments-only, N GPUs in parallel, work-stealing queue, fallback ladder, budget governor, resumable, payload for pull.

  ./routeB_run.sh --mode box8 --scrolls PHerc0211,PHerc0125,... [--steps 30000] [--soft 45 --hard 49] [--dry-run]

WHAT RUNS.  Per scroll: fetch (deploy_common/fetch_assets.py via cli.stage_fetch, prefetched by a small pool so GPUs do not wait) -> JOBS.  A job is
(scroll, tag, z0, z1) on one rung of the ladder (routeB/ladder.py + deploy_common/ladder_config.json): fit_spiral on one GPU, then tile_windings.py.
Each GPU worker thread takes the next pending job the moment it is free (work stealing; nothing is bound to a GPU), so no GPU idles while a
fetched, budget-admissible job remains.  A failed job is classified (multinomial 2^24 / OOM / host OOM / stall / env / unknown) and either retried
(memory-lean overrides), descended (the failed z-interval is re-covered by the next narrower rung, new jobs join the same queue) or failed loudly.
STATE (resumable): <home>/box8/state/<scroll>.json (jobs, attempts, rung that succeeded), <home>/box8/events.jsonl, <home>/box8/budget/budget.jsonl.
OUTPUT: <home>/box8/payload/<scroll>/{PAYLOAD.json, <tag>/..., DONE}; pull_box8.py (run from OUR side) polls for DONE.  The box never initiates a connection to us.
Only things that were MEASURED are claimed in the docs; assumptions are labelled in ladder_config.json.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from . import ladder as L
from .common import ROOT, UP_HI, UP_LO, home, is_done, say, spec


def _budget():
    for p in (ROOT / "deploy_common", ROOT.parent.parent / "deploy_common"):
        if (p / "budget.py").exists():
            sys.path.insert(0, str(p))
            break
    import budget
    return budget


# ------------------------------------------------------------------------------------------------ helpers
def discover_gpus(explicit: str | None = None, fake: int = 0) -> list[str]:
    if fake:
        return [str(i) for i in range(fake)]
    if explicit:
        return [g for g in explicit.split(",") if g != ""]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SystemExit(f"ROUTEB FAIL box8-gpus: cannot run nvidia-smi ({e})")
    if out.returncode != 0 or not out.stdout.strip():
        raise SystemExit(f"ROUTEB FAIL box8-gpus: nvidia-smi rc={out.returncode}: {out.stderr.strip()[:200]}")
    rows = [r.split(",") for r in out.stdout.strip().splitlines()]
    say("GPUs: " + "; ".join(f"{r[0].strip()}={r[1].strip()} {r[2].strip()}" for r in rows), "box8")
    return [r[0].strip() for r in rows]


def meminfo_gb() -> tuple[float, float]:
    d = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        d[k] = int(v.split()[0]) / 1048576.0
    return d["MemAvailable"], d["MemTotal"]


def pgroup_rss_gb(pgid: int) -> float:
    pg = 0
    tot = 0
    page = os.sysconf("SC_PAGE_SIZE")
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            st = open(f"/proc/{p}/stat").read()
            f = st[st.rindex(")") + 2:].split()
            if int(f[2]) == pgid:
                tot += int(open(f"/proc/{p}/statm").read().split()[1]) * page
        except (OSError, ValueError, IndexError):
            continue
    return tot / 1e9


def tail_text(p: Path, n: int = 65536) -> str:
    try:
        with open(p, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - n))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


PROG_RX = None
import re as _re
LOADED_RX = _re.compile(r"loaded ([\d,]+) tracks within z-roi")


def parse_progress(text: str):
    """last 'PROGRESS Optimizing — 1,489/1,500 iterations' -> (it, total) or None"""
    global PROG_RX
    import re
    if PROG_RX is None:
        PROG_RX = re.compile(r"PROGRESS Optimizing\s+\S+\s+([\d,]+)/([\d,]+) iterations")
    m = None
    for m in PROG_RX.finditer(text):
        pass
    return (int(m.group(1).replace(",", "")), int(m.group(2).replace(",", ""))) if m else None


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(f"{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    t.write_text(json.dumps(obj, indent=1))
    os.replace(t, p)


# ------------------------------------------------------------------------------------------------ scheduler
class Scheduler:
    def __init__(self, a, cfg, gov, fetch_fn=None, job_cmd=None, box_home: Path | None = None):
        self.a, self.cfg, self.gov = a, cfg, gov
        self.H = box_home or home()
        self.D = self.H / "box8"
        for s in ("state", "logs", "payload"):
            (self.D / s).mkdir(parents=True, exist_ok=True)
        self.cv = threading.Condition()
        self.save_lock = threading.RLock()
        self.jobs: dict[str, dict] = {}
        self.order: list[str] = []
        self.scrolls: dict[str, dict] = {}            # scroll -> {fetched: bool, fetching: bool, fetch_failed: str|None, shell, finalized}
        self.procs: dict[str, subprocess.Popen] = {}
        self.stop_reason: str | None = None
        self.fetch_fn = fetch_fn or self._fetch_real
        self.job_cmd = job_cmd
        self.refused_announced: set[str] = set()
        self.gpus: list[str] = []
        self.busy: dict[str, str] = {}               # gpu -> job id
        self.t0 = time.time()

    # ---- events / state
    dry = False

    def ev(self, kind, **kw):
        if self.dry:
            return
        row = {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "kind": kind, **kw}
        with open(self.D / "events.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")

    def save(self, scroll: str):
        with self.save_lock:
            self._save(scroll)

    def _save(self, scroll: str):
        if self.dry:
            return
        js = [self.jobs[i] for i in self.order if self.jobs[i]["scroll"] == scroll]
        sc = self.scrolls[scroll]
        ok = [j for j in js if j["status"] == "done"]
        atomic_json(self.D / "state" / f"{scroll}.json", {
            "scroll": scroll, "shell": sc["shell"], "fetched": sc["fetched"], "fetch_failed": sc.get("fetch_failed"), "finalized": sc.get("finalized"),
            "rung_succeeded": {j["tag"]: j["rung_name"] for j in ok}, "jobs": js})

    def load_state(self, scroll: str) -> bool:
        p = self.D / "state" / f"{scroll}.json"
        if not p.exists():
            return False
        st = json.loads(p.read_text())
        self.scrolls[scroll].update(fetched=st.get("fetched", False), finalized=None)
        for j in st["jobs"]:
            if j["status"] in ("running", "deferred", "hard_stopped"):
                j["status"] = "pending"
                j.setdefault("notes", []).append("requeued on restart")
            if j["status"] == "pending":                  # the budget basis may have changed since (smoke vs full, --steps): re-derive, never trust the stale number
                j["expected_h"] = L.expected_h(self.cfg, scroll, self.scrolls[scroll]["shell"], j["rung"])
                self.scale_expected(j)
            self.jobs[j["id"]] = j
            self.order.append(j["id"])
        n = {k: sum(1 for j in st["jobs"] if j["status"] == k) for k in ("done", "pending", "failed", "descended")}
        say(f"{scroll}: RESUMED from state: {n}", "box8")
        return True

    # ---- setup
    def scale_expected(self, j: dict) -> None:
        """expected_h comes from the 30000-step production fits; a smoke / short-step run must not be budgeted as a full fit (found in the pny rehearsal:
        the governor refused a 600-step smoke because it was priced at 5.5 GPU-h).  Smoke = 0.9 h (track load ~5 min + first-step compile 6-9 min + 1500
        steps at 0.7-1.5 it/s, all MEASURED on pny); other step counts: linear in steps + 0.25 h fixed (ASSUMED)."""
        if self.a.smoke:
            j["expected_h"] = 0.9
        elif self.a.steps != 30000:
            j["expected_h"] = j["expected_h"] * self.a.steps / 30000.0 + 0.25

    def add_scroll(self, scroll: str, z0: int, z1: int, first_rung: str | None):
        sp = spec(scroll)
        sc = {"fetched": False, "fetching": False, "shell": int(sp["shell_outer_winding_idx"]), "finalized": None}
        self.scrolls[scroll] = sc
        if self.load_state(scroll):
            return
        dbm = sum(v for k, v in sp["tracks"]["files"].items() if k.endswith(".dbm"))
        start, why = (L.rung_index(self.cfg, first_rung), "forced by --first-rung") if first_rung else L.first_rung(self.cfg, scroll, z0, z1)
        jobs = L.plan(self.cfg, scroll, sc["shell"], z0, z1, start=start)
        for j in jobs:
            self.scale_expected(j)
            j["dbm_bytes"] = dbm
            self.jobs[j["id"]] = j
            self.order.append(j["id"])
        say(f"{scroll}: {len(jobs)} job(s) on rung {self.cfg['rungs'][start]['name']} ({why}); expected {sum(j['expected_h'] for j in jobs):.1f} GPU-h", "box8")
        self.ev("plan", scroll=scroll, rung=self.cfg["rungs"][start]["name"], why=why, jobs=[j["id"] for j in jobs])
        self.save(scroll)

    # ---- fetch
    def _fetch_real(self, scroll: str) -> float:
        from . import cli
        ns = SimpleNamespace(with_crossings=False, full_lasagna=False)
        before = _total_net(self.H)
        cli.stage_fetch(scroll, UP_LO if self.a.z0 is None else self.a.z0, UP_HI if self.a.z1 is None else self.a.z1, ns)
        return max(0.0, _total_net(self.H) - before) / 1e9

    def fetch_worker(self, scroll: str):
        sc = self.scrolls[scroll]
        err = None
        for attempt in (1, 2, 3):
            try:
                gb = self.fetch_fn(scroll)
                self.gov.transfer(box_download_gb=gb, what=f"fetch {scroll}")
                with self.cv:
                    sc["fetched"], sc["fetching"] = True, False
                    self.cv.notify_all()
                self.ev("fetched", scroll=scroll, gb=gb)
                with self.cv:
                    self.save(scroll)
                return
            except BaseException as e:               # noqa: BLE001 - reported, then retried; a failed fetch never kills the other scrolls
                err = f"{type(e).__name__}: {e}"
                say(f"FETCH FAILED {scroll} (attempt {attempt}/3): {err[:300]}", "box8")
                time.sleep(5 if attempt < 3 else 0)
        with self.cv:
            sc["fetching"], sc["fetch_failed"] = False, err
            for j in self.jobs.values():
                if j["scroll"] == scroll and j["status"] == "pending":
                    j["status"] = "failed"
                    j["fail_why"] = f"fetch failed: {err[:200]}"
            self.cv.notify_all()
        self.ev("fetch_failed", scroll=scroll, err=err)
        with self.cv:
            self.save(scroll)

    def start_fetches(self):
        self.fpool = ThreadPoolExecutor(max_workers=self.a.fetch_workers)
        for s, sc in self.scrolls.items():
            if sc["fetched"]:
                continue
            sc["fetching"] = True
            self.fpool.submit(self.fetch_worker, s)

    # ---- claim
    def claim(self, gpu: str):
        warned = 0.0
        with self.cv:
            while True:
                if self.stop_reason:
                    return None
                pend = [self.jobs[i] for i in self.order if self.jobs[i]["status"] == "pending"]
                ready = [j for j in pend if self.scrolls[j["scroll"]]["fetched"]]
                waiting_fetch = [j for j in pend if self.scrolls[j["scroll"]]["fetching"]]
                for j in ready:
                    ram_ok, ram_why = self._ram_ok()
                    if not ram_ok:
                        if time.time() - warned > 60:
                            say(f"GPU {gpu}: waiting for RAM: {ram_why}", "box8")
                            warned = time.time()
                        break
                    ok, why = self.gov.may_launch(j["id"], j["expected_h"], j["payload_gb"], 0.0)
                    if ok:
                        j["status"] = "running"
                        j["gpu"] = gpu
                        self.busy[gpu] = j["id"]
                        self.gov.fit_start(j["id"], j["expected_h"], j["payload_gb"])
                        self.ev("launch", job=j["id"], gpu=gpu, why=why)
                        say(why, "budget")
                        self.save(j["scroll"])
                        return j
                    j["status"] = "deferred"
                    j["fail_why"] = why
                    self.ev("defer", job=j["id"], why=why)
                    if j["id"] not in self.refused_announced:
                        self.refused_announced.add(j["id"])
                        say("NOT LAUNCHED (budget): " + why, "budget")
                    self.save(j["scroll"])
                running = [j for j in self.jobs.values() if j["status"] == "running"]
                still = [i for i in self.order if self.jobs[i]["status"] == "pending"]
                if not still and not running and not waiting_fetch:
                    return None
                self.cv.wait(timeout=5)

    def _ram_ok(self):
        if self.a.fake_gpus:
            return True, ""
        avail, total = meminfo_gb()
        need = self.a.ram_need_gb
        if avail < need:
            return False, f"MemAvailable {avail:.0f} GB < need {need:.0f} GB per fit"
        du = shutil.disk_usage(self.H)
        if du.free / 1e9 < self.a.min_free_gb:
            return False, f"free disk {du.free / 1e9:.0f} GB on {self.H} < --min-free-gb {self.a.min_free_gb:.0f}"
        return True, ""

    # ---- run one job
    def job_dirs(self, j):
        rd = self.H / "runs" / j["scroll"] / j["tag"]
        return rd, rd / "fit" / "fit.log", self.D / "logs" / (j["id"].replace("/", "__") + ".log")

    def run_job(self, gpu: str, j: dict) -> tuple[str, dict]:
        rd, fitlog, jlog = self.job_dirs(j)
        rung = self.cfg["rungs"][j["rung"]]
        ov = {**rung.get("fit_overrides", {}), **j.get("extra_overrides", {})}
        att = {"n": len(j["attempts"]) + 1, "gpu": gpu, "overrides": ov, "t0": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if j.pop("fresh", False) and (rd / "fit").exists():
            old = rd / f"fit.attempt{len(j['attempts'])}"
            shutil.rmtree(old, ignore_errors=True)
            for c in (rd / "fit").rglob("*.ckpt"):
                c.unlink()
            (rd / "fit").rename(old)
            say(f"{j['id']}: previous fit dir kept as {old.name} (checkpoints removed)", "box8")
        jj = self.D / "logs" / (j["id"].replace("/", "__") + f".a{att['n']}.json")
        jj.write_text(json.dumps({"job": j, "overrides": ov, "steps": self.a.steps, "smoke": bool(self.a.smoke)}))
        cmd = (self.job_cmd or [os.environ.get("ROUTEB_PYTHON", sys.executable), "-m", "routeB.box8", "job"]) + ["--job-json", str(jj), "--gpu", gpu]
        if self.a.smoke:
            cmd += ["--smoke"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1")
        with open(jlog, "ab") as lf:
            lf.write(f"\n=== attempt {att['n']} gpu {gpu} {att['t0']} overrides {json.dumps(ov)}\n".encode())
            lf.flush()
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True, cwd=str(ROOT))
        self.procs[j["id"]] = p
        say(f"{j['id']}: rung {j['rung_name']} z[{j['z0']},{j['z1']}) on GPU {gpu} pid {p.pid} (attempt {att['n']}, overrides {ov})", "box8")
        last_sig, last_change, peak_rss, stalled, host_oom, early_mt = None, time.time(), 0.0, False, False, False
        t_start = time.time()
        stall_s = self.a.stall_minutes * 60.0
        while p.poll() is None:
            time.sleep(self.a.poll_s)
            src = fitlog if fitlog.exists() else jlog
            txt = tail_text(src)
            pr = parse_progress(txt)
            sig = (src.stat().st_size if src.exists() else 0, pr)
            if sig != last_sig:
                last_sig, last_change = sig, time.time()
            if pr and pr[1]:
                self.gov.progress(j["id"], pr[0] / pr[1])
            if not early_mt:
                m = LOADED_RX.search(txt) if LOADED_RX else None
                if m:
                    n_loaded = int(m.group(1).replace(",", ""))
                    j["n_loaded"] = n_loaded
                    if n_loaded > self.cfg["limits"]["multinomial_categories"]:
                        early_mt = True
                        say(f"{j['id']}: DETECT-EARLY: fit.log says {n_loaded:,} tracks loaded > {self.cfg['limits']['multinomial_categories']:,} (2^24 multinomial limit); killing and descending", "box8")
                        self._kill(p)
            rss = pgroup_rss_gb(p.pid)
            peak_rss = max(peak_rss, rss)
            if not self.a.fake_gpus:
                av, _tot = meminfo_gb()
                if av < self.a.ram_floor_gb and rss == peak_rss and rss > 0:
                    host_oom = True
                    say(f"{j['id']}: HOST RAM GUARD: MemAvailable {av:.1f} GB < floor {self.a.ram_floor_gb}; killing it (rss {rss:.0f} GB)", "box8")
                    self._kill(p)
            if time.time() - last_change > stall_s:
                stalled = True
                say(f"{j['id']}: WATCHDOG: no log growth/iteration progress for {self.a.stall_minutes:g} min (last progress {pr}); killing", "box8")
                self._kill(p)
            if self.stop_reason and not p.poll():
                self._kill(p)
        rc = p.returncode
        self.procs.pop(j["id"], None)
        att.update(rc=rc, wall_s=round(time.time() - t_start, 1), peak_rss_gb=round(peak_rss, 1), t1=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        txt = tail_text(fitlog, 20000) + "\n" + tail_text(jlog, 20000)
        if self.stop_reason:
            att["class"] = "hard_stop"
        elif early_mt:
            att["class"] = "multinomial"
            att["log_tail"] = f"detect-early: {j.get('n_loaded')} tracks loaded"
        elif rc == 0:
            att["class"] = "ok"
        else:
            att["class"] = L.classify(txt, rc, stalled, host_oom)
            att["log_tail"] = txt[-600:]
        j["attempts"].append(att)
        return att["class"], att

    def _kill(self, p: subprocess.Popen):
        try:
            os.killpg(p.pid, signal.SIGTERM)
            for _ in range(60):
                if p.poll() is not None:
                    return
                time.sleep(1)
            os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    # ---- after a job
    def settle(self, j: dict, cls: str, att: dict):
        with self.cv:
            self.busy.pop(j.get("gpu"), None)
            sc = self.scrolls[j["scroll"]]
            if cls == "ok":
                j["status"] = "done"
                self.gov.fit_end(j["id"], True, j["payload_gb"])
                say(f"{j['id']}: DONE rung {j['rung_name']} in {att['wall_s']:.0f} s, peak RSS {att['peak_rss_gb']} GB", "box8")
                self.ev("done", job=j["id"], rung=j["rung_name"], wall_s=att["wall_s"], peak_rss_gb=att["peak_rss_gb"])
            elif cls == "hard_stop":
                j["status"] = "hard_stopped"
                self.gov.fit_end(j["id"], False)
                self.ev("hard_stopped", job=j["id"])
            else:
                self.gov.fit_end(j["id"], False)
                d = L.decide(self.cfg, j, cls, sc["shell"], j.get("dbm_bytes"))
                self.ev("failure", job=j["id"], cls=cls, decision=d["action"], why=d["why"], tail=att.get("log_tail", "")[-300:])
                say(f"{j['id']}: FAILED class={cls} -> {d['action'].upper()}: {d['why']}", "box8")
                if d["action"] == "retry":
                    j["status"] = "pending"
                    j["extra_overrides"] = d["extra"]
                    j["fresh"] = d["fresh"]
                elif d["action"] == "descend":
                    j["status"] = "descended"
                    j["descended_to"] = d["to_rung"]
                    for k in d["jobs"]:
                        self.scale_expected(k)
                        k["dbm_bytes"] = j.get("dbm_bytes")
                        if k["id"] in self.jobs and self.jobs[k["id"]]["status"] == "done":
                            continue                    # stripe already fitted in an earlier descent
                        self.jobs[k["id"]] = k
                        self.order.append(k["id"])
                else:
                    j["status"] = "failed"
                    j["fail_why"] = d["why"]
                    say(f"{j['id']}: TERMINAL FAILURE: {d['why']}", "box8")
            self.save(j["scroll"])
            self.maybe_finalize(j["scroll"])
            self.cv.notify_all()

    def worker(self, gpu: str):
        while True:
            j = self.claim(gpu)
            if j is None:
                say(f"GPU {gpu}: no more admissible work, worker exits", "box8")
                return
            try:
                cls, att = self.run_job(gpu, j)
            except Exception as e:                       # noqa: BLE001 - scheduler bug must not strand the job as 'running'
                cls, att = "unknown", {"wall_s": 0, "peak_rss_gb": 0, "log_tail": f"scheduler exception {type(e).__name__}: {e}"}
                j["attempts"].append(att)
            self.settle(j, cls, att)

    # ---- payload / DONE
    def maybe_finalize(self, scroll: str):
        js = [j for j in self.jobs.values() if j["scroll"] == scroll]
        if any(j["status"] in ("pending", "running") for j in js) or self.scrolls[scroll].get("finalized"):
            return
        done = [j for j in js if j["status"] == "done"]
        bad = [j for j in js if j["status"] in ("failed", "deferred", "hard_stopped")]
        if not done:
            self.scrolls[scroll]["finalized"] = "nothing"
            self.ev("finalized", scroll=scroll, status="nothing", bad=[j["id"] for j in bad])
            return
        status = "complete" if not bad else "partial"
        try:
            n_files, tot = self.build_payload(scroll, done, bad, status)
        except Exception as e:                          # noqa: BLE001
            say(f"PAYLOAD FAILED {scroll}: {type(e).__name__}: {e}", "box8")
            self.ev("payload_failed", scroll=scroll, err=str(e))
            return
        self.scrolls[scroll]["finalized"] = status
        say(f"{scroll}: payload {status}: {n_files} files, {tot / 1e9:.3f} GB -> {self.D / 'payload' / scroll} (DONE marker written)", "box8")
        self.ev("finalized", scroll=scroll, status=status, files=n_files, bytes=tot)
        self.save(scroll)

    def build_payload(self, scroll: str, done: list[dict], bad: list[dict], status: str):
        pd = self.D / "payload" / scroll
        shutil.rmtree(pd, ignore_errors=True)
        pd.mkdir(parents=True)
        files: list[tuple[Path, Path]] = []
        segs = []
        for j in sorted(done, key=lambda j: (j["z0"], j["tag"])):
            rd = self.H / "runs" / scroll / j["tag"]
            dst = pd / j["tag"]
            for src in self._payload_sources(rd, j):
                rel = src.relative_to(rd)
                files.append((src, dst / rel))
            man = rd / "tiled" / "manifest.json"
            nt = None
            if man.exists():
                try:
                    nt = json.loads(man.read_text()).get("n_tiles")
                except ValueError:
                    pass
            segs.append({"tag": j["tag"], "rung": j["rung_name"], "z": [j["z0"], j["z1"]], "n_tiles": nt, "manifest": f"{j['tag']}/tiled/manifest.json",
                         "tile_name_stem": f"{scroll}_{j['tag']}_wNNN_zRRxC", "attempts": len(j["attempts"])})
        for src, dst in files:
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        # gzip the log tail (full logs can be large; the tail holds the metrics)
        for j in done:
            fl = self.job_dirs(j)[1]
            if fl.exists():
                with gzip.open(pd / j["tag"] / "fit.log.tail.gz", "wb") as g:
                    g.write(tail_text(fl, 400000).encode())
        with ThreadPoolExecutor(8) as ex:
            allf = sorted(p for p in pd.rglob("*") if p.is_file())
            ents = list(ex.map(lambda p: {"path": str(p.relative_to(pd)), "size": p.stat().st_size, "md5": md5_file(p)}, allf))
        tot = sum(e["size"] for e in ents)
        meta = {"scroll": scroll, "status": status, "segments": segs, "failed_intervals": [{"id": j["id"], "z": [j["z0"], j["z1"]], "why": j.get("fail_why")} for j in bad],
                "files": ents, "total_bytes": tot, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "note": "no checkpoints; tiled/manifest.json is the producer's verbatim manifest (box-side absolute paths inside are informational)"}
        atomic_json(pd / "PAYLOAD.json", meta)
        (pd / "DONE.tmp").write_text(f"{status} {len(ents)} files {tot} bytes\n")
        os.replace(pd / "DONE.tmp", pd / "DONE")
        return len(ents), tot

    @staticmethod
    def _payload_sources(rd: Path, j: dict):
        out = []
        for sub in ("tiled",):
            out += [p for p in (rd / sub).rglob("*") if p.is_file()]
        fit = rd / "fit"
        out += [p for p in (fit / "out").rglob("*") if p.is_file() and not p.name.endswith(".ckpt") and ("meshes" in p.parts or p.suffix in (".json", ".txt"))]
        for n in ("fit_config.json", ".done.fit.json"):
            if (fit / n).exists():
                out.append(fit / n)
        out += [p for p in rd.glob(".done.*.json")] + [p for p in (rd / "tiles.log",) if p.exists()]
        seen, uniq = set(), []
        for p in out:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        return uniq

    # ---- monitor: hard stop, STOP file, pulled markers
    def monitor(self):
        seen_pulled: set[str] = set()
        while not self.done_evt.is_set():
            time.sleep(self.a.poll_s)
            for pj in (self.D / "payload").glob("*/PULLED.json"):
                if str(pj) in seen_pulled:
                    continue
                seen_pulled.add(str(pj))
                try:
                    b = json.loads(pj.read_text()).get("bytes", 0)
                except ValueError:
                    continue
                self.gov.transfer(box_upload_gb=b / 1e9, what=f"pulled {pj.parent.name}")
            if (self.D / "STOP").exists() and not self.stop_reason:
                self._hard_stop("STOP file present (operator request)")
            elif self.gov.hard_stop() and not self.stop_reason:
                self.gov.note_hard_stop()
                self._hard_stop(f"HARD BUDGET STOP: spent ${self.gov.spent():.2f} (+unpulled payload) >= ${self.gov.r.hard_usd}")

    def _hard_stop(self, why: str):
        say(f"*** {why}: stopping running fits (SIGTERM, 60 s, SIGKILL). fit_spiral autosaves every {os.environ.get('FIT_SPIRAL_AUTOSAVE_INTERVAL', '1000')} steps; "
            f"anything newer than the last autosave is lost. Finished payloads stay pullable. ***", "box8")
        self.ev("stop", why=why)
        with self.cv:
            self.stop_reason = why
            self.cv.notify_all()

    # ---- main
    def run(self) -> int:
        self.done_evt = threading.Event()
        self.start_fetches()
        mon = threading.Thread(target=self.monitor, daemon=True)
        mon.start()
        ths = [threading.Thread(target=self.worker, args=(g,), name=f"gpu{g}") for g in self.gpus]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        self.done_evt.set()
        for s in self.scrolls:
            with self.cv:
                self.maybe_finalize(s)
        self.summary()
        st = [j["status"] for j in self.jobs.values()]
        if any(x in ("failed", "hard_stopped") for x in st) or any(sc.get("fetch_failed") for sc in self.scrolls.values()):
            return 1                                      # something failed loudly
        return 4 if "deferred" in st else 0               # 4 = nothing failed, but the budget governor left work unlaunched

    def summary(self):
        st = self.gov.status()
        by = {}
        for j in self.jobs.values():
            by[j["status"]] = by.get(j["status"], 0) + 1
        summ = {"wall_s": round(time.time() - self.t0), "jobs": by, "stop": self.stop_reason, "budget": st,
                "scrolls": {s: sc.get("finalized") or ("fetch_failed" if sc.get("fetch_failed") else "unfinished") for s, sc in self.scrolls.items()}}
        atomic_json(self.D / "ALLDONE.json", summ)
        say(f"ALLDONE {json.dumps(summ)}", "box8")


def _total_net(h: Path) -> float:
    p = h / "assets" / ".fetched" / "_totals.json"
    try:
        return float(sum(x.get("bytes_net", 0) for x in json.loads(p.read_text())))
    except (OSError, ValueError):
        return 0.0


# ------------------------------------------------------------------------------------------------ plan / CLI
def skip_reason(scroll: str) -> str | None:
    try:
        sp = spec(scroll)
    except Exception as e:                              # noqa: BLE001
        return str(e)
    if not sp.get("tracks"):
        return "no published tracks upstream (extract them first: README 'scrolls without published tracks')"
    if not sp.get("lasagna"):
        return "no published lasagna normal fields (nx/ny) upstream"
    if sp.get("spiral_outward_sense") not in ("CW", "ACW"):
        return "spiral_outward_sense unknown in the registry (needs the sense A/B fits; pass them via the pipeline mode with --sense)"
    if not sp.get("umbilicus_file") or not (ROOT / sp["umbilicus_file"]).is_file():
        return "no umbilicus file"
    return None


def build_parser():
    ap = argparse.ArgumentParser(prog="routeB_run.sh --mode box8")
    ap.add_argument("--scrolls", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--z0", type=int, default=None)
    ap.add_argument("--z1", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="z 9000-9500, 1500 steps, 2 windings, 1 tile: a proof run (jobs get --smoke)")
    ap.add_argument("--gpus", default=None, help="comma list of GPU indices (default: all from nvidia-smi)")
    ap.add_argument("--fake-gpus", type=int, default=0, help="TEST ONLY: N logical workers, no nvidia-smi, no RAM guard")
    ap.add_argument("--order", default="given", choices=["given", "spt", "lpt"], help="queue order: as listed | shortest expected first (most scrolls per $) | longest first")
    ap.add_argument("--first-rung", default=None, help="force the starting rung name (default: full, or the first under the track limit when n_tracks is known)")
    ap.add_argument("--ladder-config", default=None)
    ap.add_argument("--ladder", default=None, help="rungs in order, e.g. full,sw2800 (default from ladder_config.json) | 'all' = full,w13000,q4500,sw2800")
    ap.add_argument("--fetch-workers", type=int, default=2)
    ap.add_argument("--stall-minutes", type=float, default=None)
    ap.add_argument("--poll-s", type=float, default=5.0)
    ap.add_argument("--ram-need-gb", type=float, default=float(os.environ.get("ROUTEB_FIT_RAM_GB", 40)), help="MemAvailable needed to admit one fit (ASSUMED 40; peak RSS is logged per attempt so this can be set from data)")
    ap.add_argument("--ram-floor-gb", type=float, default=float(os.environ.get("ROUTEB_RAM_FLOOR_GB", 12)), help="below this MemAvailable the largest fit is killed (host_oom) instead of letting the kernel pick")
    ap.add_argument("--min-free-gb", type=float, default=200.0, help="do not admit a fit when ROUTEB_HOME has less free disk than this (fit dirs are written there)")
    ap.add_argument("--dry-run", action="store_true", help="print queue, skips, expected hours and the budget projection; launch nothing")
    ap.add_argument("--target-cm2", type=float, default=26.0)
    ap.add_argument("--rows", type=int, default=400)
    ap.add_argument("--windings", default=None)
    g = ap.add_argument_group("budget (deploy_common/budget.py; env BUDGET_* also read)")
    g.add_argument("--soft", type=float, default=None, help="stop launching new fits when the projection reaches this (USD, default 45)")
    g.add_argument("--hard", type=float, default=None, help="hard stop (USD, default 49)")
    g.add_argument("--hour-usd", type=float, default=None)
    g.add_argument("--ingress-per-tb", type=float, default=None, help="box DOWNLOADS from the web, USD/TB (default 2.70)")
    g.add_argument("--egress-per-tb", type=float, default=None, help="box UPLOADS to us (our payload pull), USD/TB (default 4.00)")
    g.add_argument("--swap-directions", action="store_true")
    g.add_argument("--max-run-hours", type=float, default=None, help="wall-clock hours since rental start; no fit is launched whose projected end passes it, running fits wind down (default 12; 0 = unlimited)")
    g.add_argument("--budget-config", default=None, help="json file with any of hour_usd, soft_usd, hard_usd, max_run_hours, ingress_per_tb, egress_per_tb, swap_directions (precedence: defaults < file < BUDGET_* env < flags)")
    g.add_argument("--box-start", type=float, default=None, help="epoch seconds when the rental started (default: first ledger write)")
    g.add_argument("--clock-scale", type=float, default=None, help="REHEARSAL: simulated seconds per real second (3600 = 1 s is 1 billed hour)")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    cfg = L.load_config(a.ladder_config, a.ladder)
    if a.stall_minutes is None:
        a.stall_minutes = cfg["watchdog"]["stall_minutes"]
    if a.smoke:
        a.z0, a.z1 = a.z0 or 9000, a.z1 or 9500
        a.steps = 1500 if a.steps == 30000 else a.steps
        a.windings = a.windings or "60:62"
        a.target_cm2 = min(a.target_cm2, 3.0)
        a.first_rung = a.first_rung or "full"
    z0 = UP_LO if a.z0 is None else a.z0
    z1 = UP_HI if a.z1 is None else a.z1
    B = _budget()
    if a.swap_directions:
        os.environ["BUDGET_SWAP_DIRECTIONS"] = "1"
    kw = B.Rates.from_file(a.budget_config) if a.budget_config else {}
    kw.update({k: v for k, v in dict(hour_usd=a.hour_usd, ingress_per_tb=a.ingress_per_tb, egress_per_tb=a.egress_per_tb,
                                     soft_usd=a.soft, hard_usd=a.hard, max_run_hours=a.max_run_hours).items() if v is not None})
    if a.swap_directions:
        kw["swap_directions"] = True
    rates = B.Rates.from_env(**kw)
    say(rates.describe(), "budget")
    H = home()
    clock = B.ScaledClock(a.clock_scale) if a.clock_scale else time.time
    import tempfile
    gov = B.Governor(Path(tempfile.mkdtemp(prefix="box8_dry_")) if a.dry_run else H / "box8" / "budget", rates, clock=clock, box_start=a.box_start, say=lambda m: say(m, "budget"))
    sch = Scheduler(a, cfg, gov)
    sch.dry = a.dry_run                                  # a dry run reads state (if any) but writes nothing
    names = [s.strip() for s in a.scrolls.split(",") if s.strip()]
    run_names = []
    for s in names:
        why = skip_reason(s)
        if why:
            say(f"SKIP {s}: {why}", "box8")
            sch.ev("skip", scroll=s, why=why)
        else:
            run_names.append(s)
    if not run_names:
        say("nothing to run: every requested scroll was skipped (see SKIP lines above)", "box8")
        return 1
    for s in run_names:
        sch.add_scroll(s, z0, z1, a.first_rung)
    if a.order != "given":
        tot = {}
        for i in sch.order:                          # snapshot first: list.sort() empties the list while it runs
            tot[sch.jobs[i]["scroll"]] = tot.get(sch.jobs[i]["scroll"], 0.0) + sch.jobs[i]["expected_h"]
        sch.order.sort(key=lambda i: tot[sch.jobs[i]["scroll"]], reverse=(a.order == "lpt"))
    tot_h = sum(j["expected_h"] for j in sch.jobs.values() if j["status"] == "pending")
    sch.gpus = discover_gpus(a.gpus, a.fake_gpus)
    ng = max(1, len(sch.gpus))
    wall = max(tot_h / ng, max([j["expected_h"] for j in sch.jobs.values() if j["status"] == "pending"] or [0]))
    proj = gov.projected(extra_expected_h=wall, extra_payload_gb=sum(j["payload_gb"] for j in sch.jobs.values()))
    say(f"PLAN: {len(sch.jobs)} job(s) over {len(run_names)} scroll(s), {tot_h:.1f} GPU-h expected on {ng} GPU(s) = {wall:.1f} h wall at perfect packing; "
        f"projected total ${proj['projected_total']:.2f} vs soft ${rates.soft_usd} / hard ${rates.hard_usd} (expected hours are EXTRAPOLATED, see ladder_config.json)", "box8")
    if proj["projected_total"] > rates.soft_usd:
        say(f"the whole queue does NOT fit the budget: the governor will stop launching once the projection reaches ${rates.soft_usd}; use --order spt to maximise finished scrolls", "box8")
    if a.dry_run:
        for i in sch.order:
            j = sch.jobs[i]
            say(f"  {j['id']:<30} rung {j['rung_name']:<7} z[{j['z0']},{j['z1']}) expected {j['expected_h']:.1f} GPU-h payload {j['payload_gb']} GB", "plan")
        return 0
    return sch.run()


# ------------------------------------------------------------------------------------------------ worker subcommand
def job_main(argv) -> int:
    ap = argparse.ArgumentParser(prog="routeB.box8 job")
    ap.add_argument("--job-json", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    from . import cli
    from . import fit as FIT
    from .common import StageError
    spec_ = json.loads(Path(a.job_json).read_text())
    j, ov, steps = spec_["job"], spec_["overrides"], spec_["steps"]
    rd = home() / "runs" / j["scroll"] / j["tag"]
    try:
        FIT.run_fit(j["scroll"], j["tag"], j["z0"], j["z1"], steps, None, None, None, a.gpu, ov or None, workdir=rd)
        ns = SimpleNamespace(target_cm2=3.0 if a.smoke else 26.0, rows=400, windings="60:62" if a.smoke else None)
        cli.stage_tiles(j["scroll"], j["tag"], rd, ns)
    except StageError as e:
        print(f"JOBFAIL {j['id']}: {e}", flush=True)
        return 3
    print(f"JOBOK {j['id']}", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "job":
        sys.exit(job_main(sys.argv[2:]))
    sys.exit(main())
