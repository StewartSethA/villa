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
from . import planner as PL
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
        for s in ("state", "logs"):
            (self.D / s).mkdir(parents=True, exist_ok=True)
        self.out = self.H / "out"                      # ONE directory holding every payload unit: one rsync of it pulls everything as it arrives
        if not getattr(a, "dry_run", False):
            self.out.mkdir(parents=True, exist_ok=True)
        if not getattr(a, "dry_run", False) and not (self.D / "payload").exists():
            try:
                (self.D / "payload").symlink_to(self.out)    # old name kept for older pull scripts
            except OSError:
                pass
        self.host = None
        self.tail_done = False
        self.staged_gb: dict[str, float] = {}           # scroll -> input GB currently held on disk (fetch started .. inputs released)
        self.released: set[str] = set()
        self.heights_path = self.D / "heights.json"
        self.routea_proc = None
        self.disk_fn = (lambda: (10e12, 10e12)) if a.fake_gpus else (lambda: tuple(shutil.disk_usage(self.H)[:2]))
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
        if self.cfg.get("dynamic") and not first_rung:
            vram = self.host.min_vram if self.host else 40.0
            h, why = L.start_height_for(self.cfg, scroll, vram, z1 - z0, max_height=getattr(self.a, "max_height", L.FULL_SPAN), heights_path=self.heights_path)
            jobs = L.jobs_for_height(self.cfg, scroll, sc["shell"], z0, z1, h, PL.OVERLAP)
            start = jobs[0]["rung"]
            why = f"height {h}: {why}"
        else:
            start, why = (L.rung_index(self.cfg, first_rung), "forced by --first-rung") if first_rung else L.first_rung(self.cfg, scroll, z0, z1)
            jobs = L.plan(self.cfg, scroll, sc["shell"], z0, z1, start=start)
        for j in jobs:
            self.scale_expected(j)
            j["dbm_bytes"] = dbm
            self.jobs[j["id"]] = j
            self.order.append(j["id"])
        say(f"{scroll}: {len(jobs)} job(s) on rung {self.cfg['rungs'][start]['name']} ({why}); expected {sum(j['expected_h'] for j in jobs):.1f} GPU-h "
            f"(p90 {sum(j.get('p90_h', j['expected_h']) for j in jobs):.1f})", "box8")
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

    def input_gb(self, scroll: str) -> float:
        try:
            return PL.scroll_facts(scroll, self.cfg, self.host or PL.Host([PL.Gpu("0", 40.0)]), self.a.z0 or UP_LO, self.a.z1 or UP_HI).input_gb
        except Exception:                              # noqa: BLE001 - an unknown size must not stop staging; announced
            say(f"{scroll}: input size unknown, assuming 80 GB for disk admission", "box8")
            return 80.0

    def start_fetches(self):
        """Fetch-ahead stager: scrolls are staged in dispatch order, just ahead of the GPUs, only while (a) fewer than fetch_parallel fetches run, (b) fewer than
        free-GPUs + fetch_ahead fetched scrolls wait unstarted, (c) the disk high-water mark allows the scroll's input+work volume.  Announces every block."""
        self.fpool = ThreadPoolExecutor(max_workers=max(1, self.a.fetch_workers))
        self.stager = threading.Thread(target=self._stage_loop, daemon=True, name="stager")
        self.stager.start()

    def _disk_state(self):
        total, free = self.disk_fn()
        return total / 1e9, free / 1e9, total * self.a.disk_high_water / 1e9

    def _stage_loop(self):
        blocked_msg = {}
        used0 = None
        while not self.done_evt.is_set() and not self.stop_reason:
            time.sleep(min(1.0, self.a.poll_s))
            with self.cv:
                todo = [s for s in self.order_scrolls() if not self.scrolls[s]["fetched"] and not self.scrolls[s]["fetching"] and not self.scrolls[s].get("fetch_failed")]
                if not todo:
                    return
                if used0 is None:
                    total, free, hw = self._disk_state()
                    used0 = (total - free) - sum(self.staged_gb.values())
                active = sum(1 for sc in self.scrolls.values() if sc["fetching"])
                unstarted = sum(1 for k, sc in self.scrolls.items() if (sc["fetched"] or sc["fetching"]) and not any(j["scroll"] == k and j["status"] != "pending" for j in self.jobs.values()))
                free_gpus = max(0, len(self.gpus) - len(self.busy))
                s = todo[0]
                if active >= self.a.fetch_parallel or unstarted >= free_gpus + self.a.fetch_ahead:
                    continue
                need = self.input_gb(s)
                total, free, hw = self._disk_state()
                held = sum(self.staged_gb.values())
                if used0 + held + need > hw or free < need:
                    msg = f"STAGING {s} BLOCKED by disk: base {used0:.0f} + held {held:.0f} + need {need:.0f} > high-water {hw:.0f} GB (free {free:.0f} GB); waits for a payload to be pulled"
                    if blocked_msg.get(s) != int(held):
                        blocked_msg[s] = int(held)
                        say(msg, "stage")
                        self.ev("stage_blocked", scroll=s, held_gb=held, need_gb=need)
                    continue
                self.scrolls[s]["fetching"] = True
                self.staged_gb[s] = need
            self.ev("stage", scroll=s, need_gb=need)
            self.fpool.submit(self.fetch_worker, s)

    def order_scrolls(self) -> list[str]:
        seen = []
        for i in self.order:
            sc = self.jobs[i]["scroll"]
            if sc not in seen:
                seen.append(sc)
        return seen + [s for s in self.scrolls if s not in seen]

    # ---- claim
    def claim(self, gpu: str):
        warned = 0.0
        with self.cv:
            while True:
                if self.stop_reason:
                    return None
                self._maybe_tail_split()
                pend = [self.jobs[i] for i in self.order if self.jobs[i]["status"] == "pending"]
                ready = [j for j in pend if self.scrolls[j["scroll"]]["fetched"]]
                if self.tail_done:
                    ready.sort(key=lambda j: -j["expected_h"])          # tail: largest first (LPT)
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

    def _maybe_tail_split(self):
        """LPT tail balancing (caller holds self.cv).  Once fewer unstarted jobs remain than GPUs, split the largest remaining scrolls into z-stripes across the
        GPUs that would otherwise idle -- accepted only if the simulated makespan shortens (planner.tail_split)."""
        if self.tail_done or not self.cfg.get("dynamic") or getattr(self.a, "no_tail_split", False):
            return
        pend = [j for j in (self.jobs[i] for i in self.order) if j["status"] == "pending" and not j["attempts"]]     # never split a retry (it would lose its checkpoint)
        if not pend or len(pend) >= len(self.gpus):
            return
        avail = [0.0] * max(0, len(self.gpus) - len(self.busy))
        for j in self.jobs.values():
            if j["status"] == "running":
                try:
                    avail.append(self.gov.remaining_h(self.gov.running[j["id"]]))
                except KeyError:
                    avail.append(j["expected_h"])
        shells = {k: sc["shell"] for k, sc in self.scrolls.items()}
        new, notes = PL.tail_split(self.cfg, avail, pend, shells, "p50")
        self.tail_done = True
        if not notes:
            say(f"tail balancing: {len(pend)} unstarted job(s) < {len(self.gpus)} GPUs, no stripe split shortens the simulated tail", "box8")
            return
        old_ids = {j["id"] for j in pend}
        new_ids = {j["id"] for j in new}
        for j in pend:
            if j["id"] not in new_ids:
                j["status"] = "split"
        for k in new:
            if k["id"] not in old_ids:
                k["dbm_bytes"] = self.jobs[k["parent"]].get("dbm_bytes") if k.get("parent") in self.jobs else None
                self.jobs[k["id"]] = k
                self.order.append(k["id"])
        for n_ in notes:
            say("TAIL SPLIT " + n_, "box8")
        self.ev("tail_split", notes=notes, jobs=sorted(new_ids))
        for sc in {j["scroll"] for j in new}:
            self.save(sc)

    def _ram_ok(self):
        if self.a.fake_gpus:
            return True, ""
        avail, total = meminfo_gb()
        need = self.a.ram_need_gb
        if avail < need:
            return False, f"MemAvailable {avail:.0f} GB < need {need:.0f} GB per fit"
        free = self.disk_fn()[1] / 1e9
        if free < self.a.min_free_gb:
            return False, f"free disk {free:.0f} GB on {self.H} < --min-free-gb {self.a.min_free_gb:.0f}"
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
        if cls == "ok":                                   # outside the lock: md5 of a unit takes seconds and must not stall the other GPUs' claims
            try:
                self.publish_unit(j)
            except Exception as e:                  # noqa: BLE001 - reported loudly; the unit is retried at finalize
                say(f"UNIT PAYLOAD FAILED {j['id']}: {type(e).__name__}: {e}", "box8")
                self.ev("unit_failed", job=j["id"], err=str(e))
            try:
                L.record_height(self.heights_path, j["scroll"], max(j["z1"] - j["z0"], (L.load_heights(self.heights_path).get(j["scroll"]) or {}).get("height", 0)),
                                self.host.min_vram if self.host else 0.0)
            except OSError:
                pass
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

    # ---- payload / DONE.  Layout (ONE directory, ONE rsync): <home>/out/<scroll>/<tag>/{files, PAYLOAD.json (md5 of every file), DONE} per finished stripe/job,
    #      written the moment that job finishes; <home>/out/<scroll>/{SCROLL.json, DONE} when the scroll has no pending work; <home>/out/{STATUS.json, ALLDONE.json}.
    def publish_unit(self, j: dict) -> tuple[int, int]:
        scroll, rd = j["scroll"], self.H / "runs" / j["scroll"] / j["tag"]
        unit = self.out / scroll / j["tag"]
        tmp = self.out / f".tmp_{scroll}__{j['tag']}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        for src in self._payload_sources(rd, j):
            dst = tmp / src.relative_to(rd)
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        fl = self.job_dirs(j)[1]
        if fl.exists():
            with gzip.open(tmp / "fit.log.tail.gz", "wb") as g:
                g.write(tail_text(fl, 400000).encode())
        with ThreadPoolExecutor(8) as ex:
            allf = sorted(p for p in tmp.rglob("*") if p.is_file())
            ents = list(ex.map(lambda p: {"path": str(p.relative_to(tmp)), "size": p.stat().st_size, "md5": md5_file(p)}, allf))
        tot = sum(e["size"] for e in ents)
        man = rd / "tiled" / "manifest.json"
        nt = None
        if man.exists():
            try:
                nt = json.loads(man.read_text()).get("n_tiles")
            except ValueError:
                pass
        meta = {"scroll": scroll, "tag": j["tag"], "rung": j["rung_name"], "z": [j["z0"], j["z1"]], "n_tiles": nt, "attempts": len(j["attempts"]), "files": ents,
                "total_bytes": tot, "status": "complete", "tile_name_stem": f"{scroll}_{j['tag']}_wNNN_zRRxC", "manifest": "tiled/manifest.json",
                "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "note": "no checkpoints; tiled/manifest.json is the producer's verbatim manifest (box-side absolute paths inside are informational)"}
        atomic_json(tmp / "PAYLOAD.json", meta)
        (tmp / "DONE").write_text(f"complete {len(ents)} files {tot} bytes\n")
        unit.parent.mkdir(parents=True, exist_ok=True)
        if unit.exists():
            shutil.rmtree(unit, ignore_errors=True)
        os.replace(tmp, unit)                                 # the unit appears whole, DONE inside; rsync excludes .tmp_*
        self.ev("unit", job=j["id"], files=len(ents), bytes=tot)
        say(f"{j['id']}: payload unit out/{scroll}/{j['tag']} ({len(ents)} files, {tot / 1e9:.3f} GB, DONE written)", "box8")
        return len(ents), tot

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
            for j in done:
                if not (self.out / scroll / j["tag"] / "DONE").exists():
                    self.publish_unit(j)                      # a unit whose publish failed earlier is retried here
            sd = self.out / scroll
            units = sorted(j["tag"] for j in done)
            atomic_json(sd / "SCROLL.json", {"scroll": scroll, "status": status, "units": units, "z_covered": sorted([j["z0"], j["z1"]] for j in done),
                                             "failed_intervals": [{"id": j["id"], "z": [j["z0"], j["z1"]], "why": j.get("fail_why")} for j in bad],
                                             "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            (sd / "DONE.tmp").write_text(f"{status} {len(units)} units\n")
            os.replace(sd / "DONE.tmp", sd / "DONE")
        except Exception as e:                          # noqa: BLE001
            say(f"PAYLOAD FAILED {scroll}: {type(e).__name__}: {e}", "box8")
            self.ev("payload_failed", scroll=scroll, err=str(e))
            return
        self.scrolls[scroll]["finalized"] = status
        say(f"{scroll}: {status}: {len(done)} unit(s) in out/{scroll}/ (scroll DONE marker written)", "box8")
        self.ev("finalized", scroll=scroll, status=status, units=len(done))
        self.save(scroll)

    def release_inputs(self, scroll: str):
        """Delete a scroll's fetched inputs (re-fetchable) once its payload is pulled+verified: the disk is the scarce resource."""
        if scroll in self.released:
            return
        self.released.add(scroll)
        before = self._disk_state()[1]
        shutil.rmtree(self.H / "assets" / scroll, ignore_errors=True)
        for c in (self.H / "runs" / scroll).glob("*/fit/cache"):
            shutil.rmtree(c, ignore_errors=True)
        with self.cv:
            self.staged_gb.pop(scroll, None)
            self.cv.notify_all()
        after = self._disk_state()[1]
        say(f"{scroll}: inputs released after pull (free disk {before:.0f} -> {after:.0f} GB)", "box8")
        self.ev("released", scroll=scroll, free_before=before, free_after=after)

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
            for pj in self.out.glob("*/*/PULLED.json"):
                if str(pj) in seen_pulled:
                    continue
                seen_pulled.add(str(pj))
                try:
                    b = json.loads(pj.read_text()).get("bytes", 0)
                except ValueError:
                    continue
                self.gov.transfer(box_upload_gb=b / 1e9, what=f"pulled {pj.parent.parent.name}/{pj.parent.name}")
            for sc, st in list(self.scrolls.items()):                    # inputs are released when every unit is pulled (or, with --free-inputs-on done, when finished)
                if sc in self.released or not st.get("finalized") or st["finalized"] == "nothing":
                    continue
                units = [d for d in (self.out / sc).iterdir() if d.is_dir()] if (self.out / sc).is_dir() else []
                if self.a.free_inputs_on == "done" or (units and all((d / "PULLED.json").exists() for d in units)):
                    self.release_inputs(sc)
            self.write_status()
            if (self.D / "STOP").exists() and not self.stop_reason:
                self._hard_stop("STOP file present (operator request)")
            elif self.gov.hard_stop() and not self.stop_reason:
                self.gov.note_hard_stop()
                self._hard_stop(f"HARD BUDGET STOP: spent ${self.gov.spent():.2f} (+unpulled payload) >= ${self.gov.r.hard_usd}")

    def write_status(self):
        by = {}
        for j in self.jobs.values():
            by[j["status"]] = by.get(j["status"], 0) + 1
        try:
            atomic_json(self.out / "STATUS.json", {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "jobs": by, "busy_gpus": sorted(self.busy), "budget": self.gov.status(),
                                                   "scrolls": {s: sc.get("finalized") or ("fetched" if sc["fetched"] else "staging" if sc["fetching"] else "waiting") for s, sc in self.scrolls.items()},
                                                   "stop": self.stop_reason})
        except OSError:
            pass

    def _hard_stop(self, why: str):
        say(f"*** {why}: stopping running fits (SIGTERM, 60 s, SIGKILL). fit_spiral autosaves every {os.environ.get('FIT_SPIRAL_AUTOSAVE_INTERVAL', '1000')} steps; "
            f"anything newer than the last autosave is lost. Finished payloads stay pullable. ***", "box8")
        self.ev("stop", why=why)
        with self.cv:
            self.stop_reason = why
            self.cv.notify_all()

    # ---- Route A on the spare cores
    def _routea_thread(self):
        a = self.a
        t0 = time.time()
        while not self.done_evt.is_set() and not self.busy and time.time() - t0 < 1800:
            time.sleep(2)                                    # start after the first fit is on a GPU: the fits come first
        if self.done_evt.is_set() or self.stop_reason:
            return
        busy = len(self.gpus)
        h = self.host or PL.Host([PL.Gpu(g, 40.0) for g in self.gpus])
        slots, why = PL.routea_slots(h.phys_cores, busy, a.routea_reserve_cores, h.ram_gb, a.ram_need_gb, h.ram_reserve_gb, a.routea_ram_per_grow_gb, a.routea_slots)
        if slots < 1:
            say(f"Route A NOT started: 0 slots ({why})", "routeA")
            self.ev("routea_skip", why=why)
            return
        from . import routea_side as RA
        names = [s for s in (a.routea_scrolls or "").split(",") if s]
        if not names:
            pick = RA.pick_scrolls(ROOT / "pins" / "scrolls.json", a.routea_disk_gb, list(self.scrolls))
            names = [s for s, _ in pick]
            say(f"Route A scrolls (auto, <= {a.routea_disk_gb:g} GB of prediction+grids, Route B scrolls first, smallest first): " + ", ".join(f"{s} {g:.0f} GB" for s, g in pick), "routeA")
        if not names:
            say("Route A NOT started: no scroll chosen (pins/scrolls.json missing or --routea-disk-gb too small)", "routeA")
            return
        hours = a.routea_hours or max(1.0, (getattr(a, 'planned_makespan_h', None) or 4.0) * 0.9)
        cmd = RA.command(self.H, names, slots, hours, a.routea_seeds)
        say(f"Route A START on spare cores: {slots} slots = {why}; {len(names)} scroll(s), {a.routea_seeds} seeds each, {hours:.1f} h; log box8/logs/routeA.log", "routeA")
        self.ev("routea_start", slots=slots, why=why, scrolls=names, hours=hours)
        self.routea_proc = RA.launch(self.H, cmd, self.D / "logs" / "routeA.log")
        self.routea_stop = threading.Event()
        RA.publisher(self.H / "routeA_work", self.out, self.routea_stop, self.ev, every=max(5.0, a.poll_s * 4))
        while self.routea_proc.poll() is None and not self.done_evt.is_set():
            time.sleep(5)
        if self.routea_proc.poll() not in (None, 0):
            say(f"Route A EXITED rc={self.routea_proc.returncode} (see box8/logs/routeA.log); Route B is unaffected", "routeA")
            self.ev("routea_exit", rc=self.routea_proc.returncode)

    def _routea_finish(self):
        p = self.routea_proc
        if p is not None and p.poll() is None:
            say("Route B finished: stopping Route A (SIGTERM; its finished segments are published, a rerun resumes the rest)", "routeA")
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        if getattr(self, "routea_stop", None):
            self.routea_stop.set()
            time.sleep(1)

    # ---- main
    def run(self) -> int:
        self.done_evt = threading.Event()
        self.start_fetches()
        mon = threading.Thread(target=self.monitor, daemon=True)
        mon.start()
        if getattr(self.a, "routea", False) and (not self.a.fake_gpus or os.environ.get("ROUTEB_TEST_ROUTEA")):
            threading.Thread(target=self._routea_thread, daemon=True, name="routea").start()
        ths = [threading.Thread(target=self.worker, args=(g,), name=f"gpu{g}") for g in self.gpus]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        self.done_evt.set()
        self._routea_finish()
        for s in self.scrolls:
            with self.cv:
                self.maybe_finalize(s)
        self.summary()
        self.write_status()
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
        atomic_json(self.out / "ALLDONE.json", summ)
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
    ap.add_argument("--scrolls", default="auto", help="comma list | 'auto' (default) = EVERY eligible scroll that is runnable (tracks + lasagna + sense + umbilicus in the registry); "
                    "non-runnable ones are listed with the reason")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--z0", type=int, default=None)
    ap.add_argument("--z1", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="z 9000-9500, 1500 steps, 2 windings, 1 tile: a proof run (jobs get --smoke)")
    ap.add_argument("--gpus", default=None, help="comma list of GPU indices (default: all from nvidia-smi)")
    ap.add_argument("--fake-gpus", type=int, default=0, help="TEST ONLY: N logical workers, no nvidia-smi, no RAM guard")
    ap.add_argument("--order", default="spt", choices=["given", "spt", "lpt"], help="queue order: shortest expected first (default; most scrolls finished per $) | as listed | longest first")
    ap.add_argument("--legacy-ladder", action="store_true", help="old static ladder (full -> sw2800), no planner/admission/tail split; also implied by --ladder")
    ap.add_argument("--first-rung", default=None, help="force the starting rung name (default: full, or the first under the track limit when n_tracks is known)")
    ap.add_argument("--ladder-config", default=None)
    ap.add_argument("--ladder", default=None, help="rungs in order, e.g. full,sw2800 (default from ladder_config.json) | 'all' = full,w13000,q4500,sw2800")
    ap.add_argument("--fetch-workers", type=int, default=8, help="threads in the scroll-fetch pool (each scroll fetch itself uses ROUTEB_FETCH_WORKERS=128 connections)")
    ap.add_argument("--fetch-parallel", type=int, default=8, help="max concurrent scroll fetches")
    ap.add_argument("--fetch-ahead", type=int, default=2, help="fetched-but-unstarted scrolls kept ready beyond the free GPUs (GPUs must not wait)")
    ap.add_argument("--fetch-files-per-s", type=float, default=58.0, help="planner: objects/s of ONE scroll's lasagna fetch (MEASURED 58 on pny with 128 workers under load 95, n = 1)")
    ap.add_argument("--disk-high-water", type=float, default=0.85, help="never stage inputs that would push disk use above this fraction of the volume")
    ap.add_argument("--disk-total-gb", type=float, default=None, help="planner/test override of the volume size (default: df of ROUTEB_HOME)")
    ap.add_argument("--disk-free-gb", type=float, default=None)
    ap.add_argument("--free-inputs-on", default="pulled", choices=["pulled", "done"], help="delete a scroll's fetched inputs once its payload is pulled+verified (default) or as soon as it is finished")
    ap.add_argument("--max-height", type=int, default=L.FULL_SPAN, help="largest stripe height (z slices) the planner may start a scroll at")
    ap.add_argument("--vram-margin-gib", type=float, default=1.5)
    ap.add_argument("--gpu-speed", type=float, default=1.0, help="speed of these GPUs relative to the V100-class basis of the GPU-hour model (A100 UNMEASURED -> 1.0)")
    ap.add_argument("--plan-frac", type=float, default=0.8, help="the p90 plan must finish within this fraction of the soft budget and of the max run hours; the rest is deferred")
    ap.add_argument("--priority", default=None, help="comma list of scrolls, most important first (deferral drops from the end); default: both prizes first, then cheapest")
    ap.add_argument("--no-plan", action="store_true", help="skip the planner's admission/deferral (run every runnable scroll; the governor still guards the budget)")
    ap.add_argument("--no-tail-split", action="store_true", help="never split the tail scrolls into z-stripes")
    ap.add_argument("--phys-cores", type=int, default=None)
    ap.add_argument("--fake-vram-gib", type=float, default=40.0, help="TEST ONLY with --fake-gpus")
    ap.add_argument("--fake-ram-gb", type=float, default=516.0, help="TEST ONLY with --fake-gpus")
    ap.add_argument("--routea", dest="routea", action="store_true", default=True, help="Route A guarded grows on the spare CPU cores (default ON)")
    ap.add_argument("--no-routea", dest="routea", action="store_false", help="disable Route A on the spare cores")
    ap.add_argument("--routea-slots", type=int, default=None, help="override: concurrent Route A grows (default physical_cores - 2 x busy GPUs - reserve, RAM-checked)")
    ap.add_argument("--routea-reserve-cores", type=int, default=4)
    ap.add_argument("--routea-ram-per-grow-gb", type=float, default=6.0, help="ASSUMED RSS of one Route A grow (seed proposer ~1.5 GB measured; tracer unmeasured)")
    ap.add_argument("--routea-scrolls", default=None, help="comma list (default auto: Route B scrolls first, smallest first, within --routea-disk-gb)")
    ap.add_argument("--routea-disk-gb", type=float, default=150.0, help="disk reserved for Route A inputs (prediction+grids, 4-93 GB per scroll)")
    ap.add_argument("--routea-seeds", type=int, default=8)
    ap.add_argument("--routea-hours", type=float, default=None, help="Route A grow budget (default 0.9 x the planned Route B makespan)")
    ap.add_argument("--stall-minutes", type=float, default=None)
    ap.add_argument("--poll-s", type=float, default=5.0)
    ap.add_argument("--ram-need-gb", type=float, default=float(os.environ.get("ROUTEB_FIT_RAM_GB", 40)), help="MemAvailable needed to admit one fit (ASSUMED 40; peak RSS is logged per attempt so this can be set from data)")
    ap.add_argument("--ram-floor-gb", type=float, default=float(os.environ.get("ROUTEB_RAM_FLOOR_GB", 12)), help="below this MemAvailable the largest fit is killed (host_oom) instead of letting the kernel pick")
    ap.add_argument("--min-free-gb", type=float, default=60.0, help="do not admit a fit when ROUTEB_HOME has less free disk than this (fit dirs are written there)")
    ap.add_argument("--dry-run", action="store_true", help="print queue, skips, expected hours and the budget projection; launch nothing")
    ap.add_argument("--target-cm2", type=float, default=26.0)
    ap.add_argument("--rows", type=int, default=400)
    ap.add_argument("--windings", default=None)
    g = ap.add_argument_group("budget (deploy_common/budget.py; env BUDGET_* also read)")
    g.add_argument("--soft", type=float, default=None, help="stop launching new fits when the projection reaches this (USD, default 45)")
    g.add_argument("--hard", type=float, default=None, help="hard stop (USD, default 49)")
    g.add_argument("--hour-usd", type=float, default=None, help="machine USD/h (default 4.276, the 8x A100 40GB quote)")
    g.add_argument("--disk-usd-per-16gb-hour", type=float, default=None, help="allocated disk, USD per 16 GB per hour (default 0.009), billed for the whole run")
    g.add_argument("--disk-gb", type=float, default=None, help="allocated (billed) disk in GB (default 934)")
    g.add_argument("--ingress-per-tb", type=float, default=None, help="box DOWNLOADS from the web, USD/TB (default 2.70)")
    g.add_argument("--egress-per-tb", type=float, default=None, help="box UPLOADS to us (our payload pull), USD/TB (default 4.00)")
    g.add_argument("--swap-directions", action="store_true")
    g.add_argument("--max-run-hours", type=float, default=None, help="wall-clock hours since rental start; no fit is launched whose projected end passes it, running fits wind down (default 12; 0 = unlimited)")
    g.add_argument("--budget-config", default=None, help="json file with any of hour_usd, soft_usd, hard_usd, max_run_hours, ingress_per_tb, egress_per_tb, swap_directions (precedence: defaults < file < BUDGET_* env < flags)")
    g.add_argument("--box-start", type=float, default=None, help="epoch seconds when the rental started (default: first ledger write)")
    g.add_argument("--clock-scale", type=float, default=None, help="REHEARSAL: simulated seconds per real second (3600 = 1 s is 1 billed hour)")
    return ap


def gpu_info(a) -> list[PL.Gpu]:
    """GPU list with VRAM (GiB): fake (tests), --gpus subset, or nvidia-smi."""
    if a.fake_gpus:
        return [PL.Gpu(str(i), a.fake_vram_gib, a.gpu_speed) for i in range(a.fake_gpus)]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SystemExit(f"ROUTEB FAIL box8-gpus: cannot run nvidia-smi ({e})")
    if out.returncode != 0 or not out.stdout.strip():
        raise SystemExit(f"ROUTEB FAIL box8-gpus: nvidia-smi rc={out.returncode}: {out.stderr.strip()[:200]}")
    rows = [[x.strip() for x in r.split(",")] for r in out.stdout.strip().splitlines()]
    want = [g for g in a.gpus.split(",") if g] if a.gpus else None
    gs = [PL.Gpu(r[0], float(r[2]) / 1024.0, a.gpu_speed) for r in rows if want is None or r[0] in want]
    say("GPUs: " + "; ".join(f"{r[0]}={r[1]} {float(r[2]) / 1024:.1f} GiB" for r in rows if want is None or r[0] in want), "box8")
    return gs


def physical_cores(a) -> int:
    if a.phys_cores:
        return a.phys_cores
    try:
        out = subprocess.run(["lscpu", "-p=core,socket"], capture_output=True, text=True, timeout=10).stdout
        n = len({l for l in out.splitlines() if l and not l.startswith("#")})
        if n:
            return n
    except (OSError, subprocess.TimeoutExpired):
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def build_host(a, H: Path) -> PL.Host:
    gs = gpu_info(a)
    ram = a.fake_ram_gb if a.fake_gpus else meminfo_gb()[1]
    du = shutil.disk_usage(H if H.exists() else H.parent)
    total = a.disk_total_gb if a.disk_total_gb is not None else du.total / 1e9
    free = a.disk_free_gb if a.disk_free_gb is not None else du.free / 1e9
    return PL.Host(gs, phys_cores=physical_cores(a), ram_gb=ram, disk_total_gb=total, disk_free_gb=free, disk_high_water_frac=a.disk_high_water,
                   fetch_files_per_s=a.fetch_files_per_s, fetch_parallel=a.fetch_parallel, fetch_ahead=a.fetch_ahead, ram_per_fit_gb=a.ram_need_gb,
                   reserve_cores=a.routea_reserve_cores)


def all_scroll_names() -> list[str]:
    return sorted(p.stem for p in (ROOT / "routeB" / "scrolls").glob("PHerc*.json"))


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.smoke:
        a.no_plan = True                                 # a smoke is not priced by the full-fit planner
    legacy = bool(a.legacy_ladder or a.ladder)
    cfg = L.load_config(a.ladder_config, a.ladder)
    cfg["gpu_speed"] = a.gpu_speed
    if not legacy:
        cfg["dynamic"] = True
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
    kw.update({k: v for k, v in dict(hour_usd=a.hour_usd, ingress_per_tb=a.ingress_per_tb, egress_per_tb=a.egress_per_tb, soft_usd=a.soft, hard_usd=a.hard,
                                     max_run_hours=a.max_run_hours, disk_usd_per_16gb_hour=a.disk_usd_per_16gb_hour, disk_gb=a.disk_gb).items() if v is not None})
    if a.swap_directions:
        kw["swap_directions"] = True
    rates = B.Rates.from_env(**kw)
    say(rates.describe(), "budget")
    H = home()
    clock = B.ScaledClock(a.clock_scale) if a.clock_scale else time.time
    import tempfile
    gov = B.Governor(Path(tempfile.mkdtemp(prefix="box8_dry_")) if a.dry_run else H / "box8" / "budget", rates, clock=clock, box_start=a.box_start, say=lambda m: say(m, "budget"))
    names = all_scroll_names() if a.scrolls in ("auto", "all") else [s.strip() for s in a.scrolls.split(",") if s.strip()]
    sch = Scheduler(a, cfg, gov)
    sch.dry = a.dry_run                                  # a dry run reads state (if any) but writes nothing
    run_names, not_runnable = [], []
    for s in names:
        why = skip_reason(s)
        if why:
            say(f"SKIP {s}: {why}", "box8")
            sch.ev("skip", scroll=s, why=why)
            not_runnable.append((s, why))
        else:
            run_names.append(s)
    if not run_names:
        say("nothing to run: every requested scroll was skipped (see SKIP lines above)", "box8")
        return 1
    host = build_host(a, H)
    sch.host = host
    sch.gpus = [g.idx for g in host.gpus]
    ng = max(1, len(sch.gpus))
    plan = None
    if not legacy and not a.no_plan:
        plan = PL.make_plan(host, rates, run_names, cfg, z0, z1, a.plan_frac, a.max_height, [x for x in (a.priority or "").split(",") if x] or None, sch.heights_path, a.order)
        slots, sw = PL.routea_slots(host.phys_cores, ng, a.routea_reserve_cores, host.ram_gb, a.ram_need_gb, host.ram_reserve_gb, a.routea_ram_per_grow_gb, a.routea_slots)
        ra = (f"PLAN Route A on the spare cores: {'ON' if a.routea else 'OFF (--no-routea)'}; {slots} grow slot(s) = {sw}" if a.routea else "PLAN Route A: OFF (--no-routea)")
        print(PL.render(host, rates, plan, not_runnable, ra), flush=True)
        if plan["keep"]:
            a.planned_makespan_h = plan["p50"]["makespan_h"]
        sch.ev("plan", keep=plan["keep"], deferred=plan["deferred"])
        if not plan["keep"]:
            return 3
        for n_, why in plan["deferred"]:
            sch.ev("deferred", scroll=n_, why=why)
        run_names = list(plan["keep"])
        if a.dry_run:
            return 0 if plan["fits"] else 3
    for s in run_names:
        sch.add_scroll(s, z0, z1, a.first_rung)
    if a.order != "given":
        tot = {}
        for i in sch.order:                          # snapshot first: list.sort() empties the list while it runs
            tot[sch.jobs[i]["scroll"]] = tot.get(sch.jobs[i]["scroll"], 0.0) + sch.jobs[i]["expected_h"]
        sch.order.sort(key=lambda i: tot[sch.jobs[i]["scroll"]], reverse=(a.order == "lpt"))
    tot_h = sum(j["expected_h"] for j in sch.jobs.values() if j["status"] == "pending")
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
