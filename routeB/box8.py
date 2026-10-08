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
import inspect
import json
import os
import re
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
def real_disk_fn(H):
    """(total, FREE) bytes of the volume holding H.  NOT shutil.disk_usage()[:2]: that is (total, USED), which made a 24 GB-used 1 TB box read as 24 GB free and block all staging."""
    def fn():
        u = shutil.disk_usage(H)
        return u.total, u.free
    return fn


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
        self.routea_slots = None
        self.routea_expected_exit = False
        self.routea_launch_fn = None
        self.control = self.D / "control"
        self.allowed_init = None                         # set by main(): the --gpus subset; None = every worker GPU
        self.allowed = None                              # live allowed set (control dir); None = allowed_init / all
        self.paused = False
        self.stop_launch = False                         # control STOP: no new launches, running fits finish, then the run ends
        self.kill_req: set[str] = set()
        self.foreign_skip: dict[str, float] = {}
        self.gpu_mem_fn = None                           # tests: gpu -> memory.used MiB
        self._mem_cache = (0.0, {})
        self.priority: list[str] = []
        self.alarms: dict[str, dict] = {}
        self.idle_since: dict[str, float] = {}
        self.rx_hist: list[tuple[float, int]] = []
        self.rx_fn = None                                # tests: () -> cumulative rx bytes
        self.link_base: float | None = None
        self.link_state = "ok"
        self.link_probe_fn = None                        # tests: () -> measurement dict
        self.last_shrink = 0.0
        self.replan_log: list[str] = []
        self._warned_unknown_gpus: set[str] = set()
        self.disk_fn = (lambda: (10e12, 10e12)) if a.fake_gpus else real_disk_fn(self.H)
        self.cv = threading.Condition()
        self.save_lock = threading.RLock()
        self.jobs: dict[str, dict] = {}
        self.order: list[str] = []
        self.scrolls: dict[str, dict] = {}            # scroll -> {fetched: bool, fetching: bool, fetch_failed: str|None, shell, finalized}
        self.procs: dict[str, subprocess.Popen] = {}
        self.stop_reason: str | None = None
        self.fetch_fn = fetch_fn or self._fetch_real
        try:
            self.stripe_mode = (not a.no_stripe_staging) and len(inspect.signature(self.fetch_fn).parameters) >= 3
        except (TypeError, ValueError):
            self.stripe_mode = False
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
            "rung_succeeded": {j["tag"]: j["rung_name"] for j in ok}, "planned_with": self.planned_with(), "jobs": js})

    def planned_with(self) -> dict:
        """The planning inputs recorded in each state file: a resume compares them with the current run and says what it did with the pending jobs."""
        gp = sorted(self.allowed_init if self.allowed_init is not None else self.gpus, key=str)
        return {"gpus": gp, "max_height": getattr(self.a, "max_height", None), "vram": round(self.host.min_vram, 1) if self.host else None}

    def load_state(self, scroll: str) -> bool:
        p = self.D / "state" / f"{scroll}.json"
        if not p.exists():
            return False
        st = json.loads(p.read_text())
        self._resumed_with = getattr(self, "_resumed_with", {})
        self._resumed_with[scroll] = st.get("planned_with")
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
            j["p90_h"] = 1.4
        elif self.a.steps != 30000:
            j["expected_h"] = j["expected_h"] * self.a.steps / 30000.0 + 0.25
            if "p90_h" in j:
                j["p90_h"] = j["p90_h"] * self.a.steps / 30000.0 + 0.25

    def add_scroll(self, scroll: str, z0: int, z1: int, first_rung: str | None):
        sp = spec(scroll)
        sc = {"fetched": False, "fetching": False, "shell": int(sp["shell_outer_winding_idx"]), "finalized": None}
        self.scrolls[scroll] = sc
        if self.load_state(scroll):
            self._resume_report(scroll)
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

    def _resume_report(self, scroll: str):
        """A resumed scroll keeps its old jobs: say so, and say it LOUDLY when --gpus / --max-height / VRAM differ from the run that planned them."""
        pend = [j for j in self.jobs.values() if j["scroll"] == scroll and j["status"] == "pending" and not j["attempts"]]
        old, now = self._resumed_with.get(scroll), self.planned_with()
        if old is None:
            say(f"RESUME {scroll}: the state file has no planning record (older build): {len(pend)} pending job(s) KEPT as planned then", "resume")
            return
        diff = {k: (old.get(k), now.get(k)) for k in now if old.get(k) != now.get(k)}
        if not diff:
            say(f"RESUME {scroll}: --gpus / --max-height / VRAM unchanged: {len(pend)} pending job(s) kept", "resume")
            return
        what = "; ".join(f"{k}: {a_} -> {b_}" for k, (a_, b_) in diff.items())
        if not (self.a.replan_pending_on_resume and self.cfg.get("dynamic")):
            say(f"RESUME {scroll}: planning inputs CHANGED ({what}); the {len(pend)} pending job(s) {[j['id'] for j in pend][:6]} were KEPT as planned "
                f"(NOT re-planned; pass --replan-pending-on-resume to re-plan them)", "resume")
            self.ev("resume_kept", scroll=scroll, changed=what, pending=[j["id"] for j in pend])
            return
        vram = self.host.min_vram if self.host else 40.0
        h, why = L.start_height_for(self.cfg, scroll, vram, self.cfg["full_span_z"], max_height=getattr(self.a, "max_height", L.FULL_SPAN))
        n_new = 0
        for j in pend:
            new = L.jobs_for_height(self.cfg, scroll, self.scrolls[scroll]["shell"], j["z0"], j["z1"], min(h, j["z1"] - j["z0"]), PL.OVERLAP)
            if len(new) == 1 and (new[0]["z0"], new[0]["z1"], new[0]["tag"]) == (j["z0"], j["z1"], j["tag"]):
                continue
            if any(k["id"] in self.jobs for k in new):
                continue
            j["status"], j["cancelled_by"] = "cancelled", "resume_replan"
            for k in new:
                self.scale_expected(k)
                k["provenance"] = "resume_replan"
                k["parent"] = j["id"]
                k["dbm_bytes"] = j.get("dbm_bytes")
                self.jobs[k["id"]] = k
                self.order.append(k["id"])
                n_new += 1
        say(f"RESUME {scroll}: planning inputs CHANGED ({what}); RE-PLANNED the pending jobs at height {h} ({why}): {len(pend)} cancelled/kept -> {n_new} new job(s)", "resume")
        self.ev("resume_replanned", scroll=scroll, changed=what, new_jobs=n_new)

    # ---- fetch
    def fetch_est_h(self, scroll: str) -> float:
        """Planned hours to stage `scroll` (planner.fetch_time at the current link/objects rate)."""
        try:
            if self.host is None:
                return 0.0
            return PL.fetch_time(self.host, PL.scroll_facts(scroll, self.cfg, self.host, self.a.z0 or UP_LO, self.a.z1 or UP_HI))
        except Exception:                                # noqa: BLE001
            return 0.0

    def _fetch_real(self, scroll: str, z0=None, z1=None) -> float:
        """Stage `scroll` for [z0, z1) (default: the whole requested range): tracks (skipped when already verified) + the lasagna chunks of that z-range only."""
        from . import cli
        ns = SimpleNamespace(with_crossings=False, full_lasagna=False)
        before = _total_net(self.H)
        lo = z0 if z0 is not None else (UP_LO if self.a.z0 is None else self.a.z0)
        hi = z1 if z1 is not None else (UP_HI if self.a.z1 is None else self.a.z1)
        cli.stage_fetch(scroll, lo, hi, ns)
        return max(0.0, _total_net(self.H) - before) / 1e9

    def job_ready(self, j: dict, _d: int = 0) -> bool:
        """A job can start when its inputs are on disk: the whole scroll is staged, the job itself was staged, or its parent (whose z-range contains it) was."""
        if j.get("ready") or self.scrolls[j["scroll"]]["fetched"]:
            return True
        par = self.jobs.get(j.get("parent") or "")
        return bool(par) and _d < 6 and self.job_ready(par, _d + 1)

    def fetch_worker_stripes(self, scroll: str):
        """Per-stripe staging: stripes in z order; stripe 1 (tracks + its lasagna chunks) lands first and its fit can start while the next stripes' chunks keep downloading."""
        sc = self.scrolls[scroll]
        t_s0 = time.time()
        est_h = self.fetch_est_h(scroll)
        total_gb = 0.0
        while True:
            with self.cv:
                nxt = sorted((self.jobs[i] for i in self.order if self.jobs[i]["scroll"] == scroll and self.jobs[i]["status"] == "pending" and not self.job_ready(self.jobs[i])),
                             key=lambda x: (x["z0"], x["z1"]))
                nxt = nxt[0] if nxt else None
            if nxt is None:
                break
            err = None
            for attempt in (1, 2, 3):
                try:
                    gb = self.fetch_fn(scroll, nxt["z0"], nxt["z1"])
                    self.gov.transfer(box_download_gb=gb, what=f"fetch {scroll} z[{nxt['z0']},{nxt['z1']})")
                    total_gb += gb
                    with self.cv:
                        nxt["ready"] = True
                        self.save(scroll)
                        self.cv.notify_all()
                    self.ev("job_ready", job=nxt["id"], gb=gb)
                    say(f"{nxt['id']}: inputs staged ({gb:.2f} GB new, {time.time() - t_s0:.0f} s since the scroll's fetch began): the fit can start", "stage")
                    err = None
                    break
                except BaseException as e:               # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"
                    say(f"FETCH FAILED {nxt['id']} (attempt {attempt}/3): {err[:300]}", "box8")
                    time.sleep(5 if attempt < 3 else 0)
            if err:
                with self.cv:
                    sc["fetching"], sc["fetch_failed"] = False, err
                    for j in self.jobs.values():
                        if j["scroll"] == scroll and j["status"] == "pending":
                            j["status"], j["fail_why"] = "failed", f"fetch failed: {err[:200]}"
                    self.cv.notify_all()
                self.ev("fetch_failed", scroll=scroll, err=err)
                with self.cv:
                    self.save(scroll)
                return
        with self.cv:
            sc["fetching"] = False
            sc["fetched"] = all(self.job_ready(j) for j in self.jobs.values() if j["scroll"] == scroll and j["status"] == "pending")
            self.cv.notify_all()
            self.save(scroll)
        self._observe_fetch(scroll, total_gb, time.time() - t_s0, est_h)
        self.ev("fetched", scroll=scroll, gb=total_gb)

    def fetch_worker(self, scroll: str):
        sc = self.scrolls[scroll]
        err = None
        for attempt in (1, 2, 3):
            try:
                t_f0 = time.time()
                est_h = self.fetch_est_h(scroll)
                gb = self.fetch_fn(scroll)
                self.gov.transfer(box_download_gb=gb, what=f"fetch {scroll}")
                self._observe_fetch(scroll, gb, time.time() - t_f0, est_h)
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

    def _observe_fetch(self, scroll: str, gb: float, secs: float, est_h: float):
        """Compare the planned staging time with what happened; a >1.5x miss either way rescales the planner's fetch rates (the link/object rates were quoted or probed
        under other load) and triggers a re-plan -- GPUs waiting for data bill like busy ones."""
        if self.host is None or secs < 60 or est_h <= 0:
            return
        ratio = secs / (est_h * 3600.0)
        say(f"FETCH OBSERVED {scroll}: {gb:.1f} GB in {secs / 60:.1f} min = {gb * 1000 / secs:.0f} MB/s; planned {est_h * 60:.1f} min (x{ratio:.2f})", "fetch")
        self.ev("fetch_observed", scroll=scroll, gb=gb, secs=secs, planned_h=est_h)
        if ratio > 1.5 or ratio < 0.5:
            self.host.fetch_files_per_s /= ratio
            self.host.net_down_mb_s /= ratio
            say(f"FETCH RATE CHANGE: staging is x{ratio:.2f} of plan -> planner rates scaled to {self.host.fetch_files_per_s:.0f} files/s and {self.host.net_down_mb_s:.0f} MB/s; "
                f"{'DOWNLOAD IS SLOWER THAN PLANNED: idle GPUs waiting for data are billed; consider fewer GPUs (routeB_ctl.sh gpus ...)' if ratio > 1.5 else 'faster than planned'}", "fetch")
            threading.Thread(target=self.replan, args=(f"fetch observed x{ratio:.2f} of plan",), daemon=True).start()

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
                todo = [s for s in self.order_scrolls() if not self.scrolls[s]["fetched"] and not self.scrolls[s]["fetching"] and not self.scrolls[s].get("fetch_failed")
                        and any(self.jobs[i]["scroll"] == s and self.jobs[i]["status"] == "pending" and not self.job_ready(self.jobs[i]) for i in self.order)]
                if not todo:
                    if all(sc["fetched"] or sc.get("fetch_failed") for sc in self.scrolls.values()):
                        return
                    continue                                   # deferred scrolls may be restored by a re-plan: keep watching
                if used0 is None:
                    total, free, hw = self._disk_state()
                    used0 = (total - free) - sum(self.staged_gb.values())
                active = sum(1 for sc in self.scrolls.values() if sc["fetching"])
                unstarted = sum(1 for k, sc in self.scrolls.items() if (sc["fetched"] or sc["fetching"]) and not any(j["scroll"] == k and j["status"] != "pending" for j in self.jobs.values()))
                free_gpus = max(0, len(self.allowed_set()) - len([g for g in self.busy if g in self.allowed_set()]))
                s = todo[0]
                if active >= self.a.fetch_parallel or unstarted >= free_gpus + self.a.fetch_ahead:
                    continue
                need = self.input_gb(s)
                total, free, hw = self._disk_state()
                held = sum(self.staged_gb.values())
                if used0 + held + need > hw or free < need:
                    msg = f"STAGING {s} BLOCKED by disk: base {used0:.0f} + held {held:.0f} + need {need:.0f} > high-water {hw:.0f} GB (free {free:.0f} GB); waits for a payload to be pulled"
                    self._raise("stage_blocked", msg)
                    if blocked_msg.get(s) != int(held):
                        blocked_msg[s] = int(held)
                        say(msg, "stage")
                        self.ev("stage_blocked", scroll=s, held_gb=held, need_gb=need)
                    continue
                self._clear("stage_blocked")
                self.scrolls[s]["fetching"] = True
                self.staged_gb[s] = need
            self.ev("stage", scroll=s, need_gb=need)
            self.fpool.submit(self.fetch_worker_stripes if self.stripe_mode else self.fetch_worker, s)

    def order_scrolls(self) -> list[str]:
        seen = []
        for i in self.order:
            sc = self.jobs[i]["scroll"]
            if sc not in seen:
                seen.append(sc)
        return seen + [s for s in self.scrolls if s not in seen]

    # ---- claim
    def allowed_set(self) -> set:
        base = self.allowed if self.allowed is not None else (self.allowed_init if self.allowed_init is not None else set(self.gpus))
        return set(base) & set(self.gpus)

    def _gpu_clear(self, gpu: str) -> bool:
        """False when a FOREIGN process holds VRAM on a GPU we are about to launch on (appeared after startup).  --force-gpus disables the check."""
        if self.a.force_gpus:
            return True
        try:
            if self.gpu_mem_fn is not None:
                used = float(self.gpu_mem_fn(gpu))
            elif self.a.fake_gpus:
                return True
            else:
                t, cache = self._mem_cache
                if time.time() - t > 15:
                    cache = {r[0]: float(r[1]) for r in nvsmi_query("index,memory.used")}
                    self._mem_cache = (time.time(), cache)
                used = cache.get(gpu, 0.0)
        except (SystemExit, ValueError):
            return True                                   # cannot measure: do not block on a monitoring failure
        return used <= self.a.foreign_mib

    def claim(self, gpu: str):
        warned = 0.0
        idle_warned = 0.0
        with self.cv:
            while True:
                if self.stop_reason:
                    return None
                pend = [self.jobs[i] for i in self.order if self.jobs[i]["status"] == "pending"]
                running = [j for j in self.jobs.values() if j["status"] == "running"]
                waiting_fetch = [j for j in pend if self.scrolls[j["scroll"]]["fetching"]]
                if not pend and not running and not waiting_fetch:
                    return None
                if self.stop_launch:
                    if not running:
                        return None                       # graceful STOP: nothing running any more -> workers end, ALLDONE is written
                    self.cv.wait(timeout=2)
                    continue
                if self.paused or gpu not in self.allowed_set() or time.time() < self.foreign_skip.get(gpu, 0.0):
                    if not self.allowed_set() and not running and time.time() - idle_warned > 600:
                        say("IDLE: no allowed GPU and nothing running (control dir); the box is still billing", "box8")
                        idle_warned = time.time()
                    self.cv.wait(timeout=2)
                    continue
                self._maybe_fill_idle()
                self._maybe_tail_split()
                pend = [self.jobs[i] for i in self.order if self.jobs[i]["status"] == "pending"]
                ready = [j for j in pend if self.job_ready(j)]
                if self.tail_done:
                    ready.sort(key=lambda j: -j["expected_h"])          # tail: largest first (LPT)
                for j in ready:
                    ram_ok, ram_why = self._ram_ok()
                    if not ram_ok:
                        if time.time() - warned > 60:
                            say(f"GPU {gpu}: waiting for RAM: {ram_why}", "box8")
                            warned = time.time()
                        break
                    if not self._gpu_clear(gpu):
                        self.foreign_skip[gpu] = time.time() + 60
                        say(f"WARNING GPU {gpu}: a foreign process now holds VRAM (> --foreign-mib {self.a.foreign_mib:g} MiB); not launching on it for 60 s (--force-gpus overrides)", "box8")
                        self.ev("foreign_gpu", gpu=gpu)
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
                self.cv.wait(timeout=5)

    def _maybe_fill_idle(self):
        """--fill-idle-gpus (caller holds self.cv): allowed GPUs idle, jobs ready fewer than idle GPUs, and another scroll still downloading -> re-split the NOT-YET-STARTED ready
        jobs into more z-stripes (each >= --fill-min-height, overlap as in the tail split) so every idle GPU gets one.  Old pending job -> `cancelled`, new jobs carry
        provenance 'fill_idle'.  Running and finished jobs, and retries, are never touched."""
        if not self.a.fill_idle_gpus or not self.cfg.get("dynamic") or self.paused or self.stop_launch:
            return
        al = self.allowed_set()
        idle = [g for g in al if g not in self.busy]
        pend = [self.jobs[i] for i in self.order if self.jobs[i]["status"] == "pending"]
        ready = [j for j in pend if self.job_ready(j)]
        waiting = [j for j in pend if not self.job_ready(j)]
        if not idle or not waiting or len(ready) >= len(idle):
            return
        need = len(idle) - len(ready)
        hmin = max(self.a.fill_min_height, PL.GRID)
        done_any = []
        for j in sorted([x for x in ready if not x["attempts"]], key=lambda x: -(x["z1"] - x["z0"])):
            if need <= 0:
                break
            shell = self.scrolls[j["scroll"]]["shell"]
            best = None
            for k in range(need + 1, 1, -1):
                ps = PL.pieces_for(self.cfg, j, k, shell)
                if len(ps) >= 2 and all(p_["z1"] - p_["z0"] >= hmin for p_ in ps):
                    best = ps
                    break
            if not best:
                continue
            j["status"], j["cancelled_by"] = "cancelled", "fill_idle"
            for p_ in best:
                p_["provenance"], p_["parent"], p_["dbm_bytes"] = "fill_idle", j["id"], j.get("dbm_bytes")
                self.scale_expected(p_)
                self.jobs[p_["id"]] = p_
                self.order.append(p_["id"])
            need -= len(best) - 1
            done_any.append((j, best))
            self.save(j["scroll"])
        for j, best in done_any:
            msg = (f"FILL-IDLE: {len(idle)} allowed GPU(s) idle while {sorted({w['scroll'] for w in waiting})} download: {j['id']} (not started, {j['z1'] - j['z0']} slices) -> "
                   f"{len(best)} stripes of ~{best[0]['z1'] - best[0]['z0']} slices ({sum(b_['expected_h'] for b_ in best):.1f} vs {j['expected_h']:.1f} GPU-h)")
            say(msg, "fill")
            self.ev("fill_idle", job=j["id"], pieces=[b_["id"] for b_ in best])
        key = (len(idle), len(ready), need)
        if need > 0 and getattr(self, "_fill_warned", None) != key:
            self._fill_warned = key
            say(f"FILL-IDLE limited: {need} idle GPU(s) stay idle (no pending job can be split into stripes >= --fill-min-height {hmin}); they start when the next scroll arrives", "fill")
            self.ev("fill_idle_limited", idle=len(idle), still_idle=need)

    def _maybe_tail_split(self):
        """LPT tail balancing (caller holds self.cv).  Once fewer unstarted jobs remain than GPUs, split the largest remaining scrolls into z-stripes across the
        GPUs that would otherwise idle -- accepted only if the simulated makespan shortens (planner.tail_split)."""
        if self.tail_done or not self.cfg.get("dynamic") or getattr(self.a, "no_tail_split", False):
            return
        pend = [j for j in (self.jobs[i] for i in self.order) if j["status"] == "pending" and not j["attempts"]]     # never split a retry (it would lose its checkpoint)
        al = self.allowed_set()
        if not pend or len(pend) >= len(al):
            return
        avail = [0.0] * max(0, len(al) - len([g for g in self.busy if g in al]))
        for j in self.jobs.values():
            if j["status"] == "running" and j.get("gpu") in al:
                try:
                    avail.append(self.gov.remaining_h(self.gov.running[j["id"]]))
                except KeyError:
                    avail.append(j["expected_h"])
        shells = {k: sc["shell"] for k, sc in self.scrolls.items()}
        new, notes = PL.tail_split(self.cfg, avail, pend, shells, "p50")
        self.tail_done = True
        if not notes:
            say(f"tail balancing: {len(pend)} unstarted job(s) < {len(al)} allowed GPUs, no stripe split shortens the simulated tail", "box8")
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
        ctl_killed = False
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
            if gpu in self.kill_req and not ctl_killed and p.poll() is None:
                ctl_killed = True
                say(f"{j['id']}: CONTROL KILL of GPU {gpu}: SIGTERM now; the interval is re-queued and resumes from its last autosave "
                    f"(FIT_SPIRAL_AUTOSAVE_INTERVAL {os.environ.get('FIT_SPIRAL_AUTOSAVE_INTERVAL', '1000')} steps)", "box8")
                self._kill(p)
        self.kill_req.discard(gpu)
        rc = p.returncode
        self.procs.pop(j["id"], None)
        att.update(rc=rc, wall_s=round(time.time() - t_start, 1), peak_rss_gb=round(peak_rss, 1), t1=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        txt = tail_text(fitlog, 20000) + "\n" + tail_text(jlog, 20000)
        if self.stop_reason:
            att["class"] = "hard_stop"
        elif ctl_killed:
            att["class"] = "ctl_kill"
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
            elif cls == "ctl_kill":
                j["status"] = "pending"                      # never lost: back in the queue, resumes from its last checkpoint, no attempt charged to the ladder
                j["fresh"] = False
                j.setdefault("notes", []).append("re-queued after a control-dir KILL")
                self.gov.fit_end(j["id"], False)
                self.ev("ctl_kill_requeued", job=j["id"], gpu=j.get("gpu"))
                say(f"{j['id']}: re-queued after the control KILL (not a failure)", "box8")
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
            try:
                self.check_alarms()
            except Exception as e:                       # noqa: BLE001 - an alarm bug must not stop the monitor
                say(f"alarm check error {type(e).__name__}: {e}", "alarm")
            self.write_status()
            if (self.D / "STOP").exists() and not self.stop_reason:
                self._hard_stop("STOP file present (operator request)")
            elif self.gov.hard_stop() and not self.stop_reason:
                self.gov.note_hard_stop()
                self._hard_stop(f"HARD BUDGET STOP: spent ${self.gov.spent():.2f} (+unpulled payload) >= ${self.gov.r.hard_usd}")

    # ---- D11 alarms: idle capacity beside a queue is a fault, announced loudly
    @staticmethod
    def _rx_bytes() -> int:
        tot = 0
        try:
            for line in open("/proc/net/dev").read().splitlines()[2:]:
                name, rest = line.split(":", 1)
                if name.strip() != "lo":
                    tot += int(rest.split()[0])
        except (OSError, ValueError, IndexError):
            pass
        return tot

    def _raise(self, key: str, text: str, sev: str = "red"):
        if key not in self.alarms:
            self.alarms[key] = {"since": time.time(), "text": text, "sev": sev}
            say(f"ALARM [{key}] {text}", "alarm")
            self.ev("alarm", key=key, text=text)
        else:
            self.alarms[key]["text"] = text

    def _clear(self, key: str):
        if key in self.alarms:
            say(f"ALARM CLEARED [{key}] after {time.time() - self.alarms[key]['since']:.0f} s", "alarm")
            self.ev("alarm_clear", key=key)
            del self.alarms[key]

    def _idle_cause(self, gpu: str, pend: list) -> str:
        if self.paused:
            return "launching is PAUSED (control dir)"
        if time.time() < self.foreign_skip.get(gpu, 0.0):
            return "a foreign process holds VRAM on this GPU"
        ready = [j for j in pend if self.job_ready(j)]
        if ready:
            ok, why = self._ram_ok()
            if not ok:
                return why
            return f"{len(ready)} ready job(s) but the budget governor / admission is refusing (see 'NOT LAUNCHED' lines)"
        fetching = [k for k, sc in self.scrolls.items() if sc["fetching"]]
        if fetching:
            return f"waiting for DATA: {', '.join(fetching)} still fetching (download slower than the GPUs; consider fewer GPUs)"
        if "stage_blocked" in self.alarms:
            return "STAGING BLOCKED by the disk high-water mark (nothing pulled yet)"
        return "no job is staged: the stager has not started a fetch (fetch-parallel / fetch-ahead limits or disk)"

    def check_alarms(self):
        now = time.time()
        lim = self.a.idle_alarm_s
        with self.cv:
            pend = [self.jobs[i] for i in self.order if self.jobs[i]["status"] == "pending"]
            al = self.allowed_set()
            fetching = [k for k, sc in self.scrolls.items() if sc["fetching"]]
        for g in self.gpus:
            key = f"idle:gpu{g}"
            if g in al and g not in self.busy and pend and not self.stop_launch and not self.stop_reason:
                self.idle_since.setdefault(g, now)
                if now - self.idle_since[g] > lim:
                    self._raise(key, f"GPU {g} IDLE {now - self.idle_since[g]:.0f} s beside {len(pend)} pending job(s): {self._idle_cause(g, pend)}")
            else:
                self.idle_since.pop(g, None)
                self._clear(key)
        rx = (self.rx_fn or self._rx_bytes)()
        self.rx_hist.append((now, rx))
        self.rx_hist = [x for x in self.rx_hist if now - x[0] <= lim + 30]
        if fetching and self.rx_hist and now - self.rx_hist[0][0] >= lim * 0.95:
            rate = (rx - self.rx_hist[0][1]) / max(1e-6, now - self.rx_hist[0][0])
            if rate < 50e3:
                self._raise("fetch_stall", f"fetch of {', '.join(fetching)} NOT PROGRESSING: {rate / 1e3:.0f} KB/s network ingress over the last {now - self.rx_hist[0][0]:.0f} s "
                            f"(link down, source throttling, or the fetch pool is stuck)")
            else:
                self._clear("fetch_stall")
        elif not fetching:
            self._clear("fetch_stall")
        try:
            atomic_json(self.out / "ALERTS.json", {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "alarms": [{"key": k, **v} for k, v in self.alarms.items()]})
        except OSError:
            pass

    # ---- link re-probe: trend, degrade/recover announcements, auto-shrink through the control dir
    def link_check_once(self) -> dict | None:
        from . import linkcheck as LK
        m = self.link_probe_fn() if self.link_probe_fn else LK.measure_hosts(sorted(self.scrolls)[0], link_fn())
        d = m["data"]["mbs"]
        LK.trend_append(self.H, d, m["second"]["mbs"])
        if d is None:
            say(f"link re-probe FAILED ({m['data']['why']})", "link")
            return m
        if self.link_base is None:
            self.link_base = d
        base, mn = self.link_base, self.a.min_link_mb_s
        if self.host is not None:
            self.host.net_down_mb_s = d
            self.host.link_measured = True
        say(f"link re-probe {d:.0f} MB/s (start {base:.0f}; {LK.diagnose(m, mn)})", "link")
        if self.link_state == "ok" and (d < 0.6 * base or d < mn):
            self.link_state = "degraded"
            say(f"LINK DEGRADED: {d:.0f} MB/s vs {base:.0f} MB/s at start (gate {mn:g})", "link")
            self.ev("link_degraded", mbs=d, base=base)
            self._raise("link", f"link DEGRADED to {d:.0f} MB/s (was {base:.0f})", "yellow")
            self._maybe_autoshrink(d)
            threading.Thread(target=self.replan, args=(f"link degraded to {d:.0f} MB/s",), daemon=True).start()
        elif self.link_state == "degraded" and d >= 0.85 * base and d >= mn:
            self.link_state = "ok"
            say(f"LINK RECOVERED: {d:.0f} MB/s (start {base:.0f})", "link")
            self.ev("link_recovered", mbs=d)
            self._clear("link")
            threading.Thread(target=self.replan, args=(f"link recovered to {d:.0f} MB/s",), daemon=True).start()
        return m

    def _maybe_autoshrink(self, d: float):
        """When staging now dominates (remaining transfer > 0.5 x remaining compute wall), shrink the allowed GPUs via control/gpus (never below 1; at most once per 30 min)."""
        if self.a.no_auto_shrink or time.time() - self.last_shrink < 1800 or self.host is None:
            return
        with self.cv:
            al = sorted(self.allowed_set(), key=lambda x: int(x) if str(x).isdigit() else 0)
            need_gb = sum(PL.scroll_facts(k, self.cfg, self.host, self.a.z0 or UP_LO, self.a.z1 or UP_HI).fetch_gb for k, sc in self.scrolls.items()
                          if not sc["fetched"] and any(j["scroll"] == k and j["status"] == "pending" for j in self.jobs.values()))
            gpu_h = sum(j["expected_h"] for j in self.jobs.values() if j["status"] == "pending")
        if need_gb <= 0 or not al:
            return
        from . import linkcheck as LK
        sh = LK.shrink_for({"data": {"mbs": d}}, need_gb, gpu_h, len(al), self.a.fetch_parallel)
        if sh["gpus"] < len(al):
            self.last_shrink = time.time()
            self.control.mkdir(parents=True, exist_ok=True)
            (self.control / "gpus").write_text(",".join(al[:sh["gpus"]]) + "\n")
            say(f"AUTO-SHRINK via the control dir: link {d:.0f} MB/s cannot feed {len(al)} GPUs ({need_gb:.0f} GB left to stage vs {gpu_h:.0f} GPU-h of work): "
                f"allowed GPUs -> {al[:sh['gpus']]}; routeB_ctl.sh gpus ... overrides", "link")
            self.ev("autoshrink", gpus=al[:sh["gpus"]], mbs=d)

    def link_loop(self):
        while not self.done_evt.wait(self.a.link_reprobe_s):
            try:
                self.link_check_once()
            except Exception as e:                       # noqa: BLE001
                say(f"link re-probe error {type(e).__name__}: {e}", "link")

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

    # ---- runtime control: <home>/box8/control/ re-read every --control-poll-s
    #   gpus          comma list of allowed GPU indices (live; absent = the --gpus list / all usable)
    #   STOP          graceful: stop launching, let running fits finish, then the run ends (resume = re-run; state is on disk)
    #   PAUSE/RESUME  PAUSE: no new launches while the file exists; RESUME (a command file, consumed) removes PAUSE and STOP
    #   drain.<gpu>   finish that GPU's current fit, then do not use it     kill.<gpu>   SIGTERM that GPU's fit; the interval is re-queued from its last autosave
    # box8/STOP (outside control/) keeps its old meaning: HARD stop of everything.  routeB_ctl.sh writes these files.
    def read_control(self) -> dict:
        c = self.control
        c.mkdir(parents=True, exist_ok=True)
        base = set(self.allowed_init) if self.allowed_init is not None else set(self.gpus)
        gf = c / "gpus"
        if gf.exists():
            want = {x for x in re.split(r"[,\s]+", gf.read_text()) if x}
            for u in sorted(want - set(self.gpus)):
                if u not in self._warned_unknown_gpus:
                    self._warned_unknown_gpus.add(u)
                    say(f"CONTROL: gpus file lists GPU {u} which is not a usable GPU of this box (ignored)", "control")
            base = want & set(self.gpus)
        if (c / "RESUME").exists():
            for n in ("PAUSE", "STOP", "RESUME"):
                try:
                    (c / n).unlink()
                except OSError:
                    pass
        kills = sorted(f.name.split(".", 1)[1] for f in c.glob("kill.*") if not f.name.endswith(".done"))
        for g in kills:                                   # a killed GPU stays out until a `gpus` command names it again
            self.kill_req.add(g)
            (c / f"drain.{g}").write_text("killed\n")
            try:
                (c / f"kill.{g}").rename(c / f"kill.{g}.done")
            except OSError:
                pass
        drain = {f.name.split(".", 1)[1] for f in c.glob("drain.*")}
        return {"allowed": base - drain, "paused": (c / "PAUSE").exists(), "stop": (c / "STOP").exists(), "kills": kills, "drain": sorted(drain)}

    def apply_control(self, why_prefix: str = "CONTROL") -> bool:
        st = self.read_control()
        with self.cv:
            changed = (st["allowed"] != self.allowed_set()) or st["paused"] != self.paused or st["stop"] != self.stop_launch or bool(st["kills"])
            old = self.allowed_set()
            self.allowed, self.paused, self.stop_launch = st["allowed"], st["paused"], st["stop"]
            self.cv.notify_all()
        if changed:
            what = (f"allowed GPUs {sorted(old, key=str)} -> {sorted(st['allowed'], key=str)}; paused={st['paused']} stop={st['stop']} drain={st['drain']} kill={st['kills']}")
            say(f"{why_prefix}: {what}", "control")
            self.ev("control", what=what)
            self.replan(what)
        return changed

    def control_loop(self):
        while not self.done_evt.is_set():
            try:
                self.apply_control()
            except Exception as e:                       # noqa: BLE001 - a bad control file must not kill the scheduler
                say(f"CONTROL ERROR {type(e).__name__}: {e}", "control")
            self.done_evt.wait(self.a.control_poll_s)

    def replan(self, reason: str) -> dict | None:
        """Re-plan the UNSTARTED work for the current set of allowed GPUs (shrink or regrow): tail split re-armed, scrolls that no longer fit the remaining budget/time are
        DEFERRED explicitly (never a mid-fit cutoff; running fits always finish), earlier re-plan deferrals are re-admitted when capacity returns, Route A slots follow."""
        if not self.cfg.get("dynamic") or self.dry:
            return None
        with self.cv:
            al = self.allowed_set()
            groups: dict = {}
            for i in self.order:
                j = self.jobs[i]
                sc = j["scroll"]
                g = groups.setdefault(sc, {"jobs": [], "committed": False})
                if j["status"] == "pending" or (j["status"] == "deferred" and j.get("replan_deferred")):
                    g["jobs"].append(j)
                    if j["attempts"]:
                        g["committed"] = True
                elif j["status"] in ("running", "done", "split", "descended", "failed"):
                    g["committed"] = True
            groups = {k: v for k, v in groups.items() if v["jobs"]}
            avail, extra = [], 0.0
            for sc_, g_ in groups.items():                 # release times: a job cannot start before its scroll is staged (idle GPUs waiting for data cost the box bill)
                rh = 0.0 if self.scrolls[sc_]["fetched"] else self.fetch_est_h(sc_)
                for j_ in g_["jobs"]:
                    j_["_ready_h"] = 0.0 if self.job_ready(j_) else rh
            for g in sorted(al, key=str):
                jid = self.busy.get(g)
                avail.append(self.gov.remaining_h(self.gov.running[jid]) if jid in self.gov.running else 0.0)
            for g, jid in self.busy.items():
                if g not in al and jid in self.gov.running:
                    extra = max(extra, self.gov.remaining_h(self.gov.running[jid]))
            shells = {k: sc["shell"] for k, sc in self.scrolls.items()}
            prio = self.priority or self.order_scrolls()
            res = PL.replan(self.cfg, avail, extra, groups, prio, shells, self.gov.spent(), self.gov.hours(), self.gov.r,
                            self.gov.unpulled_gb + sum(f.payload_gb for f in self.gov.running.values()), self.a.plan_frac)
            restored, deferred = [], []
            for sc, g in groups.items():
                for j in g["jobs"]:
                    if sc in [d for d, _ in res["deferred"]]:
                        if j["status"] == "pending":
                            j["status"], j["replan_deferred"] = "deferred", True
                            j["fail_why"] = next(w for d, w in res["deferred"] if d == sc)
                    elif j["status"] == "deferred" and j.get("replan_deferred"):
                        j["status"], j["replan_deferred"] = "pending", False
                        j["fail_why"] = None
                if sc in [d for d, _ in res["deferred"]]:
                    deferred.append(sc)
                elif any(j.get("replan_deferred") is False for j in g["jobs"]):
                    restored.append(sc)
            self.tail_done = False                          # re-arm the LPT tail split for the new GPU count
            n_run = len([g for g in self.busy if g in al])
            tot_h = sum(j_["expected_h"] for g_ in groups.values() for j_ in g_["jobs"] if j_["scroll"] in res["keep"] or g_["committed"])
            longest = max([j_["expected_h"] for g_ in groups.values() for j_ in g_["jobs"]] or [0.0])
            lines = [f"REPLAN ({reason}): allowed {len(al)} of {len(self.gpus)} GPU(s) {sorted(al, key=str)}, {n_run} busy; unstarted work {tot_h:.1f} GPU-h (p50) -> "
                     f">= {max(tot_h / max(1, len(al)), longest):.1f} h wall on the allowed set; spent ${self.gov.spent():.2f} at {self.gov.hours():.2f} h; "
                     f"{sum(len(g['jobs']) for g in groups.values())} unstarted job(s) in {len(groups)} scroll(s)",
                     f"REPLAN p50 {res['mk50']:.1f} h / ${res['usd50']:.2f}; p90 {res['mk90']:.1f} h / ${res['usd90']:.2f} vs limit ${res['limit_usd']:.2f} / {res['limit_h']:.1f} h"]
            for sc in res["keep"]:
                lines.append(f"REPLAN keep {sc}")
            for sc, w in res["deferred"]:
                lines.append(f"REPLAN DEFERRED {sc}: {w}")
            for sc in restored:
                lines.append(f"REPLAN RESTORED {sc} (capacity is back)")
            slots, sw = self._routea_slots_now()
            lines.append(f"REPLAN Route A: {slots} slot(s) = {sw}")
            nfill = sum(1 for j_ in self.jobs.values() if j_.get("provenance") == "fill_idle")
            if self.a.fill_idle_gpus or nfill:
                lines.append(f"REPLAN fill-idle: {'ON' if self.a.fill_idle_gpus else 'off'}, {nfill} stripe job(s) created by fill_idle so far (min height {self.a.fill_min_height})")
            lines.append(f"REPLAN disk: free {self._disk_state()[1]:.0f} of {self._disk_state()[0]:.0f} GB, staged inputs {sum(self.staged_gb.values()):.0f} GB")
            for sc in groups:
                self.save(sc)
            self.cv.notify_all()
        for l in lines:
            say(l, "replan")
        self.replan_log = lines
        try:
            (self.control).mkdir(parents=True, exist_ok=True)
            (self.control / "PLAN.txt").write_text("\n".join(lines) + "\n")
        except OSError:
            pass
        self.ev("replan", reason=reason, allowed=sorted(al, key=str), keep=res["keep"], deferred=[d for d, _ in res["deferred"]], restored=restored, mk90=res["mk90"], usd90=res["usd90"])
        self._routea_rescale(slots)
        return res

    # ---- Route A on the spare cores
    def _routea_slots_now(self):
        h = self.host or PL.Host([PL.Gpu(g, 40.0) for g in self.gpus])
        return PL.routea_slots(h.phys_cores, len(self.allowed_set()), self.a.routea_reserve_cores, h.ram_gb, self.a.ram_need_gb, h.ram_reserve_gb,
                               self.a.routea_ram_per_grow_gb, self.a.routea_slots)

    def _routea_launch(self, slots: int, why: str):
        from . import routea_side as RA
        cmd = RA.command(self.H, self.routea_names, slots, self.routea_hours, self.a.routea_seeds)
        say(f"Route A START on spare cores: {slots} slots = {why}; {len(self.routea_names)} scroll(s), {self.a.routea_seeds} seeds each, {self.routea_hours:.1f} h; log box8/logs/routeA.log", "routeA")
        self.ev("routea_start", slots=slots, why=why, scrolls=self.routea_names, hours=self.routea_hours)
        self.routea_slots = slots
        self.routea_proc = (self.routea_launch_fn or RA.launch)(self.H, cmd, self.D / "logs" / "routeA.log")

    def _routea_rescale(self, slots: int):
        """Route A's worker count is fixed per process; on a GPU-count change it is restarted (it resumes: finished seeds are skipped) when the slot count moved by >= max(2, 20 %)."""
        p = self.routea_proc
        if p is None or self.routea_slots is None or p.poll() is not None:
            return
        if abs(slots - self.routea_slots) < max(2, 0.2 * self.routea_slots) or slots < 1:
            return
        say(f"Route A RESCALE {self.routea_slots} -> {slots} slots after the GPU change: restarting it (resumable; the grows in flight restart from their last round)", "routeA")
        self.routea_expected_exit = True
        try:
            os.killpg(p.pid, signal.SIGTERM)
            for _ in range(30):
                if p.poll() is not None:
                    break
                time.sleep(1)
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        self._routea_launch(slots, f"rescaled (GPU change): {self._routea_slots_now()[1]}")
        self.ev("routea_rescale", slots=slots)
        self.routea_expected_exit = False

    def routea_should_wait(self) -> bool:
        """--routea-after-first-fit: on = always wait for the first Route B fit; off = start Route A at once; auto = wait only when the link is below the gate (a slow link must
        not be shared with Route A's multi-GB input fetch before the first fit has its data)."""
        m = self.a.routea_after_first_fit
        if m != "auto":
            return m == "on"
        link = self.link_base if self.link_base is not None else (self.host.net_down_mb_s if self.host is not None else 1e9)
        return link < self.a.min_link_mb_s

    def _routea_thread(self):
        a = self.a
        t0 = time.time()
        while self.routea_should_wait() and not self.done_evt.is_set() and not self.busy and time.time() - t0 < 1800:
            time.sleep(2)                                    # start after the first fit is on a GPU: the fits come first
        if self.done_evt.is_set() or self.stop_reason:
            return
        slots, why = self._routea_slots_now()
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
        self.routea_names = names
        self.routea_hours = a.routea_hours or max(1.0, (getattr(a, "planned_makespan_h", None) or 4.0) * 0.9)
        self._routea_launch(slots, why)
        self.routea_stop = threading.Event()
        RA.publisher(self.H / "routeA_work", self.out, self.routea_stop, self.ev, every=max(5.0, a.poll_s * 4))
        while not self.done_evt.is_set():
            time.sleep(5)
            p = self.routea_proc
            if p.poll() not in (None, 0) and not self.routea_expected_exit:
                say(f"Route A EXITED rc={p.returncode} (see box8/logs/routeA.log); Route B is unaffected", "routeA")
                self.ev("routea_exit", rc=p.returncode)
                break

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
        if not self.dry:
            if (self.control / "STOP").exists():
                say("CONTROL: a control/STOP file was left from an earlier run; removing it (restarting is deliberate)", "control")
                (self.control / "STOP").unlink()
            self.apply_control("CONTROL (startup)")
            threading.Thread(target=self.control_loop, daemon=True, name="control").start()
        if self.a.link_reprobe_s > 0 and not self.dry and (not self.a.fake_gpus or self.link_probe_fn):
            threading.Thread(target=self.link_loop, daemon=True, name="link").start()
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
    ap.add_argument("--gpus", default=None, help="comma list of GPU indices allowed at start (default: all usable); change at run time with routeB_ctl.sh gpus 0,1,2")
    ap.add_argument("--foreign-mib", type=float, default=1500.0, help="a GPU whose memory.used exceeds this before we launch anything is held by a foreign process: warned and skipped")
    ap.add_argument("--force-gpus", action="store_true", help="use GPUs even if a foreign process holds VRAM")
    ap.add_argument("--control-poll-s", type=float, default=10.0, help="how often the control dir <home>/box8/control/ is re-read")
    ap.add_argument("--link-mb-s", type=float, default=None, help="aggregate ingress MB/s; skips the probe (default: measure 8 parallel 8 MB range reads of a real input at start)")
    ap.add_argument("--no-link-probe", action="store_true", help="do not measure the link; use the quoted 860 Mbps")
    ap.add_argument("--min-link-mb-s", type=float, default=20.0, help="GATE: a data-host ingress below this STOPS the run (exit 5, nothing fetched) unless --accept-slow-link")
    ap.add_argument("--accept-slow-link", action="store_true", help="continue on a slow link; the planner auto-shrinks --gpus / --fetch-parallel to what the link can feed")
    ap.add_argument("--link-reprobe-s", type=float, default=600.0, help="re-measure the link this often during the run (0 = never); trend in box8/link/trend.jsonl")
    ap.add_argument("--no-auto-shrink", action="store_true", help="do not shrink the allowed GPUs through the control dir when the link degrades")
    ap.add_argument("--idle-alarm-s", type=float, default=180.0, help="D11: an allowed GPU idle this long beside pending work, or a fetch with no network progress this long, raises a red alarm")
    ap.add_argument("--torch-cuda", default="auto", choices=["auto", "cu126", "cu128", "cu129"],
                    help="torch build for the env: cu126 = validated; Blackwell (compute capability >= 12) auto-selects cu129 (same torch 2.13.0, UNVALIDATED numerically); "
                         "cu128 = torch 2.11.0, UNVALIDATED. The GPU kernel smoke test must pass before any fetch or fit")
    ap.add_argument("--fill-idle-gpus", action="store_true",
                    help="(default OFF) when allowed GPUs idle while the next scroll downloads, re-split the NOT-YET-STARTED jobs of the staged scroll into more z-stripes so every idle GPU "
                         "gets one. Trade-off: stripes cost ~1.1-1.35x the GPU-h of the whole-scroll fit and add seams, but idle GPUs bill the same wall-clock; running/finished jobs are never touched")
    ap.add_argument("--fill-min-height", type=int, default=2800, help="smallest stripe the fill may create (2800 = the known-good 16 GB size; never below)")
    ap.add_argument("--replan-pending-on-resume", action="store_true", help="on resume with a changed --gpus/--max-height/VRAM, re-plan the pending jobs instead of keeping them")
    ap.add_argument("--routea-after-first-fit", default="auto", choices=["auto", "on", "off"],
                    help="start Route A (and its input fetch) only once the first Route B fit is running: auto = on when the link is below --min-link-mb-s")
    ap.add_argument("--no-stripe-staging", action="store_true",
                    help="stage a scroll's inputs as ONE unit (default: per stripe; a stripe fit starts as soon as the tracks file and ITS lasagna z-range have landed)")
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


def nvsmi_query(cols: str) -> list[list[str]]:
    """Rows of `nvidia-smi --query-gpu=<cols>` (patched in tests)."""
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={cols}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SystemExit(f"ROUTEB FAIL box8-gpus: cannot run nvidia-smi ({e})")
    if out.returncode != 0 or not out.stdout.strip():
        raise SystemExit(f"ROUTEB FAIL box8-gpus: nvidia-smi rc={out.returncode}: {out.stderr.strip()[:200]}")
    return [[x.strip() for x in r.split(",")] for r in out.stdout.strip().splitlines()]


def gpu_info(a) -> tuple[list[PL.Gpu], list[PL.Gpu]]:
    """(all usable GPUs, initially allowed GPUs).  Usable = every card without a FOREIGN process: memory.used above --foreign-mib at startup (e.g. a llama-server holding
    VRAM) is warned about and skipped unless --force-gpus.  Allowed = the --gpus list (default all usable); the control dir's `gpus` file changes it at run time."""
    if a.fake_gpus:
        gs = [PL.Gpu(str(i), a.fake_vram_gib, a.gpu_speed) for i in range(a.fake_gpus)]
        want = [g for g in (a.gpus or "").split(",") if g]
        return gs, [g for g in gs if not want or g.idx in want]
    rows = nvsmi_query("index,name,memory.total,memory.used")
    usable, skipped = [], []
    for r in rows:
        used = float(r[3])
        if used > a.foreign_mib and not a.force_gpus:
            skipped.append(r[0])
            say(f"WARNING GPU {r[0]} ({r[1]}) holds {used:.0f} MiB > --foreign-mib {a.foreign_mib:g} before we launched anything (a foreign process, e.g. a llama-server): "
                f"SKIPPED. Use --force-gpus to use it anyway.", "box8")
            continue
        usable.append(PL.Gpu(r[0], float(r[2]) / 1024.0, a.gpu_speed))
    say("GPUs: " + "; ".join(f"{r[0]}={r[1]} {float(r[2]) / 1024:.1f} GiB used {float(r[3]):.0f} MiB{' [SKIPPED foreign]' if r[0] in skipped else ''}" for r in rows), "box8")
    want = [g for g in (a.gpus or "").split(",") if g]
    for w in want:
        if w in skipped:
            say(f"--gpus lists GPU {w} but it is held by a foreign process and was skipped (--force-gpus overrides)", "box8")
        elif w not in [g.idx for g in usable]:
            say(f"--gpus lists GPU {w} which nvidia-smi does not report", "box8")
    allowed = [g for g in usable if not want or g.idx in want]
    if not allowed:
        raise SystemExit("ROUTEB FAIL box8-gpus: no allowed GPU left (all requested GPUs are missing or held by foreign processes; --force-gpus to override)")
    return usable, allowed


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


def measure_link(url: str, n_conn: int = 8, chunk: int = 8 << 20, timeout: float = 60.0, budget_s: float | None = None):
    from . import linkcheck
    return linkcheck.measure_link(url, n_conn, chunk, timeout, budget_s)


def link_fn():
    """Probe function for linkcheck.measure_hosts; tests patch LINK_FN(url) -> (MB/s | None, why)."""
    if LINK_FN is None:
        return None
    return lambda url, budget_s: LINK_FN(url)


def probe_link(a, host, scroll: str) -> dict | None:
    """Measure the link (both hosts) or reuse a fresh <home>/box8/link/first.json written by routeB.linkcheck before the env was built; set the planner's link rate;
    return the measurement dict (or None when it could not be measured: announced, the quoted rate stays)."""
    from . import linkcheck
    if a.link_mb_s:
        host.net_down_mb_s, host.link_measured = a.link_mb_s, True
        say(f"link {a.link_mb_s:g} MB/s from --link-mb-s", "link")
        return {"data": {"mbs": a.link_mb_s, "why": "--link-mb-s"}, "second": {"mbs": None, "why": "not probed"}, "t": time.time()}
    if a.no_link_probe or (a.fake_gpus and LINK_FN is None):
        say(f"link NOT measured ({'--no-link-probe' if a.no_link_probe else 'fake GPUs'}): using the quoted {host.net_down_mb_s:.0f} MB/s", "link")
        return None
    first = linkcheck.load_first(home())
    m = first["measured"] if first else linkcheck.measure_hosts(scroll, link_fn())
    d = m["data"]["mbs"]
    if d is None and m["second"]["mbs"] is None:
        say(f"link probe FAILED (data: {m['data']['why']}; second: {m['second']['why']}); using the quoted {host.net_down_mb_s:.0f} MB/s", "link")
        return None
    if d is None:
        say(f"link probe: data host failed ({m['data']['why']}); using the quoted {host.net_down_mb_s:.0f} MB/s", "link")
        return m
    host.net_down_mb_s, host.link_measured = d, True
    say(f"link MEASURED {d:.0f} MB/s data host ({m['data']['why']}); second host "
        f"{'n/a' if m['second']['mbs'] is None else format(m['second']['mbs'], '.0f') + ' MB/s'}; quoted {860 / 8:.0f} MB/s", "link")
    if not first:
        linkcheck.trend_append(home(), d, m["second"]["mbs"])
    return m


LINK_FN = None                                  # tests patch this


def disk_measure(a, H: Path) -> tuple[float, float, list[str]]:
    """(total GB, free GB, the df lines used) of the volume that actually holds ROUTEB_HOME (created first).  Overrides are announced; the base use is clamped to the volume."""
    H.mkdir(parents=True, exist_ok=True)
    du = shutil.disk_usage(H)
    lines = [f"df {H} -> total {du.total / 1e9:.0f} GB, used {du.used / 1e9:.0f} GB, free {du.free / 1e9:.0f} GB (statvfs of the path itself, not of its parent)"]
    total, free = du.total / 1e9, du.free / 1e9
    if a.disk_total_gb is not None:
        lines.append(f"OVERRIDE --disk-total-gb {a.disk_total_gb:g} (df said {total:.0f})")
        total = a.disk_total_gb
        free = min(free, total) if a.disk_free_gb is None else free
    if a.disk_free_gb is not None:
        lines.append(f"OVERRIDE --disk-free-gb {a.disk_free_gb:g} (df said {free:.0f})")
        free = a.disk_free_gb
    if free > total:
        lines.append(f"free {free:.0f} > total {total:.0f}: clamped to total")
        free = total
    return total, free, lines


def build_host(a, H: Path, gpus=None, allowed=None) -> PL.Host:
    if gpus is None:
        gpus, allowed = gpu_info(a)
    ram = a.fake_ram_gb if a.fake_gpus else meminfo_gb()[1]
    total, free, lines = disk_measure(a, H)
    for l in lines:
        say(l, "disk")
    h = PL.Host(allowed, phys_cores=physical_cores(a), ram_gb=ram, disk_total_gb=total, disk_free_gb=free, disk_high_water_frac=a.disk_high_water,
                fetch_files_per_s=a.fetch_files_per_s, fetch_parallel=a.fetch_parallel, fetch_ahead=a.fetch_ahead, ram_per_fit_gb=a.ram_need_gb,
                reserve_cores=a.routea_reserve_cores, stripe_staging=not a.no_stripe_staging)
    h.all_gpus = list(gpus)
    h.disk_lines = lines
    h.env_in_base = not (a.disk_free_gb is not None)        # df of the real home already counts the env; an override does not
    return h


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
    sch.dry = a.dry_run
    pf = H / "box8" / "prefetch.json"
    if not a.dry_run and pf.exists():                    # bytes the early prefetch pulled before this process existed: account the ingress once
        try:
            d_ = json.loads(pf.read_text())
            if not d_.get("accounted"):
                gov.transfer(box_download_gb=float(d_.get("gb", 0.0)), what="early prefetch")
                d_["accounted"] = True
                pf.write_text(json.dumps(d_, indent=1))
                say(f"early prefetch: {d_.get('gb', 0):.2f} GB already on disk ({len(d_.get('done', []))} stripe(s)); accounted as ingress", "box8")
        except (OSError, ValueError):
            pass                                  # a dry run reads state (if any) but writes nothing
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
    sch.gpus = [g.idx for g in host.all_gpus]
    sch.allowed_init = {g.idx for g in host.gpus}
    ng = max(1, len(host.gpus))                          # the ALLOWED set: wall and budget projection use it, not every card on the box
    m_link = probe_link(a, host, run_names[0])
    if m_link and m_link["data"]["mbs"] is not None:
        from . import linkcheck as LK
        d_ = m_link["data"]["mbs"]
        if d_ < a.min_link_mb_s and not a.accept_slow_link:
            print(LK.box([f"LINK CHECK   VERDICT: BAD   data host {d_:.1f} MB/s < --min-link-mb-s {a.min_link_mb_s:g}", f"diagnosis: {LK.diagnose(m_link, a.min_link_mb_s)}", "",
                          "DESTROY THIS BOX, or re-run with --accept-slow-link to continue anyway."]), flush=True)
            return 5
        if d_ < a.min_link_mb_s or (d_ < 2 * a.min_link_mb_s and a.accept_slow_link):
            fb = LK.load_first(H)
            sh = (fb or {}).get("shrink")
            if not sh:
                facts_ = [PL.scroll_facts(s_, cfg, host, z0, z1) for s_ in run_names]
                sh = LK.shrink_for(m_link, sum(f.fetch_gb for f in facts_), sum(L.fit_hours(f.name, f.shell, f.z1 - f.z0) for f in facts_), len(host.gpus), a.fetch_parallel)
            if a.accept_slow_link and not a.gpus and sh["gpus"] < len(host.gpus):
                say(f"AUTO-SHRINK (slow link {d_:.1f} MB/s accepted): allowed GPUs {len(host.gpus)} -> {sh['gpus']}, --fetch-parallel {a.fetch_parallel} -> {sh['fetch_parallel']}", "link")
                host.gpus = host.gpus[:sh["gpus"]]
                a.fetch_parallel = host.fetch_parallel = sh["fetch_parallel"]
    sch.allowed_init = {g.idx for g in host.gpus}
    if m_link and m_link["data"]["mbs"] is not None:
        sch.link_base = m_link["data"]["mbs"]
    plan = None
    if not legacy and not a.no_plan:
        plan = PL.make_plan(host, rates, run_names, cfg, z0, z1, a.plan_frac, a.max_height, [x for x in (a.priority or "").split(",") if x] or None, sch.heights_path, a.order)
        slots, sw = PL.routea_slots(host.phys_cores, ng, a.routea_reserve_cores, host.ram_gb, a.ram_need_gb, host.ram_reserve_gb, a.routea_ram_per_grow_gb, a.routea_slots)
        ra = (f"PLAN Route A on the spare cores: {'ON' if a.routea else 'OFF (--no-routea)'}; {slots} grow slot(s) = {sw}" if a.routea else "PLAN Route A: OFF (--no-routea)")
        print(PL.render(host, rates, plan, not_runnable, ra), flush=True)
        if a.fill_idle_gpus:
            print(f"PLAN --fill-idle-gpus ON: while the next scroll downloads, idle allowed GPUs get z-stripes of the staged scroll (each >= {a.fill_min_height} slices). "
                  f"TRADE-OFF: stripes cost ~1.1-1.35x the GPU-h of the whole-scroll fit and add seams/overlap (200 slices); the box bills wall-clock, so idle GPUs cost the same as busy ones. "
                  f"Never touches running or finished jobs.", flush=True)
        if plan["keep"]:
            a.planned_makespan_h = plan["p50"]["makespan_h"]
        sch.ev("plan", keep=plan["keep"], deferred=plan["deferred"])
        if not plan["keep"]:
            return 3
        for n_, why in plan["deferred"]:
            sch.ev("deferred", scroll=n_, why=why)
        run_names = list(plan["keep"])
        sch.priority = list(plan.get("priority", []))
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
    lead = min([PL.fetch_time(host, PL.scroll_facts(s_, cfg, host, z0, z1)) for s_ in run_names] or [0.0])     # GPUs idle until the first scroll is staged
    wall = max(tot_h / ng, max([j["expected_h"] for j in sch.jobs.values() if j["status"] == "pending"] or [0])) + lead
    proj = gov.projected(extra_expected_h=wall, extra_payload_gb=sum(j["payload_gb"] for j in sch.jobs.values()))
    say(f"PLAN: {len(sch.jobs)} job(s) over {len(run_names)} scroll(s), {tot_h:.1f} GPU-h expected on {ng} allowed GPU(s) of {len(sch.gpus)} = {wall:.1f} h wall at perfect packing (incl. {lead:.2f} h until the first scroll is staged); "
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
