"""Box planner: compute total load, payloads, GPU assignment, stagger timeline, disk timeline and $ BEFORE any compute is launched.  Pure functions, no I/O except
reading the scroll specs; unit-tested with fake GPUs / disks / RAM (routeB/tests/test_planner.py).  `routeB_run.sh --mode box8 --dry-run` prints it.

WHAT IT DECIDES
  1. stripe height per scroll   = min(span, VRAM model (ladder.computed_height: peak GiB ~ 3.97 + 0.00208 x slices), 2^24-track limit [lifted by default when tracks.py has _multinomial_chunked; automatic fallback], --max-height)
  2. GPU schedule               = ONE GPU PER SCROLL, SMALLEST FIRST (SPT: most scrolls finished per dollar), every GPU busy; once fewer unstarted jobs remain than
                                  GPUs, the largest remaining scrolls are split into z-STRIPES across the idle GPUs (LPT tail balancing; the split is accepted only
                                  if it shortens the simulated makespan).  The parallel unit is the z-stripe: fit_spiral is single-GPU.
  3. staging                    = scroll inputs are fetched just ahead of the GPUs (fetch-ahead) and only while the disk high-water mark allows; a scroll's inputs are
                                  deleted once its payload is pulled+verified.
  4. admission                  = the whole plan must finish its p90 case within --plan-frac (0.8) of the soft budget AND of the max run hours; scrolls that do not fit
                                  are DEFERRED explicitly by priority, here, up front -- never cut off mid-fit.
  5. Route A on spare cores     = slots = physical_cores - 2 x busy_GPUs - reserve, RAM-checked.
COST MODEL (every constant is a named parameter with provenance; see ladder.py for the GPU-hour model)
  one fit of height h:  t = t_full(scroll) x (h/13000)^0.90 / gpu_speed          (V100-class basis; A100 speed UNMEASURED -> 1.0 unless --gpu-speed)
  wall bill = makespan x (machine $/h + disk $/h); + fetched GB x ingress $/TB + payload GB x egress $/TB.
IS ONE SCROLL ON SEVERAL GPUs FASTER / IS FULL HEIGHT IN PARALLEL AS EFFICIENT AS SERIAL?  Faster in wall-clock, not cheaper in GPU-h: see docs/ROUTEB_SCHEDULING.md
  (5 stripes of 2,800 = ~1.25x the GPU-h of one 13,000-slice fit but ~1/5 of the wall; hence stripes only where a GPU would otherwise idle).
"""
from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass, field

from . import ladder as L
from .common import UP_HI, UP_LO, spec

OVERLAP = 200
GRID = L.GRID

# measured / assumed input volumes ------------------------------------------------------------------------------------------------------------------------------
LAS_GB_PER_SLICE = 0.245 / 1012.0            # PHerc0211 smoke slab z 9000-9500 (+-256 margin = 1012 slices): nx+ny+grad_mag = 0.245 GB (n = 1 scroll)
LAS_OBJ_PER_SLICE = 3 * 43731 / 13512.0      # MEASURED on the PRO 5000 run: ny 43,219 and grad_mag 44,243 objects for z 4500-17500 (+-256 margin = 13,512 slices) = 3.24 per slice per field,
                                             # x3 fields (the earlier 17.8 came from a 1,012-slice slab whose z-chunk rounding inflated it ~1.8x); varies a little with the scroll's xy size
WORK_GB_BASE, WORK_GB_PER_TRACK_GB = 8.7, 5.0   # fit dir / caches / run outputs per scroll: ASSUMED, calibrated so input+work = 35 GB (smallest scroll) .. 90 GB (largest)
                                               # (the coordinator's per-scroll disk figures); refine from a measured full-height fit dir


@dataclass
class Gpu:
    idx: str
    vram_gib: float
    speed: float = 1.0


@dataclass
class Host:
    gpus: list
    phys_cores: int = 48
    ram_gb: float = 516.0
    disk_total_gb: float = 934.0
    disk_free_gb: float = 934.0
    disk_high_water_frac: float = 0.85
    env_gb: float = 15.0                       # python env + tools on the box (pny readiness proof: env 7.4 + tools 7.6 GB)
    net_down_mb_s: float = 860 / 8.0           # 860 Mbps (quoted)
    net_up_mb_s: float = 913 / 8.0
    fetch_files_per_s: float = 180.0           # objects/s of ONE process with keep-alive connections, 128 workers: MEASURED 181-236 on pny under load ~95 (n = 1 run per setting, 600 objects); urllib new-connection-per-object gave 13-121
    fetch_parallel: int = 8                    # concurrent scroll fetches
    fetch_ahead: int = 2                       # fetched-but-unstarted scrolls kept ready beyond the free GPUs
    pull_latency_h: float = 0.15               # DONE -> pulled+verified (poll interval + transfer) before the inputs are deleted
    ram_per_fit_gb: float = 40.0               # host RSS per fit 22-40 GB (measured on a 4060 Ti box and a V100 box)
    ram_reserve_gb: float = 30.0
    reserve_cores: int = 4
    stripe_staging: bool = True                 # inputs are staged per stripe (tracks, then each stripe's lasagna chunks); False = whole scroll at once

    @property
    def n(self) -> int:
        return len(self.gpus)

    @property
    def high_water_gb(self) -> float:
        return self.disk_total_gb * self.disk_high_water_frac

    @property
    def used0_gb(self) -> float:
        """Disk already in use before we stage anything, clamped to the volume.  A df of the real ROUTEB_HOME already counts the env; with a free-space override it
        does not, so env_gb is added then (env_in_base False)."""
        base = max(0.0, self.disk_total_gb - self.disk_free_gb) + (0.0 if getattr(self, "env_in_base", False) else self.env_gb)
        return min(base, self.disk_total_gb)

    @property
    def min_vram(self) -> float:
        return min(g.vram_gib for g in self.gpus)

    def max_concurrent_fits(self) -> int:
        return max(0, min(self.n, int((self.ram_gb - self.ram_reserve_gb) // self.ram_per_fit_gb)))


@dataclass
class Facts:
    name: str
    shell: int
    tracks_gb: float
    roles: list
    z0: int
    z1: int
    height: int
    height_why: str
    las_gb: float
    las_objects: int
    work_gb: float
    startup_split: bool = False

    @property
    def input_gb(self) -> float:
        return self.tracks_gb + self.las_gb + self.work_gb

    @property
    def fetch_gb(self) -> float:
        return self.tracks_gb + self.las_gb


def scroll_facts(scroll: str, cfg: dict, host: Host, z0: int = UP_LO, z1: int = UP_HI, max_height: int = L.FULL_SPAN, heights_path=None) -> Facts:
    sp = spec(scroll)
    tr = sum(v for k, v in sp["tracks"]["files"].items() if k.endswith(".dbm")) / 1e9
    span = z1 - z0
    h, why = L.start_height_for(cfg, scroll, host.min_vram, span, max_height=max_height, heights_path=heights_path)
    sl = span + 512
    return Facts(scroll, int(sp["shell_outer_winding_idx"]), tr, sp.get("roles", []), z0, z1, h, why, LAS_GB_PER_SLICE * sl, int(LAS_OBJ_PER_SLICE * sl),
                 WORK_GB_BASE + WORK_GB_PER_TRACK_GB * tr)


def make_jobs(cfg: dict, f: Facts) -> list[dict]:
    return L.jobs_for_height(cfg, f.name, f.shell, f.z0, f.z1, f.height, OVERLAP)


def hours(j: dict, q: str) -> float:
    return j["p90_h"] if q == "p90" else j["expected_h"]


def lpt_makespan(avail: list[float], hs: list[float]) -> float:
    a = sorted(avail)
    for h in sorted(hs, reverse=True):
        a[0] += h
        a.sort()
    return max(a) if a else 0.0


def release_makespan(avail: list, jobs: list) -> float:
    """List scheduling with release times (a job cannot start before its scroll's inputs are staged): jobs = [(ready_h, hours)], taken in release order, larger first."""
    a = sorted(avail)
    for ready, h in sorted(jobs, key=lambda x: (x[0], -x[1])):
        a[0] = max(a[0], ready) + h
        a.sort()
    return max(a)


def pieces_for(cfg: dict, j: dict, k: int, shell: int) -> list[dict]:
    S = j["z1"] - j["z0"]
    h = max(L.MIN_HEIGHT, -(-int((S + (k - 1) * OVERLAP) / k) // GRID) * GRID)
    if h >= S:
        return [j]
    return L.plan(cfg, j["scroll"], shell, j["z0"], j["z1"], start=L.dyn_rung(cfg, h, OVERLAP), parent=j["id"])


def tail_split(cfg: dict, avail: list[float], jobs: list[dict], shells: dict, q: str = "p50", min_gain: float = 0.03) -> tuple[list[dict], list[str]]:
    """LPT tail balancing.  `avail` = time each GPU becomes free; `jobs` = unstarted jobs.  For every candidate target T (a job's hours / k), each job longer than
    T is split into the fewest z-stripes whose pieces are <= T; the T with the smallest simulated LPT makespan (fewest pieces on ties) wins, and is accepted only if it
    beats the unsplit makespan by >= min_gain.  (A one-job-at-a-time greedy cannot see that two equal jobs must BOTH be split.)  Returns (new jobs, notes)."""
    n = len(avail)
    base = lpt_makespan(avail, [hours(j, q) for j in jobs])
    cands = sorted({round(hours(j, q) / k, 3) for j in jobs for k in range(1, n + 1)})
    cache: dict = {}

    def pieces(j, k):
        key = (j["id"], k)
        if key not in cache:
            cache[key] = pieces_for(cfg, j, k, shells[j["scroll"]]) if k > 1 and not j.get("no_split") and j["z1"] - j["z0"] >= 2 * L.MIN_HEIGHT else [j]
        return cache[key]

    best = None
    for T in cands:
        new = []
        for j in jobs:
            for k in range(1, n + 1):
                ps = pieces(j, k)
                if max(hours(x, q) for x in ps) <= T + 1e-9 or k == n:
                    break
            new += ps
        m = lpt_makespan(avail, [hours(x, q) for x in new])
        if best is None or (m, len(new)) < (best[0], len(best[1])):
            best = (m, new)
    if not best or best[0] > base * (1 - min_gain) or len(best[1]) == len(jobs):
        return list(jobs), []
    m, new = best
    notes = [f"tail split to a ~{max(hours(x, q) for x in new):.1f} GPU-h target: {len(jobs)} job(s) -> {len(new)} stripe-job(s), {sum(hours(x, q) for x in jobs):.1f} -> "
             f"{sum(hours(x, q) for x in new):.1f} GPU-h total, simulated tail {base:.2f} h -> {m:.2f} h (x{m / base:.2f})"]
    for j in jobs:
        ps = [x for x in new if x["scroll"] == j["scroll"] and x["z0"] >= j["z0"] and x["z1"] <= j["z1"]]
        if len(ps) > 1:
            notes.append(f"  {j['id']} ({j['z1'] - j['z0']} slices, {hours(j, q):.1f} GPU-h) -> {len(ps)} stripes of ~{ps[0]['z1'] - ps[0]['z0']} slices")
    for x in new:
        x["no_split"] = True
    return new, notes


def routea_slots(phys_cores: int, busy_gpus: int, reserve: int, ram_gb: float, fits_ram_gb: float, ram_reserve_gb: float, ram_per_grow_gb: float = 6.0,
                 override: int | None = None) -> tuple[int, str]:
    """Route A grow slots on the spare cores: physical_cores - 2 x busy_GPUs - reserve, capped by RAM left after the fits; --routea-slots overrides."""
    by_cores = max(0, phys_cores - 2 * busy_gpus - reserve)
    by_ram = max(0, int((ram_gb - ram_reserve_gb - busy_gpus * fits_ram_gb) // ram_per_grow_gb))
    n = min(by_cores, by_ram)
    if override is not None:
        return override, f"override {override} (cores formula {by_cores}, RAM cap {by_ram})"
    return n, f"min(cores {phys_cores} - 2 x {busy_gpus} busy GPUs - {reserve} reserve = {by_cores}, RAM ({ram_gb:g} - {ram_reserve_gb:g} - {busy_gpus}x{fits_ram_gb:g})/{ram_per_grow_gb:g} GB per grow = {by_ram})"


# ------------------------------------------------------------------------------------------------ simulation
def fetch_time(host: Host, f: Facts) -> float:
    """Hours to stage one scroll: the slower of its object count at the per-scroll fetch rate and its bytes at this scroll's share of the link."""
    by_obj = f.las_objects / host.fetch_files_per_s / 3600.0
    by_net = f.fetch_gb * 1000.0 / (host.net_down_mb_s / max(1, host.fetch_parallel)) / 3600.0
    return max(by_obj, by_net)


def stage_schedule(host: Host, f: Facts, jobs: list[dict], t0: float) -> float:
    """Per-stripe staging: the tracks file first, then the lasagna chunks of each stripe in z order.  Sets job['_ready_t'] (when that stripe's fit may start) and returns the time the
    whole scroll is staged.  With host.stripe_staging False every job is ready only when the whole scroll is."""
    share = host.net_down_mb_s / max(1, host.fetch_parallel)
    if not getattr(host, "stripe_staging", True) or len(jobs) <= 1:
        done = t0 + fetch_time(host, f)
        for j in jobs:
            j["_ready_t"] = done
        return done
    t = t0 + f.tracks_gb * 1000.0 / share / 3600.0
    for j in sorted(jobs, key=lambda x: (x["z0"], x["z1"])):
        sl = j["z1"] - j["z0"] + 512
        las_t = max(LAS_OBJ_PER_SLICE * sl / host.fetch_files_per_s / 3600.0, LAS_GB_PER_SLICE * sl * 1000.0 / share / 3600.0)
        t += las_t
        j["_ready_t"] = t
    return t


def vram_lines(host: Host, margin_gib: float = 1.5) -> list[str]:
    """The height model's max stripe per card: this box's cards first, then the reference table."""
    out = []
    mem = sorted({round(g.vram_gib, 1) for g in host.gpus})
    for m in mem:
        h = L.computed_height(m, margin_gib)
        n = 1 if h >= L.FULL_SPAN else -(-L.FULL_SPAN // h)
        out.append(f"PLAN VRAM: this box {sum(1 for g in host.gpus if round(g.vram_gib, 1) == m)} x {m} GiB -> max stripe {h:,} slices "
                   f"({'a 13,000-slice scroll fits one GPU' if n == 1 else f'a 13,000-slice scroll needs >= {n} stripes'}); model {L.VRAM_A_GIB} + {L.VRAM_B_GIB_PER_SLICE:.5f} GiB/slice, margin {margin_gib} GiB")
    out.append("PLAN VRAM reference (usable GiB -> max stripe): " + "; ".join(f"{n} {u:g} -> {h:,}" for n, _nom, u, h, _k in L.card_table(margin_gib)))
    return out


def download_report(host: Host, rates, plan: dict, frac: float = 0.5) -> list[str]:
    """Link line + per-scroll fetch vs compute + LOUD warnings when download time dominates (fetch_h > frac x compute_h) with a recommendation."""
    out = []
    facts = [plan["facts"][n] for n in plan["keep"]]
    if not facts:
        return out
    gb = sum(f.fetch_gb for f in facts)
    n = max(1, plan["p50"]["n_gpus_used"])
    link = host.net_down_mb_s
    h_bytes = gb * 1000.0 / link / 3600.0
    h_obj = sum(f.las_objects for f in facts) / (host.fetch_files_per_s * max(1, host.fetch_parallel)) / 3600.0
    src = "MEASURED at start" if getattr(host, "link_measured", False) else "QUOTED/assumed, not measured"
    out.append(f"PLAN link {link:.0f} MB/s ({src}) -> fetching {gb:.0f} GB takes {h_bytes:.2f} h of pure transfer (objects bound: {h_obj:.2f} h at {host.fetch_files_per_s:g} files/s x {host.fetch_parallel} scrolls in parallel)")
    gpu_h = plan["p50"]["gpu_h"]
    compute_h = gpu_h / n
    idle = plan["p50"]["idle_gpu_h_before_tail"]
    out.append(f"PLAN idle GPUs waiting for data (p50): {idle:.1f} GPU-h = {idle / n:.2f} h of the box bill = ${idle / n * rates.eff_hour_usd:.2f} of ${plan['m50']['total_usd']:.2f}")
    bad = []
    for f in facts:
        fh = fetch_time(host, f)
        ch = sum(j["expected_h"] for j in make_jobs(plan["cfg"], f)) if plan.get("cfg") else L.fit_hours(f.name, f.shell, f.z1 - f.z0, "p50")
        out.append(f"PLAN   {f.name:<10} fetch {f.fetch_gb:5.1f} GB, {f.las_objects:,} objects -> {fh:5.2f} h to stage vs {ch:5.1f} GPU-h of fit" + ("   <-- DOWNLOAD DOMINATES" if fh > frac * ch else ""))
        if fh > frac * ch:
            bad.append(f.name)
    agg = max(h_bytes, h_obj)
    if bad or agg > frac * compute_h:
        k = max(1, int(frac * gpu_h / agg)) if agg > 0 else n
        need = gb * 1000.0 / (frac * compute_h * 3600.0) if compute_h > 0 else 0.0
        out.append(f"PLAN !!! DOWNLOAD TIME DOMINATES: staging all inputs needs {agg:.2f} h vs {compute_h:.2f} h of compute wall on {n} GPU(s) (limit {frac:.0%}); "
                   f"per scroll: {bad or 'none individually'}.")
        out.append(f"PLAN !!! RECOMMENDATION: shrink --gpus to ~{k} (the GPUs beyond that mostly wait for data), or rent a box with ingress >= {need:.0f} MB/s "
                   f"(this link: {link:.0f} MB/s), or restrict --z0/--z1 so each scroll fetches less lasagna.")
    return out


def simulate(host: Host, facts: list[Facts], cfg: dict, q: str = "p50", tail: bool = True) -> dict:
    """Discrete-event simulation of the whole box run.  `facts` is the dispatch order (SPT).  Returns timeline, makespan, disk peak, GB moved."""
    n = min(host.n, max(1, host.max_concurrent_fits()))
    S = {}
    for f in facts:
        S[f.name] = dict(f=f, jobs=make_jobs(cfg, f), state="unstaged", ready=None, running=0, finished=0, rel=None, pay=0.0, first=None, pulled=None)
    shells = {f.name: f.shell for f in facts}
    order = [f.name for f in facts]
    t = 0.0
    running: list = []                  # (end, seq, gpu, job)
    fetching: list = []                 # (t_done, seq, scroll)
    pulls: list = []                    # (t_done, seq, scroll)
    seq = 0
    free = list(range(n))
    used = host.used0_gb
    peak = used
    disk_log = [(0.0, used)]
    sched, fetch_log, arrivals, notes, idle_gpu_h = [], [], [], [], 0.0
    gb_in = gb_out = 0.0
    last_t = 0.0
    blocked = False
    tail_done = False

    def n_pending() -> int:
        return sum(len(s["jobs"]) for s in S.values())

    for _guard in range(100000):
        # ---- completions at t
        while running and running[0][0] <= t + 1e-12:
            end, _s, g, j = heapq.heappop(running)
            free.append(g)
            s = S[j["scroll"]]
            s["running"] -= 1
            s["finished"] += 1
            s["pay"] += j["payload_gb"]
            arrivals.append({"id": j["id"], "t_h": round(end, 3), "gb": j["payload_gb"], "gpu": g})
            if not s["jobs"] and s["running"] == 0:
                seq += 1
                heapq.heappush(pulls, (end + host.pull_latency_h + s["pay"] * 1000.0 / host.net_up_mb_s / 3600.0, seq, j["scroll"]))
                gb_out += s["pay"]
        while fetching and fetching[0][0] <= t + 1e-12:
            td, _s, sc = heapq.heappop(fetching)
            if sc != "#wake":
                S[sc]["state"], S[sc]["ready"] = "ready", td
        while pulls and pulls[0][0] <= t + 1e-12:
            tp, _s, sc = heapq.heappop(pulls)
            s = S[sc]
            s["pulled"] = tp
            used -= s["f"].input_gb + s["pay"]
            disk_log.append((round(tp, 3), round(used, 1)))
        # ---- tail balancing first (so a split scroll is staged stripe by stripe)
        if tail and not tail_done and 0 < n_pending() < n and free:
            allj = [j for sc in order for j in S[sc]["jobs"]]
            avail = [t] * len(free) + [e for e, _s, _g, _j in running]
            newj, nn = tail_split(cfg, avail, allj, shells, q)
            if nn:
                notes += [f"t={t:.2f} h: {x}" for x in nn]
                for sc in order:
                    S[sc]["jobs"] = [j for j in newj if j["scroll"] == sc]
                par = {j["id"]: j for j in allj}
                for j in newj:
                    if "_ready_t" not in j and j.get("parent") in par and "_ready_t" in par[j["parent"]]:
                        j["_ready_t"] = par[j["parent"]]["_ready_t"]
            tail_done = True
        # ---- stager: fetch ahead while the disk and the lookahead allow
        for sc in order:
            s = S[sc]
            if s["state"] != "unstaged":
                continue
            active = sum(1 for x in S.values() if x["state"] == "fetching")
            staged_unstarted = sum(1 for x in S.values() if x["state"] in ("fetching", "ready") and x["first"] is None)
            if active >= host.fetch_parallel or staged_unstarted >= len(free) + host.fetch_ahead:
                break
            need = s["f"].input_gb
            if used + need > host.high_water_gb:
                notes.append(f"t={t:.2f} h: staging {sc} blocked by the disk high-water mark ({used:.0f} + {need:.0f} > {host.high_water_gb:.0f} GB)") if not any(
                    x.startswith(f"t=") and f"staging {sc} blocked" in x for x in notes) else None
                break
            t_done = stage_schedule(host, s["f"], s["jobs"], t)
            ft = t_done - t
            s["state"] = "fetching"
            s["fetch_t0"] = t
            used += need
            peak = max(peak, used)
            disk_log.append((round(t, 3), round(used, 1)))
            gb_in += s["f"].fetch_gb
            fetch_log.append({"scroll": sc, "t0_h": round(t, 3), "t1_h": round(t + ft, 3), "gb": round(s["f"].fetch_gb, 1), "objects": s["f"].las_objects})
            seq += 1
            heapq.heappush(fetching, (t + ft, seq, sc))
            for j_ in s["jobs"]:                                   # wake the loop when each stripe's inputs have landed
                seq += 1
                heapq.heappush(fetching, (j_["_ready_t"], seq, "#wake"))
        # ---- dispatch
        while free:
            rdy = lambda c: [j for j in S[c]["jobs"] if j.get("_ready_t") is not None and j["_ready_t"] <= t + 1e-9]
            cand = [sc for sc in order if S[sc]["state"] in ("fetching", "ready") and rdy(sc)]
            if not cand:
                break
            if tail_done or n_pending() < n:      # tail: largest first
                sc = max(cand, key=lambda c: max(hours(j, q) for j in rdy(c)))
                j = max(rdy(sc), key=lambda x: hours(x, q))
            else:
                sc = cand[0]
                j = sorted(rdy(sc), key=lambda x: (x["z0"], x["z1"]))[0]
            S[sc]["jobs"].remove(j)
            g = free.pop(0)
            seq += 1
            h = hours(j, q)
            heapq.heappush(running, (t + h, seq, g, j))
            S[sc]["running"] += 1
            S[sc]["first"] = S[sc]["first"] if S[sc]["first"] is not None else t
            sched.append({"gpu": g, "id": j["id"], "scroll": sc, "t0_h": round(t, 3), "t1_h": round(t + h, 3), "h": round(h, 2), "z": [j["z0"], j["z1"]]})
        # ---- next event
        nxt = [x[0][0] for x in (running, fetching, pulls) if x]
        if not nxt:
            if n_pending():
                blocked = True
                notes.append(f"t={t:.2f} h: DEADLOCK: {n_pending()} job(s) pending but nothing running/fetching (disk high-water {host.high_water_gb:.0f} GB too small for one scroll?)")
            break
        nt = max(t, min(nxt))
        idle_gpu_h += (nt - t) * len(free) if n_pending() else 0.0
        t = nt
    mk = max([s["t1_h"] for s in sched] or [0.0])
    first_fit = min([x["t0_h"] for x in sched] or [0.0])
    filled = []
    for st_ in sorted({x["t0_h"] for x in sched}):
        filled.append((st_, sum(1 for x in sched if x["t0_h"] <= st_ < x["t1_h"])))
    end_all = max([mk] + [s["pulled"] or 0 for s in S.values()])
    return {"q": q, "first_fit_h": first_fit, "gpus_filled": filled, "makespan_h": mk, "all_pulled_h": end_all, "sched": sched, "fetch": fetch_log, "arrivals": arrivals, "disk_peak_gb": round(peak, 1), "disk_log": disk_log,
            "gb_in": gb_in, "gb_out": gb_out, "idle_gpu_h_before_tail": round(idle_gpu_h, 2), "notes": notes, "blocked": blocked, "n_gpus_used": n,
            "gpu_h": sum(s["h"] for s in sched), "n_jobs": len(sched)}


def money(rates, sim: dict) -> dict:
    t = sim["all_pulled_h"] if sim["all_pulled_h"] else sim["makespan_h"]
    run = max(sim["makespan_h"], 0) * rates.eff_hour_usd
    pull_tail = (t - sim["makespan_h"]) * rates.eff_hour_usd
    ing = sim["gb_in"] * rates.ingress_per_tb / 1000.0
    egr = sim["gb_out"] * rates.egress_per_tb / 1000.0
    return {"machine_disk_usd": run + pull_tail, "ingress_usd": ing, "egress_usd": egr, "total_usd": run + pull_tail + ing + egr, "billed_h": t}


# ------------------------------------------------------------------------------------------------ the plan (admission + deferral)
def priority_key(f: Facts, cfg: dict):
    """Higher = keep.  Both prizes first, then cheaper (more scrolls per $)."""
    return (len(f.roles), -L.fit_hours(f.name, f.shell, f.z1 - f.z0, "p50"))


def startup_stripes(host: Host, ordered: list[Facts], min_h: int = 2800, thr_h: float = 0.15) -> tuple[list[Facts], list[str]]:
    """DATA-BOUND START: when staging a whole scroll takes > thr_h, a one-job-per-scroll plan leaves the GPUs idle until a whole scroll has landed (object-bound fetch: ~1.2 h seen on
    the PRO 5000 box with 0 MiB used on all four cards).  The first scroll(s) are therefore planned as k z-stripes (k = min(GPUs, floor((span+overlap)/min_h))) so the first stripe's
    fit starts after tracks + 1/k of the lasagna while the other stripes and the next scroll keep downloading.  Cost: ~1.1-1.35x the GPU-h of those scrolls (stripe overlap + startup)."""
    import dataclasses
    out, notes, gpus_left = [], [], host.n
    for f in ordered:
        span = f.z1 - f.z0
        whole = f.height >= span
        if gpus_left > 0 and whole and fetch_time(host, f) > thr_h and host.n >= 2:
            k = min(max(1, gpus_left), max(1, (span + OVERLAP) // max(min_h, GRID)))
            if k >= 2:
                h = -(-int((span + (k - 1) * OVERLAP) / k) // GRID) * GRID
                if h >= min_h:
                    g = dataclasses.replace(f, height=h, startup_split=True, height_why=f"startup stripes: {k} x ~{h} (was whole scroll; staging takes {fetch_time(host, f):.2f} h)")
                    first = f.tracks_gb * 1000.0 / (host.net_down_mb_s / max(1, host.fetch_parallel)) / 3600.0 + max(
                        LAS_OBJ_PER_SLICE * (h + 512) / host.fetch_files_per_s / 3600.0, LAS_GB_PER_SLICE * (h + 512) * 1000.0 / (host.net_down_mb_s / max(1, host.fetch_parallel)) / 3600.0)
                    extra = k * L.fit_hours(f.name, f.shell, h) - L.fit_hours(f.name, f.shell, span)
                    notes.append(f"startup stripes: {f.name} -> {k} stripes of ~{h} slices: first fit after ~{first:.2f} h instead of ~{fetch_time(host, f):.2f} h "
                                 f"(whole-scroll staging), +{extra:.1f} GPU-h p50 for the overlap/startup of the split")
                    out.append(g)
                    gpus_left -= k
                    continue
        out.append(f)
        gpus_left -= 1
    return out, notes


def make_plan(host: Host, rates, scrolls: list[str], cfg: dict, z0: int = UP_LO, z1: int = UP_HI, plan_frac: float = 0.8, max_height: int = L.FULL_SPAN,
              priority: list[str] | None = None, heights_path=None, order: str = "spt", startup: bool = False, startup_min_height: int = 2800, startup_thr_h: float = 0.15) -> dict:
    cfg.setdefault("dynamic", True)
    allf = [scroll_facts(s, cfg, host, z0, z1, max_height, heights_path) for s in scrolls]
    if priority:
        rank = {s: i for i, s in enumerate(priority)}
        prio = sorted(allf, key=lambda f: rank.get(f.name, len(rank)))           # first = most important
    else:
        prio = sorted(allf, key=lambda f: priority_key(f, cfg), reverse=True)
    keep, deferred = list(prio), []
    limit_usd = plan_frac * rates.soft_usd
    limit_h = plan_frac * rates.max_run_hours if rates.max_run_hours else float("inf")
    res = None
    while keep:
        ordered = sorted(keep, key=lambda f: sum(j["expected_h"] for j in make_jobs(cfg, f)), reverse=(order == "lpt"))
        snotes = []
        if startup:
            ordered, snotes = startup_stripes(host, ordered, startup_min_height, startup_thr_h)
        s50 = simulate(host, ordered, cfg, "p50")
        s90 = simulate(host, ordered, cfg, "p90")
        m50, m90 = money(rates, s50), money(rates, s90)
        ok = (m90["total_usd"] <= limit_usd and m90["billed_h"] <= limit_h and not s90["blocked"] and not s50["blocked"])
        res = dict(keep=[f.name for f in ordered], facts={f.name: f for f in ordered}, p50=s50, p90=s90, m50=m50, m90=m90, fits=ok, startup_notes=snotes)
        if ok:
            break
        victim = keep.pop()                    # lowest priority
        deferred.append((victim.name, f"p90 plan with it needs ${m90['total_usd']:.2f} / {m90['billed_h']:.1f} h vs limit ${limit_usd:.2f} / "
                                      f"{limit_h:.1f} h ({plan_frac:.0%} of soft ${rates.soft_usd:g} / {rates.max_run_hours:g} h max run)"))
    if res is None:
        return {"keep": [], "deferred": deferred, "fits": False, "facts": {}, "all_facts": {f.name: f for f in allf}, "plan_frac": plan_frac, "limit_usd": limit_usd, "limit_h": limit_h}
    import dataclasses
    sens = {}
    for lab, h2, c2 in (("fetch 3x faster", dataclasses.replace(host, fetch_files_per_s=host.fetch_files_per_s * 3), cfg),):
        sim = simulate(h2, [res["facts"][n] for n in res["keep"]], c2, "p50")
        sens[lab] = money(rates, sim)["total_usd"], sim["makespan_h"]
    res["sensitivity"] = sens
    res["cfg"] = cfg
    res.update(priority=[f.name for f in prio], deferred=deferred, all_facts={f.name: f for f in allf}, plan_frac=plan_frac, limit_usd=limit_usd, limit_h=limit_h)
    return res


# ------------------------------------------------------------------------------------------------ re-plan on a capacity change
def replan_eval(cfg: dict, avail: list, extra_h: float, jobs: list, shells: dict, q: str) -> float:
    """Simulated hours (from now) until every given unstarted job is done on GPUs free at `avail`, tail-split included; running jobs on GPUs that are no longer allowed
    (`extra_h`) still bill until they finish."""
    if not jobs:
        return extra_h
    if not avail:
        return float("inf")
    js = list(jobs)
    if len(js) < len(avail):
        js, _ = tail_split(cfg, avail, js, shells, q)
    if any(j.get("_ready_h") for j in js):
        return max(release_makespan(avail, [(j.get("_ready_h", 0.0), hours(j, q)) for j in js]), extra_h)
    return max(lpt_makespan(avail, [hours(j, q) for j in js]), extra_h)


def replan(cfg: dict, avail: list, extra_h: float, groups: dict, priority: list, shells: dict, spent_usd: float, now_h: float, rates, pending_payload_gb: float = 0.0,
           plan_frac: float = 0.8) -> dict:
    """Which unstarted scrolls still fit after a GPU-count change (shrink OR regrow).  `groups` = {scroll: {"jobs": [unstarted jobs], "committed": bool}}; committed scrolls
    (something already ran or was retried) always stay.  Uncommitted scrolls are admitted greedily in priority order while the p90 case keeps
    spent + makespan x effective $/h + payload egress <= plan_frac x soft AND now + makespan <= plan_frac x max run hours; the first that does not fit and everything after it is
    DEFERRED with the numbers.  Pure; the same function re-admits earlier deferrals when GPUs come back."""
    limit_usd = plan_frac * rates.soft_usd
    limit_h = plan_frac * rates.max_run_hours if rates.max_run_hours else float("inf")
    base = [j for g in groups.values() if g["committed"] for j in g["jobs"]]
    order = [s for s in priority if s in groups and not groups[s]["committed"]] + [s for s in groups if not groups[s]["committed"] and s not in priority]

    def cost(jobs, q):
        mk = replan_eval(cfg, avail, extra_h, jobs, shells, q)
        pay = pending_payload_gb + sum(j["payload_gb"] for j in jobs)
        return mk, spent_usd + mk * rates.eff_hour_usd + pay * rates.egress_per_tb / 1000.0

    keep, deferred, jobs = [], [], list(base)
    blocked = None
    for sc in order:
        if blocked:
            deferred.append((sc, blocked))
            continue
        trial = jobs + groups[sc]["jobs"]
        mk, usd = cost(trial, "p90")
        if usd <= limit_usd and now_h + mk <= limit_h and mk != float("inf"):
            keep.append(sc)
            jobs = trial
        else:
            blocked = (f"p90 re-plan on {len(avail)} allowed GPU(s): with it ${usd:.2f} / +{mk:.1f} h (now {now_h:.1f} h) vs limit ${limit_usd:.2f} / {limit_h:.1f} h "
                       f"({plan_frac:.0%} of soft ${rates.soft_usd:g} / {rates.max_run_hours:g} h)")
            deferred.append((sc, blocked))
    mk50, usd50 = cost(jobs, "p50")
    mk90, usd90 = cost(jobs, "p90")
    return {"keep": keep, "deferred": deferred, "committed": [s for s, g in groups.items() if g["committed"]], "mk50": mk50, "mk90": mk90, "usd50": usd50, "usd90": usd90,
            "limit_usd": limit_usd, "limit_h": limit_h, "n_allowed": len(avail)}


# ------------------------------------------------------------------------------------------------ printing
def gantt(sim: dict, n: int, width: int = 72) -> list[str]:
    mk = max(sim["makespan_h"], 1e-9)
    lines = [f"  GPU x time (p{'90' if sim['q'] == 'p90' else '50'} case; one column = {mk / width:.2f} h; letters = scrolls, '.' = idle)"]
    letters = {}
    for s in sim["sched"]:
        letters.setdefault(s["scroll"], chr(ord("A") + len(letters) % 26))
    for g in range(n):
        row = ["."] * width
        for s in sim["sched"]:
            if s["gpu"] != g:
                continue
            a, b = int(s["t0_h"] / mk * width), max(int(s["t1_h"] / mk * width), int(s["t0_h"] / mk * width) + 1)
            for c in range(a, min(b, width)):
                row[c] = letters[s["scroll"]]
        lines.append(f"  gpu{g} |{''.join(row)}|")
    lines.append("  key: " + ", ".join(f"{v}={k}" for k, v in letters.items()) + f"   (0 h ... {mk:.1f} h)")
    return lines


def render(host: Host, rates, plan: dict, runnable_note: list[str] | None = None, routea: str | None = None) -> str:
    out = []
    P = out.append
    P(f"PLAN host: {host.n} GPU(s) VRAM {sorted({round(g.vram_gib, 1) for g in host.gpus})} GiB (speed x{host.gpus[0].speed:g}), {host.phys_cores} physical cores, "
      f"RAM {host.ram_gb:g} GB (<= {host.max_concurrent_fits()} concurrent fits at {host.ram_per_fit_gb:g} GB each), disk {host.disk_free_gb:.0f}/{host.disk_total_gb:.0f} GB free "
      f"(high-water {host.high_water_gb:.0f} GB), net {host.net_down_mb_s:.0f}/{host.net_up_mb_s:.0f} MB/s down/up, fetch {host.fetch_files_per_s:g} files/s x {host.fetch_parallel} parallel")
    for l in getattr(host, "disk_lines", []):
        P(f"PLAN disk: {l}")
    P(f"PLAN disk base {host.used0_gb:.0f} GB of {host.disk_total_gb:.0f} GB (clamped to the volume; high-water {host.high_water_gb:.0f} GB)")
    P(f"PLAN budget: {rates.describe()}")
    for n_, why in runnable_note or []:
        P(f"  NOT RUNNABLE {n_}: {why}")
    if not plan["keep"]:
        P("PLAN: NOTHING FITS the budget/time (even one scroll); see DEFERRED lines")
    P(f"PLAN limit: p90 case must finish within {plan['plan_frac']:.0%} of soft budget = ${plan['limit_usd']:.2f} and {plan['limit_h']:.1f} h (max run hours x {plan['plan_frac']:g})")
    for n_, why in plan.get("deferred", []):
        P(f"  DEFERRED {n_}: {why}")
    if not plan["keep"]:
        return "\n".join(out)
    P("PLAN scrolls (dispatch order = smallest first; hours = GPU-h per fit on the V100-class basis / speed):")
    P(f"  {'scroll':<10} {'shell':>5} {'trk GB':>6} {'disk GB':>7} {'height':>6} {'jobs':>4} {'p50 GPU-h':>9} {'p90 GPU-h':>9}  height rule")
    for nme in plan["keep"]:
        f = plan["facts"][nme]
        s50 = [x for x in plan["p50"]["sched"] if x["scroll"] == nme]
        s90 = [x for x in plan["p90"]["sched"] if x["scroll"] == nme]
        P(f"  {nme:<10} {f.shell:>5} {f.tracks_gb:>6.1f} {f.input_gb:>7.0f} {f.height:>6} {len(s50):>4} {sum(x['h'] for x in s50):>9.1f} {sum(x['h'] for x in s90):>9.1f}  {f.height_why}")
    for n_ in plan.get("startup_notes", []):
        P("PLAN " + n_)
    ff = plan["p50"]
    P(f"PLAN time to first fit {ff['first_fit_h']:.2f} h ({'per-stripe staging: tracks + stripe 1 lasagna first' if getattr(host, 'stripe_staging', True) else 'whole-scroll staging'}); "
      "GPUs busy over time (p50): " + ", ".join(f"{n_} @ {t_:.2f} h" for t_, n_ in ff["gpus_filled"][:12]))
    for key, lab in (("p50", "p50"), ("p90", "p90")):
        sim, mon = plan[key], plan["m" + key[1:]]
        P(f"PLAN {lab}: makespan {sim['makespan_h']:.2f} h (all pulled {sim['all_pulled_h']:.2f} h), {sim['n_jobs']} job(s), {sim['gpu_h']:.1f} GPU-h on {sim['n_gpus_used']} GPU(s), "
          f"idle GPU-h while work waited on fetch/disk {sim['idle_gpu_h_before_tail']}; disk peak {sim['disk_peak_gb']} GB; in {sim['gb_in']:.0f} GB, payload out {sim['gb_out']:.1f} GB")
        P(f"PLAN {lab} $: machine+disk {mon['machine_disk_usd']:.2f} + ingress {mon['ingress_usd']:.2f} + egress {mon['egress_usd']:.2f} = ${mon['total_usd']:.2f} "
          f"(soft ${rates.soft_usd:g}, hard ${rates.hard_usd:g}; budget fits: {'YES' if mon['total_usd'] <= plan['limit_usd'] else 'NO'})")
    for lab, (usd, mk) in plan.get("sensitivity", {}).items():
        P(f"PLAN sensitivity (p50, same scrolls): {lab}: makespan {mk:.2f} h, ${usd:.2f}   [the lasagna fetch rate {host.fetch_files_per_s:g} files/s is the least certain input: measured on a loaded pny, not on this box]")
    for nline in plan["p50"]["notes"]:
        P("PLAN note: " + nline)
    P("PLAN stagger timeline (p50): fetch")
    for fl in plan["p50"]["fetch"]:
        P(f"  fetch {fl['scroll']:<10} {fl['t0_h']:>6.2f} -> {fl['t1_h']:>6.2f} h  {fl['gb']:>5.1f} GB, {fl['objects']:,} objects")
    P("PLAN stagger timeline (p50): jobs")
    for s in sorted(plan["p50"]["sched"], key=lambda x: (x["gpu"], x["t0_h"])):
        P(f"  gpu{s['gpu']} {s['id']:<26} z[{s['z'][0]},{s['z'][1]}) {s['t0_h']:>6.2f} -> {s['t1_h']:>6.2f} h ({s['h']:.1f} GPU-h)")
    out += gantt(plan["p50"], plan["p50"]["n_gpus_used"])
    P("PLAN expected payload arrivals in out/ (p50; a unit is downloadable the moment its DONE marker is written):")
    for a in sorted(plan["p50"]["arrivals"], key=lambda x: x["t_h"]):
        P(f"  t={a['t_h']:>6.2f} h  out/{a['id']}/  {a['gb']:.2f} GB")
    P("PLAN disk timeline (p50, GB used incl. env; <t h: GB>): " + " ".join(f"{t:.1f}h:{g:.0f}" for t, g in plan["p50"]["disk_log"][:40]) + f"  (peak {plan['p50']['disk_peak_gb']}, high-water {host.high_water_gb:.0f})")
    out += vram_lines(host)
    out += download_report(host, rates, plan)
    if routea:
        P(routea)
    return "\n".join(out)
