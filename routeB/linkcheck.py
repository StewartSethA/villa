"""Link check: the FIRST thing a run does, before anything that costs money-time (apt, uv, pip, compile, fetch).

Measures aggregate ingress with parallel HTTP range reads from TWO independent hosts and prints a plain verdict box:
  data host    dl.ash2txt.org  (a real tracks file: the bytes the run will actually fetch)
  second host  speed.cloudflare.com  (a CDN: tells 'the box link is slow' from 'the data source is slow')
Verdict: MB/s measured, hours to fetch the plan's GB (planner per-scroll sizes), rental $ spent waiting for data, GOOD / MARGINAL / BAD.
Gate: data-host rate below --min-link-mb-s (default 20) STOPS the run (exit 3, nothing built) with 'DESTROY THIS BOX or re-run with
--accept-slow-link'; with the flag the run continues and the planner auto-shrinks --gpus / --fetch-parallel (written to <home>/box8/link/first.json).
A probe that cannot run (no network to either host) never blocks: it is announced and the quoted rate stays.
Entry points: `python3 -m routeB.linkcheck [box8 flags]` (called by routeB_run.sh --mode box8 before the python env is built; stdlib only) and
the functions below (box8 re-probes every --link-reprobe-s during the run).
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SECOND_URL = "https://speed.cloudflare.com/__down?bytes=16000000"
DATA_TIME_S = 8.0
SECOND_TIME_S = 5.0
FRESH_S = 1800.0


def measure_link(url: str, n_conn: int = 8, chunk: int = 8 << 20, timeout: float = 60.0, budget_s: float | None = None) -> tuple[float | None, str]:
    """Aggregate ingress MB/s: n_conn parallel connections; each reads its own `chunk`-byte range (no Range header for the CDN's ?bytes= endpoint).
    With budget_s every connection stops reading at that deadline, so a slow link costs at most budget_s seconds."""
    use_range = "__down" not in url
    t0 = time.time()
    deadline = t0 + budget_s if budget_s else None

    def one(i):
        hdr = {"User-Agent": "Mozilla/5.0 routeB-linkcheck"}
        if use_range:
            hdr["Range"] = f"bytes={i * chunk}-{(i + 1) * chunk - 1}"
        n = 0
        with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=timeout) as r:
            while True:
                if deadline and time.time() >= deadline:
                    break
                b = r.read(1 << 18)
                if not b:
                    break
                n += len(b)
        return n

    try:
        with ThreadPoolExecutor(n_conn) as ex:
            got = sum(ex.map(one, range(n_conn)))
    except Exception as e:                      # noqa: BLE001 - announced by the caller; the quoted rate stays in force
        return None, f"{type(e).__name__}: {e}"
    dt = max(time.time() - t0, 1e-6)
    return got / 1e6 / dt, f"{got / 1e6:.0f} MB in {dt:.1f} s over {n_conn} connections"


def data_url(scroll: str) -> str:
    from .common import spec
    sp = spec(scroll)
    name = next(k for k in sp["tracks"]["files"] if k.endswith(".dbm"))
    return sp["tracks"]["base_url"] + name


def measure_hosts(scroll: str, fn=None) -> dict:
    """Both probes, sequential (they would split the link if concurrent).  fn(url, budget_s) -> (MB/s | None, why); default measure_link."""
    fn = fn or (lambda url, budget_s: measure_link(url, budget_s=budget_s))
    out = {"t": time.time()}
    for key, url, b in (("data", None, DATA_TIME_S), ("second", SECOND_URL, SECOND_TIME_S)):
        try:
            u = url or data_url(scroll)
        except Exception as e:                  # noqa: BLE001
            out[key] = {"mbs": None, "why": f"no url: {e}", "url": None}
            continue
        mbs, why = fn(u, b)
        out[key] = {"mbs": mbs, "why": why, "url": u}
    return out


def diagnose(m: dict, min_mbs: float) -> str:
    d, s = m["data"]["mbs"], m["second"]["mbs"]
    if d is None and s is None:
        return "no probe could run (no route to either host): link unknown"
    if d is None:
        return "data host unreachable but the CDN answered: SOURCE (dl.ash2txt.org) problem, box link works"
    if s is None:
        return "CDN unreachable, data host answered: judging on the data host alone"
    if d < min_mbs and s >= 3 * d:
        return "SOURCE slow: the CDN is >= 3x faster than the data host, so the box link is fine"
    if d < min_mbs and s < min_mbs:
        return "BOX LINK slow: both independent hosts are slow"
    if d < min_mbs:
        return "data host slow, CDN not much faster: link or source, re-probe later"
    return "link and source both look fine"


def verdict(m: dict, plan_gb: float, compute_wall_h: float, eff_hour_usd: float, min_mbs: float, n_gpus: int = 8) -> dict:
    d = m["data"]["mbs"]
    if d is None:
        return {"verdict": "UNKNOWN", "hours": None, "usd": None}
    hours = plan_gb * 1000.0 / d / 3600.0 if d > 0 else float("inf")
    usd = hours * eff_hour_usd                     # the box bill while every GPU waits for the transfer (upper bound; the planner prices the real overlap)
    if d < min_mbs:
        v = "BAD"
    elif hours > 0.25 * compute_wall_h or d < 2 * min_mbs:
        v = "MARGINAL"
    else:
        v = "GOOD"
    return {"verdict": v, "hours": hours, "usd": usd}


def box(lines: list[str], width: int = 92) -> str:
    bar = "+" + "-" * (width + 2) + "+"
    return "\n".join([bar] + [f"| {l[:width]:<{width}} |" for l in lines] + [bar])


def render(m: dict, v: dict, plan_gb: float, compute_wall_h: float, min_mbs: float, eff: float, note: str = "", shrink: dict | None = None) -> str:
    f = lambda x: "n/a" if x is None else f"{x:.1f} MB/s"
    lines = [f"LINK CHECK   VERDICT: {v['verdict']}   (gate: data host >= {min_mbs:g} MB/s)",
             f"data host   dl.ash2txt.org : {f(m['data']['mbs'])}   [{m['data']['why'][:44]}]",
             f"2nd host    speed.cloudflare: {f(m['second']['mbs'])}   [{m['second']['why'][:44]}]",
             f"diagnosis: {diagnose(m, min_mbs)}"]
    if v["hours"] is not None:
        lines += [f"the plan fetches {plan_gb:.0f} GB -> {v['hours']:.2f} h of pure transfer at {m['data']['mbs']:.1f} MB/s",
                  f"compute wall of the plan ~{compute_wall_h:.1f} h",
                  f"rental $ spent waiting for data if every GPU waits: ${v['usd']:.2f} at ${eff:.3f}/h"]
    if shrink:
        lines.append(f"AUTO-SHRINK (accepted slow link): --gpus {shrink['gpus']} --fetch-parallel {shrink['fetch_parallel']}")
    if v["verdict"] == "BAD":
        lines += ["", "DESTROY THIS BOX, or re-run with --accept-slow-link to continue anyway."]
    if note:
        lines.append(note)
    return box(lines)


def shrink_for(m: dict, plan_gb: float, compute_gpu_h: float, n_gpus: int, fetch_parallel: int, frac: float = 0.5) -> dict:
    """GPUs worth keeping when staging is the bottleneck: transfer h <= frac x compute wall.  Each concurrent fetch should get >= 10 MB/s."""
    d = m["data"]["mbs"] or 1.0
    agg = plan_gb * 1000.0 / d / 3600.0
    k = max(1, min(n_gpus, int(frac * compute_gpu_h / agg))) if agg > 0 else n_gpus
    fp = max(1, min(fetch_parallel, int(d / 10.0)))
    return {"gpus": k, "fetch_parallel": fp}


def first_path(home: Path) -> Path:
    return Path(home) / "box8" / "link" / "first.json"


def load_first(home: Path, max_age_s: float = FRESH_S) -> dict | None:
    try:
        d = json.loads(first_path(home).read_text())
    except (OSError, ValueError):
        return None
    return d if time.time() - d.get("measured", {}).get("t", 0) < max_age_s else None


def trend_append(home: Path, mbs: float | None, second: float | None = None) -> None:
    p = Path(home) / "box8" / "link" / "trend.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps({"t": time.time(), "data_mbs": mbs, "second_mbs": second}) + "\n")


def trend_read(home: Path, n: int = 12) -> list[dict]:
    try:
        rows = [json.loads(l) for l in (Path(home) / "box8" / "link" / "trend.jsonl").read_text().splitlines() if l.strip()]
    except OSError:
        return []
    return rows[-n:]


def main(argv=None) -> int:
    """python3 -m routeB.linkcheck <box8 flags>: measure, print the verdict box, gate.  Exit 0 = go (GOOD/MARGINAL/UNKNOWN/accepted), 3 = BAD and not accepted."""
    from . import box8 as B8
    from . import planner as PL
    from . import ladder as L
    from .common import home as _home
    a, _rest = B8.build_parser().parse_known_args(argv)
    a.force_gpus = True                      # sizing only: a foreign process on a card must not stop the link check
    H = _home()
    cfg = L.load_config(a.ladder_config, a.ladder)
    cfg["dynamic"] = True
    B = B8._budget()
    rates = B.Rates.from_env(**({k: v for k, v in dict(hour_usd=a.hour_usd, disk_gb=a.disk_gb, disk_usd_per_16gb_hour=a.disk_usd_per_16gb_hour).items() if v is not None}))
    names = B8.all_scroll_names() if a.scrolls in ("auto", "all") else [s.strip() for s in a.scrolls.split(",") if s.strip()]
    run = [s for s in names if not B8.skip_reason(s)]
    if not run:
        print("linkcheck: no runnable scroll; nothing to measure")
        return 0
    if a.link_mb_s:
        m = {"t": time.time(), "data": {"mbs": a.link_mb_s, "why": "--link-mb-s", "url": None}, "second": {"mbs": None, "why": "not probed", "url": None}}
    else:
        m = measure_hosts(run[0], B8.link_fn())
    gpus, allowed = B8.gpu_info(a)
    host = B8.build_host(a, H, gpus, allowed)
    if m["data"]["mbs"]:
        host.net_down_mb_s, host.link_measured = m["data"]["mbs"], True
    plan = PL.make_plan(host, rates, run, cfg, plan_frac=100.0)            # sizes and compute wall only: no deferral pressure here
    keep = plan["keep"] or run
    facts = [plan["facts"][n] for n in plan["keep"]] if plan["keep"] else [PL.scroll_facts(s, cfg, host) for s in run]
    plan_gb = sum(f.fetch_gb for f in facts)
    gpu_h = plan["p50"]["gpu_h"] if plan["keep"] else sum(L.fit_hours(f.name, f.shell, f.z1 - f.z0) for f in facts)
    n = max(1, len(host.gpus))
    wall = gpu_h / n
    min_mbs = a.min_link_mb_s
    v = verdict(m, plan_gb, wall, rates.eff_hour_usd, min_mbs, n)
    shrink = None
    if v["verdict"] in ("BAD", "MARGINAL") and v["hours"] is not None and v["hours"] > 0.5 * wall:
        shrink = shrink_for(m, plan_gb, gpu_h, n, a.fetch_parallel)
    accepted = v["verdict"] == "BAD" and a.accept_slow_link
    print(render(m, v, plan_gb, wall, min_mbs, rates.eff_hour_usd, "ACCEPTED (--accept-slow-link): continuing" if accepted else "", shrink if (accepted or v["verdict"] == "MARGINAL") else None), flush=True)
    d = first_path(H)
    d.parent.mkdir(parents=True, exist_ok=True)
    d.write_text(json.dumps({"measured": m, "verdict": v, "plan_gb": plan_gb, "scrolls": keep, "shrink": shrink, "accepted": accepted, "min_link_mb_s": min_mbs}, indent=1))
    trend_append(H, m["data"]["mbs"], m["second"]["mbs"])
    if v["verdict"] == "BAD" and not a.accept_slow_link:
        print("linkcheck: STOPPING, nothing was built. DESTROY THIS BOX or re-run with --accept-slow-link.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
