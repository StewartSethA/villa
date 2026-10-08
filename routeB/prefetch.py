"""Early prefetch: start downloading the FIRST scroll's data while the python env is still being built, so nothing waits.

Started in the background by routeB_run.sh (--mode box8) right after the link check, with the system python3 (stdlib only: manifest + deploy_common/fetch_assets).  It follows the
planner's own start order: the first jobs of the p50 simulation (smallest scroll first; tail-split stripes in z order), staged per stripe -- the tracks file once, then each stripe's
lasagna z-chunks -- so stripe 1 is complete as early as possible.  routeB_run.sh stops it (SIGTERM) when the env is ready; the downloads are resumable and the scheduler then continues
exactly where it stopped (verified assets are skipped).  What it downloaded is written to <home>/box8/prefetch.json so the budget governor can account the ingress.
  python3 -m routeB.prefetch <box8 flags>   [--prefetch-scrolls N (default 3)]
"""
from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

STOP = False
STATE = {"fa": None, "out": None, "done": []}


def _term(_s, _f):
    """Hand-over: the env is ready.  Downloads are resumable (.part + block map), so stop NOW instead of finishing the stripe in flight; record the bytes for the budget."""
    global STOP
    STOP = True
    fa, out = STATE["fa"], STATE["out"]
    if fa is not None and out is not None:
        try:
            out.write_text(json.dumps({"gb": fa.NET.bytes / 1e9, "done": STATE["done"], "stopped": True, "accounted": False}, indent=1))
        except OSError:
            pass
    print(f"[{time.strftime('%H:%M:%S')}] prefetch: stopped at hand-over ({(fa.NET.bytes / 1e9) if fa else 0:.2f} GB over the network)", flush=True)
    os._exit(0)


def main(argv=None) -> int:
    from . import box8 as B8
    from . import ladder as L
    from . import manifest as MAN
    from . import planner as PL
    from .common import UP_HI, UP_LO, home, say
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy_common"))
    import fetch_assets as FA
    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    ap = B8.build_parser()
    ap.add_argument("--prefetch-scrolls", type=int, default=3)
    a, _rest = ap.parse_known_args(argv)
    a.force_gpus = True
    H = home()
    cfg = L.load_config(a.ladder_config, a.ladder)
    cfg["dynamic"] = True
    B = B8._budget()
    rates = B.Rates.from_env()
    names = B8.all_scroll_names() if a.scrolls in ("auto", "all") else [s.strip() for s in a.scrolls.split(",") if s.strip()]
    run = [s for s in names if not B8.skip_reason(s)]
    if not run:
        return 0
    gpus, allowed = B8.gpu_info(a)
    host = B8.build_host(a, H, gpus, allowed)
    first = None
    try:
        from . import linkcheck as LK
        first = LK.load_first(H)
    except Exception:                                   # noqa: BLE001
        pass
    if first and first.get("measured", {}).get("data", {}).get("mbs"):
        host.net_down_mb_s = first["measured"]["data"]["mbs"]
    plan = PL.make_plan(host, rates, run, cfg, a.z0 or UP_LO, a.z1 or UP_HI, 100.0, a.max_height)
    if not plan["keep"]:
        return 0
    sched = sorted(plan["p50"]["sched"], key=lambda x: (x["t0_h"], x["z"][0]))
    order, seen = [], []
    for s in sched:
        if s["scroll"] not in seen:
            if len(seen) >= a.prefetch_scrolls:
                continue
            seen.append(s["scroll"])
        if s["scroll"] in seen:
            order.append((s["scroll"], s["z"][0], s["z"][1]))
    total = 0
    out = H / "box8" / "prefetch.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] prefetch: {m}", flush=True)
    log(f"early prefetch of {len(order)} stripe(s) of {seen} while the env builds (stripe 1 first)")
    done = STATE["done"]
    STATE["fa"], STATE["out"] = FA, out
    for sc, z0, z1 in order:
        if STOP:
            break
        m = MAN.build(sc, z0, z1)
        from .fetchlog import Collapser
        from .common import spec
        tr = sum(v for k, v in spec(sc)["tracks"]["files"].items() if k.endswith(".dbm")) / 1e9
        col = Collapser(sc, tr + PL.LAS_GB_PER_SLICE * (z1 - z0 + 512), emit=lambda m_: print(f"[{time.strftime('%H:%M:%S')}] fetch: {m_}", flush=True))
        r = FA.fetch(m, H / "assets", log=col)
        col.final()
        total = max(total, r["bytes_net"])                 # fetch_assets' counter is cumulative for this process
        done.append({"scroll": sc, "z": [z0, z1], "ok": r["ok"]})
        log(f"{sc} z[{z0},{z1}) {'staged' if r['ok'] else 'FAILED ' + '; '.join(r['failed'])[:200]}  (cumulative {r['bytes_net'] / 1e9:.2f} GB over the network)")
        out.write_text(json.dumps({"gb": total / 1e9, "done": done, "stopped": STOP, "accounted": False}, indent=1))
    out.write_text(json.dumps({"gb": total / 1e9, "done": done, "stopped": STOP, "accounted": False}, indent=1))
    log(f"finished: {total / 1e9:.2f} GB, stopped early: {STOP}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
