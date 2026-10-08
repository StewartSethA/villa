"""RESUME / GROW GATE (user 2026-10-07, after PHerc0211_ca9207a): a segment whose newest surface fails the degeneracy gate is NOT resumed or grown further. Fail-closed, visible
as the `self_crossing` flag + a `pause_reason` metric + a grow hold; routed to the prune (mode D, maintenance.selfx_population) and released again once its newest surface passes.

check(path, pol)   -> {'pass': bool, 'reasons': {...}, 'fractions': {...}}   transverse crossings (vc_tifxyz_selfcross, ZERO tolerated) + degeneracy fractions
sweep(fleet, me, db, apply)  per-host pass (invoked by supervise): holds GROWING segments that fail (graceful: the round in flight finishes, no next round), releases held
                             segments whose newest surface now passes. Idempotent, dry-run unless apply, per-item errors counted (D28).

Thresholds are NAMED TUNABLES (TUNABLES below, printed by disclosure()). Basis: per-segment fraction of cells firing on the in-solve-guarded, transverse-free baseline
(n = 39 segments, 2.29 M cells, 12 scrolls; docs/experiments/selfx_population_2026-10-07/degeneracy): p50/p95 fold_over 0.76/2.0 %, normal_reversal 3.4/7.1 %,
self_proximity 2.1/4.4 %, hairpin 0.10/0.40 %. The gate sits just above that baseline p95 (a fresh guarded round must pass); PHerc0211_ca9207a (fold_over 2.6 %,
normal_reversal 9.7 %, close pairs 4.6 %) fails. NO human reference exists for these thresholds (0 human verdicts in guard_review): they are a proposal, not a validated gate.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

TUNABLES = {
    "GATE_FOLD_FRAC": (0.021, "fraction of cells", "fold-over cells; guarded baseline p95 2.0 % (n=39 segs)"),
    "GATE_NORMAL_REV_FRAC": (0.072, "fraction of cells", "normal-reversal seam cells; guarded baseline p95 7.1 %"),
    "GATE_PROX_FRAC": (0.045, "fraction of cells", "cells with a non-neighbour sample < MIN_SEP_VOX; guarded baseline p95 4.4 %"),
    "GATE_HAIRPIN_FRAC": (0.005, "fraction of cells", "hairpin cells; guarded baseline p95 0.4 %"),
    "GATE_TRANSVERSE_CELLS": (0, "cells", "transverse-crossing flagged cells tolerated: ZERO (user: no self-intersections, period)"),
    "GATE_MIN_CELLS": (400, "cells", "surfaces smaller than this are not gated (too few cells for a fraction)"),
}
HOLD_BY = "selfx:resume_gate"


def tun(name: str, override: dict | None = None) -> float:
    return float((override or {}).get(name, TUNABLES[name][0]))


def disclosure(override: dict | None = None) -> str:
    return "resume_gate: " + "; ".join(f"{k}={tun(k, override):g} {u}" for k, (_, u, _b) in TUNABLES.items()) + " (named tunables; basis in resume_gate.py docstring)"


def check(path: str, pol, override: dict | None = None) -> dict:
    """Gate one checkpoint dir. `pol` = GuardPolicy with selfcross_bin/env resolved. Never raises: an unrunnable check returns {'pass': False, 'ran': False} (fail closed)."""
    from dataclasses import replace
    from . import degeneracy as DG, growth_guard as GG
    pol = replace(pol, selfcross=True)                      # the gate does not follow grow.guard.selfcross (like the finish gate)
    try:
        X, Y, Z = GG._read_xyz(path)
        V = (X > 0) & (Y > 0) & (Z > 0)
        n = int(V.sum())
        out = {"pass": True, "ran": True, "cells": n, "reasons": {}, "fractions": {}}
        if n < tun("GATE_MIN_CELLS", override):
            return out
        cs, ci = GG.selfcross_contacts(path, pol, n)
        if cs is None:
            return {"pass": False, "ran": False, "cells": n, "reasons": {"unrunnable": str(ci.get("error") or ci.get("skipped"))[:160]}, "fractions": {}}
        gen = GG._read_gen(path)
        flagged = 0
        if cs:
            if gen is not None and gen.shape == V.shape:
                _s, F, _g = GG.contact_seeds(V, gen.astype(np.int64), cs)
                flagged = int(F.sum())
            else:
                flagged = len(cs)
        out["fractions"]["transverse_cells"] = flagged
        if flagged > tun("GATE_TRANSVERSE_CELLS", override):
            out["reasons"]["transverse"] = flagged
        R = DG.run_checks(X, Y, Z, V)
        for key, name, th in (("fold_over", "fold_over", "GATE_FOLD_FRAC"), ("normal_reversal", "normal_reversal", "GATE_NORMAL_REV_FRAC"),
                              ("self_proximity", "self_proximity", "GATE_PROX_FRAC"), ("hairpin", "hairpin", "GATE_HAIRPIN_FRAC")):
            fr = R[key]["n_cells"] / n
            out["fractions"][name] = round(fr, 5)
            if fr > tun(th, override):
                out["reasons"][name] = round(fr, 5)
        out["pass"] = not out["reasons"]
        return out
    except Exception as e:                                  # noqa: BLE001 - fail closed
        return {"pass": False, "ran": False, "reasons": {"error": f"{type(e).__name__}: {str(e)[:160]}"}, "fractions": {}}


def hold(db, seg: str, result: dict, note_prefix: str = "resume gate") -> None:
    """Fail-closed hold: self_crossing flag (a RECORD), pause_reason metric, graceful grow hold (the round in flight finishes)."""
    from . import flags as FL, grow_control as GC
    from .db import pipeline_db
    why = ", ".join(f"{k}={v}" for k, v in (result.get("reasons") or {}).items())
    P = pipeline_db()
    P.record_metric(db, seg, "pause_reason", text=f"{note_prefix}: {why}"[:240], stage="grow")
    P.record_metric(db, seg, "resume_gate", 0.0, text=json.dumps(result)[:480], stage="grow")
    FL.set_flag(db, seg, "self_crossing", True, reason=f"{note_prefix}: {why}"[:300], by=HOLD_BY)
    GC.request_stop(db, seg, "graceful", by=HOLD_BY, note=f"{note_prefix}: {why}")


def release(db, seg: str) -> None:
    """The newest surface now passes: clear the hold + the flag this module set. The segment becomes a normal resume candidate again."""
    from . import flags as FL
    from .db import pipeline_db
    P = pipeline_db()
    P.set_paused(db, seg, "grow", False, by="resume_gate: released")
    FL.set_flag(db, seg, "self_crossing", False, reason="resume gate: newest surface passes", by=HOLD_BY)
    P.record_metric(db, seg, "resume_gate", 1.0, text="released", stage="grow")


def sweep(fleet, me, db, apply: bool = True, limit: int | None = None, time_budget_s: float = 1200.0, log=print) -> dict:
    """Per-host pass. (1) GROWING segments (a live assignment on THIS host, newest checkpoint local) that fail -> graceful hold; (2) segments held by this module whose newest
    local surface passes -> released. Segments whose newest artifact is not on this host are another host's business."""
    from . import growth_guard as GG
    from .stages import grow as G
    from .db import pipeline_db
    st = {"host": me.name, "apply": apply, "live_checked": 0, "live_failed": 0, "held": 0, "held_checked": 0, "released": 0, "unrunnable": 0, "errors": 0}
    t0 = time.time()
    if getattr(me, "vc_bin", ""):
        os.environ.setdefault("VC_BIN", me.vc_bin)
    from dataclasses import replace
    pol = G.resolve_selfcross_bin(fleet, replace(GG.policy_from_db(db), selfcross=True, selfcross_threads=2))
    rows = db.execute("SELECT DISTINCT a.seg FROM assignment a WHERE a.stage='grow' AND a.state IN ('running','queued') AND a.host=?", (me.name,)).fetchall()   # queued ones are held too (request_stop withdraws them)
    live = [r[0] for r in rows]
    held = [r[0] for r in db.execute("SELECT seg FROM stage_flag WHERE stage='grow' AND paused=1 AND paused_by LIKE ?", (f"%{HOLD_BY}%",)).fetchall()]

    def newest(seg):
        r = db.execute("SELECT path FROM artifact WHERE seg=? AND kind='tifxyz' AND path NOT LIKE '/dev/shm%' ORDER BY mtime DESC, id DESC LIMIT 1", (seg,)).fetchone()
        return r[0] if r and os.path.exists(os.path.join(r[0], "x.tif")) else None
    for seg in live:
        if time.time() - t0 > time_budget_s or (limit is not None and st["live_checked"] >= limit):
            break
        p = newest(seg)
        if not p:
            continue
        st["live_checked"] += 1
        try:
            res = check(p, pol)
            if not res.get("ran", True):
                st["unrunnable"] += 1
                continue
            if not res["pass"]:
                st["live_failed"] += 1
                if apply:
                    hold(db, seg, res, "gate sweep (growing)")
                    db.commit()
                    st["held"] += 1
                log(f"resume_gate {seg}: FAIL {res['reasons']}")
        except Exception as e:                              # noqa: BLE001 (D28)
            st["errors"] += 1
            log(f"resume_gate {seg}: {type(e).__name__}: {str(e)[:120]}")
    for seg in held:
        if time.time() - t0 > time_budget_s:
            break
        p = newest(seg)
        if not p:
            continue
        st["held_checked"] += 1
        try:
            res = check(p, pol)
            if res.get("ran", True) and res["pass"]:
                if apply:
                    release(db, seg)
                    db.commit()
                st["released"] += 1
        except Exception as e:                              # noqa: BLE001
            st["errors"] += 1
            log(f"resume_gate release {seg}: {type(e).__name__}: {str(e)[:120]}")
    log(f"resume_gate sweep {me.name}: {st} | " + disclosure())
    return st


if __name__ == "__main__":
    import argparse
    from . import config
    from .remotedb import connect_for, hub_token
    ap = argparse.ArgumentParser(prog="python -m vesuvius_pipeline.resume_gate")
    ap.add_argument("cmd", choices=["sweep"]); ap.add_argument("--apply", action="store_true"); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--budget-s", type=float, default=1200.0)
    a = ap.parse_args()
    fl = config.load()
    print(json.dumps(sweep(fl, config.this_host(fl), connect_for(fl, hub_token(fl)), apply=a.apply, limit=a.limit, time_budget_s=a.budget_s)))
