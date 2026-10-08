"""Route A status for a human: what it is doing, what it has done, and whether it is waiting, bottlenecked, idle or stalled -- plus the guard counts and the
in-flight guard control file.   (user 2026-10-08: "guard counts ... tinker with guards, thresholds etc even while in flight to maximize recoverable area ...
update the summary monitor so it's easy to see what it's doing and what it's done and whether it's waiting on anything, bottlenecked, idle or stalled")

  python -m vesuvius_pipeline.routea_cloud.status --work /workspace/routeB/routeA_work            # 4-6 line summary
  python -m vesuvius_pipeline.routea_cloud.status --work ... --json                                # everything
  python -m vesuvius_pipeline.routea_cloud.status --work ... guards show | set GATE_FOLD_FRAC=0.03 | clear | policy roughness_enforce=0

THE CONTROL FILE  <work>/control/guards.json   {"gate": {"GATE_FOLD_FRAC": 0.03, ...}, "policy": {"roughness_enforce": "0", ...}}
  * `gate` = resume_gate TUNABLES (GATE_FOLD_FRAC, GATE_NORMAL_REV_FRAC, GATE_PROX_FRAC, GATE_HAIRPIN_FRAC, GATE_TRANSVERSE_CELLS, GATE_MIN_CELLS).  Re-read at EVERY
    round of EVERY seed (also seeds already running): the next gate check uses the new value.  The value in effect is recorded in each round of export.json.
  * `policy` = grow.guard.<key> settings (the in-solve guards).  Read when a seed STARTS: applies to seeds started afterwards, not to rounds already running.
  * Lowering a threshold makes the gate stricter.  The gate is fail-closed and its thresholds are a PROPOSAL with no human reference (resume_gate.py): loosening one
    trades verified-clean area for recovered area; `guards sweep` shows how many held seeds each multiplier would release, and the hub's import verification
    (cloud_import) still re-checks every segment with the fleet detector.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

REASON_THR = {"fold_over": "GATE_FOLD_FRAC", "normal_reversal": "GATE_NORMAL_REV_FRAC", "self_proximity": "GATE_PROX_FRAC", "hairpin": "GATE_HAIRPIN_FRAC",
              "transverse": "GATE_TRANSVERSE_CELLS"}
SHORT = {"fold_over": "fold", "normal_reversal": "nrev", "self_proximity": "prox", "hairpin": "hair", "transverse": "trans", "unrunnable": "unrun"}
GATE_KEYS = ("GATE_FOLD_FRAC", "GATE_NORMAL_REV_FRAC", "GATE_PROX_FRAC", "GATE_HAIRPIN_FRAC", "GATE_TRANSVERSE_CELLS", "GATE_MIN_CELLS")
STALL_S = 20 * 60                 # tracers running but no seed file touched for this long = stalled
DOWNLOAD_RECENT_S = 150           # a download progress line newer than this = downloading


def control_path(work) -> Path:
    return Path(work) / "control" / "guards.json"


def load_control(work) -> dict:
    try:
        d = json.loads(control_path(work).read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save_control(work, d: dict) -> Path:
    p = control_path(work)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1, sort_keys=True))
    os.replace(tmp, p)
    return p


def clean_gate(d: dict) -> tuple[dict, list]:
    """Only known TUNABLES, numeric.  Returns (accepted, rejected-with-reason) -- an unknown key is announced, never silently kept."""
    ok, bad = {}, []
    for k, v in (d or {}).items():
        if k not in GATE_KEYS:
            bad.append((k, "unknown gate tunable"))
            continue
        try:
            ok[k] = float(v)
        except (TypeError, ValueError):
            bad.append((k, f"not a number: {v!r}"))
    return ok, bad


def tracer_count() -> int:
    n = 0
    try:
        for p in os.listdir("/proc"):
            if p.isdigit():
                try:
                    with open(f"/proc/{p}/cmdline", "rb") as f:
                        if b"vc_grow_seg_from_seed" in f.read():
                            n += 1
                except OSError:
                    continue
    except OSError:
        pass
    return n


def cpu_busy_pct(sample_s: float = 0.4) -> float | None:
    def rd():
        with open("/proc/stat") as f:
            v = [float(x) for x in f.readline().split()[1:]]
        return sum(v), v[3] + (v[4] if len(v) > 4 else 0.0)
    try:
        t0, i0 = rd()
        time.sleep(sample_s)
        t1, i1 = rd()
        return round(100.0 * (1.0 - (i1 - i0) / max(1e-9, t1 - t0)), 1)
    except (OSError, ValueError, IndexError):
        return None


def newest_area(sd: Path) -> float:
    best = None
    for m in sd.glob("r*/*/meta.json"):
        k = m.stat().st_mtime
        if best is None or k > best[0]:
            best = (k, m)
    if not best:
        return 0.0
    try:
        return float(json.loads(best[1].read_text()).get("area_cm2") or 0.0)
    except (OSError, ValueError, TypeError):
        return 0.0


def area_at_round(sd: Path, r: int) -> float:
    best = None
    for m in sd.glob(f"r{r}/*/meta.json"):
        k = m.stat().st_mtime
        if best is None or k > best[0]:
            best = (k, m)
    if not best:
        return 0.0
    try:
        return float(json.loads(best[1].read_text()).get("area_cm2") or 0.0)
    except (OSError, ValueError, TypeError):
        return 0.0


def _tail(p: Path, n: int = 6000) -> str:
    try:
        with open(p, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - n))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def summarize(work, slots: int | None = None, tracers: int | None = None, cpu: float | None = None, now: float | None = None) -> dict:
    work = Path(work)
    now = now or time.time()
    home = work.parent
    out = {"work": str(work), "now": now}
    # ---- seeds: planned / finished
    planned = 0
    for sf in (work / "seeds").glob("*.json"):
        try:
            planned += len(json.loads(sf.read_text()))
        except (OSError, ValueError):
            pass
    st, area_by, tally, rows, newest = {}, {}, {}, [], 0.0
    fin_times, finals, usable, ext_done = [], [], 0.0, 0
    for exp in (work / "export").glob("*/*/export.json"):
        try:
            ex = json.loads(exp.read_text())
        except (OSError, ValueError):
            continue
        newest = max(newest, exp.stat().st_mtime)
        fin_times.append(exp.stat().st_mtime)
        s = (ex.get("run") or {}).get("status", "?")
        st[s] = st.get(s, 0) + 1
        a = newest_area(exp.parent)
        area_by[s] = area_by.get(s, 0.0) + a
        # USABLE area (user: "any segment that is usable is valid"): a held segment's usable surface is its newest round that PASSED the gate = the round before the failing one
        us = area_at_round(exp.parent, len(ex.get("rounds") or []) - 1) if s == "gate_held" else a
        usable += us
        finals.append(us)
        if "extended_from_round" in (ex.get("run") or {}):
            ext_done += 1
        if s == "gate_held":
            rd = [r for r in (ex.get("rounds") or []) if (r.get("gate") or {}).get("pass") is False]
            g = (rd[-1] if rd else {}).get("gate") or {}
            reasons = list((g.get("reasons") or {}).keys())
            for r in reasons:
                tally[r] = tally.get(r, 0) + 1
            rows.append({"seg": exp.parent.name, "area": a, "reasons": reasons, "fractions": g.get("fractions") or {}, "thresholds": g.get("thresholds") or {},
                         "rounds": len(ex.get("rounds") or [])})
    done = sum(st.values())
    out.update(seeds_planned=planned, seeds_done=done, status=st, area_cm2=round(sum(area_by.values()), 1), area_by_status={k: round(v, 1) for k, v in area_by.items()},
               held_reasons=tally, held=len(rows))
    # ---- near-miss sweep: how many held seeds a looser threshold multiplier would release (transverse is zero-tolerance: never scaled)
    # seeds finished BEFORE thresholds were recorded per round: fall back to the thresholds in effect now (pinned values + control file)
    cur = {}
    try:
        from .. import resume_gate as _RG
        cur = {k: v[0] for k, v in _RG.TUNABLES.items()}
        try:
            pin = json.loads((Path(__file__).resolve().parents[3] / "pins" / "settings_snapshot.json").read_text()).get("resume_gate") or {}
            cur.update({k: float(v) for k, v in pin.items() if k in cur})
        except (OSError, ValueError):
            pass
        cur.update(clean_gate(load_control(work).get("gate") or {})[0])
    except ImportError:
        pass
    for r in rows:
        if not r["thresholds"]:
            r["thresholds"] = dict(cur)
            r["thresholds_assumed"] = True
    out["sweep_assumed_thresholds"] = sum(1 for r in rows if r.get("thresholds_assumed"))
    sweep = {}
    for mult in (1.25, 1.5, 2.0, 3.0):
        n = 0
        for r in rows:
            need = 0.0
            blocked = False
            for k in r["reasons"]:
                thn = REASON_THR.get(k)
                if thn is None or k == "transverse":
                    blocked = True
                    break
                fr = r["fractions"].get(k)
                th = r["thresholds"].get(thn)
                if fr is None or not th:
                    blocked = True
                    break
                need = max(need, float(fr) / float(th))
            if not blocked and need <= mult:
                n += 1
        sweep[str(mult)] = n
    out["sweep"] = sweep
    # ---- report / phase
    rep = {}
    try:
        rep = json.loads((work / "report.json").read_text())
    except (OSError, ValueError):
        pass
    finished = bool(rep.get("finished_utc"))
    out["finished_utc"] = rep.get("finished_utc")
    out["verified_pass"] = (rep.get("totals") or {}).get("verified_pass")
    # ---- remaining seeds, finish rate, estimated total (crude: finished seeds' mean usable area stands in for every unfinished seed; the in-flight partial area is the floor)
    ext_only = bool((rep.get("args") or {}).get("extend_only"))
    if ext_only:
        total_jobs = sum((v or {}).get("extend_candidates", 0) for v in (rep.get("scrolls") or {}).values())
        remaining = max(0, total_jobs - ext_done)
    else:
        remaining = max(0, planned - done)
    inflight = []
    for sd in (work / "export").glob("*/*"):
        if sd.is_dir() and not (sd / "export.json").is_file() and any(sd.glob("r*/*/meta.json")):
            inflight.append(newest_area(sd))
    mean_final = (sum(finals) / len(finals)) if finals else 0.0
    est_rem = sum(max(0.0, mean_final - a_) for a_ in inflight) + max(0, remaining - len(inflight)) * mean_final
    est_final = usable + sum(inflight) + est_rem
    recent = sum(1 for t in fin_times if now - t < 1800)
    out.update(usable_cm2=round(usable, 1), est_final_cm2=round(est_final, 1), mean_final_cm2=round(mean_final, 1), seeds_remaining=remaining, inflight=len(inflight),
               finish_rate_per_h=round(recent * 2.0, 1), extend_only=ext_only)
    try:
        from . import overlap as _ov
        ov = _ov.cached_scan(work)
    except Exception:  # noqa: BLE001
        ov = None
    out["overlap"] = ov
    if ov:
        out["est_unique_final_cm2"] = round(est_final - ov.get("dup_cm2", 0.0), 1)
    # ---- processes, cpu, logs
    tr = tracer_count() if tracers is None else tracers
    cb = cpu_busy_pct() if cpu is None else cpu
    logs = sorted((home / "box8" / "logs").glob("routeA*.log"), key=lambda p: p.stat().st_mtime) if (home / "box8" / "logs").exists() else []
    tail, log_age = "", None
    if logs:
        tail, log_age = _tail(logs[-1]), now - logs[-1].stat().st_mtime
    dl = None
    for ln in reversed(tail.splitlines()[-12:]):
        m = re.search(r"s3 (\S+?)/representations.*?: (\d+)/(\d+) objects", ln)
        if m:
            dl = f"{m.group(1)} {m.group(2)}/{m.group(3)} objects"
            break
    out.update(tracers=tr, slots=slots, cpu_busy=cb, log_age_s=None if log_age is None else int(log_age), downloading=dl,
               idle_age_s=int(now - newest) if newest else None)
    # ---- state: one word + the reason, in order of what the user must act on
    if finished and tr == 0:
        state, why = "FINISHED", "the run ended (cores are idle: start another round)"
    elif tr == 0 and dl and (log_age is not None and log_age < DOWNLOAD_RECENT_S):
        state, why = "WAITING", f"downloading inputs ({dl}); grows start after ALL listed scrolls are local"
    elif tr == 0 and not finished and logs and log_age is not None and log_age < 300:
        state, why = "STARTING", "no tracers yet (kit / inputs / seeds phase)"
    elif tr == 0:
        state, why = "IDLE", "no tracers running and nothing downloading: cores are idle"
    elif newest and now - newest > STALL_S and (log_age is None or log_age > STALL_S):
        state, why = "STALLED", f"{tr} tracers but no seed file or log line for {int((now - newest) / 60)} min"
    elif slots and tr < 0.6 * slots:
        state, why = "UNDERBOOKED", f"{tr} tracers for {slots} slots (seeds queued {max(0, planned - done - tr)}: the pool drains as seeds finish)"
    else:
        state, why = "GROWING", f"{tr} tracers" + (f" of {slots} slots" if slots else "")
    out.update(state=state, why=why)
    return out


def render_lines(d: dict, width: int = 100) -> list:
    st = d["status"]
    ok = sum(v for k, v in st.items() if k not in ("gate_held", "no_checkpoint", "scrub_failed", "?"))
    cpu = "" if d.get("cpu_busy") is None else f" cpu {d['cpu_busy']:.0f}% busy"
    l1 = f"A: {d['state']}  {d['why']}{cpu}"
    est = ""
    if d.get("est_final_cm2") is not None:
        est = f" | est final ~{d['est_final_cm2']:.0f} cm2" + (f" (~{d['est_unique_final_cm2']:.0f} unique)" if d.get("est_unique_final_cm2") is not None else "")
    total = d["seeds_planned"] or "?"
    l2 = (f"A: {d['seeds_done']}/{total} seeds, {d.get('tracers', 0)} run | {d['area_cm2']:.0f} grown, {d.get('usable_cm2', 0):.0f} usable{est}")
    out = [l1[:width], l2[:width]]
    if d["held"]:
        hs = " ".join(f"{SHORT.get(k, k)} {v}" for k, v in sorted(d["held_reasons"].items(), key=lambda kv: -kv[1]))
        held_area = d["area_by_status"].get("gate_held", 0.0)
        out.append(f"A: ok {ok}, held {d['held']} by guard: {hs} | {held_area:.0f} cm2 stopped early"[:width])
        sw = d["sweep"]
        out.append(f"A: loosen x1.25 frees {sw['1.25']}, x1.5 {sw['1.5']}, x2 {sw['2.0']}, x3 {sw['3.0']} (guards set ...)"[:width])
    ov = d.get("overlap")
    if ov:
        if ov["pairs"]:
            w = ov["pairs"][0]
            out.append(f"A: OVERLAP {ov['n_overlapping']} segs share ~{ov['dup_cm2']:.0f} cm2; worst {w['small'][-9:]}~{w['large'][-9:]} {w['frac'] * 100:.0f}%"[:width])
        else:
            out.append(f"A: overlap none ({ov['n_segments']} segs, eps {ov['eps_vox']:g} vox){' PARTIAL' if ov.get('partial') else ''}"[:width])
    return out


def _fmt_control(work) -> list:
    c = load_control(work)
    lines = [f"control file: {control_path(work)} ({'present' if c else 'absent'})"]
    try:
        from .. import resume_gate as RG
        base = {k: v[0] for k, v in RG.TUNABLES.items()}
    except ImportError:
        base = {}
    over, bad = clean_gate(c.get("gate") or {})
    for k in GATE_KEYS:
        b = base.get(k)
        lines.append(f"  {k:<24} {over.get(k, b)}" + (f"   (override; pinned {b})" if k in over else f"   (pinned)"))
    for k, why in bad:
        lines.append(f"  REJECTED {k}: {why}")
    if c.get("policy"):
        lines.append("  policy (next seeds): " + json.dumps(c["policy"], sort_keys=True))
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=os.environ.get("ROUTEA_WORK") or "/workspace/routeB/routeA_work")
    ap.add_argument("--slots", type=int, default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("cmd", nargs="*", help="guards show | guards set K=V ... | guards clear | guards sweep | policy K=V ...")
    a = ap.parse_args(argv)
    work = Path(a.work)
    if a.cmd[:1] == ["guards"]:
        sub = (a.cmd[1:2] or ["show"])[0]
        if sub == "set":
            c = load_control(work)
            kv = dict(x.split("=", 1) for x in a.cmd[2:] if "=" in x)
            acc, bad = clean_gate(kv)
            c.setdefault("gate", {}).update(acc)
            save_control(work, c)
            for k, why in bad:
                print(f"REJECTED {k}: {why}")
        elif sub == "clear":
            c = load_control(work)
            c.pop("gate", None)
            save_control(work, c)
            print("gate overrides cleared (pinned thresholds apply again)")
        elif sub == "sweep":
            d = summarize(work, a.slots)
            print(json.dumps(d["sweep"]), "held:", d["held"])
            return 0
        print("\n".join(_fmt_control(work)))
        return 0
    if a.cmd[:1] == ["policy"]:
        c = load_control(work)
        kv = dict(x.split("=", 1) for x in a.cmd[1:] if "=" in x)
        c.setdefault("policy", {}).update(kv)
        save_control(work, c)
        print("\n".join(_fmt_control(work)))
        return 0
    d = summarize(work, a.slots)
    print(json.dumps(d, indent=1, sort_keys=True) if a.json else "\n".join(render_lines(d)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
