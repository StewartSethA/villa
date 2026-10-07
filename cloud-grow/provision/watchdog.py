#!/usr/bin/env python3
"""Plain (non-LLM) fleet watchdog, stdlib only. Run locally (or on a tiny always-on VM).

  watchdog.py --plan PLAN.json [--state S.json] [--once | --interval 600] [--outdir fleet_out]

Each tick, per box: reachable? bootstrap ok? runner alive? tool md5 drift? rounds/h and verified cm2/h vs the plan's model?
failure rate, guard pauses, selfx_unverified markers, disk/RAM; fleet: spend model vs cap, deadline, instance-count tripwire.
Writes <outdir>/<name>.status.json + .status.md; exit 0 ok, 1 trouble (any TROUBLE finding), 3 a tripwire fired.
AUTO-ACTIONS (only these): `restart-runner` on a box whose runner is dead (<= 5 per box, never after STOP, never on drift), and `down`
when the spend cap or deadline is reached or the instance-count tripwire fires. Everything else is reported, never fixed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import plan as P  # noqa: E402

WARMUP_H = 2.0                  # no throughput judgement before this
MAX_RESTARTS = 5
FAIL_RATE_TROUBLE = 0.05        # D14: a few per cent is a MAJOR alert
MIN_OUTCOMES_FOR_RATE = 10
THROUGHPUT_FRACTION = 0.3       # of the model's LOW prod-equivalent; the model is +-25 % extrapolated, so this is a gross-failure test only
SPEND_WARN = 0.8
PROBE_CMD = ("P=$(ls /opt/cloud-grow/cloud-grow/config/tool_pins.json /opt/cloud-grow/config/tool_pins.json 2>/dev/null | head -1); "
             "python3 /usr/local/bin/cg-box-probe.py --pins \"$P\"")
RESTART_CMD = "sudo -n systemctl restart cloud-grow || (cd /tmp && CG=$(ls -d /opt/cloud-grow/cloud-grow /opt/cloud-grow 2>/dev/null | head -1) BOX=$(hostname) nohup /usr/local/bin/cg-box-loop.sh >>/data/cloud-grow/runner.out 2>&1 &)"


def ssh_argv(target: str, remote: str, key: str = "") -> list[str]:
    t = shlex.split(target)           # vast targets look like "root@host -p 1234"
    opts = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "StrictHostKeyChecking=accept-new"] + (["-i", key] if key else [])
    return ["ssh", *opts, *t, remote]


def real_exec(argv: list[str], timeout=90) -> tuple[int, str]:
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout
    except (OSError, subprocess.SubprocessError) as e:
        return 255, f"{type(e).__name__}: {e}"


def probe_all(targets: dict, key: str, execf=real_exec) -> dict:
    out = {}
    for box, t in targets.items():
        rc, o = execf(ssh_argv(t, PROBE_CMD, key))
        try:
            out[box] = {"reachable": True, **json.loads(o.strip().splitlines()[-1])} if rc == 0 else {"reachable": False, "why": f"rc={rc}"}
        except (ValueError, IndexError):
            out[box] = {"reachable": False, "why": f"unparseable probe output rc={rc}"}
    return out


def tick(plan: dict, st: dict, probes: dict, prev: dict | None, spend: dict, n_instances_seen: int | None, now: dt.datetime | None = None) -> dict:
    """Pure judgement. Returns {findings:[{level,box,msg}], actions:[{do,box?}], boxes:{}, fleet:{}}."""
    now = now or dt.datetime.now(dt.timezone.utc)
    prev = prev or {}
    F, A = [], []
    add = lambda lv, box, msg: F.append({"level": lv, "box": box, "msg": msg})  # noqa: E731
    cap = plan["spend_cap_usd"]
    n = len(st.get("shards", []))
    started = P._utc(st["started_utc"]) if st.get("started_utc") else None
    el_h = (now - started).total_seconds() / 3600.0 if started else 0.0
    pr = P.projection(plan, started or now) if started else None
    pe_low = (pr["fleet_cm2_per_hour_prod_equiv"]["low"] / max(n, 1)) if pr else 0.0
    # ---- fleet-level tripwires (these may `down`)
    usd = spend["usd"]
    trip = None
    if usd >= cap:
        trip = f"spend model ${usd} >= cap ${cap}"
    elif now >= P._utc(plan["deadline_utc"]):
        trip = "deadline reached"
    elif n_instances_seen is not None and n_instances_seen > n:
        trip = f"{n_instances_seen} instances exist for this plan tag but the plan has {n}: runaway/orphan"
    if trip:
        add("TRIPWIRE", None, trip)
        A.append({"do": "down", "why": trip})
    elif usd >= SPEND_WARN * cap:
        add("WARN", None, f"spend model ${usd} is >= {int(SPEND_WARN * 100)} % of cap ${cap}")
    boxes = {}
    for b in st.get("shards", []):
        name = b["box"]
        pb = probes.get(name) or {"reachable": False, "why": "no probe"}
        row = {"reachable": pb.get("reachable", False)}
        pv = (prev.get("boxes") or {}).get(name, {})
        if not pb.get("reachable"):
            miss = pv.get("unreachable_ticks", 0) + 1
            row["unreachable_ticks"] = miss
            add("TROUBLE" if miss >= 2 else "WARN", name, f"unreachable ({pb.get('why')}), {miss} tick(s) in a row")
            boxes[name] = row
            continue
        row.update({k: pb.get(k) for k in ("loadavg", "ram_avail_gb", "disk_free_gb", "runner_loop_alive", "tracer_procs", "rounds_total", "rounds_last_hour",
                                           "verified_cm2_now", "segments", "guard_pauses", "selfx_unverified_markers", "alerts", "tarballs")})
        row["restarts"] = pv.get("restarts", 0)
        if pb.get("bootstrap_failed"):
            add("TROUBLE", name, "BOOTSTRAP_FAILED marker present (see /var/log/cloud-grow-bootstrap.log): runner never started")
        if pb.get("tool_md5_drift"):
            add("TROUBLE", name, f"tool md5 drift: {pb['tool_md5_drift']} -- NOT restarting; geometry may differ from the pinned build")
        elif pb.get("tool_pins_checked", 0) == 0:
            add("WARN", name, "no tool pins checked by the probe (pins file not found on box)")
        if not pb.get("runner_loop_alive") and not pb.get("stop_file") and not pb.get("bootstrap_failed") and not pb.get("tool_md5_drift"):
            if row["restarts"] < MAX_RESTARTS:
                A.append({"do": "restart-runner", "box": name, "why": "runner loop not alive"})
                row["restarts"] += 1
                add("WARN", name, f"runner dead: restart {row['restarts']}/{MAX_RESTARTS} (resume path D3)")
            else:
                add("TROUBLE", name, f"runner dead and {MAX_RESTARTS} restarts used")
        if pb.get("selfx_unverified_markers"):
            add("TROUBLE", name, f"{pb['selfx_unverified_markers']} selfx_unverified marker(s): self-crossing check did not run on some surface; the importer will refuse them")
        if pb.get("guard_pauses"):
            add("WARN", name, f"{pb['guard_pauses']} guard pause(s)")
        seg = pb.get("segments") or {}
        tot = sum(seg.values())
        failed = seg.get("failed", 0)
        row["failure_rate"] = round(failed / tot, 3) if tot else None
        if tot >= MIN_OUTCOMES_FOR_RATE and failed / tot > FAIL_RATE_TROUBLE:
            add("TROUBLE", name, f"failure rate {failed}/{tot} segments = {failed / tot:.0%} > {FAIL_RATE_TROUBLE:.0%}: STOP ADMITTING, read the `why` strings (rounds.jsonl) before more spend (D14)")
        if (pb.get("disk_free_gb") or 1e9) < 20:
            add("TROUBLE", name, f"disk free {pb['disk_free_gb']} GB < 20")
        if (pb.get("ram_avail_gb") or 1e9) < 8:
            add("WARN", name, f"RAM available {pb['ram_avail_gb']} GB < 8")
        if el_h >= WARMUP_H and pb.get("runner_loop_alive"):
            if (pb.get("rounds_last_hour") or 0) == 0:
                add("TROUBLE", name, "runner alive but 0 rounds in the last hour")
            gain = (pb.get("verified_cm2_now") or 0) - (pb.get("verified_cm2_1h_ago") or 0)
            row["verified_cm2_per_h"] = round(gain, 2)
            if pe_low and gain < THROUGHPUT_FRACTION * pe_low:
                add("WARN", name, f"verified cm2/h {gain:.1f} < {THROUGHPUT_FRACTION:.0%} of model low {pe_low:.1f} (model +-25 %, extrapolated; box claim)")
        boxes[name] = row
    trouble = any(f["level"] in ("TROUBLE", "TRIPWIRE") for f in F)
    return {"t": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "plan": plan["name"], "elapsed_h": round(el_h, 2), "spend_model": spend, "cap_usd": cap,
            "findings": F, "actions": A, "boxes": boxes, "tripwire": trip, "trouble": trouble,
            "fleet": {"boxes": n, "reachable": sum(1 for r in boxes.values() if r.get("reachable")), "rounds_last_hour": sum((r.get("rounds_last_hour") or 0) for r in boxes.values()),
                      "verified_cm2_now_box_claims": round(sum((r.get("verified_cm2_now") or 0) for r in boxes.values()), 1)}}


def to_markdown(s: dict) -> str:
    L = [f"# fleet {s['plan']} status {s['t']}", "",
         f"- elapsed {s['elapsed_h']} h; spend (MODEL, not the bill) ${s['spend_model']['usd']} of cap ${s['cap_usd']}",
         f"- boxes reachable {s['fleet']['reachable']}/{s['fleet']['boxes']}; rounds in last hour {s['fleet']['rounds_last_hour']}; verified cm2 (box claims, unvalidated D6) {s['fleet']['verified_cm2_now_box_claims']}",
         f"- verdict: **{'TRIPWIRE: ' + s['tripwire'] if s['tripwire'] else ('TROUBLE' if s['trouble'] else 'ok')}**", ""]
    if s["findings"]:
        L += ["## findings", ""] + [f"- {f['level']} {f['box'] or 'fleet'}: {f['msg']}" for f in sorted(s["findings"], key=lambda f: f['level'] != 'TRIPWIRE')] + [""]
    if s["actions"]:
        L += ["## auto-actions this tick", ""] + [f"- {a['do']} {a.get('box', '')} ({a['why']})" for a in s["actions"]] + [""]
    L += ["## boxes", "", "| box | up | runner | rounds/h | verified cm2 | fail rate | pauses | selfx_unv | disk GB | RAM GB | restarts |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in s["boxes"].items():
        L.append(f"| {k} | {r.get('reachable')} | {r.get('runner_loop_alive')} | {r.get('rounds_last_hour')} | {r.get('verified_cm2_now')} | {r.get('failure_rate')} | {r.get('guard_pauses')} | {r.get('selfx_unverified_markers')} | {r.get('disk_free_gb')} | {r.get('ram_avail_gb')} | {r.get('restarts')} |")
    return "\n".join(L) + "\n"


def write_status(outdir: str, s: dict) -> None:
    os.makedirs(outdir, exist_ok=True)
    for ext, body in (("json", json.dumps(s, indent=1)), ("md", to_markdown(s))):
        p = os.path.join(outdir, f"{s['plan']}.status.{ext}")
        with open(p + ".tmp", "w") as fh:
            fh.write(body)
        os.replace(p + ".tmp", p)


def apply_actions(plan, st, s, key, execf, down_fn) -> list[str]:
    """Only restart-runner and down. `down_fn()` is supplied by the caller (fleet.cmd_down)."""
    done = []
    for a in s["actions"]:
        if a["do"] == "restart-runner":
            tgt = st.get("targets", {}).get(a["box"])
            if tgt:
                rc, _ = execf(ssh_argv(tgt, RESTART_CMD, key))
                done.append(f"restart-runner {a['box']} rc={rc}")
        elif a["do"] == "down":
            done.append(f"down rc={down_fn()}")
    return done


def main(argv=None, execf=real_exec) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--state", default="")
    ap.add_argument("--outdir", default="./fleet_out")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=600)
    a = ap.parse_args(argv)
    plan = P.load(a.plan)
    sp = a.state or os.path.join(a.outdir, f"{plan['name']}.fleet_state.json")
    import fleet as FL   # noqa: E402
    prev = None
    while True:
        st = FL.load_state(sp)
        spend = FL.spend_so_far(plan, st)
        probes = probe_all(st.get("targets", {}), plan.get("ssh_key_path", ""), execf)
        s = tick(plan, st, probes, prev, spend, None)
        write_status(a.outdir, s)
        print(to_markdown(s))
        def down():
            ns = argparse.Namespace(state=sp, outdir=a.outdir, yes=True, dry_run=False)
            return FL.cmd_down(ns, plan)
        for d in apply_actions(plan, st, s, plan.get("ssh_key_path", ""), execf, down):
            print("[auto-action]", d)
        prev = s
        if a.once or s["tripwire"]:
            return 3 if s["tripwire"] else (1 if s["trouble"] else 0)
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
