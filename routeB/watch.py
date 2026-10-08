"""Live dashboard for a running box8 (routeB_watch.sh).  Read-only: it only reads files under $ROUTEB_HOME, nvidia-smi and /proc; Ctrl-C detaches the watcher and
never touches the run (which lives in the tmux session `routeb`).

  routeB_watch.sh [--home DIR] [--interval 5] [--plain] [--once]
    default   redraw every ~5 s on a tty
    --plain   no colour, no screen clearing: one block per interval (non-tty / log capture)
    --once    ONE diagnostic snapshot block (everything worth pasting back: STATUS, budget, link, GPUs, jobs, alerts, stream tail, control, df) and exit
Sections: header (box, commit, uptime, $ vs soft/hard, budget clock from the REAL box start, link + trend, disk, RAM), GPU rows, fetch rows, Route A, payload table,
ALERTS (grouped failures, stalls, STAGING BLOCKED, idle GPU beside a queue), the exact PULL command for your machine, and a blended event stream.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

RED, YEL, GRN, DIM, RST, BLD = "\033[31m", "\033[33m", "\033[32m", "\033[2m", "\033[0m", "\033[1m"
PROG = re.compile(r"PROGRESS Optimizing\s+\S*\s*([\d,]+)/([\d,]+) iterations[^\r\n]*")
RATE = re.compile(r"([\d.]+) it/s")
ETA = re.compile(r"ETA\s+([0-9hms ]+)")
SUMRX = re.compile(r"^(PHerc\w+) ([\d.]+)(?:/([\d.]+))? GB (\d+) MB/s (\d+) obj/s(?: ETA (\d+) min)?( DONE)?(?: \| (.*))?$")
OBJ = re.compile(r"s3 (\S+): (\d+)/(\d+) objects, ([\d.]+) GB")
BLK = re.compile(r"(\S+): (\d+)/(\d+) blocks, ([\d.]+) GB this run, ([\d.]+) MB/s")
TS = re.compile(r"^\[(\d\d):(\d\d):(\d\d)\] (\w+): (.*)$")
SEV = [(re.compile(r"FAIL|ERROR|TERMINAL|Traceback|DESTROY|!!!|ALARM \[|OOM|HARD BUDGET|KILL|IDLE|NOT PROGRESSING|crash", re.I), "red"),
       (re.compile(r"WARN|BLOCKED|DEFERRED|REPLAN|DEGRADED|SHRINK|NOT LAUNCHED|stall|foreign|re-queued|RESCALE|FILL-IDLE|RESUME", re.I), "yellow"),
       (re.compile(r"DONE|PULLED|RECOVERED|CLEARED|complete|OK\b|MEASURED", re.I), "green")]


def readj(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return default


def tail_text(p: Path, n: int = 65536) -> str:
    try:
        with open(p, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - n))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def sh(cmd: list, timeout: float = 8.0) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def gpu_query() -> list[dict]:
    rows = []
    for l in sh(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]).splitlines():
        p = [x.strip() for x in l.split(",")]
        if len(p) == 4:
            rows.append({"idx": p[0], "util": float(p[1]), "mem": float(p[2]), "total": float(p[3])})
    return rows


def meminfo() -> tuple[float, float]:
    d = {}
    try:
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1)
            d[k] = int(v.split()[0]) / 1048576.0
    except OSError:
        return 0.0, 0.0
    return d.get("MemAvailable", 0.0), d.get("MemTotal", 0.0)


def _repo_raw_base() -> str:
    """raw.githubusercontent base of THIS checkout's branch (so the user's machine can curl pull_box8.py), placeholders if unknown."""
    import subprocess
    root = Path(__file__).resolve().parents[1]
    def g(*a):
        try:
            return subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    url, br = g("config", "--get", "remote.origin.url"), g("rev-parse", "--abbrev-ref", "HEAD")
    m = re.search(r"github\.com[:/]+([^/]+/[^/.]+?)(?:\.git)?$", url)
    repo = m.group(1) if m else "StewartSethA/villa"                  # the deploy repo; the branch name is what matters
    return f"https://raw.githubusercontent.com/{repo}/{br if br and br != 'HEAD' else 'routeAB-deploy-v5.1'}"


def pull_lines(home: Path, env=None, full: bool = False) -> list[str]:
    """Copy-paste commands for the USER'S machine, every line <= 100 chars, no heredoc, no continuation backslash.  Address from SSH_CONNECTION / VAST_* /
    RUNPOD_* else placeholders.  Lines 0-1 are always H= and P= (the DIRECT address).  full=True adds: fetch pull_box8.py from the branch, the verified pull
    command, the vast.ai SSH-PROXY variant (instance id from VAST_CONTAINERLABEL -> `vastai ssh-url <id>`), and a plain-rsync fallback."""
    e = os.environ if env is None else env
    ip, port = "<BOX-IP>", "<PORT>"
    if e.get("SSH_CONNECTION"):
        parts = e["SSH_CONNECTION"].split()
        if len(parts) == 4:
            ip, port = parts[2], parts[3]
    if e.get("PUBLIC_IPADDR"):
        ip = e["PUBLIC_IPADDR"]
        port = e.get("VAST_TCP_PORT_22") or port
    if e.get("RUNPOD_PUBLIC_IP"):
        ip = e["RUNPOD_PUBLIC_IP"]
        port = e.get("RUNPOD_TCP_PORT_22") or port
    user = "root" if (e.get("USER") in (None, "root") or os.geteuid() == 0) else e.get("USER", "root")
    base = [f"H={user}@{ip}", f"P={port}", f"RH={home}"]
    if not full:
        return base + ['while :;do rsync -aH --partial --append-verify -e "ssh -p $P" $H:$RH/out/ out/;sleep 60;done',
                       "./routeB_pull.sh --host $H --port $P --remote-home $RH --dest out --final"]
    label = e.get("VAST_CONTAINERLABEL", "")
    iid = label.split(".", 1)[1] if label.startswith("C.") else (e.get("CONTAINER_ID", "") or "<INSTANCE-ID>")
    pull = "python3 pull_box8.py --host $H --port $P --remote-home $RH --dest out --poll 120 --final"
    return base + [
        "# A) DIRECT.  On YOUR machine (needs ssh + rsync + python3; verifies md5, marks units pulled, resumable):",
        "B=" + _repo_raw_base(),
        "curl -fsSL $B/pull_box8.py -o pull_box8.py",
        pull,
        "# B) via the vast.ai SSH PROXY (if the direct port is blocked).  Get its host/port from the instance 'Connect' dialog,",
        f"#    or with the vast CLI:  vastai ssh-url {iid}   (prints ssh://root@sshN.vast.ai:PORT).  Then:",
        "H=root@sshN.vast.ai",
        "P=<PROXY-PORT>",
        pull,
        "# C) no script at all (checksum afterwards with: python3 pull_box8.py --verify-only --dest out):",
        'while :;do rsync -aH --partial --append-verify -e "ssh -p $P" $H:$RH/out/ out/;sleep 60;done',
        "# add  -i ~/.ssh/<your key>  to the ssh/pull commands if you use a key file (both pull_box8.py -i and rsync -e 'ssh -i ..')",
    ]


def newest_area(sd: Path):
    best = None
    for m in sd.glob("r*/*/meta.json"):
        k = m.stat().st_mtime
        if best is None or k > best[0]:
            best = (k, m)
    if not best:
        return 0.0
    try:
        return float(json.loads(best[1].read_text()).get("area_cm2") or 0.0)
    except (OSError, ValueError):
        return 0.0


def fetch_total_gb(scroll: str) -> float:
    """Expected GB to stage for a scroll: tracks (registry sizes) + the z-slab lasagna estimate (planner constants)."""
    try:
        from .common import spec
        from . import planner as PL
        sp = spec(scroll)
        tr = sum(v for k, v in sp["tracks"]["files"].items() if k.endswith(".dbm")) / 1e9
        return tr + PL.LAS_GB_PER_SLICE * (13000 + 512)
    except Exception:                                    # noqa: BLE001 - unknown scroll: no total
        return 0.0


def collect(home: Path, now: float | None = None, gpus=None, log_path: Path | None = None, env=None, tree: Path | None = None) -> dict:
    now = now or time.time()
    H = Path(home)
    D = H / "box8"
    snap = {"now": now, "home": str(H)}
    st = readj(H / "out" / "STATUS.json", {}) or {}
    snap["status"] = st
    snap["budget"] = st.get("budget") or {}
    snap["alerts_file"] = (readj(H / "out" / "ALERTS.json", {}) or {}).get("alarms", [])
    first = readj(D / "link" / "first.json", {}) or {}
    snap["link_first"] = first
    trend = []
    try:
        trend = [json.loads(l) for l in (D / "link" / "trend.jsonl").read_text().splitlines() if l.strip()][-8:]
    except OSError:
        pass
    snap["link_trend"] = trend
    # jobs / state
    jobs = []
    scrolls = {}
    for p in sorted((D / "state").glob("*.json")):
        sj = readj(p, {}) or {}
        scrolls[sj.get("scroll", p.stem)] = {"fetched": sj.get("fetched"), "fetch_failed": sj.get("fetch_failed"), "finalized": sj.get("finalized")}
        jobs += sj.get("jobs", [])
    snap["jobs"] = jobs
    snap["scrolls"] = scrolls
    byid = {j["id"]: j for j in jobs}
    rows = []
    gq = gpus() if gpus else gpu_query()
    for g in gq:
        r = {**g, "job": None}
        for j in jobs:
            if j.get("status") == "running" and str(j.get("gpu")) == g["idx"]:
                scroll, tag = j["scroll"], j["tag"]
                txt = tail_text(H / "runs" / scroll / tag / "fit" / "fit.log", 30000)
                prog = None
                for m in PROG.finditer(txt.replace("\r", "\n")):
                    prog = m.group(0)
                a = b = 0
                if prog:
                    m = PROG.search(prog)
                    a, b = int(m.group(1).replace(",", "")), int(m.group(2).replace(",", ""))
                rate = RATE.search(prog or "")
                eta = ETA.search(prog or "")
                chain, p_ = 0, j.get("parent")
                while p_ and p_ in byid:
                    chain += 1
                    p_ = byid[p_].get("parent")
                ooms = sum(1 for at in j.get("attempts", []) if at.get("class") in ("oom", "host_oom"))
                r["job"] = {"id": j["id"], "scroll": scroll, "tag": tag, "rung": j.get("rung_name"), "z": [j["z0"], j["z1"]], "step": a, "steps": b,
                            "rate": rate.group(1) if rate else None, "eta": eta.group(1).strip() if eta else None, "descents": chain, "ooms": ooms,
                            "no_progress_line": prog is None, "prov": j.get("provenance")}
        rows.append(r)
    snap["gpu_rows"] = rows
    # log tails
    lp = log_path or Path(os.environ.get("ROUTEB_LOG") or (H / "box8.log"))
    snap["log_path"] = str(lp)
    pl_ = tail_text(H / "prefetch.log", 60000).splitlines()
    snap["log_tail"] = sorted(tail_text(lp, 250000).splitlines() + [x for x in pl_ if TS.match(x)], key=lambda x: (TS.match(x).groups()[:3] if TS.match(x) else ("00", "00", "00")))
    snap["env"] = readj(D / "env_installed.json", {}) or {}
    # fetch rows: GB done / total, MB/s, ETA from the progress lines of the fetch log (timestamps give the rate)
    fetch = {}
    for line in snap["log_tail"]:
        m = TS.match(line)
        if not m or m.group(4) != "fetch":
            continue
        txt = m.group(5)
        sm = SUMRX.match(txt)
        if sm:
            fetch.setdefault(sm.group(1), {"tracks": 0.0, "las": {}, "pts": []})["sum"] = {
                "gb": float(sm.group(2)), "tot": float(sm.group(3) or 0), "mbs": int(sm.group(4)), "ops": int(sm.group(5)),
                "eta": sm.group(6), "done": bool(sm.group(7)), "det": sm.group(8) or ""}
            continue
        mo = re.search(r"(PHerc(?:Paris)?\d+[A-Z]?)", txt)
        if not mo:
            continue
        sc = mo.group(1)
        f = fetch.setdefault(sc, {"tracks": 0.0, "las": {}, "pts": []})
        t = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
        mb = BLK.search(txt)
        ob = OBJ.search(txt)
        if mb:
            f["tracks"] = float(mb.group(4))
        if ob:
            fld = "nx" if "_nx" in ob.group(1) else "ny" if "_ny" in ob.group(1) else "grad" if "grad_mag" in ob.group(1) else "las"
            f["las"][fld] = float(ob.group(4))
            f["obj"] = f"{fld} {ob.group(2)}/{ob.group(3)} obj"
        if mb or ob:
            f["pts"].append((t, f["tracks"] + sum(f["las"].values())))
    ev = []
    try:
        ev = [json.loads(l) for l in (D / "events.jsonl").read_text().splitlines() if l.strip()]
    except OSError:
        pass
    snap["events"] = ev
    blocked = {e["scroll"]: e for e in ev if e.get("kind") == "stage_blocked"}
    staged = {e["scroll"] for e in ev if e.get("kind") == "stage"}
    frows = []
    for sc, s_ in scrolls.items():
        if s_["fetch_failed"]:
            state = "FAILED"
        elif s_["fetched"]:
            state = "done"
        elif sc in staged:
            state = "fetching"
        elif sc in blocked:
            state = "BLOCKED by disk"
        else:
            state = "waiting"
        f = fetch.get(sc)
        detail, rate = "", None
        if state == "fetching" and f and f.get("sum"):
            q = f["sum"]
            detail = f"{q['gb']:.1f}/{q['tot']:.1f} GB {q['mbs']} MB/s {q['ops']} obj/s ETA {q['eta'] or '?'} min {q['det']}"
            rate = q["mbs"]
        elif state == "fetching" and f and f["pts"]:
            done = f["pts"][-1][1]
            total = fetch_total_gb(sc)
            old = next((p_ for p_ in reversed(f["pts"][:-1]) if f["pts"][-1][0] - p_[0] >= 10), None)
            if old and f["pts"][-1][0] > old[0]:
                rate = (done - old[1]) * 1000.0 / (f["pts"][-1][0] - old[0])
            eta = (total - done) * 1000.0 / rate / 60.0 if rate and rate > 0 and total else None
            detail = f"{done:.1f}/{total:.1f} GB  {('%.0f' % rate) if rate else '?'} MB/s  ETA {('%.0f min' % eta) if eta is not None else '?'}  {f.get('obj', '')}"
        frows.append({"scroll": sc, "state": state, "detail": detail, "rate": rate})
    snap["fetch_rows"] = frows
    # Route A
    ra = {"slots": None, "segments": 0, "area": 0.0, "events": []}
    for e in ev:
        if e.get("kind") in ("routea_start", "routea_rescale"):
            ra["slots"] = e.get("slots")
        if e.get("kind") in ("routea_exit", "routea_skip"):
            ra["events"].append(e.get("kind") + ": " + str(e.get("rc", e.get("why", ""))))
    ra["segments"] = len(list((H / "out" / "routeA").glob("*/DONE")))
    for sd in (H / "routeA_work" / "export").glob("*/*"):
        ra["area"] += newest_area(sd)
    snap["routea"] = ra
    # payload
    units = []
    for d in sorted((H / "out").glob("*/*/DONE")):
        u = d.parent
        meta = readj(u / "PAYLOAD.json", {}) or {}
        units.append({"unit": f"{u.parent.name}/{u.name}", "gb": meta.get("total_bytes", 0) / 1e9, "files": len(meta.get("files", [])), "pulled": (u / "PULLED.json").exists()})
    snap["units"] = units
    # host
    snap["host"] = {"name": socket.gethostname()}
    try:
        du = shutil.disk_usage(H if H.exists() else H.parent)
        snap["disk"] = (du.used / 1e9, du.free / 1e9, du.total / 1e9)
    except OSError:
        snap["disk"] = (0, 0, 0)
    snap["ram"] = meminfo()
    t = tree or Path(__file__).resolve().parents[1]
    snap["commit"] = sh(["git", "-C", str(t), "rev-parse", "--short", "HEAD"]).strip() or "?"
    snap["branch"] = sh(["git", "-C", str(t), "rev-parse", "--abbrev-ref", "HEAD"]).strip() or "?"
    snap["pull"] = pull_lines(H, env)
    snap["control"] = sorted(p.name for p in (D / "control").glob("*")) if (D / "control").is_dir() else []
    snap["replan"] = (D / "control" / "PLAN.txt").read_text().splitlines() if (D / "control" / "PLAN.txt").exists() else []
    return snap


def blend(snap: dict, n: int = 16) -> list[tuple[float, str, str]]:
    """Merge box8.log + Route A log + running jobs' logs, time ordered, prefixed; noisy fetch lines are dropped (they are in the fetch rows)."""
    H = Path(snap["home"])
    out = []
    day = time.strftime("%Y-%m-%d", time.gmtime(snap["now"]))

    def tsec(h, m, s):
        return h * 3600 + m * 60 + s

    for line in snap["log_tail"][-400:]:
        m = TS.match(line)
        if not m:
            continue
        tag, txt = m.group(4), m.group(5)
        if tag == "fetch" and not re.search(r"FAIL|ERROR|retry", txt, re.I):
            continue
        pref = {"box8": "[box8]", "budget": "[budget]", "routeA": "[routeA]", "link": "[link]", "alarm": "[ALARM]", "control": "[ctl]", "replan": "[replan]", "fetch": "[fetch]",
                "stage": "[stage]", "disk": "[disk]"}.get(tag, f"[{tag}]")
        fit = re.match(r"(PHerc\w+/\S+): ", txt)
        if tag == "box8" and fit:
            j = next((x for x in snap["jobs"] if x["id"] == fit.group(1)), None)
            if j and j.get("gpu") is not None:
                pref = f"[fit gpu{j['gpu']} {j['scroll']}]"
        out.append((tsec(int(m.group(1)), int(m.group(2)), int(m.group(3))), pref, txt))
    ra = tail_text(H / "box8" / "logs" / "routeA.log", 8000).splitlines()[-6:]
    mt = time.localtime(os.path.getmtime(H / "box8" / "logs" / "routeA.log")) if (H / "box8" / "logs" / "routeA.log").exists() else None
    for l in ra:
        if l.strip() and mt:
            out.append((tsec(mt.tm_hour, mt.tm_min, mt.tm_sec), "[routeA]", l.strip().replace("[routeA] ", "")[:160]))
    for r in snap["gpu_rows"]:
        j = r["job"]
        if not j:
            continue
        for l in [x for x in tail_text(H / "box8" / "logs" / (j["id"].replace("/", "__") + ".log"), 6000).splitlines() if x.strip() and "PROGRESS" not in x][-2:]:
            mt = time.localtime(os.path.getmtime(H / "box8" / "logs" / (j["id"].replace("/", "__") + ".log")))
            out.append((tsec(mt.tm_hour, mt.tm_min, mt.tm_sec), f"[fit gpu{r['idx']} {j['scroll']}]", l.strip()[:160]))
    out.sort(key=lambda x: x[0])
    return out[-n:]


def alerts(snap: dict) -> list[tuple[str, str]]:
    """(severity, text) list: scheduler alarms + grouped failures + derived checks."""
    out = []
    for a in snap["alerts_file"]:
        age = snap["now"] - a.get("since", snap["now"])
        out.append((a.get("sev", "red"), f"{a['text']}   [{age:.0f} s]"))
    fails = {}
    for e in snap["events"]:
        if e.get("kind") == "failure":
            k = f"{e.get('cls')} -> {e.get('decision')}"
            fails[k] = fails.get(k, 0) + 1
        elif e.get("kind") in ("unit_failed", "payload_failed", "fetch_failed"):
            fails[e["kind"]] = fails.get(e["kind"], 0) + 1
    tot_jobs = max(1, len(snap["jobs"]))
    for k, c in sorted(fails.items(), key=lambda x: -x[1]):
        out.append(("red" if c / tot_jobs > 0.05 or "terminal" in k or "fail" in k else "yellow", f"failures: {k} x{c} ({c / tot_jobs:.0%} of {tot_jobs} jobs)"))
    b = snap["budget"]
    if b and b.get("spent") is not None and b.get("hard") and b["spent"] >= 0.8 * b["soft"]:
        out.append(("yellow", f"budget: ${b['spent']:.2f} spent = {b['spent'] / b['soft']:.0%} of soft ${b['soft']:g}"))
    if snap["status"].get("stop"):
        out.append(("red", f"STOPPED: {snap['status']['stop']}"))
    for e in snap["events"][-200:]:
        if e.get("kind") == "stage_blocked" and not any("STAGING" in t for _s, t in out):
            out.append(("red", f"STAGING BLOCKED by disk: {e.get('scroll')} needs {e.get('need_gb', 0):.0f} GB (held {e.get('held_gb', 0):.0f} GB): nothing pulled yet?"))
    return out


def c(txt, col, color):
    return f"{col}{txt}{RST}" if color else txt


def bar(a, b, w=16):
    f = 0 if not b else int(w * min(1.0, a / b))
    return "#" * f + "." * (w - f)


def hms(h):
    return f"{int(h)}h{int((h - int(h)) * 60):02d}m"


def render(snap: dict, color: bool = False, stream_n: int = 16) -> str:
    L = []
    P = L.append
    b = snap["budget"]
    st = snap["status"]
    up = f"{hms(b['hours'])} since the REAL box start" if b.get("hours") is not None else "box8 not started yet"
    P(c(f"== routeB box8 @ {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(snap['now']))}   host {snap['host']['name']}   {snap['branch']}@{snap['commit']}   {up}", BLD, color))
    if b:
        pc = b["spent"] / b["soft"] if b.get("soft") else 0
        col = RED if b["spent"] >= b["hard"] or b.get("hard_stop") else YEL if pc >= 0.8 else GRN
        P(c(f"$ spent {b['spent']:.2f}  projected {b['projected_total']:.2f}   soft {b['soft']:g}  hard {b['hard']:g}   rate {b.get('eff_hour_usd', 0):.3f}/h   "
            f"[{bar(b['spent'], b['soft'])}] {pc:.0%} of soft   clock {hms(b['hours'])} of max {b['max_run_hours']:g} h", col, color))
    tr = snap["link_trend"]
    if tr:
        vals = [x.get("data_mbs") for x in tr if x.get("data_mbs") is not None]
        now_v = vals[-1] if vals else None
        arrow = "" if len(vals) < 2 else (" falling" if vals[-1] < 0.8 * vals[0] else " rising" if vals[-1] > 1.2 * vals[0] else " steady")
        P(f"link {('%.0f' % now_v) if now_v is not None else 'n/a'} MB/s (data host){arrow}; last {len(vals)}: " + " ".join(f"{v:.0f}" for v in vals[-8:]) +
          (f"   verdict {snap['link_first'].get('verdict', {}).get('verdict', '?')}" if snap["link_first"] else ""))
    else:
        P("link: no measurement yet")
    u, f_, t_ = snap["disk"]
    av, tot = snap["ram"]
    P(f"disk used {u:.0f} / {t_:.0f} GB (free {f_:.0f})   RAM available {av:.0f} / {tot:.0f} GB   jobs {st.get('jobs', {})}")
    P("")
    P("GPU  util  VRAM        job (scroll/stripe)            step           it/s   ETA        rung      ladder")
    for r in snap["gpu_rows"]:
        j = r["job"]
        vram = f"{r['mem'] / 1024:5.1f}/{r['total'] / 1024:.0f}G"
        if j:
            stp = f"{j['step']}/{j['steps']}" if j["steps"] else "starting"
            ld = f"desc {j['descents']} oom {j['ooms']}" if j["descents"] or j["ooms"] else "-"
            if j.get("prov"):
                ld = (ld if ld != "-" else "") + f" [{j['prov']}]"
            P(f"{r['idx']:>3}  {r['util']:>3.0f}%  {vram:<10}  {j['id']:<30} {stp:<14} {j['rate'] or '-':<6} {j['eta'] or '-':<10} {j['rung'] or '-':<9} {ld}")
        else:
            idle = "IDLE" if r["util"] < 5 else "(non-box8 load)"
            P(c(f"{r['idx']:>3}  {r['util']:>3.0f}%  {vram:<10}  -- {idle} --", YEL if idle == "IDLE" else DIM, color))
    P("")
    P("FETCH")
    if snap["fetch_rows"]:
        for fr in snap["fetch_rows"]:
            col = RED if fr["state"] in ("FAILED", "BLOCKED by disk") else GRN if fr["state"] == "done" else YEL if fr["state"] == "fetching" else DIM
            P(c(f"  {fr['scroll']:<10} {fr['state']:<16} {fr['detail']}", col, color))
    else:
        P("  (no scrolls yet)")
    ra = snap["routea"]
    P(f"ROUTE A  slots {ra['slots'] if ra['slots'] is not None else '-'}   segments published {ra['segments']}   area grown so far {ra['area']:.1f} cm2   {'; '.join(ra['events'])}")
    P("")
    npull = sum(1 for x in snap["units"] if x["pulled"])
    P(f"PAYLOAD  {len(snap['units'])} unit(s) DONE, {sum(x['gb'] for x in snap['units']):.2f} GB, {npull} pulled")
    for x in snap["units"][-8:]:
        P(f"  {x['unit']:<34} {x['gb']:>6.3f} GB {x['files']:>5} files  pulled: {'yes' if x['pulled'] else 'no'}")
    P("")
    al = alerts(snap)
    P(c(f"ALERTS ({len(al)})", RED if any(s == 'red' for s, _ in al) else BLD, color))
    for sev, txt in al:
        P(c(f"  ! {txt}", RED if sev == "red" else YEL, color))
    if not al:
        P(c("  none", GRN, color))
    P("")
    P("PULL from your machine (paste each line):")
    for l in snap["pull"]:
        P("  " + l)
    P("")
    P("EVENTS")
    for _t, pref, txt in blend(snap, stream_n):
        col = next((cc for rx, cc in SEV if rx.search(txt)), None)
        P(c(f"  {pref:<28} {txt[:150]}", {"red": RED, "yellow": YEL, "green": GRN}.get(col, ""), color and bool(col)))
    return "\n".join(L)



W = 100


def clip(t: str, w: int = W) -> str:
    return t if len(t) <= w else t[: w - 1] + "~"


def render_compact(snap: dict, color: bool = False, brief: bool = False) -> str:
    """ONE SCREEN: <= 40 lines x 100 columns (brief: <= 28, no event stream).  One line per GPU, one per fetching scroll (GB done/total, MB/s, objects/s, ETA),
    one Route A + payload line, <= 3 alert lines, the 5-line pull block, 6 event lines."""
    L = []
    b, st = snap["budget"], snap["status"]
    tm = time.strftime("%H:%M:%S", time.localtime(snap["now"]))
    L.append(c(clip(f"== routeB box8 {tm} {snap['host']['name']} {snap['branch']}@{snap['commit']}  clock {hms(b['hours']) if b.get('hours') is not None else '-'}"), BLD, color))
    if b:
        pc = b["spent"] / b["soft"] if b.get("soft") else 0
        col = RED if b["spent"] >= b["hard"] or b.get("hard_stop") else YEL if pc >= 0.8 else GRN
        j = st.get("jobs", {})
        L.append(c(clip(f"$ {b['spent']:.2f} spent, {b['projected_total']:.2f} proj | soft {b['soft']:g} hard {b['hard']:g} | {pc:.0%} | "
                        f"jobs run {j.get('running', 0)} done {j.get('done', 0)} pend {j.get('pending', 0)} fail {j.get('failed', 0)}"), col, color))
    tr = [x.get("data_mbs") for x in snap["link_trend"] if x.get("data_mbs") is not None]
    u, f_, t_ = snap["disk"]
    av, tot = snap["ram"]
    en = snap.get("env") or {}
    L.append(clip(f"link {('%.0f' % tr[-1]) if tr else '?'} MB/s{' (' + ' '.join('%.0f' % v for v in tr[-4:]) + ')' if len(tr) > 1 else ''} | disk {u:.0f}/{t_:.0f} GB"
                  f" | RAM {av:.0f}/{tot:.0f} | torch {en.get('torch', '?')}"))
    idle_s = {}
    for a_ in snap["alerts_file"]:
        mm = re.match(r"idle:gpu(\d+)", a_.get("key", ""))
        if mm:
            idle_s[mm.group(1)] = snap["now"] - a_.get("since", snap["now"])
    for r in snap["gpu_rows"]:
        j = r["job"]
        head = f"g{r['idx']} {r['util']:>3.0f}% {r['mem'] / 1024:4.1f}/{r['total'] / 1024:.0f}G"
        if j:
            stp = f"{j['step']}/{j['steps']}" if j["steps"] else "start"
            ld = ("d%d" % j["descents"] if j["descents"] else "") + ("o%d" % j["ooms"] if j["ooms"] else "") + ("+" + j["prov"][:4] if j.get("prov") else "")
            L.append(clip(f"{head} {j['id'][:26]:<26} {stp:>11} {(j['rate'] or '-') + 'it/s':>8} {(j['eta'] or '-'):<8} {j['rung'] or '-':<8} {ld}"))
        else:
            idle = f" IDLE {idle_s[r['idx']] / 60:.0f}m" if r["idx"] in idle_s else (" idle" if r["util"] < 5 else " (other load)")
            L.append(c(clip(f"{head}{idle}"), RED if r["idx"] in idle_s else YEL if r["util"] < 5 else DIM, color))
    fr = snap["fetch_rows"]
    act = [x for x in fr if x["state"] not in ("done",)]
    for x in act[:4]:
        col = RED if x["state"] in ("FAILED", "BLOCKED by disk") else YEL if x["state"] == "fetching" else DIM
        L.append(c(clip(f"f {x['scroll']:<10} {x['state']:<9} {x['detail']}"), col, color))
    if len(act) > 4 or any(x["state"] == "done" for x in fr):
        L.append(clip(f"f +{max(0, len(act) - 4)} more | {sum(1 for x in fr if x['state'] == 'done')} scroll(s) fully staged"))
    ra = snap["routea"]
    units = snap["units"]
    L.append(clip(f"A: {ra['slots'] if ra['slots'] is not None else '-'} slots, {ra['segments']} seg, {ra['area']:.1f} cm2 {'; '.join(ra['events'])[:30]} | "
                  f"P: {len(units)} units {sum(x['gb'] for x in units):.2f} GB, {sum(1 for x in units if x['pulled'])} pulled"))
    al = alerts(snap)
    for sev, txt in al[:3]:
        L.append(c(clip("! " + txt), RED if sev == "red" else YEL, color))
    if len(al) > 3:
        L.append(c(clip(f"! +{len(al) - 3} more alert(s) (routeB_watch.sh --once --full)"), YEL, color))
    L.append("PULL (your machine):")
    for l in snap["pull"][:5]:
        L.append(clip("  " + l))
    if not brief:
        L.append("EVENTS")
        for _t, pref, txt in blend(snap, 6):
            colr = next((cc for rx, cc in SEV if rx.search(txt)), None)
            L.append(c(clip(f"  {pref[:18]:<18} {txt}"), {"red": RED, "yellow": YEL, "green": GRN}.get(colr, ""), color and bool(colr)))
    return "\n".join(L)


def snapshot_text(snap: dict) -> str:
    """The single block to paste back for diagnosis: dashboard + raw STATUS/budget/link/control/REPLAN + log tail + df/nvidia-smi."""
    H = snap["home"]
    parts = ["=== routeB snapshot " + time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(snap["now"])) + f" home={H} ===", render(snap, color=False, stream_n=25), ""]
    parts.append("--- STATUS.json ---")
    parts.append(json.dumps(snap["status"], indent=1)[:2500])
    parts.append("--- link first.json / trend ---")
    parts.append(json.dumps({"first": snap["link_first"].get("verdict"), "measured": snap["link_first"].get("measured"), "trend": snap["link_trend"]})[:1500])
    parts.append("--- control dir / latest REPLAN ---")
    parts.append(" ".join(snap["control"]) or "(empty)")
    parts += snap["replan"][:14]
    parts.append("--- event counts ---")
    cnt = {}
    for e in snap["events"]:
        cnt[e.get("kind")] = cnt.get(e.get("kind"), 0) + 1
    parts.append(json.dumps(cnt))
    parts.append("--- last 20 lines of " + snap["log_path"] + " (fetch noise removed) ---")
    parts += [l[:200] for l in snap["log_tail"] if " fetch: " not in l][-20:]
    parts.append("--- nvidia-smi ---")
    parts.append(sh(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader"]).strip() or "(none)")
    parts.append("--- tmux ---")
    parts.append(sh(["tmux", "ls"]).strip() or "(no tmux sessions)")
    parts.append("=== end snapshot ===")
    return "\n".join(parts)


def ctl_status(home: Path) -> str:
    """What routeB_ctl.sh status prints: control files, status line, latest REPLAN, recent control events."""
    h = Path(home)
    c = h / "box8" / "control"
    drain = sorted(x.name[6:] for x in c.glob("drain.*")) if c.is_dir() else []
    gf = (c / "gpus").read_text().strip() if (c / "gpus").exists() else None
    flags = " ".join(n for n in ("STOP", "PAUSE") if (c / n).exists())
    L = [f"control dir: {c}", f"  gpus file : {gf!r} | drain: {drain} | {flags or 'no STOP/PAUSE'}" + (" | HARD STOP file present!" if (h / "box8" / "STOP").exists() else "")]
    st = readj(h / "out" / "STATUS.json")
    if st:
        b = st.get("budget", {})
        L.append(f"status @ {st.get('t')}: jobs {st.get('jobs')} busy GPUs {st.get('busy_gpus')}; spent ${b.get('spent', 0):.2f} projected ${b.get('projected_total', 0):.2f} "
                 f"(soft {b.get('soft')}, hard {b.get('hard')}) at {b.get('hours', 0):.2f} h; stop={st.get('stop')}")
        L.append(f"  scrolls: {st.get('scrolls')}")
    else:
        L.append("no out/STATUS.json yet")
    plan = (c / "PLAN.txt").read_text().strip() if (c / "PLAN.txt").exists() else None
    L.append("latest REPLAN:\n  " + (plan.replace("\n", "\n  ") if plan else "(none yet)"))
    try:
        rows = [json.loads(l) for l in (h / "box8" / "events.jsonl").read_text().splitlines() if any(k in l for k in ('"control"', '"replan"', "ctl_kill", '"alarm"'))]
        for r in rows[-6:]:
            L.append(f"  event {r.get('t')} {r.get('kind')} {r.get('what') or r.get('reason') or r.get('job') or r.get('text') or ''}"[:200])
    except OSError:
        pass
    return "\n".join(L)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="routeB_watch.sh")
    ap.add_argument("--home", default=os.environ.get("ROUTEB_HOME"))
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--plain", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--ctl-status", action="store_true")
    ap.add_argument("--pull", action="store_true", help="print the full pull block for the USER'S machine (direct + vast.ai proxy) and exit")
    ap.add_argument("--brief", action="store_true", help="compact AND without the event stream (<= 28 lines)")
    ap.add_argument("--full", action="store_true", help="the long dashboard / the long diagnostic snapshot (default is ONE SCREEN: <= 40 lines x 100 columns)")
    a = ap.parse_args(argv)
    if a.pull:
        print("\n".join(pull_lines(Path(a.home or os.environ.get('ROUTEB_HOME') or '/workspace/routeB'), full=True)))
        return 0
    if not a.home:
        print("routeB_watch: set ROUTEB_HOME or pass --home", file=sys.stderr)
        return 2
    H = Path(a.home)
    if a.ctl_status:
        print(ctl_status(H))
        return 0
    if a.once:
        sn = collect(H)
        print(snapshot_text(sn) if a.full else render_compact(sn, color=False, brief=a.brief))
        return 0
    tty = sys.stdout.isatty() and not a.plain
    try:
        while True:
            sn_ = collect(H)
            txt = render(sn_, color=tty) if a.full else render_compact(sn_, color=tty, brief=a.brief)
            if tty:
                sys.stdout.write("\033[H\033[2J" + txt + f"\n\n{DIM}watching every {a.interval:g} s; Ctrl-C detaches the watcher only (the run continues in tmux session routeb){RST}\n")
            else:
                sys.stdout.write("----- " + time.strftime("%H:%M:%S") + " -----\n" + txt + "\n")
            sys.stdout.flush()
            time.sleep(a.interval)
    except KeyboardInterrupt:
        print("\nwatcher detached; the run continues (reattach: routeB_watch.sh; raw console: tmux attach -t routeb)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
