#!/usr/bin/env python3
"""On-box probe (stdlib only). Prints ONE JSON object about this box; the watchdog runs it over ssh. Read-only.
Usage: box_probe.py [--workdir /data/cloud-grow/run1] [--root /data/cloud-grow] [--kit /opt/vc_kit/bin] [--pins pins.json]"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sqlite3
import time


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def procs(needle):
    n = 0
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                c = open(f"/proc/{d}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
            except OSError:
                continue
            if needle in c and "box_probe" not in c:
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="/data/cloud-grow/run1")
    ap.add_argument("--root", default="/data/cloud-grow")
    ap.add_argument("--kit", default="/opt/vc_kit/bin")
    ap.add_argument("--pins", default="")
    a = ap.parse_args()
    o = {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": os.uname().nodename}
    o["loadavg"] = os.getloadavg()[0]
    mem = {}
    for ln in open("/proc/meminfo"):
        k, v = ln.split(":")[0], int(ln.split()[1])
        mem[k] = v
    o["ram_avail_gb"] = round(mem.get("MemAvailable", 0) / 1048576, 1)
    o["ram_total_gb"] = round(mem.get("MemTotal", 0) / 1048576, 1)
    o["disk_free_gb"] = round(shutil.disk_usage(a.root if os.path.isdir(a.root) else "/").free / 1e9, 1)
    o["runner_loop_alive"] = procs("cg-box-loop") > 0
    o["tracer_procs"] = procs("vc_grow_seg_from_seed")
    o["stop_file"] = os.path.exists(os.path.join(a.root, "STOP"))
    o["bootstrap_failed"] = os.path.exists(os.path.join(a.root, "BOOTSTRAP_FAILED"))
    # tool md5 drift vs pins
    drift = []
    try:
        pins = json.load(open(a.pins))["tools"] if a.pins else {}
    except (OSError, ValueError):
        pins = {}
    for t, p in pins.items():
        f = os.path.join(a.kit, t)
        if not os.path.isfile(f):
            drift.append(f"{t}: missing")
            continue
        m = md5(f)
        if "md5" in p and m != p["md5"]:
            drift.append(f"{t}: {m} != {p['md5']}")
        elif "md5_prefix" in p and not m.startswith(p["md5_prefix"]):
            drift.append(f"{t}: {m[:8]} != prefix {p['md5_prefix']}")
    o["tool_pins_checked"] = len(pins)
    o["tool_md5_drift"] = drift
    # segment/round counters from the file-backed state shim (read-only)
    o["rounds_total"] = o["verified_cm2_now"] = o["verified_cm2_1h_ago"] = None
    o["rounds_last_hour"] = None
    o["segments"] = {}
    db = os.path.join(a.workdir, "state.sqlite")
    if os.path.isfile(db):
        try:
            c = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
            cut = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 3600))
            q = lambda s, *p: c.execute(s, p).fetchall()  # noqa: E731
            o["rounds_total"] = q("SELECT COUNT(*) FROM metric WHERE name='area_cm2'")[0][0]
            o["rounds_last_hour"] = q("SELECT COUNT(*) FROM metric WHERE name='area_cm2' AND replace(replace(recorded_utc,'T',' '),'Z','')>=replace(replace(?,'T',' '),'Z','')", cut)[0][0]
            latest = "SELECT seg, value FROM metric m WHERE name='verified_cm2' AND id=(SELECT MAX(id) FROM metric WHERE seg=m.seg AND name='verified_cm2'%s)"
            o["verified_cm2_now"] = round(sum(v or 0 for _, v in q(latest % "")), 3)
            old = q(latest % " AND replace(replace(recorded_utc,'T',' '),'Z','')<replace(replace(?,'T',' '),'Z','')", cut)
            o["verified_cm2_1h_ago"] = round(sum(v or 0 for _, v in old), 3)
            outs = q("SELECT text_value FROM metric WHERE name='grow_outcome'")
            st = {}
            for (t,) in outs:
                k = (t or "?").split(":")[0]
                st[k] = st.get(k, 0) + 1
            o["segments"] = st
            o["guard_pauses"] = q("SELECT COUNT(*) FROM metric WHERE name='pause_reason'")[0][0]
            o["selfx_unverified_metrics"] = q("SELECT COUNT(*) FROM metric WHERE name='guard_selfx_unverified'")[0][0]
            c.close()
        except sqlite3.Error as e:
            o["db_error"] = f"{type(e).__name__}: {e}"
    else:
        o["db_error"] = "state.sqlite not present yet"
    o["selfx_unverified_markers"] = len(glob.glob(os.path.join(a.workdir, "segments", "*", "*", "*", "selfx_unverified.json"))) + \
        len(glob.glob(os.path.join(a.workdir, "segments", "*", "*", "selfx_unverified.json")))
    al = {}
    ap_ = os.path.join(a.root, "alerts.jsonl")
    if os.path.isfile(ap_):
        for ln in open(ap_, errors="replace"):
            try:
                lv = json.loads(ln).get("level", "?")
            except ValueError:
                lv = "unparsed"
            al[lv] = al.get(lv, 0) + 1
    o["alerts"] = al
    o["tarballs"] = len(glob.glob(os.path.join(a.root, "export", "*.tar.gz")))
    print(json.dumps(o))


if __name__ == "__main__":
    main()
