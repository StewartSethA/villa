#!/usr/bin/env python3
"""Pull finished box8 payloads FROM the rental box TO here.  Run on OUR side; the box never connects to us and holds no credentials for our network.

  python3 pull_box8.py --host root@1.2.3.4 [--port 22 -i ~/.ssh/key] --remote-home /workspace/routeB_work --dest <local dir> [--poll 120] [--final] [--once]

Loop: ssh `ls <remote-home>/box8/payload/*/DONE` -> for each scroll not yet verified locally: read PAYLOAD.json (sizes + md5 of every file), rsync the scroll dir
(--partial, resumable), recompute md5 of EVERY file locally, compare with the manifest, log sizes/seconds/MB/s to <dest>/pull_ledger.jsonl, write
<dest>/<scroll>/PULLED.json, then write <remote>/payload/<scroll>/PULLED.json on the box (the box's budget governor reads it to account the egress).
A scroll whose md5 verification fails is re-synced once with --checksum, then reported LOUDLY and retried on the next poll; it is never marked pulled.
--final: keep polling until the box wrote box8/ALLDONE.json and every DONE payload is pulled, then print the box's budget status and exit 0 (1 if anything unpulled).
--once: single pass.  Without either flag: poll forever.  Rerun any time: verified scrolls are skipped (resumable).
Needs: ssh and rsync locally AND on the box.  Nothing here spends money except the egress it causes (4 USD/TB at the quoted rate): payloads are checkpoints-free.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def say(m):
    print(f"[{time.strftime('%H:%M:%S')}] pull: {m}", flush=True)


class Box:
    def __init__(self, a):
        self.a = a
        o = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", "-o", "StrictHostKeyChecking=accept-new"]
        if a.port:
            o += ["-p", str(a.port)]
        if a.identity:
            o += ["-i", a.identity]
        self.ssh_opts = o

    def ssh(self, cmd: str, timeout=120, inp: str | None = None):
        r = subprocess.run(["ssh", *self.ssh_opts, self.a.host, cmd], capture_output=True, text=True, timeout=timeout, input=inp)
        return r.returncode, r.stdout, r.stderr

    def rsync(self, src_rel: str, dst: Path, checksum=False) -> int:
        e = "ssh " + " ".join(shlex.quote(x) for x in self.ssh_opts)
        cmd = ["rsync", "-a", "--partial", "--info=stats1", "-e", e]
        if checksum:
            cmd.append("--checksum")
        cmd += [f"{self.a.host}:{self.a.remote_home}/box8/payload/{src_rel}/", str(dst) + "/"]
        dst.mkdir(parents=True, exist_ok=True)
        return subprocess.run(cmd, timeout=self.a.rsync_timeout).returncode


def verify(dst: Path, meta: dict, workers=8):
    bad = []

    def one(e):
        p = dst / e["path"]
        if not p.is_file():
            return (e["path"], "missing")
        if p.stat().st_size != e["size"]:
            return (e["path"], f"size {p.stat().st_size} != {e['size']}")
        if md5_file(p) != e["md5"]:
            return (e["path"], "md5 mismatch")
        return None

    with ThreadPoolExecutor(workers) as ex:
        for r in ex.map(one, meta["files"]):
            if r:
                bad.append(r)
    return bad


def pass_once(box: Box, a, ledger: Path) -> tuple[int, int, list[str]]:
    rc, out, err = box.ssh(f"ls -d {shlex.quote(a.remote_home)}/box8/payload/*/DONE 2>/dev/null")
    if rc not in (0, 1, 2):
        say(f"SSH FAILED rc={rc}: {err.strip()[:200]}")
        return 0, 0, ["ssh"]
    scrolls = sorted(Path(x).parent.name for x in out.split())
    pulled = new = 0
    problems = []
    for s in scrolls:
        dst = Path(a.dest) / s
        if (dst / "PULLED.json").exists():
            pulled += 1
            continue
        rc, txt, err = box.ssh(f"cat {shlex.quote(a.remote_home)}/box8/payload/{s}/PAYLOAD.json", timeout=300)
        if rc != 0:
            say(f"{s}: cannot read PAYLOAD.json ({err.strip()[:200]})")
            problems.append(s)
            continue
        meta = json.loads(txt)
        say(f"{s}: {meta['status']}, {len(meta['files'])} files, {meta['total_bytes'] / 1e9:.3f} GB (manifest); pulling")
        t0 = time.time()
        r = box.rsync(s, dst)
        if r != 0:
            say(f"{s}: RSYNC FAILED rc={r} (will resume next poll)")
            problems.append(s)
            continue
        bad = verify(dst, meta)
        if bad:
            say(f"{s}: {len(bad)} file(s) failed verification (first: {bad[0]}); re-syncing with --checksum")
            box.rsync(s, dst, checksum=True)
            bad = verify(dst, meta)
        wall = time.time() - t0
        got = sum((dst / e["path"]).stat().st_size for e in meta["files"] if (dst / e["path"]).exists())
        row = {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "scroll": s, "status": meta["status"], "files": len(meta["files"]), "bytes_manifest": meta["total_bytes"],
               "bytes_local": got, "wall_s": round(wall, 1), "MB_per_s": round(got / 1e6 / max(wall, 1e-6), 2), "md5_failures": len(bad), "failed_intervals": meta.get("failed_intervals")}
        with open(ledger, "a") as f:
            f.write(json.dumps(row) + "\n")
        if bad:
            say(f"{s}: *** VERIFICATION FAILED after re-sync: {bad[:5]} -- NOT marked pulled ***")
            problems.append(s)
            continue
        (dst / "PULLED.json").write_text(json.dumps({**row, "md5_verified_files": len(meta["files"])}, indent=1))
        box.ssh(f"cat > {shlex.quote(a.remote_home)}/box8/payload/{s}/PULLED.json", inp=json.dumps({"bytes": got, "t": row["t"]}))
        say(f"{s}: PULLED+VERIFIED {got / 1e9:.3f} GB in {wall:.0f} s ({row['MB_per_s']} MB/s), md5 of {len(meta['files'])} files OK")
        new += 1
        pulled += 1
    return pulled, new, problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="user@host of the rental box")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("-i", "--identity", default=None)
    ap.add_argument("--remote-home", required=True, help="the box's ROUTEB_HOME (contains box8/)")
    ap.add_argument("--remote-tree", default=None, help="the box's branch checkout (for the budget status line)")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--poll", type=float, default=120)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--rsync-timeout", type=float, default=6 * 3600)
    a = ap.parse_args(argv)
    Path(a.dest).mkdir(parents=True, exist_ok=True)
    ledger = Path(a.dest) / "pull_ledger.jsonl"
    box = Box(a)
    while True:
        pulled, new, problems = pass_once(box, a, ledger)
        say(f"pass: {pulled} scroll(s) pulled in total, {new} new, problems: {problems or 'none'}")
        if a.remote_tree:
            rc, out, _ = box.ssh(f"python3 {shlex.quote(a.remote_tree)}/deploy_common/budget.py status {shlex.quote(a.remote_home)}/box8/budget")
            if rc == 0:
                st = json.loads(out)
                say(f"BOX BUDGET: spent ${st['spent']:.2f}, projected ${st['projected_total']:.2f} (soft ${st['soft']}, hard ${st['hard']}), hard_stop={st['hard_stop']}")
        if a.once:
            return 1 if problems else 0
        if a.final:
            rc, out, _ = box.ssh(f"cat {shlex.quote(a.remote_home)}/box8/ALLDONE.json")
            if rc == 0 and not problems:
                rc2, ls, _ = box.ssh(f"ls -d {shlex.quote(a.remote_home)}/box8/payload/*/DONE 2>/dev/null | wc -l")
                if int(ls.strip() or 0) == pulled:
                    say("ALLDONE on the box and everything pulled+verified: " + out.replace("\n", " ")[:400])
                    return 0
        time.sleep(a.poll)


if __name__ == "__main__":
    sys.exit(main())
