"""Pull ALL finished payloads FROM the rental box TO here with ONE rsync of ONE directory.  Run on OUR side; the box never connects to us and holds no credentials for our network.

  python3 pull_box8.py --host root@1.2.3.4 [--port 22 -i ~/.ssh/key] --remote-home <box ROUTEB_HOME> --dest ./out [--poll 120] [--final] [--once]

The box keeps everything downloadable in <ROUTEB_HOME>/out/ :  out/<scroll>/<tag>/{files, PAYLOAD.json (size+md5 of every file), DONE}  per finished stripe/job (written the moment
that job finishes), out/routeA/<scroll>__<seg>/ likewise for Route A segments, out/<scroll>/{SCROLL.json,DONE} at scroll end, out/STATUS.json, out/ALLDONE.json.
Loop:  rsync -aH --partial --append-verify --exclude '.tmp_*' box:out/ dest/   (resumable, appends only the missing tail of a partly copied file and verifies the overlap)
       -> for every unit whose DONE arrived: recompute md5 of EVERY file against its PAYLOAD.json -> write <unit>/PULLED.json locally, append <dest>/pull_ledger.jsonl, and write
       the unit's PULLED.json on the box (the box's budget governor accounts the egress from it and then deletes the scroll's inputs).
A unit that fails verification is re-synced once with --checksum, then reported LOUDLY and retried next pass; it is never marked pulled.
--final: keep polling until the box wrote out/ALLDONE.json and every DONE unit is pulled, print the box's budget status, exit 0 (1 if anything is unpulled).  --once: single pass.
The same thing with no python at all (checksum verification afterwards by `python3 pull_box8.py --verify-only --dest ./out`):
  while :; do rsync -aH --partial --append-verify --exclude '.tmp_*' -e 'ssh -p PORT' USER@HOST:<ROUTEB_HOME>/out/ ./out/; sleep 60; done
Needs: ssh and rsync locally AND on the box.  Costs only the egress it causes (payloads are checkpoint-free: ~0.1-1 GB per stripe).
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

    def rsync(self, dst: Path, checksum=False, sub: str = "") -> int:
        e = "ssh " + " ".join(shlex.quote(x) for x in self.ssh_opts)
        cmd = ["rsync", "-aH", "--partial", "--append-verify", "--info=stats1", "--exclude", ".tmp_*", "--exclude", "PULLED.json", "-e", e]
        if checksum:
            cmd.append("--checksum")
        cmd += [f"{self.a.host}:{self.a.remote_home}/out/{sub}", str(dst / sub) + ("" if sub else "/")]
        dst.mkdir(parents=True, exist_ok=True)
        (dst / sub).mkdir(parents=True, exist_ok=True)
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


def local_units(dest: Path):
    return sorted(p.parent for p in dest.glob("*/*/DONE"))


def verify_units(dest: Path, a, box: "Box | None", ledger: Path) -> tuple[int, int, list[str]]:
    pulled = new = 0
    problems = []
    for u in local_units(dest):
        name = f"{u.parent.name}/{u.name}"
        if (u / "PULLED.json").exists():
            pulled += 1
            continue
        try:
            meta = json.loads((u / "PAYLOAD.json").read_text())
        except (OSError, ValueError) as e:
            say(f"{name}: DONE but PAYLOAD.json unreadable ({e}); will retry")
            problems.append(name)
            continue
        t0 = time.time()
        bad = verify(u, meta)
        if bad and box is not None:
            say(f"{name}: {len(bad)} file(s) failed verification (first: {bad[0]}); re-syncing the unit with --checksum")
            box.rsync(dest, checksum=True, sub=f"{u.parent.name}/{u.name}/")
            bad = verify(u, meta)
        got = sum((u / e["path"]).stat().st_size for e in meta["files"] if (u / e["path"]).exists())
        row = {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "unit": name, "status": meta.get("status"), "files": len(meta["files"]), "bytes_manifest": meta["total_bytes"],
               "bytes_local": got, "verify_s": round(time.time() - t0, 1), "md5_failures": len(bad)}
        with open(ledger, "a") as f:
            f.write(json.dumps(row) + "\n")
        if bad:
            say(f"{name}: *** VERIFICATION FAILED: {bad[:5]} -- NOT marked pulled ***")
            problems.append(name)
            continue
        (u / "PULLED.json").write_text(json.dumps({**row, "md5_verified_files": len(meta["files"])}, indent=1))
        if box is not None:
            box.ssh(f"cat > {shlex.quote(a.remote_home)}/out/{shlex.quote(name)}/PULLED.json", inp=json.dumps({"bytes": got, "t": row["t"]}))
        say(f"{name}: PULLED+VERIFIED {got / 1e9:.3f} GB, md5 of {len(meta['files'])} files OK")
        new += 1
        pulled += 1
    return pulled, new, problems


def pass_once(box: Box, a, ledger: Path) -> tuple[int, int, list[str]]:
    dest = Path(a.dest)
    t0 = time.time()
    r = box.rsync(dest)
    problems = []
    if r not in (0, 23, 24):
        say(f"RSYNC FAILED rc={r} (resumes next pass)")
        problems.append("rsync")
    say(f"rsync of out/ took {time.time() - t0:.0f} s")
    pulled, new, pr = verify_units(dest, a, box, ledger)
    rc, out, _ = box.ssh(f"ls -d {shlex.quote(a.remote_home)}/out/*/*/DONE 2>/dev/null")
    remote = sorted(str(Path(x).parent.relative_to(Path(a.remote_home) / "out")) for x in out.split())
    missing = [u for u in remote if not (dest / u / "DONE").exists()]
    return pulled, new, problems + pr + [f"missing:{u}" for u in missing]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="", help="user@host of the rental box")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("-i", "--identity", default=None)
    ap.add_argument("--remote-home", default="", help="the box's ROUTEB_HOME (contains box8/)")
    ap.add_argument("--remote-tree", default=None, help="the box's branch checkout (for the budget status line)")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--poll", type=float, default=120)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--rsync-timeout", type=float, default=6 * 3600)
    ap.add_argument("--verify-only", action="store_true", help="no ssh: just md5-verify every DONE unit already under --dest")
    a = ap.parse_args(argv)
    Path(a.dest).mkdir(parents=True, exist_ok=True)
    ledger = Path(a.dest) / "pull_ledger.jsonl"
    if a.verify_only:
        pulled, new, problems = verify_units(Path(a.dest), a, None, ledger)
        say(f"verify-only: {pulled} unit(s) verified ({new} new), problems: {problems or 'none'}")
        return 1 if problems else 0
    box = Box(a)
    while True:
        pulled, new, problems = pass_once(box, a, ledger)
        say(f"pass: {pulled} unit(s) pulled in total, {new} new, problems: {problems or 'none'}")
        if a.remote_tree:
            rc, out, _ = box.ssh(f"python3 {shlex.quote(a.remote_tree)}/deploy_common/budget.py status {shlex.quote(a.remote_home)}/box8/budget")
            if rc == 0:
                st = json.loads(out)
                say(f"BOX BUDGET: spent ${st['spent']:.2f}, projected ${st['projected_total']:.2f} (soft ${st['soft']}, hard ${st['hard']}), hard_stop={st['hard_stop']}")
        if a.once:
            return 1 if problems else 0
        if a.final and not problems and (Path(a.dest) / "ALLDONE.json").exists():
            say("ALLDONE on the box and every unit pulled+verified")
            return 0
        time.sleep(a.poll)


if __name__ == "__main__":
    sys.exit(main())
