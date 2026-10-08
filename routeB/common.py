"""Shared plumbing: paths, logging, stage markers, subprocess runner.  No private defaults: everything is relative to the branch root or ROUTEB_HOME."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # branch root (contains routeB_run.sh)
SCROLLS = ROOT / "routeB" / "scrolls"
UP_LO, UP_HI = 4500, 17500                          # upstream tracks cover z [4500, 17500) only


class StageError(RuntimeError):
    """A stage failed; the message names the stage and the cause.  The CLI prints it and continues with the next scroll."""


def home() -> Path:
    return Path(os.environ.get("ROUTEB_HOME") or (ROOT / "routeB_work")).resolve()


def env_python() -> str:
    p = os.environ.get("ROUTEB_PYTHON") or str(home() / "env" / "bin" / "python")
    return p


def spec(scroll: str) -> dict:
    p = SCROLLS / f"{scroll}.json"
    if not p.exists():
        known = ", ".join(sorted(x.stem for x in SCROLLS.glob("*.json")))
        raise StageError(f"spec: unknown scroll {scroll!r}; known: {known}")
    return json.loads(p.read_text())


def say(msg: str, tag: str = "routeB") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {tag}: {msg}", flush=True)


def marker(d: Path, name: str) -> Path:
    return d / f".done.{name}.json"


def is_done(d: Path, name: str) -> bool:
    return marker(d, name).exists()


def mark_done(d: Path, name: str, **info) -> None:
    d.mkdir(parents=True, exist_ok=True)
    marker(d, name).write_text(json.dumps({"stage": name, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **info}, indent=1))


def run(cmd, log: Path | None = None, env: dict | None = None, cwd: str | Path | None = None, timeout: float | None = None,
        check_msg: str | None = None) -> int:
    """Run a command, tee its output to `log`.  Returns the exit code (the caller decides what failure means)."""
    e = dict(os.environ)
    if env:
        e.update({k: str(v) for k, v in env.items()})
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log, "ab") if log else None
    try:
        p = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=e, cwd=str(cwd) if cwd else None)
        t0 = time.time()
        for line in iter(p.stdout.readline, b""):
            if fh:
                fh.write(line)
                fh.flush()
            if os.environ.get("ROUTEB_VERBOSE", "0") == "1":
                sys.stdout.write(line.decode("utf-8", "replace"))
            if timeout and time.time() - t0 > timeout:
                p.kill()
                break
        p.wait()
        return p.returncode
    finally:
        if fh:
            fh.close()


def tail(log: Path, n: int = 25) -> str:
    try:
        return "\n".join(log.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return "(no log)"


def stripes(width: str | int, z0: int = UP_LO, z1: int = UP_HI, overlap: int = 1500):
    """[(tag, z_begin, z_end)] windows tiling [z0, z1).  'full' -> one window.  width W: starts every W-overlap, last clipped to z1
    (4500 -> 4500,7500,10500,13500 = the production quarters q4500..q13500; the last is 4000 wide)."""
    if str(width) == "full" or int(width) >= z1 - z0:
        return [("full", z0, z1)]
    W = int(width)
    out, s = [], z0
    while s < z1:
        e = min(s + W, z1)
        out.append((f"s{s}", s, e))
        if e >= z1:
            break
        s += W - overlap
    return out
