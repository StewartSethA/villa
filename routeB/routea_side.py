"""Route A on the spare CPU cores of a Route B box: guarded grows (routeA_run.sh) beside the GPU fits, publishing finished segments into the same out/ directory.

  slots = physical_cores - 2 x busy_GPUs - reserve   (planner.routea_slots; RAM-checked; --routea-slots overrides; --no-routea disables)
The Route A driver is a separate process tree (its own python env under <home>/routeA_work); a failure there is announced and never touches the Route B fits.
Each finished grow tree work/export/<scroll>/<seg>/ is hardlinked into out/routeA/<scroll>__<seg>/ with an md5 PAYLOAD.json and a DONE marker, exactly like a Route B
unit, so the one rsync of out/ pulls both."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

from .common import ROOT, say


def pick_scrolls(pins_path: Path, budget_gb: float, prefer: list[str]) -> list[tuple[str, float]]:
    """Route A inputs (surface prediction + normal grids, GB per scroll from pins/scrolls.json): the `prefer` scrolls first (in the Route B set), then smallest first,
    until the disk budget is spent."""
    try:
        d = json.loads(Path(pins_path).read_text())["scrolls"]
    except (OSError, ValueError, KeyError):
        return []
    size = {k: (v["prediction_bytes"] + v["grids_bytes"]) / 1e9 for k, v in d.items() if isinstance(v, dict) and "prediction_bytes" in v and "grids_bytes" in v}
    order = [s for s in prefer if s in size] + sorted((s for s in size if s not in prefer), key=lambda s: size[s])
    # smallest first within the preferred set too: more scrolls per GB
    pref = sorted((s for s in order if s in prefer), key=lambda s: size[s])
    rest = [s for s in order if s not in prefer]
    out, used = [], 0.0
    for s in pref + rest:
        if used + size[s] <= budget_gb:
            out.append((s, size[s]))
            used += size[s]
    return out


def command(home: Path, scrolls: list[str], slots: int, hours: float, seeds: int) -> list[str]:
    return [str(ROOT / "routeA_run.sh"), "--scrolls", ",".join(scrolls), "--seeds", str(seeds), "--hours", f"{hours:.2f}", "--workers", str(slots), "--workdir", str(home / "routeA_work")]


def launch(home: Path, cmd: list[str], logp: Path) -> subprocess.Popen:
    logp.parent.mkdir(parents=True, exist_ok=True)
    lf = open(logp, "ab")
    return subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True, cwd=str(ROOT), env=dict(os.environ, PYTHONUNBUFFERED="1"))


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def publish_exports(work: Path, out: Path, seen: set, ev=lambda *a, **k: None) -> int:
    """Publish every finished Route A export tree (export.json present) not yet published.  Returns the number of new units."""
    n = 0
    for exp in sorted(work.glob("export/*/*/export.json")):
        sd = exp.parent
        key = str(sd)
        if key in seen:
            continue
        unit = out / "routeA" / f"{sd.parent.name}__{sd.name}"
        if (unit / "DONE").exists():
            seen.add(key)
            continue
        if time.time() - exp.stat().st_mtime < 20:             # still being written
            continue
        tmp = out / f".tmp_routeA__{sd.parent.name}__{sd.name}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        for src in (p for p in sd.rglob("*") if p.is_file()):
            dst = tmp / src.relative_to(sd)
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        ents = [{"path": str(p.relative_to(tmp)), "size": p.stat().st_size, "md5": _md5(p)} for p in sorted(tmp.rglob("*")) if p.is_file()]
        tot = sum(e["size"] for e in ents)
        (tmp / "PAYLOAD.json").write_text(json.dumps({"route": "A", "scroll": sd.parent.name, "tag": sd.name, "status": "complete", "files": ents, "total_bytes": tot,
                                                       "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
        (tmp / "DONE").write_text(f"complete {len(ents)} files {tot} bytes\n")
        unit.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(unit, ignore_errors=True)
        os.replace(tmp, unit)
        seen.add(key)
        n += 1
        ev("routea_unit", unit=str(unit.relative_to(out)), files=len(ents), bytes=tot)
        say(f"Route A unit published: out/routeA/{unit.name} ({len(ents)} files, {tot / 1e6:.1f} MB)", "routeA")
    return n


def publisher(work: Path, out: Path, stop: threading.Event, ev, every: float = 60.0) -> threading.Thread:
    seen: set = set()

    def loop():
        while not stop.is_set():
            try:
                publish_exports(work, out, seen, ev)
            except Exception as e:                      # noqa: BLE001 - announced, retried next pass
                say(f"Route A publish error {type(e).__name__}: {e}", "routeA")
            stop.wait(every)
        try:
            publish_exports(work, out, seen, ev)
        except Exception:                               # noqa: BLE001
            pass

    t = threading.Thread(target=loop, daemon=True, name="routeA-publisher")
    t.start()
    return t
