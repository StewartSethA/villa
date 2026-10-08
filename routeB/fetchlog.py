"""Collapse the fetcher's per-file chatter into ONE summary line per scroll per 30 s (the console and box8.log stay readable; errors pass through at once).

deploy_common/fetch_assets.py logs '  s3 <prefix>: i/N objects, X GB' and '<file>: a/b blocks, X GB this run, Y MB/s' lines.  The Collapser folds them into
    [HH:MM:SS] fetch: PHerc0191 3.3/9.7 GB 41 MB/s 380 obj/s ETA 4 min | nx 6000/8573 obj
(GB done / expected total, aggregate MB/s and objects/s since the previous line, ETA at that rate); routeB_watch.sh parses exactly this line.
"""
from __future__ import annotations

import re
import time

OBJ = re.compile(r"s3 (\S+): (\d+)/(\d+) objects, ([\d.]+) GB")
BLK = re.compile(r"(\S+): (\d+)/(\d+) blocks, ([\d.]+) GB this run")
BAD = re.compile(r"FAIL|ERROR|retry|Traceback|mismatch|missing", re.I)


def field_name(prefix: str) -> str:
    for k, v in (("_nx", "nx"), ("_ny", "ny"), ("grad_mag", "grad"), ("_cos", "cos")):
        if k in prefix:
            return v
    return "las"


class Collapser:
    def __init__(self, scroll: str, total_gb: float = 0.0, interval: float = 30.0, emit=print, clock=time.time):
        self.scroll, self.total_gb, self.interval, self.emit, self.clock = scroll, total_gb, interval, emit, clock
        self.tracks_gb = 0.0
        self.fields: dict[str, list] = {}           # field -> [done, total, gb]
        self.last_t = self.t0 = clock()
        self.last_gb = 0.0
        self.last_obj = 0

    def gb(self) -> float:
        return self.tracks_gb + sum(f[2] for f in self.fields.values())

    def objs(self) -> int:
        return sum(f[0] for f in self.fields.values())

    def __call__(self, line: str) -> None:
        txt = line.strip()
        m = OBJ.search(txt)
        b = BLK.search(txt)
        if m:
            self.fields[field_name(m.group(1))] = [int(m.group(2)), int(m.group(3)), float(m.group(4))]
        elif b:
            self.tracks_gb = float(b.group(4))
        elif BAD.search(txt):
            self.emit(f"{self.scroll}: {txt[:200]}")
            return
        else:
            return                                   # [get] lines, notes, per-file noise: collapsed away
        if self.clock() - self.last_t >= self.interval:
            self.summary()

    def summary(self, final: bool = False) -> None:
        now = self.clock()
        dt = max(1e-6, now - self.last_t)
        gb, ob = self.gb(), self.objs()
        mbs, ops = (gb - self.last_gb) * 1000.0 / dt, (ob - self.last_obj) / dt
        eta = ""
        if self.total_gb and mbs > 0.1 and not final:
            eta = f" ETA {max(0.0, (self.total_gb - gb) * 1000.0 / mbs / 60.0):.0f} min"
        det = " ".join(f"{k} {v[0]}/{v[1]}" for k, v in self.fields.items())
        tot = f"/{self.total_gb:.1f}" if self.total_gb else ""
        self.emit(f"{self.scroll} {gb:.1f}{tot} GB {mbs:.0f} MB/s {ops:.0f} obj/s{eta}{' DONE' if final else ''} | {det}{' obj' if det else ''}")
        self.last_t, self.last_gb, self.last_obj = now, gb, ob

    def final(self) -> None:
        self.summary(final=True)
