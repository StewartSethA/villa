#!/usr/bin/env python3
"""Budget governor for a rented, pro-rata-hourly GPU box.  stdlib only; one implementation (Route B box8 mode and the shakedown both use it).

COST MODEL (all rates are CONFIG, defaults = the user's quote of 2026-10-07 for the 8x V100 box)
    spent = hours_since_box_start * hour_usd
          + gb_box_download * ingress_per_tb / 1000       data INTO the box   (CT, tracks, lasagna: fetch_assets bytes_net)
          + gb_box_upload   * egress_per_tb  / 1000       data OUT of the box (the payload our pull script rsyncs home)
DIRECTION NAMING.  Providers name transfers from the BOX's point of view: "download" = the box downloads from the web (our $2.70/TB),
"upload" = the box uploads to the internet, i.e. what we PULL back (our $4/TB).  The user's quote "+$4/TB upload, +$2.70/TB download" is read
that way (matches vast.ai's ingest-cheaper-than-egress pattern in docs/experiments/readerB_2026-10-08/ROUTE_B_ALL_SCROLLS_PLAN.md).  If the user
meant the opposite, swap with --swap-directions / BUDGET_SWAP_DIRECTIONS=1 (or set the two rates): nothing else changes.  Ledger rows say
"box_download_gb" / "box_upload_gb", never a bare "up"/"down".

POLICY
    soft cap (default $45): STOP LAUNCHING NEW FITS when  projected_total(with the candidate) > soft.
        projected_total = spent_now
                        + horizon_h * hour_usd             horizon = hours until the LAST running fit (or the candidate) finishes; the box
                                                           bills by wall clock, so parallel fits share one clock (cost is NOT per-fit)
                        + (unpulled + running + candidate payload GB) * egress   payload that still has to come home
                        + candidate ingress GB * ingress
        Running fits finish (their cost is already inside the horizon).
    hard cap (default $49): hard_stop(now) is True when spent_now + pending payload egress >= hard -> the scheduler checkpoints and stops.
Expected remaining hours of a running fit: from progress (iteration fraction) when known: elapsed*(1-f)/f; else expected_h - elapsed;
an overrunning fit (elapsed >= expected) is assumed to need `overrun_frac` (25 %) of its expected hours more -- an ASSUMPTION, announced in the ledger.

Ledger: <dir>/budget.jsonl, append-only, one JSON object per line (kind = start|transfer|fit_start|fit_end|decision|hard_stop).  State is rebuilt from the
ledger on load, so a restarted scheduler continues the same account.  Budget overrides: env BUDGET_SOFT_USD, BUDGET_HARD_USD, BUDGET_HOUR_USD, BUDGET_DISK_USD_PER_16GB_HOUR, BUDGET_DISK_GB,
BUDGET_INGRESS_PER_TB, BUDGET_EGRESS_PER_TB, BUDGET_BOX_START (epoch s), BUDGET_SWAP_DIRECTIONS.
CLI:  python budget.py status <dir>      prints spent / projected (dry run, reads the ledger only)
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Rates:
    hour_usd: float = 4.276                # machine, USD/h (8x A100 40GB quote 2026-10-08); the DISK term below is billed on top
    disk_usd_per_16gb_hour: float = 0.009  # allocated disk, USD per 16 GB per hour, billed for the WHOLE run whether or not it is full
    disk_gb: float = 934.0                 # allocated disk (GB); 934/16*0.009 = $0.525/h
    ingress_per_tb: float = 2.70           # box downloads from the web
    egress_per_tb: float = 4.00            # box uploads to the internet (our payload pull)
    soft_usd: float = 45.0
    hard_usd: float = 49.0
    overrun_frac: float = 0.25
    max_run_hours: float = 12.0            # wall-clock limit since rental start: no launch whose projected end passes it; 0 = no limit

    @property
    def disk_hour_usd(self) -> float:
        return self.disk_gb / 16.0 * self.disk_usd_per_16gb_hour

    @property
    def eff_hour_usd(self) -> float:
        """What one wall-clock hour of the rental costs: machine + allocated disk.  Every projection uses this, never hour_usd alone."""
        return self.hour_usd + self.disk_hour_usd

    def affordable_hours(self, which: str = "soft") -> float:
        return (self.soft_usd if which == "soft" else self.hard_usd) / self.eff_hour_usd

    @classmethod
    def from_file(cls, path) -> dict:
        """Settings from ONE json file ({"hour_usd":..,"soft_usd":..,"hard_usd":..,"max_run_hours":..,"ingress_per_tb":..,"egress_per_tb":..,
        "swap_directions":bool}); precedence: defaults < file < BUDGET_* env < CLI flags."""
        d = json.loads(Path(path).read_text())
        bad = set(d) - set(cls.__dataclass_fields__) - {"swap_directions", "_doc"}
        if bad:
            raise ValueError(f"unknown budget setting(s) {sorted(bad)} in {path}")
        return {k: v for k, v in d.items() if k != "_doc"}

    def describe(self) -> str:
        return (f"BUDGET SETTINGS: machine ${self.hour_usd}/h + disk {self.disk_gb:g} GB x ${self.disk_usd_per_16gb_hour}/16GB/h = ${self.disk_hour_usd:.3f}/h "
                f"-> effective ${self.eff_hour_usd:.3f}/h ({self.affordable_hours('soft'):.1f} h to the soft cap, {self.affordable_hours('hard'):.1f} h to the hard cap) | soft stop ${self.soft_usd} | hard stop ${self.hard_usd} | max run time "
                f"{self.max_run_hours if self.max_run_hours else 'unlimited'} h | ingress (box downloads) ${self.ingress_per_tb}/TB | "
                f"egress (box uploads) ${self.egress_per_tb}/TB | overrun assumption {self.overrun_frac:.0%}")

    @classmethod
    def from_env(cls, **kw) -> "Rates":
        swap = kw.pop("swap_directions", False)
        r = cls(**kw)
        for env, attr in (("BUDGET_HOUR_USD", "hour_usd"), ("BUDGET_INGRESS_PER_TB", "ingress_per_tb"), ("BUDGET_EGRESS_PER_TB", "egress_per_tb"),
                          ("BUDGET_DISK_USD_PER_16GB_HOUR", "disk_usd_per_16gb_hour"), ("BUDGET_DISK_GB", "disk_gb"), ("BUDGET_SOFT_USD", "soft_usd"), ("BUDGET_HARD_USD", "hard_usd"), ("BUDGET_MAX_RUN_HOURS", "max_run_hours")):
            if os.environ.get(env):
                setattr(r, attr, float(os.environ[env]))
        if swap or os.environ.get("BUDGET_SWAP_DIRECTIONS") == "1":
            r.ingress_per_tb, r.egress_per_tb = r.egress_per_tb, r.ingress_per_tb
        if r.hard_usd < r.soft_usd:
            raise ValueError(f"hard cap {r.hard_usd} below soft cap {r.soft_usd}")
        return r


@dataclass
class _Fit:
    fid: str
    started: float
    expected_h: float
    payload_gb: float
    frac: float = 0.0


class Governor:
    """Thread-safe.  `clock` is injectable (tests / the rehearsal use a fake or scaled clock)."""

    def __init__(self, ledger_dir, rates: Rates | None = None, clock=time.time, box_start: float | None = None, say=print):
        self.dir = Path(ledger_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "budget.jsonl"
        self.r = rates or Rates.from_env()
        self.clock = clock
        self.say = say
        self.lock = threading.RLock()
        self.ingress_gb = 0.0
        self.egress_gb = 0.0
        self.running: dict[str, _Fit] = {}
        self.unpulled_gb = 0.0            # payload produced but not yet counted as pulled
        self.box_start = None
        self._replay()
        if self.box_start is None:
            env = os.environ.get("BUDGET_BOX_START")
            self.box_start = box_start if box_start is not None else (float(env) if env else self.clock())
            src = "argument" if box_start is not None else ("BUDGET_BOX_START" if env else "FIRST LEDGER WRITE (rental start not given -> hours before this are NOT billed here)")
            self._log("start", box_start=self.box_start, source=src, rates=self.r.__dict__)
            self.say(f"budget: box start = {self.box_start:.0f} ({src}); rates {self.r.__dict__}")

    # ---------------------------------------------------------------- ledger
    def _log(self, kind, **kw):
        row = {"t": self.clock(), "kind": kind, **kw}
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def _replay(self):
        if not self.path.exists():
            return
        for line in self.path.read_text().splitlines():
            try:
                x = json.loads(line)
            except ValueError:
                continue
            k = x.get("kind")
            if k == "start":
                self.box_start = x["box_start"]
            elif k == "transfer":
                self.ingress_gb += x.get("box_download_gb", 0.0)
                self.egress_gb += x.get("box_upload_gb", 0.0)
                self.unpulled_gb = max(0.0, self.unpulled_gb - x.get("box_upload_gb", 0.0))
            elif k == "fit_start":
                self.running[x["fid"]] = _Fit(x["fid"], x["t"], x["expected_h"], x.get("payload_gb", 0.0))
            elif k == "fit_end":
                self.running.pop(x["fid"], None)
                self.unpulled_gb += x.get("payload_gb", 0.0)

    # ---------------------------------------------------------------- accounting
    def hours(self, now=None):
        return max(0.0, ((self.clock() if now is None else now) - self.box_start) / 3600.0)

    def spent(self, now=None) -> float:
        with self.lock:
            return (self.hours(now) * self.r.eff_hour_usd + self.ingress_gb * self.r.ingress_per_tb / 1000.0
                    + self.egress_gb * self.r.egress_per_tb / 1000.0)

    def transfer(self, box_download_gb: float = 0.0, box_upload_gb: float = 0.0, what: str = ""):
        with self.lock:
            self.ingress_gb += box_download_gb
            self.egress_gb += box_upload_gb
            self.unpulled_gb = max(0.0, self.unpulled_gb - box_upload_gb)
            self._log("transfer", box_download_gb=box_download_gb, box_upload_gb=box_upload_gb, what=what, spent=round(self.spent(), 4))

    def fit_start(self, fid: str, expected_h: float, payload_gb: float = 0.0):
        with self.lock:
            now = self.clock()
            self.running[fid] = _Fit(fid, now, expected_h, payload_gb)
            self._log("fit_start", fid=fid, expected_h=expected_h, payload_gb=payload_gb)

    def progress(self, fid: str, frac: float):
        with self.lock:
            if fid in self.running:
                self.running[fid].frac = max(0.0, min(1.0, frac))

    def fit_end(self, fid: str, ok: bool, payload_gb: float = 0.0):
        """Fit left the box clock's critical path.  A successful one adds payload that still has to be pulled."""
        with self.lock:
            f = self.running.pop(fid, None)
            if ok:
                self.unpulled_gb += payload_gb
            self._log("fit_end", fid=fid, ok=ok, payload_gb=payload_gb if ok else 0.0, elapsed_h=None if f is None else (self.clock() - f.started) / 3600.0)

    def remaining_h(self, f: _Fit) -> float:
        el = (self.clock() - f.started) / 3600.0
        if f.frac >= 0.02:
            return el * (1.0 - f.frac) / f.frac
        if el < f.expected_h:
            return f.expected_h - el
        return self.r.overrun_frac * f.expected_h

    def projected(self, extra_expected_h: float = 0.0, extra_payload_gb: float = 0.0, extra_ingress_gb: float = 0.0) -> dict:
        with self.lock:
            rem = [self.remaining_h(f) for f in self.running.values()]
            horizon = max(rem + [extra_expected_h, 0.0])
            payload = self.unpulled_gb + sum(f.payload_gb for f in self.running.values()) + extra_payload_gb
            spent = self.spent()
            total = (spent + horizon * self.r.eff_hour_usd + payload * self.r.egress_per_tb / 1000.0 + extra_ingress_gb * self.r.ingress_per_tb / 1000.0)
            return {"spent": spent, "horizon_h": horizon, "payload_gb": payload, "projected_total": total, "running": len(self.running)}

    def may_launch(self, fid: str, expected_h: float, payload_gb: float = 0.0, ingress_gb: float = 0.0) -> tuple[bool, str]:
        """Decision for a candidate fit; always logged (a refusal is announced, never silent)."""
        with self.lock:
            p = self.projected(expected_h, payload_gb, ingress_gb)
            end_h = self.hours() + p["horizon_h"]
            late = bool(self.r.max_run_hours) and end_h > self.r.max_run_hours
            ok = p["projected_total"] <= self.r.soft_usd and not self.hard_stop() and not late
            why = (f"{'LAUNCH' if ok else 'REFUSE'} {fid}: projected ${p['projected_total']:.2f} (spent ${p['spent']:.2f} + horizon {p['horizon_h']:.2f} h x "
                   f"${self.r.eff_hour_usd:.3f}/h (machine+disk) + payload {p['payload_gb']:.1f} GB egress) vs soft cap ${self.r.soft_usd:.2f}"
                   + (f"; projected end {end_h:.2f} h > max run time {self.r.max_run_hours} h" if late else ""))
            self._log("decision", fid=fid, ok=ok, why=why, **p)
            return ok, why

    def hard_stop(self) -> bool:
        with self.lock:
            return self.spent() + self.unpulled_gb * self.r.egress_per_tb / 1000.0 >= self.r.hard_usd

    def note_hard_stop(self):
        self._log("hard_stop", spent=self.spent())

    def status(self) -> dict:
        p = self.projected()
        return {**p, "eff_hour_usd": self.r.eff_hour_usd, "disk_hour_usd": self.r.disk_hour_usd, "hours": self.hours(), "ingress_gb": self.ingress_gb, "egress_gb": self.egress_gb, "soft": self.r.soft_usd, "hard": self.r.hard_usd, "max_run_hours": self.r.max_run_hours,
                "past_max_run": bool(self.r.max_run_hours) and self.hours() >= self.r.max_run_hours, "hard_stop": self.hard_stop()}


class ScaledClock:
    """Rehearsal clock: `scale` simulated seconds per real second (e.g. 3600 -> one real second = one billed hour), offset-able."""

    def __init__(self, scale=1.0, t0=None):
        self.scale, self.real0 = scale, time.time()
        self.t0 = self.real0 if t0 is None else t0

    def __call__(self):
        return self.t0 + (time.time() - self.real0) * self.scale


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "status":
        g = Governor(sys.argv[2], say=lambda *_: None)
        print(json.dumps(g.status(), indent=1))
    else:
        print(__doc__)
