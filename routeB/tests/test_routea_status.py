"""Route A status + in-flight guard control (user 2026-10-08)."""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vesuvius_pipeline.routea_cloud import settings as RS_settings, status as RS     # noqa: E402


def _seg(work, scroll, name, status, gate=None, area=1.0):
    d = work / "export" / scroll / name
    (d / "r1" / "x").mkdir(parents=True)
    (d / "r1" / "x" / "meta.json").write_text(json.dumps({"area_cm2": area}))
    rounds = [{"cpu_s": 10, "gate": gate}] if gate else [{"cpu_s": 10}]
    (d / "export.json").write_text(json.dumps({"run": {"status": status, "wall_s_total": 5}, "rounds": rounds, "identity": {"seg": name}}))


def _gate(reasons, fractions, thresholds=None):
    return {"pass": False, "reasons": reasons, "fractions": fractions,
            "thresholds": thresholds or {"GATE_FOLD_FRAC": 0.021, "GATE_NORMAL_REV_FRAC": 0.072, "GATE_PROX_FRAC": 0.045, "GATE_HAIRPIN_FRAC": 0.005, "GATE_TRANSVERSE_CELLS": 0}}


def _work(tmp_path):
    w = tmp_path / "routeA_work"
    (w / "seeds").mkdir(parents=True)
    (w / "seeds" / "PHerc0125.json").write_text(json.dumps([{"x": i} for i in range(10)]))
    _seg(w, "PHerc0125", "a", "deadline", area=20.0)
    _seg(w, "PHerc0125", "b", "gate_held", _gate({"fold_over": 120}, {"fold_over": 0.03}), area=2.0)      # needs x1.43
    _seg(w, "PHerc0125", "c", "gate_held", _gate({"fold_over": 1, "normal_reversal": 1}, {"fold_over": 0.04, "normal_reversal": 0.08}), area=1.0)   # needs x1.9
    _seg(w, "PHerc0125", "d", "gate_held", _gate({"transverse": 5}, {"transverse_cells": 5}), area=0.5)     # zero tolerance: never freed by a multiplier
    return w


def test_guard_counts_and_sweep(tmp_path):
    d = RS.summarize(_work(tmp_path), slots=38, tracers=0, cpu=5.0)
    assert d["seeds_planned"] == 10 and d["seeds_done"] == 4 and d["held"] == 3
    assert d["held_reasons"] == {"fold_over": 2, "normal_reversal": 1, "transverse": 1}
    assert d["sweep"] == {"1.25": 0, "1.5": 1, "2.0": 2, "3.0": 2}                 # b at x1.43; c at x1.9; d never
    assert abs(d["area_cm2"] - 23.5) < 1e-6 and abs(d["area_by_status"]["gate_held"] - 3.5) < 1e-6


def test_states(tmp_path):
    w = _work(tmp_path)
    now = time.time()
    assert RS.summarize(w, 38, tracers=0, cpu=3.0, now=now)["state"] == "IDLE"                      # nothing running, not finished
    (w / "report.json").write_text(json.dumps({"finished_utc": "2026-10-08T15:44:09Z", "totals": {"verified_pass": 1}}))
    d = RS.summarize(w, 38, tracers=0, cpu=3.0, now=now)
    assert d["state"] == "FINISHED" and d["verified_pass"] == 1
    (w / "report.json").unlink()
    assert RS.summarize(w, 38, tracers=36, cpu=80.0, now=now)["state"] == "GROWING"
    assert RS.summarize(w, 38, tracers=10, cpu=20.0, now=now)["state"] == "UNDERBOOKED"
    assert RS.summarize(w, 38, tracers=36, cpu=80.0, now=now + 3 * 3600)["state"] == "STALLED"       # tracers but nothing touched for hours
    lines = RS.render_lines(RS.summarize(w, 38, tracers=36, cpu=80.0, now=now))
    assert len(lines) >= 4 and all(len(x) <= 100 for x in lines) and "by guard" in lines[2] and "usable" in lines[1]


def test_waiting_while_downloading(tmp_path):
    w = _work(tmp_path)
    logs = tmp_path / "box8" / "logs"
    logs.mkdir(parents=True)
    (logs / "routeA.log").write_text("[13:12:27]   s3 PHerc0826/representations/predictions/surfaces/x.zarr: 6000/54836 objects, 1.3 GB\n")
    d = RS.summarize(w, 38, tracers=0, cpu=3.0, now=time.time())
    assert d["state"] == "WAITING" and "PHerc0826" in d["why"]


def test_control_file_is_read_on_every_call_and_unknown_keys_rejected(tmp_path, monkeypatch):
    w = tmp_path / "routeA_work"
    monkeypatch.setenv("ROUTEA_WORK", str(w))
    base = RS_settings.gate_override()["GATE_FOLD_FRAC"]
    RS.main(["--work", str(w), "guards", "set", "GATE_FOLD_FRAC=0.05", "GATE_BOGUS=1"])
    assert RS_settings.gate_override()["GATE_FOLD_FRAC"] == 0.05 != base                 # takes effect on the very next call (= next round)
    assert "GATE_BOGUS" not in RS_settings.gate_override()
    RS.main(["--work", str(w), "guards", "clear"])
    assert RS_settings.gate_override()["GATE_FOLD_FRAC"] == base


def test_usable_estimate_remaining_and_rate(tmp_path):
    w = _work(tmp_path)
    # a held segment with two rounds: its usable surface is the round BEFORE the failing one (r1 = 4 cm2), not the failing r2 (5 cm2)
    d = w / "export" / "PHerc0125" / "e"
    (d / "r1" / "x").mkdir(parents=True)
    (d / "r1" / "x" / "meta.json").write_text(json.dumps({"area_cm2": 4.0}))
    (d / "r2" / "x").mkdir(parents=True)
    (d / "r2" / "x" / "meta.json").write_text(json.dumps({"area_cm2": 5.0}))
    (d / "export.json").write_text(json.dumps({"run": {"status": "gate_held"}, "rounds": [{"gate": {"pass": True}}, {"gate": _gate({"fold_over": 3}, {"fold_over": 0.03})}], "identity": {"seg": "e"}}))
    now = time.time()
    x = RS.summarize(w, 38, tracers=0, cpu=3.0, now=now)
    assert x["seeds_planned"] == 10 and x["seeds_remaining"] == 5               # 10 planned, 5 finished
    assert abs(x["usable_cm2"] - (20.0 + 4.0)) < 1e-6                           # a (20) + e's passing round (4); b, c, d have no earlier round
    assert x["finish_rate_per_h"] == 10.0                                       # 5 exports finished within the last 30 min -> 10 / h
    assert x["est_final_cm2"] >= x["usable_cm2"]
    assert abs(x["mean_final_cm2"] - (20.0 + 0 + 0 + 0 + 4.0) / 5) < 1e-6
