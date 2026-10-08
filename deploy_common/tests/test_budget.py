import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import budget as B  # noqa: E402


class Clk:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def adv_h(self, h):
        self.t += h * 3600.0


def mk(tmp_path, **kw):
    kw.setdefault("hour_usd", 2.33)       # legacy tests were written for the 2.33 $/h V100 quote with no disk term
    kw.setdefault("disk_gb", 0.0)
    c = Clk()
    g = B.Governor(tmp_path, B.Rates(**kw), clock=c, box_start=c.t, say=lambda *_: None)
    return g, c


def test_spent_formula(tmp_path):
    g, c = mk(tmp_path)
    c.adv_h(10)
    g.transfer(box_download_gb=500, box_upload_gb=100)
    # 10 h x 2.33 + 500 GB x 2.70/TB + 100 GB x 4/TB
    assert abs(g.spent() - (23.3 + 1.35 + 0.4)) < 1e-9


def test_soft_cap_stops_new_fits_and_running_finish(tmp_path):
    g, c = mk(tmp_path, max_run_hours=0)       # isolates the money cap from the 12 h default run limit
    launched = []
    for i in range(8):
        ok, _ = g.may_launch(f"f{i}", expected_h=5.0, payload_gb=1.0)
        if ok:
            g.fit_start(f"f{i}", 5.0, 1.0)
            launched.append(i)
    assert len(launched) == 8                       # 5 h x 2.33 = 11.65 + payload: fine; parallel fits share one clock
    c.adv_h(15)                                      # spent 34.95; the running fits are overrunning (25 % more = 1.25 h)
    ok, why = g.may_launch("late", expected_h=5.0)
    assert not ok and "REFUSE" in why               # 34.95 + 5 h x 2.33 = 46.6 > 45
    ok, _ = g.may_launch("short", expected_h=3.0)   # horizon = max(running remaining 1.25, 3.0) -> 34.95 + 6.99 = 41.9 + payload
    assert ok
    assert not g.hard_stop()                         # running fits are never killed by the soft cap
    rows = [json.loads(x) for x in (tmp_path / "budget.jsonl").read_text().splitlines()]
    assert sum(1 for r in rows if r["kind"] == "decision" and not r["ok"]) == 1   # refusals are logged


def test_hard_stop_at_49(tmp_path):
    g, c = mk(tmp_path)
    c.adv_h(20.9)                                    # 48.7
    assert not g.hard_stop()
    c.adv_h(0.2)                                     # 49.17
    assert g.hard_stop()


def test_hard_stop_counts_unpulled_payload(tmp_path):
    g, c = mk(tmp_path)
    g.fit_start("a", 1.0, payload_gb=100.0)
    c.adv_h(20.8)                                    # 48.46 + 100 GB x $4/TB (0.4) = 48.86 -> not yet
    g.fit_end("a", True, payload_gb=100.0)
    assert not g.hard_stop()
    g.fit_start("b", 1.0, payload_gb=200.0)
    g.fit_end("b", True, payload_gb=200.0)          # +0.8 -> 48.46 + 1.2 = 49.66
    assert g.hard_stop()


def test_ledger_replay_continues_account(tmp_path):
    c = Clk()
    g = B.Governor(tmp_path, B.Rates(), clock=c, box_start=c.t, say=lambda *_: None)
    c.adv_h(2)
    g.transfer(box_download_gb=1000)
    g.fit_start("x", 4.0, 5.0)
    s1 = g.spent()
    g2 = B.Governor(tmp_path, B.Rates(), clock=c, say=lambda *_: None)   # restart: no box_start given
    assert abs(g2.spent() - s1) < 1e-9 and "x" in g2.running


def test_progress_drives_remaining(tmp_path):
    g, c = mk(tmp_path)
    g.fit_start("p", 10.0)
    c.adv_h(2)
    g.progress("p", 0.5)                             # half done after 2 h -> 2 h more (expected said 8 more)
    assert abs(g.projected()["horizon_h"] - 2.0) < 1e-9


def test_swap_directions_and_env(tmp_path, monkeypatch):
    monkeypatch.setenv("BUDGET_SWAP_DIRECTIONS", "1")
    monkeypatch.setenv("BUDGET_SOFT_USD", "10")
    r = B.Rates.from_env()
    assert (r.ingress_per_tb, r.egress_per_tb, r.soft_usd) == (4.0, 2.7, 10.0)


def test_scaled_clock():
    c = B.ScaledClock(scale=3600.0, t0=0.0)
    import time as _t
    _t.sleep(0.05)
    assert 100 < c() < 400                           # 0.05 s real = ~180 simulated s


def test_max_run_hours_refuses_launch_that_would_end_late(tmp_path):
    g, c = mk(tmp_path, max_run_hours=5.0, soft_usd=1000.0, hard_usd=2000.0)
    c.adv_h(3)
    ok, why = g.may_launch("a", 1.5)
    assert ok, why
    ok, why = g.may_launch("b", 2.5)          # 3 + 2.5 = 5.5 h > 5 h
    assert not ok and "max run time" in why
    assert g.status()["max_run_hours"] == 5.0


def test_max_run_hours_zero_is_unlimited(tmp_path):
    g, c = mk(tmp_path, max_run_hours=0, soft_usd=1e6, hard_usd=2e6)
    c.adv_h(100)
    assert g.may_launch("a", 50)[0]


def test_rates_file_env_flag_precedence_and_describe(tmp_path, monkeypatch):
    f = tmp_path / "b.json"
    f.write_text(json.dumps({"hour_usd": 7.0, "max_run_hours": 9, "swap_directions": True}))
    kw = B.Rates.from_file(f)
    r = B.Rates.from_env(**kw)
    assert r.hour_usd == 7.0 and r.max_run_hours == 9 and (r.ingress_per_tb, r.egress_per_tb) == (4.0, 2.7)
    monkeypatch.setenv("BUDGET_HOUR_USD", "8")
    assert B.Rates.from_env(**B.Rates.from_file(f)).hour_usd == 8.0
    assert "max run time 9" in r.describe()
    f.write_text(json.dumps({"bogus": 1}))
    import pytest
    with pytest.raises(ValueError):
        B.Rates.from_file(f)


def test_disk_term_is_billed_for_the_whole_run_and_projected(tmp_path):
    r = B.Rates()                                              # defaults: 4.276 $/h machine, 934 GB x 0.009 $/16GB/h
    assert abs(r.disk_hour_usd - 934 / 16 * 0.009) < 1e-12 and abs(r.disk_hour_usd - 0.525375) < 1e-9
    assert abs(r.eff_hour_usd - 4.801375) < 1e-9
    assert abs(r.affordable_hours("soft") - 45 / 4.801375) < 1e-9 and 9.3 < r.affordable_hours("soft") < 9.4 and 10.3 < 50 / r.eff_hour_usd < 10.5
    g, c = mk(tmp_path, hour_usd=4.276, disk_gb=934.0)
    c.adv_h(10)
    assert abs(g.spent() - 10 * 4.801375) < 1e-9               # disk billed although nothing is stored: allocated, not used
    g2, c2 = mk(tmp_path / "b", hour_usd=4.276, disk_gb=0.0)
    c2.adv_h(10)
    assert abs(g.spent() - g2.spent() - 10 * 0.525375) < 1e-9
    # projection and decision text carry the disk term
    p = g.projected(extra_expected_h=2.0)
    assert abs(p["projected_total"] - (10 * 4.801375 + 2 * 4.801375)) < 1e-9
    ok, why = g.may_launch("f", 2.0)
    assert "machine+disk" in why and "4.801" in why
    assert g.status()["eff_hour_usd"] > 4.8 and "disk 934" in r.describe()


def test_disk_flags_env_and_file(tmp_path, monkeypatch):
    f = tmp_path / "b.json"
    f.write_text(json.dumps({"disk_gb": 500, "disk_usd_per_16gb_hour": 0.01}))
    r = B.Rates.from_env(**B.Rates.from_file(f))
    assert r.disk_gb == 500 and abs(r.disk_hour_usd - 500 / 16 * 0.01) < 1e-12
    monkeypatch.setenv("BUDGET_DISK_GB", "160")
    assert B.Rates.from_env().disk_gb == 160
