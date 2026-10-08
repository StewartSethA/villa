"""Route A --extend-only: continue a finished grow from its last checkpoint instead of regrowing (user 2026-10-08: 'Will this resume previous grows?')."""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vesuvius_pipeline import cloud_box as CB            # noqa: E402
from vesuvius_pipeline import growth_guard as GG, resume_gate as RG   # noqa: E402


def _kit(tmp_path):
    k = tmp_path / "kit"
    (k / "bin").mkdir(parents=True, exist_ok=True)
    (k / "lib").mkdir(exist_ok=True)
    for f in ("vc_grow_seg_from_seed", "vc_tifxyz_selfcross"):
        (k / "bin" / f).write_text("x")
    return str(k)


def _fake_run(calls):
    def run(cmd, stdout=None, stderr=None, env=None, timeout=None):
        tgt = Path(cmd[cmd.index("-t") + 1])
        ck = tgt / "ck"
        ck.mkdir(parents=True, exist_ok=True)
        (ck / "meta.json").write_text(json.dumps({"area_cm2": 1.0 * len(calls)}))
        (ck / "x.tif").write_text("x")
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0)
    return run


def _patch(monkeypatch, scrubs, gate_pass=True):
    def scrub(path, pol, vox):
        scrubs.append(path)
        return path, {"ran": True, "clean": True}
    monkeypatch.setattr(GG, "selfx_scrub", scrub)
    monkeypatch.setattr(RG, "check", lambda cur, pol, go=None: {"pass": gate_pass, "reasons": {} if gate_pass else {"hairpin": 9}, "fractions": {}})


def _grow(tmp_path, rounds, calls, extend=False):
    return CB.grow_seed(_kit(tmp_path), "vol", "grids", "PHercT", (10, 20, 30), tmp_path / "export" / "PHercT", rounds, 20, 9.4, pol=object(), run=_fake_run(calls), extend=extend)


def test_extend_continues_from_the_last_checkpoint(tmp_path, monkeypatch):
    scrubs, calls = [], []
    _patch(monkeypatch, scrubs)
    ex = _grow(tmp_path, 2, calls)
    assert ex["run"]["status"] == "grown" and len(ex["rounds"]) == 2 and "--seed" in calls[0] and "--resume" in calls[1]
    n_before = len(calls)
    ex2 = _grow(tmp_path, 5, calls, extend=True)
    assert len(ex2["rounds"]) == 5 and ex2["run"]["extended_from_round"] == 2
    assert len(calls) == n_before + 3                                       # ONLY rounds 3,4,5 ran
    assert all("--resume" in c and "--seed" not in c for c in calls[n_before:])       # resumed, never regrown from the seed
    assert any("generations" in Path(c[c.index("--params") + 1]).read_text() for c in calls[n_before:])
    assert json.loads((tmp_path / "export" / "PHercT" / ex["identity"]["seg"] / "export.json").read_text())["run"]["extended_from_round"] == 2


def test_extend_never_touches_a_held_or_complete_export(tmp_path, monkeypatch):
    scrubs, calls = [], []
    _patch(monkeypatch, scrubs, gate_pass=False)
    held = _grow(tmp_path, 3, calls)
    assert held["run"]["status"] == "gate_held"
    n = len(calls)
    _patch(monkeypatch, scrubs, gate_pass=True)
    again = _grow(tmp_path, 6, calls, extend=True)
    assert again["run"]["status"] == "gate_held" and len(calls) == n          # held stays held: no round ran
    # already complete (rounds reached): nothing runs either
    tmp2 = tmp_path / "b"
    tmp2.mkdir()
    _patch(monkeypatch, [], True)
    c2 = []
    _grow(tmp2, 2, c2)
    n2 = len(c2)
    _grow(tmp2, 2, c2, extend=True)
    assert len(c2) == n2


def test_without_extend_a_rerun_regrows_exactly_as_before(tmp_path, monkeypatch):
    scrubs, calls = [], []
    _patch(monkeypatch, scrubs)
    _grow(tmp_path, 2, calls)
    n = len(calls)
    _grow(tmp_path, 2, calls, extend=False)
    assert len(calls) == n + 2 and "--seed" in calls[n]                        # the old behaviour, unchanged
