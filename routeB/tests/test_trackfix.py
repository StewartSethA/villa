"""The 2^24-track limit: lifted when tracks.py has the chunked multinomial, with an AUTOMATIC fallback to the old cap on a real overflow (user 2026-10-08)."""
import json

from routeB import ladder as L


def _cfg(tmp_path, fixed=True, env=None):
    tp = tmp_path / "tracks.py"
    tp.write_text("def _multinomial_chunked(p, k):\n    pass\n" if fixed else "x = 1\n")
    cfg = L.load_config()
    cfg["limits"]["multinomial_categories"] = L.ORIG_CATEGORIES
    cfg["_track_limit"] = L.lift_track_limit_if_fixed(cfg, tp, env or {})
    return cfg


def test_lifted_when_fix_present_and_not_when_absent_or_overridden(tmp_path):
    assert _cfg(tmp_path)["_track_limit"]["lifted"] and _cfg(tmp_path)["limits"]["multinomial_categories"] == L.LIFTED_CATEGORIES
    assert not _cfg(tmp_path, fixed=False)["_track_limit"]["lifted"]
    c = _cfg(tmp_path, env={"ROUTEB_KEEP_TRACK_LIMIT": "1"})
    assert not c["_track_limit"]["lifted"] and c["limits"]["multinomial_categories"] == L.ORIG_CATEGORIES
    assert not L.lift_track_limit_if_fixed({"limits": {}}, tmp_path / "missing.py", {})["lifted"]


def test_height_is_no_longer_capped_by_the_limit_when_lifted(tmp_path):
    c = _cfg(tmp_path)
    c["scrolls"] = {"PHerc0191": {"n_tracks": 22_757_127}}
    h, why = L.start_height_for(c, "PHerc0191", 47.8, 13000)
    assert h == 13000 and "2^24" in why
    old = _cfg(tmp_path, fixed=False)
    old["scrolls"] = {"PHerc0191": {"n_tracks": 22_757_127}}
    assert L.start_height_for(old, "PHerc0191", 47.8, 13000)[0] < 13000


def test_real_overflow_engages_fallback_once_and_caps_again(tmp_path):
    c = _cfg(tmp_path)
    c["dynamic"] = True
    job = L.make_job(c, "PHerc0191", 146, L.dyn_rung(c, 13000), 4500, 17500)
    job["n_loaded"] = 22_757_127
    d = L.decide(c, job, "multinomial", 146)
    assert c["_track_limit"]["fallback_engaged"] and c["limits"]["multinomial_categories"] == L.ORIG_CATEGORIES
    assert d["action"] == "descend" and d["height"] < 13000
    assert L.engage_track_limit_fallback(c, job) is False            # only the first time


def test_marker_restores_the_fallback_on_restart(tmp_path):
    c = _cfg(tmp_path)
    m = tmp_path / "track_limit_fallback.json"
    m.write_text(json.dumps({"job": "PHerc0191/full"}))
    assert L.restore_fallback_marker(c, m) and c["limits"]["multinomial_categories"] == L.ORIG_CATEGORIES
    assert not L.restore_fallback_marker(_cfg(tmp_path), tmp_path / "none.json")


def test_strict_classification_keeps_other_failures_when_lifted():
    oom_tb = "File tracks.py line 2990 in _multinomial_chunked\ntorch.cuda.OutOfMemoryError: CUDA out of memory"
    assert L.classify(oom_tb, 1, strict_multinomial=True) == "oom"
    assert L.classify(oom_tb, 1) == "multinomial"                    # the old (non-strict) behaviour, unchanged
    real = "RuntimeError: number of categories cannot exceed 2^24"
    assert L.classify(real, 1, strict_multinomial=True) == "multinomial"
