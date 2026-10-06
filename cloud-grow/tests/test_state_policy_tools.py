import json
import os

import pytest

from cloud_grow import growth_guard as GG
from cloud_grow import runner as R
from cloud_grow import state as ST
from cloud_grow import tools as T

POLICY = os.path.join(os.path.dirname(__file__), "..", "config", "guard_policy.production.json")


def test_state_shim_has_production_interface_and_rows_index_both_ways():
    db = ST.connect(None)
    ST.upsert_segment(db, "s1", scroll="X", route="A")
    ST.record_metric(db, "s1", "a", 1.5, stage="grow")
    ST.record_metric(db, "s1", "t", text="hello")
    r = db.execute("SELECT seg, name, value FROM metric WHERE seg=? AND name='a'", ("s1",)).fetchone()
    assert r["value"] == 1.5 and r[2] == 1.5
    seg, name, val = r                                   # the unpacking idiom the fleet's RemoteDB once broke
    assert (seg, name, val) == ("s1", "a", 1.5)
    assert ST.latest_metric(db, "s1", "t") == "hello"
    assert not ST.is_paused(db, "s1", "grow")
    ST.set_paused(db, "s1", "grow", True, by="t")
    assert ST.is_paused(db, "s1", "grow")


def test_artifact_append_only_and_missing_path_ignored(tmp_path):
    db = ST.connect(None)
    ST.upsert_segment(db, "s1")
    d = tmp_path / "ck"; d.mkdir(); (d / "meta.json").write_text("{}")
    ST.record_artifact(db, "s1", "grow", "tifxyz", str(d))
    ST.record_artifact(db, "s1", "grow", "tifxyz", str(d))          # same path+mtime: INSERT OR IGNORE
    assert ST.record_artifact(db, "s1", "grow", "tifxyz", str(tmp_path / "nope")) is None
    assert db.execute("SELECT count(*) FROM artifact").fetchone()[0] == 1


def test_production_policy_values_are_applied_through_policy_from_db():
    db = ST.connect(None)
    R.load_policy_into_state(db, POLICY)
    pol = GG.policy_from_db(db)
    assert pol.enabled and pol.selfcross
    assert pol.selfcross_min_cells == 1 and pol.selfcross_cut_interior is True and pol.selfcross_stop_inherited is False
    assert pol.selfcross_hairpin_abort_ratio == 0.66
    assert pol.selfcross_fail_closed is True                      # default, must stay on
    for k in ("quad_flip", "stretch", "wrap_spacing", "curvature", "ridge_hit", "seam", "normal_dev", "empty_space",
              "roughness", "flatten_feedback"):
        assert getattr(pol, k + "_enforce") is True, k
    assert pol.merge_pause is False


def test_policy_typo_is_an_error_not_silently_ignored(tmp_path):
    p = tmp_path / "p.json"
    p.write_text(json.dumps({"settings": {"grow.guard.selfcros": "1"}}))
    with pytest.raises(ValueError, match="not a GuardPolicy field"):
        R.load_policy_into_state(ST.connect(None), str(p))


def test_tool_gate(kit, pins):
    ident = T.identify(kit, pins=pins)
    assert T.gate(ident) == "PINNED"
    assert "python" in ident["vc_grow_seg_from_seed"]["file"].lower() or "script" in ident["vc_grow_seg_from_seed"]["file"].lower()
    bad = T.identify(kit, pins={**pins, "vc_grow_seg_from_seed": {"md5": "0" * 32}})
    with pytest.raises(T.ToolError, match="pin gate FAILED"):
        T.gate(bad)
    msgs = []
    assert T.gate(bad, allow_unpinned=True, announce=msgs.append) == "UNPINNED-ALLOWED" and msgs      # announced
    os.remove(os.path.join(kit, "vc_tifxyz_selfcross"))
    with pytest.raises(T.ToolError, match="missing"):
        T.gate(T.identify(kit, pins=pins))


def test_shipped_pins_file_names_both_production_tools():
    pins = T.load_pins()
    assert pins["vc_grow_seg_from_seed"]["md5"] == "b747f7658a21aca5bbb05e08c95e192b"
    assert pins["vc_tifxyz_selfcross"]["md5_prefix"] == "73800e99"
