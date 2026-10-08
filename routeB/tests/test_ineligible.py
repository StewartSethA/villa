import json
from pathlib import Path
from routeB import box8, routea_side

def test_auto_set_excludes_ineligible_scroll():
    assert "PHerc0343" in json.loads((box8.ROOT / "pins" / "ineligible.json").read_text())["scrolls"]
    assert "PHerc0343" not in box8.all_scroll_names()
    assert "PHerc0125" in box8.all_scroll_names()

def test_routea_pick_skips_ineligible(tmp_path):
    pins = tmp_path / "scrolls.json"
    pins.write_text(json.dumps({"scrolls": {"PHerc0343": {"prediction_bytes": 1, "grids_bytes": 1}, "PHerc0125": {"prediction_bytes": 1, "grids_bytes": 1}}}))
    (tmp_path / "ineligible.json").write_text(json.dumps({"scrolls": {"PHerc0343": {}}}))
    assert [s for s, _ in routea_side.pick_scrolls(pins, 100.0, [])] == ["PHerc0125"]
