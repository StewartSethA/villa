"""Route A overlap monitor: a segment growing onto another segment's sheet is detected; a neighbouring WRAP (~15 voxels away) is not."""
import json
from pathlib import Path

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")
from vesuvius_pipeline.routea_cloud import overlap as OV, status as RS     # noqa: E402


def _seg(work: Path, scroll: str, name: str, x0=1000.0, z0=5000.0, n=40, area=10.0, y0=1000.0):
    ck = work / "export" / scroll / name / "r1" / "ck"
    ck.mkdir(parents=True)
    j, i = np.meshgrid(np.arange(n), np.arange(n))
    X, Y, Z = x0 + j * 20.0, y0 + i * 20.0, np.full((n, n), z0)
    for a, A in zip("xyz", (X, Y, Z)):
        tifffile.imwrite(ck / f"{a}.tif", A.astype(np.float32))
    (ck / "meta.json").write_text(json.dumps({"area_cm2": area}))
    return ck


def test_duplicate_partial_and_neighbouring_wrap(tmp_path):
    w = tmp_path / "routeA_work"
    _seg(w, "PHercT", "a")                                   # the reference sheet
    _seg(w, "PHercT", "b", z0=5001.0)                        # same sheet, 1 voxel off  -> overlap ~100 %
    _seg(w, "PHercT", "c", z0=5015.0)                        # the NEXT wrap, 15 voxels away -> not an overlap
    _seg(w, "PHercT", "d", x0=1000.0 + 20 * 20.0)            # shifted half a segment along x -> ~50 %
    _seg(w, "PHercU", "e")                                   # another scroll: never compared with PHercT
    d = OV.scan(w, time_budget_s=60)
    got = {(p["small"], p["large"]): p["frac"] for p in d["pairs"]}
    pair = lambda a, b: got.get((a, b)) or got.get((b, a))
    assert pair("a", "b") > 0.9
    assert pair("a", "c") is None and pair("b", "c") is None and pair("c", "d") is None          # wraps 15 vox apart are NOT overlapping
    assert 0.35 < pair("a", "d") < 0.65
    assert all("e" not in (p["small"], p["large"]) for p in d["pairs"])
    assert d["n_segments"] == 5 and d["dup_cm2"] > 0


def test_status_reports_the_overlap_and_the_unique_estimate(tmp_path):
    w = tmp_path / "routeA_work"
    for name, z in (("a", 5000.0), ("b", 5001.0)):
        ck = _seg(w, "PHercT", name, z0=z, area=10.0)
        (ck.parents[1] / "export.json").write_text(json.dumps({"run": {"status": "grown"}, "rounds": [{"gate": {"pass": True}}], "identity": {"seg": name}}))
    (w / "seeds").mkdir()
    (w / "seeds" / "PHercT.json").write_text(json.dumps([{"x": 1}, {"x": 2}, {"x": 3}]))
    d = RS.summarize(w, 4, tracers=0, cpu=1.0)
    assert d["overlap"]["n_overlapping"] == 1 and d["overlap"]["dup_cm2"] > 5
    assert d["est_unique_final_cm2"] < d["est_final_cm2"]
    assert any(x.startswith("A: OVERLAP") for x in RS.render_lines(d))


def test_misaligned_vertex_lattices_still_overlap(tmp_path):
    """Seeds grow on their own lattices: the same sheet sampled with vertices 7/9 voxels off must still be ~100 % overlapping (vertex-to-vertex distance would hide it)."""
    w = tmp_path / "routeA_work"
    _seg(w, "PHercT", "a")
    _seg(w, "PHercT", "b", x0=1007.0, y0=1009.0, z0=5001.0)
    d = OV.scan(w, time_budget_s=60)
    assert d["pairs"] and d["pairs"][0]["frac"] > 0.8, d["pairs"]
