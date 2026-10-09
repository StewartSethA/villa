import importlib.util, pathlib
import numpy as np
spec = importlib.util.spec_from_file_location("split_guard", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "convnext_ink" / "split_guard.py")
SG = importlib.util.module_from_spec(spec); spec.loader.exec_module(SG)


def plane(x0, y0, z, n=40, step=10.0):
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    X, Y, Z = (x0 + ii * step).astype(np.float32), (y0 + jj * step).astype(np.float32), np.full((n, n), z, np.float32)
    return [X, Y, Z], np.ones((n, n), bool)


def test_overlap_is_excluded_and_a_neighbouring_wrap_is_not():
    prot = {"sub": plane(1000, 1000, 500)}                      # protected sheet z=500, x,y in [1000,1390]
    train = {"same_sheet_half": plane(1200, 1000, 502),         # 2 vox off the same sheet, half overlapping in x
             "next_wrap": plane(1000, 1000, 515),               # a neighbouring winding 15 vox away: NOT an overlap
             "far": plane(3000, 3000, 500)}
    res, n = SG.guard(prot, train, excl=6.0, margin_cells=1)
    assert res["same_sheet_half"]["hit_cells"] > 0 and res["same_sheet_half"]["min_dist_kept_vox"] >= 6.0
    assert res["next_wrap"]["hit_cells"] == 0 and res["next_wrap"]["kept_cells"] == 40 * 40
    assert res["far"]["hit_cells"] == 0


def test_margin_dilates_the_exclusion_so_context_cannot_leak():
    prot = {"sub": plane(1000, 1000, 500)}
    a, _ = SG.guard(prot, {"t": plane(1200, 1000, 500)}, excl=6.0, margin_cells=0)
    b, _ = SG.guard(prot, {"t": plane(1200, 1000, 500)}, excl=6.0, margin_cells=3)
    assert b["t"]["kept_cells"] < a["t"]["kept_cells"] and b["t"]["min_dist_kept_vox"] > a["t"]["min_dist_kept_vox"]


def test_a_broken_guard_is_caught_by_the_independent_recheck():
    prot = {"sub": plane(1000, 1000, 500)}
    res, _ = SG.guard(prot, {"t": plane(1200, 1000, 500)}, excl=0.0, margin_cells=0)       # excl=0 -> nothing excluded
    assert res["t"]["min_dist_kept_vox"] < 6.0                                              # the re-check number exposes it


def test_a_submission_overlapping_training_is_detected_as_such():
    spec2 = importlib.util.spec_from_file_location("splits_apply", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "convnext_ink" / "splits_apply.py")
    SA = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(SA)
    grids = {"cand": plane(1000, 1000, 500), "trained_same_sheet": plane(1100, 1000, 503), "trained_elsewhere": plane(5000, 5000, 500), "trained_next_wrap": plane(1000, 1000, 520)}
    ov = SA.overlaps("cand", ["trained_same_sheet", "trained_elsewhere", "trained_next_wrap"], grids)
    assert set(ov) == {"trained_same_sheet"}


def test_registry_checks_catch_a_segment_in_two_splits():
    spec3 = importlib.util.spec_from_file_location("leakage_check", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "convnext_ink" / "leakage_check.py")
    LC = importlib.util.module_from_spec(spec3); spec3.loader.exec_module(LC)
    assert LC.registry_checks({"train": ["a", "b"], "submission": ["c"], "eval": []}) == []
    assert LC.registry_checks({"train": ["a", "b"], "submission": ["b"], "eval": []}) == ["b is in both train and submission"]
    assert LC.registry_checks({"train": ["a", "a"], "submission": [], "eval": []}) == ["duplicate entries inside train"]
