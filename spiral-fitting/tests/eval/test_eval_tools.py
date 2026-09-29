"""Synthetic-spiral tests for spiral-fitting/eval. Each positive case has a paired negative one."""
from __future__ import annotations

import json

import numpy as np
import onsheet
import outer_gap
import reconcile_seams
import scroll_area_est
import window_seams
import zarr
from conftest import NWIND, PITCH, write_winding


def _rows(p):
    return [json.loads(line) for line in p.read_text().splitlines()]


def test_onsheet_lift_positive_on_sheet_and_vanishes_half_a_pitch_off(tmp_path, pred_zarr):
    on_dir, off_dir = tmp_path / "on", tmp_path / "off"
    for w in range(2, 7):
        write_winding(on_dir, w, w, 4, 40)
        write_winding(off_dir, w, w, 4, 40, dr=PITCH / 2)      # between two sheets
    for d, name in ((on_dir, "on.jsonl"), (off_dir, "off.jsonl")):
        onsheet.main([str(d), str(pred_zarr), str(tmp_path / name), "--n", "800", "--shift", str(PITCH / 2)])
    on, off = _rows(tmp_path / "on.jsonl"), _rows(tmp_path / "off.jsonl")
    assert len(on) == len(off) == 5
    for r in on:                       # on the sheet, null half a pitch away is in the gap
        assert r["on_sheet"] > 0.9 and r["null_shift"] < 0.2 and r["lift"] > 0.7
    for r in off:                      # a fit sitting in the gap: its "null" lands ON the sheets
        assert r["on_sheet"] < 0.2 and r["lift"] < 0


def test_outer_gap_flags_a_truncated_fit_and_not_a_full_one(tmp_path, pred_zarr, umbilicus, capsys):
    full, trunc = tmp_path / "full", tmp_path / "trunc"
    for w in range(NWIND):
        write_winding(full, w, w, 0, 47)
        if w < NWIND - 3:
            write_winding(trunc, w, w, 0, 47)                   # outer three windings missing
    out = {}
    for d in (full, trunc):
        j = tmp_path / f"{d.name}.json"
        outer_gap.main([str(d), str(pred_zarr), str(umbilicus), "24", "--level", "0", "--voxel-um", "10",
                        "--json", str(j)])
        out[d.name] = json.loads(j.read_text())
    g_full = out["full"]["gap_L0_p10_p50_p90"][1]
    g_trunc = out["trunc"]["gap_L0_p10_p50_p90"][1]
    assert abs(g_full) < 0.25 * PITCH
    assert abs(g_trunc - 3 * PITCH) < 0.25 * PITCH
    assert out["trunc"]["pred_runs_outside_fit_p10_p50_p90"][1] >= 2
    assert np.isclose(out["trunc"]["gap_mm_p10_p50_p90"][1], g_trunc * 10e-3)


def test_window_seams_recovers_a_planted_plus_one_offset(tmp_path, umbilicus):
    a, b, same = tmp_path / "a", tmp_path / "b", tmp_path / "same"
    for w in range(1, 8):
        write_winding(a, w, w, 0, 30)
        write_winding(same, w, w, 20, 47)
        write_winding(b, w + 1, w, 20, 47)                      # window b numbers every sheet one higher
    r_shift, r_same = tmp_path / "shift.json", tmp_path / "same.json"
    window_seams.main([str(a), str(b), str(umbilicus), "20", "30", str(r_shift), "--pitch", str(PITCH)])
    window_seams.main([str(a), str(same), str(umbilicus), "20", "30", str(r_same), "--pitch", str(PITCH)])
    s, z = json.loads(r_shift.read_text()), json.loads(r_same.read_text())
    assert s["best_offset"] == 1 and s["best_med_vox"] < 0.1 * PITCH
    assert z["best_offset"] == 0 and z["best_med_vox"] < 0.1 * PITCH


def test_reconcile_seams_matches_consistent_windings_and_never_forces_a_displaced_one(tmp_path, umbilicus):
    a, b = tmp_path / "a", tmp_path / "b"
    for w in range(1, 8):
        write_winding(a, w, w, 0, 30)
    for w in range(1, 8):
        if w == 4:
            write_winding(b, w, w, 20, 47, dr=PITCH / 2)        # sits between two sheets: must stay unmatched
        elif w == 5:
            write_winding(b, w, 6, 20, 47)                      # planted swap: b's 5 and 6 exchanged
        elif w == 6:
            write_winding(b, w, 5, 20, 47)
        else:
            write_winding(b, w, w, 20, 47)
    out = tmp_path / "chains.json"
    reconcile_seams.main([str(umbilicus), str(out), "--windows", f"A:0:30:{a}", f"B:20:47:{b}"])
    r = json.loads(out.read_text())
    seam = r["seams"][0]
    m = {wa: wb for wa, wb in seam["matches"]}
    assert 4 not in m and 4 not in m.values()                   # not forced onto a neighbour
    assert m[5] == 6 and m[6] == 5                              # the swap is followed, not flattened to k = 0
    for w in (1, 2, 3, 7):
        assert m[w] == w
    assert seam["offset_hist"] == {"-1": 1, "0": 4, "1": 1}      # JSON keys are strings
    # measured from the windings, not assumed (22.5 here: the planted anomalies in b bias it upward)
    assert abs(seam["pitch_vox"] - PITCH) < 0.15 * PITCH


def test_scroll_area_est_matches_body_volume_over_pitch(tmp_path):
    # a solid cylinder of radius 100 voxels, 40 slices, 10 um voxels
    ct = tmp_path / "ct.zarr"
    g = zarr.open_group(str(ct), mode="w")
    yy, xx = np.mgrid[0:256, 0:256]
    disk = ((xx - 128) ** 2 + (yy - 128) ** 2 <= 100 ** 2).astype(np.uint8) * 200
    g.create_dataset("0", data=np.repeat(disk[None], 40, axis=0), chunks=(8, 128, 128))
    j = tmp_path / "a.json"
    scroll_area_est.main([str(ct), "--z0", "0", "--z1", "40", "--step", "10", "--level", "0",
                          "--pitch", "20", "20", "--vox-um", "10", "--json", str(j)])
    r = json.loads(j.read_text())
    vol = np.pi * 100 ** 2 * 40 * (10e-4) ** 3                   # cm3
    assert abs(r["body_volume_cm3"] - vol) / vol < 0.02
    assert abs(r["sheet_area_cm2_range"][0] - vol / (20 * 10e-4)) / (vol / 0.02) < 0.02
