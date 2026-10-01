#!/usr/bin/env python3
"""Re-register all 19 PHerc0125 screenshots trying all 8 D4 orientations
(identity, fliplr, flipud, 180 rot, transpose, and the 3 remaining 90-degree/
diagonal combinations) plus scale+translation, picking each slice's best by NCC.
Coarse (level2, 22 scales) search run for ALL 8 orientations (cheap); fine
refine (level1/0) run only for the orientation with the best coarse NCC."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/seth/ScrollPrizeTutorial/scripts/labels/mush")
import numpy as np
import zarr
import register as R

SRC_DIR = Path("/home/seth/ScrollPrizeTutorial/data/Scrolls/crushed scroll label")
OUT_DIR = Path(
    "/tmp/claude-1000/-home-seth-ScrollPrizeTutorial/2009ddb2-9fa9-4865-b0ef-fcf612867acf/scratchpad/mush/reg_out_d4"
)
OUT_DIR.mkdir(exist_ok=True)
ZARR_PATH = "/mnt/raid7/scroll_volume_cache/PHerc0125.zarr"

ALL_Z = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
         11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000]

ORIENTATIONS = {
    "identity": lambda a: a,
    "fliplr": lambda a: a[:, ::-1],
    "flipud": lambda a: a[::-1, :],
    "rot180": lambda a: a[::-1, ::-1],
    "transpose": lambda a: a.T,
    "transpose_fliplr": lambda a: a.T[:, ::-1],
    "transpose_flipud": lambda a: a.T[::-1, :],
    "anti_transpose": lambda a: a.T[::-1, ::-1],
}


def filename_for(z):
    if z == 11000:
        return SRC_DIR / "125-z11000spiral.png"
    return SRC_DIR / f"0125-z{z}spiral.png"


def load_canvas_gray(z):
    from PIL import Image
    shot = np.array(Image.open(filename_for(z)).convert("L"))
    r0, r1, c0, c1 = R.canvas_bbox(shot)
    return shot[r0:r1, c0:c1], (r0, r1, c0, c1)


def main():
    g = zarr.open(ZARR_PATH, mode="r")
    lvl0, lvl1, lvl2 = g["0"], g["1"], g["2"]
    coarse_scales = np.geomspace(0.15, 4.0, 22)
    summary = []
    for z in ALL_Z:
        t0 = time.time()
        canvas0, (r0, r1, c0, c1) = load_canvas_gray(z)
        slice2, zl2 = R.get_slice(lvl2, z, 2)

        # coarse NCC search for all 8 orientations
        orient_results = {}
        for name, fn in ORIENTATIONS.items():
            canvas = fn(canvas0)
            coarse = R.ncc_search_scale(canvas, slice2, coarse_scales, level0_per_srcpx=4.0)
            if coarse:
                orient_results[name] = (coarse[0], canvas.shape)
            else:
                orient_results[name] = (None, canvas.shape)

        ranked = sorted(
            ((name, v) for name, v in orient_results.items() if v[0] is not None),
            key=lambda kv: -kv[1][0][1],
        )
        if not ranked:
            summary.append({"z": z, "error": "no orientation fit"})
            print("NOFIT", z, flush=True)
            continue
        best_name, ((best_s, best_ncc, (ly, lx), (th, tw)), canvas_shape) = ranked[0]
        coarse_table = {name: round(v[0][1], 4) if v[0] else None for name, v in orient_results.items()}

        # fine refine only for the winning orientation
        canvas = ORIENTATIONS[best_name](canvas0)
        level0_per_srcpx = 1.0
        level_arr, level = lvl0, 0
        if canvas.shape[1] * best_s > 6000:
            level_arr, level0_per_srcpx, level = lvl1, 2.0, 1
        slice_r, zl_r = R.get_slice(level_arr, z, level)
        fine_scales = np.geomspace(best_s * 0.85, best_s * 1.15, 9)
        fine = R.ncc_search_scale(canvas, slice_r, fine_scales, level0_per_srcpx)
        if not fine:
            fine = [(best_s, best_ncc, (int(ly * 4 / level0_per_srcpx), int(lx * 4 / level0_per_srcpx)), (th, tw))]
        fs, fncc, (fy, fx), (fth, ftw) = fine[0]
        lvl_to_l0 = 2 ** level
        x0_l0 = fx * lvl_to_l0
        y0_l0 = fy * lvl_to_l0

        rec = {
            "z_level0": z,
            "canvas_bbox_screenshot_px": [int(r0), int(r1), int(c0), int(c1)],
            "best_orientation": best_name,
            "coarse_ncc_by_orientation": coarse_table,
            "canvas_shape_oriented": list(canvas.shape),
            "refine_level": level,
            "fit_scale_vox_per_screenshot_px": fs,
            "fit_ncc": fncc,
            "fit_residual_1_minus_ncc": 1.0 - fncc,
            "match_topleft_level0_xy": [float(x0_l0), float(y0_l0)],
            "match_template_shape_level_px": [int(fth), int(ftw)],
            "elapsed_s": time.time() - t0,
        }
        summary.append(rec)
        print(f"z={z} best={best_name} ncc={fncc:.3f} coarse_table={coarse_table}", flush=True)
        with open(OUT_DIR / f"reg_z{z}.json", "w") as fh:
            json.dump(rec, fh, indent=1)

    with open(OUT_DIR / "reg_summary_d4.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print("DONE_D4", len(summary), "slices")


if __name__ == "__main__":
    main()
