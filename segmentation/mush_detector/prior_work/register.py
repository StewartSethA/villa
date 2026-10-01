#!/usr/bin/env python3
"""Register PHerc0125 mush-label GIMP screenshots against the hub's PHerc0125.zarr.

For each labelled z-slice: find (scale, tx, ty) mapping screenshot-canvas pixels
(u,v) -> level-0 volume (x,y) at that z, by coarse multi-scale NCC search at
zarr level 2, refined at level 0/1. No cv2 available in this venv; NCC is
computed by hand (FFT cross-correlation + integral images for the normalizer),
which is numerically identical to OpenCV's TM_CCOEFF_NORMED.

The scale-bar reading ("300" over 103 px, same for every slice) gives two
candidate hypotheses (300 um -> 0.311 vox/px, or 300 vox -> 2.913 vox/px) but
is NEVER trusted directly -- both are included in the coarse sweep and the
actual scale is whichever wins on NCC, per task instruction "never guess the
transform".
"""
import json
import time
from pathlib import Path

import numpy as np
import zarr
from scipy.ndimage import zoom as ndzoom
from scipy.signal import fftconvolve

SRC_DIR = Path("/home/seth/ScrollPrizeTutorial/data/Scrolls/crushed scroll label")
EXPORT_DIR = Path(
    "/tmp/claude-1000/-home-seth-ScrollPrizeTutorial/2009ddb2-9fa9-4865-b0ef-fcf612867acf/scratchpad/mush"
)
ZARR_PATH = "/mnt/raid7/scroll_volume_cache/PHerc0125.zarr"
OUT_DIR = EXPORT_DIR / "reg_out"
OUT_DIR.mkdir(exist_ok=True)

SLICES = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
GRAY = 128


def canvas_bbox(gray_img, thresh=50):
    """Viewer canvas bbox. A plain 'not == letterbox gray' test is contaminated
    by a crosshair overlay that is drawn across the WHOLE screenshot, including
    the gray margins (1-3 px wide, visible as isolated non-gray pixels far outside
    the real canvas). Require a minimum count of non-gray pixels per row/column
    so a thin global overlay line cannot masquerade as canvas edge."""
    nongray = gray_img != GRAY
    cols = np.flatnonzero(nongray.sum(axis=0) > thresh)
    rows = np.flatnonzero(nongray.sum(axis=1) > thresh)
    return rows.min(), rows.max() + 1, cols.min(), cols.max() + 1


def load_screenshot_gray(z):
    from PIL import Image

    f = SRC_DIR / f"0125-z{z}spiral.png"
    im = np.array(Image.open(f).convert("L"))
    return im


def get_slice(level_arr, z_level0, level):
    zl = int(round(z_level0 / (2 ** level)))
    zl = max(0, min(level_arr.shape[0] - 1, zl))
    return np.asarray(level_arr[zl]), zl


def ncc_map(source, template):
    """Valid-mode normalized cross-correlation map, == cv2 TM_CCOEFF_NORMED."""
    src = source.astype(np.float64)
    tpl = template.astype(np.float64)
    th, tw = tpl.shape
    n = th * tw
    tpl_mean = tpl.mean()
    tpl0 = tpl - tpl_mean
    tpl_ss = float((tpl0 ** 2).sum())
    # cross term: sum(I_patch * T) via correlation (flip template for convolution)
    cross = fftconvolve(src, tpl0[::-1, ::-1], mode="valid")
    # integral images for sum(I) and sum(I^2) over each tw x th window
    ii = np.zeros((src.shape[0] + 1, src.shape[1] + 1))
    ii[1:, 1:] = np.cumsum(np.cumsum(src, axis=0), axis=1)
    ii2 = np.zeros_like(ii)
    s2 = src * src
    ii2[1:, 1:] = np.cumsum(np.cumsum(s2, axis=0), axis=1)

    H2, W2 = cross.shape
    win_sum = (ii[th:, tw:] - ii[:-th or None, tw:] - ii[th:, : -tw or None] + ii[: -th or None, : -tw or None])[
        :H2, :W2
    ]
    win_sumsq = (ii2[th:, tw:] - ii2[:-th or None, tw:] - ii2[th:, : -tw or None] + ii2[: -th or None, : -tw or None])[
        :H2, :W2
    ]
    win_ss = win_sumsq - (win_sum ** 2) / n
    denom = np.sqrt(np.clip(win_ss, 1e-6, None) * tpl_ss)
    ncc = cross / np.clip(denom, 1e-6, None)
    return ncc


def ncc_search_scale(template_full, source, scales, level0_per_srcpx):
    results = []
    H, W = source.shape
    for s in scales:
        f = s / level0_per_srcpx
        tw = max(8, int(round(template_full.shape[1] * f)))
        th = max(8, int(round(template_full.shape[0] * f)))
        if tw >= W or th >= H or tw < 8 or th < 8:
            continue
        templ = ndzoom(template_full.astype(np.float64), (th / template_full.shape[0], tw / template_full.shape[1]), order=1)
        m = ncc_map(source, templ)
        idx = np.unravel_index(np.argmax(m), m.shape)
        results.append((s, float(m[idx]), (int(idx[0]), int(idx[1])), (th, tw)))
    results.sort(key=lambda r: -r[1])
    return results


def main():
    g = zarr.open(ZARR_PATH, mode="r")
    lvl0, lvl1, lvl2 = g["0"], g["1"], g["2"]

    coarse_scales = np.geomspace(0.15, 4.0, 22)
    summary = []
    for z in SLICES:
        t0 = time.time()
        shot = load_screenshot_gray(z)
        r0, r1, c0, c1 = canvas_bbox(shot)
        canvas = shot[r0:r1, c0:c1]

        slice2, zl2 = get_slice(lvl2, z, 2)
        coarse = ncc_search_scale(canvas, slice2, coarse_scales, level0_per_srcpx=4.0)
        if not coarse:
            summary.append({"z": z, "error": "no coarse candidates fit"})
            print("NOFIT", z, flush=True)
            continue
        best_s, best_ncc, (ly, lx), (th, tw) = coarse[0]

        level0_per_srcpx = 1.0
        level_arr, level = lvl0, 0
        if canvas.shape[1] * best_s > 6000:
            level_arr, level0_per_srcpx, level = lvl1, 2.0, 1
        slice_r, zl_r = get_slice(level_arr, z, level)

        fine_scales = np.geomspace(best_s * 0.85, best_s * 1.15, 9)
        fine = ncc_search_scale(canvas, slice_r, fine_scales, level0_per_srcpx)
        if not fine:
            fine = [(best_s, best_ncc, (int(ly * 4 / level0_per_srcpx), int(lx * 4 / level0_per_srcpx)), (th, tw))]
        fs, fncc, (fy, fx), (fth, ftw) = fine[0]

        lvl_to_l0 = 2 ** level
        x0_l0 = fx * lvl_to_l0
        y0_l0 = fy * lvl_to_l0
        residual = 1.0 - fncc

        rec = {
            "z_level0": z,
            "canvas_bbox_screenshot_px": [int(r0), int(r1), int(c0), int(c1)],
            "canvas_shape": list(canvas.shape),
            "coarse_best_scale_vox_per_px": best_s,
            "coarse_best_ncc": best_ncc,
            "refine_level": level,
            "refine_zlevel_index": int(zl_r),
            "fit_scale_vox_per_screenshot_px": fs,
            "fit_ncc": fncc,
            "fit_residual_1_minus_ncc": residual,
            "match_topleft_level0_xy": [float(x0_l0), float(y0_l0)],
            "match_template_shape_level_px": [int(fth), int(ftw)],
            "elapsed_s": time.time() - t0,
        }
        summary.append(rec)
        print(json.dumps(rec), flush=True)
        with open(OUT_DIR / f"reg_z{z}.json", "w") as fh:
            json.dump(rec, fh, indent=1)

    with open(OUT_DIR / "reg_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print("DONE", len(summary), "slices")


if __name__ == "__main__":
    main()
