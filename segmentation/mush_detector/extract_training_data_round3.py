#!/usr/bin/env python3
"""Re-extract the label-dependent data using the CORRECTED (fliplr) registration in
reg_out_d4/, with BINARIZED labels (user correction: translucency was a labelling
mistake; any painted pixel = 1). Also produces full-resolution EDGE-ONLY alignment
overlays for 3 slices. Does NOT redo the level0 CT sweep (extract2.py's sweep_l0/ is
orientation/label-independent and stays valid)."""
import json
import numpy as np
import zarr
from pathlib import Path
from PIL import Image
from scipy.ndimage import zoom, distance_transform_edt, binary_erosion

import os
os.environ.setdefault("BLOSC_NTHREADS", "2")

REG_DIR = Path("/tmp/claude-1000/-home-seth-ScrollPrizeTutorial/2009ddb2-9fa9-4865-b0ef-fcf612867acf/scratchpad/mush/reg_out_d4")
LBL_DIR = Path("/home/seth/ScrollPrizeTutorial/data/labels/mush/PHerc0125")
RAW_DIR = Path("/home/seth/ScrollPrizeTutorial/data/Scrolls/crushed scroll label")
OUT = Path("/tmp/claude-1000/-home-seth-ScrollPrizeTutorial/2009ddb2-9fa9-4865-b0ef-fcf612867acf/scratchpad/mush_bundle3")
OUT.mkdir(exist_ok=True)
EDGE_OUT = Path("/tmp/claude-1000/-home-seth-ScrollPrizeTutorial/2009ddb2-9fa9-4865-b0ef-fcf612867acf/scratchpad/edge_overlays")
EDGE_OUT.mkdir(exist_ok=True)

PAINTED = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
HELD_OUT = [2000, 6000, 9000]
TRAIN = [z for z in PAINTED if z not in HELD_OUT]
EDGE_CHECK = [2000, 6000, 9000]  # all 3 held-out, full-res edge-only overlay

g0125 = zarr.open("/mnt/raid7/scroll_volume_cache/PHerc0125.zarr", mode="r")
lvl0 = g0125["0"]
VOX_UM_L0 = 9.362


def label_layer(z):
    for suf in ("label", "Label"):
        p = LBL_DIR / f"0125-z{z}spiral__{suf}.png"
        if p.exists():
            return p
    raise FileNotFoundError(z)


def oriented_canvas_alpha(z):
    """Alpha channel, canvas-cropped, then fliplr (the confirmed correct orientation
    for every slice -- see reg_out_d4/reg_summary_d4.json / reg_summary_fliplr.json)."""
    reg = json.loads((REG_DIR / f"reg_z{z}.json").read_text())
    r0, r1, c0, c1 = reg["canvas_bbox_screenshot_px"]
    alpha_full = np.array(Image.open(label_layer(z)))[..., 3]
    alpha = alpha_full[r0:r1, c0:c1]
    assert reg["best_orientation"] == "fliplr", reg["best_orientation"]
    return alpha[:, ::-1], reg


def load_native_slice_pair(z):
    alpha, reg = oriented_canvas_alpha(z)
    fs = reg["fit_scale_vox_per_screenshot_px"]
    x0, y0 = reg["match_topleft_level0_xy"]
    th, tw = alpha.shape
    x1 = int(round(x0 + tw * fs))
    y1 = int(round(y0 + th * fs))
    x0i, y0i = max(0, int(round(x0))), max(0, int(round(y0)))
    x1 = min(lvl0.shape[2], x1)
    y1 = min(lvl0.shape[1], y1)
    ct = np.asarray(lvl0[z, y0i:y1, x0i:x1])
    Hc, Wc = ct.shape
    zy, zx = Hc / th, Wc / tw

    # BINARIZE FIRST (the paint's intent was opaque, per user correction), THEN resample
    # the 0/1 mask to the level0 grid -- any blur at edges is from resampling geometry
    # only, not from paint translucency.
    painted_before_canvas_px = int((alpha > 0).sum())  # "before": raw painted-pixel count
    alpha_bin = (alpha > 0).astype(np.float32)
    alpha_l0 = zoom(alpha_bin, (zy, zx), order=1)
    alpha_l0 = alpha_l0[:Hc, :Wc]
    if alpha_l0.shape != ct.shape:
        pad_h = Hc - alpha_l0.shape[0]
        pad_w = Wc - alpha_l0.shape[1]
        alpha_l0 = np.pad(alpha_l0, ((0, max(0, pad_h)), (0, max(0, pad_w))))
        alpha_l0 = alpha_l0[:Hc, :Wc]
    y = (alpha_l0 > 0.5).astype(np.float32)  # final binary target at level0 res
    painted_after_l0_px = int(y.sum())
    return ct.astype(np.uint8), y, reg["fit_ncc"], painted_before_canvas_px, painted_after_l0_px


def painted_z_to_y(z):
    ct, y, ncc, before, after = load_native_slice_pair(z)
    material = (ct > 5)
    return ct, y, material, ncc, before, after


print("=== BEFORE/AFTER binarization painted-pixel counts (per slice) ===")
counts = {}
for z in PAINTED:
    _, y, _, ncc, before, after = painted_z_to_y(z)
    counts[z] = {"ncc": ncc, "painted_px_canvas_before_binarize": before, "painted_px_level0_after_binarize": after}
    print(f"  z={z} ncc={ncc:.3f} before(canvas,alpha>0)={before} after(level0,binary)={after}")
json.dump(counts, open(OUT / "binarize_counts.json", "w"), indent=1)

print("=== training slices (full native res, binary labels, corrected orientation) ===")
for z in TRAIN:
    ct, y, material, ncc, before, after = painted_z_to_y(z)
    np.savez_compressed(OUT / f"train_z{z}.npz", ct=ct, y=y, material=material, ncc=ncc)
    print(f"  z={z} shape={ct.shape} ncc={ncc:.3f} painted_frac={float(y.mean()):.4f}")

print("=== held-out slices (full native res) + band-around-paint mask ===")
BAND_UM = 800.0
band_px_l0 = BAND_UM / VOX_UM_L0
for z in HELD_OUT:
    ct, y, material, ncc, before, after = painted_z_to_y(z)
    painted = y > 0
    dist = distance_transform_edt(~painted) if painted.any() else np.full(ct.shape, 1e9, dtype=np.float32)
    band = (dist <= band_px_l0) & material
    np.savez_compressed(OUT / f"heldout_z{z}.npz", ct=ct, y=y, material=material, band=band, ncc=ncc)
    print(f"  z={z} shape={ct.shape} ncc={ncc:.3f} painted_px={int(painted.sum())} "
          f"material_px={int(material.sum())} band_px={int(band.sum())}")

print("=== full-resolution EDGE-ONLY alignment overlays, 3 slices ===")
for z in EDGE_CHECK:
    ct, y, material, ncc, before, after = painted_z_to_y(z)
    edge = (y.astype(bool) & ~binary_erosion(y.astype(bool), iterations=3))
    ct_u8 = np.clip(ct, 0, 255).astype(np.uint8)
    rgb = np.stack([ct_u8] * 3, axis=-1)
    rgb[edge] = [255, 30, 30]
    Image.fromarray(rgb).save(EDGE_OUT / f"edge_overlay_FULLRES_z{z}.png")
    print(f"  z={z} edge px={int(edge.sum())} saved full-res {rgb.shape[1]}x{rgb.shape[0]}")

print("=== overlay-check for 2 unpainted slices, corrected orientation ===")
for z in [12000, 18000]:
    regp = REG_DIR / f"reg_z{z}.json"
    if not regp.exists():
        print(f"  z={z} SKIPPED (registration not ready yet)")
        continue
    reg = json.loads(regp.read_text())
    r0, r1, c0, c1 = reg["canvas_bbox_screenshot_px"]
    fs = reg["fit_scale_vox_per_screenshot_px"]
    x0, y0 = reg["match_topleft_level0_xy"]
    fn = RAW_DIR / ("125-z11000spiral.png" if z == 11000 else f"0125-z{z}spiral.png")
    shot = np.array(Image.open(fn).convert("L"))[r0:r1, c0:c1][:, ::-1]  # fliplr
    th, tw = shot.shape
    x1 = int(x0 + tw * fs); y1 = int(y0 + th * fs)
    x0i, y0i = max(0, int(x0)), max(0, int(y0))
    x1 = min(lvl0.shape[2], x1); y1 = min(lvl0.shape[1], y1)
    ct = np.asarray(lvl0[z, y0i:y1, x0i:x1])
    f_ct = 1000 / ct.shape[1]
    ct_small = zoom(ct.astype(np.float32), f_ct, order=1)
    shot_small = zoom(shot.astype(np.float32), fs * f_ct, order=1)
    H = min(ct_small.shape[0], shot_small.shape[0]); W = min(ct_small.shape[1], shot_small.shape[1])
    canvas = np.zeros((H, W * 2 + 20), dtype=np.uint8)
    canvas[:, :W] = np.clip(ct_small[:H, :W], 0, 255).astype(np.uint8)
    canvas[:, W + 20:W + 20 + W] = np.clip(shot_small[:H, :W], 0, 255).astype(np.uint8)
    Image.fromarray(canvas).save(EDGE_OUT / f"check_unpainted_z{z}_ct_vs_screenshot_CORRECTED.png")
    print(f"  z={z} ncc={reg['fit_ncc']:.3f}")

print("DONE extraction3 ->", OUT)
