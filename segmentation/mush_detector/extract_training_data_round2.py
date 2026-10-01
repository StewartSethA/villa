#!/usr/bin/env python3
"""Round 2 extraction (light I/O on hub, no GPU/training here):
 - full native-res (level0) CT + soft-label for all 9 PAINTED slices, split into
   6 training + 3 held-out (low/mid/high: z=2000,6000,9000 held out)
 - held-out band-around-paint mask (distance transform of alpha>0)
 - overlay-check images for 2 of the 10 newly-registered (unpainted) slices
 - full-resolution (level0) whole-volume CT sweep, z-stride 100 (<=100 per the ask)
 - other-scroll samples: REUSED from round 1 bundle (not re-extracted)
"""
import json
import numpy as np
import zarr
from pathlib import Path
from PIL import Image
from scipy.ndimage import zoom, distance_transform_edt

import os
os.environ.setdefault("BLOSC_NTHREADS", "2")

REG_DIR = Path("/tmp/claude-1000/-home-seth-ScrollPrizeTutorial/2009ddb2-9fa9-4865-b0ef-fcf612867acf/scratchpad/mush/reg_out")
LBL_DIR = Path("/home/seth/ScrollPrizeTutorial/data/labels/mush/PHerc0125")
RAW_DIR = Path("/home/seth/ScrollPrizeTutorial/data/Scrolls/crushed scroll label")
OUT = Path("/tmp/claude-1000/-home-seth-ScrollPrizeTutorial/2009ddb2-9fa9-4865-b0ef-fcf612867acf/scratchpad/mush_bundle2")
OUT.mkdir(exist_ok=True)
OVERLAY_OUT = Path("/tmp/claude-1000/-home-seth-ScrollPrizeTutorial/2009ddb2-9fa9-4865-b0ef-fcf612867acf/scratchpad/overlays2")
OVERLAY_OUT.mkdir(exist_ok=True)

PAINTED = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
HELD_OUT = [2000, 6000, 9000]  # low / mid / high within the labelled z-range
TRAIN = [z for z in PAINTED if z not in HELD_OUT]
UNPAINTED_NEW = [11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000]
OVERLAY_CHECK_NEW = [12000, 18000]  # 2 of the 10, spread apart

g0125 = zarr.open("/mnt/raid7/scroll_volume_cache/PHerc0125.zarr", mode="r")
lvl0 = g0125["0"]
VOX_UM_L0 = 9.362


def label_layer(z):
    for suf in ("label", "Label"):
        p = LBL_DIR / f"0125-z{z}spiral__{suf}.png"
        if p.exists():
            return p
    raise FileNotFoundError(z)


def load_native_slice_pair(z):
    reg = json.loads((REG_DIR / f"reg_z{z}.json").read_text())
    fs = reg["fit_scale_vox_per_screenshot_px"]
    x0, y0 = reg["match_topleft_level0_xy"]
    r0, r1, c0, c1 = reg["canvas_bbox_screenshot_px"]
    alpha_full = np.array(Image.open(label_layer(z)))[..., 3]
    alpha = alpha_full[r0:r1, c0:c1]
    th, tw = alpha.shape
    x1 = int(round(x0 + tw * fs))
    y1 = int(round(y0 + th * fs))
    x0i, y0i = max(0, int(round(x0))), max(0, int(round(y0)))
    x1 = min(lvl0.shape[2], x1)
    y1 = min(lvl0.shape[1], y1)
    ct = np.asarray(lvl0[z, y0i:y1, x0i:x1])
    Hc, Wc = ct.shape
    zy, zx = Hc / th, Wc / tw
    alpha_l0 = zoom(alpha.astype(np.float32), (zy, zx), order=1)
    alpha_l0 = alpha_l0[:Hc, :Wc]
    if alpha_l0.shape != ct.shape:
        pad_h = Hc - alpha_l0.shape[0]
        pad_w = Wc - alpha_l0.shape[1]
        alpha_l0 = np.pad(alpha_l0, ((0, max(0, pad_h)), (0, max(0, pad_w))))
        alpha_l0 = alpha_l0[:Hc, :Wc]
    return ct.astype(np.uint8), alpha_l0.astype(np.float32), reg["fit_ncc"]


def painted_z_to_y(z):
    ct, alpha, ncc = load_native_slice_pair(z)
    nz = alpha[alpha > 0]
    amax = float(nz.max()) if nz.size else 1.0
    y = np.clip(alpha / max(amax, 1.0), 0.0, 1.0).astype(np.float32)
    material = (ct > 5)
    return ct, y, material, ncc, amax


print("=== training slices (full native res) ===")
for z in TRAIN:
    ct, y, material, ncc, amax = painted_z_to_y(z)
    np.savez_compressed(OUT / f"train_z{z}.npz", ct=ct, y=y, material=material, ncc=ncc, amax=amax)
    print(f"  z={z} shape={ct.shape} ncc={ncc:.3f} amax={amax:.0f} painted_frac={float((y>0).mean()):.4f}")

print("=== held-out slices (full native res) + band-around-paint mask ===")
BAND_UM = 800.0
band_px_l0 = BAND_UM / VOX_UM_L0
for z in HELD_OUT:
    ct, y, material, ncc, amax = painted_z_to_y(z)
    painted = y > 0
    if painted.any():
        dist = distance_transform_edt(~painted)
    else:
        dist = np.full(ct.shape, 1e9, dtype=np.float32)
    band = (dist <= band_px_l0) & material
    np.savez_compressed(OUT / f"heldout_z{z}.npz", ct=ct, y=y, material=material, band=band, ncc=ncc, amax=amax)
    print(f"  z={z} shape={ct.shape} ncc={ncc:.3f} painted_px={int(painted.sum())} "
          f"material_px={int(material.sum())} band_px={int(band.sum())} (band={BAND_UM}um={band_px_l0:.1f}px)")

print("=== overlay-check images for 2 of the 10 newly-registered (unpainted) slices ===")
for z in OVERLAY_CHECK_NEW:
    reg = json.loads((REG_DIR / f"reg_z{z}.json").read_text())
    fs = reg["fit_scale_vox_per_screenshot_px"]
    x0, y0 = reg["match_topleft_level0_xy"]
    r0, r1, c0, c1 = reg["canvas_bbox_screenshot_px"]
    fn = RAW_DIR / ("125-z11000spiral.png" if z == 11000 else f"0125-z{z}spiral.png")
    shot = np.array(Image.open(fn).convert("L"))[r0:r1, c0:c1]
    th, tw = shot.shape
    x1 = int(x0 + tw * fs); y1 = int(y0 + th * fs)
    x0i, y0i = max(0, int(x0)), max(0, int(y0))
    x1 = min(lvl0.shape[2], x1); y1 = min(lvl0.shape[1], y1)
    ct = np.asarray(lvl0[z, y0i:y1, x0i:x1])
    target_w = 1000
    f_ct = target_w / ct.shape[1]
    ct_small = zoom(ct.astype(np.float32), f_ct, order=1)
    shot_small = zoom(shot.astype(np.float32), fs * f_ct, order=1)
    H = min(ct_small.shape[0], shot_small.shape[0])
    W = min(ct_small.shape[1], shot_small.shape[1])
    # side by side: registered screenshot-gray vs CT, to confirm the SAME scan is being matched
    canvas = np.zeros((H, W * 2 + 20), dtype=np.uint8)
    canvas[:, :W] = np.clip(ct_small[:H, :W], 0, 255).astype(np.uint8)
    canvas[:, W + 20:W + 20 + W] = np.clip(shot_small[:H, :W], 0, 255).astype(np.uint8)
    Image.fromarray(canvas).save(OVERLAY_OUT / f"check_unpainted_z{z}_ct_vs_screenshot.png")
    print(f"  z={z} ncc={reg['fit_ncc']:.3f} -> {OVERLAY_OUT}/check_unpainted_z{z}_ct_vs_screenshot.png")

print("=== level0 (full native res) whole-volume sweep, z-stride 100 ===")
STRIDE = 100
zs = list(range(0, 20840, STRIDE))
print(f"{len(zs)} slices at level0 native res")
import time
t0 = time.time()
sweep_dir = OUT / "sweep_l0"
sweep_dir.mkdir(exist_ok=True)
z_list = []
for i, z in enumerate(zs):
    s = np.asarray(lvl0[z])
    np.savez_compressed(sweep_dir / f"s{i:04d}.npz", ct=s, z=z)
    z_list.append(z)
    if i % 20 == 0:
        print(f"  extracted {i}/{len(zs)} z={z} elapsed={time.time()-t0:.0f}s", flush=True)
json.dump(z_list, open(sweep_dir / "z_list.json", "w"))
print(f"level0 sweep extraction done in {time.time()-t0:.0f}s, {len(zs)} slices")
print("DONE extraction2 ->", OUT)
