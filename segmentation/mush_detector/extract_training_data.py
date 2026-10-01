#!/usr/bin/env python3
"""Light I/O extraction on the hub (no training/GPU here, per CLAUDE.md hub rule).

Builds a compact .npz bundle of:
 - 8 training slices' native-res (level0) label alpha + CT, cropped to the
   registered canvas bbox (full-res, as required).
 - 1 held-out slice (z6000, mid-z, good NCC 0.78 and visually verified) at
   native res, same crop.
 - a sparse full-z sweep of PHerc0125 CT at level1 (18.72 um/px) across the
   whole volume (stride chosen for the time budget).
 - a few sample slices from PHerc0211, PHerc0191, PHerc1203, PHerc0172 at
   level1, full frame.
All CT reads only (no model compute). Single-process, modest thread count.
"""
import json
import numpy as np
import zarr
from pathlib import Path
from PIL import Image

import os
os.environ.setdefault("BLOSC_NTHREADS", "2")

REG_DIR = Path("/tmp/claude-1000/-home-seth-ScrollPrizeTutorial/2009ddb2-9fa9-4865-b0ef-fcf612867acf/scratchpad/mush/reg_out")
LBL_DIR = Path("/home/seth/ScrollPrizeTutorial/data/labels/mush/PHerc0125")
OUT = Path("/tmp/claude-1000/-home-seth-ScrollPrizeTutorial/2009ddb2-9fa9-4865-b0ef-fcf612867acf/scratchpad/mush_bundle")
OUT.mkdir(exist_ok=True)

ALL_SLICES = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
HELD_OUT = 6000  # mid-z, good NCC (0.78), visually verified alignment

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
    # native-res CT crop matching alpha's screenshot-px grid 1:1 would need
    # per-pixel resample by fs; since fs != 1 in general, resample alpha onto
    # the *level0* grid instead (alpha is screenshot px, CT is level0 vox).
    x1 = int(round(x0 + tw * fs))
    y1 = int(round(y0 + th * fs))
    x0i, y0i = max(0, int(round(x0))), max(0, int(round(y0)))
    x1 = min(lvl0.shape[2], x1)
    y1 = min(lvl0.shape[1], y1)
    ct = np.asarray(lvl0[z, y0i:y1, x0i:x1])  # level0 native res, shape (Hc, Wc)
    # resample alpha (screenshot px) onto the same level0 grid via nearest/linear zoom
    from scipy.ndimage import zoom
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


print("Extracting labelled slices at level0 native res...")
train_patches = {"ct": [], "y": [], "z": []}
rng = np.random.default_rng(0)
PATCH = 512
for z in ALL_SLICES:
    ct, alpha, ncc = load_native_slice_pair(z)
    # per-slice stretch: y = alpha / max_nonzero_alpha (never threshold)
    nz = alpha[alpha > 0]
    amax = float(nz.max()) if nz.size else 1.0
    y = np.clip(alpha / max(amax, 1.0), 0.0, 1.0).astype(np.float32)
    material = ct > 5
    print(f"  z={z} ct={ct.shape} ncc={ncc:.3f} amax={amax:.0f} painted_frac={float((alpha>0).mean()):.4f} material_frac={float(material.mean()):.3f}")

    if z == HELD_OUT:
        np.savez_compressed(OUT / f"heldout_z{z}.npz", ct=ct, y=y, material=material, ncc=ncc)
        continue

    H, W = ct.shape
    if H < PATCH or W < PATCH:
        continue
    # stratified patch centers: half near painted pixels, half uniform over material
    vs, us = np.nonzero(alpha > 0)
    centers = []
    if vs.size:
        idx = rng.choice(vs.size, size=min(5, vs.size), replace=False)
        for i in idx:
            centers.append((int(vs[i]), int(us[i])))
    mv, mu = np.nonzero(material)
    if mv.size:
        idx = rng.choice(mv.size, size=min(5, mv.size), replace=False)
        for i in idx:
            centers.append((int(mv[i]), int(mu[i])))
    for (cy, cx) in centers:
        y0p = int(np.clip(cy - PATCH // 2, 0, H - PATCH))
        x0p = int(np.clip(cx - PATCH // 2, 0, W - PATCH))
        train_patches["ct"].append(ct[y0p:y0p + PATCH, x0p:x0p + PATCH].copy())
        train_patches["y"].append(y[y0p:y0p + PATCH, x0p:x0p + PATCH].copy())
        train_patches["z"].append(z)

np.savez_compressed(
    OUT / "train_patches.npz",
    ct=np.stack(train_patches["ct"]),
    y=np.stack(train_patches["y"]),
    z=np.array(train_patches["z"]),
)
print("train patches:", len(train_patches["z"]))

# ---- sparse full-z sweep of PHerc0125, level1 (18.72 um/px), full frame ----
lvl1 = g0125["1"]
Z1, H1, W1 = lvl1.shape
STRIDE = 260  # level0-z units -> picks ~80 slices across the volume, budget-fit
zs = list(range(0, 20840, STRIDE))
print(f"level1 sweep: {len(zs)} slices, shape {H1}x{W1}")
sweep = np.zeros((len(zs), H1, W1), dtype=np.uint8)
for i, z in enumerate(zs):
    zl = min(Z1 - 1, z // 2)
    sweep[i] = np.asarray(lvl1[zl])
np.savez_compressed(OUT / "sweep_l1.npz", ct=sweep, z_level0=np.array(zs))
print("sweep saved", sweep.shape)

# ---- other scrolls, a few slices each, level1 full frame ----
OTHER = {
    "PHerc0211": [4000, 9000, 14000],
    "PHerc0191": [4000, 9000, 14000],
    "PHerc1203": [4000, 9000, 14000],
    "PHerc0172": [4000, 9000, 14000],
}
for sc, zlist in OTHER.items():
    try:
        gs = zarr.open(f"/mnt/raid7/scroll_volume_cache/{sc}.zarr", mode="r")
        l1 = gs["1"]
        Zs = l1.shape[0]
        imgs = []
        used_z = []
        for z in zlist:
            zl = min(Zs - 1, z // 2)
            imgs.append(np.asarray(l1[zl]))
            used_z.append(z)
        np.savez_compressed(OUT / f"other_{sc}.npz", ct=np.stack(imgs), z_level0=np.array(used_z))
        print(sc, "ok", imgs[0].shape)
    except Exception as e:
        print(sc, "FAILED", e)

print("DONE extraction ->", OUT)
