#!/usr/bin/env python3
"""Loads the already-trained wide_LAST.pt checkpoint and runs ONLY the full-res
level0 sweep + other-scroll inference, with NON-overlapping tiles (stride=PATCH)
for speed -- training and held-out validation already completed and were saved
with proper overlap-blended inference; this sweep is the exploratory full-volume
view and does not need the same inference quality."""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BUNDLE = Path("/home/seth/mush_bundle3")
OUT = Path("/home/seth/mush_out3")
DEV = "cuda:0"
PATCH = 512
CTX_WIN = 2048


def clamp01_straight_through(x):
    class _F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.clamp(0, 1)

        @staticmethod
        def backward(ctx, g):
            return g
    return _F.apply(x)


def conv_block(ci, co, dilation=1):
    pad = dilation
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, padding=pad, dilation=dilation), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
        nn.Conv2d(co, co, 3, padding=pad, dilation=dilation), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
    )


class SmallUNetBN(nn.Module):
    IN_CH = 1

    def __init__(self, ch=24):
        super().__init__()
        self.e1 = conv_block(1, ch)
        self.e2 = conv_block(ch, ch * 2)
        self.e3 = conv_block(ch * 2, ch * 4)
        self.b = conv_block(ch * 4, ch * 4)
        self.d3 = conv_block(ch * 4 + ch * 4, ch * 2)
        self.d2 = conv_block(ch * 2 + ch * 2, ch)
        self.d1 = conv_block(ch + ch, ch)
        self.out = nn.Conv2d(ch, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.b(self.pool(e3))
        d3 = self.d3(torch.cat([F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False), e1], 1))
        return self.out(d1)


def normalize_slice(ct, material):
    mct = ct[material].astype(np.float32) if material.sum() else ct.astype(np.float32)
    mu, sd = float(mct.mean()), float(mct.std() + 1e-6)
    return (ct.astype(np.float32) - mu) / sd, mu, sd


@torch.no_grad()
def infer_fast(model, ct_full, material, stride=PATCH):
    """Non-overlapping tiled inference -- fast, for the exploratory sweep only."""
    H, W = ct_full.shape
    ct_n, _, _ = normalize_slice(ct_full, material)
    pred = np.zeros((H, W), dtype=np.float32)
    ys = sorted(set(list(range(0, max(H - PATCH, 1), stride)) + [max(H - PATCH, 0)]))
    xs = sorted(set(list(range(0, max(W - PATCH, 1), stride)) + [max(W - PATCH, 0)]))
    for yy in ys:
        for xx in xs:
            crop_ct = ct_n[yy:yy + PATCH, xx:xx + PATCH]
            ph, pw = crop_ct.shape
            xin = np.zeros((1, 1, PATCH, PATCH), dtype=np.float32)
            xin[0, 0, :ph, :pw] = crop_ct
            t = torch.from_numpy(xin).to(DEV)
            out = clamp01_straight_through(model(t))[0, 0].cpu().numpy()
            pred[yy:yy + ph, xx:xx + pw] = out[:ph, :pw]
    return pred


model = SmallUNetBN().to(DEV)
model.load_state_dict(torch.load(OUT / "small_LAST.pt", map_location=DEV))
model.eval()
print("loaded small_LAST.pt")

sweep_dir = BUNDLE.parent / "mush_bundle2" / "sweep_l0"
z_list = json.load(open(sweep_dir / "z_list.json"))
print(f"level0 sweep (non-overlap, fast): {len(z_list)} slices")
t1 = time.time()
sweep_out = OUT / "sweep_pred_l0_small"
sweep_out.mkdir(exist_ok=True)
for i, z in enumerate(z_list):
    outp = sweep_out / f"p{i:04d}.npz"
    if outp.exists():
        continue
    d = np.load(sweep_dir / f"s{i:04d}.npz")
    ct = d["ct"]
    material = ct > 5
    pred = infer_fast(model, ct, material)
    np.savez_compressed(outp, pred=(np.clip(pred * 255, 0, 255)).astype(np.uint8), z=z)
    if i % 10 == 0:
        print(f"  sweep {i}/{len(z_list)} z={z} elapsed={time.time()-t1:.0f}s", flush=True)
print(f"level0 sweep done in {time.time()-t1:.0f}s")
print("SWEEP_DONE")

for f in sorted(Path("/home/seth/mush_bundle").glob("other_*.npz")):
    name = f.stem.replace("other_", "")
    d = np.load(f)
    imgs, zs = d["ct"], d["z_level0"]
    preds = np.zeros_like(imgs, dtype=np.uint8)
    for i in range(imgs.shape[0]):
        img = imgs[i]
        m = img > 5
        pred = infer_fast(model, img, m)
        preds[i] = np.clip(pred * 255, 0, 255).astype(np.uint8)
    np.savez_compressed(OUT / f"other_pred_small_{name}.npz", pred=preds, ct=imgs, z_level0=zs)
    print(name, "inferred", imgs.shape)

print("ALL DONE")
