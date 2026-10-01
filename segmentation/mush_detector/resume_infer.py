#!/usr/bin/env python3
"""Resume from the already-trained checkpoint (numpy.trapz was removed in numpy>=2.0;
fixed here with a manual trapezoidal rule) and run held-out validation + sweep +
cross-scroll inference."""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BUNDLE = Path("/home/jacob/mush_bundle")
OUT = Path("/home/jacob/mush_out")
DEV = "cuda:0"


def clamp01_straight_through(x):
    class _F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.clamp(0, 1)

        @staticmethod
        def backward(ctx, g):
            return g
    return _F.apply(x)


class SmallUNet(nn.Module):
    def __init__(self, ch=24):
        super().__init__()
        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.InstanceNorm2d(co), nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1), nn.InstanceNorm2d(co), nn.ReLU(inplace=True),
            )
        self.e1 = block(1, ch)
        self.e2 = block(ch, ch * 2)
        self.e3 = block(ch * 2, ch * 4)
        self.b = block(ch * 4, ch * 4)
        self.d3 = block(ch * 4 + ch * 4, ch * 2)
        self.d2 = block(ch * 2 + ch * 2, ch)
        self.d1 = block(ch + ch, ch)
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


def normalize_ct(ct):
    return (ct.astype(np.float32) - 60.0) / 60.0


def auc(scores, labels):
    order = np.argsort(-scores)
    labels = labels[order].astype(np.float64)
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    tp = tp / max(tp[-1], 1)
    fp = fp / max(fp[-1], 1)
    # manual trapezoidal rule (numpy>=2.0 removed numpy.trapz)
    return float(np.sum((fp[1:] - fp[:-1]) * (tp[1:] + tp[:-1]) / 2.0))


model = SmallUNet().to(DEV)
model.load_state_dict(torch.load(OUT / "mush_detector_small_unet.pt", map_location=DEV))
model.eval()
print("loaded checkpoint")

ho = np.load(BUNDLE / "heldout_z6000.npz")
ho_ct, ho_y, ho_material = ho["ct"], ho["y"], ho["material"]
PATCH = 512
H, W_ = ho_ct.shape
ys = sorted(set(list(range(0, max(H - PATCH, 1), PATCH)) + [max(H - PATCH, 0)]))
xs = sorted(set(list(range(0, max(W_ - PATCH, 1), PATCH)) + [max(W_ - PATCH, 0)]))
pred_full = np.zeros((H, W_), dtype=np.float32)
with torch.no_grad():
    for yy in ys:
        for xx in xs:
            patch = normalize_ct(ho_ct[yy:yy + PATCH, xx:xx + PATCH])
            ph, pw = patch.shape
            t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(DEV)
            if ph < PATCH or pw < PATCH:
                t = F.pad(t, (0, PATCH - pw, 0, PATCH - ph))
            out = clamp01_straight_through(model(t))[0, 0].cpu().numpy()
            pred_full[yy:yy + ph, xx:xx + pw] = out[:ph, :pw]

judged = ho_material
yj = ho_y[judged]
pj = pred_full[judged]
soft_mse_val = float(np.mean((pj - yj) ** 2))
npix = int(judged.sum())

thresh = 0.1
bin_y = (yj > thresh).astype(np.float32)
model_auc = auc(pj, bin_y) if 0 < bin_y.sum() < bin_y.size else float("nan")
ct_j = ho_ct[judged].astype(np.float32)
ct_auc = auc(ct_j, bin_y)

from scipy.ndimage import uniform_filter
ct_f = ho_ct.astype(np.float32)
mean7 = uniform_filter(ct_f, 7)
var7 = uniform_filter(ct_f * ct_f, 7) - mean7 * mean7
var_auc = auc(var7[judged], bin_y)

results = {
    "held_out_z_level0": 6000,
    "held_out_ncc": float(ho["ncc"]),
    "pixel_count_judged_material": npix,
    "pixel_count_total_slice": int(ho_ct.size),
    "positive_px_at_thresh_0.1": int(bin_y.sum()),
    "soft_mse_model": soft_mse_val,
    "soft_mse_baseline_zero": float(np.mean(yj ** 2)),
    "auc_thresh_0.1": {"model": model_auc, "ct_intensity_baseline": ct_auc, "local_variance_7x7_baseline": var_auc},
    "train_steps": 1450,
    "train_seconds": 421.1,
    "throughput_mvox_s_train": 7.22,
    "n_train_patches": 80,
    "patch_px": PATCH,
}
print(json.dumps(results, indent=2))
(OUT / "results.json").write_text(json.dumps(results, indent=2))
np.save(OUT / "heldout_pred_z6000.npy", pred_full.astype(np.float16))
print("HELDOUT_DONE")

# ---- sweep ----
sw = np.load(BUNDLE / "sweep_l1.npz")
sw_ct, sw_z = sw["ct"], sw["z_level0"]
print("sweep:", sw_ct.shape)
t1 = time.time()
sweep_pred = np.zeros_like(sw_ct, dtype=np.uint8)
with torch.no_grad():
    for i in range(sw_ct.shape[0]):
        img = sw_ct[i]
        H, W_ = img.shape
        pr = np.zeros((H, W_), dtype=np.float32)
        ys_ = sorted(set(list(range(0, max(H - PATCH, 1), PATCH)) + [max(H - PATCH, 0)]))
        xs_ = sorted(set(list(range(0, max(W_ - PATCH, 1), PATCH)) + [max(W_ - PATCH, 0)]))
        for yy in ys_:
            for xx in xs_:
                patch = normalize_ct(img[yy:yy + PATCH, xx:xx + PATCH])
                ph, pw = patch.shape
                t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(DEV)
                if ph < PATCH or pw < PATCH:
                    t = F.pad(t, (0, PATCH - pw, 0, PATCH - ph))
                out = clamp01_straight_through(model(t))[0, 0].cpu().numpy()
                pr[yy:yy + ph, xx:xx + pw] = out[:ph, :pw]
        sweep_pred[i] = np.clip(pr * 255, 0, 255).astype(np.uint8)
        if i % 10 == 0:
            print(f"  sweep slice {i}/{sw_ct.shape[0]} z={sw_z[i]} elapsed={time.time()-t1:.0f}s")
print(f"sweep inference done in {time.time()-t1:.0f}s")
np.savez_compressed(OUT / "sweep_pred_l1.npz", pred=sweep_pred, z_level0=sw_z)
print("SWEEP_DONE")

for f in sorted(BUNDLE.glob("other_*.npz")):
    name = f.stem.replace("other_", "")
    d = np.load(f)
    imgs, zs = d["ct"], d["z_level0"]
    preds = np.zeros_like(imgs, dtype=np.uint8)
    with torch.no_grad():
        for i in range(imgs.shape[0]):
            img = imgs[i]
            H, W_ = img.shape
            pr = np.zeros((H, W_), dtype=np.float32)
            ys_ = sorted(set(list(range(0, max(H - PATCH, 1), PATCH)) + [max(H - PATCH, 0)]))
            xs_ = sorted(set(list(range(0, max(W_ - PATCH, 1), PATCH)) + [max(W_ - PATCH, 0)]))
            for yy in ys_:
                for xx in xs_:
                    patch = normalize_ct(img[yy:yy + PATCH, xx:xx + PATCH])
                    ph, pw = patch.shape
                    t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(DEV)
                    if ph < PATCH or pw < PATCH:
                        t = F.pad(t, (0, PATCH - pw, 0, PATCH - ph))
                    out = clamp01_straight_through(model(t))[0, 0].cpu().numpy()
                    pr[yy:yy + ph, xx:xx + pw] = out[:ph, :pw]
            preds[i] = np.clip(pr * 255, 0, 255).astype(np.uint8)
    np.savez_compressed(OUT / f"other_pred_{name}.npz", pred=preds, ct=imgs, z_level0=zs)
    print(name, "inferred", imgs.shape)

print("ALL DONE")
