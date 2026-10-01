#!/usr/bin/env python3
"""Train a small 2-D mush detector on PHerc0125 labelled cross-sections,
validate on a held-out slice, run sparse whole-volume + cross-scroll inference.

Soft labels: paint alpha stretched to [0,1] by its own per-slice max (never
thresholded). Output head: straight-through clamp. Loss: masked soft MSE
(dense_head.py's clamp01_straight_through / soft_mse, reimplemented here to
keep this a standalone, self-contained training script for the villa branch).
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)

BUNDLE = Path("/home/jacob/mush_bundle")
OUT = Path("/home/jacob/mush_out")
OUT.mkdir(exist_ok=True)
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


def soft_mse(p, y, w):
    se = (p - y) ** 2
    return (se * w).sum() / w.clamp_min(1.0).sum()


class SmallUNet(nn.Module):
    """~1.1 M params, 4 levels, instance norm. 2-D, single-slice input."""
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
    # simple fixed affine: CT uint8 -> roughly zero-mean unit-ish range
    return (ct.astype(np.float32) - 60.0) / 60.0


# ---------------- data ----------------
tp = np.load(BUNDLE / "train_patches.npz")
ct_all, y_all, z_all = tp["ct"], tp["y"], tp["z"]
print("train patches:", ct_all.shape, "zs:", sorted(set(z_all.tolist())))

ho = np.load(BUNDLE / "heldout_z6000.npz")
ho_ct, ho_y, ho_material = ho["ct"], ho["y"], ho["material"]
print("held-out z=6000:", ho_ct.shape, "ncc=", float(ho["ncc"]))

X = torch.from_numpy(normalize_ct(ct_all)).unsqueeze(1).to(DEV)
Y = torch.from_numpy(y_all).unsqueeze(1).to(DEV)
# judged region = material (papyrus), the whole slice the annotator reviewed,
# not just painted pixels -- see README for the reasoning (differs from the
# ink-label rule, which masks to painted-only because ink masks are NOT
# comprehensively reviewed).
W = (torch.from_numpy((ct_all > 5).astype(np.float32))).unsqueeze(1).to(DEV)

model = SmallUNet().to(DEV)
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=400)

n = X.shape[0]
BS = 8
t0 = time.time()
BUDGET_S = 420  # ~7 min train budget
losses = []
step = 0
while time.time() - t0 < BUDGET_S:
    perm = torch.randperm(n, device=DEV)
    for i in range(0, n, BS):
        idx = perm[i:i + BS]
        xb, yb, wb = X[idx], Y[idx], W[idx]
        # light augmentation: random flip
        if np.random.rand() < 0.5:
            xb = xb.flip(-1); yb = yb.flip(-1); wb = wb.flip(-1)
        if np.random.rand() < 0.5:
            xb = xb.flip(-2); yb = yb.flip(-2); wb = wb.flip(-2)
        opt.zero_grad()
        logit = model(xb)
        p = clamp01_straight_through(logit)
        loss = soft_mse(p, yb, wb)
        loss.backward()
        opt.step()
        sched.step()
        step += 1
        losses.append(float(loss))
    if time.time() - t0 > BUDGET_S:
        break
print(f"trained {step} steps in {time.time()-t0:.1f}s, final train soft-MSE {np.mean(losses[-20:]):.5f}")

MVOX = (n * ct_all.shape[1] * ct_all.shape[2] * step * BS / n) / 1e6  # rough
throughput_mvox_s = (step * BS * ct_all.shape[1] * ct_all.shape[2]) / 1e6 / (time.time() - t0)
print(f"throughput ~{throughput_mvox_s:.2f} MVox/s (train, 2-D, RTX 4060 Ti)")

torch.save(model.state_dict(), OUT / "mush_detector_small_unet.pt")

# ---------------- held-out validation ----------------
model.eval()
PATCH = 512
H, W_ = ho_ct.shape
ys = list(range(0, H - PATCH, PATCH)) + ([H - PATCH] if H > PATCH else [0])
xs = list(range(0, W_ - PATCH, PATCH)) + ([W_ - PATCH] if W_ > PATCH else [0])
pred_full = np.zeros((H, W_), dtype=np.float32)
with torch.no_grad():
    for yy in sorted(set(ys)):
        for xx in sorted(set(xs)):
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

# AUC against a stated threshold (0.1 of the stretched soft label = "any visible paint")
def auc(scores, labels):
    # manual trapezoidal rule: numpy>=2.0 removed numpy.trapz
    order = np.argsort(-scores)
    labels = labels[order].astype(np.float64)
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    tp = tp / max(tp[-1], 1)
    fp = fp / max(fp[-1], 1)
    return float(np.sum((fp[1:] - fp[:-1]) * (tp[1:] + tp[:-1]) / 2.0))

thresh = 0.1
bin_y = (yj > thresh).astype(np.float32)
model_auc = auc(pj, bin_y) if bin_y.sum() > 0 and bin_y.sum() < bin_y.size else float("nan")

# untrained baselines: raw CT intensity, local variance
ct_j = ho_ct[judged].astype(np.float32)
ct_auc = auc(ct_j, bin_y)

from scipy.ndimage import generic_filter, uniform_filter
ct_f = ho_ct.astype(np.float32)
mean5 = uniform_filter(ct_f, 7)
var5 = uniform_filter(ct_f * ct_f, 7) - mean5 * mean5
var_j = var5[judged]
var_auc = auc(var_j, bin_y)

results = {
    "held_out_z_level0": 6000,
    "held_out_ncc": float(ho["ncc"]),
    "pixel_count_judged_material": npix,
    "pixel_count_total_slice": int(ho_ct.size),
    "positive_px_at_thresh_0.1": int(bin_y.sum()),
    "soft_mse_model": soft_mse_val,
    "soft_mse_baseline_zero": float(np.mean(yj ** 2)),
    "auc_thresh_0.1": {"model": model_auc, "ct_intensity_baseline": ct_auc, "local_variance_7x7_baseline": var_auc},
    "train_steps": step,
    "train_seconds": time.time() - t0,
    "throughput_mvox_s_train": throughput_mvox_s,
    "n_train_patches": int(n),
    "patch_px": PATCH,
}
print(json.dumps(results, indent=2))
(OUT / "results.json").write_text(json.dumps(results, indent=2))
np.save(OUT / "heldout_pred_z6000.npy", pred_full.astype(np.float16))

# ---------------- sparse full-z sweep inference ----------------
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
        ys_ = list(range(0, max(H - PATCH, 1), PATCH)) + [max(H - PATCH, 0)]
        xs_ = list(range(0, max(W_ - PATCH, 1), PATCH)) + [max(W_ - PATCH, 0)]
        for yy in sorted(set(ys_)):
            for xx in sorted(set(xs_)):
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

# ---------------- other scrolls (unvalidated) ----------------
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
