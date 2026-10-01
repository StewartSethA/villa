#!/usr/bin/env python3
"""Round 2: fixes applied vs round 1 (both per coordinator instruction):
  1. Per-slice intensity normalisation (z-score over each WHOLE slice's material
     region, computed once) replaces the fixed global affine -- consistent across
     every inference tile within a slice, not tile-dependent.
  2. BatchNorm2d (fixed running stats in eval()) replaces InstanceNorm2d, which
     was the actual source of the round-1 tiling checkerboard (each tile got its
     OWN per-sample normalisation statistics; BatchNorm with running stats gives
     every tile the same fixed affine at inference).
  3. Thousands of random 512x512 crops sampled ON THE FLY from 6 FULL training
     slices (not a fixed small patch pool).
  4. 3 held-out slices (low/mid/high z: 2000/6000/9000), each scored two ways:
     whole-cross-section material region, AND a band within 800 um of any
     painted pixel in that slice (painted-attention region; unpainted-but-far
     papyrus is NOT treated as confirmed non-mush, per the judged-region
     reconsideration).
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

BUNDLE = Path("/home/jacob/mush_bundle2")
OUT = Path("/home/jacob/mush_out2")
OUT.mkdir(exist_ok=True)
DEV = "cuda:0"
PATCH = 512
TRAIN_Z = [3000, 4000, 5000, 7000, 8000, 10000]
HELD_OUT_Z = [2000, 6000, 9000]


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


def auc(scores, labels):
    order = np.argsort(-scores)
    labels = labels[order].astype(np.float64)
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    tp = tp / max(tp[-1], 1)
    fp = fp / max(fp[-1], 1)
    return float(np.sum((fp[1:] - fp[:-1]) * (tp[1:] + tp[:-1]) / 2.0))


class SmallUNetBN(nn.Module):
    """Same topology as round 1's SmallUNet, InstanceNorm2d -> BatchNorm2d."""
    def __init__(self, ch=24):
        super().__init__()
        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
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


# ---------------- load training slices (full res) ----------------
train_slices = []
for z in TRAIN_Z:
    d = np.load(BUNDLE / f"train_z{z}.npz")
    ct, y, material = d["ct"], d["y"], d["material"]
    # per-slice z-score, computed ONCE over the whole slice's material region
    mct = ct[material].astype(np.float32)
    mu, sd = float(mct.mean()), float(mct.std() + 1e-6)
    ct_n = (ct.astype(np.float32) - mu) / sd
    train_slices.append({"z": z, "ct": ct_n, "y": y, "material": material, "mu": mu, "sd": sd, "H": ct.shape[0], "W": ct.shape[1]})
    print(f"train z={z} shape={ct.shape} mu={mu:.1f} sd={sd:.1f} painted_frac={float((y>0).mean()):.4f}")

# bias crop sampling toward the material bbox of each slice (many fewer all-air crops)
for s in train_slices:
    ys, xs = np.nonzero(s["material"])
    s["bbox"] = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())) if ys.size else (0, s["H"], 0, s["W"])

model = SmallUNetBN().to(DEV)
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2000)

BS = 16
BUDGET_S = 450  # ~7.5 min
rng = np.random.default_rng(0)


def sample_batch():
    xb = np.empty((BS, 1, PATCH, PATCH), dtype=np.float32)
    yb = np.empty((BS, 1, PATCH, PATCH), dtype=np.float32)
    wb = np.empty((BS, 1, PATCH, PATCH), dtype=np.float32)
    for i in range(BS):
        s = train_slices[rng.integers(0, len(train_slices))]
        y0, y1, x0, x1 = s["bbox"]
        cy = rng.integers(max(y0, 0), max(y1 - PATCH, y0 + 1)) if y1 - y0 > PATCH else max(0, (y0 + y1 - PATCH) // 2)
        cx = rng.integers(max(x0, 0), max(x1 - PATCH, x0 + 1)) if x1 - x0 > PATCH else max(0, (x0 + x1 - PATCH) // 2)
        cy = int(np.clip(cy, 0, max(s["H"] - PATCH, 0)))
        cx = int(np.clip(cx, 0, max(s["W"] - PATCH, 0)))
        crop_ct = s["ct"][cy:cy + PATCH, cx:cx + PATCH]
        crop_y = s["y"][cy:cy + PATCH, cx:cx + PATCH]
        crop_m = s["material"][cy:cy + PATCH, cx:cx + PATCH].astype(np.float32)
        if crop_ct.shape != (PATCH, PATCH):
            pad = ((0, PATCH - crop_ct.shape[0]), (0, PATCH - crop_ct.shape[1]))
            crop_ct = np.pad(crop_ct, pad)
            crop_y = np.pad(crop_y, pad)
            crop_m = np.pad(crop_m, pad)
        if rng.random() < 0.5:
            crop_ct, crop_y, crop_m = crop_ct[:, ::-1], crop_y[:, ::-1], crop_m[:, ::-1]
        if rng.random() < 0.5:
            crop_ct, crop_y, crop_m = crop_ct[::-1], crop_y[::-1], crop_m[::-1]
        xb[i, 0] = crop_ct
        yb[i, 0] = crop_y
        wb[i, 0] = crop_m
    return (torch.from_numpy(xb.copy()).to(DEV), torch.from_numpy(yb.copy()).to(DEV), torch.from_numpy(wb.copy()).to(DEV))


t0 = time.time()
step = 0
losses = []
n_patches_seen = 0
model.train()
while time.time() - t0 < BUDGET_S:
    xb, yb, wb = sample_batch()
    opt.zero_grad()
    logit = model(xb)
    p = clamp01_straight_through(logit)
    loss = soft_mse(p, yb, wb)
    loss.backward()
    opt.step()
    if step < 2000:
        sched.step()
    step += 1
    n_patches_seen += BS
    losses.append(float(loss.detach()))

elapsed = time.time() - t0
mvox_s = n_patches_seen * PATCH * PATCH / 1e6 / elapsed
print(f"trained {step} steps ({n_patches_seen} patches) in {elapsed:.1f}s, final train soft-MSE {np.mean(losses[-50:]):.5f}, throughput {mvox_s:.2f} MVox/s")
torch.save(model.state_dict(), OUT / "mush_detector_bn_unet.pt")
json.dump({"train_steps": step, "n_patches": n_patches_seen, "train_seconds": elapsed, "throughput_mvox_s": mvox_s,
           "final_train_soft_mse": float(np.mean(losses[-50:])), "train_z": TRAIN_Z, "held_out_z": HELD_OUT_Z,
           "batch_size": BS, "patch_px": PATCH},
          open(OUT / "train_meta.json", "w"), indent=1)

# ---------------- held-out validation, 3 slices, two judged regions ----------------
model.eval()


def infer_full(ct_n, H, W_):
    ys = sorted(set(list(range(0, max(H - PATCH, 1), PATCH)) + [max(H - PATCH, 0)]))
    xs = sorted(set(list(range(0, max(W_ - PATCH, 1), PATCH)) + [max(W_ - PATCH, 0)]))
    pred = np.zeros((H, W_), dtype=np.float32)
    with torch.no_grad():
        for yy in ys:
            for xx in xs:
                patch = ct_n[yy:yy + PATCH, xx:xx + PATCH]
                ph, pw = patch.shape
                t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(DEV)
                if ph < PATCH or pw < PATCH:
                    t = F.pad(t, (0, PATCH - pw, 0, PATCH - ph))
                out = clamp01_straight_through(model(t))[0, 0].cpu().numpy()
                pred[yy:yy + ph, xx:xx + pw] = out[:ph, :pw]
    return pred


results = {"train_meta": {"train_steps": step, "train_seconds": elapsed, "throughput_mvox_s": mvox_s,
                           "n_patches": n_patches_seen, "train_z": TRAIN_Z},
           "held_out": {}}
for z in HELD_OUT_Z:
    d = np.load(BUNDLE / f"heldout_z{z}.npz")
    ct, y, material, band, ncc = d["ct"], d["y"], d["material"], d["band"], float(d["ncc"])
    mct = ct[material].astype(np.float32)
    mu, sd = float(mct.mean()), float(mct.std() + 1e-6)
    ct_n = (ct.astype(np.float32) - mu) / sd
    pred = infer_full(ct_n, *ct.shape)
    np.save(OUT / f"heldout_pred_z{z}.npy", pred.astype(np.float16))

    thresh = 0.1
    entry = {"z_level0": z, "ncc": ncc, "pixel_count_material": int(material.sum()),
             "pixel_count_band": int(band.sum()), "pixel_count_total": int(ct.size),
             "positive_px_thresh_0.1_in_material": int((y[material] > thresh).sum()),
             "positive_px_thresh_0.1_in_band": int((y[band] > thresh).sum()) if band.any() else 0}
    for region_name, region in (("whole_material", material), ("band_800um_around_paint", band)):
        if region.sum() == 0:
            entry[region_name] = {"note": "empty region"}
            continue
        yj = y[region]; pj = pred[region]; ctj = ct[region].astype(np.float32)
        bin_y = (yj > thresh).astype(np.float32)
        model_auc = auc(pj, bin_y) if 0 < bin_y.sum() < bin_y.size else float("nan")
        ct_auc = auc(ctj, bin_y) if 0 < bin_y.sum() < bin_y.size else float("nan")
        from scipy.ndimage import uniform_filter
        mean7 = uniform_filter(ct.astype(np.float32), 7)
        var7 = uniform_filter(ct.astype(np.float32) ** 2, 7) - mean7 ** 2
        var_auc = auc(var7[region], bin_y) if 0 < bin_y.sum() < bin_y.size else float("nan")
        entry[region_name] = {
            "n_px": int(region.sum()), "n_positive_px": int(bin_y.sum()),
            "soft_mse_model": float(np.mean((pj - yj) ** 2)),
            "soft_mse_zero_baseline": float(np.mean(yj ** 2)),
            "auc_thresh_0.1": {"model": model_auc, "ct_intensity": ct_auc, "local_variance_7x7": var_auc},
        }
    results["held_out"][str(z)] = entry
    print(z, json.dumps(entry, indent=1))

json.dump(results, open(OUT / "results2.json", "w"), indent=1)
print("HELDOUT_DONE")

# ---------------- full-resolution (level0) whole-volume sweep ----------------
sweep_dir = BUNDLE / "sweep_l0"
z_list = json.load(open(sweep_dir / "z_list.json"))
print(f"level0 sweep: {len(z_list)} slices")
t1 = time.time()
sweep_out = OUT / "sweep_pred_l0"
sweep_out.mkdir(exist_ok=True)
with torch.no_grad():
    for i, z in enumerate(z_list):
        d = np.load(sweep_dir / f"s{i:04d}.npz")
        ct = d["ct"]
        material = ct > 5
        if material.sum() > 0:
            mu, sd = float(ct[material].mean()), float(ct[material].std() + 1e-6)
        else:
            mu, sd = 60.0, 60.0
        ct_n = (ct.astype(np.float32) - mu) / sd
        pred = infer_full(ct_n, *ct.shape)
        np.savez_compressed(sweep_out / f"p{i:04d}.npz", pred=(np.clip(pred * 255, 0, 255)).astype(np.uint8), z=z)
        if i % 20 == 0:
            print(f"  sweep {i}/{len(z_list)} z={z} elapsed={time.time()-t1:.0f}s", flush=True)
print(f"level0 sweep inference done in {time.time()-t1:.0f}s")
print("SWEEP_DONE")

# ---------------- other-scroll samples: re-infer round-1's CT crops with the new model ----------------
for f in sorted(Path("/home/jacob/mush_bundle").glob("other_*.npz")):
    name = f.stem.replace("other_", "")
    d = np.load(f)
    imgs, zs = d["ct"], d["z_level0"]
    preds = np.zeros_like(imgs, dtype=np.uint8)
    for i in range(imgs.shape[0]):
        img = imgs[i]
        m = img > 5
        mu, sd = (float(img[m].mean()), float(img[m].std() + 1e-6)) if m.sum() else (60.0, 60.0)
        img_n = (img.astype(np.float32) - mu) / sd
        pred = infer_full(img_n, *img.shape)
        preds[i] = np.clip(pred * 255, 0, 255).astype(np.uint8)
    np.savez_compressed(OUT / f"other_pred_{name}.npz", pred=preds, ct=imgs, z_level0=zs)
    print(name, "inferred", imgs.shape)

print("ALL DONE")
