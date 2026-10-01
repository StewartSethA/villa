#!/usr/bin/env python3
"""Round 3: binarized labels (already baked into mush_bundle3), corrected (fliplr)
registration (already baked in), BatchNorm + per-slice normalisation + overlapping-tile
blended inference (fixes the tiling artifact properly: a Hann-windowed sliding-window
average removes any residual seam, on top of the BatchNorm fix from round 2), a
wide-context 2-channel model (fine 512x512 @ 9.362um/px + a 2048x2048 window
average-pooled to 512x512, i.e. ~19.2mm of context) compared against the small
single-scale model, and measured receptive field (backprop of a center-pixel delta)
for both. Trains BEST+LAST checkpoints for each model.
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

BUNDLE = Path("/home/jacob/mush_bundle3")
OUT = Path("/home/jacob/mush_out3")
OUT.mkdir(exist_ok=True)
DEV = "cuda:0"
PATCH = 512
CTX_WIN = 2048  # physical window for the context channel, px at level0 (19.2 mm)
TRAIN_Z = [3000, 4000, 5000, 7000, 8000, 10000]
HELD_OUT_Z = [2000, 6000, 9000]
VOX_UM = 9.362


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


def conv_block(ci, co, dilation=1):
    pad = dilation
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, padding=pad, dilation=dilation), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
        nn.Conv2d(co, co, 3, padding=pad, dilation=dilation), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
    )


class SmallUNetBN(nn.Module):
    """Round 2's architecture: 3 downsample levels, single-channel (fine only) input."""
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


class WideUNetBN(nn.Module):
    """2-channel input (fine + ~19.2mm downsampled context), 4 downsample levels,
    dilated-conv bottleneck (dilation 2,4,8) for extra receptive field on the fine path."""
    IN_CH = 2

    def __init__(self, ch=24):
        super().__init__()
        self.e1 = conv_block(2, ch)
        self.e2 = conv_block(ch, ch * 2)
        self.e3 = conv_block(ch * 2, ch * 4)
        self.e4 = conv_block(ch * 4, ch * 8)
        self.b1 = conv_block(ch * 8, ch * 8, dilation=2)
        self.b2 = conv_block(ch * 8, ch * 8, dilation=4)
        self.d4 = conv_block(ch * 8 + ch * 8, ch * 4)
        self.d3 = conv_block(ch * 4 + ch * 4, ch * 2)
        self.d2 = conv_block(ch * 2 + ch * 2, ch)
        self.d1 = conv_block(ch + ch, ch)
        self.out = nn.Conv2d(ch, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.b2(self.b1(self.pool(e4)))
        d4 = self.d4(torch.cat([F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False), e4], 1))
        d3 = self.d3(torch.cat([F.interpolate(d4, scale_factor=2, mode="bilinear", align_corners=False), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False), e1], 1))
        return self.out(d1)


def measure_receptive_field(model, in_ch):
    """RF by backprop of a center-pixel delta: zero input, grad on, forward, backward
    from the center OUTPUT pixel, measure the bounding box of nonzero-gradient INPUT
    pixels on channel 0 (the fine/native-resolution channel)."""
    model.eval()
    size = 1024
    x = torch.zeros(1, in_ch, size, size, device=DEV, requires_grad=True)
    y = model(x)
    c = size // 2
    loss = y[0, 0, c, c]
    grad = torch.autograd.grad(loss, x)[0][0, 0].abs()
    nz = (grad > grad.max() * 1e-6) if grad.max() > 0 else grad > -1
    ys, xs = torch.nonzero(nz, as_tuple=True)
    if ys.numel() == 0:
        return {"rf_px": None, "rf_mm": None, "note": "zero gradient everywhere (dead net at init?)"}
    h = int(ys.max() - ys.min() + 1)
    w = int(xs.max() - xs.min() + 1)
    rf_px = max(h, w)
    return {"rf_px": rf_px, "rf_h_px": h, "rf_w_px": w, "rf_mm": round(rf_px * VOX_UM / 1000.0, 2)}


def normalize_slice(ct, material):
    mct = ct[material].astype(np.float32) if material.sum() else ct.astype(np.float32)
    mu, sd = float(mct.mean()), float(mct.std() + 1e-6)
    return (ct.astype(np.float32) - mu) / sd, mu, sd


# ---------------- load training slices ----------------
train_slices = []
for z in TRAIN_Z:
    d = np.load(BUNDLE / f"train_z{z}.npz")
    ct, y, material = d["ct"], d["y"], d["material"]
    ct_n, mu, sd = normalize_slice(ct, material)
    ys_, xs_ = np.nonzero(material)
    bbox = (int(ys_.min()), int(ys_.max()), int(xs_.min()), int(xs_.max())) if ys_.size else (0, ct.shape[0], 0, ct.shape[1])
    train_slices.append({"z": z, "ct_n": ct_n, "y": y, "material": material, "H": ct.shape[0], "W": ct.shape[1], "bbox": bbox})
    print(f"train z={z} shape={ct.shape} mu={mu:.1f} sd={sd:.1f} painted_frac={float(y.mean()):.4f}")

rng = np.random.default_rng(0)


def sample_crop_center(s):
    y0, y1, x0, x1 = s["bbox"]
    cy = rng.integers(max(y0, 0), max(y1 - PATCH, y0 + 1)) if y1 - y0 > PATCH else max(0, (y0 + y1 - PATCH) // 2)
    cx = rng.integers(max(x0, 0), max(x1 - PATCH, x0 + 1)) if x1 - x0 > PATCH else max(0, (x0 + x1 - PATCH) // 2)
    cy = int(np.clip(cy, 0, max(s["H"] - PATCH, 0)))
    cx = int(np.clip(cx, 0, max(s["W"] - PATCH, 0)))
    return cy, cx


def get_context(s, cy, cx):
    """2048x2048 window centered on the fine crop, avg-pooled 4x -> 512x512."""
    H, W = s["H"], s["W"]
    half = CTX_WIN // 2
    fcy, fcx = cy + PATCH // 2, cx + PATCH // 2
    y0 = int(np.clip(fcy - half, 0, max(H - CTX_WIN, 0)))
    x0 = int(np.clip(fcx - half, 0, max(W - CTX_WIN, 0)))
    win = s["ct_n"][y0:y0 + CTX_WIN, x0:x0 + CTX_WIN]
    if win.shape != (CTX_WIN, CTX_WIN):
        pad = ((0, CTX_WIN - win.shape[0]), (0, CTX_WIN - win.shape[1]))
        win = np.pad(win, pad, mode="edge")
    t = torch.from_numpy(win.copy()).unsqueeze(0).unsqueeze(0)
    ctx = F.avg_pool2d(t, kernel_size=CTX_WIN // PATCH).numpy()[0, 0]
    return ctx


def sample_batch(bs, in_ch):
    xb = np.empty((bs, in_ch, PATCH, PATCH), dtype=np.float32)
    yb = np.empty((bs, 1, PATCH, PATCH), dtype=np.float32)
    wb = np.empty((bs, 1, PATCH, PATCH), dtype=np.float32)
    for i in range(bs):
        s = train_slices[rng.integers(0, len(train_slices))]
        cy, cx = sample_crop_center(s)
        crop_ct = s["ct_n"][cy:cy + PATCH, cx:cx + PATCH]
        crop_y = s["y"][cy:cy + PATCH, cx:cx + PATCH]
        crop_m = s["material"][cy:cy + PATCH, cx:cx + PATCH].astype(np.float32)
        if crop_ct.shape != (PATCH, PATCH):
            pad = ((0, PATCH - crop_ct.shape[0]), (0, PATCH - crop_ct.shape[1]))
            crop_ct = np.pad(crop_ct, pad); crop_y = np.pad(crop_y, pad); crop_m = np.pad(crop_m, pad)
        flip_lr = rng.random() < 0.5
        flip_ud = rng.random() < 0.5
        if in_ch == 2:
            crop_ctx = get_context(s, cy, cx)
        if flip_lr:
            crop_ct = crop_ct[:, ::-1]; crop_y = crop_y[:, ::-1]; crop_m = crop_m[:, ::-1]
            if in_ch == 2:
                crop_ctx = crop_ctx[:, ::-1]
        if flip_ud:
            crop_ct = crop_ct[::-1]; crop_y = crop_y[::-1]; crop_m = crop_m[::-1]
            if in_ch == 2:
                crop_ctx = crop_ctx[::-1]
        xb[i, 0] = crop_ct
        if in_ch == 2:
            xb[i, 1] = crop_ctx
        yb[i, 0] = crop_y
        wb[i, 0] = crop_m
    return (torch.from_numpy(xb.copy()).to(DEV), torch.from_numpy(yb.copy()).to(DEV), torch.from_numpy(wb.copy()).to(DEV))


def train_one(ModelCls, name, budget_s):
    model = ModelCls().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2000)
    BS = 12 if ModelCls.IN_CH == 2 else 16
    t0 = time.time()
    step = 0
    losses = []
    n_patches = 0
    best_val = float("inf")
    model.train()
    VAL_EVERY = 150
    val_crops = []  # small fixed val subsample for best-checkpoint tracking
    for z in HELD_OUT_Z[:1]:
        d = np.load(BUNDLE / f"heldout_z{z}.npz")
        ct, y, material = d["ct"], d["y"], d["material"]
        ct_n, _, _ = normalize_slice(ct, material)
        val_crops.append({"ct_n": ct_n, "y": y, "material": material, "H": ct.shape[0], "W": ct.shape[1],
                           "bbox": (0, ct.shape[0], 0, ct.shape[1])})
    while time.time() - t0 < budget_s:
        xb, yb, wb = sample_batch(BS, ModelCls.IN_CH)
        opt.zero_grad()
        p = clamp01_straight_through(model(xb))
        loss = soft_mse(p, yb, wb)
        loss.backward()
        opt.step()
        if step < 2000:
            sched.step()
        step += 1
        n_patches += BS
        losses.append(float(loss.detach()))
        if step % VAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                s = val_crops[0]
                cy, cx = sample_crop_center(s)
                crop_ct = s["ct_n"][cy:cy + PATCH, cx:cx + PATCH]
                crop_y = s["y"][cy:cy + PATCH, cx:cx + PATCH]
                crop_m = s["material"][cy:cy + PATCH, cx:cx + PATCH].astype(np.float32)
                xin = np.zeros((1, ModelCls.IN_CH, PATCH, PATCH), dtype=np.float32)
                xin[0, 0] = crop_ct
                if ModelCls.IN_CH == 2:
                    xin[0, 1] = get_context(s, cy, cx)
                t = torch.from_numpy(xin).to(DEV)
                pv = clamp01_straight_through(model(t)).cpu().numpy()[0, 0]
                vm = crop_m > 0
                vloss = float(np.mean((pv[vm] - crop_y[vm]) ** 2)) if vm.any() else float("inf")
            if vloss < best_val:
                best_val = vloss
                torch.save(model.state_dict(), OUT / f"{name}_BEST.pt")
            model.train()
    elapsed = time.time() - t0
    torch.save(model.state_dict(), OUT / f"{name}_LAST.pt")
    mvox_s = n_patches * PATCH * PATCH / 1e6 / elapsed
    print(f"[{name}] trained {step} steps ({n_patches} patches) in {elapsed:.1f}s, "
          f"final train soft-MSE {np.mean(losses[-50:]):.5f}, best_val_sample {best_val:.5f}, {mvox_s:.2f} MVox/s")
    rf = measure_receptive_field(model, ModelCls.IN_CH)
    print(f"[{name}] receptive field: {rf}")
    return model, {"train_steps": step, "n_patches": n_patches, "train_seconds": elapsed,
                    "throughput_mvox_s": mvox_s, "final_train_soft_mse": float(np.mean(losses[-50:])),
                    "best_val_sample_soft_mse": best_val, "receptive_field": rf}


def infer_blended(model, in_ch, ct_full, material, stride=256):
    """Overlapping-tile inference with a 2-D Hann blend window -- removes tile seams."""
    model.eval()
    H, W = ct_full.shape
    ct_n, _, _ = normalize_slice(ct_full, material)
    win1d = torch.hann_window(PATCH, periodic=False).numpy()
    win1d = np.clip(win1d, 0.05, 1.0)  # avoid exact-zero weight at the very edge
    win2d = np.outer(win1d, win1d).astype(np.float32)
    acc = np.zeros((H, W), dtype=np.float32)
    wsum = np.zeros((H, W), dtype=np.float32)
    ys = list(range(0, max(H - PATCH, 1), stride)) + [max(H - PATCH, 0)]
    xs = list(range(0, max(W - PATCH, 1), stride)) + [max(W - PATCH, 0)]
    ys = sorted(set(int(v) for v in ys)); xs = sorted(set(int(v) for v in xs))
    s_full = {"ct_n": ct_n, "H": H, "W": W}
    with torch.no_grad():
        for yy in ys:
            for xx in xs:
                crop_ct = ct_n[yy:yy + PATCH, xx:xx + PATCH]
                ph, pw = crop_ct.shape
                xin = np.zeros((1, in_ch, PATCH, PATCH), dtype=np.float32)
                xin[0, 0, :ph, :pw] = crop_ct
                if in_ch == 2:
                    xin[0, 1] = get_context(s_full, yy, xx)
                t = torch.from_numpy(xin).to(DEV)
                out = clamp01_straight_through(model(t))[0, 0].cpu().numpy()
                acc[yy:yy + ph, xx:xx + pw] += out[:ph, :pw] * win2d[:ph, :pw]
                wsum[yy:yy + ph, xx:xx + pw] += win2d[:ph, :pw]
    return acc / np.clip(wsum, 1e-6, None)


def score_region(pred, y, ct, region, thresh=0.1):
    if region.sum() == 0:
        return {"note": "empty region"}
    yj = y[region]; pj = pred[region]; ctj = ct[region].astype(np.float32)
    bin_y = (yj > thresh).astype(np.float32)
    if not (0 < bin_y.sum() < bin_y.size):
        return {"n_px": int(region.sum()), "n_positive_px": int(bin_y.sum()), "note": "degenerate (all/none positive)"}
    model_auc = auc(pj, bin_y)
    ct_auc = auc(ctj, bin_y)
    from scipy.ndimage import uniform_filter
    mean7 = uniform_filter(ct.astype(np.float32), 7)
    var7 = uniform_filter(ct.astype(np.float32) ** 2, 7) - mean7 ** 2
    var_auc = auc(var7[region], bin_y)
    return {"n_px": int(region.sum()), "n_positive_px": int(bin_y.sum()),
            "soft_mse_model": float(np.mean((pj - yj) ** 2)), "soft_mse_zero_baseline": float(np.mean(yj ** 2)),
            "auc_thresh_0.1": {"model": model_auc, "ct_intensity": ct_auc, "local_variance_7x7": var_auc}}


SERVE_DIR = Path("/mnt/raid10T/experiments/ink_transfer/mush_detector_v2")


def main():
    results = {}
    models = {}
    for ModelCls, name, budget in [(SmallUNetBN, "small", 300), (WideUNetBN, "wide", 360)]:
        model, meta = train_one(ModelCls, name, budget)
        models[name] = model
        results[name] = {"train_meta": meta, "held_out": {}}

    for name, model in models.items():
        in_ch = 2 if name == "wide" else 1
        for z in HELD_OUT_Z:
            d = np.load(BUNDLE / f"heldout_z{z}.npz")
            ct, y, material, band = d["ct"], d["y"], d["material"], d["band"]
            pred = infer_blended(model, in_ch, ct, material)
            np.save(OUT / f"heldout_pred_{name}_z{z}.npy", pred.astype(np.float16))
            entry = {"z_level0": z, "whole_material": score_region(pred, y, ct, material),
                      "band_800um_around_paint": score_region(pred, y, ct, band)}
            results[name]["held_out"][str(z)] = entry
            print(name, z, json.dumps(entry))

    json.dump(results, open(OUT / "results3.json", "w"), indent=1)
    print("HELDOUT_DONE")

    # full-res level0 sweep + other-scroll samples: run with the WIDE model (the one
    # meant to be deployed, per the context-fix ask), to keep total compute bounded
    sweep_dir = Path("/home/jacob/mush_bundle2/sweep_l0")
    z_list = json.load(open(sweep_dir / "z_list.json"))
    print(f"level0 sweep (wide model): {len(z_list)} slices")
    t1 = time.time()
    sweep_out = OUT / "sweep_pred_l0_wide"
    sweep_out.mkdir(exist_ok=True)
    wide = models["wide"]
    for i, z in enumerate(z_list):
        d = np.load(sweep_dir / f"s{i:04d}.npz")
        ct = d["ct"]
        material = ct > 5
        pred = infer_blended(wide, 2, ct, material, stride=384)  # coarser stride to fit budget
        np.savez_compressed(sweep_out / f"p{i:04d}.npz", pred=(np.clip(pred * 255, 0, 255)).astype(np.uint8), z=z)
        if i % 20 == 0:
            print(f"  sweep {i}/{len(z_list)} z={z} elapsed={time.time()-t1:.0f}s", flush=True)
    print(f"level0 sweep done in {time.time()-t1:.0f}s")
    print("SWEEP_DONE")

    for f in sorted(Path("/home/jacob/mush_bundle").glob("other_*.npz")):
        name = f.stem.replace("other_", "")
        d = np.load(f)
        imgs, zs = d["ct"], d["z_level0"]
        preds = np.zeros_like(imgs, dtype=np.uint8)
        for i in range(imgs.shape[0]):
            img = imgs[i]
            m = img > 5
            pred = infer_blended(wide, 2, img, m, stride=384)
            preds[i] = np.clip(pred * 255, 0, 255).astype(np.uint8)
        np.savez_compressed(OUT / f"other_pred_{name}.npz", pred=preds, ct=imgs, z_level0=zs)
        print(name, "inferred", imgs.shape)

    print("ALL DONE")


if __name__ == "__main__":
    main()
