#!/usr/bin/env python3
"""reader_v2 FAST inference, run under villa's interpreter (it imports villa's vesuvius.ink_detection, nothing of ours).

Why: the villa CLI (`vesuvius.ink_detection.inference.infer`) reads ~100 patches/s however it is fed (DataLoader worker
spawn ~20 s fixed, then per-patch host overhead), so production reader_v2 faces sat at 0 % GPU for minutes (overnight
operator, 2026-10-06). This file applies the SAME checkpoint contract with villa's OWN primitives, batched on the GPU:
`configure_model` (weights, preprocessing, AMP dtype from the checkpoint), 128 px patches at stride 64
(`_sliding_positions_1d`, boundary-aligned last position), per-patch robust MAD (`normalize_batch_on_device`, documented
bit-exact with the CPU path), `logits_to_probabilities`, floored Hann blend (`compute_importance_map_2d`) weighted by the
validity mask, uint8 = floor(255 p). Agreement with the CLI is MEASURED, not assumed (scripts/reader_v2/fast_check.py):
bar r >= 0.999 and mean |d| <= 1 LSB on valid pixels.

usage: _reader_v2_villa.py <x17.npy (17,H,W) uint8, face order as given> <ckpt> <out.png> [--mask mask.png] [--batch 32]
Writes <out.png> and <out>.fast.json (timings, MVox/s, device).
"""
import argparse
import json
import os
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("x")
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--mask", default="")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    t0 = time.time()
    import torch
    from PIL import Image
    from dataclasses import replace
    from vesuvius.ink_detection.inference import infer as INF
    x = torch.from_numpy(np.load(a.x))                       # uint8, CPU
    Z, H, W = x.shape
    assert Z == 17, x.shape
    valid = x[Z // 2] > 0
    if a.mask and os.path.exists(a.mask):
        mk = np.asarray(Image.open(a.mask).convert("L"))
        m = np.zeros((H, W), bool)
        h, w = min(H, mk.shape[0]), min(W, mk.shape[1])
        m[:h, :w] = mk[:h, :w] > 127
        valid &= torch.from_numpy(m)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cm = INF.configure_model(INF.parse_args(["/dev/null", a.ckpt, "/dev/null"]))
    cm = replace(cm, model=cm.model.to(dev).eval())
    P = int(cm.patch_size)
    Hp, Wp = max(H, P), max(W, P)
    if (Hp, Wp) != (H, W):
        x = torch.nn.functional.pad(x, (0, Wp - W, 0, Hp - H))
        valid = torch.nn.functional.pad(valid, (0, Wp - W, 0, Hp - H))
    wmap = INF.compute_importance_map_2d(patch_size=(P, P), mode="hann").to(dev)
    ys = INF._sliding_positions_1d(Hp, P, 64)
    xs = INF._sliding_positions_1d(Wp, P, 64)
    pos = [(y, xx) for y in ys for xx in xs if bool(valid[y:y + P, xx:xx + P].any())]
    acc = torch.zeros((Hp, Wp), device=dev)
    wac = torch.zeros((Hp, Wp), device=dev)
    vf = valid.to(dev).float()
    ac = torch.autocast(dev.type, dtype=cm.amp_dtype) if (cm.amp_dtype is not None and dev.type == "cuda") else torch.autocast(dev.type, enabled=False)
    t1 = time.time()
    with torch.inference_mode():
        for i in range(0, len(pos), a.batch):
            pb = pos[i:i + a.batch]
            imgs = torch.stack([x[:, y:y + P, xx:xx + P] for y, xx in pb]).to(dev, non_blocking=True).float()[:, None]
            imgs = INF.normalize_batch_on_device(imgs, cm.preprocessing)
            with ac:
                pr = INF.logits_to_probabilities(cm.model(imgs), image_hw=(P, P))[:, 0].float()
            for (y, xx), p in zip(pb, pr, strict=True):
                w = wmap * vf[y:y + P, xx:xx + P]
                acc[y:y + P, xx:xx + P] += p * w
                wac[y:y + P, xx:xx + P] += w
    out = torch.where(wac > 0, acc / wac.clamp(min=1e-12), torch.zeros_like(acc))[:H, :W]
    q = (out.clamp(0, 1) * 255).floor().to(torch.uint8).cpu().numpy()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t2 = time.time()
    Image.fromarray(q).save(a.out)
    nv = int(valid[:H, :W].sum())
    json.dump({"setup_s": round(t1 - t0, 2), "gpu_s": round(t2 - t1, 2), "patches": len(pos), "valid_px": nv,
               "mvox_per_s": round(nv * 17 / max(t2 - t1, 1e-6) / 1e6, 2), "batch": a.batch,
               "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"},
              open(os.path.splitext(a.out)[0] + ".fast.json", "w"), indent=1)


if __name__ == "__main__":
    main()
