#!/usr/bin/env python3
"""Does this render have the intensity distribution the model was trained on?  Prints mean / std / p1 / p99 of the middle plane over tissue pixels and the normalisation to apply.

Models that take raw uint8 (the Grand-Prize-era stacks) expect tissue mean ~150, std ~20-25 on the middle layer; models with per-window robust-MAD normalisation (reader-v3) do not care.
Measured example (Scroll 5, same mesh, same model): our render mean 87 / std 32 -> AUC 0.52; reversed plane order 0.87; reversed + normalised to 150 / 25: 0.92 (the published stack: 0.91).
usage: intensity_check.py STACK.npy [--target-mean 150 --target-std 25]"""
import argparse
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("stack"); ap.add_argument("--target-mean", type=float, default=150.0); ap.add_argument("--target-std", type=float, default=25.0); a = ap.parse_args()
S = np.load(a.stack, mmap_mode="r"); mid = np.asarray(S[S.shape[0] // 2]).astype(np.float32); t = mid[mid > 0]
m, s = float(t.mean()), float(t.std()); p1, p99 = np.percentile(t, [1, 99])
print(dict(middle_plane=S.shape[0] // 2, tissue_px=int(t.size), mean=round(m, 1), std=round(s, 1), p1=round(float(p1), 1), p99=round(float(p99), 1)))
print(f"normalise with  x' = clip((x - {m:.1f}) / {s:.1f} * {a.target_std:g} + {a.target_mean:g}, 0, 255)  (on tissue pixels; leave 0 outside the surface)")
