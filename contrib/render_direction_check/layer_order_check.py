#!/usr/bin/env python3
"""Is my render's plane k the published stack's plane k, or plane N-1-k?  (And do the two agree at all?)

Given two stacks of the SAME surface window, (P, H, W) arrays saved as .npy (or a zarr array path), correlate every plane of A with every plane of B on the common valid pixels.
A correct, non-flipped render has its best match on the diagonal; a depth-reversed one on the anti-diagonal.
usage: layer_order_check.py A.npy B.npy [--sample 400000]      prints diag / anti-diagonal mean correlation and the verdict"""
import argparse
import numpy as np


def plane_corr(A: np.ndarray, B: np.ndarray, sample: int = 400000, seed: int = 0) -> np.ndarray:
    P = min(A.shape[0], B.shape[0]); H = min(A.shape[1], B.shape[1]); W = min(A.shape[2], B.shape[2])
    A, B = A[:P, :H, :W].reshape(P, -1).astype(np.float32), B[:P, :H, :W].reshape(P, -1).astype(np.float32)
    valid = (A > 0).any(0) & (B > 0).any(0); idx = np.flatnonzero(valid)
    idx = np.random.default_rng(seed).choice(idx, min(sample, idx.size), replace=False)
    a = A[:, idx]; b = B[:, idx]; a = (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-6); b = (b - b.mean(1, keepdims=True)) / (b.std(1, keepdims=True) + 1e-6)
    return a @ b.T / idx.size


def order_verdict(C: np.ndarray) -> dict:
    P = C.shape[0]; d = float(np.mean(np.diag(C))); ad = float(np.mean(np.diag(C[:, ::-1])))
    return dict(planes=P, mean_corr_same_order=round(d, 3), mean_corr_reversed_order=round(ad, 3), verdict=("same order" if d > ad + 0.1 else ("REVERSED" if ad > d + 0.1 else "ambiguous")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--sample", type=int, default=400000); x = ap.parse_args()
    print(order_verdict(plane_corr(np.load(x.a, mmap_mode="r"), np.load(x.b, mmap_mode="r"), x.sample)))
