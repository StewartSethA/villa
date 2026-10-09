#!/usr/bin/env python3
"""Profile + check the trainer's batched 3-D warp / domain augmentation (rv2_distill.augment) outside training.
Checks: identity at zero amplitude; mid-plane / target alignment under warp+zoom (layers identical -> exact);
times each part on the given device. Usage: aug_bench.py [--device cuda] [--batch 32]"""
import argparse, math, os, sys, textwrap, time, types
import numpy as np
import torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda"); ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--chunk", type=int, default=4)
a0 = ap.parse_args()
src = open(os.path.join(HERE, "rv2_distill.py")).read()
seg = src[src.index("    def _gblur("):src.index("    AUGLOG = {")] + src[src.index("    def _sfield("):src.index("    def _pull2d(img, dy, dx):")]
sys.path.insert(0, os.path.join(HERE, "..", "ensemble_rv2")); import warp3d as W3


def make(warp, dom, mode="full"):
    g = dict(math=math, np=np, torch=torch, F=F, W3=W3, HL=8, E=24, AUGLOG={"n": 0, "warp": 0, "dom": 0},
             a=types.SimpleNamespace(warp3d=warp, domain_aug=dom, warp_mode=mode))
    exec(textwrap.dedent(seg), g)
    return g["augment"]


dev = a0.device; B, Z = a0.batch, 17; Hx = 256 + 2 * 32
rng = np.random.default_rng(0)
xb = (torch.rand(B, Z, Hx, Hx, device=dev) * 200 + 20).to(torch.uint8)
I = F.avg_pool2d(torch.rand(B, 1, Hx, Hx, device=dev), 5, 1, 2)[:, 0] * 200 + 20
xs = I[:, None].expand(B, Z, Hx, Hx).contiguous()
t = I[:, 8:-8, 8:-8] / 255.0; w = torch.ones_like(t)
xo, to, wo = make(0, 0)(xs, t, w, rng, a0.chunk)
print("identity max |x'-x| %.4g, |t'-t| %.3g" % (float((xo - xs[:, :, 24:-24, 24:-24]).abs().max()), float((to - t[:, 24:-24, 24:-24]).abs().max())))
xo, to, wo = make(1, 0)(xs, t, w, rng, a0.chunk)
m = wo > 0
print("warp: max |mid - target| on valid %.4g (valid %.3f)" % (float((xo[:, 8, 8:-8, 8:-8] / 255 - to)[m].abs().max()), float(m.float().mean())))
for mode in ("inplane", "depth"):
    xo, to, wo = make(1, 0, mode)(xs, t, w, rng, a0.chunk); m = wo > 0
    print("%s: max |mid - target| on valid %.4g" % (mode, float((xo[:, 8, 8:-8, 8:-8] / 255 - to)[m].abs().max())))
for name, wp, dm in (("warp", 0.7, 0), ("domain", 0, 0.7), ("both", 0.7, 0.7)):
    f = make(wp, dm)
    f(xb, t, w, rng, a0.chunk)
    if dev == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        f(xb, t, w, rng, a0.chunk)
    if dev == "cuda": torch.cuda.synchronize()
    print(f"{name}: {(time.time() - t0) / 5 * 1000:.0f} ms per batch of {B} on {dev} (chunk {a0.chunk})")
if dev == "cuda":
    from torch.profiler import profile, ProfilerActivity
    f = make(0.7, 0.7)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        f(xb, t, w, rng, a0.chunk); torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))
