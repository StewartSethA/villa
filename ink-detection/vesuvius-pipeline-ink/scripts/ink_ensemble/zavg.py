"""Z-score-average consensus of finished ink maps for one segment (Nieuwlaar's z-score-average recipe, applied to our published family PNGs).
Each member map is z-scored with pooled-free per-map stats over its valid (non-zero) pixels, the z maps are averaged, then ONE stretch p1..p99.5 -> uint8.
Members default to the pipeline's best families; faces are averaged separately and also jointly (the reading face is the *_reversed order).
usage: zavg.py SEG [--dir DIR] [--members a,b,c] [--out DIR]   (D33: PCT_LO/PCT_HI named below and printed)"""
import argparse, sys
from pathlib import Path
import numpy as np
from PIL import Image
PCT_LO, PCT_HI = 1.0, 99.5
DEF = "dense_native,reader_v2,reader_v2_dense,ink9um_student,sharp_dense"
def load(p):
    a = np.asarray(Image.open(p).convert("L"), np.float32); return a
def zavg(arrs):
    zs = []
    for a in arrs:
        v = a > 0
        m, s = a[v].mean(), max(a[v].std(), 1e-6)
        z = np.where(v, (a - m) / s, np.nan); zs.append(z)
    st = np.nanmean(np.stack(zs), 0)
    ok = np.isfinite(st)
    lo, hi = np.percentile(st[ok], [PCT_LO, PCT_HI])
    return np.where(ok, np.clip((st - lo) / (hi - lo), 0, 1) * 255, 0).astype(np.uint8)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("seg"); ap.add_argument("--dir"); ap.add_argument("--members", default=DEF); ap.add_argument("--out")
    a = ap.parse_args()
    d = Path(a.dir or f"ink_detection_results/{a.seg}"); out = Path(a.out or d); out.mkdir(parents=True, exist_ok=True)
    print(f"percentiles p{PCT_LO}..p{PCT_HI}; members {a.members}")
    for face, suf in (("fwd", ""), ("rev", "_reversed")):
        got, arrs = [], []
        for m in a.members.split(","):
            p = d / f"{m}{suf}.png"
            if p.exists(): got.append(m); arrs.append(load(p))
        if len(arrs) < 2: print(face, "fewer than 2 members:", got); continue
        sh = {x.shape for x in arrs}
        if len(sh) > 1: print(face, "shape mismatch", sh); continue
        Image.fromarray(zavg(arrs)).save(out / f"consensus_zavg{suf}.png"); print(face, "n_members", len(got), got, "->", out / f"consensus_zavg{suf}.png")
main()
