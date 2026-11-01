#!/usr/bin/env python3
"""Do adjacent sheets carry the same ink picture? 
For an Overlap-3D tile with sheetspx.u8 (sheet label per voxel: 1 own, 2 +1, 3 -1, 4 +2, 5 -2, 6 +3.., 7 -3..) and ink3d.u8 (ink score), project the ink of every sheet to 2-D
(max over the planes of that sheet's run in each column) and report, over the columns where both sheets exist: Pearson r of the two maps, IoU of the maps thresholded at THR,
each next to a NULL (the second map rolled laterally by --null-shift px, same pixels, same statistics, no registration) and to the human label (ink_human_label.png when present).
usage: sheet_leakage_metrics.py TILE_DIR [--thr 0.5] [--null-shift 97]"""
import argparse, json, os, sys
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("tile"); ap.add_argument("--thr", type=float, default=0.5); ap.add_argument("--null-shift", type=int, default=97); a = ap.parse_args()
def load_lab(t, sh):
    if os.path.exists(t + "/sheetspx.u8"): return np.fromfile(t + "/sheetspx.u8", np.uint8).reshape(sh)
    vol = np.fromfile(t + "/vol.u8", np.uint8).reshape(sh); fg = vol > 127; NP = sh[0]; MID = NP // 2                       # no stored labels: number the predicted-sheet runs along depth from the one nearest the mesh plane (as build_s1_full_tiles.py)
    start = fg & ~np.concatenate([np.zeros((1,) + sh[1:], bool), fg[:-1]], 0); rid = np.cumsum(start, 0) * fg; dist = np.where(fg, np.abs(np.arange(NP) - MID)[:, None, None], 99); jn = dist.argmin(0); has = fg.any(0) & (dist.min(0) <= 7)
    own = np.where(has, np.take_along_axis(rid, jn[None], 0)[0], 0); rel = rid - own[None]; lab = np.zeros(sh, np.uint8); m = fg & has[None]
    for l, c in ((1, rel == 0), (2, rel == 1), (3, rel == -1), (4, rel == 2), (5, rel == -2), (6, rel >= 3), (7, rel <= -3)): lab[m & c] = l
    return lab
v = json.load(open(a.tile + "/vol.json")); sh = tuple(v["shape"]); lab = load_lab(a.tile, sh); ink = np.fromfile(a.tile + "/ink3d.u8", np.uint8).reshape(sh).astype(np.float32) / 255
names = {1: "own", 2: "+1", 3: "-1", 4: "+2", 5: "-2", 6: "+3..", 7: "-3.."}; M = {}; E = {}
for l in names:
    m = lab == l; E[l] = m.any(0); M[l] = np.where(m, ink, 0).max(0)
def r(x, y, ok):
    x, y = x[ok], y[ok]
    return float(np.corrcoef(x, y)[0, 1]) if ok.sum() > 100 and x.std() > 0 and y.std() > 0 else None
def iou(x, y, ok):
    bx, by = (x > a.thr)[ok], (y > a.thr)[ok]; u = (bx | by).sum()
    return float((bx & by).sum() / u) if u > 50 else None
out = []; hl = None
if os.path.exists(a.tile + "/ink_human_label.png"):
    from PIL import Image; hl = (np.asarray(Image.open(a.tile + "/ink_human_label.png")) > 0).astype(np.float32)
for p, q in [(1, 2), (1, 3), (2, 1), (1, 4), (1, 5), (2, 4), (3, 5), (1, 6), (1, 7)]:
    ok = E[p] & E[q]
    if ok.sum() < 1000: continue
    qs = np.roll(M[q], a.null_shift, axis=1); oks = E[p] & np.roll(E[q], a.null_shift, axis=1)
    out.append(dict(pair=f"{names[p]} vs {names[q]}", n_columns=int(ok.sum()), pearson=r(M[p], M[q], ok), pearson_null_shifted=r(M[p], qs, oks), iou=iou(M[p], M[q], ok), iou_null_shifted=iou(M[p], qs, oks)))
    if hl is not None: out[-1]["pearson_with_human_label_p_q"] = [r(M[p], hl, E[p]), r(M[q], hl, E[q])]
print(a.tile, "ink columns per sheet:", {names[l]: int((M[l] > a.thr).sum()) for l in names})
for o in out: print(json.dumps(o))
