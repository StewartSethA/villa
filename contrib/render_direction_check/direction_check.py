#!/usr/bin/env python3
"""Which way does depth run in this surface render?  Decide forward vs reverse from two ink maps and one reference map.

A surface-conditioned render stacks planes along the mesh normal; whether plane 0 is the recto or the verso side depends on the mesh orientation, and a model that is depth-aware (a 3-D stem, Hecate,
the 3-D student) reads chance on the wrong direction.  Run the model once on the render as it is (`fwd`) and once with the plane order reversed (`rev`), reduce each to a 2-D map
(for a 3-D output: max over planes), and compare each against any independent reference map of the same area (another model's 2-D map, a human label).  The direction that agrees in more blocks wins.

usage: direction_check.py FWD.png REV.png REF.png [--block 256] [--min-valid 0.7]      (maps must be the same size; uint8 grey; 0 = outside the surface)
prints the per-block Pearson r, the share of blocks where fwd > rev, the mean r of each, and a verdict (fwd / rev / ambiguous)."""
import argparse
import numpy as np
from PIL import Image


def block_correlations(fwd: np.ndarray, rev: np.ndarray, ref: np.ndarray, block: int = 256, min_valid: float = 0.7):
    """-> list of (r_fwd, r_rev) over blocks where >= min_valid of the pixels are valid (ref > 0 and any of fwd/rev > 0) and the reference varies."""
    out = []
    valid = (ref > 0) & ((fwd > 0) | (rev > 0))
    H, W = ref.shape
    for y in range(0, H - block + 1, block):
        for x in range(0, W - block + 1, block):
            m = valid[y:y + block, x:x + block]
            if m.mean() < min_valid:
                continue
            a = ref[y:y + block, x:x + block][m].astype(np.float64)
            if a.std() < 1e-6:
                continue
            rf = np.corrcoef(a, fwd[y:y + block, x:x + block][m].astype(np.float64))[0, 1] if fwd[y:y + block, x:x + block][m].std() > 0 else 0.0
            rr = np.corrcoef(a, rev[y:y + block, x:x + block][m].astype(np.float64))[0, 1] if rev[y:y + block, x:x + block][m].std() > 0 else 0.0
            out.append((rf, rr))
    return out


def verdict(pairs, margin: float = 0.10, share: float = 0.75):
    if not pairs:
        return dict(n_blocks=0, verdict="ambiguous")
    p = np.array(pairs)
    mf, mr = float(p[:, 0].mean()), float(p[:, 1].mean()); sh = float((p[:, 0] > p[:, 1]).mean())
    v = "fwd" if (mf - mr > margin and sh >= share) else ("rev" if (mr - mf > margin and sh <= 1 - share) else "ambiguous")
    return dict(n_blocks=len(pairs), mean_r_fwd=round(mf, 3), mean_r_rev=round(mr, 3), share_blocks_fwd_better=round(sh, 3), verdict=v)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("fwd"); ap.add_argument("rev"); ap.add_argument("ref"); ap.add_argument("--block", type=int, default=256); ap.add_argument("--min-valid", type=float, default=0.7)
    a = ap.parse_args(); Image.MAX_IMAGE_PIXELS = None
    f, r, g = [np.asarray(Image.open(p).convert("L")) for p in (a.fwd, a.rev, a.ref)]
    print(verdict(block_correlations(f, r, g, a.block, a.min_valid)))
