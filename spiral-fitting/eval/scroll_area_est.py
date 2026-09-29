#!/usr/bin/env python3
"""A physical estimate of a scroll's TOTAL papyrus sheet area, the denominator for "what
fraction of the scroll does a route cover".

At every --step L0 slices: the CT slice at level L, thresholded (0.5 x the slice's p99 of
non-zero voxels, the same body rule as outer_gap.py), holes filled -> body cross-section area
(cm2). Integrated over z -> body volume (cm3). A rolled scroll is sheets stacked at the winding
pitch p, so sheet area ~ body volume / p. p is taken from the spiral fit's own meshes (the
radial spacing of adjacent windings, 16.0-18.9 L0 voxels measured on PHerc0211) and reported as a
RANGE, because the estimate is only as good as p: area = V / p_hi .. V / p_lo.

What it does NOT do: subtract the voids inside the body (delaminated gaps count as papyrus), or
tell the outer case apart from papyrus if the case is dense. It is an upper-bound-ish scale, not
a measurement of sheet area; say so beside any fraction computed from it.

usage: scroll_area_est.py CT_ZARR --z0 Z0 --z1 Z1 --vox-um UM [--step 250] [--level 2] [--pitch 16 19] [--json out]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr
from scipy import ndimage as ndi


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ct")
    ap.add_argument("--z0", type=int, required=True)
    ap.add_argument("--z1", type=int, required=True)
    ap.add_argument("--step", type=int, default=250)
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--pitch", type=float, nargs=2, default=[16.0, 19.0], help="winding pitch range, L0 voxels")
    ap.add_argument("--vox-um", type=float, required=True, help="level-0 voxel size (um); read it from the volume name or metadata")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    s = 2 ** a.level
    arr = zarr.open(a.ct, mode="r")[str(a.level)]
    px_cm2 = (s * a.vox_um * 1e-4) ** 2
    rows = []
    for z in range(a.z0, a.z1, a.step):
        c = np.asarray(arr[z // s]).astype(np.float32)
        if not (c > 0).any():
            rows.append({"z": z, "body_cm2": 0.0})
            continue
        th = 0.5 * float(np.percentile(c[c > 0], 99))
        m = ndi.binary_fill_holes(ndi.binary_closing(c > th, iterations=2))
        lab, n = ndi.label(m)
        if n:
            sizes = ndi.sum(m, lab, range(1, n + 1))
            m = np.isin(lab, 1 + np.flatnonzero(sizes >= 0.01 * sizes.max()))
        rows.append({"z": z, "body_cm2": float(m.sum() * px_cm2), "th": th})
    zz = np.array([r["z"] for r in rows], float)
    ar = np.array([r["body_cm2"] for r in rows])
    vol_cm3 = float(np.sum(ar) * a.step * a.vox_um * 1e-4)
    p_lo, p_hi = (p * a.vox_um * 1e-4 for p in a.pitch)
    out = {"ct": a.ct, "z": [a.z0, a.z1], "step": a.step, "level": a.level, "vox_um": a.vox_um,
           "body_volume_cm3": vol_cm3, "pitch_L0": a.pitch,
           "sheet_area_cm2_range": [vol_cm3 / p_hi, vol_cm3 / p_lo],
           "body_cm2_max": float(ar.max()) if ar.size else 0.0,
           "z_with_body": [float(zz[ar > 0.05 * ar.max()].min()), float(zz[ar > 0.05 * ar.max()].max())] if ar.size and ar.max() > 0 else None,
           "rows": rows}
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
