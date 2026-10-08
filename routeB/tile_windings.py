#!/usr/bin/env python3
"""Cut a Route B spiral fit's per-winding meshes into finishable tifxyz TILES.

WHY
---
A whole-scroll fit writes one tifxyz per winding, each ~13,000 voxels tall and up to
~2*pi*1,700 voxels long: tens of cm2 each. The fleet's finish stage refuses anything
above FINISH_MAX_CM2 (30 cm2) because a surface that size costs ~100 GB at inference
(scheduler.py), and FINDINGS 36.7 measured the GP model OOM-ing on a 71 Mpx strip. So a
winding is cut, on its own parameter lattice, into the FEWEST equal pieces of at most
`--target-cm2` (default 26 cm2; +overlap stays under the 30 cm2 finish cap) and `--rows` lattice rows, with `--overlap` cells added on
each side so no letter is lost on a seam.

WHAT IT DOES
------------
1. Dedupes the `_spliced` twins (FINDINGS 36.1: `save_mesh` writes wNNN_<tag> and
   wNNN_spliced_<tag>; with one segment per winding they are md5-identical). Keeps ONE
   per winding and REPORTS any pair that differs, rather than assuming.
2. Measures each winding from its own rasters: valid cells, quad area (cm2), z extent.
3. Writes tiles that hold at least `--min-valid` of their cells and `--min-cm2` of
   surface, cropped to their valid bounding box. The winding lattice is copied, never
   resampled: a tile's x/y/z are the fit's own numbers.
4. Writes manifest.json (every winding and every tile, with md5s of the source rasters).

Never modifies the fit's meshes. Output goes under --out.

usage: tile_windings.py MESHES_DIR --scroll PHerc0211 --fit full0211 --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import tifffile



def xyz_md5(d: Path) -> str:
    h = hashlib.md5()
    for n in "xyz":
        h.update((d / f"{n}.tif").read_bytes())
    return h.hexdigest()


def load(d: Path):
    X = tifffile.imread(d / "x.tif").astype(np.float32)
    Y = tifffile.imread(d / "y.tif").astype(np.float32)
    Z = tifffile.imread(d / "z.tif").astype(np.float32)
    V = (X != -1) & (Z > 0) & np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    return X, Y, Z, V


def quad_area_vx2(X, Y, Z, V) -> np.ndarray:
    """Per-cell surface area in voxel^2 for cells whose four corners are valid (two
    triangles). Shape (H-1, W-1); invalid cells are 0."""
    P = np.stack([X, Y, Z], -1).astype(np.float64)
    p00, p10, p01, p11 = P[:-1, :-1], P[1:, :-1], P[:-1, 1:], P[1:, 1:]
    ok = V[:-1, :-1] & V[1:, :-1] & V[:-1, 1:] & V[1:, 1:]
    a = 0.5 * np.linalg.norm(np.cross(p10 - p00, p01 - p00), axis=-1) \
        + 0.5 * np.linalg.norm(np.cross(p10 - p11, p01 - p11), axis=-1)
    return np.where(ok, a, 0.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("meshes", type=Path)
    ap.add_argument("--scroll", required=True)
    ap.add_argument("--voxel-um", type=float, required=True, help="level-0 voxel pitch of the scroll (um); from the scroll spec")
    ap.add_argument("--wind-min", type=int, default=None, help="only windings >= this index (smoke runs)")
    ap.add_argument("--wind-max", type=int, default=None, help="only windings <= this index (smoke runs)")
    ap.add_argument("--fit", required=True, help="fit tag, recorded as segment.fit_id")
    ap.add_argument("--prefix", required=True, help="segment-name stem, e.g. PHerc0211_sp1")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", type=int, default=400, help="max lattice rows per tile")
    ap.add_argument("--target-cm2", type=float, default=26.0, help="max tile area (cm2, 3D)")
    ap.add_argument("--overlap", type=int, default=4)
    ap.add_argument("--min-valid", type=float, default=0.25)
    ap.add_argument("--min-cm2", type=float, default=0.5)
    ap.add_argument("--yield", dest="yld", type=Path,
                    help="segment_yield.py jsonl over the windings; windings below --min-material "
                         "are measured and listed but NOT tiled (the 0.75 VACUUM line, FINDINGS 35.4)")
    ap.add_argument("--min-material", type=float, default=0.75)
    a = ap.parse_args(argv)
    vum = a.voxel_um
    cm2_per_vx2 = (vum * 1e-4) ** 2
    pat = re.compile(r"^w(\d{3})(_spliced)?_")
    groups: dict[int, dict[str, Path]] = {}
    for d in sorted(a.meshes.iterdir()):
        m = pat.match(d.name)
        if m and (d / "x.tif").exists():
            groups.setdefault(int(m.group(1)), {})["spliced" if m.group(2) else "plain"] = d
    if not groups:
        sys.exit(f"no wNNN_* tifxyz under {a.meshes}")
    a.out.mkdir(parents=True, exist_ok=True)
    yld = {}
    if a.yld is not None:
        for line in a.yld.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                yld[Path(r["tifxyz"]).resolve()] = r
    windings, tiles, differ = [], [], []
    for w in sorted(groups):
        if (a.wind_min is not None and w < a.wind_min) or (a.wind_max is not None and w > a.wind_max):
            continue
        g = groups[w]
        src = g.get("spliced") or g["plain"]
        md5s = {k: xyz_md5(p) for k, p in g.items()}
        if len(set(md5s.values())) > 1:
            differ.append(w)
        X, Y, Z, V = load(src)
        A = quad_area_vx2(X, Y, Z, V)
        wseg = f"{a.prefix}_w{w:03d}"
        zs = Z[V]
        wrow = {"seg": wseg, "winding": w, "src": str(src), "md5_xyz": md5s,
                "twins_identical": len(set(md5s.values())) == 1,
                "grid": list(X.shape), "valid_frac": float(V.mean()),
                "area_cm2": float(A.sum() * cm2_per_vx2),
                "z_min": float(zs.min()) if zs.size else None, "z_max": float(zs.max()) if zs.size else None,
                "tiles": []}
        y = yld.get(src.resolve())
        if y is not None:
            wrow["material_frac"] = y.get("material_frac")
            wrow["bbox_occupancy"] = y.get("bbox_occupancy")
        elif a.yld is not None:
            wrow["material_frac"] = None
        wrow["excluded"] = (a.yld is not None and (wrow["material_frac"] is None
                                                   or wrow["material_frac"] < a.min_material))
        H, W = X.shape if not wrow["excluded"] else (0, 0)
        # equal pieces sized to --target-cm2 (default 26; with overlap still under FINISH_MAX_CM2 = 30): as few
        # renders as possible, each with a --overlap cell margin on both sides
        n_r = max(1, -(-H // a.rows)) if H else 0
        n_c = max(1, -(-int(np.ceil(wrow["area_cm2"])) // max(1, int(n_r * a.target_cm2)))) if H else 0
        re_ = np.linspace(0, H, n_r + 1).round().astype(int) if H else []
        ce_ = np.linspace(0, W, n_c + 1).round().astype(int) if H else []
        spans = [(bi, ci, max(0, re_[bi] - a.overlap), min(H, re_[bi + 1] + a.overlap),
                  max(0, ce_[ci] - a.overlap), min(W, ce_[ci + 1] + a.overlap))
                 for bi in range(n_r) for ci in range(n_c)]
        for bi, ci, r0, r1, c0, c1 in spans:
            v = V[r0:r1, c0:c1]
            if not v.any():
                continue
            rr, cc = np.where(v)
            R0, R1, C0, C1 = r0 + rr.min(), r0 + rr.max() + 1, c0 + cc.min(), c0 + cc.max() + 1
            vv = V[R0:R1, C0:C1]
            area = float(A[R0:max(R0, R1 - 1), C0:max(C0, C1 - 1)].sum() * cm2_per_vx2)
            if vv.mean() < a.min_valid or area < a.min_cm2:
                continue
            tseg = f"{wseg}_z{bi:02d}x{ci}"
            td = a.out / "tiles" / tseg
            td.mkdir(parents=True, exist_ok=True)
            for n, arr in zip("xyz", (X, Y, Z), strict=True):
                t = arr[R0:R1, C0:C1].copy()
                t[~vv] = -1.0
                tifffile.imwrite(td / f"{n}.tif", t.astype(np.float32))
            xs, ys, zz = X[R0:R1, C0:C1][vv], Y[R0:R1, C0:C1][vv], Z[R0:R1, C0:C1][vv]
            meta = {"scale": json.loads((src / "meta.json").read_text()).get("scale", [0.05, 0.05]),
                    "bbox": [[float(xs.min()), float(ys.min()), float(zz.min())],
                             [float(xs.max()), float(ys.max()), float(zz.max())]],
                    "area_vx2": float(area / cm2_per_vx2), "area_cm2": area,
                    "format": "tifxyz", "type": "seg", "uuid": tseg,
                    "source": f"fit_spiral fitted, route B, tile of {wseg}",
                    "routeB": {"fit": a.fit, "winding": w, "parent": wseg, "src": str(src),
                               "rows": [int(R0), int(R1)], "cols": [int(C0), int(C1)],
                               "valid_frac": float(vv.mean())}}
            (td / "meta.json").write_text(json.dumps(meta, indent=1))
            row = {"seg": tseg, "parent": wseg, "dir": str(td), "area_cm2": area,
                   "valid_frac": float(vv.mean()), "rows": [int(R0), int(R1)], "cols": [int(C0), int(C1)],
                   "z": [float(zz.min()), float(zz.max())]}
            tiles.append(row)
            wrow["tiles"].append(tseg)
        windings.append(wrow)
        print(f"w{w:03d} grid {X.shape[0]}x{X.shape[1]} material {wrow.get('material_frac')} "
              f"{'EXCLUDED ' if wrow['excluded'] else ''}valid {wrow['valid_frac']:.3f} area {wrow['area_cm2']:.2f} cm2 "
              f"z {wrow['z_min']:.0f}-{wrow['z_max']:.0f} tiles {len(wrow['tiles'])}"
              + ("" if wrow["twins_identical"] else "  TWINS DIFFER"), flush=True)
    man = {"scroll": a.scroll, "fit": a.fit, "meshes": str(a.meshes), "voxel_um": vum,
           "params": {k: getattr(a, k) for k in ("rows", "target_cm2", "overlap", "min_valid", "min_cm2")},
           "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "code_md5": hashlib.md5(Path(__file__).read_bytes()).hexdigest(),
           "n_windings": len(windings), "n_tiles": len(tiles),
           "winding_area_cm2": sum(w["area_cm2"] for w in windings),
           "tile_area_cm2": sum(t["area_cm2"] for t in tiles),
           "twins_differ": differ, "min_material": a.min_material if a.yld else None,
           "n_excluded": sum(1 for w in windings if w["excluded"]),
           "windings": windings, "tiles": tiles}
    (a.out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"windings {len(windings)} ({man['winding_area_cm2']:.1f} cm2, {man['n_excluded']} excluded), tiles {len(tiles)} "
          f"({man['tile_area_cm2']:.1f} cm2 incl. overlap), twins differing: {differ or 'none'}")


if __name__ == "__main__":
    main()
