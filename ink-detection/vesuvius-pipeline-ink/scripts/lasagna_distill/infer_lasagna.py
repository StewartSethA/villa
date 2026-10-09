#!/usr/bin/env python3
"""Run the distilled Lasagna model over a scroll's CT volume and write the fields in the PUBLISHED layout.

Output (one directory per scroll, <out>/<scroll>/):
  <scroll>_nx.ome.zarr/ <scroll>_ny.ome.zarr/ <scroll>_grad_mag.ome.zarr/ <scroll>_cos.ome.zarr/
    .zgroup, .zattrs (OME multiscales, one dataset "2" with scale [4,4,4], like the published stores), 2/.zarray + 2/z/y/x chunks
    uint8, chunks 32^3, blosc lz4 clevel 5 shuffle 1, fill_value 0 (0 = unset, as published), shape == the volume's level-2 shape
  <scroll>.lasagna.json     provenance: checkpoint sha256, code git, input volume, DISTILLED (not upstream) flag
Tiling: 128^3 input tiles at stride 96; only the central 96^3 of each tile is kept (16-voxel margin discarded), so no blending
and every output chunk is written once.  Tiles whose CT is entirely 0 are skipped; the output is 0 wherever CT == 0 and >= 1
elsewhere (a predicted 0 would read as "unset").  Run one process per GPU:  --shard i --nshards n  (z-slab sharding).
  python infer_lasagna.py --scroll PHerc0490A --ckpt best.pt --vol-root DIR --out DIR [--zmin Z --zmax Z] --shard 0 --nshards 4
--vol-root holds the volume's level-2 chunks as fetched by `rclone copy` (<vol>/2/z/y/x raw uint8 128^3 + .zarray).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_lasagna import UNet3D, prep  # noqa: E402

CH = ("nx", "ny", "grad_mag", "cos")
T, M, S = 128, 16, 96          # input tile, discarded margin, kept core
VC, FC = 128, 32


def say(m):
    print(f"[{time.strftime('%H:%M:%S')}] infer: {m}", flush=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


class Vol:
    """Level-2 CT chunks on local disk; a missing chunk file is all zeros."""

    def __init__(self, root: Path, level=2):
        za = list(root.glob(f"*/{level}/.zarray")) or list(root.glob(f"{level}/.zarray"))
        if not za:
            raise SystemExit(f"no {level}/.zarray under {root}")
        self.dir = za[0].parent
        self.shape = tuple(json.loads(za[0].read_text())["shape"])

    def read(self, z0, y0, x0, n=T):
        out = np.zeros((n, n, n), np.uint8)
        for z in range(max(0, z0) // VC, (z0 + n - 1) // VC + 1):
            for y in range(max(0, y0) // VC, (y0 + n - 1) // VC + 1):
                for x in range(max(0, x0) // VC, (x0 + n - 1) // VC + 1):
                    f = self.dir / str(z) / str(y) / str(x)
                    if not f.exists():
                        continue
                    ch = np.fromfile(f, np.uint8)
                    if ch.size != VC ** 3:
                        continue
                    ch = ch.reshape(VC, VC, VC)
                    zs, ys, xs = z * VC, y * VC, x * VC
                    a0, a1 = max(z0, zs), min(z0 + n, zs + VC)
                    b0, b1 = max(y0, ys), min(y0 + n, ys + VC)
                    c0, c1 = max(x0, xs), min(x0 + n, xs + VC)
                    if a0 < a1 and b0 < b1 and c0 < c1:
                        out[a0 - z0:a1 - z0, b0 - y0:b1 - y0, c0 - x0:c1 - x0] = ch[a0 - zs:a1 - zs, b0 - ys:b1 - ys, c0 - xs:c1 - xs]
        return out


def write_meta(odir: Path, scroll: str, shape, ckpt_sha, git, vol_name):
    for c in CH:
        d = odir / f"{scroll}_{c}.ome.zarr"
        (d / "2").mkdir(parents=True, exist_ok=True)
        (d / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
        (d / ".zattrs").write_text(json.dumps({"multiscales": [{"version": "0.4", "name": c, "axes": [
            {"name": a, "type": "space", "unit": "pixel"} for a in "zyx"],
            "datasets": [{"path": "2", "coordinateTransformations": [{"type": "scale", "scale": [4.0, 4.0, 4.0]}]}]}],
            "lasagna_source": "DISTILLED from the published fields by scripts/lasagna_distill (NOT an upstream release)"}))
        (d / "2" / ".zarray").write_text(json.dumps({
            "shape": list(shape), "chunks": [FC] * 3, "dtype": "|u1", "fill_value": 0, "order": "C", "filters": None,
            "dimension_separator": "/", "compressor": {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1, "blocksize": 0},
            "zarr_format": 2}))
    (odir / f"{scroll}.lasagna.json").write_text(json.dumps({
        "version": 2, "source": "distilled", "scroll": scroll, "checkpoint_sha256": ckpt_sha, "code_git": git, "volume": vol_name,
        "groups": {c: {"zarr": f"{scroll}_{c}.ome.zarr/2", "scaledown": 2, "channels": [c]} for c in CH},
        "base_shape_zyx": [int(s) * 4 for s in shape], "note": "distilled fields; validate against published fields on held-out scrolls before trusting"}, indent=1))


def write_core(odir, scroll, codec, arr, z0, y0, x0):
    """arr (4, a, b, c) uint8 whose origin is chunk-aligned (multiples of 32)."""
    for ci, c in enumerate(CH):
        base = odir / f"{scroll}_{c}.ome.zarr" / "2"
        for z in range(0, arr.shape[1], FC):
            for y in range(0, arr.shape[2], FC):
                for x in range(0, arr.shape[3], FC):
                    blk = arr[ci, z:z + FC, y:y + FC, x:x + FC]
                    if not blk.any():
                        continue
                    full = np.zeros((FC, FC, FC), np.uint8)
                    full[:blk.shape[0], :blk.shape[1], :blk.shape[2]] = blk
                    p = base / str((z0 + z) // FC) / str((y0 + y) // FC)
                    p.mkdir(parents=True, exist_ok=True)
                    (p / str((x0 + x) // FC)).write_bytes(codec.encode(full))


def main():
    from numcodecs import Blosc
    ap = argparse.ArgumentParser()
    ap.add_argument("--scroll", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vol-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--zmin", type=int, default=0)
    ap.add_argument("--zmax", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--bs", type=int, default=4)
    a = ap.parse_args()
    dev = torch.device("cuda")
    ck = torch.load(a.ckpt, map_location="cpu")
    model = UNet3D(ck["args"].get("base", 24)).to(dev).eval().to(memory_format=torch.channels_last_3d)
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()})
    vol = Vol(Path(a.vol_root))
    odir = Path(a.out) / a.scroll
    git = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    if a.shard == 0:
        odir.mkdir(parents=True, exist_ok=True)
        write_meta(odir, a.scroll, vol.shape, sha256(a.ckpt), git, str(vol.dir.parent.name))
    else:
        for _ in range(120):
            if (odir / f"{a.scroll}.lasagna.json").exists():
                break
            time.sleep(1)
    codec = Blosc(cname="lz4", clevel=5, shuffle=1)
    Z, Y, X = vol.shape
    zmax = a.zmax or Z
    slabs = list(range(a.zmin // S * S, zmax, S))[a.shard::a.nshards]
    t0, nt, nskip, nvox = time.time(), 0, 0, 0
    for zc in slabs:
        z0 = zc - M
        zl = min(S, zmax - zc, Z - zc)
        core = np.zeros((4, S, ((Y + S - 1) // S) * S, ((X + S - 1) // S) * S), np.uint8)
        batch, where = [], []

        def flush():
            nonlocal nt, nvox
            if not batch:
                return
            x = torch.from_numpy(np.stack(batch)).to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                xin, _, _ = prep(x, torch.zeros(len(batch), 4, T, T, T, dtype=torch.uint8, device=dev))
                p = model(xin.contiguous(memory_format=torch.channels_last_3d)).float()
            pu = (p[:, :, M:M + S, M:M + S, M:M + S] * 255).round().clamp(1, 255).to(torch.uint8).cpu().numpy()
            for k, (yc, xc) in enumerate(where):
                inside = np.stack(batch)[k][M:M + S, M:M + S, M:M + S] > 0
                core[:, :, yc:yc + S, xc:xc + S] = pu[k] * inside[None]
            nt += len(batch)
            nvox += len(batch) * S ** 3
            batch.clear()
            where.clear()

        for yc in range(0, Y, S):
            for xc in range(0, X, S):
                t = vol.read(z0, yc - M, xc - M)
                if not t[M:M + S, M:M + S, M:M + S].any():
                    nskip += 1
                    continue
                batch.append(t)
                where.append((yc, xc))
                if len(batch) == a.bs:
                    flush()
        flush()
        core[:, zl:] = 0
        write_core(odir, a.scroll, codec, core[:, :, :Y, :X][:, :, :, :], zc, 0, 0)
        say(f"shard {a.shard}: slab z={zc} done; {nt} tiles run, {nskip} empty skipped, {nvox / 1e6 / max(1e-9, time.time() - t0):.0f} MVox/s kept output, {time.time() - t0:.0f} s")
    say(f"shard {a.shard} finished {len(slabs)} slab(s)")


if __name__ == "__main__":
    main()
