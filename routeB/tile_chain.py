"""Stages SNAP -> FLATTEN -> RENDER -> INK for one tile (tifxyz dir).  The same arithmetic as src/vesuvius_pipeline/stages/render.py + ink.py of the
fleet pipeline, without its database:
  snap     scripts/sheet_snap.py (recto face of the nearest material run, radius 10 vox)  [default on, as in production VPIPE_SHEET_SNAP=recto]
  flatten  villa lasagna fit.py configs/flatten_fast_nofilter.json  (the First Letters flattener; GPU if present)
  frame    um/px at render scale 1 = edge_med_vox / step_size(20) * voxel_um ; render scale = that / 9.5 um  (D: never hand-set)
  render   gpu_render/render_gpu.py (NVRTC kernels, needs an uncompressed 128^3 zarr -> a sparse local copy of just the needed chunks)
  ink      vendored ink_models: sharp_dense, reader_v2_dense, ink9um_student, sharp_dense_human on the 17 centred layers at 9.5 um/px
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from .common import ROOT, StageError, env_python, home, is_done, mark_done, run, say, spec, tail

STEP_SIZE = 20.0               # flatten_output_step in configs/flatten_fast_nofilter.json
TARGET_UM = 9.5                # all four ink families read ~9.5 um/px
LAYERS = 17                    # centred 17 layers; the fleet renders 65 and the models take the centre 17 (same offsets, odd count)
SNAP_RADIUS = 10
FAMILIES = {                    # family -> (module, arm)
    "sharp_dense": ("reader_v2_dense", "sharp"),
    "reader_v2_dense": ("reader_v2_dense", None),
    "ink9um_student": ("ink9um_student", None),
    "sharp_dense_human": ("reader_v2_dense", "human_cv14k"),
}
DEFAULT_FAMILIES = "sharp_dense,reader_v2_dense,ink9um_student,sharp_dense_human"


def load_xyz(d: Path):
    import tifffile
    return [tifffile.imread(d / f"{a}.tif").astype(np.float32) for a in "xyz"]


def needed_chunks(tile: Path, margin_vox: int = SNAP_RADIUS + LAYERS // 2 + 4, chunk: int = 128) -> list[str]:
    """'z/y/x' level-0 chunk keys within margin_vox of any lattice point (what snap + render can read)."""
    X, Y, Z = load_xyz(tile)
    v = (X > 0) & (Y > 0) & (Z > 0)
    P = np.stack([X[v], Y[v], Z[v]], 1)
    keys = set()
    for dz in (-margin_vox, 0, margin_vox):
        for dy in (-margin_vox, 0, margin_vox):
            for dx in (-margin_vox, 0, margin_vox):
                q = np.floor((P + np.array([dx, dy, dz])) / chunk).astype(np.int64)
                q = np.unique(q, axis=0)
                keys.update(map(tuple, q.tolist()))
    return [f"{z}/{y}/{x}" for x, y, z in sorted(keys, key=lambda k: (k[2], k[1], k[0])) if min(x, y, z) >= 0]


def ensure_volume(scroll: str, tiles: list[Path], log=say) -> Path:
    """Sparse local copy of the CT (level 0) holding only the chunks the tiles need; returns the zarr dir (render_gpu / sheet_snap / recto test read it)."""
    sp = spec(scroll)
    sys.path.insert(0, str(ROOT / "deploy_common"))
    import fetch_assets as FA
    from .manifest import BUCKET
    keys = sorted({k for t in tiles for k in needed_chunks(t)})
    vol = home() / "assets" / scroll / "volume" / sp["volume_zarr"]
    pre = sp["volume_s3_prefix"]
    man = {"assets": [
        {"name": f"{scroll}:volume:meta", "kind": "http", "url": f"https://{BUCKET}.s3.amazonaws.com/{pre}/0/.zarray",
         "dest": f"{scroll}/volume/{sp['volume_zarr']}/0/.zarray"},
        {"name": f"{scroll}:volume:zattrs", "kind": "http", "url": f"https://{BUCKET}.s3.amazonaws.com/{pre}/.zattrs",
         "dest": f"{scroll}/volume/{sp['volume_zarr']}/.zattrs", "optional": True},
        {"name": f"{scroll}:volume:chunks:{len(keys)}:{abs(hash(tuple(keys))) % 10**8}", "kind": "s3keys", "bucket": BUCKET, "prefix": pre + "/0",
         "dest": f"{scroll}/volume/{sp['volume_zarr']}/0", "keys": keys}]}
    log(f"CT chunks needed: {len(keys)} x 128^3 (<= {len(keys) * 2.0 / 1024:.2f} GB if all present)", "volume")
    r = FA.fetch(man, home() / "assets", log=lambda m: log(m, "volume"))
    if not r["ok"]:
        raise StageError("volume: chunk fetch failed: " + "; ".join(r["failed"]))
    za = json.loads((vol / "0" / ".zarray").read_text())
    z, y, x = za["shape"]
    (vol / "meta.json").write_text(json.dumps({"type": "vol", "format": "zarr", "uuid": f"{scroll}_sparse", "name": f"{scroll}-sparse", "width": x, "height": y,
                                              "slices": z, "voxelsize": sp["voxel_um"], "min": 0.0, "max": 255.0}))
    (vol / ".zgroup").write_text('{"zarr_format": 2}')
    return vol


def edge_med_vox(d: Path) -> float:
    X, Y, Z = load_xyz(d)
    v = (X > 0) & (Y > 0) & (Z > 0)
    e = []
    for a, b, va, vb in ((slice(None, -1), slice(1, None), v[:-1], v[1:]), ):
        pass
    P = np.stack([X, Y, Z], -1).astype(np.float64)
    ee = np.concatenate([np.linalg.norm(P[:, 1:] - P[:, :-1], axis=-1)[v[:, 1:] & v[:, :-1]],
                         np.linalg.norm(P[1:] - P[:-1], axis=-1)[v[1:] & v[:-1]]])
    if ee.size == 0:
        return float("nan")
    med = float(np.median(ee))
    ee = ee[ee < 3 * med]
    return float(np.median(ee)) if ee.size else med


def recto_is_reversed(flat: Path, vol: Path):
    import tifffile
    X, Y, Z = (tifffile.imread(flat / f"{a}.tif").astype(np.float64)[::4, ::4] for a in "xyz")
    P = np.stack([X, Y, Z], -1)
    valid = X > 0
    dc, dr = P[:, 1:] - P[:, :-1], P[1:] - P[:-1]
    n = np.cross(dc[:-1], dr[:, :-1])
    m = valid[:-1, :-1] & valid[1:, 1:] & valid[:-1, 1:] & valid[1:, :-1]
    n, pts = n[m], P[:-1, :-1][m]
    if len(n) < 20:
        return None
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    meta = json.loads((vol / "meta.json").read_text())
    ax = np.array([float(meta["width"]) / 2, float(meta["height"]) / 2])
    to = np.stack([ax[0] - pts[:, 0], ax[1] - pts[:, 1], np.zeros(len(pts))], -1)
    to /= np.linalg.norm(to, axis=1, keepdims=True) + 1e-9
    d = float((n * to).sum(1).mean())
    return None if abs(d) < 0.2 else d < 0


def make_mask(layers: Path, out_png: Path):
    import tifffile
    from PIL import Image
    from scipy import ndimage
    files = sorted(layers.glob("*.tif"))
    mid = tifffile.imread(files[len(files) // 2]) > 0
    m = ndimage.binary_fill_holes(ndimage.binary_closing(mid, iterations=3))
    Image.fromarray((m * 255).astype(np.uint8)).save(out_png)


def process_tile(scroll: str, tile: Path, work: Path, vol: Path, families: list[str], snap: str = "recto", gpu: str = "0",
                 device: str = "cuda") -> dict:
    sp = spec(scroll)
    vox = sp["voxel_um"]
    work.mkdir(parents=True, exist_ok=True)
    py, log = env_python(), work / "tile.log"
    rec: dict = {"tile": tile.name, "families": {}}
    t0 = time.time()
    genv = {"CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "4")}
    # ---- snap
    src = tile
    if snap != "off":
        sn = work / "snapped"
        if not (sn / "x.tif").exists():
            r = run([py, ROOT / "routeB" / "sheet_snap.py", tile, vol, sn, "--target", snap, "--radius", str(SNAP_RADIUS)], log=log, env=genv)
            if r != 0 or not (sn / "x.tif").exists():
                raise StageError(f"snap: sheet_snap.py rc={r} on {tile.name}\n{tail(log)}")
        src = sn
    # ---- flatten
    flat = work / "flat"
    if not (flat / "x.tif").exists():
        lw = work / "lasagna"
        shutil.rmtree(lw, ignore_errors=True)
        lw.mkdir(parents=True)
        (lw / "input.json").write_text(json.dumps({"external_surfaces": [{"path": str(src.resolve())}]}))
        t1 = time.time()
        r = run([py, "fit.py", "configs/flatten_fast_nofilter.json", lw / "input.json", "--out-dir", lw, "--device", device], log=log, env=genv,
                cwd=ROOT / "lasagna")
        prod = None
        for c in sorted(lw.rglob("x.tif")):
            prod = c.parent
            break
        if r != 0 or prod is None:
            raise StageError(f"flatten: lasagna fit.py rc={r}, no tifxyz produced for {tile.name}\n{tail(log)}")
        shutil.copytree(prod, flat)
        rec["flatten_s"] = round(time.time() - t1, 1)
    import tifffile
    Xf = tifffile.imread(flat / "x.tif")
    valid = float((Xf > 0).mean())
    if min(Xf.shape) < 32 or valid < 0.2:
        raise StageError(f"flatten: collapsed flat {Xf.shape} valid {valid:.3f} for {tile.name}")
    rec["flat_grid"], rec["flat_valid"] = list(Xf.shape), round(valid, 3)
    # ---- frame / render
    em = edge_med_vox(flat)
    um1 = em / STEP_SIZE * vox
    rscale = round(um1 / TARGET_UM, 4)
    rir = recto_is_reversed(flat, vol)
    rec.update(edge_med_vox=round(em, 3), um_per_px_scale1=round(um1, 4), render_scale=rscale, recto_is_reversed=rir)
    layers = work / "layers"
    if len(glob.glob(str(layers / "*.tif"))) != LAYERS:
        shutil.rmtree(layers, ignore_errors=True)
        layers.mkdir(parents=True)
        t1 = time.time()
        r = run([py, ROOT / "gpu_render" / "render_gpu.py", "--seg", flat, "--vol", vol, "--out", layers, "-n", LAYERS, "--scale", f"{rscale:.4f}",
                 "-g", "0", "--device", "0", "--pool-gb", os.environ.get("ROUTEB_RENDER_POOL_GB", "3"), "--io-threads", "8"], log=log, env=genv)
        n = len(glob.glob(str(layers / "*.tif")))
        if r != 0 or n != LAYERS:
            raise StageError(f"render: render_gpu.py rc={r}, {n}/{LAYERS} layers for {tile.name}\n{tail(log)}")
        rec["render_s"] = round(time.time() - t1, 1)
    mid = tifffile.imread(sorted(layers.glob("*.tif"))[LAYERS // 2])
    frac = float((mid > 0).mean())
    rec["render_shape"], rec["render_valid_frac"] = list(mid.shape), round(frac, 4)
    if min(mid.shape) < 128 or frac < 0.01:
        raise StageError(f"render: unusable render {mid.shape} valid {frac:.4f} for {tile.name} (no CT chunks present for this region?)")
    mask = work / "mask.png"
    make_mask(layers, mask)
    use = layers
    if rir:                                    # forward = the recto face: when the recto is the reversed stack, read reversed
        use = work / "layers_reversed"
        shutil.rmtree(use, ignore_errors=True)
        use.mkdir()
        files = sorted(layers.glob("*.tif"))
        for i, f in enumerate(reversed(files)):
            os.symlink(f.resolve(), use / files[i].name)
    # ---- ink
    out = work / "ink"
    out.mkdir(exist_ok=True)
    t1 = time.time()
    r = run([py, ROOT / "routeB" / "ink_run.py", "--layers", use, "--mask", mask, "--out", out, "--families", ",".join(families)], log=log, env=genv,
            cwd=ROOT / "models")
    rec["ink_s"] = round(time.time() - t1, 1)
    for f in families:
        png = out / f"{f}.png"
        rec["families"][f] = {"png": f"ink/{f}.png" if png.exists() else None}
        if png.exists():
            a = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(png))
            rec["families"][f].update(shape=list(a.shape), mean=round(float(a.mean()) / 255, 4), p99=round(float(np.percentile(a, 99)) / 255, 4))
    bad = [f for f in families if rec["families"][f]["png"] is None]
    rec["wall_s"] = round(time.time() - t0, 1)
    (work / "tile_result.json").write_text(json.dumps(rec, indent=1))
    if bad:
        raise StageError(f"ink: families produced no PNG for {tile.name}: {bad}\n{tail(log)}")
    return rec
