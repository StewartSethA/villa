#!/usr/bin/env python3
"""Hecate 9.6 um 3-D probability volumes over whole surface-volume arrays, written band by band (resumable).

Input  ROOT/<TAG>/vol17.zarr  (17, H, W) uint8: 17 planes of a surface-conditioned render, 9.6 um apart, mesh at plane 8.
Output ROOT/<TAG>/hec3d.zarr  (16, H, W) uint8 = round(255 x p) for the planes 0..15, plus hec3d.DONE when finished.
Needs hecate.py + hecate_9.6um.pth from https://huggingface.co/scrollprize/hecate (set HECATE_PY / HECATE_CKPT).  Use --reverse when the
render's depth direction is the other way round (see ../render_direction_check).
usage: hecate_teacher_volumes.py ROOT TAG [TAG ...] [--reverse] [--band 1536] [--halo 64] [--batch 4]"""
import argparse, importlib.util, json, os, time
import numpy as np, zarr

ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("tags", nargs="+")
ap.add_argument("--band", type=int, default=1536); ap.add_argument("--halo", type=int, default=64); ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--reverse", action="store_true")
a = ap.parse_args()
spec = importlib.util.spec_from_file_location("hecate", os.environ.get("HECATE_PY", "hecate.py")); H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
model = H.load_model(os.environ.get("HECATE_CKPT", "hecate_9.6um.pth"), "cuda"); t0 = time.time()
for tag in a.tags:
    d = f"{a.root}/{tag}"
    if os.path.exists(d + "/hec3d.DONE"):
        print(tag, "done already"); continue
    V = zarr.open(d + "/vol17.zarr", mode="r"); _, h, w = V.shape
    O = zarr.open(d + "/hec3d.zarr", mode="a", shape=(16, h, w), chunks=(16, 512, 512), dtype="uint8")
    prog = d + "/hec3d.progress"; done = int(open(prog).read()) if os.path.exists(prog) else 0
    for y0 in range(done, h, a.band):
        y1 = min(h, y0 + a.band); ya, yb = max(0, y0 - a.halo), min(h, y1 + a.halo)          # halo rows of context, so bands have no seams
        vol = np.asarray(V[:, ya:yb, :]); res = np.zeros((vol.shape[0], yb - ya, w), np.uint8)
        H.predict(model, vol, res, reverse=a.reverse, stride=None, batch_size=a.batch, precision="fp32", three_d=True)
        O[:, y0:y1, :] = res[:16, y0 - ya:y1 - ya, :]; open(prog, "w").write(str(y1))
        print(f"{tag} rows {y1}/{h} {time.time() - t0:.0f}s", flush=True)
    json.dump(dict(tag=tag, shape=[16, h, w], model="scrollprize/hecate hecate_9.6um.pth", reverse=a.reverse, built=time.strftime("%FT%TZ", time.gmtime())), open(d + "/hec3d.json", "w"))
    open(d + "/hec3d.DONE", "w").write("ok\n")
