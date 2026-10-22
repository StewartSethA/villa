"""RENDER STUDY, the cost of PRE-rendering: render each whole segment canvas (all 28 planes) once on the CPU and once on the GPU, store it as the uncompressed zarr that training would read later, and record wall seconds, Mvoxel/s and bytes.
usage: bench_full.py SEGMENT_JSON OUT_JSON [--gpu 1] [--threads 16]"""
import sys, os, json, time, subprocess, argparse, tempfile, shutil, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ap = argparse.ArgumentParser(); ap.add_argument("segs"); ap.add_argument("out"); ap.add_argument("--threads", type=int, default=16); ap.add_argument("--only", default=None); a = ap.parse_args()
import torch, zarr, render_gpu
from common import load_config
_C = load_config(); D = _C["data_dir"]; VOL = _C["volume"]; CPU = _C["cpu_renderer"]; NS = _C["num_slices"]; T = _C["tmp_dir"]
def du(p): return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs)
rows = []
for seg in json.load(open(a.segs))["segments"]:
    if a.only and a.only not in seg: continue
    d = f"{D}/{seg}"; mesh = f"{d}/mesh.tifxyz"
    if not os.path.exists(d + "/sv.zarr/.done"): continue
    H, W = zarr.open_array(d + "/sv.zarr/0", mode="r").shape[1:]; rec = dict(segment=seg, canvas=[H, W], voxels=NS * H * W)
    o = tempfile.mkdtemp(dir=T); t0 = time.perf_counter()
    subprocess.run([CPU, "-v", VOL, "--scale", "1", "-g", "0", "-s", mesh, "-n", str(NS), "--flip-normals", "--tif-output", o, "--threads", str(a.threads)], capture_output=True); rec["cpu_full_wall_s_with_tiff_write"] = time.perf_counter() - t0; rec["cpu_full_tiff_bytes"] = du(o); shutil.rmtree(o, ignore_errors=True)
    t0 = time.perf_counter(); r = render_gpu.render(mesh, VOL, "/tmp/jm_rs_unused", num_slices=NS, slice_step=-1.0, device=0, write_tif=False, verbose=False, io_threads=16); torch.cuda.synchronize(); rec["gpu_full_wall_s_no_write"] = time.perf_counter() - t0
    z = tempfile.mkdtemp(dir=T); t0 = time.perf_counter(); zz = zarr.open_array(z + "/0", mode="w", shape=r["out"].shape, chunks=(NS, 128, 128), dtype="u1", compressor=None, zarr_format=2); zz[:] = r["out"]; rec["gpu_zarr_write_s"] = time.perf_counter() - t0; rec["prerendered_zarr_bytes"] = du(z); shutil.rmtree(z, ignore_errors=True)
    for k in ("cpu_full_wall_s_with_tiff_write", "gpu_full_wall_s_no_write"): rec[k.replace("wall_s", "Mvox_s")] = rec["voxels"] / 1e6 / rec[k]
    rows.append(rec); print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in rec.items() if k != "canvas"}), flush=True); json.dump(rows, open(a.out, "w"))
print("FULL_DONE", len(rows))
