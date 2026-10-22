"""RENDER STUDY harness (Part 1): the same window of one labelled segment obtained three ways -
  P  pre-rendered : read the window from the stored surface volume (the upstream open-data render, one uncompressed zarr per segment)
  C  CPU dynamic  : vc_render_tifxyz (CPU, OpenMP) renders the window from the CT volume and the mesh on demand
  G  GPU dynamic  : render_gpu.py (in this folder) renders the same window on a GPU on demand
For each (segment, window) it records bytes, wall seconds, Mvoxel/s and the md5 of every output, and compares C, G and P voxel by voxel.
Positive control: C with --flip-normals must reproduce P (the upstream surface volume was made by this same tool from the same CT volume and mesh).
FLEXIBILITY (same call, other depth / face / offset), one window per segment: G at D = 17 / 33 / 65, depth offsets, and the VERSO side (negative slice step) - checked against G at D = 65 (identical voxels where the planes coincide).
usage: bench_render.py SEGMENT_JSON OUT_JSON [--windows 3] [--size 1024] [--gpu 1] [--threads 16] [--seed 0]"""
import sys, os, json, time, subprocess, hashlib, glob, argparse, shutil, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ap = argparse.ArgumentParser(); ap.add_argument("segs"); ap.add_argument("out"); ap.add_argument("--windows", type=int, default=3); ap.add_argument("--size", type=int, default=1024); ap.add_argument("--gpu", type=int, default=1)
ap.add_argument("--threads", type=int, default=16); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--only", default=None); ap.add_argument("--flex", type=int, default=1); ap.add_argument("--gpu-first", action="store_true", help="render with the GPU BEFORE the CPU touches the window: the GPU call then reads the CT chunks cold (page cache not warmed by the CPU run)"); a = ap.parse_args()
import tifffile, zarr, torch
import render_gpu
from common import load_config
_C = load_config(); D = _C["data_dir"]; VOL = _C["volume"]; CPU = _C["cpu_renderer"]; NS = _C["num_slices"]
def md5(x): return hashlib.md5(np.ascontiguousarray(x).tobytes()).hexdigest()
def du(p): return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs)
def cpu_render(mesh, y0, x0, h, w, n=NS, flip=True, tmp=None):
    t0 = time.perf_counter(); o = tempfile.mkdtemp(dir=tmp or _C["tmp_dir"])
    cmd = [CPU, "-v", VOL, "--scale", "1", "-g", "0", "-s", mesh, "-n", str(n), "--crop-x", str(x0), "--crop-y", str(y0), "--crop-width", str(w), "--crop-height", str(h), "--tif-output", o, "--threads", str(a.threads)] + (["--flip-normals"] if flip else [])
    r = subprocess.run(cmd, capture_output=True); t1 = time.perf_counter()
    arr = np.stack([tifffile.imread(f) for f in sorted(glob.glob(o + "/*.tif"))]); t2 = time.perf_counter(); shutil.rmtree(o, ignore_errors=True)
    return arr, t1 - t0, t2 - t0
def gpu_render(mesh, y0, x0, h, w, n=NS, step=-1.0, zoff=0.0):
    t0 = time.perf_counter(); r = render_gpu.render(mesh, VOL, os.path.join(_C["tmp_dir"], "unused"), num_slices=n, slice_step=step, rect=(y0, y0 + h, x0, x0 + w), device=0, write_tif=False, verbose=False, z_offset=zoff, io_threads=16); torch.cuda.synchronize()
    return r["out"], time.perf_counter() - t0
os.makedirs(_C["tmp_dir"], exist_ok=True)
segs = json.load(open(a.segs))["segments"]
if a.only: segs = [s for s in segs if a.only in s]
rows = []; rng = np.random.default_rng(a.seed)
for seg in segs:
    d = f"{D}/{seg}"; mesh = f"{d}/mesh.tifxyz"; svp = f"{d}/sv.zarr"
    if not (os.path.exists(mesh + "/x.tif") and os.path.exists(svp + "/.done")): print("SKIP (data not pulled yet)", seg, flush=True); continue
    sv = zarr.open_array(svp + "/0", mode="r"); NZ, H, W = sv.shape; mid = np.asarray(sv[NS // 2, ::16, ::16]) > 0
    sz = a.size; ok = [(y, x) for y in range(0, H - sz, 128) for x in range(0, W - sz, 128) if mid[y // 16:(y + sz) // 16, x // 16:(x + sz) // 16].mean() > 0.97]
    if not ok: print("no valid window", seg); continue
    pick = [ok[i] for i in rng.choice(len(ok), min(a.windows, len(ok)), replace=False)]
    bytes_pre = du(svp); bytes_mesh = du(mesh)
    for wi, (y0, x0) in enumerate(pick):
        rec = dict(segment=seg, window=wi, y0=y0, x0=x0, size=sz, planes=NS, canvas=[H, W], disk_bytes_prerendered_segment=bytes_pre, disk_bytes_mesh=bytes_mesh)
        t0 = time.perf_counter(); P = np.asarray(sv[:, y0:y0 + sz, x0:x0 + sz]); rec["P_wall_s"] = time.perf_counter() - t0; rec["P_md5"] = md5(P)
        if a.gpu_first: G1, tg1 = gpu_render(mesh, y0, x0, sz, sz)          # cold chunk reads (nothing else has touched this window)
        C, tc_proc, tc_all = cpu_render(mesh, y0, x0, sz, sz); rec.update(C_wall_s=tc_proc, C_wall_incl_read_s=tc_all, C_md5=md5(C))
        if not a.gpu_first: G1, tg1 = gpu_render(mesh, y0, x0, sz, sz)
        G, tg = gpu_render(mesh, y0, x0, sz, sz); rec["gpu_first_order"] = bool(a.gpu_first)             # first call (CUDA / pool warm-up) and steady call
        rec.update(G_first_wall_s=tg1, G_wall_s=tg, G_md5=md5(G))
        for nm, X, Y in (("C_vs_P", C, P), ("G_vs_P", G, P), ("G_vs_C", G, C)): rec[nm] = dict(frac_equal=float((X == Y).mean()), max_abs=int(np.abs(X.astype(int) - Y).max()), n_diff=int((X != Y).sum()))
        # unflipped CPU = reversed plane order: the other orientation (documented, not an error)
        rec["C_noflip_vs_P_reversed_equal"] = None
        if wi == 0 and a.flex:                                                        # the SAME call for other depth / offset / face
            G65, t65 = gpu_render(mesh, y0, x0, sz, sz, n=65, step=-1.0); rec["flex"] = {"D65_wall_s": t65}
            for Dn in (17, 33):
                for zo in (0.0, 3.0, -4.0):
                    Gd, td = gpu_render(mesh, y0, x0, sz, sz, n=Dn, step=-1.0, zoff=zo)
                    # plane zi of this call is at offset (zi - (Dn-1)/2 + zo) steps; in the D65 render plane k is at (k - 32): k = zi - (Dn-1)/2 + zo + 32
                    k0 = 32 - (Dn - 1) / 2 + zo; ok_int = abs(k0 - round(k0)) < 1e-6
                    rec["flex"][f"D{Dn}_z{zo:+.1f}"] = dict(wall_s=td, md5=md5(Gd), equal_to_D65_subset=(bool(np.array_equal(Gd, G65[int(round(k0)):int(round(k0)) + Dn])) if ok_int and 0 <= round(k0) and round(k0) + Dn <= 65 else None))
            Gv, tv = gpu_render(mesh, y0, x0, sz, sz, n=33, step=+1.0, zoff=0.0)       # the opposite face / direction: step +1 = un-flipped normals = planes in reverse order of the D65 render
            rec["flex"]["opposite_direction_D33"] = dict(wall_s=tv, equal_to_reversed_D65_subset=bool(np.array_equal(Gv, G65[::-1][16:49])), md5=md5(Gv))
            prof = [float(G65[k][G65[k] > 0].mean()) if (G65[k] > 0).any() else 0.0 for k in range(65)]; rec["flex"]["depth_profile_mean_grey_D65"] = [round(x, 1) for x in prof]
        rows.append(rec); print(json.dumps({k: rec[k] for k in ("segment", "window", "P_wall_s", "C_wall_s", "G_wall_s", "G_first_wall_s")}), "C_vs_P", rec["C_vs_P"]["frac_equal"], "G_vs_P", rec["G_vs_P"]["frac_equal"], "G_vs_C", rec["G_vs_C"]["frac_equal"], flush=True)
        json.dump(rows, open(a.out, "w"))
print("BENCH_DONE", len(rows), flush=True)
