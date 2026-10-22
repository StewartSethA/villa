"""RENDER STUDY, EXCLUSIVE-DEVICE run (Part 1 redo): same three paths and the same windows as bench_render.py, but
  * one GPU reserved (no other process on it), benchmark pinned to cores 16-31 (taskset), the machine state is logged before / during / after;
  * every (window, path) is timed REPEATS times; the order of the three paths is RANDOMISED in every repeat;
  * cache state: repeats 0 and 1 are COLD (the file pages of the stored zarr and of the CT chunks this window needs are dropped with posix_fadvise DONTNEED before EACH timed call), repeats 2.. are WARM (after an untimed warm-up of all three paths);
  * one untimed GPU call per process initialises CUDA; the first timed GPU call is reported anyway.
Records per call: segment, window, repeat, state, path, wall seconds, md5 of the output. Identity against the stored render is taken from every call (all repeats).
usage: bench_render_excl.py SEGMENT_JSON OUT_JSON [--repeats 5] [--windows 3] [--size 1024] [--threads 16]   (run under: taskset -c 16-31, CUDA_VISIBLE_DEVICES=<reserved card>)"""
import sys, os, json, time, subprocess, hashlib, glob, argparse, shutil, tempfile, random
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
ap = argparse.ArgumentParser(); ap.add_argument("segs"); ap.add_argument("out"); ap.add_argument("--repeats", type=int, default=5); ap.add_argument("--windows", type=int, default=3); ap.add_argument("--size", type=int, default=1024)
ap.add_argument("--threads", type=int, default=16); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--only", default=None); a = ap.parse_args()
import tifffile, zarr, torch
import render_gpu
from common import load_config
_C = load_config(); D = _C["data_dir"]; VOL = _C["volume"]; CPU = _C["cpu_renderer"]; NS = _C["num_slices"]; T = _C["tmp_dir"]
def md5(x): return hashlib.md5(np.ascontiguousarray(x).tobytes()).hexdigest()
def drop(paths):
    for p in paths:
        try: fd = os.open(p, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
        except OSError: pass
def ct_chunks(mesh, y0, x0, sz):
    z = tifffile.imread(mesh + "/z.tif"); x = tifffile.imread(mesh + "/x.tif"); y = tifffile.imread(mesh + "/y.tif"); g0, g1, h0, h1 = y0 // 20, (y0 + sz) // 20 + 2, x0 // 20, (x0 + sz) // 20 + 2
    zz, yy, xx = z[g0:g1, h0:h1], y[g0:g1, h0:h1], x[g0:g1, h0:h1]; ok = zz > 0; pad = 40
    lo = [int(v) - pad for v in (zz[ok].min(), yy[ok].min(), xx[ok].min())]; hi = [int(v) + pad for v in (zz[ok].max(), yy[ok].max(), xx[ok].max())]
    # tifxyz x.tif = X, y.tif = Y, z.tif = Z (volume axes z,y,x): chunk files VOL/0/<z>/<y>/<x>
    fs = []
    for cz in range(max(lo[0], 0) // 128, hi[0] // 128 + 1):
        for cy in range(max(lo[1], 0) // 128, hi[1] // 128 + 1):
            for cx in range(max(lo[2], 0) // 128, hi[2] // 128 + 1):
                p = f"{VOL}/0/{cz}/{cy}/{cx}"
                if os.path.exists(p): fs.append(p)
    return fs
def cpu_render(mesh, y0, x0, sz):
    t0 = time.perf_counter(); o = tempfile.mkdtemp(dir=T)
    subprocess.run([CPU, "-v", VOL, "--scale", "1", "-g", "0", "-s", mesh, "-n", str(NS), "--crop-x", str(x0), "--crop-y", str(y0), "--crop-width", str(sz), "--crop-height", str(sz), "--tif-output", o, "--threads", str(a.threads), "--flip-normals"], capture_output=True)
    t1 = time.perf_counter(); arr = np.stack([tifffile.imread(f) for f in sorted(glob.glob(o + "/*.tif"))]); shutil.rmtree(o, ignore_errors=True); return arr, t1 - t0
def gpu_render(mesh, y0, x0, sz):
    t0 = time.perf_counter(); r = render_gpu.render(mesh, VOL, "/tmp/jm_rs_unused", num_slices=NS, slice_step=-1.0, rect=(y0, y0 + sz, x0, x0 + sz), device=0, write_tif=False, verbose=False, io_threads=16); torch.cuda.synchronize(); return r["out"], time.perf_counter() - t0
os.makedirs(T, exist_ok=True); rng = np.random.default_rng(a.seed); rnd = random.Random(a.seed); rows = []
gpu_render(f"{D}/{json.load(open(a.segs))['segments'][0]}/mesh.tifxyz", 3000, 2000, 256)        # untimed CUDA / pool initialisation
for seg in json.load(open(a.segs))["segments"]:
    if a.only and a.only not in seg: continue
    d = f"{D}/{seg}"; mesh = f"{d}/mesh.tifxyz"; svp = f"{d}/sv.zarr"
    if not (os.path.exists(mesh + "/x.tif") and os.path.exists(svp + "/.done")): continue
    sv = zarr.open_array(svp + "/0", mode="r"); NZ, H, W = sv.shape; mid = np.asarray(sv[NS // 2, ::16, ::16]) > 0; sz = a.size
    ok = [(y, x) for y in range(0, H - sz, 128) for x in range(0, W - sz, 128) if mid[y // 16:(y + sz) // 16, x // 16:(x + sz) // 16].mean() > 0.97]
    if not ok: continue
    pick = [ok[i] for i in rng.choice(len(ok), min(a.windows, len(ok)), replace=False)]; svfiles = [os.path.join(r, f) for r, _, fs in os.walk(svp) for f in fs]
    for wi, (y0, x0) in enumerate(pick):
        ctf = ct_chunks(mesh, y0, x0, sz)
        def run(path, state):
            if state == "cold": drop(svfiles if path == "P" else ctf)
            if path == "P": t0 = time.perf_counter(); out = np.asarray(sv[:, y0:y0 + sz, x0:x0 + sz]); return out, time.perf_counter() - t0
            return (cpu_render if path == "C" else gpu_render)(mesh, y0, x0, sz)
        for p in "PCG": run(p, "warm")                                                     # untimed warm-up of all three for the warm repeats
        ref = None
        for rep in range(a.repeats):
            state = "cold" if rep < 2 else "warm"; order = list("PCG"); rnd.shuffle(order)
            for p in order:
                out, dt = run(p, state)
                if p == "P" and ref is None: ref = out
                rows.append(dict(segment=seg, window=wi, y0=y0, x0=x0, repeat=rep, state=state, path=p, wall_s=dt, md5=md5(out), order="".join(order), ct_chunk_files=len(ctf)))
                if p != "P" and ref is not None: rows[-1]["frac_equal_vs_stored"] = float((out == ref).mean()); rows[-1]["max_abs_vs_stored"] = int(np.abs(out.astype(int) - ref).max())
        print(seg[:30], wi, "done", flush=True)
    json.dump(rows, open(a.out, "w"))
print("EXCL_DONE", len(rows), flush=True)
