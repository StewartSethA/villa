"""CPU-ONLY comparison (no torch, no GPU, no CUDA): renders windows with vc_render_tifxyz and compares them with the stored surface volume,
voxel by voxel, and times both. Uses the windows recorded in results/results_bench.json (same segment, y0, x0, size), so the numbers are comparable.
usage: cpu_only_check.py [--bench results/results_bench.json] [--limit 6] [--threads 16]   (paths from config.json, see README)"""
import argparse, glob, hashlib, json, os, shutil, subprocess, sys, tempfile, time
import numpy as np, tifffile, zarr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config
ap = argparse.ArgumentParser(); ap.add_argument("--bench", default=None); ap.add_argument("--limit", type=int, default=6); ap.add_argument("--threads", type=int, default=16); a = ap.parse_args()
C = load_config(); bench = a.bench or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "results_bench.json")
rows = [r for r in json.load(open(bench)) if r.get("window") == 0][:a.limit]; NS = C["num_slices"]; out = []
for r in rows:
    d = os.path.join(C["data_dir"], r["segment"]); mesh = os.path.join(d, "mesh.tifxyz"); svp = os.path.join(d, "sv.zarr", "0")
    if not (os.path.exists(os.path.join(mesh, "x.tif")) and os.path.exists(svp)): print("SKIP (data not pulled)", r["segment"]); continue
    y0, x0, sz = r["y0"], r["x0"], r["size"]
    t0 = time.perf_counter(); P = np.asarray(zarr.open_array(svp, mode="r")[:, y0:y0 + sz, x0:x0 + sz]); tp = time.perf_counter() - t0
    o = tempfile.mkdtemp(dir=C["tmp_dir"]); t0 = time.perf_counter()
    cmd = [C["cpu_renderer"], "-v", C["volume"], "--scale", "1", "-g", "0", "-s", mesh, "-n", str(NS), "--crop-x", str(x0), "--crop-y", str(y0), "--crop-width", str(sz), "--crop-height", str(sz), "--tif-output", o, "--threads", str(a.threads), "--flip-normals"]
    subprocess.run(cmd, capture_output=True, check=True); tc = time.perf_counter() - t0
    X = np.stack([tifffile.imread(f) for f in sorted(glob.glob(o + "/*.tif"))]); shutil.rmtree(o, ignore_errors=True)
    rec = dict(segment=r["segment"], stored_read_s=round(tp, 3), cpu_render_s=round(tc, 3), frac_equal=float((X == P).mean()), max_abs=int(np.abs(X.astype(int) - P).max()), md5_cpu=hashlib.md5(np.ascontiguousarray(X).tobytes()).hexdigest())
    print(json.dumps(rec), flush=True); out.append(rec)
json.dump(out, open(os.path.join(C["out_dir"], "cpu_only_check.json"), "w"), indent=1)
