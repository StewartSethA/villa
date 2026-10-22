"""FLATTEN STUDY (Part 2): the same 3-D mesh flattened by the CPU flattener (vc_flatten, ABF++ with a sparse direct solver) and by the GPU flattener of the pipeline (lasagna fit.py, Adam on the GPU, config flatten_fast_nofilter), on labelled PHerc0139 segments.
Input = the open-data tifxyz of each segment (its 3-D points); output = a new tifxyz grid per flattener. Recorded: wall seconds, intrinsic distortion of each result, and the agreement of the two parameterisations with each other and with the open-data one.
  distortion of a tifxyz grid (1 vertex per 1/scale = 20 output px): stretch = |3-D edge| / 20 along rows and along columns (1 = isometric), shear = |angle between the two edge directions - 90 deg|, per valid quad.
  agreement: every vertex of result A is matched to the nearest vertex of result B in 3-D; the grid indices of the matched vertices are related by a 2-D similarity (rotation or reflection, scale, shift) fitted by least squares; the residual in output px (index x 20) is the disagreement of the two parameterisations.
usage: flatten_study.py SEGMENTS (.json list or {"segments": [...]}, or a text file of ids) OUT_JSON [--gpu 1]"""
import sys, os, json, time, subprocess, argparse, shutil
import numpy as np, tifffile
from scipy.spatial import cKDTree
ap = argparse.ArgumentParser(); ap.add_argument("segs"); ap.add_argument("out"); ap.add_argument("--gpu", default="1"); ap.add_argument("--only", default=None); a = ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config
_C = load_config(); D = _C["data_dir"]; F = _C["flat_dir"]; LAS = _C["lasagna_dir"]
def load(p):
    x, y, z = (tifffile.imread(f"{p}/{c}.tif").astype(np.float64) for c in "xyz"); v = (z > 0) & np.isfinite(z) & (x > 0) & (y > 0); return np.stack([x, y, z], -1), v
def distortion(P, v, step=20.0):
    ex = np.linalg.norm(P[:, 1:] - P[:, :-1], axis=-1) / step; ey = np.linalg.norm(P[1:] - P[:-1], axis=-1) / step
    q = v[:-1, :-1] & v[1:, :-1] & v[:-1, 1:] & v[1:, 1:]; tx = (P[:-1, 1:] - P[:-1, :-1]); ty = (P[1:, :-1] - P[:-1, :-1])
    ca = (tx * ty).sum(-1) / (np.linalg.norm(tx, axis=-1) * np.linalg.norm(ty, axis=-1) + 1e-9); sh = np.abs(np.degrees(np.arccos(np.clip(ca, -1, 1))) - 90)
    sx, sy = ex[:-1, :][q], ey[:, :-1][q]; qs = lambda t: [float(np.percentile(t, p)) for p in (10, 50, 90)]
    return dict(stretch_rows_p10_50_90=qs(sx), stretch_cols_p10_50_90=qs(sy), shear_deg_p10_50_90=qs(sh[q]), area_stretch_p50=float(np.median(sx * sy)), quads=int(q.sum()))
def agree(PA, vA, PB, vB):
    ia = np.argwhere(vA); pa = PA[vA]; ib = np.argwhere(vB); pb = PB[vB]; tree = cKDTree(pb); d, j = tree.query(pa); keep = d < 30; ia = ia[keep]; jb = ib[j[keep]]
    A = ia.astype(float) * 20; B = jb.astype(float) * 20; best = None
    for refl in (False, True):
        Bm = B * np.array([1, -1]) if refl else B; ma, mb = A.mean(0), Bm.mean(0); H = (Bm - mb).T @ (A - ma); U, S, Vt = np.linalg.svd(H); R = (U @ Vt).T if False else (Vt.T @ U.T); s = S.sum() / ((Bm - mb) ** 2).sum()
        Ahat = s * (Bm - mb) @ R.T + ma; res = np.linalg.norm(Ahat - A, axis=1)
        if best is None or res.mean() < best[0]: best = (res.mean(), res, refl, s)
    r = best[1]; return dict(matched=int(len(r)), nn3d_median=float(np.median(d[keep])), resid_px_p50_p90_p99=[float(np.percentile(r, q)) for q in (50, 90, 99)], reflection=bool(best[2]), scale=float(best[3]))
rows = []
if a.segs.endswith(".json"): _j = json.load(open(a.segs)); segs = _j["segments"] if isinstance(_j, dict) else _j
else: segs = open(a.segs).read().split()
for seg in segs:
    if a.only and a.only not in seg: continue
    src = f"{D}/{seg}/mesh.tifxyz"; o = f"{F}/{seg}"; os.makedirs(o, exist_ok=True); rec = dict(segment=seg)
    t0 = time.perf_counter(); subprocess.run([_C["cpu_flattener"], "-i", src, "-o", f"{o}/abf_cpu", "--iterations", "10"], capture_output=True); rec["cpu_abf_wall_s"] = time.perf_counter() - t0
    wk = f"{o}/lasagna"; os.makedirs(wk, exist_ok=True); json.dump({"external_surfaces": [{"path": src}]}, open(wk + "/input.json", "w"))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, PYTHONUNBUFFERED="1"); t0 = time.perf_counter()
    subprocess.run([_C["lasagna_python"], "fit.py", "configs/flatten_fast_nofilter.json", wk + "/input.json", "--out-dir", wk, "--device", "cuda"], cwd=LAS, env=env, capture_output=True); rec["gpu_lasagna_wall_s"] = time.perf_counter() - t0
    try:
        ref, vr = load(src); cpu, vc = load(f"{o}/abf_cpu"); gpu, vg = load(f"{wk}/tifxyz/flatten.tifxyz")
        rec.update(valid_ref=int(vr.sum()), valid_cpu=int(vc.sum()), valid_gpu=int(vg.sum()), dist_ref=distortion(ref, vr), dist_cpu=distortion(cpu, vc), dist_gpu=distortion(gpu, vg), cpu_vs_ref=agree(cpu, vc, ref, vr), gpu_vs_ref=agree(gpu, vg, ref, vr), gpu_vs_cpu=agree(gpu, vg, cpu, vc))
    except Exception as e: rec["error"] = repr(e)[:200]
    rows.append(rec); json.dump(rows, open(a.out, "w"), indent=1); print(json.dumps({k: rec.get(k) for k in ("segment", "cpu_abf_wall_s", "gpu_lasagna_wall_s", "error")}), flush=True)
print("FLATTEN_DONE", len(rows))
