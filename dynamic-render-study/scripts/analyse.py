"""RENDER STUDY analysis: read results_bench.json (+ results_full.json), print the tables used in the study (p10/p50/p90 over windows and over segments)."""
import sys, json, numpy as np, os
B = json.load(open(sys.argv[1])); F = json.load(open(sys.argv[2])) if len(sys.argv) > 2 and os.path.exists(sys.argv[2]) else []
def q(x): x = np.asarray(x, float); return "%.4g / %.4g / %.4g" % tuple(np.percentile(x, [10, 50, 90]))
n = len(B); segs = sorted({r["segment"] for r in B}); vox = B[0]["planes"] * B[0]["size"] ** 2 / 1e6
print(f"windows n = {n}; segments n = {len(segs)}; window {B[0]['size']}x{B[0]['size']} px x {B[0]['planes']} planes = {vox:.2f} Mvoxel")
print("\nWALL SECONDS per window (p10 / p50 / p90):")
for k, nm in (("P_wall_s", "pre-rendered read"), ("C_wall_s", "CPU render (process)"), ("C_wall_incl_read_s", "CPU render + read TIFFs back"), ("G_first_wall_s", "GPU render, first call"), ("G_wall_s", "GPU render, steady")): print(f"  {nm:34s} {q([r[k] for r in B])}")
print("\nMvoxel/s per window (p10 / p50 / p90):")
for k, nm in (("P_wall_s", "pre-rendered read"), ("C_wall_s", "CPU render"), ("C_wall_incl_read_s", "CPU render + read-back"), ("G_wall_s", "GPU render, steady")): print(f"  {nm:34s} {q([vox / r[k] for r in B])}")
print("\nPER-SEGMENT medians of the window medians (n segments):")
for k, nm in (("P_wall_s", "pre-rendered read s"), ("C_wall_s", "CPU render s"), ("G_wall_s", "GPU render s")):
    print(f"  {nm:34s} {q([np.median([r[k] for r in B if r['segment'] == s]) for s in segs])}")
print("\nIDENTITY (fraction of voxels exactly equal; max |diff| grey levels):")
for k in ("C_vs_P", "G_vs_P", "G_vs_C"): print(f"  {k}: frac_equal p10/p50/p90 {q([r[k]['frac_equal'] for r in B])}; max|diff| over all windows {max(r[k]['max_abs'] for r in B)}; windows with ZERO differing voxels {sum(r[k]['n_diff']==0 for r in B)}/{n}; total differing voxels {sum(r[k]['n_diff'] for r in B)} of {n*vox*1e6:.4g}")
print("\nDISK (bytes per segment): pre-rendered 28-plane zarr p10/p50/p90 GB", q([r['disk_bytes_prerendered_segment'] / 1e9 for r in {x['segment']: x for x in B}.values()]), "; mesh (dynamic) MB", q([r['disk_bytes_mesh'] / 1e6 for r in {x['segment']: x for x in B}.values()]))
if F:
    print("\nPRE-RENDER COST (whole canvas, once), n segments =", len(F))
    for k in ("voxels", "cpu_full_wall_s_with_tiff_write", "cpu_full_Mvox_s_with_tiff_write", "gpu_full_wall_s_no_write", "gpu_full_Mvox_s_no_write", "gpu_zarr_write_s", "prerendered_zarr_bytes", "cpu_full_tiff_bytes"): print(f"  {k:36s} {q([r[k] for r in F])}")
fl = [r["flex"] for r in B if "flex" in r]
if fl:
    print("\nFLEXIBILITY (same GPU call; one window per segment, n =", len(fl), ")")
    keys = [k for k in fl[0] if k.startswith("D") and k != "D65_wall_s" or k.startswith("opposite")]
    for k in keys:
        ws = [f[k]["wall_s"] for f in fl if k in f]; eq = [f[k].get("equal_to_D65_subset", f[k].get("equal_to_reversed_D65_subset")) for f in fl if k in f]
        print(f"  {k:26s} wall s p10/p50/p90 {q(ws)}; identical to the matching planes of the D65 render: {sum(1 for e in eq if e is True)} true / {sum(1 for e in eq if e is False)} false / {sum(1 for e in eq if e is None)} not comparable")
    print("  D65 wall s", q([f['D65_wall_s'] for f in fl]))
