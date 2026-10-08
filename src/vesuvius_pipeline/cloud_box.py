"""DB-FREE grow driver for an UNTRUSTED rented box (cloud_grow_2026-10-08). No hub token, no ssh keys, no DB, no scroll other than the one being grown. It runs the tracer rounds exactly as
stages.grow.run_round builds the command (same params.json keys, env), applies the GEOMETRY-ONLY guards that need nothing but the checkpoint and the kit's vc_tifxyz_selfcross
(growth_guard.selfx_scrub = transverse crossings cut to verified zero; resume_gate.check = degeneracy gate, fail closed: a failing surface is not grown on) between rounds, and writes the
export tree the hub imports (export.json + md5.txt; cloud_import.py verifies it). CT-dependent guard criteria (vacuum / ridge / seam / curvature) are NOT run here: the hub re-scores.

  python -m vesuvius_pipeline.cloud_box --kit KIT --pred PRED.zarr --grids GRIDS --scroll PHerc0211 --seed X Y Z [--seed ...] --out OUT --rounds 6 --gens 20 --voxel-um 9.362 --threads 1
Per seed: OUT/<seg>/r<k>/ (tracer targets), OUT/<seg>/export.json, OUT/<seg>/md5.txt. A seed whose surface fails the gate stops there (status `gate_held`) and is still exported (the hub quarantines it).
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path

SCHEMA = "vpipe-remote-grow-export/1"


def _md5(p) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def newest_checkpoint(tgt: Path):
    cks = [p.parent for p in tgt.glob("*/meta.json") if (p.parent / "x.tif").exists()]
    return max(cks, key=lambda d: ((d / "meta.json").stat().st_mtime, d.name)) if cks else None


def tracer_cmd(kit, vol, tgt, params_json, resume=None, seed=None):
    cmd = [os.path.join(kit, "bin", "vc_grow_seg_from_seed"), "-v", str(vol), "-t", str(tgt), "--params", str(params_json)]
    return cmd + (["--resume", str(resume)] if resume else ["--seed", *(str(int(s)) for s in seed)])


def grow_seed(kit, vol, grids, scroll, seed, out, rounds, gens, voxel_um, threads=1, rng=1, pol=None, self_collision=True, round_timeout=6 * 3600, run=subprocess.run,
              deadline=None, gate_override=None):
    """One seed, `rounds` rounds, geometry guards between rounds. Returns the export dict (also written to disk)."""
    from . import growth_guard as GG, resume_gate as RG
    out = Path(out)
    seg = f"{scroll}_c{hashlib.md5(('%s:%s:%s:%s' % ((scroll,) + tuple(seed))).encode()).hexdigest()[:7]}"
    sd = out / seg
    sd.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, LD_LIBRARY_PATH=os.path.join(kit, "lib"), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS=str(threads),
               VC_GRID_CACHE_BYTES=str(256 * 1024 * 1024), VC_GROWPATCH_RNG_SEED=str(rng))
    pol = pol or GG.GuardPolicy(selfcross=True, selfcross_bin=os.path.join(kit, "bin", "vc_tifxyz_selfcross"), selfcross_env={"LD_LIBRARY_PATH": os.path.join(kit, "lib")}, selfcross_threads=2)
    cur, status, rr = None, "grown", []
    t_all = time.time()
    for k in range(1, rounds + 1):
        if deadline is not None and time.time() >= deadline:
            status = "deadline"                                                      # --hours budget reached: stop between rounds (checkpoint kept, exported)
            break
        tgt = sd / f"r{k}"
        params = {"mode": "seed", "generations": gens * k, "min_area_cm": 0.002, "step_size": 20.0, "thread_limit": threads, "voxelsize": voxel_um,
                  "normal_grid_path": str(grids), "snapshot-interval": 1 if cur else 5, "cache_root": str(sd / "cache_root")}
        if self_collision:
            params["self_collision"] = {"enabled": True, "weight": 0}
        pj = sd / f"params_r{k}.json"
        pj.write_text(json.dumps(params))
        cmd = tracer_cmd(kit, vol, tgt, pj, resume=cur, seed=seed)
        r0 = resource.getrusage(resource.RUSAGE_CHILDREN); t0 = time.time()
        with open(sd / f"round{k}.log", "ab") as lg:
            lg.write((" ".join(cmd) + "\n").encode())
            try:
                p = run(cmd, stdout=lg, stderr=subprocess.STDOUT, env=env, timeout=(round_timeout if deadline is None else max(60, min(round_timeout, deadline - time.time()))))
                timed_out = False
            except subprocess.TimeoutExpired:                                           # budget reached mid-round: the tracer's last snapshot is a valid checkpoint
                p, timed_out = None, True
        r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        ck = newest_checkpoint(tgt)
        rr.append({"round": k, "rc": getattr(p, "returncode", None), "timed_out": timed_out, "wall_s": round(time.time() - t0, 1), "cpu_s": round((r1.ru_utime + r1.ru_stime) - (r0.ru_utime + r0.ru_stime), 1),
                   "checkpoint": str(ck.relative_to(sd)) if ck else None})
        if ck is None:
            status = "no_checkpoint"
            break
        path, info = GG.selfx_scrub(str(ck), pol, voxel_um)                         # transverse crossings cut to a verified zero (never smaller than needed)
        rr[-1]["scrub"] = {k2: info.get(k2) for k2 in ("ran", "clean", "removed_cells", "iters", "error")}
        if path is None:
            status = "scrub_failed"
            break
        cur = path
        gate = RG.check(cur, pol, gate_override)                                    # degeneracy gate, fail closed
        rr[-1]["gate"] = {"pass": gate.get("pass"), "reasons": gate.get("reasons"), "fractions": gate.get("fractions")}
        if not gate.get("pass"):
            status = "gate_held"
            break
        if timed_out:
            status = "deadline"
            break
    libs = {}
    for n in ("libvc_tracer.so", "libvc_core.so"):
        fp = Path(kit) / "lib" / n
        libs[n] = _md5(fp) if fp.exists() else None
    ex = {"schema": SCHEMA, "identity": {"seg": seg, "scroll": scroll, "seed_xyz": list(seed), "rng_seed": rng},
          "tool": {"md5": _md5(Path(kit) / "bin" / "vc_grow_seg_from_seed"), "kit_libs_md5": libs, "selfcross_md5": _md5(Path(kit) / "bin" / "vc_tifxyz_selfcross")},
          "run": {"host": os.uname().nodename, "threads": threads, "status": status, "wall_s_total": round(time.time() - t_all, 1)}, "rounds": rr,
          "guards_on_box": ["in-solve self_collision" if self_collision else "none", "selfx_scrub (transverse -> 0)", "resume_gate (degeneracy, fail closed)"],
          "guards_NOT_on_box": ["CT-dependent criteria (vacuum/ridge/seam/curvature): hub re-scores"]}
    (sd / "export.json").write_text(json.dumps(ex, indent=1))
    import shutil
    shutil.rmtree(sd / "cache_root", ignore_errors=True)
    lines = [f"{_md5(fp)}  {fp.relative_to(sd)}" for fp in sorted(sd.rglob("*")) if fp.is_file() and not fp.is_symlink() and fp.name != "md5.txt"]
    (sd / "md5.txt").write_text("\n".join(lines) + "\n")
    return ex


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m vesuvius_pipeline.cloud_box")
    ap.add_argument("--kit", required=True); ap.add_argument("--pred", required=True); ap.add_argument("--grids", required=True); ap.add_argument("--scroll", required=True)
    ap.add_argument("--seed", nargs=3, type=int, action="append", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=6); ap.add_argument("--gens", type=int, default=20); ap.add_argument("--voxel-um", type=float, required=True)
    ap.add_argument("--threads", type=int, default=1); ap.add_argument("--rng", type=int, default=1)
    a = ap.parse_args(argv)
    for s in a.seed:
        ex = grow_seed(a.kit, a.pred, a.grids, a.scroll, s, a.out, a.rounds, a.gens, a.voxel_um, a.threads, a.rng)
        print(json.dumps({"seg": ex["identity"]["seg"], "status": ex["run"]["status"], "rounds": len(ex["rounds"])}), flush=True)


if __name__ == "__main__":
    main()
