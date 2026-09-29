"""guarded-grow: run VC3D's `vc_grow_seg_from_seed` round by round with the growth guard between rounds.

Each round grows the current checkpoint for a fixed number of generations (`--seed` first,
`--resume` afterwards), then `growth_guard.guard_round` checks the new surface (self-intersections
via upstream's `vc_tifxyz_selfcross`, plus the per-cell criteria of the chosen policy), writes a
trimmed checkpoint beside the tracer's when it cuts anything, and says whether to stop. The next
round resumes from the trimmed checkpoint. The tracer, its parameters and its binary are not
changed: this is what a person does by hand when they stop a grow, inspect it, trim it and resume.

Outputs under --out:
  params_rN.json, roundN.log      the exact tracer parameters and log of each round
  rN/<checkpoint>/                the tracer's checkpoint; guarded_g_<checkpoint>/ when trimmed
  guard_report.jsonl              one line per round: tracer CPU, guard seconds, guard summary
  result.json                     final checkpoint, stop reason, area, CPU, provenance

Usage:
  guarded-grow --volume VOL.zarr --normal-grids GRIDS/ --seed X Y Z --out OUT/ \
      --policy presets/D.json [--ct CT.zarr] [--pred PRED.zarr] [--umbilicus UMB.json] \
      [--vc-bin DIR] [--params extra.json] [--first-gens 10] [--gen-step 10] [--wall-s 1800]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import growth_guard as GG

EXHAUSTED_MM2 = 50.0        # a resumed round that adds less than this is "exhausted"


def newest_checkpoint(tgt: str) -> str | None:
    """The checkpoint directory with the newest meta.json by modification time (ties by name).

    Not a name sort: a trimmed checkpoint is named `guarded_g_<name>`, which sorts after every
    `auto_grown_<timestamp>` and would otherwise be resumed from forever (observed in production:
    one segment regrew from the same stale checkpoint 89 times)."""
    if not os.path.isdir(tgt):
        return None
    best, best_key = None, None
    for d in os.listdir(tgt):
        p = os.path.join(tgt, d)
        try:
            key = (os.path.getmtime(os.path.join(p, "meta.json")), d)
        except OSError:
            continue
        if best_key is None or key > best_key:
            best, best_key = p, key
    return best


def read_meta(ckpt: str | None) -> dict:
    if not ckpt:
        return {}
    try:
        return json.loads((Path(ckpt) / "meta.json").read_text())
    except (OSError, ValueError):
        return {}


def md5(path: str) -> str | None:
    try:
        h = hashlib.md5()
        with open(path, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()
    except OSError:
        return None


def git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=10, check=False)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def resolve_tool(name: str, vc_bin: str | None) -> str:
    p = os.path.join(vc_bin, name) if vc_bin else shutil.which(name)
    if not p or not os.path.exists(p):
        raise FileNotFoundError(f"{name} not found (vc_bin={vc_bin!r}, PATH)")
    return p


def voxel_um_of(volume: str, given: float | None) -> float:
    """--voxel-um, else the VC3D volume's own meta.json `voxelsize`. Never a hard-coded default:
    area (cm2) scales with its square."""
    if given:
        return float(given)
    m = read_meta(volume)
    if m.get("voxelsize"):
        return float(m["voxelsize"])
    raise ValueError(f"no voxelsize in {volume}/meta.json; pass --voxel-um")


def run_round(a, rnd: int, resume: str | None, gen_to: int, voxel_um: float, grow_bin: str) -> dict:
    params = {"mode": "seed", "generations": gen_to, "min_area_cm": 0.002, "step_size": a.step_size,
              "thread_limit": a.thread_limit, "voxelsize": voxel_um, "normal_grid_path": a.normal_grids,
              "snapshot-interval": 1 if resume else 5}
    params.update(getattr(a, "preset_tracer_params", None) or {})     # a preset's own tracer params (E, F)
    if a.params:
        params.update(json.loads(Path(a.params).read_text()))
    pj = os.path.join(a.out, f"params_r{rnd}.json")
    Path(pj).write_text(json.dumps(params, indent=1))
    tgt = os.path.join(a.out, f"r{rnd}")
    cmd = [grow_bin, "-v", a.volume, "-t", tgt, "--params", pj]
    cmd += ["--resume", resume] if resume else ["--seed", *[str(int(v)) for v in a.seed]]
    env = dict(os.environ, OMP_NUM_THREADS=str(a.thread_limit), OPENBLAS_NUM_THREADS="1")
    if a.rng_seed is not None:
        env["VC_GROWPATCH_RNG_SEED"] = str(a.rng_seed)
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.time()
    with open(os.path.join(a.out, f"round{rnd}.log"), "ab") as lg:
        lg.write((" ".join(cmd) + "\n").encode())
        try:
            rc = subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT, env=env,
                                timeout=a.round_timeout_s, check=False).returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            rc, timed_out = -9, True
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    ck = newest_checkpoint(tgt)
    return {"round": rnd, "rc": rc, "timed_out": timed_out, "checkpoint": ck, "cpu_s": round(cpu, 3),
            "wall_s": round(time.time() - t0, 3), "gen_to": gen_to, "area_cm2": float(read_meta(ck).get("area_cm2") or 0.0)}


def build_context(a):
    sampler = GG.ZarrSampler(a.ct or a.volume, a.ct_level) if (a.ct or a.volume) else None
    pred = GG.ZarrSampler(a.pred, 0) if a.pred else None
    umb = None
    if a.umbilicus:
        import numpy as np
        d = json.loads(Path(a.umbilicus).read_text())
        pts = sorted(d.get("control_points", d) if isinstance(d, dict) else d, key=lambda p: p["z"])
        zs = np.array([p["z"] for p in pts], float)
        xs = np.array([p["x"] for p in pts], float)
        ys = np.array([p["y"] for p in pts], float)

        def umb(z):
            z = np.clip(z, zs[0], zs[-1])
            return np.interp(z, zs, xs), np.interp(z, zs, ys)
    ctx = GG.ShadowContext(pred_sampler=pred, normal_sampler=None, umbilicus_of_z=umb, step_size=a.step_size)
    return sampler, ctx


def grow(a) -> dict:
    os.makedirs(a.out, exist_ok=True)
    grow_bin = resolve_tool("vc_grow_seg_from_seed", a.vc_bin)
    pol = GG.load_policy(a.policy) if a.policy else GG.GuardPolicy()
    a.preset_tracer_params = json.loads(Path(a.policy).read_text()).get("tracer_params") if a.policy else None
    if a.set:
        over = {}
        for kv in a.set:
            k, _, v = kv.partition("=")
            over[k] = v
        base = {f: getattr(pol, f) for f in pol.__dataclass_fields__}
        base.update(over)
        pol = GG.policy_from_dict(base)
    if pol.enabled and pol.selfcross:
        from dataclasses import replace
        pol = replace(pol, selfcross_bin=resolve_tool("vc_tifxyz_selfcross", a.vc_bin), selfcross_env=dict(os.environ))
    voxel_um = voxel_um_of(a.volume, a.voxel_um)
    if pol.enabled:
        ok, detail = GG.self_test(pol.selfcross_bin or None, selfcross_required=pol.selfcross, env=pol.selfcross_env)
        if not ok:
            raise GG.GuardBroken(f"self-test failed: {detail}")
    sampler, ctx = build_context(a) if pol.enabled else (None, None)
    state = GG.GuardState()
    report = open(os.path.join(a.out, "guard_report.jsonl"), "a")
    cur, gen, area = a.resume, int(read_meta(a.resume).get("max_gen") or 0), float(read_meta(a.resume).get("area_cm2") or 0.0)
    t0, rounds, cpu, guard_s = time.time(), 0, 0.0, 0.0
    status, why = "done", "max_rounds"
    while rounds < a.max_rounds:
        if time.time() - t0 >= a.wall_s:
            why = "wall_ceiling"
            break
        rounds += 1
        gen_to = (gen + a.gen_step) if cur else a.first_gens
        rr = run_round(a, rounds, cur, gen_to, voxel_um, grow_bin)
        cpu += rr["cpu_s"]
        line = dict(rr)
        if rr["checkpoint"] is None or (rr["rc"] != 0 and rr["area_cm2"] <= area):
            status, why = "failed", ("timeout" if rr["timed_out"] else f"exit_{rr['rc']}")
            report.write(json.dumps(line) + "\n")
            break
        gained = rr["area_cm2"] - area
        cur, gen, area = rr["checkpoint"], gen_to, rr["area_cm2"]
        stop = None
        if pol.enabled:
            g0 = time.time()
            cur, ginfo = GG.guard_round(cur, pol, voxel_um, state, sampler=sampler, shadow_ctx=ctx)
            guard_s += time.time() - g0
            line["guard_s"] = round(time.time() - g0, 3)
            line["guard"] = ginfo
            area = float(read_meta(cur).get("area_cm2") or 0.0)
            stop = ginfo.get("stop")
        line["area_after_guard_cm2"] = area
        report.write(json.dumps(line, default=str) + "\n")
        report.flush()
        if stop == "nothing_left":
            status, why, area = "failed", "guard_nothing_left", 0.0
            break
        if stop:
            why = f"guard_{stop}"
            break
        if rounds > 1 and gained * 100.0 < a.exhausted_mm2:
            why = f"exhausted: gained {gained * 100:.1f} mm2"
            break
    report.close()
    res = {"status": status, "stop_reason": why, "checkpoint": cur, "area_cm2": area, "rounds": rounds,
           "tracer_cpu_s": round(cpu, 3), "guard_s": round(guard_s, 3), "wall_s": round(time.time() - t0, 3),
           "voxel_um": voxel_um, "seed": a.seed, "resume": a.resume,
           "tracer_params_extra": {**(a.preset_tracer_params or {}), **(json.loads(Path(a.params).read_text()) if a.params else {})},
           "policy": {f: getattr(pol, f) for f in pol.__dataclass_fields__ if f != "selfcross_env"},
           "provenance": {"vc_grow_seg_from_seed": grow_bin, "vc_grow_seg_from_seed_md5": md5(grow_bin),
                          "vc_tifxyz_selfcross_md5": md5(pol.selfcross_bin) if pol.selfcross_bin else None,
                          "tifxyz_tools_git": git_commit(), "host": os.uname().nodename,
                          "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}
    Path(a.out, "result.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="guarded-grow", description=__doc__.split("\n")[0])
    ap.add_argument("--volume", required=True, help="the -v volume passed to the tracer (OME-Zarr)")
    ap.add_argument("--normal-grids", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--seed", nargs=3, type=float, metavar=("X", "Y", "Z"))
    g.add_argument("--resume")
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", help="preset JSON (presets/*.json); default: guard disabled")
    ap.add_argument("--set", action="append", metavar="FIELD=VALUE", help="override one GuardPolicy field")
    ap.add_argument("--params", help="extra tracer params JSON merged into each round's params")
    ap.add_argument("--ct", help="CT OME-Zarr for the vacuum criterion (default: --volume; pass the CT "
                                 "explicitly when --volume is a surface prediction)")
    ap.add_argument("--ct-level", type=int, default=1)
    ap.add_argument("--pred", help="surface-prediction OME-Zarr: enables ridge_hit / seam / empty_space")
    ap.add_argument("--umbilicus", help="umbilicus JSON ({x,y,z} points): enables wrap_spacing / curvature")
    ap.add_argument("--vc-bin", help="directory holding vc_grow_seg_from_seed and vc_tifxyz_selfcross")
    ap.add_argument("--voxel-um", type=float)
    ap.add_argument("--step-size", type=float, default=20.0)
    ap.add_argument("--thread-limit", type=int, default=1)
    ap.add_argument("--rng-seed", type=int)
    ap.add_argument("--first-gens", type=int, default=10)
    ap.add_argument("--gen-step", type=int, default=10)
    ap.add_argument("--max-rounds", type=int, default=50)
    ap.add_argument("--wall-s", type=float, default=1800.0)
    ap.add_argument("--round-timeout-s", type=float, default=3600.0)
    ap.add_argument("--exhausted-mm2", type=float, default=EXHAUSTED_MM2,
                    help="stop when a resumed round adds less than this (mm2)")
    a = ap.parse_args(argv)
    res = grow(a)
    print(json.dumps({k: res[k] for k in ("status", "stop_reason", "area_cm2", "rounds", "tracer_cpu_s", "guard_s")}))
    return 0 if res["status"] == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
