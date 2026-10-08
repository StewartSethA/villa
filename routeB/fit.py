"""Stage FIT: one spiral fit per z-window (fit_spiral.py), checkpoint-restart chain (the chain142.sh logic), reduced-resolution knobs.

Recipe provenance (all from our v100 production runs, docs/experiments/routeB_allscrolls + routeB_stripes):
  base overrides      = scripts/routeB/arm142.sh (patches/outer-shell/fibres/winding-inference OFF, grad-mag dense spacing, 24000 tracks/step)
  gradient magnitude  = EXTRA_OVERRIDES of the converged full-width fit f0125_res_flowonly (input_use_gradient_magnitude, loss_weight_dense_spacing 12)
  model_flow_voxel_resolution 32 = the knob that let a full-z (13,000-slice) fit converge on a 32 GB card (flow grid z-cells scale with the z span)
  shell sizing        = per-scroll shell_outer_winding_idx (spec); gap expander = shell+10, capacity = shell+14
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from .common import ROOT, StageError, env_python, home, is_done, mark_done, run, say, spec, tail

CKPT_STEP_PY = ("import sys,torch;ck=torch.load(sys.argv[1],map_location='cpu',weights_only=False,mmap=True);"
                "print(int(ck.get('completed_iterations',-1)))")


def overrides(sp: dict, z0: int, z1: int, steps: int, shell: int, extra: dict | None) -> dict:
    o = {"z_begin": z0, "z_end": z1, "input_disable_patches": True, "loss_weight_shell_outer": 0, "loss_weight_shell_patch_radius": 0,
         "dense_spacing_mode": "grad_mag", "loss_weight_dense_spacing": 12.0, "input_use_outer_shell": False, "input_use_fibers": False,
         "input_use_winding_inference": False, "input_use_gradient_magnitude": True,
         "model_gap_expander_num_windings": shell + 10, "model_gap_expander_capacity_windings": shell + 14,
         "shell_outer_winding_idx": shell, "sample_count_tracks_per_step": 24000, "optimizer_num_training_steps": steps}
    if z1 - z0 > 6000:
        o["model_flow_voxel_resolution"] = 32          # full-width convergence knob (see module docstring)
    if extra:
        o.update(extra)
    return o


def lasagna_scale(nx_dir: Path) -> int:
    za = json.loads((nx_dir / ".zattrs").read_text())
    for ds in za["multiscales"][0]["datasets"]:
        if str(ds["path"]) == "2":
            for t in ds.get("coordinateTransformations", []):
                if t.get("type") == "scale":
                    return int(round(float(t["scale"][-1])))
    raise StageError(f"fit: cannot read the group-2 scale from {nx_dir}/.zattrs")


def prepare_dataset(scroll: str, sp: dict, ds_base: Path, ds: Path, sense: str, umb: Path) -> Path:
    """Per-run dataset dir: symlinked tracks + lasagna inputs (shared across stripes), own umbilicus.json and spiral-scroll.json."""
    ds.mkdir(parents=True, exist_ok=True)
    for name in ("tracks", "lasagna_inputs"):
        src = ds_base / name
        if not src.exists():
            raise StageError(f"fit: missing {src} (did the fetch stage run?)")
        lk = ds / name
        if lk.is_symlink() or lk.exists():
            lk.unlink() if lk.is_symlink() else shutil.rmtree(lk)
        lk.symlink_to(src, target_is_directory=True)
    shutil.copy(umb, ds / "umbilicus.json")
    dbms = sorted((ds_base / "tracks").glob("*.dbm"))
    if not dbms:
        raise StageError(f"fit: no .dbm under {ds_base}/tracks")
    scl = lasagna_scale(ds_base / "lasagna_inputs" / "las_008_nx.ome.zarr")
    (ds / "spiral-scroll.json").write_text(json.dumps({
        "schema_version": 1, "name": scroll, "voxel_size_um": sp["voxel_um"], "spiral_outward_sense": sense,
        "normal_zarr_group": "2", "lasagna_scale": scl,
        "paths": {"tracks_dbm": f"tracks/{dbms[0].name}"},
        "_provenance": {"generated_by": "routeB/fit.py", "lasagna_scale": "read from the nx store's .zattrs (group 2), not from lasagna.json",
                        "spiral_outward_sense": f"{sense} ({sp.get('spiral_outward_sense_source')})"}}, indent=1))
    return ds


def ckpt_step(ckpt: Path) -> int:
    import subprocess
    r = subprocess.run([env_python(), "-c", CKPT_STEP_PY, str(ckpt)], capture_output=True, text=True)
    try:
        return int(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return 0


def run_fit(scroll: str, tag: str, z0: int, z1: int, steps: int, sense: str | None, shell: int | None, umbilicus: str | None,
            gpu: str, extra: dict | None, max_chunks: int = 40, workdir: Path | None = None) -> Path:
    sp = spec(scroll)
    H = home()
    sense = sense or sp.get("spiral_outward_sense")
    if sense not in ("CW", "ACW"):
        raise StageError(f"fit: spiral_outward_sense for {scroll} is unknown ({sp.get('spiral_outward_sense_source')}); pass --sense CW|ACW")
    if not sp.get("spiral_outward_sense"):
        say(f"WARNING {scroll}: sense {sense} given on the command line, registry has none", "fit")
    shell = int(shell or sp["shell_outer_winding_idx"])
    if "ASSUMED" in (sp.get("shell_source") or "") and not shell:
        say(f"WARNING {scroll}: shell is ASSUMED", "fit")
    umb = Path(umbilicus) if umbilicus else ROOT / (sp.get("umbilicus_file") or "")
    if not umb.is_file():
        raise StageError(f"fit: no umbilicus file for {scroll} ({umb}); pass --umbilicus")
    run_dir = (workdir or H / "runs" / scroll / tag) / "fit"
    if is_done(run_dir, "fit"):
        say(f"{scroll}/{tag}: fit already done", "fit")
        return run_dir
    ds = prepare_dataset(scroll, sp, H / "assets" / scroll / "dataset", run_dir.parent / "ds", sense, umb)
    sf = ROOT / "spiral-fitting"
    ov = overrides(sp, z0, z1, steps, shell, extra)
    out_dir = run_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "checkpoint_fitted.ckpt"
    log = run_dir / "fit.log"
    env = {"FIT_SPIRAL_OUT_DIR": out_dir.parent / "outroot", "FIT_SPIRAL_RUN_DIR": out_dir, "FIT_SPIRAL_CACHE_DIR": run_dir / "cache",
           "FIT_SPIRAL_NUM_THREADS": os.environ.get("FIT_SPIRAL_NUM_THREADS", str(min(8, os.cpu_count() or 4))),
           "OMP_NUM_THREADS": os.environ.get("FIT_SPIRAL_NUM_THREADS", str(min(8, os.cpu_count() or 4))),
           "FIT_SPIRAL_RUN_TAG": tag, "CUDA_VISIBLE_DEVICES": gpu, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
           "PYTHONUNBUFFERED": "1", "FIT_SPIRAL_CONFIG_OVERRIDES": json.dumps(ov),
           "TORCHINDUCTOR_CACHE_DIR": H / "compile_cache" / "inductor", "TRITON_CACHE_DIR": H / "compile_cache" / "triton",
           "FIT_SPIRAL_AUTOSAVE_INTERVAL": os.environ.get("FIT_SPIRAL_AUTOSAVE_INTERVAL", "1000")}
    for k in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "FIT_SPIRAL_CACHE_DIR"):
        Path(env[k]).mkdir(parents=True, exist_ok=True)
    (run_dir / "fit_config.json").write_text(json.dumps({"scroll": scroll, "tag": tag, "z": [z0, z1], "steps": steps, "sense": sense,
                                                         "shell": shell, "overrides": ov, "umbilicus": str(umb)}, indent=1))
    say(f"{scroll}/{tag}: fit z[{z0},{z1}) steps={steps} sense={sense} shell={shell} gpu={gpu}", "fit")
    say(f"   overrides: {json.dumps(ov)}", "fit")
    last = 0
    for chunk in range(1, max_chunks + 1):
        got = ckpt_step(ckpt) if ckpt.exists() else 0
        if got > 0:
            env["FIT_SPIRAL_RESUME_PATH"], env["FIT_SPIRAL_RESUME_STEP"] = ckpt, got
        if chunk > 1 and got <= last:
            raise StageError(f"fit: NO PROGRESS between chunks (checkpoint at step {got}); see {log}\n{tail(log)}")
        last = got
        if got >= steps:
            break
        say(f"   chunk {chunk} from step {got}", "fit")
        t0 = time.time()
        rc = run([env_python(), "fit_spiral.py", "--dataset", ds, "--cache", env["FIT_SPIRAL_CACHE_DIR"]], log=log, env=env, cwd=sf)
        now = ckpt_step(ckpt) if ckpt.exists() else 0
        say(f"   chunk {chunk} exited rc={rc} after {time.time() - t0:.0f} s, checkpoint at step {now}", "fit")
        if rc == 0:
            break
    met = list(out_dir.rglob("satisfaction_metrics_fitted.json"))
    meshes = [d for d in out_dir.rglob("meshes") if d.is_dir()]
    if not met or not meshes:
        raise StageError(f"fit: finished without satisfaction metrics/meshes under {out_dir}; log tail:\n{tail(log)}")
    m = json.loads(met[0].read_text())
    sat = m.get("satisfied_track_points_fraction") or m.get("satisfied_track_points")
    mark_done(run_dir, "fit", z=[z0, z1], steps=steps, satisfied_track_points=sat, metrics=str(met[0].relative_to(run_dir)),
              meshes=str(meshes[0].relative_to(run_dir)), sense=sense, shell=shell)
    say(f"{scroll}/{tag}: fit done, satisfied_track_points={sat}", "fit")
    return run_dir
