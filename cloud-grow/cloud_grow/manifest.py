"""Export manifest (`<seg>.export.json`, schema vpipe-remote-grow-export/1) + md5 tree, written as NEW SIDECARS (D10).

Never modifies a meta.json / result.json / manifest. The manifest is assembled from `rounds.jsonl` (one line per
round, written by the runner), `run_context.json` (written once at run start) and the seed row of the local state DB.
Fields follow docs/experiments/vast_benchmark_2026-10-06/manifest_spec.json (copy: schema/export_manifest_spec.json).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path

SCHEMA = "vpipe-remote-grow-export/1"
EXCLUDE_DIRS = {"cache_root"}
EXCLUDE_SUFFIX = (".provenance.json",)
MANIFEST_NAME = "{seg}.export.json"
TREE_NAME = "{seg}.md5tree"


def md5_file(p, chunk=1 << 20) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_file(p, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def shippable_files(seg_dir: str, seg: str) -> list[str]:
    """Relative paths of every file that ships: everything except cache_root/, *.provenance.json, the manifest and its md5 tree."""
    skip = {MANIFEST_NAME.format(seg=seg), TREE_NAME.format(seg=seg)}
    out = []
    for r, dirs, files in os.walk(seg_dir):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDE_DIRS)
        for f in sorted(files):
            rel = os.path.relpath(os.path.join(r, f), seg_dir)
            if f in skip and r == seg_dir:
                continue
            if f.endswith(EXCLUDE_SUFFIX) or os.path.islink(os.path.join(r, f)):
                continue
            out.append(rel)
    return sorted(out)


def md5_tree(seg_dir: str, seg: str) -> str:
    """`md5  relpath` lines (the maintenance.py convention)."""
    return "".join(f"{md5_file(os.path.join(seg_dir, rel))}  {rel}\n" for rel in shippable_files(seg_dir, seg))


def cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo") as fh:
            for ln in fh:
                if ln.startswith("model name"):
                    return ln.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def ram_gb() -> float | None:
    try:
        with open("/proc/meminfo") as fh:
            return round(int(fh.readline().split()[1]) / 1048576.0, 1)
    except (OSError, ValueError):
        return None


def git_release(repo: str | None = None) -> dict:
    repo = repo or str(Path(__file__).resolve().parents[2])
    try:
        sha = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", repo, "status", "--porcelain", "--", "cloud-grow"], capture_output=True,
                                    text=True, timeout=20).stdout.strip())
        return {"release_sha": sha or None, "git_dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"release_sha": None, "git_dirty": None}


def kit_libs_md5(bin_dir: str) -> str | None:
    lib = os.path.join(os.path.dirname(os.path.abspath(bin_dir)), "lib")
    if not os.path.isdir(lib):
        return None
    h = hashlib.md5()
    for f in sorted(os.listdir(lib)):
        p = os.path.join(lib, f)
        if os.path.isfile(p):
            h.update(f.encode() + md5_file(p).encode())
    return h.hexdigest()


def store_identity(path: str | None) -> dict:
    """sha256 of the store's METADATA files (not its bytes), like provenance.store_identity of the fleet."""
    if not path:
        return {"path": None, "sha256": None}
    h, n = hashlib.sha256(), 0
    for name in (".zattrs", ".zarray", ".zgroup", "zarr.json", "meta.json", "metadata.json"):
        q = Path(path) / name
        if q.is_file():
            h.update(name.encode() + b"\0" + q.read_bytes())
            n += 1
    return {"path": os.path.basename(str(path).rstrip("/")), "sha256": h.hexdigest() if n else None}


def write_run_context(out_root: str, cfg, ident: dict, bin_dir: str, pin_status: str, policy_settings: dict, extra_inputs: dict | None = None,
                      not_ported=()) -> str:
    """Once per run: what the box was and what it ran. Written beside the segments, never into them."""
    rel = git_release()
    ctx = {
        "schema": "vpipe-cloud-grow-run-context/1", "t_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scroll": cfg.scroll, "instance_id": cfg.instance_id or None,
        "host": {"hostname": socket.gethostname(), "cpu_model": cpu_model(), "threads": os.cpu_count(), "ram_gb": ram_gb(),
                 "python": platform.python_version(), "platform": platform.platform()},
        "release_sha": cfg.release_sha or rel["release_sha"], "git_dirty": rel["git_dirty"],
        "tool": {"identification": ident, "pin_status": pin_status, "kit_libs_md5": kit_libs_md5(bin_dir)},
        "inputs": {"voxel_um": cfg.voxel_um, "ct_level_for_guard": cfg.ct_level_guard,
                   "tracer_volume": cfg.tracer_volume,
                   "prediction": store_identity(cfg.prediction_zarr), "ct": store_identity(cfg.ct_zarr),
                   "normal_grids": store_identity(cfg.normal_grids),
                   "umbilicus_md5": md5_file(cfg.umbilicus_json) if cfg.umbilicus_json and os.path.isfile(cfg.umbilicus_json) else None,
                   **(extra_inputs or {})},
        "run": {"thread_limit": cfg.thread_limit, "rng_seed": cfg.rng_seed, "step_size": cfg.step_size,
                "env": {"VC_GRID_CACHE_BYTES": str(cfg.grid_cache_bytes), "VC_GROWPATCH_RNG_SEED": str(cfg.rng_seed),
                        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": str(cfg.thread_limit)},
                "spaceline_always": cfg.spaceline_always},
        "policy": {"settings": policy_settings, "sha256": hashlib.sha256(json.dumps(policy_settings, sort_keys=True).encode()).hexdigest()},
        "not_ported": list(not_ported),
        "validation_status": "UNVALIDATED against human annotation (D6)",
    }
    os.makedirs(out_root, exist_ok=True)
    p = os.path.join(out_root, "run_context.json")
    tmp = p + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(ctx, fh, indent=1)
    os.replace(tmp, p)
    return p


def read_rounds(seg_dir: str) -> list[dict]:
    p = os.path.join(seg_dir, "rounds.jsonl")
    if not os.path.exists(p):
        return []
    with open(p) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def build_manifest(seg_dir: str, seg: str, run_context: dict, seed: dict | None = None) -> dict:
    """Write `<seg>.md5tree` then `<seg>.export.json` into seg_dir; returns the manifest dict."""
    rounds = read_rounds(seg_dir)
    tree_text = md5_tree(seg_dir, seg)
    tree_path = os.path.join(seg_dir, TREE_NAME.format(seg=seg))
    with open(tree_path, "w") as fh:
        fh.write(tree_text)
    ident = (run_context.get("tool") or {}).get("identification") or {}
    tr = ident.get("vc_grow_seg_from_seed") or {}
    man = {
        "schema": SCHEMA, "t_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "identity": {"seg": seg, "scroll": run_context.get("scroll"), "seed": seed, "seed_provenance": (seed or {}).get("provenance"),
                     "rng_seed": (run_context.get("run") or {}).get("rng_seed")},
        "inputs": run_context.get("inputs"),
        "tool": {"path": tr.get("path"), "md5": tr.get("md5"), "file_type": tr.get("file"), "pin_status": (run_context.get("tool") or {}).get("pin_status"),
                 "identification": ident, "kit_libs_md5": (run_context.get("tool") or {}).get("kit_libs_md5"),
                 "release_sha": run_context.get("release_sha"), "git_dirty": run_context.get("git_dirty")},
        "run": {**(run_context.get("host") or {}), "instance_id": run_context.get("instance_id"), **(run_context.get("run") or {})},
        "policy": run_context.get("policy"),
        "not_ported": run_context.get("not_ported"),
        "rounds": rounds,
        "ship": {"files": shippable_files(seg_dir, seg), "md5tree_file": TREE_NAME.format(seg=seg),
                 "md5tree_sha256": hashlib.sha256(tree_text.encode()).hexdigest()},
        "claims": {"validation_status": "UNVALIDATED against human annotation (D6)",
                   "verified_cm2_meaning": "the BOX'S OWN claim, scored at CT level %s; the hub re-scores on import" % (run_context.get("inputs") or {}).get("ct_level_for_guard")},
    }
    last = rounds[-1] if rounds else {}
    man["final"] = {"checkpoint": last.get("checkpoint"), "area_cm2_postguard": last.get("area_cm2_postguard"),
                    "guard_stop": last.get("guard_stop")}
    p = os.path.join(seg_dir, MANIFEST_NAME.format(seg=seg))
    tmp = p + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(man, fh, indent=1, default=str)
    os.replace(tmp, p)
    return man
