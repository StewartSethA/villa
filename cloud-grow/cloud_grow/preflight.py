"""Preflight: can THIS box run the grow, and what should it cost? Exit 0 = every hard check passed.

Checks (each prints PASS/FAIL/WARN with the measured value): CPU cores, RAM (>= max(64 GB, 3 GB x concurrent grows + 40 GB),
DEPLOY.md), disk free vs inputs + scratch, tool md5 pins + `file -L`, CT level availability (guard needs level 1, seeding level 4),
prediction + grids present, normal-grid / prediction `meta.json`, python deps, and the expected throughput/cost from the vast
benchmark model (EXTRAPOLATED; needs --passmark-st). Nothing is installed or downloaded.
"""
from __future__ import annotations

import importlib
import json
import os
import shutil
import sys

from . import cost as COST
from . import tools as T


def _row(res, name, status, detail):
    res.append({"check": name, "status": status, "detail": detail})


def run(cfg: dict, grows: int | None = None, passmark_st: float | None = None, usd_per_hour: float | None = None,
        target_cm2: float = 100.0, cores: int | None = None, ram_gb: float | None = None, free_gb: float | None = None,
        pins: dict | None = None, need_inputs_gb: float | None = None, allow_unpinned: bool = False) -> dict:
    res: list = []
    phys = cores if cores is not None else (os.cpu_count() or 1)
    grows = grows or max(1, phys)
    _row(res, "cpu", "PASS" if phys >= 1 else "FAIL", f"{phys} logical cpus visible (physical cores may be half with SMT); planned concurrent grows = {grows}")
    if ram_gb is None:
        from .manifest import ram_gb as _r
        ram_gb = _r()
    need_ram = COST.ram_needed_gb(grows)
    ok = ram_gb is not None and ram_gb >= need_ram
    _row(res, "ram", "PASS" if ok else "FAIL",
         f"{ram_gb} GB present, {need_ram:.0f} GB needed (max(64, {COST.GB_PER_GROW:g} x {grows} + {COST.RAM_BASE_GB:g})); with VC_GRID_CACHE_BYTES=256 MB per-tool RSS p50 215 MB / max 540 MB (benchmark workload, resume rounds untested)")
    wd = cfg.get("workdir") or "."
    probe = wd if os.path.isdir(wd) else (os.path.dirname(os.path.abspath(wd)) or ".")
    free = free_gb if free_gb is not None else shutil.disk_usage(probe).free / 2**30
    inputs = need_inputs_gb
    if inputs is None:
        inputs = 0.0
        for k in ("ct_zarr", "prediction_zarr", "normal_grids"):
            p = cfg.get(k)
            if p and os.path.isdir(p):
                inputs += sum(os.path.getsize(os.path.join(r, f)) for r, _d, fs in os.walk(p) for f in fs) / 2**30
    need_disk = 20.0
    _row(res, "disk", "PASS" if free >= need_disk else "FAIL", f"{free:.1f} GB free at {probe}; >= {need_disk:g} GB scratch (cache_root) required; inputs already on disk {inputs:.1f} GB")
    kit = cfg.get("kit_bin") or os.environ.get("VC_BIN", "")
    try:
        bd = T.kit_bin_dir(kit or None)
        ident = T.identify(bd, pins=pins)
        for n, v in ident.items():
            st = {"PINNED": "PASS", "MISSING": "FAIL", "MISMATCH": "FAIL" if not allow_unpinned else "WARN", "NO-PIN": "WARN"}[v["pin_status"]]
            _row(res, f"tool:{n}", st, f"md5 {v['md5']} file: {v['file']} pin: {v['pin_status']}")
        if os.path.isdir(os.path.join(os.path.dirname(os.path.abspath(bd)), "lib")):
            _row(res, "kit_lib", "PASS", "kit lib/ present (LD_LIBRARY_PATH will be set)")
        else:
            _row(res, "kit_lib", "WARN", "no <kit>/lib: the binaries must resolve their libraries from the system")
    except T.ToolError as e:
        _row(res, "tools", "FAIL", str(e))
    ct = cfg.get("ct_zarr")
    for lv, why in ((1, "guard CT sampler"), (4, "seeding support field")):
        okl = bool(ct) and os.path.isdir(os.path.join(ct, str(lv))) and (os.path.exists(os.path.join(ct, str(lv), ".zarray")) or os.path.exists(os.path.join(ct, str(lv), "zarr.json")))
        _row(res, f"ct_level_{lv}", "PASS" if okl else "FAIL", f"{why}: {'present' if okl else 'MISSING'} in {ct!r} (zarr level dir with .zarray)")
    pz, ng = cfg.get("prediction_zarr"), cfg.get("normal_grids")
    _row(res, "prediction", "PASS" if pz and os.path.isdir(os.path.join(pz, "0")) else "FAIL", f"{pz!r}")
    _row(res, "prediction_meta", "PASS" if pz and os.path.exists(os.path.join(pz, "meta.json")) else "WARN", "VC3D needs <zarr>/meta.json (fetch writes it)")
    _row(res, "normal_grids", "PASS" if ng and all(os.path.isdir(os.path.join(ng, a)) for a in ("xy", "xz", "yz")) else "FAIL", f"{ng!r} needs xy/ xz/ yz/")
    _row(res, "voxel_um", "PASS" if cfg.get("voxel_um") else "FAIL", f"{cfg.get('voxel_um')} (registry value; 0/None would corrupt every area)")
    umb = cfg.get("umbilicus_json")
    _row(res, "umbilicus", "PASS" if umb and os.path.isfile(umb) else "WARN", "absent: guard criteria wrap_spacing + curvature are SKIPPED (not scored clean)")
    for mod in ("numpy", "scipy", "tifffile", "zarr"):
        try:
            importlib.import_module(mod)
            _row(res, f"py:{mod}", "PASS", "importable")
        except ImportError:
            _row(res, f"py:{mod}", "FAIL", "not installed (pip install -r cloud-grow/requirements.txt)")
    est = None
    if passmark_st:
        est = COST.estimate(passmark_st, max(1, phys // 2) if cores is None else phys, target_cm2, usd_per_hour)
        _row(res, "estimate", "INFO", json.dumps(est))
    else:
        _row(res, "estimate", "INFO", "pass --passmark-st <single-thread score> --usd-per-hour <x> for expected time/cost (EXTRAPOLATED)")
    return {"ok": not any(r["status"] == "FAIL" for r in res), "checks": res, "estimate": est}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="preflight")
    ap.add_argument("--config", required=True)
    ap.add_argument("--grows", type=int)
    ap.add_argument("--passmark-st", type=float)
    ap.add_argument("--usd-per-hour", type=float)
    ap.add_argument("--target-cm2", type=float, default=100.0)
    ap.add_argument("--allow-unpinned", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    cfg = json.load(open(a.config))
    r = run(cfg, a.grows, a.passmark_st, a.usd_per_hour, a.target_cm2, allow_unpinned=a.allow_unpinned)
    if a.json:
        print(json.dumps(r, indent=1))
    else:
        for c in r["checks"]:
            print(f"{c['status']:5s} {c['check']:18s} {c['detail']}")
        print("PREFLIGHT", "OK" if r["ok"] else "FAILED")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
