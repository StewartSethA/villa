#!/usr/bin/env python3
"""CLI: verify and register remote-grow exports on the hub (see cloud_grow/importer.py for the trust model).

  import_remote_grow.py --registry DIR [--voxel-um 9.362] [--selfcross-bin PATH --kit-bin DIR] [--dry-run] EXPORT.tar.gz|DIR ...
Exit 0 only if every export PASSED (or was already imported); 1 if any was REFUSED; 2 on usage error.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cloud_grow import importer as I      # noqa: E402
from cloud_grow import tools as T          # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exports", nargs="+")
    ap.add_argument("--registry", required=True, help="append-only sidecar registry directory")
    ap.add_argument("--sqlite", default="", help="optional sqlite file; an INSERT-only table remote_grow_import is created")
    ap.add_argument("--voxel-um", type=float, default=None, help="hub registry voxel pitch (upstream wins over the box's)")
    ap.add_argument("--area-tol", type=float, default=0.01)
    ap.add_argument("--max-selfx-density", type=float, default=0.0)
    ap.add_argument("--kit-bin", default="", help="hub VC3D bin dir (for vc_tifxyz_selfcross + LD_LIBRARY_PATH)")
    ap.add_argument("--selfcross-bin", default="")
    ap.add_argument("--accept-unpinned", action="store_true")
    ap.add_argument("--skip-selfx", action="store_true")
    ap.add_argument("--ct-zarr", default="")
    ap.add_argument("--expect-sha256", default="")
    ap.add_argument("--dry-run", action="store_true", help="verify only; write nothing")
    a = ap.parse_args(argv)
    kit = a.kit_bin or os.environ.get("VC_BIN", "")
    opt = I.ImportOptions(voxel_um=a.voxel_um, area_tol=a.area_tol, max_selfx_density=a.max_selfx_density,
                          accept_unpinned=a.accept_unpinned, selfcross_bin=a.selfcross_bin or (os.path.join(kit, "vc_tifxyz_selfcross") if kit else ""),
                          selfcross_env=T.tool_env(kit) if kit else None, ct_zarr=a.ct_zarr, skip_selfx=a.skip_selfx,
                          dry_run=a.dry_run, expect_sha256=a.expect_sha256)
    rc = 0
    for e in a.exports:
        v = I.import_export(e, a.registry, opt, a.sqlite)
        print(json.dumps({"export": e, "seg": v.seg, "status": v.status, "reasons": v.reasons, "dry_run": a.dry_run,
                          "registered": v.registered, "checks": v.checks}, default=str))
        if v.status != "PASS":
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
