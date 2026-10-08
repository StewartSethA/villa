"""Fetch-manifest builder: which upstream assets one (scroll, z-window) needs.  Output feeds deploy_common/fetch_assets.py."""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

from .common import spec

BUCKET = "vesuvius-challenge-open-data"
FETCH_WORKERS = int(os.environ.get("ROUTEB_FETCH_WORKERS", "128"))   # bench on pny (n = 1 run per setting): 32 workers 37 files/s, 128 workers 58 files/s, 256 workers 49 files/s


PINS = Path(__file__).resolve().parent / "scrolls" / "pins.json"       # md5/sha256 of upstream files we measured, keyed by file name


def pinned() -> dict:
    return json.loads(PINS.read_text()) if PINS.exists() else {}


def lasagna_z_chunks(z0: int, z1: int, scale: int = 4, chunk: int = 32, margin_vox: int = 256) -> list[int]:
    lo = max(0, (z0 - margin_vox) // scale // chunk)
    hi = (z1 + margin_vox) // scale // chunk
    return list(range(lo, hi + 1))


def lasagna_z_include(z0: int, z1: int, scale: int = 4, chunk: int = 32, margin_vox: int = 256) -> str:
    """Regex over keys relative to a field's zarr root: metadata + only the group-2 z-chunks covering [z0, z1) (L0 voxels) +- margin."""
    lo = max(0, (z0 - margin_vox) // scale // chunk)
    hi = (z1 + margin_vox) // scale // chunk
    zs = "|".join(str(i) for i in range(lo, hi + 1))
    return rf"^(\.zattrs|\.zgroup|2/\.zarray|2/\.zattrs|2/({zs})/)"


def build(scroll: str, z0: int, z1: int, with_crossings: bool = False, full_lasagna: bool = False) -> dict:
    s = spec(scroll)
    pins = pinned()
    a = []
    tr = s.get("tracks")
    if not tr:
        raise ValueError(f"{scroll}: no upstream tracks published (extract them first: README 'scrolls without published tracks')")
    for name, size in tr["files"].items():
        if name.endswith(".crossings.npz") and not with_crossings:
            continue
        pn = pins.get(name, {})
        a.append({"name": f"{scroll}:tracks:{name.split(scroll + '_' + tr['ts'] + '_')[-1]}", "kind": "http", "url": tr["base_url"] + name,
                  "dest": f"{scroll}/dataset/tracks/{name}", "size": size, **{k: v for k, v in pn.items() if k in ("md5", "sha256")}})
    las = s.get("lasagna")
    if not las:
        raise ValueError(f"{scroll}: no upstream lasagna fields (nx/ny) published")
    inc = None if full_lasagna else lasagna_z_include(z0, z1)
    for fld, dirname in (("nx", "las_008_nx"), ("ny", "las_008_ny"), ("grad_mag", "las_008_grad_mag")):
        pref = las["fields"].get(fld)
        if not pref:
            continue
        a.append({"name": f"{scroll}:lasagna:{fld}", "kind": "s3prefix", "bucket": BUCKET, "prefix": pref,
                  "dest": f"{scroll}/dataset/lasagna_inputs/{dirname}.ome.zarr", "workers": FETCH_WORKERS,
                  **({"include": inc, "subprefixes": [".zattrs", ".zgroup", "2/.zarray", "2/.zattrs", *[f"2/{c}/" for c in lasagna_z_chunks(z0, z1)]]} if inc else {"include": r"^(\.zattrs|\.zgroup|2/)"})})
    return {"scroll": scroll, "z": [z0, z1], "assets": a}


def volume_chunks_asset(scroll: str, keys: list[str], name: str) -> dict:
    """s3 volume chunks (level 0) needed by one tile set; `keys` are 'z/y/x' chunk keys."""
    s = spec(scroll)
    return {"scroll": scroll, "assets": [{"name": name, "kind": "s3keys", "bucket": BUCKET, "prefix": s["volume_s3_prefix"] + "/0",
                                           "dest": f"{scroll}/volume/{s['volume_zarr']}/0", "keys": keys, "workers": FETCH_WORKERS}]}
