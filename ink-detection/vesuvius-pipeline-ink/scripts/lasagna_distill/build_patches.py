#!/usr/bin/env python3
"""Patch builder for Lasagna distillation: CT volume (input) + published nx/ny/grad_mag/cos (targets).

All five arrays are read at pyramid level LEVEL (default 2, 1/4 resolution; the volume and the published fields share
that grid exactly: verified PHerc0125 volume/2 == nx/2 == cos/2 == 5210x2097x2097).  Volume chunks are 128^3 raw, field
chunks 32^3 blosc.  A missing chunk file is an all-zero chunk (zarr fill_value 0 = "unset" in the published encoding).

Measured (5000 box, 2026-10-08): per-patch zarr/s3fs reads do not scale with threads (32 workers: 0.13 patches/s);
`rclone copy --files-from` of chunk files moves ~550 requests/s (4096 requests, 7.5 s), so chunks are fetched in bulk and
assembled locally.  Per scroll, in batches of --batch cubes:
  A. fetch only the 64 nx chunks of each candidate cube, keep cubes with >= MIN_FILL non-zero nx voxels
  B. fetch ny/grad_mag/cos chunks + the volume chunk of the kept cubes, assemble, write, delete the downloaded chunks
Output per scroll: x.npy (N,P,P,P) u8 CT; y.npy (N,4,P,P,P) u8 nx,ny,grad_mag,cos (published encoding, untouched);
meta.json (origins z,y,x at LEVEL, counts, kept fraction, seed, done flag).  Resumable per scroll (done flag).
  python build_patches.py --out DIR --list
  python build_patches.py --out DIR --per-scroll 450 --scrolls all
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

BUCKET = "vesuvius-challenge-open-data"
CHANS = ("nx", "ny", "grad_mag", "cos")
RCLONE = ["rclone", "--s3-provider", "AWS", "--s3-region", "us-east-1", "--s3-env-auth=false"]
FC = 32      # field chunk edge
VC = 128     # volume chunk edge (patch size is fixed to it)


def say(m):
    print(f"[{time.strftime('%H:%M:%S')}] build_patches: {m}", flush=True)


def rc(*a):
    return subprocess.run(RCLONE + list(a), capture_output=True, text=True, timeout=600).stdout


def discover(cache):
    if os.path.exists(cache):
        return {k: tuple(v) for k, v in json.load(open(cache)).items()}
    out = {}
    for s in rc("lsf", f":s3:{BUCKET}/").split():
        if not s.startswith("PHerc"):
            continue
        s = s.rstrip("/")
        runs = [r.rstrip("/") for r in rc("lsf", f":s3:{BUCKET}/{s}/representations/predictions/lasagna/").split()]
        if not runs:
            continue
        vols = [v.rstrip("/") for v in rc("lsf", f":s3:{BUCKET}/{s}/volumes/").split() if v.rstrip("/").endswith(".zarr")]
        pref = [v for v in vols if v.split("-")[0] == runs[-1].split("-")[0]] or [v for v in vols if "masked" in v] or vols
        if pref:
            out[s] = (runs[-1], pref[0])
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    json.dump(out, open(cache, "w"))
    return out


def fetch(remote, files, dest):
    """Bulk-fetch chunk files (paths relative to remote) into dest/c. Missing files are all-zero chunks."""
    os.makedirs(dest, exist_ok=True)
    lst = os.path.join(dest, "list.txt")
    with open(lst, "w") as f:
        f.write("\n".join(files) + "\n")
    subprocess.run(RCLONE + ["copy", "--files-from", lst, remote, os.path.join(dest, "c"), "--transfers", "64",
                             "--checkers", "64", "--no-traverse", "--retries", "5", "--low-level-retries", "20"],
                   capture_output=True, text=True, timeout=3600)


def field_files(prefix, c, o, level):
    z0, y0, x0 = (v // FC for v in o)
    n = VC // FC
    return [f"{prefix}_{c}.ome.zarr/{level}/{a}/{b}/{d}" for a in range(z0, z0 + n) for b in range(y0, y0 + n)
            for d in range(x0, x0 + n)]


def read_field(root, prefix, c, o, level, codec):
    out = np.zeros((VC, VC, VC), np.uint8)
    z0, y0, x0 = (v // FC for v in o)
    n = VC // FC
    for a in range(n):
        for b in range(n):
            for d in range(n):
                f = os.path.join(root, f"{prefix}_{c}.ome.zarr/{level}/{z0 + a}/{y0 + b}/{x0 + d}")
                if os.path.exists(f):
                    ch = np.frombuffer(codec.decode(open(f, "rb").read()), np.uint8).reshape(FC, FC, FC)
                    out[a * FC:(a + 1) * FC, b * FC:(b + 1) * FC, d * FC:(d + 1) * FC] = ch
    return out


def build_scroll(scroll, run, vol, a):
    from numcodecs import Blosc
    codec = Blosc()
    d = os.path.join(a.out, scroll)
    mp = os.path.join(d, "meta.json")
    if os.path.exists(mp) and json.load(open(mp)).get("done"):
        say(f"{scroll}: done, skipping")
        return
    os.makedirs(d, exist_ok=True)
    P = VC
    lroot = f":s3:{BUCKET}/{scroll}/representations/predictions/lasagna/{run}"
    shp = json.loads(rc("cat", f"{lroot}/{scroll}_nx.ome.zarr/{a.level}/.zarray"))["shape"]
    vshp = json.loads(rc("cat", f":s3:{BUCKET}/{scroll}/volumes/{vol}/{a.level}/.zarray"))["shape"]
    if list(shp) != list(vshp):
        say(f"{scroll}: GRID MISMATCH volume {vshp} vs field {shp}: skipped")
        return
    rng = np.random.default_rng(a.seed + sum(map(ord, scroll)))
    grid = [s // P for s in shp]
    xs = np.lib.format.open_memmap(os.path.join(d, "x.npy"), "w+", np.uint8, (a.per_scroll, P, P, P))
    yy = np.lib.format.open_memmap(os.path.join(d, "y.npy"), "w+", np.uint8, (a.per_scroll, len(CHANS), P, P, P))
    origins, tried, seen = [], 0, set()
    t0 = time.time()
    while len(origins) < a.per_scroll and tried < a.per_scroll * 12:
        cand = []
        while len(cand) < a.batch:
            o = tuple(int(rng.integers(0, g)) * P for g in grid)
            if o not in seen:
                seen.add(o)
                cand.append(o)
        tried += len(cand)
        tmp = tempfile.mkdtemp(prefix="lasagna_bld_", dir=a.tmp)
        try:
            fetch(lroot, [f for o in cand for f in field_files(scroll, "nx", o, a.level)], tmp)
            root = os.path.join(tmp, "c")
            nx = {o: read_field(root, scroll, "nx", o, a.level, codec) for o in cand}
            keep = [o for o in cand if (nx[o] > 0).mean() >= a.min_fill][: a.per_scroll - len(origins)]
            if keep:
                fetch(lroot, [f for o in keep for c in CHANS[1:] for f in field_files(scroll, c, o, a.level)], tmp)
                fetch(f":s3:{BUCKET}/{scroll}/volumes",
                      [f"{vol}/{a.level}/{o[0] // VC}/{o[1] // VC}/{o[2] // VC}" for o in keep], os.path.join(tmp, "v"))
                for o in keep:
                    k = len(origins)
                    vf = os.path.join(tmp, "v", "c", vol, str(a.level), str(o[0] // VC), str(o[1] // VC), str(o[2] // VC))
                    xs[k] = np.fromfile(vf, np.uint8).reshape(P, P, P) if os.path.exists(vf) else 0
                    yy[k, 0] = nx[o]
                    for ci, c in enumerate(CHANS[1:], 1):
                        yy[k, ci] = read_field(root, scroll, c, o, a.level, codec)
                    origins.append(list(o))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        say(f"{scroll}: {len(origins)}/{a.per_scroll} kept, {tried} tried, {time.time() - t0:.0f} s")
    n = len(origins)
    xs.flush()
    yy.flush()
    json.dump({"scroll": scroll, "run": run, "volume": vol, "level": a.level, "patch": P, "shape": list(shp), "n": n,
               "tried": tried, "kept_frac": round(n / max(1, tried), 3), "min_fill": a.min_fill, "seed": a.seed,
               "origins": origins, "channels": list(CHANS), "secs": round(time.time() - t0), "done": n > 0},
              open(mp, "w"))
    say(f"{scroll}: DONE {n} patches of {P}^3 ({n * P ** 3 * 5 / 1e9:.1f} GB), kept {n}/{tried}, {time.time() - t0:.0f} s, "
        f"{n / max(1, time.time() - t0):.2f} patches/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scrolls", default="all")
    ap.add_argument("--per-scroll", type=int, default=450)
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--min-fill", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tmp", default=None, help="scratch for downloaded chunks (deleted per batch); default system tmp")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    found = discover(os.path.join(a.out, "discover.json"))
    if a.list:
        for s, (r, v) in sorted(found.items()):
            print(s, r, v)
        return 0
    for s in (sorted(found) if a.scrolls == "all" else a.scrolls.split(",")):
        try:
            build_scroll(s, *found[s], a)
        except Exception as e:  # noqa: BLE001 - one bad scroll must not cancel the rest
            say(f"{s}: FAILED {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
