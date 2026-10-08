"""Route A: does a segment overlap another Route A segment of the same scroll?  (user 2026-10-08: "detect when a Route A segment begins to overlap an existing Route A segment")

Two seeds that grow onto the SAME sheet end up covering the same papyrus twice: the cm2 counter double-counts it and the hub would import duplicates.  Detection: the newest
checkpoint of each segment is a tifxyz grid (x/y/z.tif, voxel coordinates); a vertex of segment A is "shared" with segment B when B has a vertex within EPS_VOX of it.  The overlap
fraction of A (the smaller one) is the share of its vertices that are shared; overlap cm2 ~= fraction x A's area.

TUNABLES (named, with a basis; D33).  EPS_VOX = 5 voxels (~47 um at 9.4 um/voxel): under half the wrap pitch (~15 voxels measured in the wrap-stack study), so a neighbouring WRAP is not
called an overlap; STRIDE = 3 (every 3rd grid row and column: a speed/accuracy trade-off, the fraction is an estimate); MIN_FRAC = 0.02 (pairs below 2 % of the smaller segment are noise from
touching borders and are not reported).  Not validated against a human reference: it is a monitor, never a gate -- it does not stop or delete anything.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

EPS_VOX = 5.0
STRIDE = 3
MIN_FRAC = 0.02


def _round_no(d: Path) -> int:
    m = re.search(r"(\d+)", d.parent.name)
    return int(m.group(1)) if m else -1


def newest_checkpoint(sd: Path):
    cks = [p.parent for p in sd.glob("r*/*/x.tif") if all((p.parent / f).is_file() for f in ("y.tif", "z.tif", "meta.json"))]
    return max(cks, key=lambda d: (_round_no(d), (d / "meta.json").stat().st_mtime, d.name)) if cks else None


def load_grid(ck: Path):
    """(X, Y, Z float32 grids with NaN at invalid cells, area_cm2); None when unreadable."""
    import numpy as np
    from .. import growth_guard as GG
    try:
        X, Y, Z = GG._read_xyz(str(ck))
        V = (X > 0) & (Y > 0) & (Z > 0)
        G = [np.where(V, A, np.nan).astype("float32") for A in (X, Y, Z)]
        area = float(json.loads((ck / "meta.json").read_text()).get("area_cm2") or 0.0)
        return G, area
    except Exception:  # noqa: BLE001 - a monitor must not die on one bad checkpoint
        return None


def _spacing(G) -> float:
    import numpy as np
    d = np.sqrt(sum((np.diff(A, axis=1)) ** 2 for A in G))
    d = d[np.isfinite(d)]
    return float(np.median(d)) if d.size else 20.0


def dense_points(G, lo, hi, eps: float):
    """Bilinear-densified vertices of grid G inside the box [lo, hi], spaced <= eps*0.9, so that 'distance to the nearest point' ~ 'distance to the SURFACE' (a vertex lattice 20 voxels
    apart would otherwise hide a perfect overlap: vertices of one surface sit up to ~14 voxels from the other's nearest vertex)."""
    import numpy as np
    from scipy.ndimage import map_coordinates
    f = max(1, int(np.ceil(_spacing(G) / (eps * 1.2))))
    H, W = G[0].shape
    keep = (G[0] >= lo[0]) & (G[0] <= hi[0]) & (G[1] >= lo[1]) & (G[1] <= hi[1]) & (G[2] >= lo[2]) & (G[2] <= hi[2])
    if not keep.any():
        return np.zeros((0, 3), "float32")
    r = np.where(keep.any(1))[0]
    c = np.where(keep.any(0))[0]
    r0, r1, c0, c1 = max(0, r.min() - 1), min(H - 1, r.max() + 1), max(0, c.min() - 1), min(W - 1, c.max() + 1)
    sub = [A[r0:r1 + 1, c0:c1 + 1] for A in G]
    h, w = sub[0].shape
    yy, xx = np.meshgrid(np.linspace(0, h - 1, (h - 1) * f + 1), np.linspace(0, w - 1, (w - 1) * f + 1), indexing="ij")
    P = np.stack([map_coordinates(A, [yy, xx], order=1, cval=np.nan, mode="constant").ravel() for A in sub], axis=1)
    P = P[np.isfinite(P).all(1)]
    m = (P >= lo).all(1) & (P <= hi).all(1)
    return P[m].astype("float32")


def scan_scroll(scdir: Path, eps: float, stride: int, min_frac: float, deadline: float):
    """One scroll: (pairs, best, n_segments, finished).  Cheap COARSE test first (the smaller segment's vertices against the larger one's raw vertices within 0.8 x the grid spacing -- every
    true overlap passes it); only the pairs that pass get the fine test against a densified surface."""
    import numpy as np
    from scipy.spatial import cKDTree
    segs = []
    for sd in sorted(p for p in scdir.glob("*") if p.is_dir()):
        ck = newest_checkpoint(sd)
        if ck is None:
            continue
        lg = load_grid(ck)
        if lg is None:
            continue
        G, area = lg
        pts = np.stack([A[::stride, ::stride].ravel() for A in G], axis=1)
        pts = pts[np.isfinite(pts).all(1)]
        allp = np.stack([A.ravel() for A in G], axis=1)
        allp = allp[np.isfinite(allp).all(1)]
        if len(pts) < 10 or len(allp) < 10:
            continue
        segs.append({"seg": sd.name, "G": G, "P": pts, "ALL": allp, "area": area, "n": len(allp), "lo": allp.min(0) - eps, "hi": allp.max(0) + eps, "tree": None, "sp": _spacing(G)})
    segs.sort(key=lambda s_: s_["n"])
    pairs, best, finished = [], {}, True
    for i, A in enumerate(segs):
        for B in segs[i + 1:]:
            if time.time() > deadline:
                return pairs, best, len(segs), False
            if np.any(A["hi"] < B["lo"]) or np.any(B["hi"] < A["lo"]):
                continue
            if B["tree"] is None:
                B["tree"] = cKDTree(B["ALL"])
            dc, _ = B["tree"].query(A["P"], k=1, distance_upper_bound=0.8 * B["sp"])
            if float(np.isfinite(dc).mean()) < min_frac:
                continue                                                            # not even close: skip the expensive test
            lo, hi = np.maximum(A["lo"], B["lo"]), np.minimum(A["hi"], B["hi"])
            D = dense_points(B["G"], lo - eps, hi + eps, eps)
            if len(D) < 3:
                continue
            d, _ = cKDTree(D).query(A["P"], k=1, distance_upper_bound=eps)
            frac = float(np.isfinite(d).mean())
            if frac >= min_frac:
                cm2 = round(frac * A["area"], 2)
                pairs.append({"scroll": scdir.name, "small": A["seg"], "large": B["seg"], "frac": round(frac, 3), "cm2": cm2})
                if frac > best.get(A["seg"], (0.0, 0.0))[0]:
                    best[A["seg"]] = (frac, cm2)
    return pairs, best, len(segs), finished


def _sig(scdir: Path) -> str:
    """Cheap change signature of a scroll: every segment's newest checkpoint dir name + mtime."""
    parts = []
    for sd in sorted(p for p in scdir.glob("*") if p.is_dir()):
        ck = newest_checkpoint(sd)
        if ck is not None:
            parts.append(f"{sd.name}:{ck.parent.name}/{ck.name}:{int((ck / 'meta.json').stat().st_mtime)}")
    return "|".join(parts)


def scan(work, eps: float = EPS_VOX, stride: int = STRIDE, min_frac: float = MIN_FRAC, time_budget_s: float = 25.0, use_cache: bool = True) -> dict:
    """Pairwise overlap of every Route A segment with every other segment of the same scroll.  Per-scroll results are cached (<work>/status/overlap_scrolls.json, keyed by the scroll's
    checkpoint signature), so an unchanged scroll costs nothing; a scan that runs out of `time_budget_s` returns what it has with partial=True and the next call continues."""
    t0 = time.time()
    work = Path(work)
    cp = work / "status" / "overlap_scrolls.json"
    cache = {}
    if use_cache:
        try:
            cache = json.loads(cp.read_text())
        except (OSError, ValueError):
            cache = {}
    pairs, best, nseg, partial = [], {}, 0, False
    for scdir in sorted((work / "export").glob("*")):
        if not scdir.is_dir():
            continue
        sig = _sig(scdir)
        ent = cache.get(scdir.name)
        if ent and ent.get("sig") == sig and ent.get("eps") == eps and ent.get("stride") == stride:
            sp, sb, sn = ent["pairs"], {k: tuple(v) for k, v in ent["best"].items()}, ent["nseg"]
        else:
            if partial or time.time() - t0 > time_budget_s:
                partial = True
                continue
            sp, sb, sn, fin = scan_scroll(scdir, eps, stride, min_frac, t0 + time_budget_s)
            if fin:
                cache[scdir.name] = {"sig": sig, "eps": eps, "stride": stride, "pairs": sp, "best": {k: list(v) for k, v in sb.items()}, "nseg": sn}
            else:
                partial = True
        pairs += sp
        best.update(sb)
        nseg += sn
    if use_cache:
        try:
            cp.parent.mkdir(parents=True, exist_ok=True)
            tmp = cp.with_suffix(".tmp")
            tmp.write_text(json.dumps(cache))
            os.replace(tmp, cp)
        except OSError:
            pass
    pairs.sort(key=lambda p_: -p_["cm2"])
    return {"eps_vox": eps, "stride": stride, "min_frac": min_frac, "pairs": pairs, "n_segments": nseg, "n_overlapping": len(best),
            "dup_cm2": round(sum(v[1] for v in best.values()), 1), "partial": partial, "seconds": round(time.time() - t0, 1)}


def cached_scan(work, max_age_s: float = 300.0, **kw) -> dict | None:
    """What the dashboard calls.  Returns the aggregated result in <work>/status/overlap.json at once; when it is older than max_age_s a DETACHED refresh process is started (so the
    watcher never blocks).  With no result at all it scans synchronously for up to 8 s (partial results are flagged) and starts the background refresh for the rest."""
    import subprocess
    import sys
    work = Path(work)
    cpath = work / "status" / "overlap.json"
    lock = work / "status" / "overlap.lock"
    d, age = None, None
    try:
        d, age = json.loads(cpath.read_text()), time.time() - cpath.stat().st_mtime
    except (OSError, ValueError):
        pass
    if d is None or age > max_age_s:
        busy = False
        try:
            busy = time.time() - lock.stat().st_mtime < 600
        except OSError:
            pass
        if not busy:
            try:
                lock.parent.mkdir(parents=True, exist_ok=True)
                lock.write_text(str(os.getpid()))
                src = str(Path(__file__).resolve().parents[2])
                subprocess.Popen([sys.executable, "-m", "vesuvius_pipeline.routea_cloud.overlap", "--work", str(work), "--refresh"], env={**os.environ, "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", "")},
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True, preexec_fn=lambda: os.nice(19))      # low priority: Route A's cores are the product
            except OSError:
                pass
        if d is None:
            try:
                d = scan(work, time_budget_s=8.0, **kw)
            except Exception:  # noqa: BLE001
                return None
    return d


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", default=os.environ.get("ROUTEA_WORK") or "/workspace/routeB/routeA_work")
    ap.add_argument("--eps", type=float, default=EPS_VOX)
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--refresh", action="store_true", help="scan everything (no time budget) and write <work>/status/overlap.json for the dashboard")
    a = ap.parse_args(argv)
    d = scan(a.work, eps=a.eps, stride=a.stride, time_budget_s=1e9)
    if a.refresh:
        cp = Path(a.work) / "status" / "overlap.json"
        cp.parent.mkdir(parents=True, exist_ok=True)
        tmp = cp.with_suffix(".tmp")
        tmp.write_text(json.dumps(d))
        os.replace(tmp, cp)
        try:
            (Path(a.work) / "status" / "overlap.lock").unlink()
        except OSError:
            pass
    print(f"{d['n_segments']} segments, {d['n_overlapping']} overlap another (eps {d['eps_vox']} vox, stride {d['stride']}), ~{d['dup_cm2']} cm2 duplicated, {d['seconds']} s")
    for p in d["pairs"][:20]:
        print(f"  {p['scroll']}  {p['small']}  ~  {p['large']}   {p['frac'] * 100:.0f} % of the smaller  (~{p['cm2']} cm2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
