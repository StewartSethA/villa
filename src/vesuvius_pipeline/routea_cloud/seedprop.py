"""DB-free seed proposer (the production seeder, stages/seed.py, needs the coverage field + DB; a fresh box has neither). Same ideas: candidates = prediction > PRED_TH at a coarse
pyramid level, random draw weighted by distance to the support edge (thick interior preferred), greedy acceptance with a minimum separation and an edge margin at level 0, a level-0
window check; PLUS spreading across wraps by umbilicus radius bins when an umbilicus is available. Deterministic for a given rng_seed. Seeds are (x, y, z) in LEVEL-0 voxel
coordinates of the prediction zarr = the tracer's -v frame."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PRED_TH = 180          # stages/seed.py: thresholded m7 prediction is 0/255; >180 keeps only positives
EDGE_MARGIN_VOX = 100
SEED_WINDOW_MEAN = 20.0
SEED_CENTRE_MIN = 20.0
MIN_SEP_VOX = 250.0
MAX_COARSE_VOXELS = 40_000_000          # keeps the seeder under ~1.5 GB RAM (a 254 M-voxel level OOM-killed the first proof run on a 14 GB box)


def open_levels(pred_dir):
    import zarr
    z = zarr.open(str(pred_dir), mode="r")
    names = sorted((k for k in z.keys() if k.isdigit()), key=int)
    if "0" not in names:
        raise ValueError(f"{pred_dir}: no level 0")
    return z, names


def level_factor(pred_dir, z, name: str) -> float:
    """Voxel size of level `name` relative to level 0 from multiscales metadata when present, else 2**level."""
    try:
        zat = json.loads((Path(pred_dir) / ".zattrs").read_text())
        ds = zat["multiscales"][0]["datasets"]
        sc = {d["path"]: d["coordinateTransformations"][0]["scale"][-1] for d in ds}
        if sc.get("0") and sc.get(name):
            return float(sc[name]) / float(sc["0"])
    except Exception:                                                  # noqa: BLE001
        pass
    return float(2 ** int(name))


def umbilicus_fn(path):
    pts = json.loads(Path(path).read_text())
    pts = pts.get("control_points") if isinstance(pts, dict) else pts
    pts = sorted(pts, key=lambda p: p["z"])
    zs = np.array([p["z"] for p in pts], float); xs = np.array([p["x"] for p in pts], float); ys = np.array([p["y"] for p in pts], float)
    return lambda z: (np.interp(np.clip(z, zs[0], zs[-1]), zs, xs), np.interp(np.clip(z, zs[0], zs[-1]), zs, ys))


def propose(pred_dir, n: int, rng_seed: int = 1, min_sep: float = MIN_SEP_VOX, umbilicus: str | None = None, log=print) -> list[dict]:
    from scipy import ndimage as ndi
    z, names = open_levels(pred_dir)
    l0 = z["0"]
    pick = None
    for nm in names:                                                       # finest level that fits the budget
        if int(np.prod(z[nm].shape)) <= MAX_COARSE_VOXELS:
            pick = nm
            break
    if pick is None:
        pick = names[-1]
    arr = np.asarray(z[pick][:])
    F = level_factor(pred_dir, z, pick)
    sup = arr > PRED_TH
    if not sup.any():
        raise ValueError(f"{pred_dir}: no prediction voxels above {PRED_TH} at level {pick}")
    dist = np.minimum(ndi.distance_transform_edt(sup), 12.0).astype(np.float32)
    idx = np.flatnonzero(dist.ravel() > 0)
    rng = np.random.default_rng(rng_seed)
    w = dist.ravel()[idx]
    n_draw = min(idx.size, 20000)
    draw = rng.choice(idx, size=n_draw, replace=False, p=w / w.sum())
    zz, yy, xx = np.unravel_index(draw, dist.shape)
    X = (xx * F + F / 2).astype(np.int64); Y = (yy * F + F / 2).astype(np.int64); Z = (zz * F + F / 2).astype(np.int64)
    score = dist.ravel()[draw]
    shape0 = l0.shape
    ok = (Z >= EDGE_MARGIN_VOX) & (Z < shape0[0] - EDGE_MARGIN_VOX) & (Y >= EDGE_MARGIN_VOX) & (Y < shape0[1] - EDGE_MARGIN_VOX) & (X >= EDGE_MARGIN_VOX) & (X < shape0[2] - EDGE_MARGIN_VOX)
    cand = np.flatnonzero(ok)
    rad = None
    if umbilicus and Path(umbilicus).is_file():
        ux, uy = umbilicus_fn(umbilicus)(Z.astype(float))
        rad = np.hypot(X - ux, Y - uy)
    # radius bins: round-robin over wraps (quantile bins of candidate radius) so seeds land on different sheets; without an umbilicus one bin
    bins = [cand]
    if rad is not None and n > 1:
        q = np.quantile(rad[cand], np.linspace(0, 1, n + 1))
        bins = [cand[(rad[cand] >= q[i]) & (rad[cand] <= q[i + 1])] for i in range(n)]
    bins = [b[np.argsort(-score[b], kind="stable")] for b in bins if len(b)]
    out: list[dict] = []
    taken: list[tuple[int, int, int]] = []
    tries = 0
    pos = [0] * len(bins)
    while len(out) < n and tries < 200000 and any(pos[i] < len(bins[i]) for i in range(len(bins))):
        for bi in range(len(bins)):
            if len(out) >= n:
                break
            while pos[bi] < len(bins[bi]):
                i = bins[bi][pos[bi]]; pos[bi] += 1; tries += 1
                x, y, zc = int(X[i]), int(Y[i]), int(Z[i])
                if any((x - a) ** 2 + (y - b) ** 2 + (zc - c) ** 2 < min_sep ** 2 for a, b, c in taken):
                    continue
                win = np.asarray(l0[zc - 3:zc + 4, y - 3:y + 4, x - 3:x + 4])
                if not (float(win.mean()) > SEED_WINDOW_MEAN and float(win[3, 3, 3]) > SEED_CENTRE_MIN):
                    continue
                taken.append((x, y, zc))
                out.append({"x": x, "y": y, "z": zc, "score": float(score[i]), "level": pick, "radius_vox": None if rad is None else float(rad[i])})
                break
    if len(out) < n:
        log(f"seedprop: only {len(out)} of {n} seeds found (min_sep {min_sep}, level {pick})")
    return out
