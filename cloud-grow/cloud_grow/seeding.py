"""Coverage-driven seeding without the hub: port of `stages/seed.py` (`propose`) + `stages/coverage.py`.

score(v) = support(v) * dist_to_coverage(v) / MAX_DIST,   support = pred > 180 AND ct > 5   (pyramid level 4)

Coverage comes from the grown tifxyz directories of THIS run (a directory scan, not the hub's inventory), dilated by
RADIUS_L4. Candidates are drawn by score, then accepted greedily with separation `min_sep`, the array-edge margin
(EDGE_MARGIN_VOX: the PHerc0490B failure at 16-24 vox from the edge), and a verification read.

DIFFERENCES from production, all recorded in each seed's provenance (`verify`):
  * production verifies a seed on CT LEVEL 0 (window mean > 20 and centre > 20). A cloud box does not hold level 0
    (PHerc0211: 252 GB), so verification reads the CT at LEVEL 1 (same window, in level-1 voxels) and, when given,
    the level-0 PREDICTION (centre >= PRED_TH). The thresholds are production's; the level is not. NOT measured
    against production's seed acceptance rate.
  * no hub `blocked`/`active` seed lists: the caller passes `exclude` (the seeds already in the local state DB).
  * `seed_dedup` (prototype, default OFF in production) and `mush_gate` (OFF) are not shipped.
UNVALIDATED against human annotation (D6): a seed is a starting point, not a claim about the sheet.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass

import numpy as np

LEVEL = 4
FACTOR = 2 ** LEVEL
RADIUS_L4 = 2
MAX_DIST_L4 = 12
DIST_POOL = 4
PRED_TH = 180
CT_MIN = 5
SEED_WINDOW_MEAN = 20.0
SEED_CENTRE_MIN = 20.0
EDGE_MARGIN_VOX = 100
NEAR_SEP = 64.0


@dataclass
class Seed:
    x: int
    y: int
    z: int
    score: float
    dist_l4: float
    centre: float
    window_mean: float

    def as_dict(self):
        return asdict(self)


def seg_id(scroll: str, seed: Seed) -> str:
    return f"{scroll}_c{hashlib.sha1(f'{seed.x},{seed.y},{seed.z}'.encode()).hexdigest()[:6]}"


def mark_tifxyz(cov: np.ndarray, tifxyz: str, stride: int = 2) -> int:
    import tifffile
    X, Y, Z = (tifffile.imread(os.path.join(tifxyz, f"{a}.tif"))[::stride, ::stride].astype(np.float32) for a in "xyz")
    m = (X > 0) & (Y > 0) & (Z > 0)
    if not m.any():
        return 0
    zi, yi, xi = ((A[m] / FACTOR).astype(np.int64) for A in (Z, Y, X))
    ok = (zi >= 0) & (zi < cov.shape[0]) & (yi >= 0) & (yi < cov.shape[1]) & (xi >= 0) & (xi < cov.shape[2])
    cov[zi[ok], yi[ok], xi[ok]] = True
    return int(ok.sum())


def coverage_mask(shape_l4, tifxyz_dirs) -> np.ndarray:
    from scipy import ndimage
    cov = np.zeros(shape_l4, bool)
    for p in tifxyz_dirs:
        try:
            mark_tifxyz(cov, p)
        except (OSError, ValueError, KeyError, TypeError) as e:
            print(f"[cloud-grow] coverage: skipping {p}: {e}", flush=True)
    return ndimage.binary_dilation(cov, iterations=RADIUS_L4) if cov.any() else cov


def distance(cov: np.ndarray) -> np.ndarray:
    """Level-4 voxels to the nearest covered voxel, capped at MAX_DIST_L4 (coverage.CoverageField.distance)."""
    from scipy import ndimage
    if not cov.any():
        return np.full(cov.shape, float(MAX_DIST_L4), np.float32)
    k = DIST_POOL
    z, y, x = cov.shape
    pz, py, px = -(-z // k), -(-y // k), -(-x // k)
    pad = np.zeros((pz * k, py * k, px * k), bool)
    pad[:z, :y, :x] = cov
    coarse = pad.reshape(pz, k, py, k, px, k).any(axis=(1, 3, 5))
    d = np.minimum(ndimage.distance_transform_edt(~coarse) * k, MAX_DIST_L4).astype(np.float32)
    full = np.repeat(np.repeat(np.repeat(d, k, 0), k, 1), k, 2)[:z, :y, :x]
    full[cov] = 0.0
    return full


def support_from_zarrs(ct_zarr: str, pred_zarr: str | None):
    """Boolean support at level 4. Returns (support, source). `ct-only` fallback = ct > 40 (production's)."""
    import zarr
    ct4 = np.asarray(zarr.open(ct_zarr, mode="r")[str(LEVEL)][:])
    if not pred_zarr:
        return ct4 > 40, "ct-only"
    p4 = np.asarray(zarr.open(pred_zarr, mode="r")[str(LEVEL)][:])
    if p4.shape != ct4.shape:
        zz, yy, xx = (min(a, b) for a, b in zip(p4.shape, ct4.shape))
        sup = np.zeros(ct4.shape, bool)
        sup[:zz, :yy, :xx] = (p4[:zz, :yy, :xx] > PRED_TH) & (ct4[:zz, :yy, :xx] > CT_MIN)
        return sup, "prediction (shape-clipped)"
    return (p4 > PRED_TH) & (ct4 > CT_MIN), "prediction"


def band_support(support_l4: np.ndarray, zmin: int | None, zmax: int | None) -> np.ndarray:
    """Restrict seeding to the level-0 z band [zmin, zmax) (fleet sharding: boxes of one scroll get disjoint bands so no two
    boxes seed the same region). Returns a copy; cells outside the band are unsupported. FACTOR = level-4 -> level-0."""
    out = support_l4.copy()
    if zmin is not None:
        out[: max(0, int(zmin) // FACTOR)] = False
    if zmax is not None:
        out[max(0, int(zmax) // FACTOR):] = False
    return out


def propose(support_l4: np.ndarray, cov_l4: np.ndarray, shape0, count: int = 1, min_sep: float = 250.0,
            rng_seed: int | None = None, exclude=None, blocked=None, near_sep: float = NEAR_SEP,
            verify=None) -> list[Seed]:
    """`verify(x, y, z) -> (centre, window_mean) | None` reads the CT around the L0 point (see module docstring);
    None = no verification (tests / dry planning). `shape0` = level-0 (z, y, x)."""
    dist = distance(cov_l4)
    score = support_l4.astype(np.float32) * (dist / MAX_DIST_L4)
    score[dist < 1.0] = 0.0
    idx = np.flatnonzero(score > 0)
    if idx.size == 0:
        return []
    rng = np.random.default_rng(rng_seed if rng_seed is not None else int.from_bytes(os.urandom(8), "little"))
    w = score.ravel()[idx]
    draw = rng.choice(idx, size=min(idx.size, 4000), replace=False, p=w / w.sum())
    zz, yy, xx = np.unravel_index(draw, score.shape)
    order = np.argsort(-score.ravel()[draw])
    taken = list(exclude or [])
    out: list[Seed] = []
    for i in order:
        if len(out) >= count:
            break
        x, y, z = (int(a[i] * FACTOR + FACTOR // 2) for a in (xx, yy, zz))
        if not (EDGE_MARGIN_VOX <= z < shape0[0] - EDGE_MARGIN_VOX and EDGE_MARGIN_VOX <= y < shape0[1] - EDGE_MARGIN_VOX
                and EDGE_MARGIN_VOX <= x < shape0[2] - EDGE_MARGIN_VOX):
            continue
        if any((x - a) ** 2 + (y - b) ** 2 + (z - c) ** 2 < min_sep ** 2 for a, b, c in taken):
            continue
        if blocked and any((x - a) ** 2 + (y - b) ** 2 + (z - c) ** 2 < near_sep ** 2 for a, b, c in blocked):
            continue
        centre, wmean = 255.0, 255.0
        if verify is not None:
            got = verify(x, y, z)
            if got is None:
                continue
            centre, wmean = got
            if not (wmean > SEED_WINDOW_MEAN and centre > SEED_CENTRE_MIN):
                continue
        taken.append((x, y, z))
        out.append(Seed(x, y, z, float(score.ravel()[draw[i]]), float(dist[zz[i], yy[i], xx[i]]), centre, wmean))
    return out


def make_ct_level1_verifier(ct_zarr: str, level: int = 1):
    """Window-7 mean + centre at CT `level` around a level-0 point (production's check is the same window at level 0)."""
    import zarr
    a = zarr.open(ct_zarr, mode="r")[str(level)]
    f = 2 ** level

    def verify(x, y, z):
        cx, cy, cz = int(round(x / f)), int(round(y / f)), int(round(z / f))
        if min(cx, cy, cz) < 3:
            return None
        win = np.asarray(a[cz - 3:cz + 4, cy - 3:cy + 4, cx - 3:cx + 4])
        if win.shape != (7, 7, 7):
            return None
        return float(win[3, 3, 3]), float(win.mean())
    return verify


def reserve(db, scroll: str, seed: Seed, source: str = "cloud-grow seed", level_note: str = "ct_level1_window7") -> str:
    """Record the seed in the local state DB with its provenance (seed row id analogue, finder, level)."""
    from . import state as ST
    seg = seg_id(scroll, seed)
    ST.record_seed(db, scroll, seg, (seed.x, seed.y, seed.z), source=source, ct_value=seed.centre, score=seed.score,
                   dist_l4=seed.dist_l4, window_mean=seed.window_mean,
                   provenance={"seeder": "cloud_grow.seeding.propose", "coverage_level": LEVEL, "radius_l4": RADIUS_L4,
                               "pred_th": PRED_TH, "ct_min": CT_MIN, "verify": level_note, "edge_margin_vox": EDGE_MARGIN_VOX})
    return seg
