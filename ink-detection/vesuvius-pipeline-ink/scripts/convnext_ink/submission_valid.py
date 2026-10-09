"""Which submission tiles are still VALID, and how much of each?  (user, 2026-10-08: "which submission tiles are still valid? can you just mask out the
overlapped portion from the submission tiles, with a small buffer?")

A submission region must not overlap ANY training data: (1) the segments registered for training, (2) the segments the existing ink models trained on
(model_training_sets.json).  For each submission tile this masks every cell within EXCL_VOX (3-D, same sheet) of such a segment's surface, dilated by BUFFER_MM
(a named, disclosed parameter; the model context window is the reason it exists), and reports what is left:
    valid_cm2            area of cells that survive
    largest_component_cm2   the largest CONNECTED valid region (a submission needs ONE 4 cm2 area with >= 10 letters)
    largest_square_mm    side of the largest axis-aligned valid square in the tile's own grid (4 cm2 = a 20 x 20 mm square)
    passes_4cm2          largest_component_cm2 >= 4.0
Bounding-box pre-filter: only segments whose 3-D box meets a submission tile's box (+ EXCL_VOX + buffer) are densified, so 400+ candidates stay cheap.
Writes <out>/<seg>.submission_exclude.npy (bool grid, True = overlapped/buffered) and <out>/submission_valid.json.   Sensitivity at 0.5 / 1 / 2 mm is reported.
usage: submission_valid.py --out DIR [--buffer-mm 1.0]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
import split_guard as SG  # noqa: E402

BUFFER_MM = 1.0
WINDOW_CM2 = 4.0
EXCL_SLACK = 12.0     # candidate vertices are not densified: ~half its grid spacing (10 vox) of slack so no touching segment is missed


def bbox(G):
    return [np.nanmin(A) for A in G] + [np.nanmax(A) for A in G]


def meet(a, b, pad):
    return all(a[i] - pad <= b[i + 3] and b[i] - pad <= a[i + 3] for i in range(3))


def largest_square(keep):
    h, w = keep.shape
    dp = np.zeros((h + 1, w + 1), np.int32)
    best = 0
    for i in range(1, h + 1):
        row = keep[i - 1]
        for j in range(1, w + 1):
            if row[j - 1]:
                dp[i, j] = 1 + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
                if dp[i, j] > best:
                    best = dp[i, j]
    return int(best)


def tile_report(seg, G, V, cands, buffer_mm, um, area_cm2):
    X = G[0]
    spacing = float(np.nanmedian(np.sqrt(np.diff(X, axis=1) ** 2 + np.diff(G[1], axis=1) ** 2 + np.diff(G[2], axis=1) ** 2)))
    cell_mm = spacing * um / 1000.0
    mc = int(np.ceil(buffer_mm / max(cell_mm, 1e-6)))
    res, _ = SG.guard({k: v for k, v in cands.items()}, {seg: (G, V)}, excl=SG.EXCL_VOX, margin_cells=mc)
    r = res[seg]
    keep = V & ~r["exclude"]
    from scipy import ndimage as ndi
    lab, n = ndi.label(keep)
    cell_cm2 = (area_cm2 / max(1, int(V.sum()))) if area_cm2 else (cell_mm ** 2) / 100.0
    big = int(np.bincount(lab.ravel())[1:].max()) if n else 0
    sq = largest_square(keep) if keep.sum() < 4_000_000 else 0
    return r["exclude"], dict(cell_mm=round(cell_mm, 3), buffer_cells=mc, valid_cells=int(V.sum()), kept_cells=int(keep.sum()), valid_cm2=round(float(keep.sum()) * cell_cm2, 2),
                              masked_frac=round(1 - float(keep.sum()) / max(1, int(V.sum())), 4), largest_component_cm2=round(big * cell_cm2, 2),
                              largest_square_mm=round(sq * cell_mm, 1), passes_4cm2=bool(big * cell_cm2 >= WINDOW_CM2), min_dist_kept_vox=r["min_dist_kept_vox"])


def touching(G, cands, slack=None):
    """Candidates with ANY vertex within EXCL_VOX + slack of tile G's densified surface (exact, vertex-level; the 3-D box of a wrapped ribbon filters nothing)."""
    from scipy.spatial import cKDTree
    slack = EXCL_SLACK if slack is None else slack
    tree = cKDTree(SG.densify(G, SG.EPS_VOX))
    out = {}
    for k, (CG, CV) in cands.items():
        q = np.stack([CG[0][CV], CG[1][CV], CG[2][CV]], 1)
        if len(q) and float(tree.query(q, k=1)[0].min()) < SG.EXCL_VOX + slack:
            out[k] = (CG, CV)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--buffer-mm", type=float, default=BUFFER_MM)
    ap.add_argument("--splits", default=str(ROOT / "docs/experiments/cnx_ink/splits.json"))
    ap.add_argument("--models", default=str(ROOT / "docs/experiments/cnx_ink/model_training_sets.json"))
    a = ap.parse_args()
    from vesuvius_pipeline import config, scrolls as SC
    from vesuvius_pipeline.db import pipeline_db
    fl = config.load(None)
    db = pipeline_db().connect(str(fl.pipeline_db))
    sp, ms = json.load(open(a.splits)), json.load(open(a.models))
    work = Path(a.out) / "_pulled"
    Path(a.out).mkdir(parents=True, exist_ok=True)
    subs = {}
    for s in sp["submission"]:
        g, why = SG.load_grid(s, fl, db, work)
        if g:
            subs[s] = g
        else:
            print(f"  {s}: NO MESH ({why})")
    sbox = {s: bbox(g[0]) for s, g in subs.items()}
    pad = SG.EXCL_VOX + 400
    cands, boxes, unknown = {}, {}, []
    names = [s for s in sp["train"] if s not in subs] + [s for s in ms.get("distill_rv2", []) if s not in subs and s not in sp["train"]]
    for i, s in enumerate(names):
        g, why = SG.load_grid(s, fl, db, work)
        if not g:
            unknown.append((s, why))
            continue
        cands[s] = g
        if i % 50 == 0:
            print(f"  loaded {i}/{len(names)} candidates", flush=True)
    print(f"candidates loaded: {len(cands)} of {len(names)}; no obtainable mesh: {len(unknown)}")
    rep = dict(parameters=dict(EXCL_VOX=SG.EXCL_VOX, BUFFER_MM=a.buffer_mm, WINDOW_CM2=WINDOW_CM2), candidates=len(cands), unknown_geometry=unknown, tiles={})
    for s, (G, V) in subs.items():
        um = SC.voxel_um(s.split("_")[0]) or 9.0
        area = db.execute("SELECT value FROM metric WHERE seg=? AND name='area_cm2' ORDER BY id DESC LIMIT 1", (s,)).fetchone()
        near = touching(G, cands)
        ex, r = tile_report(s, G, V, near, a.buffer_mm, um, float(area[0]) if area else None)
        np.save(Path(a.out) / f"{s}.submission_exclude.npy", ex)
        r["overlapping_segments"] = len(near)
        sens = {}
        for b in (0.5, 1.0, 2.0):
            _e, rb = tile_report(s, G, V, near, b, um, float(area[0]) if area else None)
            sens[str(b)] = dict(valid_cm2=rb["valid_cm2"], largest_component_cm2=rb["largest_component_cm2"], largest_square_mm=rb["largest_square_mm"])
        r["sensitivity_buffer_mm"] = sens
        rep["tiles"][s] = r
        print(f"{s:34s} area {area[0] if area else '?':>6.4} cm2 | near segs {len(near):3d} | masked {100 * r['masked_frac']:5.1f} % | valid {r['valid_cm2']:6.2f} cm2 | largest region {r['largest_component_cm2']:6.2f} cm2 | square {r['largest_square_mm']:5.1f} mm | 4 cm2 window: {'YES' if r['passes_4cm2'] else 'no'}", flush=True)
    json.dump(rep, open(Path(a.out) / "submission_valid.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
