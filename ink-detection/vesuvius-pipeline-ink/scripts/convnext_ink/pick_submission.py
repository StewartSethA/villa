"""Pick submission tiles with NO training overlap (user, 2026-10-09: "pick 1-3 submission ones from each scroll with no training overlap; can we grab the
overlapped ones into submission, or remove them from training?").  Protected-from set = registry TRAIN segments + the segments existing models trained on
(model_training_sets.json distill_rv2).  One KD-tree over their densified surfaces (EPS_VOX) with a source label per point; a candidate overlaps a source when any
candidate vertex is within EXCL_VOX + SLACK of it.  Candidates = Route B tiles (z..x.. names) with a tifxyz on this host; N_CAND random (seeded) per scroll.
Reports per candidate: overlapping train segs, overlapping model-pool segs, area cm2.  Also the pair matrix for the CURRENT submission tiles so the
'promote overlapping train segments into submission' option can be counted.  Writes <out>/pick.json.
usage: pick_submission.py --out DIR [--n-cand 150] [--seed 7] [--min-cm2 4]"""
import argparse, json, random, re, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "src"))
import split_guard as SG
from scipy.spatial import cKDTree
SLACK = 6.0
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--n-cand", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7); ap.add_argument("--min-cm2", type=float, default=4.0); a = ap.parse_args()
    from vesuvius_pipeline import config
    from vesuvius_pipeline.db import pipeline_db
    fl = config.load(None); db = pipeline_db().connect(str(fl.pipeline_db))
    sp = json.load(open(ROOT / "docs/experiments/cnx_ink/splits.json")); ms = json.load(open(ROOT / "docs/experiments/cnx_ink/model_training_sets.json"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); work = out / "_pulled"
    src_names = {}; pts = []; lab = []
    names = [(s, "train") for s in sp["train"]] + [(s, "model") for s in ms.get("distill_rv2", []) if s not in sp["train"] and s not in sp["submission"]]
    for i, (s, role) in enumerate(names):
        g, why = SG.load_grid(s, fl, db, work)
        if not g: continue
        P = SG.densify(g[0], SG.EPS_VOX); pts.append(P); lab.append(np.full(len(P), len(src_names), np.int32)); src_names[len(src_names)] = (s, role)
    tree = cKDTree(np.concatenate(pts)); L = np.concatenate(lab); print(f"protected-from set: {len(src_names)} segments, {len(L)} pts", flush=True)
    def overlaps(G):
        q = np.stack([A[np.isfinite(A)] for A in G], 1)
        d, i = tree.query(q, k=1, distance_upper_bound=SG.EXCL_VOX + SLACK)
        hit = np.isfinite(d); ids = np.unique(L[i[hit]]) if hit.any() else []
        return float(hit.mean()), [src_names[int(k)] for k in ids]
    rep = {"params": dict(EXCL_VOX=SG.EXCL_VOX, SLACK=SLACK, EPS_VOX=SG.EPS_VOX, N_CAND=a.n_cand, seed=a.seed, MIN_CM2=a.min_cm2), "current_submission": {}, "candidates": {}}
    for s in sp["submission"]:
        g, why = SG.load_grid(s, fl, db, work)
        if g:
            f, o = overlaps(g[0]); rep["current_submission"][s] = dict(frac=round(f, 3), train=[x[0] for x in o if x[1] == "train"], model=[x[0] for x in o if x[1] == "model"])
            print(s, rep["current_submission"][s]["frac"], len(rep["current_submission"][s]["train"]), len(rep["current_submission"][s]["model"]), flush=True)
    rng = random.Random(a.seed)
    rows = db.execute("SELECT s.seg, s.scroll, a.path FROM segment s JOIN artifact a ON a.seg=s.seg AND a.kind='tifxyz' WHERE s.route='B' AND s.seg GLOB '*_w[0-9]*_z[0-9]*x[0-9]*' GROUP BY s.seg").fetchall()
    by = {}
    for seg, sc, p in rows:
        if p and Path(p, "x.tif").exists(): by.setdefault(sc, []).append((seg, p))
    for sc, lst in sorted(by.items()):
        rng.shuffle(lst); res = []
        for seg, p in lst[: a.n_cand]:
            try: G, V = SG.read_xyz(p)
            except Exception: continue
            f, o = overlaps(G)
            ar = db.execute("SELECT value FROM metric WHERE seg=? AND name='area_cm2' ORDER BY id DESC LIMIT 1", (seg,)).fetchone()
            res.append(dict(seg=seg, frac=round(f, 3), area=float(ar[0]) if ar else None, train=[x[0] for x in o if x[1] == "train"], model=[x[0] for x in o if x[1] == "model"]))
        rep["candidates"][sc] = dict(local_tiles=len(lst), scored=len(res), clean=[r for r in res if r["frac"] == 0 and (r["area"] or 0) >= a.min_cm2])
        print(sc, "local tiles", len(lst), "scored", len(res), "clean(frac 0, >=min cm2)", len(rep["candidates"][sc]["clean"]), flush=True)
    json.dump(rep, open(out / "pick.json", "w"), indent=1)
main()
