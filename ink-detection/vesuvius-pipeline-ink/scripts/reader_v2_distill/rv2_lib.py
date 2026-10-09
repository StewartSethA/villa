"""Shared scoring for the Reader v2 dense distillation (2026-10-06): labelled subjects, near-64 regions,
AUC, edge rise, parent-bootstrap CIs. Used by the trainer (TensorBoard label-val) and by rv2_eval.py.

Subjects are `ink9um_tta_prep.py` output (<subj_root>/<name>/{x.npy, lab.npy, meta.json}); lab bits:
1 = ink, 2 = judged. Ink labels are POSITIVE-ONLY evidence (CLAUDE.md D23), so the primary region is
near64 = judged AND within 64 px (~0.61 mm at 9.5 um/px) of labelled ink; 'judged' (the region every
earlier crossgroup table used) is reported beside it for comparability.
"""
import json
import os
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "hires_ink"))
from ink9um_sharpness import profile_rise_px  # noqa: E402

NEAR_PX = 64
# Reader v2 trained on S1 (PHercParis4), S4 (PHerc1667), S5 (PHerc0172), PHerc0139/0814/0500P2/0009B/0343P
# (reader_v2.py docstring, from its train_config.json). Only PHerc0841 and the Kaggle fragments are unseen.
RV2_SEEN_SCROLLS = {"PHercParis4", "PHerc1667", "PHerc0172", "PHerc0139", "PHerc0814", "PHerc0500P2",
                    "PHerc0009B", "PHerc0343P", "PHerc0343"}


# SEGMENT-level provenance (2026-10-06, read from the `config.datasets` embedded in reader-v2-step040000.pth, its
# init reader-v2-init-ft12k.pth and ink_9um step-075000.pth -- identical segment lists to the HF train_config.json):
# a labelled tile is SEGMENT-SEEN when its parent segment is one the Reader v2 chain trained on. PHercParis4 w04/w08
# are in NO checkpoint's list. The S1 mapping (w-number -> segment id) is scripts/letter_detect/fetch_s1_9um.py CASES.
# Segment ids the Reader v2 chain trained on are read from the checkpoint config.datasets (see the README);
# the list is not shipped. Set RV2_TRAIN_SEGMENT_KEYS_FILE to a text file with one key per line.
import os as _os
_kf = _os.environ.get("RV2_TRAIN_SEGMENT_KEYS_FILE")
RV2_TRAIN_SEGMENT_KEYS = tuple(l.strip() for l in open(_kf) if l.strip()) if _kf and _os.path.exists(_kf) else ()


def rv2_segment_seen(parent: str, scroll: str) -> str:
    """'segment-seen' | 'segment-unseen (scroll seen)' | 'scroll-unseen' for Reader v2 (and its ink_9um ancestry)."""
    if any(k in (parent or "") for k in RV2_TRAIN_SEGMENT_KEYS):
        return "segment-seen"
    return "segment-unseen (scroll seen)" if scroll in RV2_SEEN_SCROLLS else "scroll-unseen"


def auc(score, label):
    l = label.astype(bool)
    n1 = int(l.sum()); n0 = len(l) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[l].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def report_group(meta):
    """Reporting group: S1 / S4 / S5 / PHerc0841 / fragments / other-open (seen by Reader v2)."""
    sc = meta.get("scroll") or ""
    g = meta.get("group") or ""
    if g.startswith("fragment") or sc.startswith(("PHercParis2Fr", "PHercParis1Fr", "PHerc1667Cr")):
        return "fragments (Kaggle, UNSEEN)"
    if sc == "PHercParis4":
        return "S1 PHercParis4 (seen)"
    if sc == "PHerc1667":
        return "S4 PHerc1667 (seen)"
    if sc == "PHerc0172":
        return "S5 PHerc0172 (seen)"
    if sc == "PHerc0841":
        return "PHerc0841 (UNSEEN)"
    return f"{sc} (seen)"


def load_subjects(subj_root, select_layers):
    out = []
    for name in sorted(os.listdir(subj_root)):
        d = os.path.join(subj_root, name)
        if not os.path.exists(os.path.join(d, "meta.json")):
            continue
        meta = json.load(open(os.path.join(d, "meta.json")))
        lab = np.load(os.path.join(d, "lab.npy"))
        ink = (lab & 1) > 0
        judged = (lab & 2) > 0
        if judged.sum() < 20 or not ink[judged].any() or ink[judged].all():
            continue
        dist_out = distance_transform_edt(~ink)
        near = judged & (dist_out <= NEAR_PX)
        xf = np.load(os.path.join(d, "x.npy"), mmap_mode="r")
        x = np.ascontiguousarray(xf[select_layers(xf.shape[0])])
        din = distance_transform_edt(ink)
        sdist = np.where(ink, din, -dist_out).astype(np.float32)
        valid = x[x.shape[0] // 2] > 0
        dedge = distance_transform_edt(valid) if not valid.all() else np.full(valid.shape, 1e9, np.float32)
        out.append({"name": name, "meta": meta, "scroll": meta.get("scroll"), "group": report_group(meta),
                    "parent": meta.get("parent", name), "um": float(meta.get("um") or 9.5), "x": x,
                    "ink": ink, "judged": judged, "near": near, "sdist": sdist,
                    "seg_seen": rv2_segment_seen(meta.get("parent", name), meta.get("scroll") or ""),
                    "valid": valid, "dedge": dedge, "dink": dist_out,
                    # confirmed negatives (D23): fully annotated substrates only -- the S4 HF letter boxes and the
                    # Kaggle fragments ("every pixel judged"); elsewhere an unlabelled pixel is not a negative
                    "confirmed_neg": ("every pixel judged" in (meta.get("neg_rule") or "")
                                      or "fully annotated" in (meta.get("neg_rule") or ""))})
    return out


def score_map(s, p):
    """AUC on near64 and judged, edge rise (um) on judged, for one probability map."""
    r = {"near64": auc(p[s["near"]], s["ink"][s["near"]]), "judged": auc(p[s["judged"]], s["ink"][s["judged"]])}
    v = s["judged"]
    if s["ink"].sum() >= 200 and (~s["ink"][v]).sum() >= 200:
        rise, _, _ = profile_rise_px(s["sdist"][v], p[v].astype(np.float32))
        r["rise_um"] = rise * s["um"] if rise is not None else None
        # the +-15 px profile caps the 10-90 % rise near 0.8 x 30 px (~230 um): raw CT (colmean) reads ~180 um, so
        # a smooth map is PINNED at that ceiling. A +-40 px profile (cap ~610 um) is reported beside it.
        rise40, _, _ = profile_rise_px(s["sdist"][v], p[v].astype(np.float32), lo=-40, hi=40)
        r["rise40_um"] = rise40 * s["um"] if rise40 is not None else None
    else:
        r["rise_um"] = r["rise40_um"] = None
    return r


def label_edge_um(s):
    v = s["judged"]
    rise, _, _ = profile_rise_px(s["sdist"][v], s["ink"][v].astype(np.float32))
    return rise * s["um"] if rise is not None else None


def size_line(subs):
    """n tiles / parents, MPix scored (near64), cm2 near64, cm2 labelled ink inside it."""
    um = np.array([s["um"] for s in subs])
    near = np.array([s["near"].sum() for s in subs]); ink = np.array([(s["ink"] & s["near"]).sum() for s in subs])
    return {"n_tiles": len(subs), "n_parents": len({s["parent"] for s in subs}),
            "MPix_near64": round(float(near.sum()) / 1e6, 3),
            "MVox_near64_17L": round(float(near.sum()) * 17 / 1e6, 2),
            "cm2_near64": round(float((near * (um / 1e4) ** 2).sum()), 3),
            "cm2_ink": round(float((ink * (um / 1e4) ** 2).sum()), 4)}


def parent_boot(rows, key, n=2000, seed=0):
    """mean over tiles, 95 % CI by bootstrapping PARENTS (the independent unit)."""
    rng = np.random.default_rng(seed)
    byp = {}
    for r in rows:
        v = r.get(key)
        if v is not None and np.isfinite(v):
            byp.setdefault(r["parent"], []).append(v)
    if not byp:
        return None
    ps = list(byp)
    pm = np.array([np.mean(byp[p]) for p in ps])
    vals = [v for p in ps for v in byp[p]]
    if len(ps) < 2:
        return {"mean": float(np.mean(vals)), "ci95": None, "n_parents": 1}
    boots = [pm[rng.integers(0, len(ps), len(ps))].mean() for _ in range(n)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": float(np.mean(vals)), "ci95": [float(lo), float(hi)], "n_parents": len(ps)}
