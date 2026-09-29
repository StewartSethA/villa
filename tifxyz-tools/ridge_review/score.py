"""Score human ridge_hit reviews: error rates per severity stratum and scroll, reweighted to production.

  python score.py LABELS.json [LABELS.json ...] [--data ridge_review/data] [--boot 4000] [--json out.json]

Inputs (all in ridge_review/data unless --data says otherwise):
  key.json         per sample: class (removed | kept), stratum, scroll, round stats
  prevalence.json  per source x scroll x stratum: rounds, cells grown / kept / flagged / removed_by_ridge
  samples/*.npz    the review inputs (used here for the automated CT on-sheet check)
  LABELS.json      exported from the review page: {labels: {sample_id: {label, severity, note, pred_viewed}}}

Definitions (ridge_hit "positive" = the cell is off-sheet, remove it):
  false-removal share  P(human: sheet | ridge_hit removed it)       -- share of the cut that was sound
  miss share           P(human: not sheet | ridge_hit kept it)
  FPR                  P(removed | sheet)     = Nr*fr / (Nr*fr + Nk*sk)
  FNR                  P(kept | not sheet)    = Nk*mk / (Nk*mk + Nr*(1-fr))
where fr = false-removal share, sk = 1 - miss share, mk = miss share, and Nr, Nk are the production cell
counts removed by ridge_hit / kept in that stratum (prevalence.json). Pooled numbers weight each stratum
by its production cells, so the reviewed rates are reweighted to the fleet-wide cut. "unsure" labels are
excluded and counted. CIs: bootstrap over samples within each stratum (percentile, 95 %).
Several label files: each sample's label is the majority (ties -> unsure); pairwise agreement reported.

Automated comparison: the CT on-sheet check (no prediction used) -- along the normal, the brightest CT
within +-3 voxels minus the median over +-12, at the cell vs 10 voxels either side; "sheet" when the
cell's score exceeds both neighbours'. Its agreement with the human labels is reported (Cohen's kappa).
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

STRATA = ("lt10", "10to50", "gt50_or_emptied")


def ct_check(npz):
    z = np.load(npz)
    cs = z["ct_stack"].astype(np.float64)
    D, H = (cs.shape[0] - 1) // 2, (cs.shape[1] - 1) // 2
    prof = cs[:, H, H]                        # the CT along the normal through the cell

    def score(k):
        lo, hi = D + k - 12, D + k + 12
        if lo < 0 or hi >= len(prof):
            w = prof[max(0, lo):min(len(prof), hi + 1)]
        else:
            w = prof[lo:hi + 1]
        c = prof[max(0, D + k - 3):D + k + 4]
        return float(c.max() - np.median(w))
    s0, sp, sm = score(0), score(min(10, D - 3)), score(-min(10, D - 3))
    return "sheet" if s0 > max(sp, sm) else "not_sheet"


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if not len(a):
        return None
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in set(a) | set(b))
    return None if pe >= 1 else float((po - pe) / (1 - pe))


def merge(files):
    per = collections.defaultdict(list)
    who = []
    for f in files:
        d = json.load(open(f))
        who.append(d.get("reviewer") or Path(f).stem)
        for sid, L in d["labels"].items():
            per[sid].append(L["label"])
    out = {}
    for sid, ls in per.items():
        c = collections.Counter(ls).most_common()
        out[sid] = c[0][0] if len(c) == 1 or c[0][1] > c[1][1] else "unsure"
    agree = None
    if len(files) > 1:
        ds = [json.load(open(f))["labels"] for f in files]
        common = set.intersection(*[set(d) for d in ds])
        pairs = [(ds[i][s]["label"], ds[j][s]["label"]) for i in range(len(ds)) for j in range(i + 1, len(ds)) for s in common]
        agree = {"pairs": len(pairs), "raw_agreement": float(np.mean([a == b for a, b in pairs])) if pairs else None,
                 "kappa": kappa([a for a, _ in pairs], [b for _, b in pairs])}
    return out, who, agree


def rates(items):
    """items: list of (class, label). -> (false_removal_share, miss_share, n_removed, n_kept)"""
    r = [lab for c, lab in items if c == "removed"]
    k = [lab for c, lab in items if c == "kept"]
    fr = np.mean([x == "sheet" for x in r]) if r else np.nan
    mk = np.mean([x == "not_sheet" for x in k]) if k else np.nan
    return fr, mk, len(r), len(k)


def fpr_fnr(fr, mk, Nr, Nk):
    sk = 1 - mk
    fpr = Nr * fr / (Nr * fr + Nk * sk) if (Nr * fr + Nk * sk) > 0 else np.nan
    fnr = Nk * mk / (Nk * mk + Nr * (1 - fr)) if (Nk * mk + Nr * (1 - fr)) > 0 else np.nan
    return fpr, fnr


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", nargs="+")
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent / "data"))
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--json")
    ap.add_argument("--weights", default="production_all",
                    help="production_all: every production scroll's cells (default; unsampled scrolls assumed to share "
                         "their stratum's rates) | production_sampled: only the reviewed scrolls")
    a = ap.parse_args(argv)
    data = Path(a.data)
    key = {k["sample_id"]: k for k in json.load(open(data / "key.json"))}
    prev = json.load(open(data / "prevalence.json"))
    human, who, agree = merge(a.labels)
    rows = [(sid, key[sid], lab) for sid, lab in human.items() if sid in key]
    unsure = sum(1 for _, _, lab in rows if lab == "unsure")
    usable = [(sid, k, lab) for sid, k, lab in rows if lab != "unsure"]
    report = {"weights": a.weights, "reviewers": who, "inter_rater": agree, "labelled": len(rows), "unsure_excluded": unsure, "strata": [], "pooled": {}}
    # production cells per stratum (all sources, all scrolls reviewed)
    N = collections.defaultdict(lambda: [0, 0])
    Ns = collections.defaultdict(lambda: [0, 0])
    reviewed = {k["scroll"] for k in key.values()}
    for p in prev:
        if p["source"] != "production_policy_D":
            continue
        if a.weights == "production_sampled" and p["scroll"] not in reviewed:
            continue
        N[p["stratum"]][0] += p["cells_removed_by_ridge"]
        N[p["stratum"]][1] += p["cells_kept"]
        Ns[(p["scroll"], p["stratum"])][0] += p["cells_removed_by_ridge"]
        Ns[(p["scroll"], p["stratum"])][1] += p["cells_kept"]
    rng = np.random.default_rng(1)
    boots = collections.defaultdict(list)
    for st in STRATA:
        it = [(k["class"], lab) for _, k, lab in usable if k["stratum"] == st]
        fr, mk, nr, nk = rates(it)
        fpr, fnr = fpr_fnr(fr, mk, *N[st])
        bs = []
        for _ in range(a.boot):
            s = [it[i] for i in rng.integers(0, len(it), len(it))] if it else []
            f2, m2, _, _ = rates(s)
            bs.append((f2, m2) + fpr_fnr(f2, m2, *N[st]))
        bs = np.array(bs, dtype=float)
        boots[st] = bs
        ci = lambda c: [round(float(x), 3) for x in np.nanpercentile(bs[:, c], [2.5, 97.5])] if len(it) else None  # noqa: E731
        report["strata"].append({"stratum": st, "n_removed": nr, "n_kept": nk, "false_removal_share": fr, "ci": ci(0),
                                 "miss_share": mk, "ci_miss": ci(1), "FPR": fpr, "ci_FPR": ci(2), "FNR": fnr, "ci_FNR": ci(3),
                                 "production_cells_removed_by_ridge": N[st][0], "production_cells_kept": N[st][1]})
    # pooled, weighted by production cells removed (false-removal) / kept (miss)
    w_r = np.array([N[s][0] for s in STRATA], float)
    w_k = np.array([N[s][1] for s in STRATA], float)
    def pooled(col, w):
        pts = np.array([r[k] for r, k in zip(report["strata"], ["false_removal_share"] * 3 if col == 0 else ["miss_share"] * 3, strict=True)], float)
        ok = np.isfinite(pts)
        est = float((pts[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else None
        bb = [np.nansum(np.array([boots[s][b, col] for s in STRATA]) * w * ok) / w[ok].sum() for b in range(a.boot)] if ok.any() else []
        return est, ([round(float(x), 3) for x in np.percentile(bb, [2.5, 97.5])] if bb else None)
    report["pooled"]["false_removal_share_production_weighted"] = pooled(0, w_r)
    report["pooled"]["miss_share_production_weighted"] = pooled(1, w_k)
    # per scroll (unweighted within scroll, small n)
    per = collections.defaultdict(list)
    for _, k, lab in usable:
        per[k["scroll"]].append((k["class"], lab))
    report["per_scroll"] = {sc: dict(zip(("false_removal_share", "miss_share", "n_removed", "n_kept"), map(lambda x: None if isinstance(x, float) and np.isnan(x) else x, rates(it)), strict=True)) for sc, it in sorted(per.items())}
    # automated CT check vs human
    ct, hu, cls = [], [], []
    for sid, k, lab in usable:
        f = data / "samples" / f"{sid}.npz"
        if f.exists():
            ct.append(ct_check(f)); hu.append(lab); cls.append(k["class"])
    report["ct_check_vs_human"] = {"n": len(ct), "agreement": float(np.mean(np.array(ct) == np.array(hu))) if ct else None,
                                   "kappa": kappa(ct, hu),
                                   "ct_false_removal_share": float(np.mean([c == "sheet" for c, k in zip(ct, cls, strict=True) if k == "removed"])) if ct else None}
    txt = json.dumps(report, indent=1, default=lambda x: None if isinstance(x, float) and np.isnan(x) else x)
    print(txt)
    if a.json:
        Path(a.json).write_text(txt)


if __name__ == "__main__":
    main()
