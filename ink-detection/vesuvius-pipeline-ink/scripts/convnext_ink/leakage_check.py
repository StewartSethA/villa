"""THE canonical leakage check for First Letters submissions and for the ConvNeXt ink run.

One command, one verdict, run before every training start, before every evaluation write-up, and by anything that builds a submission.  It reads the registry
(docs/experiments/cnx_ink/splits.json) and the model training sets (docs/experiments/cnx_ink/model_training_sets.json) and checks, in this order:

  1. REGISTRY   no segment sits in two splits; no segment is listed twice.                                           (names)
  2. SPLITS     every TRAIN vertex is >= EXCL_VOX in 3-D from every submission/eval surface after the guard's masks   (geometry, split_guard.guard)
  3. MODELS     for every segment an EXISTING model trained on (the distillation pool lists): its surface must not overlap a submission/eval surface
                (the prize: "ink model outputs of this region should not overlap with any training data used")                        (geometry)
  4. LABELS     no pipeline label version (painted / uploaded / PSEUDO) of a submission/eval segment exists as a trainable label; any that does is reported
                (a pseudo-label made from a model's output on the submission region is training data about it).                         (database)
  5. UNKNOWNS   a segment whose mesh cannot be obtained is NOT cleared: it is listed and the verdict cannot be PASS while any remain among the models'/train sets.

Verdict: PASS only when 1-4 find nothing and 5 is empty.  FAIL lists every offender with its overlap fraction.  UNCLEARED when the only problem is unknown geometry.
Exit codes: 0 PASS, 1 FAIL, 2 UNCLEARED.  The report is written to <out>/leakage_report.json (and printed).  Named parameters are printed (D33).
usage: leakage_check.py [--splits PATH] [--models PATH] --out DIR
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
import split_guard as SG  # noqa: E402


def registry_checks(d):
    problems = []
    seen = {}
    for role in ("train", "eval", "submission"):
        lst = d.get(role, [])
        if len(lst) != len(set(lst)):
            problems.append(f"duplicate entries inside {role}")
        for s in lst:
            if s in seen and seen[s] != role:
                problems.append(f"{s} is in both {seen[s]} and {role}")
            seen.setdefault(s, role)
    return problems


def check(splits, models, out, work, fl, db):
    prot_names = list(splits.get("submission", [])) + list(splits.get("eval", []))
    rep = {"parameters": dict(EXCL_VOX=SG.EXCL_VOX, EPS_VOX=SG.EPS_VOX, CONTEXT_PX=SG.CONTEXT_PX), "protected": prot_names, "registry_problems": registry_checks(splits)}
    grids, unknown = {}, {}

    def grid(s):
        if s in grids or s in unknown:
            return grids.get(s)
        g, why = SG.load_grid(s, fl, db, work)
        if g:
            grids[s] = g
        else:
            unknown[s] = why
        return g
    prot = {s: grids[s] for s in prot_names if grid(s)}
    # 2. splits: train vs protected
    tr = {s: grids[s] for s in splits.get("train", []) if grid(s)}
    res, _ = SG.guard(prot, tr)
    rep["train_overlap"] = {s: {"overlap_frac": r["frac_overlapping"], "min_dist_kept_vox": r["min_dist_kept_vox"], "kept_cells": r["kept_cells"]} for s, r in res.items() if r["hit_cells"] > 0}
    rep["train_guarantee_holds"] = all(r["min_dist_kept_vox"] >= SG.EXCL_VOX for r in res.values() if r["kept_cells"])
    # 3. models: segments the existing models trained on
    msegs = [s for s in models.get("distill_rv2", []) if s not in prot_names]
    mg = {s: grids[s] for s in msegs if grid(s)}
    mres, _ = SG.guard(prot, mg, margin_cells=0)
    rep["model_training_overlap"] = {s: {"overlap_frac": r["frac_overlapping"], "min_dist_vox": r["min_dist_all_vox"]} for s, r in mres.items() if r["hit_cells"] > 0}
    rep["model_training_segments_checked"] = len(mg)
    # 4. labels
    labs = []
    for s in prot_names:
        for r in db.execute("SELECT version, who, source_prediction FROM ink_label WHERE seg=?", (s,)):
            labs.append({"seg": s, "version": r[0], "who": r[1], "source": r[2]})
    rep["labels_on_protected_segments"] = labs
    # 5. unknowns
    rep["unknown_geometry"] = {s: why for s, why in unknown.items()}
    rep["models_unresolved_pools"] = models.get("unresolved", {})
    bad = bool(rep["registry_problems"]) or bool(rep["train_overlap"] and not rep["train_guarantee_holds"]) or bool(rep["model_training_overlap"]) or bool(labs)
    rep["verdict"] = "FAIL" if bad else ("UNCLEARED" if (unknown or rep["models_unresolved_pools"]) else "PASS")
    Path(out).mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(Path(out) / "leakage_report.json", "w"), indent=1)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=str(ROOT / "docs/experiments/cnx_ink/splits.json"))
    ap.add_argument("--models", default=str(ROOT / "docs/experiments/cnx_ink/model_training_sets.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from vesuvius_pipeline import config
    from vesuvius_pipeline.db import pipeline_db
    fl = config.load(None)
    db = pipeline_db().connect(str(fl.pipeline_db))
    rep = check(json.load(open(a.splits)), json.load(open(a.models)), a.out, Path(a.out) / "_pulled", fl, db)
    print(f"parameters {rep['parameters']}; protected {len(rep['protected'])}")
    print(f"1 REGISTRY : {rep['registry_problems'] or 'ok'}")
    print(f"2 SPLITS   : {len(rep['train_overlap'])} train segments overlap protected surfaces (masked by the guard); guarantee holds: {rep['train_guarantee_holds']}")
    mo = rep["model_training_overlap"]
    print(f"3 MODELS   : {len(mo)} of {rep['model_training_segments_checked']} model-training segments overlap a protected surface" + ("" if not mo else ": " + ", ".join(f"{s} {100 * v['overlap_frac']:.1f}%" for s, v in sorted(mo.items()))))
    print(f"4 LABELS   : {len(rep['labels_on_protected_segments'])} label version(s) exist on protected segments {rep['labels_on_protected_segments'] or ''}")
    print(f"5 UNKNOWNS : {len(rep['unknown_geometry'])} segment(s) with no obtainable mesh; unresolved model pools {rep['models_unresolved_pools']}")
    print("VERDICT    :", rep["verdict"])
    return {"PASS": 0, "FAIL": 1, "UNCLEARED": 2}[rep["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
