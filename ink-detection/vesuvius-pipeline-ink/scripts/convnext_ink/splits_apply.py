"""Apply the operator's additions to docs/experiments/cnx_ink/splits.json -- the ONE way segments join the ConvNeXt ink registry.

  python splits_apply.py --train A B C --submission D E [--splits PATH] [--work DIR]

Rules (user, 2026-10-08):
  * "Override my decision and make it training if it overlaps an existing training segment": a segment named for SUBMISSION that overlaps ANY training
    segment (3-D surface distance < EXCL_VOX at any vertex, split_guard's definition) is registered as TRAIN instead, with the reason recorded. A
    submission must not overlap training data (prize criterion), and a stripe-width variant of a trained surface does.
  * A name given for both train and submission is decided by the same rule: submission only if it overlaps no OTHER training segment.
  * Names given with no label join the previous label (the caller passes them as --train).
  * Everything is idempotent; moves are logged under "moves" with the overlapping partner and the fraction.
Afterwards run split_guard.py: it masks the cells of TRAIN segments that lie within EXCL_VOX (+ context margin) of the remaining submission/eval surfaces.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
import split_guard as SG  # noqa: E402


def overlaps(prot_seg, train_segs, grids):
    """{train_seg: fraction of ITS valid cells within EXCL_VOX of prot_seg's surface} for every train seg with any hit."""
    G, V = grids[prot_seg]
    res, _ = SG.guard({prot_seg: (G, V)}, {t: grids[t] for t in train_segs if t in grids and t != prot_seg}, excl=SG.EXCL_VOX, margin_cells=0)
    return {t: r["frac_overlapping"] for t, r in res.items() if r["hit_cells"] > 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="*", default=[])
    ap.add_argument("--submission", nargs="*", default=[])
    ap.add_argument("--train-file", default=None, help="bulk TRAIN list (one segment per line). A segment currently registered as SUBMISSION is HELD (reported, not moved): a bulk paste must not silently undo explicit submission choices.")
    ap.add_argument("--splits", default=str(HERE.parents[1] / "docs" / "experiments" / "cnx_ink" / "splits.json"))
    ap.add_argument("--work", default="guard_pulled")
    a = ap.parse_args()
    from vesuvius_pipeline import config
    from vesuvius_pipeline.db import pipeline_db
    fl = config.load(None)
    db = pipeline_db().connect(str(fl.pipeline_db))
    d = json.load(open(a.splits))
    d.setdefault("moves", [])
    existing_train = list(d["train"])
    held = []
    if a.train_file:
        for line in open(a.train_file):
            seg = line.strip()
            if not seg:
                continue
            if seg in d["submission"]:
                held.append(seg)
            elif seg not in d["train"]:
                d["train"].append(seg)
        print(f"  bulk train file: {len(d['train']) - len(existing_train)} added, {len(held)} HELD because they are registered as submission: {held}")
    for s in a.train:
        if s in d["submission"]:
            d["submission"].remove(s)
        if s not in d["train"]:
            d["train"].append(s)
    grids, miss = {}, {}
    # geometry is only needed to TEST submission candidates against training; pure training additions skip the (slow) mesh loads
    for s in (set(d["train"]) | set(d["submission"]) | set(a.submission)) if a.submission else ():
        g, why = SG.load_grid(s, fl, db, a.work)
        if g:
            grids[s] = g
        else:
            miss[s] = why
    for s in a.submission:
        if s in miss:
            print(f"  {s}: NO GEOMETRY ({miss[s]}): cannot test overlap, NOT registered"); continue
        others = [t for t in d["train"] if t != s]
        ov = overlaps(s, others, grids)
        if ov:
            top = max(ov.items(), key=lambda kv: kv[1])
            if s in d["submission"]:
                d["submission"].remove(s)
            if s not in d["train"]:
                d["train"].append(s)
            d["moves"].append({"seg": s, "from": "submission", "to": "train", "overlaps_train": {k: round(v, 4) for k, v in ov.items()}, "rule": "user 2026-10-08: make it training if it overlaps an existing training segment"})
            print(f"  {s}: overlaps {len(ov)} training segment(s) (max {top[0]} {100 * top[1]:.1f} % of its cells) -> registered as TRAIN")
        else:
            if s in d["train"]:
                d["train"].remove(s)
            if s not in d["submission"]:
                d["submission"].append(s)
            print(f"  {s}: overlaps no training segment -> SUBMISSION")
    d["source"]["train"] = f"user 2026-10-08 'train:' messages ({len(d['train'])} segments so far; bare names with no label were taken as train)"
    json.dump(d, open(a.splits, "w"), indent=2)
    import time
    with open(Path(a.splits).with_name("splits.history.jsonl"), "a") as h:        # append-only: what was asked, when, and what the rules did with it
        h.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "asked_train": a.train, "asked_submission": a.submission,
                            "result": {"train": len(d["train"]), "submission": len(d["submission"]), "eval": len(d.get("eval", []))},
                            "moves_total": len(d["moves"]), "no_geometry": miss, "train_file": a.train_file, "held_submission": held}) + "\n")
    print(f"registry: {len(d['train'])} train, {len(d['submission'])} submission, {len(d.get('eval', []))} eval; missing geometry: {miss or 'none'}")


if __name__ == "__main__":
    main()
