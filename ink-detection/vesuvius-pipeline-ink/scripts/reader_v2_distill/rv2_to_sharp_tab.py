#!/usr/bin/env python3
"""Put Reader v2 dense distillation arms on the hub's Sharp students tab (ui/sharp_api.py reads
<eval>/crossgroup_<arm>.json summaries + <maps>/<tile>/<arm>.npy + <eval>/sharpness_all.json rows).

From rv2_eval.json: per arm a crossgroup_<arm>.json in the tab's format (summary[<original group>]["plain"] =
judged-region AUC mean + parent-bootstrap CI, the SAME region and face rule family as the ablation arms, so the
columns are comparable; near64 is added beside it as "near64"), and the arm's rise rows merged into
sharpness_all.json (rows matched by tile name, key <arm>_rise_um, summary untouched). Arm names are prefixed
'rv2d_' so they cannot collide with the ablation arms. Maps are copied by the caller (rsync).
Usage: rv2_to_sharp_tab.py rv2_eval.json <eval_dir> [--arms a,b]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rv2_lib as L  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("res"); ap.add_argument("eval_dir")
    ap.add_argument("--arms", default="")
    ap.add_argument("--prefix", default="rv2d_", help="tab arm-name prefix (conv retrain: 'conv_')")
    a = ap.parse_args()
    d = json.load(open(a.res))
    rows = d["rows"]
    arms = [x for x in (a.arms.split(",") if a.arms else d["arms"]) if x]
    groups = sorted({r["group_orig"] for r in rows})
    for arm in arms:
        summ = {}
        for g in groups:
            gr = [r for r in rows if r["group_orig"] == g]
            pj, pn = L.parent_boot(gr, f"{arm}__judged"), L.parent_boot(gr, f"{arm}__near64")
            if pj is None:
                continue
            summ[g] = {"n_tiles": len(gr), "n_parents": len({r["parent"] for r in gr}),
                       "plain": {"mean": pj["mean"], "ci95": pj["ci95"] or [pj["mean"], pj["mean"]]},
                       "near64": {"mean": pn["mean"], "ci95": pn["ci95"]} if pn else None,
                       "edgefp_in": L.parent_boot(gr, f"{arm}__edgefp_in"), "edge_out": L.parent_boot(gr, f"{arm}__edge_out")}
        out = os.path.join(a.eval_dir, f"crossgroup_{a.prefix}{arm}.json")
        json.dump({"source": os.path.abspath(a.res), "arm": arm, "meta": d["arms"].get(arm), "summary": summ,
                   "rows": [{"name": r["name"], "plain": r.get(f"{arm}__judged"), "near64": r.get(f"{arm}__near64"),
                             "face": r.get(f"{arm}__face")} for r in rows]}, open(out + ".tmp", "w"), indent=1)
        os.replace(out + ".tmp", out)
        print("wrote", out)
    sp = os.path.join(a.eval_dir, "sharpness_all.json")
    sh = json.load(open(sp))
    by = {r["name"]: r for r in rows}
    for r in sh["rows"]:
        src = by.get(r["name"])
        if src:
            for arm in arms:
                r[f"{a.prefix}{arm}_rise_um"] = src.get(f"{arm}__rise_um")
    json.dump(sh, open(sp + ".tmp", "w"), indent=1)
    os.replace(sp + ".tmp", sp)
    print("merged rise rows into", sp)


if __name__ == "__main__":
    main()
