#!/usr/bin/env python3
"""Score Reader v2 dense students against the teacher and the production ink9um_student on every labelled subject.

Per subject (ink9um_tta_prep layout, 54 tiles / 7 groups -- the sharp_student ablation's own set) and per arm:
  * near64 AUC (judged AND <= 64 px of labelled ink; labels are positive-only, D23) -- the PRIMARY metric;
  * judged AUC (the region the earlier crossgroup tables used, for comparability);
  * edge rise, um: 10-90 % rise of the mean prediction across the labelled ink edge (ink9um_sharpness.py method,
    +-15 px signed-distance profile); the label's own edge reads ~15 um;
  * face: oracle per tile per arm on near64 AUC (CLAUDE.md D5: oracle-picked sweeps are acceptable; stated).
Arms:
  --ckpt name=path      reader_v2_dense checkpoints, run here, both faces; GPU time -> MVox/s (17 layers x H x W)
  --ref  name=spec      saved maps: 'rv2maps:<dir>' = <dir>/<subj>/base__{fwd,rev}.npy uint8 (Reader v2 plain, both
                        faces); 'arm:<dir>:<arm>' = <dir>/<subj>/<arm>.npy float32 (one face, already chosen by that
                        model's own judged AUC -- e.g. the production ink9um_student = arm0)
  colmean               the untrained baseline: mean of the 17 layers (no sign flip, no face choice)
Groups: S1 / S4 / S5 / other open scrolls (all SEEN by Reader v2) and PHerc0841 / Kaggle fragments (UNSEEN).
CIs: bootstrap over PARENT segments within a group (2,000 resamples); paired deltas are bootstrapped the same way.
Usage: rv2_eval.py <subj_root> --src <repo>/src --ckpt a=... --ref teacher=rv2maps:... --out res.json [--save-maps D]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rv2_lib as L  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subj_root")
    ap.add_argument("--src", required=True)
    ap.add_argument("--ckpt", action="append", default=[])
    ap.add_argument("--ref", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-maps", default="")
    ap.add_argument("--device", default="cuda", help="cpu when the shared card is full (MVox/s then measures the CPU)")
    ap.add_argument("--tile", type=int, default=2048)
    ap.add_argument("--pairs", default="", help="comma list a:b -> paired deltas a minus b (default: every ckpt arm "
                    "vs every ref arm)")
    a = ap.parse_args()
    sys.path.insert(0, a.src)
    from vesuvius_pipeline.stages.ink_models import ink9um_student as S
    from vesuvius_pipeline.stages.ink_models import reader_v2_dense as R
    subs = L.load_subjects(a.subj_root, S.select_layers)
    print(f"[eval] {len(subs)} subjects", flush=True)

    rows = []
    arms_meta = {}
    for s in subs:
        rows.append({"name": s["name"], "parent": s["parent"], "group": s["group"], "scroll": s["scroll"],
                     "seg_seen": s["seg_seen"], "confirmed_neg": s["confirmed_neg"],
                     "group_orig": s["meta"].get("group"),
                     "um": s["um"], "near64_px": int(s["near"].sum()), "judged_px": int(s["judged"].sum()),
                     "ink_near64_px": int((s["ink"] & s["near"]).sum()), "label_edge_um": L.label_edge_um(s)})
    byname = {r["name"]: r for r in rows}

    from scipy.ndimage import distance_transform_edt as _edt
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hires_ink"))
    import metrics as MET

    def edge_metrics(s, p, other):
        """Segment-edge false positives + false strokes (2026-10-06). edgefp_in: mean within-tile percentile of the
        map on judged non-ink > 16 px from ink within 4 px of the surface edge MINUS the same > 48 px from it (0 = no
        edge effect; inside the rendered surface, i.e. what production keeps). edge_out: mean probability on the 6 px
        OUTSIDE the surface minus the judged-interior mean (raw map; production zeroes it). thr90 = the arm's own
        threshold at 90 % recall of labelled ink near64 on this tile; fs_neg = letter-like components per cm2 on
        CONFIRMED negatives (S4 letter boxes, fully annotated fragments) at thr90; fs_rev = the same on the
        reversed face over the whole surface."""
        out = {}
        J = s["judged"]
        ref = np.sort(p[J]); rk = np.searchsorted(ref, p) / max(1, len(ref))
        far = J & ~s["ink"] & (s["dink"] > 16)
        a_ = far & (s["dedge"] <= 4); b_ = far & (s["dedge"] > 48)
        out["edgefp_in"] = float(rk[a_].mean() - rk[b_].mean()) if a_.sum() > 50 and b_.sum() > 50 else None
        if not s["valid"].all():
            ring = (~s["valid"]) & (_edt(~s["valid"]) <= 6)
            out["edge_out"] = float(p[ring].mean() - p[J & s["valid"]].mean()) if ring.sum() > 50 else None
        else:
            out["edge_out"] = None
        inkn = s["ink"] & s["near"]
        thr = float(np.percentile(p[inkn], 10)) if inkn.sum() > 100 else float("nan")
        out["thr90"] = thr
        if s["confirmed_neg"]:
            fs = MET.false_strokes(p, J & ~s["ink"] & (s["dink"] > 8), thr, um_per_px=s["um"])
            out["fs_neg_n"] = fs["n_false_strokes"]; out["fs_neg_cm2"] = fs["cm2"]
        if other is not None:
            fs = MET.false_strokes(other, s["valid"], thr, um_per_px=s["um"])
            out["fs_rev_n"] = fs["n_false_strokes"]; out["fs_rev_cm2"] = fs["cm2"]
        return out

    def put(arm, s, p, face, other=None):
        sc = L.score_map(s, p)
        r = byname[s["name"]]
        r[f"{arm}__near64"] = sc["near64"]; r[f"{arm}__judged"] = sc["judged"]; r[f"{arm}__rise_um"] = sc["rise_um"]
        r[f"{arm}__rise40_um"] = sc["rise40_um"]
        r[f"{arm}__face"] = face
        for k, v in edge_metrics(s, p, other).items():
            r[f"{arm}__{k}"] = v
        if a.save_maps:
            from PIL import Image
            d = os.path.join(a.save_maps, s["name"]); os.makedirs(d, exist_ok=True)
            np.save(os.path.join(d, f"{arm}.npy"), p.astype(np.float32))
            Image.fromarray(np.clip(p * 255 + 0.5, 0, 255).astype(np.uint8)).save(os.path.join(d, f"{arm}.png"))

    def best_face(s, maps):
        best = None
        for face, p in maps.items():
            au = L.auc(p[s["near"]], s["ink"][s["near"]])
            if best is None or (np.isfinite(au) and au > best[0]):
                best = (au, face, p)
        return best[1], best[2]

    # ---- reference maps ----------------------------------------------------------------------------------
    for spec in a.ref:
        arm, how = spec.split("=", 1)
        kind, rest = how.split(":", 1)
        arms_meta[arm] = {"kind": kind, "source": rest}
        for s in subs:
            try:
                if kind == "rv2maps":
                    maps = {f: np.load(os.path.join(rest, s["name"], f"base__{f}.npy")).astype(np.float32) / 255.0
                            for f in ("fwd", "rev")}
                    face, p = best_face(s, maps)
                    other = maps["rev" if face == "fwd" else "fwd"]
                elif kind == "arm":
                    d, an = rest.rsplit(":", 1)
                    p = np.load(os.path.join(d, s["name"], f"{an}.npy")).astype(np.float32); face = "own"; other = None
                else:
                    raise SystemExit(f"unknown ref kind {kind}")
                assert p.shape == s["ink"].shape, (p.shape, s["ink"].shape)
                put(arm, s, p, face, other)
            except (FileNotFoundError, AssertionError) as e:
                print(f"[eval] {arm} {s['name']}: MISSING {type(e).__name__} {e}", flush=True)
        print(f"[eval] ref {arm} scored", flush=True)
    arms_meta["colmean"] = {"kind": "untrained", "source": "mean of the 17 centred layers / 255"}
    for s in subs:
        put("colmean", s, s["x"].astype(np.float32).mean(0) / 255.0, "n/a")

    # ---- student checkpoints -----------------------------------------------------------------------------
    import torch
    torch.backends.cudnn.benchmark = True
    for spec in a.ckpt:
        arm, path = spec.split("=", 1)
        model, ck = R.load(path, a.device)
        norm = ck["config"]["norm"]; halo = int(ck["config"]["halo"])
        arms_meta[arm] = {"kind": "reader_v2_dense", "ckpt": path, "run": ck.get("run"), "step": ck.get("step"),
                          "params": ck.get("params"), "rf_total": ck.get("rf_total"), "norm": norm,
                          "teacher": ck.get("teacher"), "targets_desc": ck.get("targets_desc"), "loss": ck.get("loss")}
        # warm-up (cuDNN autotune) outside the timing
        R.predict_array(model, [subs[0]["x"][j] for j in range(17)], norm, a.device, tile=a.tile, halo=halo)
        gs = 0.0; vox = 0
        for s in subs:
            maps = {}
            for face in ("fwd", "rev"):
                xx = s["x"] if face == "fwd" else s["x"][::-1]
                if a.device == "cuda": torch.cuda.synchronize()
                t0 = time.time()
                lg = R.predict_array(model, [xx[j] for j in range(17)], norm, a.device, tile=a.tile, halo=halo)
                if a.device == "cuda": torch.cuda.synchronize()
                gs += time.time() - t0
                vox += lg.size * 17
                maps[face] = 1.0 / (1.0 + np.exp(-lg))
            face, p = best_face(s, maps)
            put(arm, s, p, face, other=maps["rev" if face == "fwd" else "fwd"])
        arms_meta[arm]["MVox_per_s"] = round(vox / 1e6 / gs, 1)
        arms_meta[arm]["timing"] = {"vox": vox, "gpu_s": round(gs, 2), "gpu": torch.cuda.get_device_name(0) if a.device == "cuda" else "cpu",
                                    "note": "end-to-end predict_array incl. host->device, normaliser, fp16 net, "
                                            "device->host; tiles <= 2048 px (small subjects: launch-bound)"}
        print(f"[eval] ckpt {arm}: {arms_meta[arm]['MVox_per_s']} MVox/s on {arms_meta[arm]['timing']['gpu']}",
              flush=True)
        del model
        if a.device == "cuda": torch.cuda.empty_cache()

    # ---- summaries -----------------------------------------------------------------------------------------
    arms = list(arms_meta)
    groups = sorted({s["group"] for s in subs})
    ck_arms = [s.split("=", 1)[0] for s in a.ckpt]
    ref_arms = [s.split("=", 1)[0] for s in a.ref]
    pairs = [tuple(p.split(":")) for p in a.pairs.split(",") if p] or \
        [(x, y) for x in ck_arms for y in ref_arms + [c for c in ck_arms if c != x]]
    for r in rows:
        for x, y in pairs:
            for m in ("near64", "rise_um", "rise40_um"):
                vx, vy = r.get(f"{x}__{m}"), r.get(f"{y}__{m}")
                if vx is not None and vy is not None:
                    r[f"d__{x}__{y}__{m}"] = vx - vy
    summary = {}
    SEG = ["SEGMENT-SEEN by Reader v2", "SEGMENT-UNSEEN (scroll seen)", "SEGMENT-UNSEEN (all)"]
    for g in groups + ["UNSEEN (PHerc0841 + fragments)"] + SEG + ["ALL"]:
        if g == "ALL":
            gs_ = subs
        elif g == SEG[0]:
            gs_ = [s for s in subs if s["seg_seen"] == "segment-seen"]
        elif g == SEG[1]:
            gs_ = [s for s in subs if s["seg_seen"] == "segment-unseen (scroll seen)"]
        elif g == SEG[2]:
            gs_ = [s for s in subs if s["seg_seen"] != "segment-seen"]
        elif g.startswith("UNSEEN"):
            gs_ = [s for s in subs if "UNSEEN" in s["group"]]
        else:
            gs_ = [s for s in subs if s["group"] == g]
        gr = [byname[s["name"]] for s in gs_]
        e = {"size": L.size_line(gs_)}
        le = [r["label_edge_um"] for r in gr if r["label_edge_um"] is not None]
        e["label_edge_um_p50"] = float(np.median(le)) if le else None
        for arm in arms:
            e[arm] = {"near64": L.parent_boot(gr, f"{arm}__near64"), "judged": L.parent_boot(gr, f"{arm}__judged")}
            for m in ("rise_um", "rise40_um"):
                rs = [r[f"{arm}__{m}"] for r in gr if r.get(f"{arm}__{m}") is not None]
                e[arm][m] = ({"p10": float(np.percentile(rs, 10)), "p50": float(np.percentile(rs, 50)),
                              "p90": float(np.percentile(rs, 90)), "n": len(rs),
                              "n_missing": len(gr) - len(rs)} if rs else None)
            for m in ("edgefp_in", "edge_out"):
                e[arm][m] = L.parent_boot(gr, f"{arm}__{m}")
            nn = sum(r.get(f"{arm}__fs_neg_n") or 0 for r in gr); cn = sum(r.get(f"{arm}__fs_neg_cm2") or 0 for r in gr)
            nr = sum(r.get(f"{arm}__fs_rev_n") or 0 for r in gr); cr = sum(r.get(f"{arm}__fs_rev_cm2") or 0 for r in gr)
            e[arm]["false_strokes"] = {"confirmed_neg_per_cm2": nn / cn if cn else None, "confirmed_neg_n": nn,
                                       "confirmed_neg_cm2": round(cn, 3), "reversed_face_per_cm2": nr / cr if cr else None,
                                       "reversed_face_n": nr, "reversed_face_cm2": round(cr, 3),
                                       "threshold": "arm's own 90 % recall of labelled ink near64, per tile"}
        e["paired"] = {f"{x}-{y}": {"near64": L.parent_boot(gr, f"d__{x}__{y}__near64"),
                                    "rise_um": L.parent_boot(gr, f"d__{x}__{y}__rise_um"),
                                    "rise40_um": L.parent_boot(gr, f"d__{x}__{y}__rise40_um")} for x, y in pairs}
        summary[g] = e
    json.dump({"arms": arms_meta, "rows": rows, "summary": summary, "near_px": L.NEAR_PX,
               "subj_root": os.path.abspath(a.subj_root)}, open(a.out, "w"), indent=1)

    def fm(v):
        if not v:
            return "-"
        ci = v.get("ci95")
        return f"{v['mean']:.3f}" + (f" [{ci[0]:.3f},{ci[1]:.3f}]" if ci else " (1 parent)")
    for g, e in summary.items():
        z = e["size"]
        print(f"\n== {g}: {z['n_tiles']} tiles / {z['n_parents']} parents, {z['MPix_near64']} MPix near64, "
              f"{z['cm2_near64']} cm2 near64, {z['cm2_ink']} cm2 ink; label edge p50 {e['label_edge_um_p50']} um")
        for arm in arms:
            ri, r4 = e[arm]["rise_um"], e[arm]["rise40_um"]
            fs_ = e[arm]["false_strokes"]
            print(f"  {arm:28s} edgefp_in {fm(e[arm]['edgefp_in'])} edge_out {fm(e[arm]['edge_out'])} false strokes/cm2 "
                  f"conf-neg {fs_['confirmed_neg_per_cm2']} (n {fs_['confirmed_neg_n']}, {fs_['confirmed_neg_cm2']} cm2) "
                  f"rev-face {fs_['reversed_face_per_cm2']} (n {fs_['reversed_face_n']})")
            print(f"  {arm:28s} near64 {fm(e[arm]['near64'])}  judged {fm(e[arm]['judged'])}  rise15 p10/50/90 "
                  + (f"{ri['p10']:.0f}/{ri['p50']:.0f}/{ri['p90']:.0f}" if ri else "-") + "  rise40 p10/50/90 "
                  + (f"{r4['p10']:.0f}/{r4['p50']:.0f}/{r4['p90']:.0f} um (n {r4['n']})" if r4 else "-"))
        for k, v in e["paired"].items():
            print(f"  d {k:40s} near64 {fm(v['near64'])}  rise15 {fm(v['rise_um'])}  rise40 {fm(v['rise40_um'])}")
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
