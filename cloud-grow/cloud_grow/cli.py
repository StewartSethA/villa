"""`python -m cloud_grow <command>`: fetch | check-tools | seed | grow | pack | estimate."""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from . import cost as COST
from . import data_fetch as DF
from . import manifest as M
from . import pack as PK
from . import runner as R
from . import seeding as SD
from . import state as ST
from . import tools as T

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_POLICY = os.path.join(os.path.dirname(HERE), "config", "guard_policy.production.json")


def _ctx(cfg: R.RunConfig):
    wd = cfg.workdir
    os.makedirs(os.path.join(wd, "segments"), exist_ok=True)
    return ST.connect(os.path.join(wd, "state.sqlite")), os.path.join(wd, "segments")


def cmd_fetch(a) -> int:
    info = DF.scroll_info(a.scroll, a.scrolls_json)
    src = DF.DirSource(a.from_dir) if a.from_dir else DF.HttpS3Source(a.base_url or DF.DEFAULT_BASE)
    ct_name = a.volume_name or info.get("ct_zarr")
    pred_name = a.prediction_name or info.get("prediction_zarr")
    levels = {int(x) for x in a.levels.split(",")} if a.levels != "all" else None
    plans = []
    want = {"ct", "prediction", "grids"} if a.what == "all" else {a.what}
    if "ct" in want:
        if not ct_name:
            raise SystemExit(f"{a.scroll}: no CT zarr name in config/scrolls.json; pass --volume-name")
        pre = f"{a.scroll}/volumes/{ct_name}"
        plans.append(("ct", pre, DF.plan_zarr(src, pre, levels), os.path.join(a.dest, a.scroll, ct_name), ct_name))
    if "prediction" in want:
        if not pred_name:
            raise SystemExit(f"{a.scroll}: no prediction name; pass --prediction-name")
        pre = f"{a.scroll}/representations/predictions/surfaces/{pred_name}"
        plans.append(("prediction", pre, DF.plan_zarr(src, pre, None), os.path.join(a.dest, a.scroll, pred_name), pred_name))
    if "grids" in want:
        scan_ts = (ct_name or pred_name or "")[:14]
        gp = DF.find_grids_prefix(src, a.scroll, scan_ts)
        if not gp:
            raise SystemExit(f"{a.scroll}: no *.normal-grids under surfaces/; generate with vc_gen_normalgrids (CPU) from the prediction")
        objs, _ = src.list(gp)
        plans.append(("grids", gp.rstrip("/"), objs, os.path.join(a.dest, a.scroll, os.path.basename(gp.rstrip("/"))), os.path.basename(gp.rstrip("/"))))
    summ = [DF.preview(objs, dest, label) for label, _p, objs, dest, _n in plans]
    print(json.dumps({"preview": summ, "note": "nothing is downloaded until this is accepted (--yes); data transfer dominates short rentals"}, indent=1))
    if a.dry_run or not a.yes:
        print("[cloud-grow] preview only (pass --yes to download)")
        return 0
    if any(not s["fits"] for s in summ):
        print("[cloud-grow] REFUSING: a target does not fit free disk (+5 % and 20 GB scratch)", file=sys.stderr)
        return 2
    rc = 0
    for label, pre, objs, dest, name in plans:
        res = DF.fetch_objects(src, objs, pre.rstrip("/") + "/", dest, workers=a.workers)
        print(json.dumps({"label": label, **{k: v for k, v in res.items() if k != "failed"}, "n_failed": len(res["failed"])}))
        for f in res["failed"][:10]:
            print("  FAILED " + f, file=sys.stderr)
        if res["complete"] and label in ("ct", "prediction"):
            DF.write_meta(dest, a.scroll, info.get("voxel_um"), source=pre)
        rc |= 0 if res["complete"] else 1
    return rc


def cmd_check_tools(a) -> int:
    bd = T.kit_bin_dir(a.kit)
    ident = T.identify(bd)
    print(json.dumps(ident, indent=1))
    try:
        st = T.gate(ident, allow_unpinned=a.allow_unpinned)
    except T.ToolError as e:
        print("[cloud-grow] " + str(e), file=sys.stderr)
        return 1
    print("[cloud-grow] tool gate:", st)
    return 0


def cmd_seed(a) -> int:
    cfg = R.RunConfig.load(a.config)
    db, _ = _ctx(cfg)
    import zarr
    cov_dirs = [os.path.join(r, d) for r in _ckpt_roots(cfg) for d in os.listdir(r)] if os.path.isdir(os.path.join(cfg.workdir, "segments")) else []
    sup, src = SD.support_from_zarrs(cfg.ct_zarr, cfg.prediction_zarr or None)
    cov = SD.coverage_mask(sup.shape, cov_dirs)
    shape0 = zarr.open(cfg.ct_zarr, mode="r")["0"].shape if os.path.isdir(os.path.join(cfg.ct_zarr, "0")) else tuple(s * SD.FACTOR for s in sup.shape)
    excl = [(r["x"], r["y"], r["z"]) for r in db.execute("SELECT x,y,z FROM seed")]
    seeds = SD.propose(sup, cov, shape0, count=a.count, min_sep=a.min_sep, rng_seed=a.rng_seed, exclude=excl,
                       verify=SD.make_ct_level1_verifier(cfg.ct_zarr, cfg.ct_level_guard))
    for s in seeds:
        seg = SD.reserve(db, cfg.scroll, s)
        print(json.dumps({"seg": seg, **s.as_dict(), "support_source": src}))
    print(f"[cloud-grow] {len(seeds)} seed(s) of {a.count} requested", file=sys.stderr)
    return 0 if seeds else 1


def _ckpt_roots(cfg):
    root = os.path.join(cfg.workdir, "segments")
    out = []
    for seg in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        for r in sorted(os.listdir(os.path.join(root, seg))):
            p = os.path.join(root, seg, r)
            if r.startswith("r") and r[1:].isdigit() and os.path.isdir(p):
                out.append(p)
    return out


def cmd_grow(a) -> int:
    cfg = R.RunConfig.load(a.config)
    bd = T.kit_bin_dir(cfg.kit_bin or None)
    ident = T.identify(bd)
    pin = T.gate(ident, allow_unpinned=cfg.allow_unpinned or a.allow_unpinned)
    db, segroot = _ctx(cfg)
    settings = R.load_policy_into_state(db, cfg.policy_path or DEFAULT_POLICY)
    M.write_run_context(cfg.workdir, cfg, ident, bd, pin, settings, not_ported=R.NOT_PORTED)
    rows = db.execute("SELECT seg,x,y,z FROM seed WHERE scroll=? ORDER BY recorded_utc", (cfg.scroll,)).fetchall()
    if a.segs:
        rows = [r for r in rows if r["seg"] in set(a.segs)]
    todo = []
    for r in rows:
        done = db.execute("SELECT 1 FROM metric WHERE seg=? AND name='grow_outcome' LIMIT 1", (r["seg"],)).fetchone()
        if not done or a.regrow_resume:
            todo.append(r)
    print(f"[cloud-grow] {len(todo)} segment(s) to grow, parallel={a.parallel}, tool gate {pin}", flush=True)

    def one(r):
        d = ST.connect(os.path.join(cfg.workdir, "state.sqlite"))
        out = R.grow_segment(cfg, bd, d, r["seg"], os.path.join(segroot, r["seg"]), seed=(r["x"], r["y"], r["z"]))
        ST.record_metric(d, r["seg"], "grow_outcome", out.area_cm2, text=f"{out.status}:{out.why}", stage="grow")
        print(json.dumps({"seg": r["seg"], "status": out.status, "why": out.why, "area_cm2": out.area_cm2,
                          "verified_cm2_boxclaim": out.verified_cm2, "rounds": len(out.rounds)}), flush=True)
        return out.status != "failed"
    with ThreadPoolExecutor(max_workers=max(1, a.parallel)) as ex:
        ok = list(ex.map(one, todo))
    nfail = ok.count(False)
    print(f"[cloud-grow] done: {len(ok)} segment(s), {nfail} failed ({(100.0 * nfail / len(ok)) if ok else 0:.1f} %)", flush=True)
    return 0 if not nfail else 1


def cmd_pack(a) -> int:
    cfg = R.RunConfig.load(a.config)
    db, segroot = _ctx(cfg)
    ctx = json.load(open(os.path.join(cfg.workdir, "run_context.json")))
    res = PK.pack_all(segroot, cfg.scroll, ctx, a.out, db=db)
    for r in res:
        print(json.dumps(r))
    print(f"[cloud-grow] packed {len(res)} segment(s), {sum(r['bytes'] for r in res) / 1e6:.2f} MB total", file=sys.stderr)
    return 0


def cmd_estimate(a) -> int:
    print(json.dumps(COST.estimate(a.passmark_st, a.cores, a.target_cm2, a.usd_per_hour), indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cloud_grow")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="resume-safe data puller with size preview")
    f.add_argument("--scroll", required=True)
    f.add_argument("--dest", required=True)
    f.add_argument("--what", choices=["ct", "prediction", "grids", "all"], default="all")
    f.add_argument("--levels", default="1,2,3,4,5", help="CT pyramid levels, or 'all'. Seeding needs 4, guard needs 1; level 0 is huge and NOT needed")
    f.add_argument("--volume-name"); f.add_argument("--prediction-name")
    f.add_argument("--base-url"); f.add_argument("--from-dir", help="local dir laid out like the bucket (mirror / tests)")
    f.add_argument("--scrolls-json"); f.add_argument("--workers", type=int, default=16)
    f.add_argument("--yes", action="store_true"); f.add_argument("--dry-run", action="store_true")
    f.set_defaults(fn=cmd_fetch)
    c = sub.add_parser("check-tools"); c.add_argument("--kit"); c.add_argument("--allow-unpinned", action="store_true"); c.set_defaults(fn=cmd_check_tools)
    s = sub.add_parser("seed"); s.add_argument("--config", required=True); s.add_argument("--count", type=int, default=8)
    s.add_argument("--min-sep", type=float, default=250.0); s.add_argument("--rng-seed", type=int, default=None); s.set_defaults(fn=cmd_seed)
    g = sub.add_parser("grow"); g.add_argument("--config", required=True); g.add_argument("--parallel", type=int, default=1)
    g.add_argument("--segs", nargs="*"); g.add_argument("--allow-unpinned", action="store_true")
    g.add_argument("--regrow-resume", action="store_true", help="re-enter segments that already have an outcome (resumes, D3)")
    g.set_defaults(fn=cmd_grow)
    p = sub.add_parser("pack"); p.add_argument("--config", required=True); p.add_argument("--out", required=True); p.set_defaults(fn=cmd_pack)
    e = sub.add_parser("estimate"); e.add_argument("--passmark-st", type=float, required=True); e.add_argument("--cores", type=int, required=True)
    e.add_argument("--target-cm2", type=float, default=100.0); e.add_argument("--usd-per-hour", type=float); e.set_defaults(fn=cmd_estimate)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
