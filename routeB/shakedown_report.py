"""Shakedown verdict: reads ONLY artifacts under ROUTEB_HOME (never exit codes) and prints one PASS/FAIL/PENDING line per stage with seconds and GB.
usage: python -m routeB.shakedown_report [--scrolls A,B] [--json out.json]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .common import home


def jl(p: Path):
    try:
        return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    except OSError:
        return []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scrolls", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--with-ink", action="store_true", help="also grade CT-chunk fetch / snap / flatten / render / ink (off: cloud run is segment production only)")
    a = ap.parse_args(argv)
    H = home()
    D = H / "box8"
    rows = []

    def add(stage, scroll, ok, detail, secs=None, gb=None):
        rows.append({"stage": stage, "scroll": scroll, "verdict": "PASS" if ok is True else ("PENDING" if ok is None else "FAIL"), "seconds": secs, "gb": gb, "detail": detail})

    ev = jl(D / "events.jsonl")
    for sc in a.scrolls.split(","):
        fe = [e for e in ev if e.get("kind") == "fetched" and e.get("scroll") == sc]
        add("fetch(tracks+lasagna slab)", sc, bool(fe), f"{fe[-1]['gb']:.3f} GB over the network" if fe else "no 'fetched' event", gb=fe[-1]["gb"] if fe else None)
        runs = H / "runs" / sc
        fm = list(runs.glob("*/fit/.done.fit.json"))
        if fm:
            m = json.loads(fm[0].read_text())
            att = [x for e in [json.loads((D / "state" / f"{sc}.json").read_text())] for j in e["jobs"] for x in j["attempts"]] if (D / "state" / f"{sc}.json").exists() else []
            add("fit (incl. final satisfaction + meshes)", sc, True, f"satisfied_track_points={m.get('satisfied_track_points')} steps={m.get('steps')} attempts={[x.get('class') for x in att]} peak_rss_gb={[x.get('peak_rss_gb') for x in att]}",
                secs=sum(x.get("wall_s", 0) for x in att))
        else:
            add("fit (incl. final satisfaction + meshes)", sc, False, "no runs/*/fit/.done.fit.json")
        tm = list(runs.glob("*/.done.tiles.json"))
        if tm:
            m = json.loads(tm[0].read_text())
            add("tiles (tile_windings)", sc, m.get("n_tiles", 0) > 0, f"{m.get('n_windings')} windings, {m.get('n_tiles')} tiles, {m.get('tile_area_cm2')} cm2")
        else:
            add("tiles (tile_windings)", sc, False, "no .done.tiles.json")
        sd = D.parent / "out" / sc
        units = sorted(d for d in sd.iterdir() if d.is_dir() and (d / "DONE").exists()) if sd.is_dir() else []
        if (sd / "DONE").exists() and units:
            pms = [json.loads((u / "PAYLOAD.json").read_text()) for u in units]
            nf = sum(len(pm["files"]) for pm in pms)
            add("payload (DONE marker, md5 manifest, no checkpoints)", sc, not any(f["path"].endswith(".ckpt") for pm in pms for f in pm["files"]),
                f"{len(units)} unit(s), {nf} files", gb=sum(pm["total_bytes"] for pm in pms) / 1e9)
        else:
            add("payload (DONE marker, md5 manifest, no checkpoints)", sc, False, "no DONE")
        if a.with_ink:
            trs = [json.loads(p.read_text()) for p in runs.glob("*/tilework/*/tile_result.json")]
            if trs:
                t = trs[0]
                add("CT chunk fetch (sparse volume)", sc, (H / "assets" / sc / "volume").exists(), f"volume dir present; chunks ledger " + str(len(list((H / 'assets' / '.fetched').glob(f'{sc}:volume:chunks*')))))
                add("sheet snap", sc, "snap_s" in t, f"snap_s={t.get('snap_s')}", secs=t.get("snap_s"))
                add("lasagna flatten", sc, "flatten_s" in t and t.get("flat_valid", 0) >= 0.2, f"grid={t.get('flat_grid')} valid={t.get('flat_valid')}", secs=t.get("flatten_s"))
                add("render_gpu", sc, "render_s" in t and t.get("render_valid_frac", 0) > 0.01, f"shape={t.get('render_shape')} valid={t.get('render_valid_frac')} scale={t.get('render_scale')}", secs=t.get("render_s"))
                for fam, v in t.get("families", {}).items():
                    add(f"ink {fam}", sc, bool(v.get("png")), f"mean={v.get('mean')} p99={v.get('p99')}", secs=None)
                add("ink (all families, wall)", sc, all(v.get("png") for v in t.get("families", {}).values()), f"ink_s={t.get('ink_s')} tile wall_s={t.get('wall_s')}", secs=t.get("ink_s"))
            else:
                for st in ("CT chunk fetch (sparse volume)", "sheet snap", "lasagna flatten", "render_gpu", "ink (all families, wall)"):
                    add(st, sc, False, "no tile_result.json (tile stage did not complete)")
        ex = H / "export" / sc
        if a.with_ink:
          add("export (manifest + tarball)", sc, bool(list(ex.glob("*/manifest.json"))) and bool(list(ex.glob("*.tar"))), str([p.name for p in ex.glob("*.tar")]))
        add("pull-back (PULLED.json written by pull_box8.py from our side)", sc, True if (pl / "PULLED.json").exists() else None,
            "pulled+md5-verified" if (pl / "PULLED.json").exists() else "run the local pull command, then rerun this report")
    bl = jl(D / "budget" / "budget.jsonl")
    last = [x for x in bl if "spent" in x]
    out = {"rows": rows, "budget_last_spent_usd": last[-1]["spent"] if last else None, "budget_rows": len(bl)}
    w = max(len(r["stage"]) for r in rows)
    for r in rows:
        print(f"{r['verdict']:<8} {r['scroll']:<10} {r['stage']:<{w}} {'' if r['seconds'] is None else str(round(r['seconds'])) + ' s':<8} {'' if r['gb'] is None else format(r['gb'], '.3f') + ' GB':<10} {r['detail']}")
    print(f"budget ledger: last logged spent ${out['budget_last_spent_usd']} ({out['budget_rows']} rows)")
    n_fail = sum(1 for r in rows if r["verdict"] == "FAIL")
    print(f"SHAKEDOWN: {sum(1 for r in rows if r['verdict'] == 'PASS')} PASS, {n_fail} FAIL, {sum(1 for r in rows if r['verdict'] == 'PENDING')} PENDING")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
