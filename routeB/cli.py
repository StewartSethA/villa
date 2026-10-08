"""routeB_run.sh driver.  Stages per scroll: fetch -> (per stripe) fit -> tiles -> volume+snap+flatten+render+ink per tile -> export.
Each stage fails loud with a message naming the stage; a failed scroll/stripe does not stop the others (exit status 1 at the end lists them)."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import tarfile
import time
import traceback
from pathlib import Path

from . import fit as FIT
from . import manifest as MAN
from . import tile_chain as TC
from .common import ROOT, UP_HI, UP_LO, StageError, env_python, home, is_done, mark_done, run, say, spec, stripes, tail


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def stage_fetch(scroll: str, z0: int, z1: int, a) -> None:
    sys.path.insert(0, str(ROOT / "deploy_common"))
    import fetch_assets as FA
    try:
        m = MAN.build(scroll, z0, z1, with_crossings=a.with_crossings, full_lasagna=a.full_lasagna)
    except ValueError as e:
        raise StageError(f"fetch: {e}")
    t0 = time.time()
    r = FA.fetch(m, home() / "assets", log=lambda x: say(x, "fetch"))
    say(f"{scroll}: fetched {r['bytes_net'] / 1e9:.3f} GB over the network in {time.time() - t0:.0f} s", "fetch")
    if not r["ok"]:
        raise StageError("fetch: " + "; ".join(r["failed"]))


def stage_tiles(scroll: str, tag: str, run_dir: Path, a) -> Path:
    sp = spec(scroll)
    out = run_dir / "tiled"
    if is_done(run_dir, "tiles"):
        return out
    meshes_root = list((run_dir / "fit" / "out").rglob("meshes"))
    if not meshes_root:
        raise StageError(f"tiles: no meshes under {run_dir}/fit/out")
    cand = sorted(meshes_root[0].glob("*/w0*_spliced_*")) or sorted(meshes_root[0].glob("w0*_spliced_*"))
    if not cand:
        raise StageError(f"tiles: no w0NN_spliced_* mesh dirs under {meshes_root[0]}")
    mesh = cand[0].parent
    cmd = [env_python(), ROOT / "routeB" / "tile_windings.py", mesh, "--scroll", scroll, "--fit", tag, "--prefix", f"{scroll}_{tag}", "--out", out,
           "--voxel-um", sp["voxel_um"], "--target-cm2", a.target_cm2, "--rows", a.rows, "--min-material", "0"]
    if a.windings:
        lo, hi = a.windings.split(":")
        cmd += ["--wind-min", lo, "--wind-max", hi]
    log = run_dir / "tiles.log"
    rc = run(cmd, log=log)
    if rc != 0 or not (out / "manifest.json").exists():
        raise StageError(f"tiles: tile_windings.py rc={rc}\n{tail(log)}")
    m = json.loads((out / "manifest.json").read_text())
    mark_done(run_dir, "tiles", n_windings=m["n_windings"], n_tiles=m["n_tiles"], tile_area_cm2=m["tile_area_cm2"])
    say(f"{scroll}/{tag}: {m['n_windings']} windings, {m['n_tiles']} tiles, {m['tile_area_cm2']:.1f} cm2", "tiles")
    return out


def choose_tiles(tiled: Path, a) -> list[Path]:
    m = json.loads((tiled / "manifest.json").read_text())
    ts = sorted(m["tiles"], key=lambda t: t["seg"])
    if a.tile_stride > 1:
        ts = ts[::a.tile_stride]
    if a.max_tiles:
        ts = ts[: a.max_tiles]
    return [tiled / "tiles" / t["seg"] for t in ts]


def stage_export(scroll: str, tag: str, run_dir: Path, fit_info: dict, results: list[dict], a, errors: list[str]) -> Path:
    ex = home() / "export" / scroll / tag
    (ex / "ink").mkdir(parents=True, exist_ok=True)
    files = []
    for r in results:
        for fam, v in r["families"].items():
            if v.get("png"):
                src = run_dir / "tilework" / r["tile"] / v["png"]
                dst = ex / "ink" / f"{r['tile']}__{fam}.png"
                dst.write_bytes(src.read_bytes())
                files.append(dst)
    mods = {p.name: md5(p) for p in sorted((ROOT / "models" / "var" / "models").glob("*.ckpt"))}
    man = {"scroll": scroll, "stripe": tag, "z": fit_info.get("z"), "fit": fit_info, "families": a.families.split(","), "tiles": results,
           "errors": errors, "model_md5": mods, "code": {"routeB": md5(ROOT / "routeB" / "cli.py"), "fit_spiral.py": md5(ROOT / "spiral-fitting" / "fit_spiral.py")},
           "command": " ".join(sys.argv), "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "tile_manifest": json.loads((run_dir / "tiled" / "manifest.json").read_text()).get("params") if (run_dir / "tiled" / "manifest.json").exists() else None}
    (ex / "manifest.json").write_text(json.dumps(man, indent=1))
    (ex / "FILES.txt").write_text("\n".join(str(p.relative_to(ex)) for p in files + [ex / "manifest.json"]) + "\n")
    tar = ex.parent / f"routeB_{scroll}_{tag}.tar"
    with tarfile.open(tar, "w") as t:
        for p in files + [ex / "manifest.json", ex / "FILES.txt"]:
            t.add(p, arcname=f"{scroll}/{tag}/{p.relative_to(ex)}")
    tot = sum(p.stat().st_size for p in files) / 1e6
    say(f"{scroll}/{tag}: exported {len(files)} ink PNGs ({tot:.1f} MB) -> {ex}; tarball {tar} ({tar.stat().st_size / 1e6:.1f} MB); "
        f"pull with: rsync -a <host>:{ex}/ ./ ", "export")
    return ex


def run_scroll(scroll: str, a) -> list[str]:
    fails: list[str] = []
    sp = spec(scroll)
    z0 = a.z0 if a.z0 is not None else UP_LO
    z1 = a.z1 if a.z1 is not None else UP_HI
    wins = stripes(a.stripe_width, z0, z1, a.overlap)
    if a.only_stripe:
        wins = [w for w in wins if w[0] == a.only_stripe or str(w[1]) == a.only_stripe]
    say(f"{scroll}: stripe width {a.stripe_width} -> {len(wins)} window(s): {[(t, s, e) for t, s, e in wins]}")
    if "fetch" in a.stages:
        try:
            stage_fetch(scroll, min(w[1] for w in wins), max(w[2] for w in wins), a)
        except Exception as e:                                   # noqa: BLE001
            fails.append(f"{scroll}: {e}")
            say(f"STAGE FAILED {e}", "fetch")
            return fails
    for tag, s, e in wins:
        run_dir = home() / "runs" / scroll / tag
        label = f"{scroll}/{tag}"
        try:
            fit_info = {"z": [s, e]}
            if "fit" in a.stages:
                FIT.run_fit(scroll, tag, s, e, a.steps, a.sense, a.shell, a.umbilicus, a.gpu, json.loads(a.fit_overrides) if a.fit_overrides else None,
                            workdir=run_dir)
                fit_info.update(json.loads((run_dir / "fit" / ".done.fit.json").read_text()))
            if "tiles" in a.stages:
                tiled = stage_tiles(scroll, tag, run_dir, a)
            results, errs = [], []
            if "ink" in a.stages:
                tiled = run_dir / "tiled"
                tiles = choose_tiles(tiled, a)
                say(f"{label}: {len(tiles)} tile(s) to flatten/render/ink: {[t.name for t in tiles][:6]}{'...' if len(tiles) > 6 else ''}")
                vol = TC.ensure_volume(scroll, tiles)
                for t in tiles:
                    try:
                        results.append(TC.process_tile(scroll, t, run_dir / "tilework" / t.name, vol, a.families.split(","), a.snap, a.gpu, a.device))
                        say(f"{t.name}: ok {json.dumps({k: v for k, v in results[-1].items() if k.endswith('_s')})}", "tile")
                    except StageError as ex:
                        errs.append(f"{t.name}: {ex}")
                        say(f"TILE FAILED {t.name}: {ex}", "tile")
                if errs:
                    fails.append(f"{label}: {len(errs)}/{len(tiles)} tiles failed; first: {errs[0][:300]}")
                    say(f"tile failure rate {len(errs)}/{len(tiles)}", "tile")
                if "export" in a.stages:
                    stage_export(scroll, tag, run_dir, fit_info, results, a, errs)
        except StageError as e:
            fails.append(f"{label}: {e}")
            say(f"STAGE FAILED: {e}", "stripe")
        except Exception as e:                                    # noqa: BLE001 - a bug is reported, the next stripe still runs
            fails.append(f"{label}: BUG {type(e).__name__}: {e}")
            traceback.print_exc()
    return fails


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--mode" in argv:                                   # --mode box8 -> fits-and-segments-only scheduler (routeB/box8.py); --mode pipeline = default
        i = argv.index("--mode")
        mode = argv[i + 1] if i + 1 < len(argv) else ""
        rest = argv[:i] + argv[i + 2:]
        if mode == "box8":
            from . import box8
            return box8.main(rest)
        if mode != "pipeline":
            print(f"ROUTEB FAIL args: unknown --mode {mode!r} (pipeline|box8)", file=sys.stderr)
            return 2
        argv = rest
    ap = argparse.ArgumentParser(prog="routeB_run.sh")
    ap.add_argument("--scrolls", required=True, help="comma list, e.g. PHerc0211,PHerc0125")
    ap.add_argument("--stripe-width", default="full", help="4500 | 7500 | 13500 | full | any integer (z slices); default full")
    ap.add_argument("--z0", type=int, default=None, help=f"first z (default {UP_LO}, the upstream tracks' lower bound)")
    ap.add_argument("--z1", type=int, default=None, help=f"end z (default {UP_HI})")
    ap.add_argument("--overlap", type=int, default=1500, help="z overlap between consecutive stripes")
    ap.add_argument("--only-stripe", default=None, help="run just this stripe tag (e.g. s7500) or its start z")
    ap.add_argument("--steps", type=int, default=30000, help="fit steps (production 30000)")
    ap.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    ap.add_argument("--device", default="cuda", help="lasagna flatten device (cuda|cpu)")
    ap.add_argument("--families", default=TC.DEFAULT_FAMILIES)
    ap.add_argument("--snap", default="recto", choices=["recto", "centre", "off"])
    ap.add_argument("--sense", default=None, help="override spiral_outward_sense (CW|ACW)")
    ap.add_argument("--shell", type=int, default=None, help="override shell_outer_winding_idx")
    ap.add_argument("--umbilicus", default=None, help="override umbilicus.json path")
    ap.add_argument("--fit-overrides", default=None, help="JSON merged over the fit config overrides")
    ap.add_argument("--stages", default="fetch,fit,tiles,ink,export")
    ap.add_argument("--windings", default=None, help="lo:hi winding index range to tile (smoke)")
    ap.add_argument("--target-cm2", type=float, default=26.0)
    ap.add_argument("--rows", type=int, default=400)
    ap.add_argument("--max-tiles", type=int, default=0, help="0 = all tiles")
    ap.add_argument("--tile-stride", type=int, default=1, help="take every Nth tile (spread a smoke subset)")
    ap.add_argument("--with-crossings", action="store_true")
    ap.add_argument("--full-lasagna", action="store_true", help="fetch the whole lasagna group instead of the z-slab")
    ap.add_argument("--smoke", action="store_true", help="z 9000-9500, 1500 steps, 1 tile, 2 windings (proof run)")
    a = ap.parse_args(argv)
    a.stages = a.stages.split(",")
    if a.smoke:
        a.z0, a.z1 = a.z0 or 9000, a.z1 or 9500
        a.steps = 1500 if a.steps == 30000 else a.steps
        a.max_tiles = a.max_tiles or 1
        a.windings = a.windings or "60:62"
        a.target_cm2 = min(a.target_cm2, 3.0)
    t0 = time.time()
    fails: list[str] = []
    for s in a.scrolls.split(","):
        fails += run_scroll(s.strip(), a)
    say(f"DONE in {time.time() - t0:.0f} s; {len(fails)} failure(s)")
    for f in fails:
        print("FAILED:", f, file=sys.stderr)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
