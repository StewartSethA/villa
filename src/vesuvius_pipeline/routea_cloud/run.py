"""The ONE driver behind ./routeA_run.sh -- same code on a fleet host and on a rented box.

  python -m vesuvius_pipeline.routea_cloud.run --scrolls PHerc0332,PHerc0211 [--seeds N] [--hours H] [--workdir DIR] [--rounds R] [--gens G] [--threads T] [--workers W] [--kit-url URL] [--stage-only]

Phases (each timed in work/report.json): kit -> inputs per scroll (public open-data bucket) -> seeds (DB-free) -> grows (cloud_box: tracer rounds + in-solve self_collision + selfx scrub + degeneracy
gate between rounds, production guard settings snapshot) -> self-verify of every export tree (cloud_import.verify with THIS kit's detector, fail closed). Results: work/export/<scroll>/<seg>/ (export.json, md5.txt, checkpoints).
Idempotent/resumable: finished seeds are skipped, downloads continue. Any missing piece raises (no silent fallback). No hub, token, key or DB is ever read or written."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from . import bootstrap, net, seedprop, settings


def _grow_job(args):
    (kit, pred, grids, scroll, seed, out, rounds, gens, vox, threads, rng, deadline) = args
    from .. import cloud_box
    pol = settings.policy(kit, threads=2)
    return cloud_box.grow_seed(kit, pred, grids, scroll, seed, out, rounds, gens, vox, threads=threads, rng=rng, pol=pol,
                               self_collision=settings.self_collision_on(), deadline=deadline, gate_override=settings.gate_override)   # the CALLABLE: re-read every round


def _newest_area(sd: Path):
    best = None
    for m in sd.glob("r*/*/meta.json"):
        if (m.parent / "x.tif").exists():
            k = (m.stat().st_mtime, str(m))
            if best is None or k > best[0]:
                best = (k, m)
    if not best:
        return None, None
    try:
        return float(json.loads(best[1].read_text()).get("area_cm2") or 0.0), str(best[1].parent.relative_to(sd))
    except (OSError, ValueError):
        return None, None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="routeA_run.sh", description=__doc__.split("\n\n")[0])
    ap.add_argument("--scrolls", required=True, help="comma-separated, e.g. PHerc0332,PHerc0211")
    ap.add_argument("--seeds", type=int, default=4, help="seeds per scroll (default 4)")
    ap.add_argument("--hours", type=float, default=6.0, help="wall-clock budget for the grow phase (default 6)")
    ap.add_argument("--rounds", type=int, default=6); ap.add_argument("--gens", type=int, default=20, help="generations added per round")
    ap.add_argument("--threads", type=int, default=1, help="tracer threads per grow (1 = one grow per core)")
    ap.add_argument("--workers", type=int, default=0, help="concurrent grows (default: cpu_count // threads)")
    ap.add_argument("--workdir", default=os.environ.get("ROUTEA_WORK", "routeA_work"))
    ap.add_argument("--kit-url", default=None, help=f"HTTPS URL of the kit tarball (else ${bootstrap.KIT_URL_ENV}); sha256-pinned in pins/kit.json")
    ap.add_argument("--rng", type=int, default=1); ap.add_argument("--stage-only", action="store_true", help="download + verify inputs, then stop")
    ap.add_argument("--no-verify", action="store_true", help="skip the self-verify of the export trees (NOT recommended)")
    a = ap.parse_args(argv)
    work = Path(a.workdir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    os.environ["ROUTEA_WORK"] = str(work)          # worker processes inherit it: the guard control file is <work>/control/guards.json
    rep = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "args": vars(a), "phases": {}, "scrolls": {}}
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    def phase(name, t0):
        rep["phases"][name] = {"seconds": round(time.time() - t0, 1), "downloaded_gb": round(net.DOWNLOADED.bytes / 1e9, 3)}
        (work / "report.json").write_text(json.dumps(rep, indent=1))
        log(f"phase {name}: {rep['phases'][name]['seconds']} s (cumulative download {rep['phases'][name]['downloaded_gb']} GB)")

    t0 = time.time()
    kit = bootstrap.ensure_kit(work, a.kit_url, log=log)
    kdir = kit["dir"]
    phase("kit", t0)
    scrolls = [s.strip() for s in a.scrolls.split(",") if s.strip()]
    inputs = {}
    t0 = time.time()
    for sc in scrolls:
        inputs[sc] = bootstrap.ensure_scroll(work, sc, log=log)
        rep["scrolls"][sc] = {"voxel_um": inputs[sc]["voxel_um"], "stats": inputs[sc]["stats"]}
    phase("inputs", t0)
    if a.stage_only:
        return 0
    t0 = time.time()
    jobs = []
    root = bootstrap.package_root()
    for sc in scrolls:
        sf = work / "seeds" / f"{sc}.json"
        if sf.is_file() and len(json.loads(sf.read_text())) >= a.seeds:
            seeds = json.loads(sf.read_text())[:a.seeds]
        else:
            umb = root / "umbilicus" / sc / "umbilicus.json"
            seeds = seedprop.propose(inputs[sc]["pred"], a.seeds, rng_seed=a.rng, umbilicus=str(umb) if umb.is_file() else None, log=log)
            sf.parent.mkdir(parents=True, exist_ok=True)
            sf.write_text(json.dumps(seeds, indent=1))
        if not seeds:
            raise RuntimeError(f"{sc}: no seeds could be proposed")
        rep["scrolls"][sc]["seeds"] = seeds
        for s in seeds:
            jobs.append((sc, s))
    phase("seeds", t0)
    workers = a.workers or max(1, (os.cpu_count() or 1) // max(1, a.threads))
    deadline = time.time() + a.hours * 3600
    t0 = time.time()
    results, todo = [], []
    from .. import cloud_box
    import hashlib
    for sc, s in jobs:
        seg = f"{sc}_c{hashlib.md5(('%s:%s:%s:%s' % (sc, s['x'], s['y'], s['z'])).encode()).hexdigest()[:7]}"
        exp = work / "export" / sc / seg / "export.json"
        if exp.is_file():
            results.append(json.loads(exp.read_text())); log(f"resume: {seg} already exported ({results[-1]['run']['status']})"); continue
        todo.append((sc, s))
    log(f"grow: {len(todo)} seeds to grow, {len(results)} already done, {min(workers, max(1, len(todo)))} parallel, budget {a.hours} h")
    with ProcessPoolExecutor(max_workers=min(workers, max(1, len(todo)))) as ex:
        futs = {ex.submit(_grow_job, (str(kdir), str(inputs[sc]["pred"]), str(inputs[sc]["grids"]), sc, (s["x"], s["y"], s["z"]), str(work / "export" / sc), a.rounds, a.gens,
                                      inputs[sc]["voxel_um"], a.threads, a.rng, deadline)): (sc, s) for sc, s in todo}
        for f in as_completed(futs):
            ex_ = f.result()                                              # a crashed job raises: fail loud
            results.append(ex_)
            log(f"grown {ex_['identity']['seg']}: {ex_['run']['status']} rounds={len(ex_['rounds'])} wall={ex_['run']['wall_s_total']} s")
    phase("grow", t0)
    # self-verify with the box's own detector (the hub repeats this with the fleet detector on import)
    t0 = time.time()
    summ = []
    from .. import cloud_import
    pol = settings.policy(kdir, threads=2)
    tot_area = tot_cpu = 0.0
    for sc in scrolls:
        for sd in sorted((work / "export" / sc).glob("*/")):
            if not (sd / "export.json").is_file():
                continue
            ex_ = json.loads((sd / "export.json").read_text())
            area, ck = _newest_area(sd)
            cpu = sum(r.get("cpu_s") or 0.0 for r in ex_["rounds"])
            v = None if a.no_verify else cloud_import.verify(str(sd), pol)
            row = {"seg": sd.name, "scroll": sc, "status": ex_["run"]["status"], "area_cm2": area, "checkpoint": ck, "cpu_h": round(cpu / 3600.0, 3), "wall_s": ex_["run"]["wall_s_total"],
                   "rounds": len(ex_["rounds"]), "import_verify_pass": None if v is None else bool(v.get("pass")), "import_problems": None if v is None else v.get("problems"),
                   "gate_fractions": (ex_["rounds"][-1].get("gate") or {}).get("fractions") if ex_["rounds"] else None}
            summ.append(row)
            tot_area += area or 0.0; tot_cpu += cpu
    phase("verify", t0)
    rep["results"] = summ
    rep["totals"] = {"segments": len(summ), "area_cm2": round(tot_area, 3), "cpu_h": round(tot_cpu / 3600.0, 3), "cm2_per_cpu_h": round(tot_area / (tot_cpu / 3600.0), 2) if tot_cpu else None,
                     "verified_pass": sum(1 for r in summ if r["import_verify_pass"]), "downloaded_gb": round(net.DOWNLOADED.bytes / 1e9, 3)}
    rep["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (work / "report.json").write_text(json.dumps(rep, indent=1))
    log("RESULT " + json.dumps(rep["totals"]))
    for r in summ:
        log(f"  {r['seg']:<28} {r['status']:<10} area {r['area_cm2']} cm2  cpu {r['cpu_h']} h  verify={r['import_verify_pass']} {r['import_problems'] or ''}")
    log(f"export tree for the hub import (pull it, then `python -m vesuvius_pipeline.cloud_import verify --staging <dir>`): {work / 'export'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
