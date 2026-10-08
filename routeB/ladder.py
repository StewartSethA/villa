"""Fallback ladder for box8 mode: pure functions (no I/O except load_config), unit-tested in routeB/tests/test_box8.py.

A JOB is one (scroll, tag, z0, z1) fit+tile unit on one rung.  When a job fails, `decide()` says what happens next:
  retry   same rung (memory-lean overrides for OOM, plain resume for a stall/unknown error)
  descend re-cover the FAILED interval with the next narrower rung (jobs enter the same work-stealing queue)
  fail    terminal (env fault, ladder exhausted, attempt budget spent): reported loudly, never silent
Failure classes (from the fit log tail + exit status + watchdog):
  multinomial  torch.multinomial 2^24 category limit: DETERMINISTIC, not a memory problem -> never retried on the same rung, descend at once to the
               first narrower rung whose estimated track count is under the limit (estimate = n_tracks x span / full_span; n_tracks from the config or,
               failing that, the .dbm size x tracks_per_gb -- a proxy, announced).
  oom          CUDA out of memory -> retry in the rung with oom_steps[k] (fewer tracks/step, coarser flow grid), then descend.
  host_oom     the process was killed (SIGKILL / our RSS guard) -> same policy as oom.
  stall        the watchdog saw no log growth / iteration progress for stall_minutes -> one resume, then descend.
  env          missing module / GLIBCXX / no space / driver -> terminal: the environment is the bug, descending would burn money.
  tiles        the fit converged but tiling failed -> terminal (a narrower fit does not fix tiling).
  unknown      one resume retry, then terminal.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = next((p for p in (ROOT / "deploy_common" / "ladder_config.json", ROOT.parent.parent / "deploy_common" / "ladder_config.json") if p.exists()),
                ROOT / "deploy_common" / "ladder_config.json")

PATTERNS = [
    ("multinomial", re.compile(r"number of categories cannot exceed 2\^24|cannot exceed 2\^24|multinomial", re.I)),
    ("oom", re.compile(r"CUDA out of memory|OutOfMemoryError|cudaErrorMemoryAllocation|CUBLAS_STATUS_ALLOC_FAILED|CUDA error: out of memory", re.I)),
    ("env", re.compile(r"ModuleNotFoundError|ImportError|GLIBCXX|No space left on device|CUDA driver version|no CUDA-capable device|NVML|cannot open shared object|"
                       r"rebuild the Spiral native extensions", re.I)),
]


def load_config(path: Path | str | None = None, ladder: str | list | None = None) -> dict:
    """ladder: None -> cfg['default_ladder']; 'all' -> every rung; 'full,sw2800' / list -> those rungs in that order."""
    p = Path(path or CFG_PATH)
    cfg = json.loads(p.read_text())
    names = ladder if ladder is not None else cfg.get("default_ladder", "all")
    if isinstance(names, str):
        names = None if names == "all" else [n for n in names.split(",") if n]
    if names:
        byname = {r["name"]: r for r in cfg["rungs"]}
        unknown = [n for n in names if n not in byname]
        if unknown:
            raise SystemExit(f"ROUTEB FAIL ladder: unknown rung(s) {unknown}; known: {list(byname)}")
        cfg["rungs"] = [byname[n] for n in names]
    pd = cfg.get("plan_data")
    cfg["_plan"] = {}
    if pd and (p.parent / pd).exists():
        cfg["_plan"] = json.loads((p.parent / pd).read_text())
    return cfg


PLAN_KEY = {"full": "full", "w13000": "13000", "q4500": "4500", "sw2800": "2800"}


def classify(log_tail: str, rc: int | None, stalled: bool = False, host_oom: bool = False) -> str:
    """Order matters: a multinomial error is checked before OOM (its traceback can mention allocation), env before generic."""
    if stalled:
        return "stall"
    if host_oom or rc in (-9, 137):
        return "host_oom"
    for name, rx in PATTERNS:
        if rx.search(log_tail or ""):
            return name
    return "unknown"


def rung_index(cfg: dict, name: str) -> int:
    for i, r in enumerate(cfg["rungs"]):
        if r["name"] == name:
            return i
    raise KeyError(name)


def _stripes(width, z0: int, z1: int, overlap: int):
    if width == "full" or int(width) >= z1 - z0:
        return [(z0, z1)]
    W, out, s = int(width), [], z0
    while s < z1:
        e = min(s + W, z1)
        out.append((s, e))
        if e >= z1:
            break
        s += W - overlap
    return out


def tag_for(rung: dict, z0: int) -> str:
    return rung["tag"].format(z0=z0)


def expected_h(cfg: dict, scroll: str, shell: int, rung_i: int) -> float:
    r = cfg["rungs"][rung_i]
    ov = cfg.get("expected_gpuh_override", {}).get(scroll, {}).get(r["name"])
    if ov is not None:
        return float(ov)
    pl = cfg.get("_plan", {}).get(scroll, {}).get(PLAN_KEY.get(r["name"], ""))
    if pl:
        return float(pl["gpu_h_" + cfg.get("expected_quantile", "p50")]) / max(1, int(pl.get("n_stripes", 1)))
    return 5.99 * shell / 125.0 * r["gpuh_factor"]


def make_job(cfg: dict, scroll: str, shell: int, rung_i: int, z0: int, z1: int, parent: str | None = None, lineage_attempts: int = 0) -> dict:
    r = cfg["rungs"][rung_i]
    tag = tag_for(r, z0)
    return {"id": f"{scroll}/{tag}", "scroll": scroll, "rung": rung_i, "rung_name": r["name"], "tag": tag, "z0": z0, "z1": z1, "status": "pending",
            "parent": parent, "attempts": [], "oom_n": 0, "stall_n": 0, "unknown_n": 0, "lineage_attempts": lineage_attempts,
            "expected_h": expected_h(cfg, scroll, shell, rung_i), "payload_gb": cfg["payload_gb"].get(r["name"], 0.5), "extra_overrides": {}}


def tracks_estimate(cfg: dict, scroll: str, span: int, dbm_bytes: float | None = None, n_total: float | None = None) -> tuple[float | None, str]:
    """tracks per stripe of `span` slices, assuming tracks uniform in z.  n_total (tracks over the full span) wins: it comes from the fit.log
    'loaded N tracks' line, i.e. it is MEASURED on this very scroll."""
    full = cfg["full_span_z"]
    if n_total:
        return n_total * span / full, "loaded-tracks line of the failing fit"
    n = cfg.get("scrolls", {}).get(scroll, {}).get("n_tracks")
    if n:
        return n * span / full, "config n_tracks"
    if dbm_bytes:
        return dbm_bytes / 1e9 * cfg["limits"]["tracks_per_gb"] * span / full, "PROXY: .dbm size x tracks_per_gb"
    return None, "unknown"


def first_rung(cfg: dict, scroll: str, z0: int, z1: int, use_proxy: bool = False, dbm_bytes: float | None = None, start: int = 0, n_total: float | None = None) -> tuple[int, str]:
    """Initial rung.  Pre-skips rungs whose per-stripe track count is KNOWN (config) to exceed the multinomial limit; the GB proxy is used only if
    use_proxy (i.e. after a multinomial failure was observed)."""
    lim = cfg["limits"]["multinomial_categories"]
    for i in range(start, len(cfg["rungs"])):
        r = cfg["rungs"][i]
        w = r["width"]
        span = min(z1 - z0, z1 - z0 if w == "full" else int(w))
        est, how = tracks_estimate(cfg, scroll, span, dbm_bytes if use_proxy else None, n_total)
        if est is None or est <= lim:
            return i, (f"rung {r['name']} chosen" if i == start else f"rung {r['name']}: rungs before it skipped, est. {est:,.0f} tracks/stripe <= {lim:,} ({how})")
    return len(cfg["rungs"]) - 1, "no rung is under the track limit by estimate; using the narrowest"


def plan(cfg: dict, scroll: str, shell: int, z0: int, z1: int, start: int = 0, parent: str | None = None, lineage_attempts: int = 0) -> list[dict]:
    r = cfg["rungs"][start]
    return [make_job(cfg, scroll, shell, start, a, b, parent, lineage_attempts) for a, b in _stripes(r["width"], z0, z1, r["overlap"])]


def next_rung(cfg: dict, job: dict, min_index: int | None = None) -> int | None:
    span = job["z1"] - job["z0"]
    for i in range((min_index if min_index is not None else job["rung"] + 1), len(cfg["rungs"])):
        r = cfg["rungs"][i]
        if i <= job["rung"]:
            continue
        if r.get("same_span") or r["width"] != "full" and int(r["width"]) < span:
            return i
    return None


def decide(cfg: dict, job: dict, cls: str, shell: int, dbm_bytes: float | None = None) -> dict:
    """What happens to a failed job.  Mutates the counters on `job`; returns {'action': retry|descend|fail, ...}."""
    a = cfg["attempts"]
    job["lineage_attempts"] += 1
    if job["lineage_attempts"] >= a["max_attempts_per_interval"]:
        return {"action": "fail", "why": f"attempt budget {a['max_attempts_per_interval']} spent on this interval (last class {cls})"}
    if cls in ("env", "tiles"):
        return {"action": "fail", "why": f"{cls} fault is terminal: descending the ladder cannot fix it"}
    if cls == "unknown":
        job["unknown_n"] += 1
        if job["unknown_n"] <= a["unknown_retries"]:
            return {"action": "retry", "extra": dict(job["extra_overrides"]), "fresh": False, "why": "unknown error: one plain resume"}
        return {"action": "fail", "why": "unknown error persisted after retry"}
    if cls == "stall":
        job["stall_n"] += 1
        if job["stall_n"] <= a["stall_retries"]:
            return {"action": "retry", "extra": dict(job["extra_overrides"]), "fresh": False, "why": "watchdog stall: one resume"}
    if cls in ("oom", "host_oom") and job["oom_n"] < min(a["oom_retries_in_rung"], len(cfg["oom_steps"])):
        extra = dict(cfg["oom_steps"][job["oom_n"]])
        job["oom_n"] += 1
        # fewer tracks per step keeps the checkpoint compatible (resume, as the 30 observed chunk exits with progress did); a different flow grid does not
        return {"action": "retry", "extra": extra, "fresh": "model_flow_voxel_resolution" in extra, "why": f"{cls}: memory-lean retry {job['oom_n']} with {extra}"}
    # descend
    nr = next_rung(cfg, job)
    if nr is None:
        return {"action": "fail", "why": f"ladder exhausted after {job['rung_name']} ({cls})"}
    if cls == "multinomial":
        # deterministic: jump to the first narrower rung whose stripe track count is under the limit
        ntot = job["n_loaded"] * cfg["full_span_z"] / max(1, job["z1"] - job["z0"]) if job.get("n_loaded") else None
        j, _how = first_rung(cfg, job["scroll"], job["z0"], job["z1"], use_proxy=True, dbm_bytes=dbm_bytes, start=nr, n_total=ntot)
        nr = next_rung(cfg, job, min_index=max(nr, j))
        if nr is None:
            return {"action": "fail", "why": "multinomial limit and no narrower rung is under it"}
    kids = plan(cfg, job["scroll"], shell, job["z0"], job["z1"], start=nr, parent=job["id"], lineage_attempts=job["lineage_attempts"])
    return {"action": "descend", "jobs": kids, "to_rung": cfg["rungs"][nr]["name"], "why": f"{cls}: re-cover z[{job['z0']},{job['z1']}) with rung {cfg['rungs'][nr]['name']} ({len(kids)} stripe(s))"}
