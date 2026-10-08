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
    cfg["_track_limit"] = lift_track_limit_if_fixed(cfg)
    return cfg


TRACKS_PY = ROOT / "spiral-fitting" / "tracks.py"
LIFTED_CATEGORIES = 2 ** 40          # "no ceiling": the chunked multinomial has none; kept finite so the arithmetic below still works


def lift_track_limit_if_fixed(cfg: dict, tracks_py: Path | str | None = None, environ: dict | None = None) -> dict:
    """The 2^24-category ceiling of torch.multinomial was FIXED in spiral-fitting/tracks.py (`_multinomial_chunked`, villa b408d54c: exact two-level
    decomposition, no ceiling).  The planner's stripe-height cap, the DETECT-EARLY kill and the 'multinomial' failure class were written before that
    fix and kept enforcing the old limit (PHerc0191 was split 9,100 + 4,100 for no reason, and a >16.7 M-track fit would be KILLED although the
    deployed code samples it correctly).  So: when the deployed tracks.py contains the fix, the limit is lifted, loudly.  ROUTEB_KEEP_TRACK_LIMIT=1
    keeps the old limit.  UNVALIDATED END TO END: no >2^24-track fit has yet been seen to run to completion on a GPU in THIS pipeline; the first
    one is the validation (watch `loaded N tracks` then the first sampling step)."""
    import os
    env = os.environ if environ is None else environ
    tp = Path(tracks_py or TRACKS_PY)
    lim = cfg.get("limits", {}).get("multinomial_categories")
    if env.get("ROUTEB_KEEP_TRACK_LIMIT") == "1":
        return {"lifted": False, "why": "ROUTEB_KEEP_TRACK_LIMIT=1: the 2^24-track limit stays in force", "limit": lim}
    try:
        fixed = "def _multinomial_chunked" in tp.read_text()
    except OSError as e:
        return {"lifted": False, "why": f"cannot read {tp} ({e}): the 2^24-track limit stays in force (announced, not silent)", "limit": lim}
    if not fixed:
        return {"lifted": False, "why": f"{tp.name} has no _multinomial_chunked: the 2^24-track limit stays in force", "limit": lim}
    cfg.setdefault("limits", {})["multinomial_categories"] = LIFTED_CATEGORIES
    return {"lifted": True, "limit": LIFTED_CATEGORIES, "was": lim,
            "why": "tracks.py has _multinomial_chunked (villa b408d54c): the 2^24-track limit is LIFTED (stripe heights are no longer capped by it, DETECT-EARLY will not kill "
                   ">16.7 M-track fits). UNVALIDATED end to end: the first fit over 2^24 tracks is the validation."}


PLAN_KEY = {"full": "full", "w13000": "13000", "q4500": "4500", "sw2800": "2800"}


def classify(log_tail: str, rc: int | None, stalled: bool = False, host_oom: bool = False, strict_multinomial: bool = False) -> str:
    """Order matters: a multinomial error is checked before OOM (its traceback can mention allocation), env before generic."""
    if stalled:
        return "stall"
    if host_oom or rc in (-9, 137):
        return "host_oom"
    for name, rx in PATTERNS:
        if rx.search(log_tail or ""):
            # strict_multinomial (the 2^24 limit is LIFTED): only the real error text counts.  The bare word "multinomial" also appears in tracebacks of
            # OTHER failures that pass through `_multinomial_chunked` (an OOM, say), which must keep their own class.
            if name == "multinomial" and strict_multinomial and not re.search(r"cannot exceed 2\^24", log_tail or ""):
                continue
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


def expected_h(cfg: dict, scroll: str, shell: int, rung_i: int, span: int | None = None) -> float:
    r = cfg["rungs"][rung_i]
    if r.get("dynamic") or (cfg.get("dynamic") and span):
        return fit_hours(scroll, shell, span or r["height"], speed=cfg.get("gpu_speed", 1.0))
    ov = cfg.get("expected_gpuh_override", {}).get(scroll, {}).get(r["name"])
    if ov is not None:
        return float(ov)
    pl = cfg.get("_plan", {}).get(scroll, {}).get(PLAN_KEY.get(r["name"], ""))
    if pl:
        return float(pl["gpu_h_" + cfg.get("expected_quantile", "p50")]) / max(1, int(pl.get("n_stripes", 1)))
    return 5.99 * shell / 125.0 * r["gpuh_factor"]


def make_job(cfg: dict, scroll: str, shell: int, rung_i: int, z0: int, z1: int, parent: str | None = None, lineage_attempts: int = 0) -> dict:
    r = cfg["rungs"][rung_i]
    if r.get("dynamic") or cfg.get("dynamic"):
        pay = round(0.8 * (z1 - z0) / 13000.0 + 0.05, 3)
    tag = tag_for(r, z0)
    return {"id": f"{scroll}/{tag}", "scroll": scroll, "rung": rung_i, "rung_name": r["name"], "tag": tag, "z0": z0, "z1": z1, "status": "pending",
            "parent": parent, "attempts": [], "oom_n": 0, "stall_n": 0, "unknown_n": 0, "lineage_attempts": lineage_attempts,
            "expected_h": expected_h(cfg, scroll, shell, rung_i, z1 - z0),
            "p90_h": fit_hours(scroll, shell, z1 - z0, "p90", cfg.get("gpu_speed", 1.0)),
            "payload_gb": pay if (r.get("dynamic") or cfg.get("dynamic")) else cfg["payload_gb"].get(r["name"], 0.5), "extra_overrides": {}}


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
    if cls == "multinomial":
        engage_track_limit_fallback(cfg, job)       # no-op unless the limit was lifted and this is the first overflow
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
    if cfg.get("dynamic"):
        # adaptive ladder: an OOM (after its retry) re-covers the failed interval with 0.75x the failed height; a multinomial overflow jumps to the height
        # whose track count (loaded-tracks evidence, else the configured count) is under the 2^24 limit.  Heights are rounded down to GRID slices.
        span = job["z1"] - job["z0"]
        if cls == "multinomial":
            h = tracks_height(cfg, job["scroll"], job.get("n_loaded"), span)
            h = min(h, shrink_height(span) or h) if h >= span else h
        else:
            h = shrink_height(span)
        if not h or h < MIN_HEIGHT:
            return {"action": "fail", "why": f"stripe height floor ({MIN_HEIGHT} slices) reached after {cls} at {span} slices"}
        nr = dyn_rung(cfg, h)
        kids = plan(cfg, job["scroll"], shell, job["z0"], job["z1"], start=nr, parent=job["id"], lineage_attempts=job["lineage_attempts"])
        return {"action": "descend", "jobs": kids, "to_rung": cfg["rungs"][nr]["name"], "height": h,
                "why": f"{cls}: re-cover z[{job['z0']},{job['z1']}) at height {h} (was {span}; 0.75x shrink / track limit), {len(kids)} stripe(s)"}
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


# ---------------------------------------------------------------- computed stripe height (adaptive ladder)
# Peak VRAM of one fit_spiral run is modelled as  vram_gib = A + B * z_slices  (fixed cost + per-slice cost: tracks, flow field and
# optimizer state all grow with the z-extent). TWO measured points, both with the production recipe, so this is a 2-point line, not a fit:
#   * 2,800-slice PHerc0191 stripes on a 4060 Ti 16 GB: nvidia-smi peak 9.6-10.2 GiB (n = 4 stripes, 2026-10-08, includes ~0.3 GiB of other contexts), no OOM
#     through 30,000 steps' first 83 %;
#   * 13,000-slice PHerc0211 (14.36 M tracks) on a V100 32 GB: peak ~31 GiB (f0211_full, OOM at step 25000, restarted from checkpoint).
# Cross-check: a 13,000-slice fit on a 15.58 GiB 4060 Ti OOMed at iteration 61-77 in 4 of 4 runs (model says 31 GiB needed). Tracks per slice differ
# per scroll (0191 22.8 M / 0211 14.4 M): the model is per z-slice, not per track, until a third point exists.
VRAM_A_GIB = 3.97
VRAM_B_GIB_PER_SLICE = (31.0 - 9.8) / (13000 - 2800)
GRID = 100            # stripe heights are rounded DOWN to this many slices (tiling grid)


def computed_height(vram_gib: float, margin_gib: float = 1.5, overlap: int = 0, span: int | None = None, a: float = VRAM_A_GIB,
                    b: float = VRAM_B_GIB_PER_SLICE, grid: int = GRID, min_height: int = 1000) -> int:
    """Largest stripe height (z slices) whose modelled peak VRAM stays within vram_gib - margin_gib; at least min_height, at most span."""
    h = int((vram_gib - margin_gib - a) / b)
    h = max(min_height, (h // grid) * grid)
    return min(h, span) if span else h


def shrink_height(height: int, factor: float = 0.75, grid: int = GRID, min_height: int = 1000) -> int | None:
    """Next height after an OOM (0.75x, grid-rounded); None when it would fall under min_height (the scroll is then skipped, announced)."""
    h = (int(height * factor) // grid) * grid
    return h if h >= min_height else None


def load_heights(path: Path | str) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def record_height(path: Path | str, scroll: str, height: int, vram_gib: float) -> None:
    """Remember the height that SUCCEEDED for a scroll on a card of this size, so the next scroll starts from it."""
    d = load_heights(path)
    d[scroll] = {"height": height, "vram_gib": vram_gib}
    d["_last_ok"] = {"height": height, "vram_gib": vram_gib}
    Path(path).write_text(json.dumps(d, indent=1))


def start_height(path: Path | str, scroll: str, vram_gib: float, span: int, margin_gib: float = 1.5) -> int:
    """Start height for a scroll: its own recorded success, else the last scroll's success on a card of this size, else the model."""
    d = load_heights(path)
    for k in (scroll, "_last_ok"):
        if k in d and abs(d[k]["vram_gib"] - vram_gib) < 0.5:
            return min(d[k]["height"], span)
    return computed_height(vram_gib, margin_gib, span=span)


# ---------------------------------------------------------------- cost model by stripe height (planner + dynamic ladder)
# GPU-hours of ONE fit_spiral job (30,000 steps) as a function of its z height.  Anchors (all MEASURED, on different cards, so this is a model, not a fit):
#   full 13,000 slices: PHerc0125 5.99 GPU-h, PHerc0211 7.49 GPU-h (V100 32 GB, n = 1 each; the plan's n=3 range 5.99-13.27 incl. a gap-only run)
#   2,800-slice stripes: 1.76 h p50 (p10 1.30 / p90 2.54), n = 26 completed fits on the contended 8xV100 box (0125, 5 stripes); and 3.6-3.8 ks = 1.03 h each,
#     8.6-9.8 it/s, 9.6-10.2 GiB, 0 OOM, n = 4 PHerc0191 stripes on an RTX 4060 Ti 16 GB (4060 Ti fleet box, uncontended) -- a faster-than-V100 per-stripe figure.
# height exponent alpha: t(h) = t_full * (h/13000)^alpha; alpha = ln(7.49/1.88)/ln(13000/2800) = 0.90 from 0125 (5.99 full vs 9.41/5 = 1.88 per stripe).
# Consequence (printed in the doc): total GPU-h of a split scroll is ~(1+overlap)*5*(2800/13000)^0.9 / 1 = ~1.25x the full-height GPU-h, but wall-clock drops ~5x.
FULL_SPAN = 13000
ALPHA = 0.90
FULL_P50_MEASURED = {"PHerc0125": 5.99, "PHerc0211": 7.49}
HOURS_PER_SHELL = 0.048            # p50 full-height GPU-h per shell winding, from the plan table (4.5/7.0/10.6 at shell 146, 9.6 at 200, 19.6 at 408); ASSUMED for unfit scrolls
Q_MULT = {"p10": 0.64, "p50": 1.0, "p90": 1.52}      # plan table 6.1/9.6/14.6 -> 0.64/1/1.52
MIN_HEIGHT = 1000


def fit_hours(scroll: str, shell: int, height: int, q: str = "p50", speed: float = 1.0) -> float:
    """Expected GPU-hours of one fit of `height` z-slices on a V100-class card (speed>1 = faster card; A100 is UNMEASURED so the default stays 1.0)."""
    tf = FULL_P50_MEASURED.get(scroll) or max(5.99, HOURS_PER_SHELL * shell)
    return tf * Q_MULT[q] * (min(height, FULL_SPAN) / FULL_SPAN) ** ALPHA / speed


def tracks_height(cfg: dict, scroll: str, n_loaded: float | None, span: int, safety: float = 0.95) -> int:
    """Largest height whose track count stays under the 2^24 multinomial limit (tracks assumed uniform in z -- announced), with a safety factor."""
    lim = cfg["limits"]["multinomial_categories"]
    if n_loaded:
        per_slice = n_loaded / max(1, span)
    else:
        n = cfg.get("scrolls", {}).get(scroll, {}).get("n_tracks")
        if not n:
            return FULL_SPAN
        per_slice = n / cfg["full_span_z"]
    return max(MIN_HEIGHT, int(lim * safety / per_slice) // GRID * GRID)


def dyn_rung(cfg: dict, height: int, overlap: int = 200) -> int:
    name = f"h{height}"
    for i, r in enumerate(cfg["rungs"]):
        if r["name"] == name:
            return i
    cfg["rungs"].append({"name": name, "width": height, "overlap": overlap, "tag": f"h{height}s{{z0}}", "gpuh_factor": None, "fit_overrides": {},
                         "dynamic": True, "height": height, "note": "computed-height rung (routeB/planner.py), created at run time"})
    return len(cfg["rungs"]) - 1


def start_height_for(cfg: dict, scroll: str, vram_gib: float, span: int, margin_gib: float = 1.5, max_height: int = FULL_SPAN,
                     heights_path: Path | str | None = None) -> tuple[int, str]:
    """Initial stripe height = min(span, VRAM model, track limit from the configured count, --max-height); a recorded success for this scroll wins."""
    hv = computed_height(vram_gib, margin_gib, span=span)
    ht = tracks_height(cfg, scroll, None, span)
    h = min(span, hv, ht, max_height)
    why = f"min(span {span}, VRAM {vram_gib:.1f} GiB model {hv}, 2^24-track limit {ht}, max {max_height})"
    if heights_path:
        d = load_heights(heights_path)
        if scroll in d and abs(d[scroll]["vram_gib"] - vram_gib) < 0.5:
            return min(d[scroll]["height"], span), f"recorded success for {scroll} on a {d[scroll]['vram_gib']:.1f} GiB card ({why})"
    return h, why


def jobs_for_height(cfg: dict, scroll: str, shell: int, z0: int, z1: int, height: int, overlap: int = 200) -> list[dict]:
    """Jobs covering [z0,z1) at `height`: one 'full' job when height >= span, else a dynamic rung's stripes."""
    if height >= z1 - z0:
        return plan(cfg, scroll, shell, z0, z1, start=0)
    return plan(cfg, scroll, shell, z0, z1, start=dyn_rung(cfg, height, overlap))


# ---------------------------------------------------------------- reference cards: usable VRAM and the height model's max stripe
# usable = what torch sees after the driver/context (nvidia-smi memory.total is a little larger); RTX 5090 / 4090 / A100 / H100 figures are the vendor sizes minus the usual
# ~0.4-0.7 GiB (5090 31.3 as given by the coordinator's rental listing); 4060 Ti 15.58 is MEASURED here.  The height is the MODEL's (3.97 + 0.00208 GiB per slice, margin 1.5 GiB),
# which was fitted on two points (9.8 GiB @ 2,800 slices on a 4060 Ti, ~31 GiB @ 13,000 on a V100 32 GB): beyond them it is an extrapolation.
CARDS = [("RTX 4060 Ti 16 GB", 16, 15.58), ("RTX 4090 24 GB", 24, 23.5), ("RTX 5090 32 GB", 32, 31.3), ("V100 32 GB", 32, 31.7), ("RTX PRO 5000 48 GB", 48, 47.0),
         ("A100 40 GB", 40, 39.4), ("A100 80 GB", 80, 79.2), ("H100 80 GB", 80, 79.6)]


def card_table(margin_gib: float = 1.5) -> list[tuple]:
    out = []
    for name, nominal, usable in CARDS:
        h = computed_height(usable, margin_gib)
        out.append((name, nominal, usable, h, max(1, -(-FULL_SPAN // h)) if h < FULL_SPAN else 1))
    return out


ORIG_CATEGORIES = 2 ** 24


def engage_track_limit_fallback(cfg: dict, job: dict | None = None) -> bool:
    """The lifted 2^24-track limit was tried and a fit really hit `number of categories cannot exceed 2^24`: restore the old cap for the REST of the run
    (every later height decision and the DETECT-EARLY kill use it again).  Returns True the first time (caller logs + persists), else False."""
    tl = cfg.get("_track_limit") or {}
    if not tl.get("lifted") or tl.get("fallback_engaged"):
        return False
    cfg.setdefault("limits", {})["multinomial_categories"] = ORIG_CATEGORIES
    tl["fallback_engaged"] = True
    tl["fallback_job"] = (job or {}).get("id")
    tl["fallback_n_loaded"] = (job or {}).get("n_loaded")
    tl["new_fallback"] = True
    cfg["_track_limit"] = tl
    return True


def restore_fallback_marker(cfg: dict, marker: Path | str) -> bool:
    """A restart must remember that the fallback was already needed: the marker file written by box8 re-engages it before planning."""
    try:
        d = json.loads(Path(marker).read_text())
    except (OSError, ValueError):
        return False
    if engage_track_limit_fallback(cfg, {"id": d.get("job")}):
        cfg["_track_limit"]["new_fallback"] = False
        cfg["_track_limit"]["restored_from"] = str(marker)
        return True
    return False
