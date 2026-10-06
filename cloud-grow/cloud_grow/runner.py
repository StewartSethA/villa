"""Standalone grow loop: `stages/grow.py` `run_round` + `grow_segment` semantics without the hub DB or hub token.

PORT, not a reimplementation of the science: the tracer command line, params.json, SPACELINE arm, normal grids,
guard call sequence (`growth_guard.guard_round`), fail-closed selfcross, the D3 "resumes only expand" invariant and
the exhaustion rule follow `src/vesuvius_pipeline/stages/grow.py` of the source fleet (release sha recorded in
VENDORED.json). What is NOT ported (announced in every run's `run_context.json["not_ported"]`):
  * SPAGHETTI / HOLEY / MUSHY pause gates (need shape_metrics, fibre_metrics, lasagna normals),
  * good-sheet neighbour seeding, efficiency-floor pause (need fleet medians),
  * hub neighbour cover (`overlap` / `merge_pause` guard criteria are skipped, cover=None),
  * Lasagna normal sampler (`normal_dev` guard criterion is skipped unless a sampler is supplied).
UNVALIDATED against human annotation (D6): this loop produces segments by the same method as the source fleet's
Route A grow; no accuracy claim is made for it here.
"""
from __future__ import annotations

import json
import os
import resource
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import growth_guard as GG
from . import scoring as SC
from . import state as ST
from . import tools as T

VACUUM_TH = SC.VACUUM_TH
DEFAULT_STEP_SIZE = 20.0
SPACELINE = {"space_line_weight": 0.1, "space_line_threshold": 5.0, "space_line_steps": 8}
EXHAUSTED_MM2 = 50.0
SELFX_UNVERIFIED_BY = "growth_guard:selfx_unverified"
NOT_PORTED = ["SPAGHETTI/HOLEY/MUSHY pause gates", "good-sheet neighbour seeding", "efficiency-floor pause",
              "hub neighbour cover (guard overlap/merge_pause skipped)", "lasagna normal sampler (normal_dev skipped unless supplied)",
              "flatten_feedback (no flatten attempts exist on a cloud box)"]


@dataclass
class RunConfig:
    scroll: str
    workdir: str
    kit_bin: str = ""
    ct_zarr: str = ""                   # CT OME-Zarr (level 1 is what the guard needs)
    prediction_zarr: str = ""           # surface prediction zarr (tracer -v when tracer_volume == "prediction")
    normal_grids: str = ""
    umbilicus_json: str = ""            # optional; wrap_spacing / curvature are skipped without it
    voxel_um: float = 0.0
    tracer_volume: str = "prediction"   # production: grow.input.use_prediction=1
    spaceline_always: bool = True       # production: grow.input.spaceline_always=1
    step_size: float = DEFAULT_STEP_SIZE
    first_gens: int = 35
    gen_step: int = 20
    thread_limit: int = 1
    rng_seed: int = 1
    grid_cache_bytes: int = 256 * 1024 ** 2   # benchmark value; production uses 16 GiB (resume rounds at 256 MB UNTESTED)
    wall_s: float = 6 * 3600
    max_rounds: int = 0
    max_area_cm2: float = 0.0           # 0 = take the policy's max_area_cm2 when grow_to_sane, else none
    policy_path: str = ""
    ct_level_guard: int = 1
    allow_unpinned: bool = False
    round_timeout_s: int = 0            # 0 = production budget_s() rule
    instance_id: str = ""
    release_sha: str = ""

    @staticmethod
    def load(path: str) -> "RunConfig":
        with open(path) as fh:
            d = json.load(fh)
        d = {k: v for k, v in d.items() if not k.startswith("_")}
        known = set(RunConfig.__dataclass_fields__)
        extra = set(d) - known
        if extra:
            raise ValueError(f"unknown config key(s) {sorted(extra)} in {path}")
        return RunConfig(**d)


@dataclass
class RoundResult:
    round: int
    ok: bool
    why: str
    checkpoint: str | None
    gen_from: int
    gen_to: int
    area_before: float
    area_after: float | None
    cpu_s: float
    wall_s: float
    timed_out: bool
    salvaged: bool = False
    armed: bool = False
    peak_rss_mb: float | None = None
    peak_rss_mb_sampled: float | None = None
    mhz: dict | None = None
    score: SC.Score | None = None


@dataclass
class Outcome:
    status: str          # done|paused|exhausted|failed|cancelled
    why: str
    checkpoint: str | None
    area_cm2: float
    verified_cm2: float | None
    rounds: list = field(default_factory=list)


# ------------------------------------------------------------------ small helpers
def read_meta(ckpt: str) -> dict:
    with open(os.path.join(ckpt, "meta.json")) as fh:
        return json.load(fh)


def budget_s(area_cm2: float, gen: int, newgen: int, rate=0.97, safety=3.0, tmin=600, tmax=43200) -> int:
    k = (area_cm2 * 100.0) / float(max(gen, 1) ** 2)
    exp_mm2 = k * (newgen ** 2 - gen ** 2)
    return int(max(tmin, min(tmax, safety * exp_mm2 / rate)))


def newest_checkpoint(tgt: str) -> str | None:
    """Newest checkpoint dir under `tgt` by meta.json mtime (NOT by name: 'guarded_g_' sorts above 'auto_grown_',
    the 2026-09-29 production bug), ties broken by name."""
    if not os.path.isdir(tgt):
        return None
    best, key = None, None
    for d in os.listdir(tgt):
        mp = os.path.join(tgt, d, "meta.json")
        try:
            k = (os.path.getmtime(mp), d)
        except OSError:
            continue
        if key is None or k > key:
            best, key = os.path.join(tgt, d), k
    return best


def grid_fetch_failures(round_log: str, offset: int = 0) -> str | None:
    try:
        with open(round_log, "rb") as fh:
            fh.seek(offset)
            txt = fh.read().decode("utf-8", "replace")
    except OSError:
        return None
    bad = [ln for ln in txt.splitlines() if ("normal-grid fetch for" in ln and "returned HTTP" in ln)
           or "Failed to fetch remote normal-grid" in ln]
    if not bad:
        return None
    return f"ENV_FAULT: lazy normal-grid fetch failed for {len(bad)} grid file(s) (first: {bad[0].strip()[:120]}); the grow ran without normals"


def _vmhwm_mb(pid: int) -> float | None:
    try:
        with open(f"/proc/{pid}/status") as fh:
            for ln in fh:
                if ln.startswith("VmHWM:"):
                    return int(ln.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        return None
    return None


def _cpu_mhz() -> float | None:
    try:
        v = []
        with open("/proc/cpuinfo") as fh:
            for ln in fh:
                if ln.startswith("cpu MHz"):
                    v.append(float(ln.split(":")[1]))
        return float(np.mean(v)) if v else None
    except (OSError, ValueError):
        return None


def _pct(a):
    if not a:
        return None
    return {"p10": float(np.percentile(a, 10)), "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)), "n": len(a)}


def _wait_rusage(p, timeout, samples: dict | None = None):
    """Wait for Popen `p`; return (returncode, its own rusage) via os.wait4. Samples VmHWM and mean cpu MHz
    every ~2 s into `samples` (peak RSS is NULL for every production grow attempt: this is the fix)."""
    deadline = None if timeout is None else time.time() + timeout
    last = 0.0
    while True:
        pid, status, ru = os.wait4(p.pid, os.WNOHANG)
        if pid == p.pid:
            p.returncode = os.waitstatus_to_exitcode(status)
            return p.returncode, ru
        now = time.time()
        if samples is not None and now - last >= 2.0:
            last = now
            r = _vmhwm_mb(p.pid)
            if r is not None:
                samples["rss"] = max(samples.get("rss", 0.0), r)
            m = _cpu_mhz()
            if m is not None:
                samples.setdefault("mhz", []).append(m)
        if deadline is not None and now >= deadline:
            raise subprocess.TimeoutExpired(p.args, timeout)
        time.sleep(0.25)


# ------------------------------------------------------------------ one tracer round
def run_round(cfg: RunConfig, bin_dir: str, out_dir: str, rnd: int, resume: str | None,
              seed: tuple[int, int, int] | None, gen_from: int, gen_to: int, area_before: float, arm: bool,
              rng_seed: int, db=None, seg: str | None = None, attempt_id: int | None = None,
              param_override: dict | None = None, child_holder: dict | None = None) -> RoundResult:
    vol_arg = cfg.prediction_zarr if cfg.tracer_volume == "prediction" else cfg.ct_zarr
    if not vol_arg or not os.path.isdir(vol_arg) or not cfg.normal_grids or not os.path.isdir(cfg.normal_grids):
        return RoundResult(rnd, False, f"missing local input: volume={vol_arg!r} normal_grids={cfg.normal_grids!r}", None,
                           gen_from, gen_to, area_before, None, 0.0, 0.0, False)
    params = {"mode": "seed", "generations": gen_to, "min_area_cm": 0.002, "step_size": cfg.step_size,
              "thread_limit": cfg.thread_limit, "voxelsize": cfg.voxel_um, "normal_grid_path": str(cfg.normal_grids),
              "snapshot-interval": 1 if resume else 5, "cache_root": str(Path(out_dir) / "cache_root")}
    if arm:
        params.update(SPACELINE)
    if param_override:
        params.update(param_override)
    os.makedirs(out_dir, exist_ok=True)
    pj = os.path.join(out_dir, f"params_r{rnd}.json")
    with open(pj, "w") as fh:
        json.dump(params, fh)
    tgt = os.path.join(out_dir, f"r{rnd}")
    cmd = [os.path.join(bin_dir, "vc_grow_seg_from_seed"), "-v", str(vol_arg), "-t", tgt, "--params", pj]
    cmd += ["--resume", resume] if resume else ["--seed", str(seed[0]), str(seed[1]), str(seed[2])]
    env = T.tool_env(bin_dir, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS=str(cfg.thread_limit),
                     VC_GRID_CACHE_BYTES=str(cfg.grid_cache_bytes), VC_GROWPATCH_RNG_SEED=str(rng_seed))
    budget = cfg.round_timeout_s or (budget_s(area_before, gen_from, gen_to) if resume else 3600)
    w0 = time.time()
    timed_out = False
    round_log = os.path.join(out_dir, f"round{rnd}.log")
    log_off = os.path.getsize(round_log) if os.path.exists(round_log) else 0
    samples: dict = {}
    ru_child = None
    with open(round_log, "ab") as lg:
        lg.write((" ".join(cmd) + "\n").encode())
        lg.flush()
        p = subprocess.Popen(cmd, stdout=lg, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        if child_holder is not None:
            child_holder["proc"] = p
        try:
            rc, ru_child = _wait_rusage(p, budget, samples)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(p.pid), 15)
            try:
                _, ru_child = _wait_rusage(p, 60)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(p.pid), 9)
                _, ru_child = _wait_rusage(p, None)
            rc, timed_out = -9, True
        finally:
            if child_holder is not None:
                child_holder.pop("proc", None)
    cpu = (ru_child.ru_utime + ru_child.ru_stime) if ru_child is not None else 0.0
    wall = time.time() - w0
    peak = (ru_child.ru_maxrss / 1024.0) if ru_child is not None else None     # Linux: KiB
    common = dict(cpu_s=cpu, wall_s=wall, timed_out=timed_out, armed=arm, peak_rss_mb=peak,
                  peak_rss_mb_sampled=samples.get("rss"), mhz=_pct(samples.get("mhz") or []))
    ck = newest_checkpoint(tgt)
    fault = grid_fetch_failures(round_log, log_off)
    if fault:
        return RoundResult(rnd, False, fault, None, gen_from, gen_to, area_before, None, **common)
    if rc != 0 or ck is None:
        why = "timeout" if timed_out else ("cancelled" if rc in (-15, -9) else f"exit_{rc}")
        if ck is not None:
            try:
                m = read_meta(ck)
                if float(m.get("area_cm2") or 0) > area_before and int(m.get("max_gen") or 0) > gen_from:
                    return RoundResult(rnd, True, why + "_salvaged", ck, gen_from, int(m["max_gen"]), area_before,
                                       float(m["area_cm2"]), salvaged=True, **common)
            except (OSError, ValueError):
                pass
        return RoundResult(rnd, False, why, None, gen_from, gen_to, area_before, None, **common)
    m = read_meta(ck)
    return RoundResult(rnd, True, "ok", ck, gen_from, int(m.get("max_gen") or gen_to), area_before,
                       float(m.get("area_cm2") or 0.0), **common)


# ------------------------------------------------------------------ guard inputs
class GuardInputs:
    """Built ONCE per grow_segment (the fleet's guard_inputs.InputCache, minus the hub-only parts)."""

    def __init__(self, cfg: RunConfig, normal_sampler=None):
        self.cfg = cfg
        self.sampler = None
        self.pred_sampler = None
        self.umbilicus_of_z = None
        self.normal_sampler = normal_sampler
        self.available: dict = {}

    def build(self, db, seg, pol):
        cfg = self.cfg
        if self.sampler is None:
            if not cfg.ct_zarr:
                raise RuntimeError(f"guard needs CT level {pol.ct_level} (config.ct_zarr is empty): fetch it with "
                                   "cloud-grow/scripts/fetch_data.py --what ct")
            self.sampler = GG.ZarrSampler(cfg.ct_zarr, pol.ct_level)
        if self.pred_sampler is None and cfg.prediction_zarr:
            try:
                self.pred_sampler = GG.ZarrSampler(cfg.prediction_zarr, 0)
            except Exception as e:                      # noqa: BLE001 - skips ridge_hit/seam/empty_space, announced below
                print(f"[cloud-grow] prediction sampler unavailable ({type(e).__name__}: {e}); ridge_hit/seam/empty_space SKIPPED", flush=True)
        if self.umbilicus_of_z is None and cfg.umbilicus_json:
            try:
                self.umbilicus_of_z = umbilicus_xy_of_z(cfg.umbilicus_json)
            except Exception as e:                      # noqa: BLE001
                print(f"[cloud-grow] umbilicus unreadable ({type(e).__name__}: {e}); wrap_spacing/curvature SKIPPED", flush=True)
        ctx = GG.ShadowContext(pred_sampler=self.pred_sampler, normal_sampler=self.normal_sampler,
                               umbilicus_of_z=self.umbilicus_of_z, db=db, seg=seg, step_size=float(cfg.step_size))
        self.available = {"ct_sampler": True, "neighbour_cover": False, "pred_sampler": self.pred_sampler is not None,
                          "umbilicus": self.umbilicus_of_z is not None, "normal_sampler": self.normal_sampler is not None}
        return {"sampler": self.sampler, "cover": None, "neighbour_ok": None, "shadow_ctx": ctx}


def umbilicus_xy_of_z(path: str):
    with open(path) as fh:
        data = json.load(fh)
    pts = data.get("control_points") if isinstance(data, dict) else data
    if not pts:
        raise ValueError(f"no points in {path}")
    pts = sorted(pts, key=lambda p: p["z"])
    zs = np.array([p["z"] for p in pts], dtype=np.float64)
    xs = np.array([p["x"] for p in pts], dtype=np.float64)
    ys = np.array([p["y"] for p in pts], dtype=np.float64)

    def f(z):
        z = np.clip(z, zs[0], zs[-1])
        return np.interp(z, zs, xs), np.interp(z, zs, ys)
    return f


# ------------------------------------------------------------------ policy
def load_policy_into_state(db, policy_path: str) -> dict:
    """Apply config/guard_policy.production.json to the local pipeline_setting table, then the SAME
    `growth_guard.policy_from_db` code path the fleet uses reads it back. Unknown keys are an ERROR (the fleet
    silently ignores them, which is how a typo'd enforce flag stays off for ever)."""
    with open(policy_path) as fh:
        doc = json.load(fh)
    settings = doc.get("settings") or {}
    fields_ = {f for f in GG.GuardPolicy.__dataclass_fields__}
    for k, v in settings.items():
        if k.startswith(GG.PREFIX) and k[len(GG.PREFIX):] not in fields_:
            raise ValueError(f"policy key {k!r} is not a GuardPolicy field (typo? it would be silently ignored)")
        ST.set_pipeline_setting(db, k, v, by="policy-file")
    return settings


def resolve_policy(db, bin_dir: str):
    pol = GG.policy_from_db(db)
    if pol.selfcross:
        from dataclasses import replace
        upd = {}
        if not pol.selfcross_bin:
            upd["selfcross_bin"] = os.path.join(bin_dir, "vc_tifxyz_selfcross")
        if not pol.selfcross_env:
            upd["selfcross_env"] = T.tool_env(bin_dir)
        pol = replace(pol, **upd) if upd else pol
    return pol


# ------------------------------------------------------------------ fail closed
def _selfx_fail_closed(db, seg, attempt_id, held_ckpt, raw_ckpt, round_start_ckpt, round_start_area, area,
                       last_score, rounds, reason, held=0, kept=0) -> Outcome:
    """A round whose self-crossing check could not run (or whose guard raised) is never let through: the grow is
    PAUSED, an alert is raised, and the newest recorded surface is one that WAS verified (port of
    stages.grow._selfx_fail_closed; the fleet's `nofinish` flag has no equivalent here, so the
    unverified-raw case writes a `selfx_unverified.json` marker the importer refuses)."""
    note = ""
    ckpt, a = raw_ckpt, area
    if held_ckpt and kept > 0:
        ckpt, a = held_ckpt, float(read_meta(held_ckpt).get("area_cm2") or 0.0)
        note = f"held back {held} new cells, kept {kept} round-start cells"
    elif round_start_ckpt and os.path.isdir(round_start_ckpt):
        try:
            dst = str(Path(raw_ckpt or round_start_ckpt).parent / f"selfx_held_{Path(raw_ckpt or round_start_ckpt).name}")
            if os.path.exists(dst):
                raise FileExistsError(dst)
            shutil.copytree(round_start_ckpt, dst)
            Path(dst, "selfx_held.json").write_text(json.dumps({"copy_of": round_start_ckpt, "reason": reason[:300],
                                                                 "stop": "selfx_unverified"}))
            ST.record_artifact(db, seg, "grow", "tifxyz", dst, attempt_id)
            ckpt, a = dst, round_start_area
            note = "reverted to a copy of the round-start checkpoint"
        except Exception as e:                          # noqa: BLE001
            note = f"round-start copy failed ({type(e).__name__}: {e}); "
            ckpt = None
    if ckpt is None or ckpt == raw_ckpt:
        if raw_ckpt and os.path.isdir(raw_ckpt):
            Path(raw_ckpt, "selfx_unverified.json").write_text(json.dumps(
                {"by": SELFX_UNVERIFIED_BY, "reason": reason[:300],
                 "meaning": "newest surface was never self-crossing checked; the hub importer refuses it"}))
        ckpt, a = raw_ckpt, area
        note += "no verified surface: raw kept as resume point, selfx_unverified.json marker written"
    ST.record_metric(db, seg, "guard_selfx_unverified", float(held), text=f"{reason[:160]} | {note}"[:300],
                     stage="grow", attempt_id=attempt_id)
    ST.set_paused(db, seg, "grow", True, by="grow: selfx_unverified")
    ST.record_metric(db, seg, "pause_reason", text=f"selfx_unverified: {reason[:160]}", stage="grow")
    from .alerts import alert
    alert(f"grow {seg}: self-crossing check did not run ({reason[:160]}) -- {note}; grow PAUSED")
    return Outcome("paused", "selfx_unverified", ckpt, a, last_score.verified_cm2 if last_score else None, rounds)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def append_round_record(out_dir: str, rec: dict) -> None:
    """One JSON line per round (`rounds.jsonl`): the raw material of the export manifest. Append-only."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "rounds.jsonl"), "a") as fh:
        fh.write(json.dumps(_jsonable(rec)) + "\n")


def spaceline_armed(verdict, last_score, cfg: RunConfig) -> bool:
    if cfg.spaceline_always:
        return True
    return verdict == "VACUUM" and (last_score is None or last_score.material_frac <= VACUUM_TH)


def next_round_number(out_dir: str) -> int:
    n = 0
    if os.path.isdir(out_dir):
        for d in os.listdir(out_dir):
            if d.startswith("r") and d[1:].isdigit():
                n = max(n, int(d[1:]))
    return n


# ------------------------------------------------------------------ the loop
def grow_segment(cfg: RunConfig, bin_dir: str, db, seg: str, out_dir: str, seed=None, resume: str | None = None,
                 should_stop=lambda: False, attempt_id: int | None = None, normal_sampler=None) -> Outcome:
    ST.upsert_segment(db, seg, scroll=cfg.scroll, route="A")
    # D3: a segment that already has grown rounds on disk is RESUMED, never re-grown from scratch.
    prior = next_round_number(out_dir)
    if resume is None and prior:
        newest = None
        for r in range(prior, 0, -1):
            newest = newest_checkpoint(os.path.join(out_dir, f"r{r}"))
            if newest:
                break
        if newest is None:
            raise RuntimeError(f"{seg}: {out_dir} has round dirs but no checkpoint with a meta.json; refusing to "
                               "re-grow from scratch (D3). Delete the directory yourself if that is what you want.")
        print(f"[cloud-grow] D3: {seg} has {prior} earlier round(s); RESUMING from {newest} (not re-seeding)", flush=True)
        resume = newest
    if resume is not None:
        if not (os.path.isdir(resume) and all(os.path.exists(os.path.join(resume, f)) for f in ("x.tif", "y.tif", "z.tif", "meta.json"))):
            raise RuntimeError(f"{seg}: resume checkpoint {resume!r} is not a grown tifxyz (needs x/y/z.tif + meta.json); "
                               "D3: resumes continue from an existing surface, they never start over")
    elif seed is None:
        raise ValueError("grow_segment needs a seed or a resume checkpoint")
    pol = resolve_policy(db, bin_dir)
    gg_on = bool(pol.enabled)
    gg_state = GG.GuardState() if gg_on else None
    if gg_state is not None and resume:
        try:                                              # D3: protect the resume surface from round 1 on
            rX, rY, rZ = GG._read_xyz(resume)
            gg_state.prev_V = (rX > 0) & (rY > 0) & (rZ > 0)
            gg_state.prev_P, _ = GG.lattice_frame(rX, rY, rZ)
        except Exception:                                 # noqa: BLE001
            gg_state.prev_V = None
            gg_state.prev_P = None
    gi = GuardInputs(cfg, normal_sampler=normal_sampler)
    ST.record_metric(db, seg, "step_size", float(cfg.step_size), stage="grow", attempt_id=attempt_id)
    t0 = time.time()
    rounds: list[RoundResult] = []
    cur = resume
    gen = int(read_meta(cur).get("max_gen") or 0) if cur else 0
    area = float(read_meta(cur).get("area_cm2") or 0.0) if cur else 0.0
    verdict, last_score, score_errors = None, None, 0
    rnd = prior
    n_this_call = 0

    def done(status, why):
        return Outcome(status, why, cur, area, last_score.verified_cm2 if last_score else None, rounds)

    while True:
        if should_stop():
            return done("cancelled", "operator")
        if ST.is_paused(db, seg, "grow"):
            return done("paused", "operator_pause")
        if time.time() - t0 >= cfg.wall_s:
            return done("done", "wall_ceiling")
        if cfg.max_rounds and n_this_call >= cfg.max_rounds:
            return done("done", f"round_target {cfg.max_rounds}")
        if verdict == "FRAME_SUSPECT":
            ST.set_paused(db, seg, "grow", True, by="grow: frame suspect")
            return done("paused", "frame_suspect")
        arm = spaceline_armed(verdict, last_score, cfg)
        rnd += 1
        n_this_call += 1
        round_start_ckpt, round_start_area = cur, area
        gen_to = (gen + cfg.gen_step) if cur else cfg.first_gens
        rr = run_round(cfg, bin_dir, out_dir, rnd, cur, seed, gen, gen_to, area, arm, cfg.rng_seed, db=db, seg=seg,
                       attempt_id=attempt_id)
        rounds.append(rr)
        rec = {"round": rnd, "ok": rr.ok, "why": rr.why, "wall_s": rr.wall_s, "cpu_s": rr.cpu_s,
               "peak_rss_mb": rr.peak_rss_mb, "peak_rss_mb_sampled": rr.peak_rss_mb_sampled,
               "busy_mhz": rr.mhz, "gen_from": rr.gen_from, "gen_to": rr.gen_to, "armed": rr.armed,
               "area_cm2_preguard": rr.area_after, "area_before": rr.area_before,
               "raw_checkpoint": rr.checkpoint, "resume_from": round_start_ckpt}
        if not rr.ok:
            append_round_record(out_dir, rec)
            if rr.why == "cancelled":
                return done("cancelled", "signal")
            if rr.why == "timeout" and (last_score.verified_cm2 if last_score else 0) > 0:
                return done("done", "round_timeout_with_area")
            return done("failed", rr.why)
        cur, gen = rr.checkpoint, rr.gen_to
        gained = (rr.area_after or 0.0) - area
        area = rr.area_after or area
        sc = None
        try:
            if cfg.ct_zarr:
                sampler = gi.sampler or GG.ZarrSampler(cfg.ct_zarr, cfg.ct_level_guard)
                gi.sampler = sampler
                sc = SC.score_checkpoint(cur, sampler)
            score_errors = 0
        except Exception as e:                            # noqa: BLE001 - recorded; three strikes
            ST.record_metric(db, seg, "score_error", text=f"{type(e).__name__}: {str(e)[:160]}", stage="grow")
            score_errors += 1
            if score_errors >= 3:
                append_round_record(out_dir, rec)
                return done("failed", f"scorer broken: {type(e).__name__}")
        rr.score = sc
        ST.record_artifact(db, seg, "grow", "tifxyz", cur, attempt_id)
        ST.record_metric(db, seg, "area_cm2", area, stage="grow", attempt_id=attempt_id)
        ST.record_metric(db, seg, "round_cpu_s", rr.cpu_s, stage="grow", attempt_id=attempt_id)
        if sc:
            last_score, verdict = sc, sc.verdict
            ST.record_metric(db, seg, "material_frac", sc.material_frac, stage="grow", attempt_id=attempt_id,
                             control_name="bbox_occupancy", control_value=sc.bbox_occupancy)
            ST.record_metric(db, seg, "verified_cm2", sc.verified_cm2, stage="grow", attempt_id=attempt_id,
                             text="box_claim_ct_level_%d" % cfg.ct_level_guard)
            rec.update({"material_frac": sc.material_frac, "onesheet_frac": sc.onesheet_frac,
                        "bbox_occupancy": sc.bbox_occupancy, "verified_cm2_boxclaim": sc.verified_cm2,
                        "yield_verdict": sc.verdict})
        try:
            P_, V_ = GG.lattice_frame(*GG._read_xyz(cur))
            from . import geomaps as GM
            fm = GM.fold_metrics(P_, V_, voxel_um=cfg.voxel_um)
            rec["geo_planarity"] = float(GM.planarity_score(P_, V_))
            rec["geo_hairpin_lines"] = fm.get("geo_hairpin_lines")
            rec["geo_lines_read"] = fm.get("geo_lines_read")
            rec["geo_fold_frac"] = fm.get("geo_fold_frac")
        except Exception as e:                            # noqa: BLE001 - diagnostic only
            ST.record_metric(db, seg, "score_error", text=f"geo: {type(e).__name__}: {str(e)[:120]}", stage="grow")
        if gg_on and cur:
            gg_judged = False
            try:
                kw = gi.build(db, seg, pol)
                pre_guard_cur = cur
                g0 = time.time()
                cur, ginfo = GG.guard_round(cur, pol, cfg.voxel_um, gg_state, own=seg, **kw)
                gg_judged = True
                ST.record_metric(db, seg, "guard_s", time.time() - g0, stage="grow", attempt_id=attempt_id)
                rec.update({"guard_stop": ginfo.get("stop"), "guard_summary": ginfo, "guard_s": time.time() - g0,
                            "guard_inputs_available": gi.available, "guarded_checkpoint": cur if cur != pre_guard_cur else None})
                if cur != pre_guard_cur and not (ginfo.get("stop") == "selfx_unverified" and not int(ginfo.get("cells_after") or 0)):
                    ST.record_artifact(db, seg, "grow", "tifxyz", cur, attempt_id)
                ST.record_metric(db, seg, "guard_pruned_cells", float(ginfo["cells_before"] - ginfo["cells_after"]),
                                 stage="grow", attempt_id=attempt_id)
                ST.record_metric(db, seg, "guard_summary", text=json.dumps(_jsonable(ginfo)), stage="grow", attempt_id=attempt_id)
                if ginfo.get("stop") == "selfx_unverified":
                    ST.record_metric(db, seg, "guard_stop", text="selfx_unverified", stage="grow", attempt_id=attempt_id)
                    o = _selfx_fail_closed(db, seg, attempt_id, cur if cur != pre_guard_cur else None, pre_guard_cur,
                                           round_start_ckpt, round_start_area, area, last_score, rounds,
                                           str(ginfo.get("guard_selfx_error") or ""),
                                           held=int(ginfo.get("selfx_held_back_cells") or 0),
                                           kept=int(ginfo.get("cells_after") or 0))
                    rec.update({"checkpoint": o.checkpoint, "area_cm2_postguard": o.area_cm2})
                    append_round_record(out_dir, rec)
                    return o
                post_area = read_meta(cur).get("area_cm2")
                area = float(post_area) if post_area is not None else area
                if round_start_ckpt:                      # D3 invariant, in CELLS (not area: different formulas)
                    try:
                        a, b, c = GG._read_xyz(round_start_ckpt)
                        d_, e_, f_ = GG._read_xyz(cur)
                        n0 = int(((a > 0) & (b > 0) & (c > 0)).sum())
                        n1 = int(((d_ > 0) & (e_ > 0) & (f_ > 0)).sum())
                    except Exception:                     # noqa: BLE001
                        n0 = n1 = None
                    if n0 is not None and n1 < n0:
                        ST.record_metric(db, seg, "guard_invariant_violation", float(n0 - n1), stage="grow", attempt_id=attempt_id,
                                         text=f"round {rnd}: {n1} cells < round-start {n0} cells -- reverted, growth_guard bug")
                        ST.record_artifact(db, seg, "grow", "tifxyz", round_start_ckpt, attempt_id)
                        rec["guard_invariant_violation"] = n0 - n1
                        append_round_record(out_dir, rec)
                        cur, area = round_start_ckpt, round_start_area
                        return done("done", "guard_invariant_violation")
                if cur != pre_guard_cur and cfg.ct_zarr:
                    try:
                        sc2 = SC.score_checkpoint(cur, gi.sampler)
                        last_score = sc2
                        ST.record_metric(db, seg, "verified_cm2", sc2.verified_cm2, stage="grow", attempt_id=attempt_id,
                                         text="post_guard_rescore_box_claim")
                        rec.update({"material_frac": sc2.material_frac, "onesheet_frac": sc2.onesheet_frac,
                                    "verified_cm2_boxclaim": sc2.verified_cm2})
                    except Exception as e:                # noqa: BLE001
                        ST.record_metric(db, seg, "score_error", text=f"post_guard_rescore: {type(e).__name__}: {str(e)[:140]}", stage="grow")
                rec["area_cm2_postguard"] = area
                rec["checkpoint"] = cur
                if ginfo.get("stop"):
                    ST.record_metric(db, seg, "guard_stop", text=str(ginfo["stop"]), stage="grow", attempt_id=attempt_id)
                    append_round_record(out_dir, rec)
                    if ginfo["stop"] == "nothing_left":
                        return Outcome("failed", "guard_nothing_left", cur, 0.0, 0.0, rounds)
                    return done("done", f"guard_{ginfo['stop']}")
            except Exception as e:                        # noqa: BLE001 - fail CLOSED when selfcross is on
                ST.record_metric(db, seg, "guard_error", text=f"{type(e).__name__}: {str(e)[:160]}", stage="grow")
                rec["guard_error"] = f"{type(e).__name__}: {str(e)[:200]}"
                if pol.selfcross and pol.selfcross_fail_closed and not gg_judged:
                    o = _selfx_fail_closed(db, seg, attempt_id, None, cur, round_start_ckpt, round_start_area, area,
                                           last_score, rounds, f"guard raised {type(e).__name__}: {str(e)[:120]}")
                    rec.update({"checkpoint": o.checkpoint, "area_cm2_postguard": o.area_cm2})
                    append_round_record(out_dir, rec)
                    return o
        rec.setdefault("checkpoint", cur)
        rec.setdefault("area_cm2_postguard", area)
        append_round_record(out_dir, rec)
        cap = cfg.max_area_cm2 or (pol.max_area_cm2 if (gg_on and pol.grow_to_sane) else 0.0)
        if cap and area >= cap:
            return done("done", f"guard_max_area {area:.1f} cm2")
        if cur and n_this_call > 1 and gained * 100.0 < EXHAUSTED_MM2:
            return Outcome("exhausted", f"gained {gained * 100:.1f} mm2", cur, area,
                           last_score.verified_cm2 if last_score else None, rounds)
