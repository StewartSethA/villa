#!/usr/bin/env python3
"""Provider-agnostic fleet orchestrator for cloud-grow. DRY-RUN BY DEFAULT.

  fleet.py plan   PLAN.json                  shards, projected cost/time, quota hints (touches nothing)
  fleet.py up     PLAN.json --dry-run        print every provider command + the rendered bootstrap (secrets redacted)
  fleet.py up     PLAN.json --yes            execute -- ONLY if env FLEET_EXECUTE=1 as well (two interlocks), a spend cap exists, and RAM/plan validate
  fleet.py status PLAN.json [--probe]        spend-so-far model, instances, (with --probe) one ssh box_probe per box
  fleet.py collect PLAN.json [--into inbox]  print/run the pull from the object store / staging host to YOUR machine (no hub credentials)
  fleet.py down   PLAN.json --yes            terminate everything in the state file, then run the VERIFY command

Nothing in this tool signs up, rents or provisions by itself. Without FLEET_EXECUTE=1 every provider command is only printed.
No secret is stored in the plan or the state file: upload credentials are read from the env vars the plan NAMES, at `up` time.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import plan as P  # noqa: E402
from adapters import ADAPTERS, Cmd  # noqa: E402

SCROLLS_JSON = os.path.join(HERE, "..", "config", "scrolls.json")


class FleetError(RuntimeError):
    pass


# ---------------------------------------------------------------- runners
class DryRunner:
    """Prints, never executes. Used unless FLEET_EXECUTE=1 and --yes."""
    executes = False

    def __init__(self, out=print):
        self.out, self.log = out, []

    def run(self, cmd: Cmd, secrets=()):
        s = redact(cmd.shell(), secrets)
        self.log.append(s)
        self.out(f"  $ {s}" + (f"    # {cmd.note}" if cmd.note else ""))
        return 0, ""


class RealRunner:
    executes = True

    def __init__(self, out=print):
        self.out = out

    def run(self, cmd: Cmd, secrets=()):
        self.out(f"  $ {redact(cmd.shell(), secrets)}")
        try:
            r = subprocess.run(cmd.argv, capture_output=True, text=True, timeout=600)
        except (OSError, subprocess.SubprocessError) as e:
            return 127, f"{type(e).__name__}: {e}"
        return r.returncode, (r.stdout or "") + (("\n" + r.stderr) if r.returncode else "")


def redact(s: str, secrets) -> str:
    for v in secrets:
        if v:
            s = s.replace(v, "<REDACTED>")
    return s


# ---------------------------------------------------------------- rendering
def _scroll_meta(scroll: str) -> dict:
    d = json.load(open(SCROLLS_JSON))["scrolls"]
    if scroll not in d:
        raise FleetError(f"{scroll} not in config/scrolls.json")
    return d[scroll]


def run_config(plan: dict, b: dict, data_root="/data/vol") -> dict:
    m = _scroll_meta(b["scroll"])
    pz = m.get("prediction_zarr")
    if not pz:
        raise FleetError(f"{b['scroll']}: no prediction_zarr in scrolls.json (null = unknown, not guessed)")
    base = f"{data_root}/{b['scroll']}"
    box = plan["box"]
    return {"scroll": b["scroll"], "workdir": "/data/cloud-grow/run1", "kit_bin": "/opt/vc_kit/bin", "ct_zarr": f"{base}/{m['ct_zarr']}",
            "prediction_zarr": f"{base}/{pz}",
            "normal_grids": f"{base}/{pz[:-5] if pz.endswith('.zarr') else pz}.normal-grids",   # naming convention: UNVERIFIED per scroll; preflight fails if absent
            "umbilicus_json": "", "voxel_um": m["voxel_um"], "tracer_volume": "prediction", "spaceline_always": True, "step_size": 20.0,
            "first_gens": 35, "gen_step": 20, "thread_limit": 1, "rng_seed": b["rng_seed"],
            "grid_cache_bytes": int(box.get("grid_cache_bytes", 17179869184)), "wall_s": int(box.get("wall_s", 21600)), "max_rounds": 0, "max_area_cm2": 0,
            "policy_path": "", "ct_level_guard": 1, "allow_unpinned": False, "instance_id": b["box"], "release_sha": plan["code"].get("release_sha", "")}


def write_env_map(plan: dict) -> dict:
    """upload.env_vars: {box_var: LOCAL_ENV_VAR} (the box's AWS_ACCESS_KEY_ID <- your CG_WRITE_KEY_ID). A list means identity mapping."""
    e = plan["upload"].get("env_vars", {})
    return dict(e) if isinstance(e, dict) else {v: v for v in e}


def upload_env(plan: dict, b: dict, secrets: dict | None) -> tuple[str, list[str]]:
    """The box's env file. `secrets` None = redacted preview. Returns (text, secret values for redaction)."""
    up = plan["upload"]
    parallel = int(plan["box"].get("grows", plan["box"]["cores"]))
    lines = [f"UPLOAD_KIND={up['kind']}", f"UPLOAD_DEST={up['dest']}", f"SEEDS_PER_BATCH={b['seeds_per_batch']}", f"PARALLEL={parallel}",
             f"RNG_SEED={b['rng_seed']}", f"ZMIN={b['zmin'] if b['zmin'] is not None else ''}", f"ZMAX={b['zmax'] if b['zmax'] is not None else ''}"]
    vals = []
    for boxvar, local in write_env_map(plan).items():
        v = (secrets or {}).get(local)
        if secrets is not None and not v:
            raise FleetError(f"env var {local} (named in plan.upload.env_vars) is not set")
        lines.append(f"{boxvar}={v if secrets is not None else '<REDACTED from $' + local + '>'}")
        vals.append(v or "")
    return "\n".join(lines), vals


def render_bootstrap(plan: dict, b: dict, hb: dict, secrets: dict | None = None) -> tuple[str, list[str]]:
    tmpl = open(os.path.join(HERE, "bootstrap.sh.tmpl")).read()
    env, vals = upload_env(plan, b, secrets)
    rc = json.dumps(run_config(plan, b), indent=1)
    parallel = int(plan["box"].get("grows", plan["box"]["cores"]))
    mem_max = f"{int(plan['box']['ram_gb'] * 0.9)}G"
    rep = {"BOX": b["box"], "SCROLL": b["scroll"], "MINUTES": str(max(1, int(hb["hours"] * 60))), "CODE_URL": plan["code"]["url"], "CODE_SHA": plan["code"]["sha256"],
           "KIT_URL": plan["kit"]["url"], "KIT_SHA": plan["kit"]["sha256"], "UPLOAD_ENV": env, "RUN_JSON": rc, "PARALLEL": str(parallel), "MEMORY_MAX": mem_max}
    for k, v in rep.items():
        tmpl = tmpl.replace(f"@@{k}@@", v)
    if "@@" in tmpl:
        raise FleetError("unreplaced @@placeholder@@ in bootstrap")
    return tmpl, vals


def write_bootstrap(outdir: str, plan, b, hb, secrets):
    os.makedirs(outdir, exist_ok=True)
    txt, vals = render_bootstrap(plan, b, hb, secrets)
    path = os.path.join(outdir, f"{b['box']}.bootstrap.sh")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(txt)
    return path, vals


# ---------------------------------------------------------------- state
def state_path(args, plan) -> str:
    return args.state or os.path.join(args.outdir or ".", f"{plan['name']}.fleet_state.json")


def load_state(path: str) -> dict:
    return json.load(open(path)) if os.path.isfile(path) else {}


def save_state(path: str, st: dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    json.dump(st, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


def spend_so_far(plan: dict, st: dict, now=None) -> dict:
    """MODEL, not a provider bill: n live boxes x $/h x hours since `started_utc`."""
    now = now or dt.datetime.now(dt.timezone.utc)
    if not st.get("started_utc"):
        return {"usd": 0.0, "hours": 0.0}
    h = max(0.0, (now - P._utc(st["started_utc"])).total_seconds() / 3600.0)
    n = len(st.get("shards", []))
    return {"usd": round(h * n * plan["box"]["usd_per_hour"], 2), "hours": round(h, 2), "basis": "model: boxes x plan $/h x wall hours; NOT the provider's bill"}


# ---------------------------------------------------------------- commands
def cmd_plan(args, plan, out=print) -> int:
    sh = P.shards(plan)
    pr = P.projection(plan)
    ad = ADAPTERS[plan["provider"]](plan, args.outdir or "./fleet_out")
    out(f"plan {plan['name']}: provider={plan['provider']} region={plan['region']} boxes={len(sh)}")
    out(P.fmt_projection(pr))
    out(json.dumps({k: pr[k] for k in ("fleet_cm2_per_hour_prod_equiv", "per_scroll", "usd_per_100cm2")}, indent=1))
    for b in sh:
        out(f"  {b['box']}: {b['scroll']} shard {b['shard'] + 1}/{b['of']} z=[{b['zmin']},{b['zmax']}) rng_seed={b['rng_seed']} ({b['shard_kind']})")
    if any(b["zmin"] is None for b in sh):
        out("ANNOUNCE: boxes without z_extent only get distinct rng seeds; the seeder cannot see other boxes' coverage, so near-duplicate seeds are possible.")
    out("ANNOUNCE: z-band shards are disjoint at the SEED; a grown sheet may cross a band edge. Overlap is not measured here; the importer registers rows, it does not merge.")
    pred = {s["scroll"]: (_scroll_meta(s["scroll"]).get("prediction_bytes") or 0) / 1e9 for s in plan["scrolls"]}
    out("data per box (prediction only; + grids ~12 GB + CT L1-5 ~36 GB, sizes at fetch preview): " + ", ".join(f"{k} {v:.1f} GB" for k, v in pred.items()))
    out("quota / price checks to run YOURSELF before `up`:")
    for q in ad.quota_hints():
        out("  " + q)
    out("D6: nothing is validated against human annotation; self-crossing guard must run fail-closed (it does: production policy) and the importer rejects unverifiable surfaces.")
    return 0


def _interlocks(args) -> None:
    if not args.yes:
        raise FleetError("refusing: pass --yes (or use --dry-run)")
    if os.environ.get("FLEET_EXECUTE") != "1":
        raise FleetError("refusing: set FLEET_EXECUTE=1 in the environment as well (second interlock)")


def cmd_up(args, plan, runner=None, out=print) -> int:
    cap = plan["spend_cap_usd"]
    if args.cap is not None:
        cap = min(cap, args.cap)           # a CLI cap can only LOWER the plan's cap
        plan = {**plan, "spend_cap_usd": cap}
    if not cap or cap <= 0:
        raise FleetError("refusing: no spend cap")
    sh = P.shards(plan)
    pr = P.projection(plan)
    out(P.fmt_projection(pr))
    if pr["hours"] <= 0:
        raise FleetError("refusing: deadline already passed / zero horizon")
    if pr["max_compute_usd"] > cap + 1e-9:
        raise FleetError(f"refusing: max compute ${pr['max_compute_usd']} > cap ${cap}")
    if pr["binding"] == "spend cap":
        out(f"NOTE: the spend cap binds before the deadline: boxes self-destruct after {pr['hours']} h, not at the deadline.")
    dry = args.dry_run or runner is None and not (args.yes and os.environ.get("FLEET_EXECUTE") == "1")
    if not args.dry_run:
        _interlocks(args)
    outdir = args.outdir or "./fleet_out"
    secrets = None if (dry or args.dry_run) else {v: os.environ.get(v, "") for v in write_env_map(plan).values()}
    allsecrets: list[str] = []
    for b in sh:
        path, vals = write_bootstrap(outdir, plan, b, pr, secrets)
        allsecrets += vals
    ad = ADAPTERS[plan["provider"]](plan, outdir)
    cmds = ad.up_cmds(sh)
    runner = runner or (DryRunner(out) if (dry or args.dry_run) else RealRunner(out))
    out(f"{'DRY-RUN: ' if not runner.executes else ''}{len(cmds)} provider command(s); bootstrap files in {outdir}/ ({'redacted preview' if secrets is None else 'CONTAINS the write-only upload credential, mode 0600: delete after up'})")
    st = {"plan": plan["name"], "provider": plan["provider"], "shards": sh, "instances": {}, "started_utc": None}
    for c in cmds:
        rc, o = runner.run(c, allsecrets)
        if runner.executes:
            if rc != 0:
                out(f"FAILED rc={rc}: {redact(o[:300], allsecrets)}  -> stopping; run `down` to clean up what was created")
                save_state(state_path(args, plan), st)
                return 1
            if c.parse and c.box:
                tok = (o.strip().split() or [""])[-1]
                if c.parse.startswith("json:"):
                    try:
                        tok = str(json.loads(o)[c.parse[5:]])
                    except (ValueError, KeyError, TypeError):
                        tok = ""
                st["instances"][c.box] = tok if c.parse != "name" else c.box
    if runner.executes:
        st["started_utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        save_state(state_path(args, plan), st)
        for b in sh:
            try:
                os.remove(os.path.join(outdir, f"{b['box']}.bootstrap.sh"))      # credential-bearing file: do not leave it
            except OSError:
                pass
        out(f"state: {state_path(args, plan)}. NEXT: start watchdog.py; smoke-test ONE box for one hour before scaling (README).")
    return 0


def resolve_targets(plan, ad, st, runner) -> dict:
    outs = []
    for c in ad.resolve_cmds(st):
        rc, o = runner.run(c)
        outs.append(o if rc == 0 else "")
    return ad.parse_resolve(outs, st)


def cmd_status(args, plan, runner=None, out=print) -> int:
    st = load_state(state_path(args, plan))
    sp = spend_so_far(plan, st)
    out(json.dumps({"plan": plan["name"], "boxes": len(st.get("shards", [])), "instances": st.get("instances", {}), "spend_model": sp, "cap_usd": plan["spend_cap_usd"]}))
    if not st:
        out("no state file: nothing was `up`ed from this machine")
        return 0
    runner = runner or (RealRunner(out) if os.environ.get("FLEET_EXECUTE") == "1" else DryRunner(out))
    ad = ADAPTERS[plan["provider"]](plan, args.outdir or "./fleet_out")
    targets = resolve_targets(plan, ad, st, runner)
    out(f"targets: {json.dumps(targets)}")
    if args.probe:
        out("probe with: watchdog.py --plan PLAN.json --once   (it owns the probe + judgement)")
    return 0


def cmd_collect(args, plan, runner=None, out=print) -> int:
    up, into = plan["upload"], args.into or "./inbox"
    runner = runner or (RealRunner(out) if os.environ.get("FLEET_EXECUTE") == "1" else DryRunner(out))
    if up["kind"] == "s3":
        c = Cmd(["aws", "s3", "sync", up["dest"].rstrip("/") + "/", into, "--only-show-errors"], "needs a READ key (separate from the boxes' write-only keys)")
    else:
        c = Cmd(["rsync", "-a", "--partial", up["dest"].rstrip("/") + "/", into + "/"], "from the staging host you designated")
    rc, _ = runner.run(c)
    out("then, on the hub (hub-initiated, no hub token on any box):\n  hub/import_remote_grow.py --registry ./remote_registry --voxel-um <V> --kit-bin <hub kit>/bin --dry-run %s/*/*.tar.gz\n  (drop --dry-run once PASS rows look right; REFUSED rows name their reason, incl. selfx_unverified / selfx unrunnable)" % into)
    return rc


def cmd_down(args, plan, runner=None, out=print) -> int:
    if not args.dry_run:
        _interlocks(args)
    st = load_state(state_path(args, plan))
    if not st:
        raise FleetError("no state file: refusing to guess what to terminate (use the provider console / tag cloud-grow=%s)" % plan["name"])
    runner = runner or (DryRunner(out) if args.dry_run or os.environ.get("FLEET_EXECUTE") != "1" else RealRunner(out))
    ad = ADAPTERS[plan["provider"]](plan, args.outdir or "./fleet_out")
    worst = 0
    for c in ad.down_cmds(st):
        rc, o = runner.run(c)
        worst = max(worst, rc)
        if runner.executes and "VERIFY" in c.note:
            out(f"VERIFY output (must be empty): {o.strip()[:300] or '(empty)'}")
            if not ad.verify_empty(o):
                out("!!! INSTANCES REMAIN for this plan: they are BILLING. Terminate by hand in the provider console NOW.")
                worst = max(worst, 4)
    out("after down: also check the provider console for orphaned disks/IPs/snapshots, and that the object-store key of this rental is deleted.")
    return worst


def main(argv=None, runner=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("plan", "up", "status", "collect", "down"):
        s = sub.add_parser(n)
        s.add_argument("plan")
        s.add_argument("--state", default="")
        s.add_argument("--outdir", default="")
        s.add_argument("--yes", action="store_true")
        s.add_argument("--dry-run", action="store_true")
        s.add_argument("--cap", type=float, default=None, help="spend cap in USD; can only lower the plan's cap")
        s.add_argument("--probe", action="store_true")
        s.add_argument("--into", default="")
    a = ap.parse_args(argv)
    try:
        plan = P.load(a.plan)
        fn = {"plan": cmd_plan, "up": cmd_up, "status": cmd_status, "collect": cmd_collect, "down": cmd_down}[a.cmd]
        return fn(a, plan, **({"runner": runner} if a.cmd != "plan" else {}))
    except (P.PlanError, FleetError) as e:
        print(f"fleet: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
