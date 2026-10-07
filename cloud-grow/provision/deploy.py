#!/usr/bin/env python3
"""ONE entry point: plan file + your provider credentials (env vars / files, never committed) -> unattended fleet run.

  deploy.sh PLAN --checklist          the ordered list of things only YOU can do (accounts, billing, quota, keys), with URLs/commands
  deploy.sh PLAN --dry-run            the full run printed, nothing touched, nothing executed
  deploy.sh PLAN --smoke  --yes       ONE box, ONE hour, same pipeline (do this first)
  deploy.sh PLAN --yes                the fleet: preflight -> up -> watchdog loop -> collect+verify+import -> finalize -> down -> verify none left
  deploy.sh PLAN --abort              terminate everything NOW (no --yes needed; FLEET_EXECUTE=1 still required to act)

LIVE EXECUTION IS UNTESTED (written without any provider account; only fake providers were exercised). Real calls need
--yes AND env FLEET_EXECUTE=1. Without both, every provider command is printed, never run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import fleet as FL  # noqa: E402
import plan as P  # noqa: E402
import watchdog as WD  # noqa: E402
from adapters import ADAPTERS, Cmd  # noqa: E402

GRACE_MIN = 15          # STOP the boxes this long before the horizon so the final upload lands before they die
CHECKLIST = {
    "common": [
        "Decide ONE scroll (or a few, as separate per-scroll boxes) and the dollar cap you can lose. Copy provision/plan.example.json, edit: scrolls, boxes, spend_cap_usd, deadline_utc.",
        "Publish the two verified downloads the boxes pull: the cloud-grow code tarball and the VC3D kit tarball (tools/make_portable_kit.sh). Put each at an https URL the boxes can reach and record its sha256 in plan.code / plan.kit (`sha256sum file.tgz`).",
        "Create the result store: an S3/R2/B2 bucket (or a staging VM with rsync). Create (a) one WRITE-ONLY key for the boxes, (b) one READ key kept only on this machine. Export the write key into the env vars named in plan.upload.env_vars and the read key per plan.upload.read_env_vars.",
        "Make sure the hub-side importer works locally: `python hub/import_remote_grow.py --help`, and set plan.import.kit_bin to a VC3D bin dir containing vc_tifxyz_selfcross (else every export is REFUSED: fail closed).",
        "Run `deploy.sh PLAN --dry-run` and read it. Then `deploy.sh PLAN --smoke --yes` (FLEET_EXECUTE=1 in the env): ONE box, ONE hour. Look at the status Markdown, then `deploy.sh PLAN` import summary: self-crossing rate and REFUSED reasons. Only then run the fleet.",
    ],
    "aws": [
        "Create an AWS account (https://aws.amazon.com/), add a payment method, set a budget alert: https://console.aws.amazon.com/billing/home#/budgets",
        "REQUEST SPOT vCPU QUOTA (the usual blocker; new accounts get ~1-32 vCPU; approval hours to days): https://console.aws.amazon.com/servicequotas/home/services/ec2/quotas  -> 'All Standard (A, C, D, H, I, M, R, T, Z) Spot Instance Requests' (L-34B43A08). Need boxes x vCPUs.",
        "Create an IAM user/role limited to ec2:RunInstances/Describe*/TerminateInstances (+ servicequotas:GetServiceQuota); create an access key: https://console.aws.amazon.com/iam/home#/users . Export AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (or AWS_PROFILE). Install the CLI: https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html",
        "Create an EC2 key pair + a security group allowing inbound ssh (22) from YOUR IP only; put key_name, security_group_id, ami (Ubuntu 22.04 for the region), instance_type in plan.aws; ssh_key_path in the plan.",
    ],
    "gcp": [
        "Create a Google Cloud project + billing account: https://console.cloud.google.com/billing  ; enable Compute Engine API: https://console.cloud.google.com/apis/library/compute.googleapis.com",
        "REQUEST QUOTA for C3D / PREEMPTIBLE CPUs in the region (usual blocker): https://console.cloud.google.com/iam-admin/quotas",
        "Install gcloud (https://cloud.google.com/sdk/docs/install), then `gcloud auth login` and `gcloud config set project PROJECT`. Put project, zone, machine_type in plan.gcp; ssh via `gcloud compute config-ssh` or set ssh_key_path.",
        "Allow ssh (tcp:22) from your IP in the VPC firewall: https://console.cloud.google.com/net-security/firewall-manager",
    ],
    "vast": [
        "Create a vast.ai account and add prepaid credit (that balance IS your cap; add only what you accept losing): https://cloud.vast.ai/billing/",
        "Create an API key: https://cloud.vast.ai/account/ ; `pip install vastai`; `vastai set api-key KEY` (or export VAST_API_KEY). Add your ssh PUBLIC key under Account.",
        "Run the printed `vastai search offers ...` (from --dry-run), pick offers (note: vast is GPU hosts; we rent the cores; vCPU = threads), put the offer ids in plan.vast.offer_ids.",
        "Accept vast's terms at first rental (the console prompts). vast has NO provider-side auto-terminate: the watchdog / `--abort` is the only horizon.",
    ],
    "ssh": [
        "Rent the servers yourself (Hetzner AX, Latitude.sh, OVH...): create the account, billing, accept the agreement, order N servers with Ubuntu 22.04 + your ssh key: they bill until YOU cancel them in the console.",
        "Put user@ip of each in plan.hosts (outside git: the plan holds your real IPs); set ssh_key_path. `ssh HOST sudo -n true` must work.",
        "After the run, cancel the servers in the provider console: `down` only wipes our directories and credential, it cannot end the rental.",
    ],
}


def checklist(provider: str) -> list[str]:
    return CHECKLIST["common"][:2] + CHECKLIST[provider] + CHECKLIST["common"][2:]


def print_checklist(provider: str, out=print) -> None:
    out(f"CHECKLIST ({provider}): everything the human must do. The script does the rest. Nothing below is done by deploy.sh.")
    for i, s in enumerate(checklist(provider), 1):
        out(f"{i:2d}. {s}")


def smoke_plan(plan: dict, now=None) -> dict:
    """ONE box (the first scroll's first shard), ONE hour, cap = 1.25 x one box-hour."""
    now = now or dt.datetime.now(dt.timezone.utc)
    s0 = dict(plan["scrolls"][0], boxes=1)
    rate = plan["box"]["usd_per_hour"]
    p = {**plan, "name": (plan["name"] + "-smoke")[:31], "scrolls": [s0], "deadline_utc": (now + dt.timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
         "spend_cap_usd": round(min(plan["spend_cap_usd"], rate * 1.25), 2)}
    if p["provider"] == "ssh":
        p["hosts"] = plan["hosts"][:1]
    return p


# ---------------------------------------------------------------- preflight of OUR side
def preflight(plan: dict, runner, env=os.environ, check_local=True) -> list[str]:
    problems = []
    ad = ADAPTERS[plan["provider"]](plan, "./fleet_out")
    for c in ad.cred_cmds():
        rc, o = runner.run(c)
        if runner.executes:
            if rc != 0:
                problems.append(f"credential/quota check failed ({c.argv[0]} {c.argv[1] if len(c.argv) > 1 else ''}): {o.strip()[:200]}")
            elif c.parse == "quota":
                try:
                    have = float(json.loads(o)["Quota"]["Value"])
                    need = len(P.shards(plan)) * int(plan["aws"].get("vcpus", plan["box"]["cores"] * 2))
                    if have < need:
                        problems.append(f"spot vCPU quota {have:.0f} < {need} needed: request an increase first (checklist)")
                except (ValueError, KeyError, TypeError):
                    problems.append("could not parse the spot quota answer: not verified")
    for var in FL.write_env_map(plan).values():
        if runner.executes and not env.get(var):
            problems.append(f"env var {var} (upload write key) not set")
    if runner.executes and plan["upload"]["kind"] == "s3":
        wk = [env.get(v) for v in FL.write_env_map(plan).values()]
        if env.get("AWS_ACCESS_KEY_ID") and env.get("AWS_ACCESS_KEY_ID") in wk:
            problems.append("the READ key (AWS_ACCESS_KEY_ID here) equals the boxes' WRITE key: use two different keys (a rented box must not hold the read key)")
    if plan["upload"]["kind"] == "s3":
        for var in plan["upload"].get("read_env_vars", []):
            if runner.executes and not env.get(var):
                problems.append(f"env var {var} (collect READ key) not set")
    imp = plan.get("import", {})
    if runner.executes and check_local:
        kb = imp.get("kit_bin", "")
        if not kb or not os.path.isfile(os.path.join(kb, "vc_tifxyz_selfcross")):
            problems.append("plan.import.kit_bin must contain vc_tifxyz_selfcross (the hub-side self-crossing re-check; without it every export is REFUSED)")
    return problems


# ---------------------------------------------------------------- collect + verify + import
def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_landing(landing: str) -> tuple[list[str], list[dict]]:
    """Checksum every tarball against its .sha256 sidecar; failures are quarantined (moved to _bad/), never imported."""
    good, bad = [], []
    for root, dirs, files in os.walk(landing):
        dirs[:] = [d for d in dirs if d not in ("_bad",)]
        for f in files:
            if not f.endswith(".tar.gz"):
                continue
            p = os.path.join(root, f)
            try:
                want = open(p + ".sha256").read().split()[0]
            except (OSError, IndexError):
                bad.append({"tar": p, "why": "no .sha256 sidecar (upload incomplete?)"})
                continue
            got = sha256_file(p)
            if got != want:
                bad.append({"tar": p, "why": f"sha256 mismatch {got[:12]} != {want[:12]}"})
                os.makedirs(os.path.join(landing, "_bad"), exist_ok=True)
                os.replace(p, os.path.join(landing, "_bad", os.path.basename(p)))
            else:
                good.append(p)
    return good, bad


def default_import(export: str, scroll: str, dry: bool, plan: dict) -> dict:
    imp = plan.get("import", {})
    vox = FL._scroll_meta(scroll)["voxel_um"]
    argv = [sys.executable, os.path.join(HERE, "..", "hub", "import_remote_grow.py"), "--registry", imp.get("registry", "./remote_registry"), "--voxel-um", str(vox),
            "--kit-bin", imp.get("kit_bin", "")] + (["--dry-run"] if dry else []) + [export]
    r = subprocess.run(argv, capture_output=True, text=True)
    for ln in reversed((r.stdout or "").splitlines()):
        try:
            return json.loads(ln)
        except ValueError:
            continue
    return {"status": "REFUSED", "reasons": [f"importer produced no verdict (rc={r.returncode}): {(r.stderr or '')[-200:]}"], "checks": {}}


def collect_and_import(plan, runner, landing, state, import_fn=None, out=print) -> dict:
    """Pull -> verify checksums -> importer dry-run -> importer register (append-only). Idempotent: re-import is a no-op in the importer."""
    import_fn = import_fn or (lambda e, s, d: default_import(e, s, d, plan))
    up = plan["upload"]
    os.makedirs(landing, exist_ok=True)
    src = up["dest"].rstrip("/") + "/"
    c = (Cmd(["aws", "s3", "sync", src, landing, "--only-show-errors"], "READ key, held only locally") if up["kind"] == "s3"
         else Cmd(["rsync", "-a", "--partial", src, landing + "/"], "staging host"))
    rc, o = runner.run(c)
    res = {"sync_rc": rc, "good": 0, "bad": [], "pass": 0, "refused": 0, "reasons": {}, "cm2": {}, "per_box": {}}
    if rc != 0:
        out(f"collect: sync failed rc={rc}: {o[:200]}")
        return res
    good, bad = verify_landing(landing)
    res["bad"], res["good"] = bad, len(good)
    shard_by_box = {b["box"]: b for b in state.get("shards", [])}
    for tar in sorted(good):
        box = os.path.relpath(tar, landing).split(os.sep)[0]
        scroll = shard_by_box.get(box, {}).get("scroll") or plan["scrolls"][0]["scroll"]
        v = import_fn(tar, scroll, True)
        if v.get("status") == "PASS":
            v = import_fn(tar, scroll, False)
        pb = res["per_box"].setdefault(box, {"pass": 0, "refused": 0, "cm2": 0.0})
        if v.get("status") == "PASS":
            res["pass"] += 1
            pb["pass"] += 1
            areas = (v.get("checks") or {}).get("area") or []
            cm2 = (areas[-1].get("remeasured_cm2") if areas else 0) or 0
            pb["cm2"] += cm2
            res["cm2"][scroll] = res["cm2"].get(scroll, 0.0) + cm2
        else:
            res["refused"] += 1
            pb["refused"] += 1
            for r in v.get("reasons") or ["(no reason given)"]:
                k = str(r)[:100]
                res["reasons"][k] = res["reasons"].get(k, 0) + 1
    return res


def fmt_summary(plan, state, imp, spend) -> str:
    L = [f"== fleet summary {plan['name']} ==", f"spend (MODEL: boxes x $/h x hours; NOT the bill): ${spend['usd']} over {spend['hours']} h, cap ${plan['spend_cap_usd']}",
         f"imported (PASS, hub re-measured lattice cm2 of each final checkpoint): {sum(imp['cm2'].values()):.2f} cm2 by scroll {json.dumps({k: round(v, 2) for k, v in imp['cm2'].items()})}",
         f"tarballs verified {imp['good']}, checksum-bad {len(imp['bad'])}, PASS {imp['pass']}, REFUSED {imp['refused']}"]
    tot = imp["pass"] + imp["refused"]
    if tot:
        L.append(f"refusal rate {imp['refused']}/{tot} tarballs = {imp['refused'] / tot:.0%} (n = {tot} exported segments; self-crossing rate is among the reasons)")
    for r, n in sorted(imp["reasons"].items(), key=lambda kv: -kv[1]):
        L.append(f"  REFUSED x{n}: {r}")
    for b, d in sorted(imp["per_box"].items()):
        L.append(f"  {b}: PASS {d['pass']} REFUSED {d['refused']} cm2 {d['cm2']:.2f}")
    L.append("D6: PASS = the box's claims reproduce on the hub; nothing here is validated against human annotation.")
    return "\n".join(L)


# ---------------------------------------------------------------- run
def resolve_and_save(plan, ad, st, runner, sp, tries=20, sleep=time.sleep, out=print) -> dict:
    n = len(st["shards"])
    for i in range(tries):
        t = FL.resolve_targets(plan, ad, st, runner)
        if len(t) >= n:
            st["targets"] = t
            FL.save_state(sp, st)
            return t
        out(f"waiting for {n - len(t)} box address(es) ({i + 1}/{tries})")
        sleep(30)
    st["targets"] = t
    FL.save_state(sp, st)
    return t


def run_fleet(plan, args, runner, execf=WD.real_exec, sleep=time.sleep, import_fn=None, out=print, max_ticks=10 ** 9) -> int:
    outdir = args.outdir
    sp = FL.state_path(args, plan)
    probs = preflight(plan, runner)
    if probs:
        out("PREFLIGHT FAILED (nothing was created):\n  - " + "\n  - ".join(probs))
        return 2
    ns = argparse.Namespace(**{**vars(args), "dry_run": False, "cap": None})
    rc = FL.cmd_up(ns, plan, runner=runner, out=out)
    st = FL.load_state(sp)
    ad = ADAPTERS[plan["provider"]](plan, outdir)
    if rc != 0:
        out("up failed: tearing down whatever was created")
        FL.cmd_down(ns, plan, runner=runner, out=out)
        return 1
    resolve_and_save(plan, ad, st, runner, sp, sleep=sleep, out=out)
    pr = P.projection(plan, P._utc(st["started_utc"]))
    horizon_s = pr["hours"] * 3600
    stop_at = max(0.0, horizon_s - GRACE_MIN * 60)
    t0 = time.monotonic()
    prev, ticks, exit_code = None, 0, 0
    down_done = False
    landing = args.landing
    while ticks < max_ticks:
        ticks += 1
        spend = FL.spend_so_far(plan, st)
        probes = WD.probe_all(st.get("targets", {}), plan.get("ssh_key_path", ""), execf)
        n_inst = None
        s = WD.tick(plan, st, probes, prev, spend, n_inst)
        WD.write_status(outdir, s)
        out(f"[tick {ticks}] {'TRIPWIRE ' + s['tripwire'] if s['tripwire'] else ('TROUBLE' if s['trouble'] else 'ok')}; reachable {s['fleet']['reachable']}/{s['fleet']['boxes']}; spend model ${spend['usd']}/{plan['spend_cap_usd']}")
        for f in s["findings"]:
            out(f"    {f['level']} {f['box'] or 'fleet'}: {f['msg']}")
        down_fn = lambda: FL.cmd_down(ns, plan, runner=runner, out=out)  # noqa: E731
        if s["tripwire"]:
            exit_code = max(exit_code, 3)
            for d in WD.apply_actions(plan, st, s, plan.get("ssh_key_path", ""), execf, down_fn):
                out(f"    [auto] {d}")
                if d.startswith("down rc=") and d != "down rc=0":
                    exit_code = max(exit_code, 4)
            down_done = True
            break
        for d in WD.apply_actions(plan, st, s, plan.get("ssh_key_path", ""), execf, down_fn):
            out(f"    [auto] {d}")
        prev = s
        imp = collect_and_import(plan, runner, landing, st, import_fn, out)
        out(f"    collect: tarballs verified {imp['good']} (PASS {imp['pass']}, REFUSED {imp['refused']}, checksum-bad {len(imp['bad'])})")
        if s["trouble"]:
            exit_code = max(exit_code, 1)
        elapsed = time.monotonic() - t0
        if elapsed >= stop_at or _elapsed_model(st) >= stop_at:
            break
        sleep(args.interval)
    if not down_done:
        out("finalize: STOP all runners, wait for the final upload, then down")
        for b, t in st.get("targets", {}).items():
            execf(WD.ssh_argv(t, "sudo -n touch /data/cloud-grow/STOP || touch /data/cloud-grow/STOP", plan.get("ssh_key_path", "")))
        for _ in range(int(getattr(args, "final_wait_ticks", 10))):
            pr_ = WD.probe_all(st.get("targets", {}), plan.get("ssh_key_path", ""), execf)
            if not any(p.get("runner_loop_alive") for p in pr_.values()):
                break
            sleep(60)
        rcd = FL.cmd_down(ns, plan, runner=runner, out=out)
        if rcd != 0:
            exit_code = max(exit_code, 4)
    imp = collect_and_import(plan, runner, landing, st, import_fn, out)       # object store / staging outlive the boxes
    out(fmt_summary(plan, st, imp, FL.spend_so_far(plan, st)))
    if exit_code == 4:
        out("!!! EXIT 4: instances may remain and bill. Check the provider console NOW.")
    return exit_code


def _elapsed_model(st) -> float:
    return (dt.datetime.now(dt.timezone.utc) - P._utc(st["started_utc"])).total_seconds() if st.get("started_utc") else 0.0


def cmd_abort(plan, args, runner, out=print) -> int:
    ns = argparse.Namespace(**{**vars(args), "dry_run": not getattr(runner, "executes", False), "yes": True})
    out("ABORT: terminating everything now (no final collect; whatever the boxes already uploaded is kept in the store)")
    return FL.cmd_down(ns, plan, runner=runner, out=out)


def dry_run(plan, args, out=print) -> int:
    r = FL.DryRunner(out)
    out("=== DRY RUN: nothing is touched ===")
    FL.cmd_plan(args, plan, out)
    out("--- 1. preflight (read-only provider calls, run for real only with --yes + FLEET_EXECUTE=1)")
    preflight(plan, r)
    out("--- 2. provision + bootstrap (user-data / ssh), shards as above")
    ns = argparse.Namespace(**{**vars(args), "dry_run": True, "yes": False, "cap": None})
    FL.cmd_up(ns, plan, runner=r, out=out)
    out(f"--- 3. loop every {args.interval}s: watchdog probe (ssh box_probe.py) -> judge -> status JSON/MD in {args.outdir}/ ; auto-actions: restart-runner, down at cap/deadline/instance tripwire")
    out(f"--- 4. collect every tick -> {args.landing}: sync, sha256 verify (bad -> _bad/), importer --dry-run then register (append-only)")
    FL.cmd_collect(argparse.Namespace(into=args.landing), plan, runner=r, out=out)
    out(f"--- 5. finalize {GRACE_MIN} min before the horizon: STOP runners, wait for final upload, down, VERIFY none left (fail loud, exit 4)")
    st = {"shards": P.shards(plan), "instances": {b["box"]: f"<id-{b['box']}>" for b in P.shards(plan)}}
    for c in ADAPTERS[plan["provider"]](plan, args.outdir).down_cmds(st):
        r.run(c)
    out("--- UNTESTED LIVE: none of the provider commands above has ever been run against a real account.")
    return 0


def main(argv=None, runner=None, execf=WD.real_exec, sleep=time.sleep, import_fn=None, out=print, max_ticks=10 ** 9) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan")
    ap.add_argument("--checklist", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--abort", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--cap", type=float, default=None)
    ap.add_argument("--interval", type=int, default=600)
    ap.add_argument("--outdir", default="./fleet_out")
    ap.add_argument("--landing", default="./landing")
    ap.add_argument("--state", default="")
    a = ap.parse_args(argv)
    try:
        plan = P.load(a.plan)
        if a.checklist:
            print_checklist(plan["provider"], out)
            return 0
        if a.smoke:
            plan = smoke_plan(plan)
            out(f"SMOKE: 1 box, 1 hour, cap ${plan['spend_cap_usd']}")
        if a.cap is not None:
            plan = {**plan, "spend_cap_usd": min(plan["spend_cap_usd"], a.cap)}
        a.final_wait_ticks = 10
        if a.dry_run:
            return dry_run(plan, a, out)
        live = os.environ.get("FLEET_EXECUTE") == "1"
        if a.abort:
            return cmd_abort(plan, a, runner or (FL.RealRunner(out) if live else FL.DryRunner(out)), out)
        FL._interlocks(a)
        return run_fleet(plan, a, runner or FL.RealRunner(out), execf, sleep, import_fn, out, max_ticks)
    except (P.PlanError, FL.FleetError) as e:
        print(f"deploy: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
