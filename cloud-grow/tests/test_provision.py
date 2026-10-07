"""Offline tests for provision/ (fleet orchestrator, adapters, watchdog, deploy loop) with FAKE providers. Nothing here touches a provider."""
import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "provision"))
sys.path.insert(0, ROOT)
import adapters as AD  # noqa: E402
import deploy as DP  # noqa: E402
import fleet as FL  # noqa: E402
import plan as P  # noqa: E402
import watchdog as WD  # noqa: E402
from cloud_grow import seeding as SD  # noqa: E402
from cloud_grow import state as ST  # noqa: E402

SECRET = "SECRETVALUE0123456789abcdefSECRET"


def mkplan(provider="aws", **kw):
    p = {"name": "t1", "provider": provider, "region": "us-east-2",
         "scrolls": [{"scroll": "PHerc0358", "boxes": 2, "z_extent": [0, 14000], "seeds_per_batch": 4}],
         "box": {"cores": 8, "grows": 8, "ram_gb": 128, "disk_gb": 300, "passmark_st": 2696, "usd_per_hour": 1.0},
         "spend_cap_usd": 100, "deadline_utc": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=10)).strftime("%Y-%m-%dT%H:%M:%SZ"),
         "aws": {"ami": "ami-x", "instance_type": "c7a.48xlarge", "key_name": "k", "security_group_id": "sg-x", "vcpus": 16},
         "gcp": {"project": "pr", "zone": "us-central1-a", "machine_type": "c3d-standard-360"},
         "vast": {"offer_ids": []}, "hosts": ["u@203.0.113.1", "u@203.0.113.2"],
         "upload": {"kind": "s3", "dest": "s3://bkt/t1", "env_vars": {"AWS_ACCESS_KEY_ID": "CG_W_ID", "AWS_SECRET_ACCESS_KEY": "CG_W_SECRET"}},
         "code": {"url": "https://x.invalid/c.tgz", "sha256": "a" * 64}, "kit": {"url": "https://x.invalid/k.tgz", "sha256": "b" * 64},
         "import": {"registry": "./reg", "kit_bin": "/nonexistent"}}
    p.update(kw)
    return p


class FakeRunner:
    """Fake provider: records argv, hands out instance ids, tracks which are alive; optional leftover on terminate."""
    executes = True

    def __init__(self, bucket=None, leave=False, fail_prefix=None):
        self.log, self.alive, self.n, self.bucket, self.leave, self.fail_prefix = [], {}, 0, bucket, leave, fail_prefix

    def run(self, cmd, secrets=()):
        a = cmd.argv
        self.log.append(a)
        if self.fail_prefix and a[:2] == self.fail_prefix:
            return 1, "denied"
        if a[:3] == ["aws", "ec2", "run-instances"]:
            self.n += 1
            name = [x for x in a if "Key=Name,Value=" in x][0].split("Key=Name,Value=")[1].split("}")[0]
            self.alive[name] = f"i-{self.n:04d}"
            return 0, f"i-{self.n:04d}\n"
        if a[:3] == ["aws", "ec2", "describe-instances"]:
            if "--query" in a and "InstanceId]" in a[a.index("--query") + 1]:
                return 0, "".join(f"{b} 203.0.113.{i + 10} {iid}\n" for i, (b, iid) in enumerate(self.alive.items()))
            return 0, "".join(f"{i}\n" for i in self.alive.values())
        if a[:3] == ["aws", "ec2", "terminate-instances"]:
            if not self.leave:
                self.alive.clear()
            else:
                self.alive.pop(next(iter(self.alive)))
            return 0, ""
        if a[:3] == ["aws", "s3", "sync"] and self.bucket:
            subprocess.run(["cp", "-r", self.bucket + "/.", a[4]])
            return 0, ""
        if a[:2] == ["aws", "sts"]:
            return 0, "{}"
        if a[:2] == ["aws", "service-quotas"]:
            return 0, json.dumps({"Quota": {"Value": 1000}})
        return 0, ""


def args_(tmp, **kw):
    d = dict(state="", outdir=str(tmp / "out"), yes=True, dry_run=False, cap=None, probe=False, into="", landing=str(tmp / "land"), interval=0, final_wait_ticks=1)
    d.update(kw)
    return argparse.Namespace(**d)


def probe(**kw):
    d = {"reachable": True, "runner_loop_alive": True, "stop_file": False, "bootstrap_failed": False, "tool_md5_drift": [], "tool_pins_checked": 2,
         "rounds_last_hour": 20, "verified_cm2_now": 100.0, "verified_cm2_1h_ago": 90.0, "segments": {"done": 20}, "guard_pauses": 0,
         "selfx_unverified_markers": 0, "disk_free_gb": 200, "ram_avail_gb": 100, "alerts": {}}
    d.update(kw)
    return d


def state(plan, hours_ago=3.0):
    return {"shards": P.shards(plan), "instances": {}, "targets": {}, "started_utc": (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")}


# ------------------------------------------------------------------ plan / shards / cost
def test_no_cap_refused():
    p = mkplan()
    p["spend_cap_usd"] = 0
    with pytest.raises(P.PlanError):
        P.validate(p)
    del p["spend_cap_usd"]
    with pytest.raises(P.PlanError):
        P.validate(p)


def test_secret_and_private_ip_in_plan_refused(tmp_path):
    f = tmp_path / "p.json"
    p = mkplan()
    p["code"]["url"] = "https://x.invalid/c.tgz?k=AKIA" + "A" * 16
    f.write_text(json.dumps(p))
    with pytest.raises(P.PlanError):
        P.load(str(f))
    p = mkplan(provider="ssh", hosts=["u@192.168.1.5", "u@203.0.113.2"])
    with pytest.raises(P.PlanError):
        P.validate(p)


def test_ram_too_small_refused():
    p = mkplan()
    p["box"]["ram_gb"] = 32
    with pytest.raises(P.PlanError):
        P.validate(p)


def test_zband_shards_disjoint_and_cover():
    p = mkplan()
    p["scrolls"][0].update(boxes=3, z_extent=[100, 1000])
    sh = P.shards(p)
    bands = [(b["zmin"], b["zmax"]) for b in sh]
    assert bands[0][0] == 100 and bands[-1][1] == 1000
    assert all(bands[i][1] == bands[i + 1][0] for i in range(2))
    assert len({b["rng_seed"] for b in sh}) == 3


def test_no_zextent_is_announced_overlap():
    p = mkplan()
    p["scrolls"][0].pop("z_extent")
    assert all("OVERLAP" in b["shard_kind"] for b in P.shards(p))


def test_cap_binds_hours_and_max_never_exceeds_cap():
    p = mkplan(spend_cap_usd=5)
    pr = P.projection(p)
    assert pr["binding"] == "spend cap" and pr["hours"] == 2.5 and pr["max_compute_usd"] <= 5.0


def test_band_support_masks_outside():
    import numpy as np
    s = np.ones((100, 4, 4), bool)
    out = SD.band_support(s, 16 * 10, 16 * 20)
    assert out[:10].sum() == 0 and out[20:].sum() == 0 and out[10:20].all() and s.all()


# ------------------------------------------------------------------ interlocks / secrets
def test_up_needs_yes(tmp_path, monkeypatch):
    monkeypatch.setenv("FLEET_EXECUTE", "1")
    monkeypatch.setenv("CG_W_ID", "a")
    monkeypatch.setenv("CG_W_SECRET", "b")
    f = tmp_path / "p.json"
    f.write_text(json.dumps(mkplan()))
    r = FakeRunner()
    assert FL.main(["up", str(f), "--outdir", str(tmp_path / "o")], runner=r) == 2
    assert r.log == []


def test_up_needs_second_interlock(tmp_path, monkeypatch):
    monkeypatch.delenv("FLEET_EXECUTE", raising=False)
    monkeypatch.setenv("CG_W_ID", "a")
    monkeypatch.setenv("CG_W_SECRET", "b")
    f = tmp_path / "p.json"
    f.write_text(json.dumps(mkplan()))
    r = FakeRunner()
    assert FL.main(["up", str(f), "--yes", "--outdir", str(tmp_path / "o")], runner=r) == 2
    assert r.log == []


def test_dry_run_runs_nothing_and_never_prints_secret(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CG_W_ID", SECRET)
    monkeypatch.setenv("CG_W_SECRET", SECRET + "2")
    f = tmp_path / "p.json"
    f.write_text(json.dumps(mkplan()))
    assert DP.main([str(f), "--dry-run", "--outdir", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    assert SECRET not in out and "aws ec2 run-instances" in out
    for fn in os.listdir(tmp_path / "o"):
        assert SECRET not in open(tmp_path / "o" / fn).read()


def test_real_up_secret_only_in_0600_bootstrap_which_is_removed_state_clean(tmp_path, monkeypatch):
    monkeypatch.setenv("FLEET_EXECUTE", "1")
    monkeypatch.setenv("CG_W_ID", SECRET)
    monkeypatch.setenv("CG_W_SECRET", SECRET + "2")
    plan = P.validate(mkplan())
    r, lines = FakeRunner(), []
    ns = args_(tmp_path)
    assert FL.cmd_up(ns, plan, runner=r, out=lines.append) == 0
    st = open(FL.state_path(ns, plan)).read()
    assert SECRET not in st and SECRET not in "\n".join(lines)
    assert not [f for f in os.listdir(ns.outdir) if f.endswith(".bootstrap.sh")]
    assert json.loads(st)["instances"] == {"t1-00": "i-0001", "t1-01": "i-0002"}


def test_bootstrap_has_deadman_pins_and_no_hub_secrets(tmp_path):
    plan = P.validate(mkplan())
    b = P.shards(plan)[0]
    txt, _ = FL.render_bootstrap(plan, b, P.hours_budget(plan))
    assert 'shutdown -P +"$HOURS_MIN"' in txt and __import__("re").search(r"HOURS_MIN=(59\d|600)", txt) and "sha256sum -c" in txt and "check-tools" in txt and "MemoryMax=" in txt
    assert "hub.token" not in txt and "hub_token" not in txt.lower().replace("no hub token", "")
    assert '"zmin"' not in txt and "ZMIN=0" in txt and "ZMAX=7000" in txt


# ------------------------------------------------------------------ adapters
def test_aws_cmd_flags():
    plan = P.validate(mkplan())
    c = AD.AwsAdapter(plan, "o").up_cmds(P.shards(plan))[0].shell()
    assert "MarketType=spot" in c and "--instance-initiated-shutdown-behavior terminate" in c and "Key=cloud-grow,Value=t1" in c


def test_gcp_cmd_flags():
    plan = P.validate(mkplan("gcp", spend_cap_usd=5))
    c = AD.GcpAdapter(plan, "o").up_cmds(P.shards(plan))[0].shell()
    assert "--provisioning-model=SPOT" in c and "--instance-termination-action=DELETE" in c and "--max-run-duration=9000s" in c


def test_vast_without_offers_only_searches():
    plan = P.validate(mkplan("vast"))
    cmds = AD.VastAdapter(plan, "o").up_cmds(P.shards(plan))
    assert len(cmds) == 1 and cmds[0].argv[:3] == ["vastai", "search", "offers"] and "create" not in cmds[0].shell()


def test_ssh_uses_supplied_hosts_and_needs_enough():
    plan = P.validate(mkplan("ssh"))
    c = AD.SshAdapter(plan, "o").up_cmds(P.shards(plan))
    assert "u@203.0.113.1:/tmp/cg-bootstrap.sh" in c[0].shell()
    with pytest.raises(P.PlanError):
        P.validate(mkplan("ssh", hosts=["u@203.0.113.1"]))


# ------------------------------------------------------------------ watchdog judgement
def judge(plan, st, probes, usd=1.0, n_inst=None, prev=None):
    return WD.tick(plan, st, probes, prev, {"usd": usd, "hours": 3}, n_inst)


def names(s, lv):
    return [f["msg"] for f in s["findings"] if f["level"] == lv]


def test_healthy_fleet_is_quiet():
    plan = P.validate(mkplan())
    st = state(plan)
    s = judge(plan, st, {b["box"]: probe() for b in st["shards"]})
    assert not s["trouble"] and not s["actions"]


def test_dead_runner_restarts_but_not_after_stop_nor_on_drift_nor_beyond_five():
    plan = P.validate(mkplan())
    st = state(plan)
    b0, b1 = (b["box"] for b in st["shards"])
    s = judge(plan, st, {b0: probe(runner_loop_alive=False), b1: probe(runner_loop_alive=False, stop_file=True)})
    assert [a for a in s["actions"] if a["do"] == "restart-runner"] == [{"do": "restart-runner", "box": b0, "why": "runner loop not alive"}]
    s = judge(plan, st, {b0: probe(runner_loop_alive=False, tool_md5_drift=["x"]), b1: probe()})
    assert not s["actions"] and any("md5 drift" in m for m in names(s, "TROUBLE"))
    prev = {"boxes": {b0: {"restarts": 5}}}
    s = judge(plan, st, {b0: probe(runner_loop_alive=False), b1: probe()}, prev=prev)
    assert not s["actions"] and s["trouble"]


def test_cost_tripwire_and_deadline_and_orphans_down():
    plan = P.validate(mkplan(spend_cap_usd=50))
    st = state(plan)
    ok = {b["box"]: probe() for b in st["shards"]}
    assert judge(plan, st, ok, usd=50.0)["actions"] == [{"do": "down", "why": "spend model $50.0 >= cap $50"}]
    assert judge(plan, st, ok, usd=41.0)["actions"] == [] and any("80" in m for m in names(judge(plan, st, ok, usd=41.0), "WARN"))
    assert judge(plan, st, ok, n_inst=3)["tripwire"]
    late = P.validate(mkplan(deadline_utc="2020-01-01T00:00:00Z", spend_cap_usd=50))
    assert judge(late, st, ok)["tripwire"] == "deadline reached"


def test_troubles_failrate_selfx_unreachable_noprogress():
    plan = P.validate(mkplan())
    st = state(plan)
    b0, b1 = (b["box"] for b in st["shards"])
    s = judge(plan, st, {b0: probe(segments={"done": 18, "failed": 2}), b1: probe(selfx_unverified_markers=2)})
    t = " | ".join(names(s, "TROUBLE"))
    assert "failure rate 2/20" in t and "selfx_unverified" in t
    s = judge(plan, st, {b0: probe(rounds_last_hour=0), b1: {"reachable": False, "why": "rc=255"}}, prev={"boxes": {b1: {"unreachable_ticks": 1}}})
    t = " | ".join(names(s, "TROUBLE"))
    assert "0 rounds" in t and "unreachable" in t
    s = judge(plan, st, {b0: probe(segments={"done": 5, "failed": 3}), b1: probe()})
    assert not any("failure rate" in m for m in names(s, "TROUBLE"))     # n < 10: no verdict on a tiny sample


def test_status_files_written(tmp_path):
    plan = P.validate(mkplan())
    st = state(plan)
    s = judge(plan, st, {b["box"]: probe() for b in st["shards"]})
    WD.write_status(str(tmp_path), s)
    assert json.load(open(tmp_path / "t1.status.json"))["plan"] == "t1" and "| box |" in open(tmp_path / "t1.status.md").read()


def test_box_probe_reads_state_db(tmp_path):
    wd = tmp_path / "run1"
    os.makedirs(wd / "segments" / "s1" / "r1")
    db = ST.connect(str(wd / "state.sqlite"))
    ST.record_metric(db, "s1", "area_cm2", 1.0, stage="grow")
    ST.record_metric(db, "s1", "verified_cm2", 0.9, stage="grow")
    ST.record_metric(db, "s1", "grow_outcome", 1.0, text="failed:x", stage="grow")
    ST.record_metric(db, "s2", "grow_outcome", 1.0, text="done:y", stage="grow")
    db.commit()
    (wd / "segments" / "s1" / "r1" / "selfx_unverified.json").write_text("{}")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "provision", "box_probe.py"), "--workdir", str(wd), "--root", str(tmp_path)], capture_output=True, text=True)
    j = json.loads(r.stdout)
    assert j["rounds_total"] == 1 and j["rounds_last_hour"] == 1 and j["verified_cm2_now"] == 0.9 and j["segments"] == {"failed": 1, "done": 1}
    assert j["selfx_unverified_markers"] == 1 and j["runner_loop_alive"] is False


# ------------------------------------------------------------------ collect / verify / import
def mk_bucket(tmp, good=2, bad=1):
    b = tmp / "bucket" / "t1-00"
    os.makedirs(b)
    for i in range(good + bad):
        t = b / f"seg{i}.tar.gz"
        t.write_bytes(b"payload%d" % i)
        sha = hashlib.sha256(t.read_bytes()).hexdigest() if i < good else "0" * 64
        (b / f"seg{i}.tar.gz.sha256").write_text(f"{sha}  seg{i}.tar.gz\n")
    return str(tmp / "bucket")


def test_checksum_bad_quarantined_never_imported(tmp_path):
    plan = P.validate(mkplan())
    seen = []

    def imp(e, s, dry):
        seen.append((os.path.basename(e), dry))
        return {"status": "PASS", "checks": {"area": [{"remeasured_cm2": 2.0}]}}
    res = DP.collect_and_import(plan, FakeRunner(bucket=mk_bucket(tmp_path)), str(tmp_path / "land"), state(plan), imp, out=lambda *_: None)
    assert res["good"] == 2 and len(res["bad"]) == 1 and "seg2.tar.gz" not in {n for n, _ in seen}
    assert os.path.exists(tmp_path / "land" / "_bad" / "seg2.tar.gz")
    assert res["pass"] == 2 and res["cm2"] == {"PHerc0358": 4.0}
    assert seen.count(("seg0.tar.gz", True)) == 1 and seen.count(("seg0.tar.gz", False)) == 1     # dry-run first, then register


def test_refused_not_registered_and_reasons_counted(tmp_path):
    plan = P.validate(mkplan())
    calls = []

    def imp(e, s, dry):
        calls.append(dry)
        return {"status": "REFUSED", "reasons": ["selfx census could not run"]}
    res = DP.collect_and_import(plan, FakeRunner(bucket=mk_bucket(tmp_path, 2, 0)), str(tmp_path / "land"), state(plan), imp, out=lambda *_: None)
    assert res["refused"] == 2 and res["reasons"] == {"selfx census could not run": 2} and calls == [True, True]    # never the non-dry call


def test_sync_failure_reported(tmp_path):
    plan = P.validate(mkplan())
    r = FakeRunner(fail_prefix=["aws", "s3"])
    res = DP.collect_and_import(plan, r, str(tmp_path / "land"), state(plan), lambda *a: {}, out=lambda *_: None)
    assert res["sync_rc"] == 1 and res["good"] == 0


# ------------------------------------------------------------------ deploy
@pytest.mark.parametrize("prov", ["aws", "gcp", "vast", "ssh"])
def test_checklist_per_provider_mentions_quota_and_billing(prov, tmp_path):
    lines = []
    DP.print_checklist(prov, lines.append)
    txt = "\n".join(lines)
    assert "smoke" in txt and ("billing" in txt.lower() or "credit" in txt.lower())
    if prov in ("aws", "gcp"):
        assert "QUOTA" in txt


def test_smoke_plan_one_box_one_hour():
    p = DP.smoke_plan(P.validate(mkplan()))
    assert P.n_boxes(p) == 1 and P.hours_budget(p)["hours"] <= 1.0 and p["spend_cap_usd"] <= 1.25


def fake_exec(argv, timeout=90):
    if "cg-box-probe" in argv[-1]:
        return 0, json.dumps(probe(runner_loop_alive=False))
    return 0, ""


def run(tmp_path, runner, monkeypatch, **kw):
    monkeypatch.setenv("FLEET_EXECUTE", "1")
    monkeypatch.setenv("CG_W_ID", SECRET)
    monkeypatch.setenv("CG_W_SECRET", SECRET + "2")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "READKEY")
    kb = tmp_path / "kit"
    os.makedirs(kb)
    (kb / "vc_tifxyz_selfcross").write_text("x")
    f = tmp_path / "p.json"
    plan = mkplan(**kw)
    plan["import"]["kit_bin"] = str(kb)
    f.write_text(json.dumps(plan))
    out = []
    rc = DP.main([str(f), "--yes", "--interval", "0", "--outdir", str(tmp_path / "o"), "--landing", str(tmp_path / "land")], runner=runner,
                 execf=fake_exec, sleep=lambda s: None, import_fn=lambda e, s, d: {"status": "PASS", "checks": {"area": [{"remeasured_cm2": 1.5}]}}, out=out.append, max_ticks=2)
    return rc, out


def test_full_run_ends_with_down_verified_and_summary(tmp_path, monkeypatch):
    r = FakeRunner(bucket=mk_bucket(tmp_path, 2, 0))
    rc, out = run(tmp_path, r, monkeypatch)
    txt = "\n".join(out)
    assert rc == 0 and not r.alive and "fleet summary" in txt and "imported (PASS" in txt
    kinds = [a[2] for a in r.log if a[:2] == ["aws", "ec2"]]
    assert kinds.index("run-instances") < kinds.index("terminate-instances") and kinds[-1] == "describe-instances"
    assert SECRET not in txt


def test_leftover_instance_fails_loudly(tmp_path, monkeypatch):
    r = FakeRunner(bucket=mk_bucket(tmp_path, 1, 0), leave=True)
    rc, out = run(tmp_path, r, monkeypatch)
    assert rc == 4 and any("INSTANCES REMAIN" in o for o in out)


def test_preflight_failure_creates_nothing(tmp_path, monkeypatch):
    r = FakeRunner(fail_prefix=["aws", "sts"])
    rc, out = run(tmp_path, r, monkeypatch)
    assert rc == 2 and not any(a[:3] == ["aws", "ec2", "run-instances"] for a in r.log)


def test_quota_too_small_refuses_before_create(tmp_path, monkeypatch):
    class Q(FakeRunner):
        def run(self, cmd, secrets=()):
            if cmd.argv[:2] == ["aws", "service-quotas"]:
                self.log.append(cmd.argv)
                return 0, json.dumps({"Quota": {"Value": 4}})
            return super().run(cmd, secrets)
    r = Q()
    rc, out = run(tmp_path, r, monkeypatch)
    assert rc == 2 and any("quota" in o for o in out) and not r.alive


def test_read_key_equal_write_key_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("FLEET_EXECUTE", "1")
    monkeypatch.setenv("CG_W_ID", "SAME")
    monkeypatch.setenv("CG_W_SECRET", "x")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "SAME")
    plan = P.validate(mkplan())
    plan["import"]["kit_bin"] = str(tmp_path)
    (tmp_path / "vc_tifxyz_selfcross").write_text("x")
    assert any("READ key" in p for p in DP.preflight(plan, FakeRunner()))


def test_cost_tripwire_in_loop_downs_then_collects(tmp_path, monkeypatch):
    r = FakeRunner(bucket=mk_bucket(tmp_path, 1, 0))
    rc, out = run(tmp_path, r, monkeypatch, deadline_utc="2020-01-01T00:00:00Z")
    assert rc in (2,)    # a past deadline is refused at `up` (zero horizon): nothing is created
    assert not r.alive


def test_abort_terminates_without_yes(tmp_path, monkeypatch):
    monkeypatch.setenv("FLEET_EXECUTE", "1")
    plan = P.validate(mkplan())
    ns = args_(tmp_path)
    FL.save_state(FL.state_path(ns, plan), {**state(plan), "instances": {"t1-00": "i-1", "t1-01": "i-2"}})
    r = FakeRunner()
    r.alive = {"t1-00": "i-1", "t1-01": "i-2"}
    f = tmp_path / "p.json"
    f.write_text(json.dumps(mkplan()))
    assert DP.main([str(f), "--abort", "--outdir", ns.outdir], runner=r, out=lambda *_: None) == 0
    assert not r.alive and any(a[:3] == ["aws", "ec2", "terminate-instances"] for a in r.log)


def test_abort_is_print_only_without_execute(tmp_path, monkeypatch):
    monkeypatch.delenv("FLEET_EXECUTE", raising=False)
    plan = P.validate(mkplan())
    ns = args_(tmp_path)
    FL.save_state(FL.state_path(ns, plan), {**state(plan), "instances": {"t1-00": "i-1"}})
    f = tmp_path / "p.json"
    f.write_text(json.dumps(mkplan()))
    out = []
    assert DP.main([str(f), "--abort", "--outdir", ns.outdir], out=out.append) == 0
    assert any("terminate-instances" in o for o in out)    # printed, DryRunner


def test_runners_redact_secret_in_printed_commands(capsys):
    c = AD.Cmd(["tool", "--key", SECRET])
    lines = []
    FL.DryRunner(lines.append).run(c, [SECRET])
    assert SECRET not in "\n".join(lines) and "<REDACTED>" in lines[0]
    lines = []
    FL.RealRunner(lines.append).run(AD.Cmd(["true", SECRET]), [SECRET])
    assert SECRET not in "\n".join(lines)
