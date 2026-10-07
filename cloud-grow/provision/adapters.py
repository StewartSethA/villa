"""Provider adapters: THIN. Each turns (plan, shards, rendered bootstrap) into argv lists; nothing here executes anything.

Execution goes through a `Runner` (fleet.py): `DryRunner` prints, `RealRunner` (only with FLEET_EXECUTE=1 and --yes) subprocesses.
Every adapter implements: up_cmds, resolve_cmds + parse_resolve, down_cmds, quota_hints. Commands are UNTESTED against real
provider CLIs (D6/D16 honesty): flags were written from memory of the CLIs' documented interface; run `up --dry-run` and read them.
"""
from __future__ import annotations

import json
import os
import shlex
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plan as P  # noqa: E402


class Cmd:
    def __init__(self, argv: list[str], note: str = "", box: str | None = None, parse: str | None = None):
        self.argv, self.note, self.box, self.parse = list(argv), note, box, parse   # parse: how to read stdout ("id" = first token)

    def shell(self) -> str:
        return " ".join(shlex.quote(a) for a in self.argv)


class Adapter:
    name = "base"

    def __init__(self, plan: dict, outdir: str):
        self.plan, self.outdir = plan, outdir

    def ud_path(self, box: str) -> str:
        return os.path.join(self.outdir, f"{box}.bootstrap.sh")

    # --- to implement
    def up_cmds(self, sh: list[dict]) -> list[Cmd]: raise NotImplementedError
    def resolve_cmds(self, state: dict) -> list[Cmd]: raise NotImplementedError
    def parse_resolve(self, outputs: list[str], state: dict) -> dict: raise NotImplementedError   # box -> ssh target
    def down_cmds(self, state: dict) -> list[Cmd]: raise NotImplementedError
    def quota_hints(self) -> list[str]: return []
    def cred_cmds(self) -> list[Cmd]: return []                 # READ-ONLY calls proving the credentials work
    def verify_empty(self, out: str) -> bool: return not out.strip()   # output of the VERIFY cmd when nothing is left

    def ssh_user(self) -> str:
        return self.plan.get("ssh_user", "ubuntu")

    def minutes(self, hb: dict) -> int:
        return max(1, int(hb["hours"] * 60))


class SshAdapter(Adapter):
    """User-supplied hosts (Hetzner/Latitude/OVH bare metal, anything with ssh + sudo). Nothing is created or billed by us."""
    name = "ssh"

    def _opts(self) -> list[str]:
        o = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=20"]
        if self.plan.get("ssh_key_path"):
            o += ["-i", self.plan["ssh_key_path"]]
        return o

    def up_cmds(self, sh):
        out = []
        for b in sh:
            h = self.plan["hosts"][b["index"]]
            out.append(Cmd(["scp", *self._opts(), self.ud_path(b["box"]), f"{h}:/tmp/cg-bootstrap.sh"], "copy bootstrap (rendered, 0600 when it holds the upload key)", b["box"]))
            out.append(Cmd(["ssh", *self._opts(), h, "nohup sudo -n bash /tmp/cg-bootstrap.sh >/dev/null 2>&1 &"], "start bootstrap detached; progress: /var/log/cloud-grow-bootstrap.log", b["box"]))
        return out

    def resolve_cmds(self, state): return []

    def parse_resolve(self, outputs, state):
        return {b["box"]: self.plan["hosts"][b["index"]] for b in state["shards"]}

    def down_cmds(self, state):
        out = []
        for b in state["shards"]:
            h = self.plan["hosts"][b["index"]]
            out.append(Cmd(["ssh", *self._opts(), h, "sudo -n touch /data/cloud-grow/STOP; sudo -n systemctl stop cloud-grow; sudo -n rm -rf /data/cloud-grow /data/vol /etc/cloud-grow.env /opt/cloud-grow /opt/vc_kit /tmp/cg-bootstrap.sh"],
                           "stop runner, WIPE our dirs + credential. This does NOT end the provider's billing: cancel the server in the provider console", b["box"]))
        return out

    def cred_cmds(self):
        return [Cmd(["ssh", *self._opts(), h, "sudo -n true && echo ok"], f"reach {h} and passwordless sudo") for h in self.plan["hosts"]]

    def verify_empty(self, out): return True      # ssh hosts are not ours to terminate: nothing to verify via API; cancel in the provider console

    def quota_hints(self): return ["ssh: no API quotas; check the provider's server limit and that sudo -n works without a password"]


class AwsAdapter(Adapter):
    name = "aws"

    def up_cmds(self, sh):
        a, r = self.plan["aws"], self.plan["region"]
        out = []
        for b in sh:
            argv = ["aws", "ec2", "run-instances", "--region", r, "--image-id", a["ami"], "--instance-type", a["instance_type"], "--count", "1",
                    "--instance-market-options", "MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}",
                    "--instance-initiated-shutdown-behavior", "terminate",
                    "--key-name", a["key_name"], "--security-group-ids", a["security_group_id"],
                    "--block-device-mappings", f"DeviceName=/dev/sda1,Ebs={{VolumeSize={int(self.plan['box']['disk_gb'])},VolumeType=gp3,DeleteOnTermination=true}}",
                    "--user-data", f"file://{self.ud_path(b['box'])}",
                    "--tag-specifications", f"ResourceType=instance,Tags=[{{Key=cloud-grow,Value={self.plan['name']}}},{{Key=Name,Value={b['box']}}},{{Key=cg-scroll,Value={b['scroll']}}}]",
                    "--query", "Instances[0].InstanceId", "--output", "text"]
            if a.get("subnet_id"):
                argv += ["--subnet-id", a["subnet_id"]]
            out.append(Cmd(argv, f"spot {a['instance_type']}; self-terminates at the horizon (shutdown -P in user-data + terminate-on-shutdown)", b["box"], parse="id"))
        return out

    def resolve_cmds(self, state):
        return [Cmd(["aws", "ec2", "describe-instances", "--region", self.plan["region"], "--filters", f"Name=tag:cloud-grow,Values={self.plan['name']}",
                     "Name=instance-state-name,Values=pending,running", "--query", "Reservations[].Instances[].[Tags[?Key==`Name`]|[0].Value,PublicIpAddress,InstanceId]", "--output", "text"], "box name, public ip, id")]

    def parse_resolve(self, outputs, state):
        res = {}
        for ln in (outputs[0] if outputs else "").splitlines():
            f = ln.split()
            if len(f) >= 2 and f[1] != "None":
                res[f[0]] = f"{self.ssh_user()}@{f[1]}"
        return res

    def down_cmds(self, state):
        ids = [i for i in state.get("instances", {}).values() if i]
        out = [Cmd(["aws", "ec2", "terminate-instances", "--region", self.plan["region"], "--instance-ids", *ids], "terminate (EBS DeleteOnTermination=true)")] if ids else []
        out.append(Cmd(["aws", "ec2", "describe-instances", "--region", self.plan["region"], "--filters", f"Name=tag:cloud-grow,Values={self.plan['name']}",
                        "Name=instance-state-name,Values=pending,running,stopping,stopped", "--query", "Reservations[].Instances[].InstanceId", "--output", "text"],
                       "VERIFY none left (a different observation than the terminate call); also check orphan volumes in the console"))
        return out

    def cred_cmds(self):
        return [Cmd(["aws", "sts", "get-caller-identity", "--region", self.plan["region"], "--output", "json"], "credentials valid? (read-only)"),
                Cmd(["aws", "service-quotas", "get-service-quota", "--region", self.plan["region"], "--service-code", "ec2", "--quota-code", "L-34B43A08", "--output", "json"],
                    f"spot vCPU quota; need >= {len(P.shards(self.plan)) * int(self.plan['aws'].get('vcpus', self.plan['box']['cores'] * 2))}", parse="quota")]

    def quota_hints(self):
        return [f"aws service-quotas get-service-quota --region {self.plan['region']} --service-code ec2 --quota-code L-34B43A08   # All Standard Spot Instance Requests (vCPUs). Usual blocker: new accounts start at ~1-32 vCPU; request an increase BEFORE planning (hours to days)",
                "aws ec2 describe-spot-price-history --region REGION --instance-types c7a.48xlarge --product-descriptions Linux/UNIX --max-items 5   # real spot price, then put it in box.usd_per_hour"]


class GcpAdapter(Adapter):
    name = "gcp"

    def up_cmds(self, sh):
        g, hb = self.plan["gcp"], P.hours_budget(self.plan)
        out = []
        for b in sh:
            out.append(Cmd(["gcloud", "compute", "instances", "create", b["box"], "--project", g["project"], "--zone", g["zone"], "--machine-type", g["machine_type"],
                            "--provisioning-model=SPOT", "--instance-termination-action=DELETE", f"--max-run-duration={self.minutes(hb) * 60}s",
                            "--image-family=ubuntu-2204-lts", "--image-project=ubuntu-os-cloud", f"--boot-disk-size={int(self.plan['box']['disk_gb'])}GB", "--boot-disk-type=pd-balanced",
                            f"--metadata-from-file=startup-script={self.ud_path(b['box'])}", f"--labels=cloud-grow={self.plan['name']},cg-scroll={b['scroll'].lower()}"],
                           "SPOT; --max-run-duration + DELETE = provider-side horizon (verify the flag exists in your gcloud version)", b["box"], parse="name"))
        return out

    def resolve_cmds(self, state):
        g = self.plan["gcp"]
        return [Cmd(["gcloud", "compute", "instances", "list", "--project", g["project"], f"--filter=labels.cloud-grow={self.plan['name']}", "--format=value(name,networkInterfaces[0].accessConfigs[0].natIP)"], "name, external ip")]

    def parse_resolve(self, outputs, state):
        res = {}
        for ln in (outputs[0] if outputs else "").splitlines():
            f = ln.split()
            if len(f) >= 2:
                res[f[0]] = f"{self.ssh_user()}@{f[1]}"
        return res

    def down_cmds(self, state):
        g = self.plan["gcp"]
        names = [b["box"] for b in state["shards"]]
        return [Cmd(["gcloud", "compute", "instances", "delete", *names, "--project", g["project"], "--zone", g["zone"], "--quiet"], "delete (boot disk auto-deletes by default)"),
                Cmd(["gcloud", "compute", "instances", "list", "--project", g["project"], f"--filter=labels.cloud-grow={self.plan['name']}", "--format=value(name)"], "VERIFY none left")]

    def cred_cmds(self):
        return [Cmd(["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"], "an active account? (read-only)"),
                Cmd(["gcloud", "compute", "regions", "describe", self.plan["region"], "--project", self.plan["gcp"]["project"], "--format=value(quotas)"], "quota listing (read-only; read PREEMPTIBLE_CPUS / C3D_CPUS yourself)")]

    def quota_hints(self):
        return [f"gcloud compute regions describe {self.plan['region']} --project PROJECT --format='value(quotas)' | tr ';' '\\n' | grep -i -E 'PREEMPTIBLE_CPUS|C3D_CPUS|CPUS'   # spot vCPU quota is the usual blocker; request increases first"]


class VastAdapter(Adapter):
    """vast.ai: marketplace of mostly GPU hosts; GPU unused here, we rent them for cores. vCPU = threads (cores = vCPU/2), host RAM is shared
    with other tenants' limits; containers have no systemd (no MemoryMax) and the dead-man switch is weaker: `down` is REQUIRED at the horizon."""
    name = "vast"

    def search_cmd(self) -> Cmd:
        b = self.plan["box"]
        q = f"cpu_cores_effective>={int(b['cores']) * 2} cpu_ram>={int(b['ram_gb'])} disk_space>={int(b['disk_gb'])} inet_down>=500 reliability>0.98 rentable=true"
        return Cmd(["vastai", "search", "offers", q, "-o", "dph", "--raw"], "pick offers by $/h x PassMark (docs T3), then list offer ids in plan.vast.offer_ids; vCPU counts threads")

    def up_cmds(self, sh):
        v = self.plan.get("vast", {})
        offers = v.get("offer_ids") or []
        if len(offers) < len(sh):
            return [self.search_cmd()]          # not enough chosen offers: emit the search only, create nothing
        out = []
        for b, oid in zip(sh, offers):
            out.append(Cmd(["vastai", "create", "instance", str(oid), "--image", v.get("image", "ubuntu:22.04"), "--disk", str(int(self.plan["box"]["disk_gb"])),
                            "--onstart", self.ud_path(b["box"]), "--label", b["box"], "--raw"], "on-demand (not bid) by default; GPU unused", b["box"], parse="json:new_contract"))
        return out

    def resolve_cmds(self, state):
        return [Cmd(["vastai", "show", "instances", "--raw"], "json list: label, ssh_host, ssh_port, id")]

    def parse_resolve(self, outputs, state):
        res = {}
        try:
            for i in json.loads(outputs[0] or "[]"):
                if str(i.get("label", "")).startswith(self.plan["name"]) and i.get("ssh_host"):
                    res[i["label"]] = f"root@{i['ssh_host']} -p {i.get('ssh_port', 22)}"
        except (ValueError, TypeError, AttributeError):
            pass
        return res

    def down_cmds(self, state):
        return [Cmd(["vastai", "destroy", "instance", str(i)], "destroy (stopped instances still bill disk)", b) for b, i in state.get("instances", {}).items() if i] + \
               [Cmd(["vastai", "show", "instances", "--raw"], "VERIFY none left")]

    def cred_cmds(self):
        return [Cmd(["vastai", "show", "user", "--raw"], "API key valid + credit balance (read-only)")]

    def verify_empty(self, out):
        try:
            return not [i for i in json.loads(out or "[]") if str(i.get("label", "")).startswith(self.plan["name"])]
        except (ValueError, TypeError, AttributeError):
            return False       # unparseable = not verified empty (fail loud)

    def quota_hints(self): return ["vast: prepaid credit balance is the cap (set only what you accept losing); no vCPU quota, but host reliability varies"]


ADAPTERS = {"ssh": SshAdapter, "aws": AwsAdapter, "gcp": GcpAdapter, "vast": VastAdapter}
