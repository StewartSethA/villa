# provision/: push-button fleet for cloud-grow (DRY-RUN-ONLY so far)

**STATUS (2026-10-06): written and tested offline with FAKE providers only. No provider account was used, nothing was rented, no real `aws`/`gcloud`/`vastai`/`ssh` command has ever been run by this code. Every provider flag is from documentation memory and unverified live. No real VC3D binary was run (cloud-grow's own limit, TESTING.md). Nothing is validated against human annotation (D6).**

## The honest answer to "how easily can I get the Route A growths done within a day with a small CPU fleet?"

**Easily to start, not to finish in a day, and not for "all scrolls".** The script removes the operating work; it cannot remove the account, quota and verification steps.

| step | who | time |
|---|---|---|
| accounts, billing, budget alert, agreements | you | 30-60 min (AWS/GCP new accounts may also need identity/card verification) |
| **spot vCPU quota request** (the usual blocker: new accounts start at ~1-32 vCPU; 16 x 96-core = ~3000 vCPU) | you, then wait | 5 min to file; **hours to days** to be granted, partial grants common. Bare-metal/ssh hosts (Hetzner, Latitude) and vast.ai have no vCPU quota but have stock/limits |
| API key / IAM user, ssh key, security group, bucket + write-only key + read key | you | 30-45 min |
| publish the code tarball and the VC3D kit tarball (kit build is documented in tools/BUILD_TOOLS.md and has never been done end-to-end by us; tracer md5 reproducibility off the source fleet unknown) | you | 30 min if the kit exists, **unknown (hours) if it must be built** |
| edit the plan, `deploy.sh PLAN --dry-run`, read it | you | 15-20 min |
| **smoke: `--smoke --yes` = ONE box, ONE hour**, then read the import summary | script + you | ~1 h run + 30 min reading |
| the fleet | script; you watch the status Markdown | the run itself |

Human hands-on work from nothing to "16 boxes growing": **about 3-4 hours** (excluding the quota wait and the unknown kit build). Realistic calendar: day 1 accounts + quota request + kit; day 2 smoke; day 3 launch. "Within a day" is possible only for the part after the smoke test.

**What "a day" buys** (model docs/experiments/cloud_cost_2026-10-06: EXTRAPOLATED +-25 %, calibration factor 0.13-0.33, box claims not validated): 16 x 96-core boxes is about **1.1 days for the LOW all-scroll target at x1**. So plan per scroll (one scroll = its own boxes, its own data pull of 5-87 GB per box), not "all scrolls" in a day. `fleet.py plan` prints the model's low/mid/high verified cm2 and the dollar line for your plan before anything is created.

**What can break** (all unmeasured live): (1) no real binaries tested: the first real `grow` may fail in ways the fakes cannot show; (2) tracer md5 reproducibility on a new CPU is unknown, so the pin gate refuses to start on a mismatch (by design); (3) spot preemption: a terminated box loses its disk; only what was uploaded (every batch) survives, and resume across boxes needs the checkpoint pushed back (D3 resume exists, the hub push does not); (4) the self-crossing guard is a prototype, 2 human verdicts exist in total: boxes run it fail-closed (production policy) and the importer REFUSES unverifiable surfaces, but **the self-crossing rate of rented output must be read from the smoke-test import summary BEFORE scaling up**; (5) z-band shards are disjoint at the seed only, sheets may cross band edges; (6) vast.ai has no provider-side auto-terminate (watchdog/`--abort` is the only horizon) and its offers are mostly GPU hosts (vCPU = threads; the GPU is paid for and unused); (7) the data pull (30-90 GB per box from the public bucket, provider-region dependent) is the dominant short-rental cost and its real rate is unmeasured.

## Use

```
./deploy.sh PLAN.json --checklist        # ordered manual steps with URLs/commands for your provider
./deploy.sh PLAN.json --dry-run          # the whole run printed; touches nothing
FLEET_EXECUTE=1 ./deploy.sh PLAN.json --smoke --yes   # ONE box, ONE hour (FIRST)
FLEET_EXECUTE=1 ./deploy.sh PLAN.json --yes           # the fleet, unattended; watch fleet_out/<name>.status.md
FLEET_EXECUTE=1 ./deploy.sh PLAN.json --abort         # terminate everything now
```
Plan: copy `plan.example.json` OUTSIDE git (it will hold your real hosts); credentials come ONLY from env vars (names in the plan) or the provider CLI's own config, never the plan (the loader refuses secret-like content and private IPs). Real calls need **both** `--yes` and `FLEET_EXECUTE=1`; without them every provider command is printed, not run. `up` refuses without a spend cap; the effective horizon is min(deadline, cap / fleet $/h) and each box arms its own `shutdown -P +N` (provider-side `--max-run-duration` on GCP, terminate-on-shutdown on AWS) so spend stops even if this machine dies.

What `deploy.sh PLAN --yes` does, unattended: (1) preflight OUR side: read-only credential call, quota where the provider exposes it (AWS), write key != read key, importer kit present, cap + deadline; (2) `up` (spot instances + user-data / ssh bootstrap): deps, sha256-verified code and kit, tool md5 pin gate, fetch ONLY that scroll's prediction + grids + CT levels 1-5 from the public bucket on the box, preflight, `run.json`, runner under systemd with `MemoryMax=` (nohup fallback) that loops seed (z-band shard) -> grow -> pack -> upload; (3) every tick: watchdog probe + judgement, status JSON/MD, collect (object-store sync / rsync, local read key), sha256 verify (bad tarballs quarantined to `_bad/`), `hub/import_remote_grow.py --dry-run` then register (append-only); (4) 15 min before the horizon: STOP runners, wait for the last upload, `down`, VERIFY by listing instances by tag (any left: exit 4, loud), final collect/import, per-box and fleet summary (cm2 imported, REFUSED counts + reasons, $ spent as a MODEL not the bill).

Files: `plan.py` (validation, shards, cost) - `adapters.py` (SSH, AWS spot, GCP SPOT, vast.ai; thin argv builders) - `fleet.py` (plan/up/status/collect/down) - `deploy.py`+`deploy.sh` (one button) - `watchdog.py` (non-LLM monitor) - `box_probe.py`, `box_loop.sh`, `bootstrap.sh.tmpl` (on box) - `RUNBOOK.md` - `plan.example.json`. Tests: `tests/test_provision.py` (+ mutants in `tests/mutants.py`).

## Result path without exposing the hub
Boxes upload tarball + `.sha256` per segment to a store; this machine pulls and the importer runs HERE/on the hub (hub-initiated). A box never holds the hub token, a hub ssh key or a read key.
| option | boxes hold | cost | note |
|---|---|---|---|
| S3 / R2 / B2 bucket (recommended) | a WRITE-ONLY key per rental (S3: PutObject on `prefix/*` only; delete the key at the end) | storage ~0; inbound free; **egress to you**: S3 ~$0.09/GB, R2 $0 egress, B2 free up to 3x storage; tarballs are ~0.5-5 MB per segment so egress is cents | the env file on the box is mode 0600 but visible to the provider: a write-only key limits the damage to junk uploads |
| staging VM (rsync) | a forced-command ssh key to the VM only | VM + disk hours | you operate one more machine |
| direct scp/rsync to the hub | NEVER | | not offered |
Not measured: our uplink, bucket download rates per provider.

## A monitor: PushButtonLocalCoders or not
Read 2026-10-06 (README only; 0 stars, 147 commits, 5 open PRs, no test suite mentioned): it provisions **local LLM backends on NVIDIA GPUs** (llama.cpp GGUF, model/GPU placement) and launches coding frontends (Claude Code, Qwen Code, OpenCode...) against them. It is not a fleet monitor and does not drive provider CLIs or ssh by itself; a frontend it launches could, because Claude Code can run shell commands, but then **the model is the monitor** with all the unreliability that implies, and it needs a GPU box running 24 h. **Not suitable as the thing the fleet depends on.** What the fleet depends on is `watchdog.py` (plain stdlib, deterministic, the only automatic actions are restart-runner and down, plus box-side dead-man switches). A local Claude Code (hosted or local) is an optional reader of the status Markdown using RUNBOOK.md's prompt; a human can do the same with a coffee.
