# routeAB-deploy: Vesuvius Route A + Route B pushbuttons

Two one-command pipelines, identical on a local GPU/CPU box and on a rented machine. No credentials, no private hosts, no history.

| | what | command | needs |
|---|---|---|---|
| **Route A** | guarded surface **growth** from seeds with the production guard stack (VC3D tracer kit, self-collision guard, degeneracy resume gate) | `git clone -b routeAB-deploy --depth 1 https://github.com/StewartSethA/villa.git && cd villa && ./routeA_run.sh --scrolls PHerc0332,PHerc0211 --seeds 8 --hours 6` | CPU only; bash, tar, curl/wget/python3 |
| **Route B** | **spiral fit** of whole windings -> tiles -> flatten -> render -> **ink** -> export | `git clone -b routeAB-deploy --depth 1 https://github.com/StewartSethA/villa.git && cd villa && ./routeB_run.sh --scrolls PHerc0211,PHerc0125 --stripe-width 4500` | NVIDIA GPU (driver >= 525), g++ >= 13, curl/wget |

## Cloud quickstart (rented multi-GPU box): the two one-liners
Target box: 8 x A100 40 GB, 192 threads, 516 GB RAM, ~934 GB NVMe, Ubuntu with outbound HTTPS. Budget $50 (soft stop $45 / hard stop $49; billed machine $4.276/h + disk $0.525/h + $2.70/TB in + $4/TB out).

**0. Box setup (once, as root; ssh in with your key):** needs `git rsync curl g++ >= 13` and a data disk with >= 300 GB free.
```
ssh -p <PORT> root@<BOX-IP>                      # add your public key to the box's authorized_keys first (the provider console or ssh-copy-id)
df -h                                            # pick the NVMe mount (e.g. /workspace); everything lives under ROUTEB_HOME on it
apt-get update && apt-get install -y git rsync curl g++-13 && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100   # skip if `g++ --version` is already >= 13
nvidia-smi -L                                    # must list the 8 GPUs (driver >= 525)
```
**1. On the box (starts everything, returns immediately; log in `box8.log`):**
```
git clone --branch routeAB-deploy-v3 --depth 1 <REPO-URL> routeAB && cd routeAB && \
  BUDGET_BOX_START=$(date +%s) ROUTEB_HOME=/workspace/routeB nohup ./routeB_run.sh --mode box8 > /workspace/box8.log 2>&1 &
```
With no `--scrolls` this plans **every runnable eligible scroll**, defers (explicitly, in the printed plan) what does not fit 80 % of the budget at the p90 case, runs Route B on the GPUs (one scroll per GPU, smallest first, tail split into z-stripes) and Route A on the spare cores, and writes everything downloadable to **`$ROUTEB_HOME/out/`**. Preview without launching: add `--dry-run` (prints plan, stagger timeline, expected payload arrivals, disk timeline, $ per scroll). Tune: `--gpu-speed`, `--hour-usd 4.276 --disk-gb 934 --disk-usd-per-16gb-hour 0.009`, `--soft 45 --hard 49 --max-run-hours 12`, `--no-routea`, `--scrolls A,B`.

**2. On YOUR machine (pull everything as it arrives; resumable; re-run any time):**
```
./routeB_pull.sh --host root@<BOX-IP> --port <PORT> -i <KEY> --remote-home /workspace/routeB --dest ./out --final
# the same thing without any script (checksum verification afterwards):
while :; do rsync -aH --partial --append-verify --exclude '.tmp_*' -e "ssh -p <PORT>" root@<BOX-IP>:/workspace/routeB/out/ ./out/; sleep 60; done
python3 pull_box8.py --verify-only --dest ./out
```
**Scaling back / forward on the fly (no restart):** the scheduler re-reads `$ROUTEB_HOME/box8/control/` every ~10 s.
```
./routeB_ctl.sh --home /workspace/routeB status          # allowed GPUs, jobs, budget, the latest REPLAN
./routeB_ctl.sh --home /workspace/routeB gpus 0,1,2      # use ONLY these GPUs from now on (running fits on the others finish); `gpus 0,1,2,3,4,5,6,7` grows back
./routeB_ctl.sh --home /workspace/routeB drain 5         # finish GPU 5's current fit, then stop using it
./routeB_ctl.sh --home /workspace/routeB kill 5          # SIGTERM GPU 5's fit; its z-interval is re-queued and resumes from the last autosave (never lost); GPU 5 stays out
./routeB_ctl.sh --home /workspace/routeB stop            # graceful: no new launches, running fits finish, the run ends (re-run the same command to resume)
./routeB_ctl.sh --home /workspace/routeB pause | resume
```
Every change re-plans the unstarted work (tail split for the new GPU count, Route A slots = cores - 2 x allowed GPUs - reserve, RAM, disk) and prints `REPLAN ...` lines (also in `box8/control/PLAN.txt`): scrolls that no longer fit 80 % of the remaining budget/time at the p90 case are DEFERRED explicitly, and re-admitted when GPUs come back. Running fits are never cut off by a re-plan. Start-up: `--gpus 1,2,..` picks the initial set; a GPU already holding VRAM from a foreign process (e.g. a llama-server; `--foreign-mib` 1500) is warned about and skipped unless `--force-gpus`. The old hard stop (kills running fits) is still `touch $ROUTEB_HOME/box8/STOP`.

Layout of `out/`: `<scroll>/<tag>/{files, PAYLOAD.json (size+md5 of every file), DONE}` per finished stripe (written the moment it finishes), `<scroll>/{SCROLL.json,DONE}`, `routeA/<scroll>__<seg>/...`, `STATUS.json`, `ALLDONE.json`. Mark-and-free: the puller writes `PULLED.json` per unit on the box; the box then deletes that scroll's fetched inputs. Stop the box: `touch $ROUTEB_HOME/box8/STOP` (running fits get SIGTERM; finished units stay pullable). Details: `SCHEDULING.md`, `BOX8_NOTES.md`, `AGENT_GUIDE.md`, `TROUBLESHOOTING.md`.

Route B options: `--stripe-width 4500|7500|13500|full`, `--smoke` (proof-sized run), `--stages fetch,fit,tiles,ink,export`, `--help`.
Both scripts download everything they need (pinned and hash/size-verified, resumable) into `./routeA_work/` or `./routeB_work/` (`ROUTEB_HOME`, `--workdir`).

* Route A details: `README_ROUTEA.md`. Route B details: `README_ROUTEB.md`.
* A chatbot/agent picking this up: **`AGENT_GUIDE.md`** (architecture, state, how to verify each stage, failure catalogue pointers).
* Something broke: **`TROUBLESHOOTING.md`**.
* Prove it works on a new machine: `./smoke_test.sh A|B|both` (small cases; prints wall time and GB downloaded).
* Shared plumbing (one implementation): `deploy_common/` (`bootstrap_env.sh` pinned uv + python + lock; `fetch_assets.py` verified resumable downloads; `branch_scan.py` secret/IP/size scan; run `python3 deploy_common/branch_scan.py .` before pushing any change).

Layout: `routeA_run.sh pins/ src/ umbilicus/ tests/ patches/ tools/` (Route A) | `routeB_run.sh routeB/ spiral-fitting/ lasagna/ vesuvius/ gpu_render/ ink/ models/` (Route B) | `deploy_common/` (both).
Provenance: library code is copied from Seth Stewart's ScrollPrizeTutorial working repo at the commits recorded in `routeB/BUILD_INFO.txt`; upstream pieces are Vesuvius Challenge `villa` (spiral-fitting, lasagna).
