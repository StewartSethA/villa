# routeAB-deploy: Vesuvius Route A + Route B pushbuttons

Two one-command pipelines, identical on a local GPU/CPU box and on a rented machine. No credentials, no private hosts, no history.

| | what | command | needs |
|---|---|---|---|
| **Route A** | guarded surface **growth** from seeds with the production guard stack (VC3D tracer kit, self-collision guard, degeneracy resume gate) | `git clone -b routeAB-deploy --depth 1 https://github.com/StewartSethA/villa.git && cd villa && ./routeA_run.sh --scrolls PHerc0332,PHerc0211 --seeds 8 --hours 6` | CPU only; bash, tar, curl/wget/python3 |
| **Route B** | **spiral fit** of whole windings -> tiles -> flatten -> render -> **ink** -> export | `git clone -b routeAB-deploy --depth 1 https://github.com/StewartSethA/villa.git && cd villa && ./routeB_run.sh --scrolls PHerc0211,PHerc0125 --stripe-width 4500` | NVIDIA GPU (driver >= 525), g++ >= 13, curl/wget |

## Cloud quickstart (rented multi-GPU box): ONE line, then a live dashboard
Target: 8 x A100 40 GB, 192 threads, 516 GB RAM, ~934 GB NVMe, Ubuntu, outbound HTTPS, root or sudo.
Budget $50 (soft stop $45 / hard stop $49; billed machine $4.276/h + disk $0.525/h + $2.70/TB in).

**On the box (paste this one line):**
```
curl -fsSL https://raw.githubusercontent.com/StewartSethA/villa/routeAB-deploy-v5/go|bash -s run
```
What it does, in this order:
0. **Link check first (about 13 s, before apt/uv/pip/compile):** 8 parallel range reads from the real data
   host (dl.ash2txt.org) and from a CDN (tells "box link slow" from "source slow"); prints a verdict box
   (MB/s, hours to fetch the plan, box dollars idling, GOOD / MARGINAL / BAD). Below `--min-link-mb-s`
   (default 20) it STOPS with exit 5 and "DESTROY THIS BOX or re-run with --accept-slow-link", nothing built.
   With `--accept-slow-link` the planner shrinks `--gpus` / `--fetch-parallel` to what the link can feed.
   The link is re-probed every 10 min; degrade/recover is announced and GPUs auto-shrink via the control dir.
1. prints the GPUs and chooses the torch wheel from the hardware (cu128 for Blackwell / driver CUDA >= 12.8, else cu126);
   installs git rsync curl tmux g++>=13; 2. clones the branch; 3. starts the run detached in tmux session
   `routeb` (ROUTEB_HOME and BUDGET_BOX_START are passed inside the tmux command; the resolved home is printed);
4. attaches the live dashboard. Ctrl-C detaches only the watcher.
   Inside the run: link check -> early prefetch of the first scroll's stripes while the env builds -> GPU kernel
   smoke + speed test on every GPU (a failure stops the run before anything is fetched) -> per-stripe staging:
   the first stripe's fit starts as soon as its own data has landed. Optional: `--fill-idle-gpus` (see BOX8_NOTES).

Extra run flags go after `run`, e.g. `bash -s run --accept-slow-link --gpus 0,1,2,3` (`--dry-run` = plan only).

**Dashboard (reattach any time, from any ssh session):**
```
bash ~/routeAB/routeB_watch.sh
bash ~/routeAB/routeB_watch.sh --plain
bash ~/routeAB/routeB_watch.sh --once
```
It redraws every ~5 s: budget and the clock from the REAL box start, link MB/s + trend, disk, RAM, one row
per GPU (util, VRAM, scroll/stripe, step, it/s, ETA, rung, OOM-ladder descents), fetch rows per scroll,
Route A slots/segments/cm2, payload units (DONE, GB, pulled), an ALERTS area (failures grouped by reason,
stalls, STAGING BLOCKED, a GPU idle >3 min beside pending work, a fetch with no network progress for 3 min),
the exact PULL commands for your machine, and a time-ordered event stream. `--once` prints one block with
STATUS, budget, link, GPUs, alerts and the log tail: paste it back for diagnosis. Raw console: `tmux attach -t routeb`.

**On YOUR machine (the dashboard prints these with the box address filled in):**
```
H=root@<BOX-IP>
P=<PORT>
RH=/workspace/routeB
R=$RH/out/
mkdir -p out
while :;do rsync -aH --partial --append-verify -e "ssh -p $P" $H:$R out/;sleep 60;done
./routeB_pull.sh --host $H --port $P --remote-home $RH --dest out --final
```
(the last line, with md5 verification and PULLED marking, needs a repo checkout on your machine.)

Scale back on the fly: `./routeB_ctl.sh status|gpus 0,1,2|drain N|kill N|stop|pause|resume` (see BOX8_NOTES.md).
Layout of `out/`: `<scroll>/<tag>/{files, PAYLOAD.json (size+md5 of every file), DONE}` per finished stripe,
`<scroll>/{SCROLL.json,DONE}`, `routeA/<scroll>__<seg>/...`, `STATUS.json`, `ALERTS.json`, `ALLDONE.json`.
Hard stop: `touch $ROUTEB_HOME/box8/STOP`. Details: `SCHEDULING.md`, `BOX8_NOTES.md`, `AGENT_GUIDE.md`, `TROUBLESHOOTING.md`.

Route B options: `--stripe-width 4500|7500|13500|full`, `--smoke` (proof-sized run), `--stages fetch,fit,tiles,ink,export`, `--help`.
Both scripts download everything they need (pinned and hash/size-verified, resumable) into `./routeA_work/` or `./routeB_work/` (`ROUTEB_HOME`, `--workdir`).

* Route A details: `README_ROUTEA.md`. Route B details: `README_ROUTEB.md`.
* A chatbot/agent picking this up: **`AGENT_GUIDE.md`** (architecture, state, how to verify each stage, failure catalogue pointers).
* Something broke: **`TROUBLESHOOTING.md`**.
* Prove it works on a new machine: `./smoke_test.sh A|B|both` (small cases; prints wall time and GB downloaded).
* Shared plumbing (one implementation): `deploy_common/` (`bootstrap_env.sh` pinned uv + python + lock; `fetch_assets.py` verified resumable downloads; `branch_scan.py` secret/IP/size scan; run `python3 deploy_common/branch_scan.py .` before pushing any change).

Layout: `routeA_run.sh pins/ src/ umbilicus/ tests/ patches/ tools/` (Route A) | `routeB_run.sh routeB/ spiral-fitting/ lasagna/ vesuvius/ gpu_render/ ink/ models/` (Route B) | `deploy_common/` (both).
Provenance: library code is copied from Seth Stewart's ScrollPrizeTutorial working repo at the commits recorded in `routeB/BUILD_INFO.txt`; upstream pieces are Vesuvius Challenge `villa` (spiral-fitting, lasagna).
