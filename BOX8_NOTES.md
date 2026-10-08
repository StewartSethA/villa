# Box8 notes: running Route B fits on a rented multi-GPU box (8x V100, 12 TB SSD, pro-rata hourly)

## One-liner (README text for the rental box)
```
git clone --branch routeAB-deploy --single-branch <repo-url> routeAB && cd routeAB && ROUTEB_HOME=/data/routeB ./routeB_shakedown.sh      # <= $5, run this first
ROUTEB_HOME=/data/routeB BUDGET_BOX_START=$(date +%s) nohup ./routeB_run.sh --mode box8 --scrolls PHerc0211,PHerc0826,... --order spt > box8.log 2>&1 &
# on OUR machine, any time, resumable (the box never connects to us):
./routeB_pull.sh --host root@<box-ip> -i <key> --remote-home /data/routeB --remote-tree ~/routeAB --dest <local dir> --final
```
Box prerequisites (installed by hand once; `routeB_run.sh` fails loud naming whichever is missing): Linux x86_64, NVIDIA driver >= 525, `g++` >= 13 (C++23 build of vc_spiral), `rsync`, `curl` or `wget`, `git`, `python3`. Everything else (uv, Python 3.14.7, 120 pinned wheels incl. torch cu126) is fetched into `$ROUTEB_HOME/env`.

## GPU discovery
`box8.discover_gpus()` runs `nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader` and prints one line per card (`GPUs: 0=Tesla V100-SXM2-32GB 32768 MiB; ...`); one worker thread per index, each job gets `CUDA_VISIBLE_DEVICES=<index>`. Override with `--gpus 0,1,2` (e.g. leave a card for debugging). `--fake-gpus N` is test-only. If `nvidia-smi` fails the run exits with `ROUTEB FAIL box8-gpus`. A GPU that is ECC-faulty/off the bus makes `nvidia-smi` slow or fail box-wide: exclude it with `--gpus`.

## Disk layout for the 12 TB SSD (one filesystem; `ROUTEB_HOME` on it)
| path | contents | size (MEASURED on our machines; otherwise ASSUMED) |
|---|---|---|
| `env/`, `env.tools/` | uv + Python 3.14.7 + wheels | ~15 GB (not re-measured on a clean box) |
| `compile_cache/` | torch inductor + triton caches shared by all fits (saves the 6-9 min first-step compile after the first fit per GPU model) | < 2 GB |
| `assets/<scroll>/dataset/{tracks,lasagna_inputs}` | tracks .dbm 3.9-13.2 GB (MEASURED upstream sizes) + lasagna nx/ny/grad_mag (z-slab 0.25 GB for a smoke; whole z range up to ~7-20 GB) | ~20-30 GB per scroll -> ~0.6 TB for 21 scrolls |
| `runs/<scroll>/<tag>/fit/` | fit outputs: meshes ~0.56 GB for a full fit (MEASURED, f0211_full), checkpoints (<= ~15 MB each), `cache/` (UNMEASURED) | ~1-2 GB per fit |
| `runs/<scroll>/<tag>/tiled/` | tile tifxyz | 0.22 GB for a full fit (MEASURED) |
| `out/<scroll>/<tag>/` | ONE directory for everything downloadable: hardlinks of the payload files (no extra space) + `PAYLOAD.json` md5 manifest + `DONE`, written when each stripe finishes; `out/<scroll>/{SCROLL.json,DONE}`, `out/routeA/...`, `STATUS.json`, `ALLDONE.json` (`box8/payload` is a symlink to it) | = the payload (~0.8 GB per full fit) |
| `box8/{state,logs,budget,events.jsonl}` | resumable state + ledgers | MB |
Total expected < 1 TB; 12 TB leaves a large margin, so the only disk guard is `--min-free-gb` (default 200): no new fit is admitted below it. Ink/flatten/render are NOT part of box8 mode (render layers are 3.1 GB per tile, plan doc risk 1).

## Per-GPU RAM guard (host RAM; GPU memory is the ladder's business)
- Admission: a fit is started only when `MemAvailable >= --ram-need-gb` (default 40 GB, an ASSUMPTION; env `ROUTEB_FIT_RAM_GB`). The first fits log `peak_rss_gb` per attempt in `box8/state/<scroll>.json` and `box8/events.jsonl`: set the flag from that.
- Kill rule: while a fit runs, `MemAvailable < --ram-floor-gb` (default 12 GB) kills the fit with the largest RSS (SIGTERM, then SIGKILL) and classifies it `host_oom` (retried once or twice with fewer tracks/step, then the next ladder rung) instead of letting the kernel OOM-killer pick a victim at random.
- `FIT_SPIRAL_NUM_THREADS`/`OMP_NUM_THREADS` default to min(8, ncpu) per fit (fit.py): 8 fits x 8 threads.
- GPU OOM: classified from the log (`CUDA out of memory`), retried with `ladder_config.json: oom_steps`, then the next narrower rung. torch.multinomial's 2^24-category limit is deterministic and never retried by memory.

## What box8 does NOT do
No flatten/render/ink (that is the shakedown's stage 3 and the fleet's job), no D6 validation, recto only, no vacuum filter; sense/shell are the registry's (scrolls whose sense is unknown are SKIPPED with the reason). Expected-hour figures are EXTRAPOLATED (one converged full fit, n = 1); the 2800-slice factor is unmeasured.

## Governor settings (all configurable; printed at start as "BUDGET SETTINGS: ...")
Precedence: built-in defaults < `--budget-config FILE.json` < `BUDGET_*` env < CLI flags.
| setting | default | CLI flag | env | file key |
|---|---|---|---|---|
| machine $/h | 4.276 (8xA100 40 GB) | `--hour-usd` | BUDGET_HOUR_USD | hour_usd |
| allocated disk, $ per 16 GB per hour (billed the WHOLE run) | 0.009 | `--disk-usd-per-16gb-hour` | BUDGET_DISK_USD_PER_16GB_HOUR | disk_usd_per_16gb_hour |
| allocated disk GB | 934 (= $0.525/h; effective rate $4.801/h, soft $45 = 9.4 h, $50 = 10.4 h) | `--disk-gb` | BUDGET_DISK_GB | disk_gb |
| soft stop (no new fits when the projection reaches it) | $45 | `--soft` | BUDGET_SOFT_USD | soft_usd |
| hard stop | $49 | `--hard` | BUDGET_HARD_USD | hard_usd |
| MAX RUN TIME (h since rental start; no launch whose projected end passes it, running fits finish, then the run winds down; 0 = unlimited) | 12 | `--max-run-hours` | BUDGET_MAX_RUN_HOURS | max_run_hours |
| ingress, box downloads from the web, $/TB | 2.70 | `--ingress-per-tb` | BUDGET_INGRESS_PER_TB | ingress_per_tb |
| egress, box uploads to us, $/TB | 4.00 | `--egress-per-tb` | BUDGET_EGRESS_PER_TB | egress_per_tb |
| swap the two directions | off | `--swap-directions` | BUDGET_SWAP_DIRECTIONS=1 | swap_directions |
RAM/disk guards: `--ram-need-gb` (40), `--ram-floor-gb` (12), `--min-free-gb` (60). Max run time is a launch gate, not a kill: a fit already running is never killed for it.

## Computed stripe height (adaptive ladder) -- WIRED into the scheduler
`routeB/ladder.py`: `computed_height(vram_gib, margin_gib=1.5)`, `start_height_for()` (min of span, VRAM model, 2^24-track limit, `--max-height`; a recorded success wins), `decide()` (OOM after its retry -> 0.75x the failed height, grid 100, floor 1000; multinomial -> the track-limit height), `record_height()` -> `box8/heights.json`. Model: peak VRAM = 3.97 + 0.00208 GiB x z_slices, a 2-point line (2,800-slice 0191 stripes on a 4060 Ti: 9.6-10.2 GiB, n = 4; 13,000-slice 0211 on a V100-32: ~31 GiB). On a 40 GB A100 the model allows 16,600 slices, so the whole 13,000-slice scroll goes to one GPU. See `SCHEDULING.md` for the policy, cost model and the parallel-vs-serial answer.

## Runtime control directory (`$ROUTEB_HOME/box8/control/`, re-read every `--control-poll-s` = 10 s)
| file | meaning | written by |
|---|---|---|
| `gpus` | comma list of allowed GPU indices (absent = `--gpus` / all usable GPUs). Indices that are not usable GPUs of the box are ignored with a warning | `routeB_ctl.sh gpus 0,1,2` (also deletes `drain.N` for the listed N) |
| `drain.N` | finish GPU N's current fit, then do not use it | `drain N`; also created by a kill |
| `kill.N` | SIGTERM GPU N's fit process group now; the job goes back to `pending` (class `ctl_kill`, no ladder attempt charged) and resumes from its last autosave (`FIT_SPIRAL_AUTOSAVE_INTERVAL`, 1000 steps; anything newer is recomputed). Consumed as `kill.N.done`; GPU N stays drained | `kill N` |
| `STOP` | graceful: no new launches; when the running fits finish the workers end and ALLDONE is written (pending work stays in `box8/state`; re-running resumes, a leftover `STOP` is removed at start) | `stop` |
| `PAUSE` / `RESUME` | no launches while `PAUSE` exists; `RESUME` (consumed) removes `PAUSE` and `STOP` | `pause` / `resume` |
| `PLAN.txt` | the latest REPLAN printout (written by the scheduler) | scheduler |
`box8/STOP` (outside `control/`) is unchanged: HARD stop, kills running fits.

**Re-plan (`Scheduler.replan` -> `planner.replan`)** runs on every change of allowed GPUs / pause / stop / kill and at no other time. Inputs: remaining hours of the running fits (from their iteration progress), spent $ and hours, pending payload, the plan priority. Committed scrolls (something already ran or was retried) always stay. Uncommitted scrolls are admitted greedily in priority order while the **p90** case keeps `spent + makespan x effective $/h + payload egress <= plan_frac x soft` and `now + makespan <= plan_frac x max_run_hours`; the first that does not fit and all after it are `deferred` (status `deferred`, `replan_deferred`) with the numbers in `fail_why`; a later re-plan with more GPUs restores them to `pending`. The LPT tail split is re-armed for the new count; Route A is restarted (resumable) with the new `--workers` when the slot count moved by >= max(2, 20 %). Not modelled in a re-plan: lasagna fetch time for scrolls not yet staged (inputs already on disk are assumed). Budget behaviour is unchanged: the governor still only stops launching, kills nothing before the hard cap.

**Foreign processes.** At start every GPU with `memory.used > --foreign-mib` (1500 MiB) is skipped with a warning (`--force-gpus` overrides); if all requested GPUs are held the run fails loudly. A foreign process that appears later blocks launches on that GPU for 60 s at a time (checked just before each launch) and is announced.

**Disk measurement.** `ROUTEB_HOME` is created first and `statvfs` is taken on that path itself (never its parent: an overlay root next to a separate `/workspace` volume read 13 GB free on a box with 918 GB free). The planner prints the `df` line it used (`PLAN disk: ...`), clamps the base use to the volume, and counts the env only when `--disk-free-gb` is overridden (a real df already includes it). `--disk-total-gb` / `--disk-free-gb` still override and are announced.

## Allowed-set wall and the measured link (v4)
* The `PLAN:` line, the budget projection and every `REPLAN` use the ALLOWED GPU set: `... GPU-h expected on 4 allowed GPU(s) of 8 = X h wall` (X = max(GPU-h / allowed, longest job) + the hours until the first scroll is staged). A REPLAN prints `allowed 3 of 8 GPU(s) ... >= X h wall on the allowed set`.
* **Link.** At start (also in `--dry-run`) 8 parallel 8 MB HTTP range reads (64 MB, ~$0.0002) of a real input measure the aggregate ingress MB/s; `--link-mb-s X` overrides, `--no-link-probe` keeps the quoted 860 Mbps; a failed probe is announced and the quote stays. The plan prints `PLAN link X MB/s (MEASURED at start|QUOTED/assumed) -> fetching N GB takes Y h of pure transfer (objects bound: ...)`.
* **Fetch time is in the cost.** Per scroll: staging h = max(objects / (files/s), GB / (link / parallel fetches)); the simulation holds each scroll until staged, so `idle GPUs waiting for data` (GPU-h, and their share of the bill in $) is printed and priced. REPLAN gives unstaged scrolls a release time (planner.release_makespan), so a re-plan does not promise work that cannot start yet.
* **Warning.** Per scroll `fetch_h > 0.5 x fit GPU-h`, or total staging > 0.5 x compute wall, prints `!!! DOWNLOAD TIME DOMINATES` and `!!! RECOMMENDATION: shrink --gpus to ~K ..., or rent a box with ingress >= Z MB/s, or restrict --z0/--z1`.
* **Observed fetch.** Each finished fetch longer than 60 s is compared with its plan (`FETCH OBSERVED ... xR`); outside 0.5-1.5x the planner's file rate and link rate are rescaled by 1/R (ASSUMPTION: the miss is uniform), the slowdown is announced, and a re-plan runs.
