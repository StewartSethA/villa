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
| `box8/payload/<scroll>/` | hardlinks of the payload files (no extra space) + `PAYLOAD.json` md5 manifest + `DONE` | = the payload (~0.8 GB per full fit) |
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
| machine $/h | 2.33 (8xV100) | `--hour-usd` | BUDGET_HOUR_USD | hour_usd |
| soft stop (no new fits when the projection reaches it) | $45 | `--soft` | BUDGET_SOFT_USD | soft_usd |
| hard stop | $49 | `--hard` | BUDGET_HARD_USD | hard_usd |
| MAX RUN TIME (h since rental start; no launch whose projected end passes it, running fits finish, then the run winds down; 0 = unlimited) | 12 | `--max-run-hours` | BUDGET_MAX_RUN_HOURS | max_run_hours |
| ingress, box downloads from the web, $/TB | 2.70 | `--ingress-per-tb` | BUDGET_INGRESS_PER_TB | ingress_per_tb |
| egress, box uploads to us, $/TB | 4.00 | `--egress-per-tb` | BUDGET_EGRESS_PER_TB | egress_per_tb |
| swap the two directions | off | `--swap-directions` | BUDGET_SWAP_DIRECTIONS=1 | swap_directions |
RAM/disk guards: `--ram-need-gb` (40), `--ram-floor-gb` (12), `--min-free-gb` (200). Max run time is a launch gate, not a kill: a fit already running is never killed for it.
