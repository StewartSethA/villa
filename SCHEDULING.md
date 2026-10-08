# Route B on a multi-GPU box: scheduling, cost model, and "is one scroll on several GPUs faster?"

Every number below is tagged MEASURED (with n and where) or ASSUMED. `./routeB_run.sh --mode box8 --dry-run` prints the same model as a plan for the box it runs on.

## Cost model (routeB/ladder.py `fit_hours`, routeB/planner.py)
One `fit_spiral` job (30,000 steps) of z-height `h` on a V100-class card:
`GPU-h(h) = t_full(scroll) x (h/13000)^0.90 / gpu_speed`, with p10/p50/p90 multipliers 0.64 / 1 / 1.52.

| input | value | status |
|---|---|---|
| full height (13,000 slices) PHerc0125 / PHerc0211 | 5.99 / 7.49 GPU-h | MEASURED, V100 32 GB, n = 1 fit each (plan range over n = 3 full fits 5.99-13.27 incl. a gap-only run) |
| other scrolls, full height p50 | 0.048 x shell GPU-h (min 5.99) | ASSUMED from the plan table (7.0 h at shell 146, 9.6 at 200, 19.6 at 408) |
| 2,800-slice stripe (200 overlap), V100 box, contended | 1.76 h p50 (p10 1.30 / p90 2.54) | MEASURED, n = 26 completed fits (PHerc0125, 5 stripes) |
| 2,800-slice stripe, RTX 4060 Ti 16 GB (an RTX 4060 Ti fleet box), uncontended | 3.6-3.8 ks = 1.03 h, 8.6-9.8 it/s, 9.6-10.2 GiB, 0 OOM | MEASURED, n = 4 stripes of PHerc0191 |
| 13,000-slice PHerc0211 on V100 32 GB | peak ~31 GiB, 1.1 it/s | MEASURED, n = 1 |
| 13,000 slices on a 15.58 GiB 4060 Ti | OOM at iteration 61-77 in 4 of 4 runs | MEASURED |
| exponent 0.90 | fitted to 5.99 h (full) vs 9.41/5 = 1.88 h per 2,800 stripe, PHerc0125 | one scroll; scroll-independent ASSUMED |
| A100 40 GB speed vs V100 | 1.0 (no speed-up assumed) | UNMEASURED: pass `--gpu-speed` once one fit is timed on the box |
| peak VRAM | 3.97 + 0.00208 GiB x slices | 2-point line (9.8 GiB @ 2,800; 31 GiB @ 13,000) |
| host RSS per fit | 22-40 GB (planner uses 40) | MEASURED on a 4060 Ti box and a V100 box |
| lasagna fetch | 58 files/s per scroll at 128 workers (32 -> 37/s, 256 -> 49/s) | MEASURED, pny under load ~95, n = 1 run per setting. Objects per scroll (z 4500-17500): ~241 k (3 fields x 6,012 objects per 1,012 slices, PHerc0211 slab) -> ~69 min |
| inputs + work dir per scroll | 12 + 6 x tracks-GB = 35-90 GB | ASSUMED, calibrated to the coordinator's 35 / 90 GB endpoints (tracks 3.9-13.2 GB) |
| billing | machine $4.276/h + disk 934 GB x $0.009/16 GB/h = $0.525/h -> **$4.801/h**; $2.70/TB into the box, $4/TB out | quoted; $50 = 10.4 h, soft $45 = 9.4 h |

## Is one scroll on several GPUs faster? Is full height in parallel as efficient as serial?
PHerc0211 (7.49 GPU-h at full height), model `fit_hours`, p50:

| z-stripes | height | wall (h) | total GPU-h | total / full |
|---|---|---|---|---|
| 1 | 13,000 | 7.49 | 7.49 | 1.00 |
| 2 | 6,600 | 4.07 | 8.14 | 1.09 |
| 3 | 4,500 | 2.88 | 8.59 | 1.15 |
| 5 | 2,800 | 1.88 | 9.28 | 1.24 |
| 8 | 1,800 | 1.26 | 10.11 | 1.35 |

* **Faster: yes in wall-clock** (the parallel unit is the z-stripe; `fit_spiral` is single-GPU, so "one scroll on 5 GPUs" = 5 stripe fits on 5 GPUs). 5 stripes finish in ~1.9 h instead of 7.5 h.
* **As efficient as serial: no, ~1.1-1.35x the GPU-hours** (stripe overlap 200 slices + per-job fixed cost: track load ~5 min and first-step compile 6-9 min, both MEASURED on pny). The uncontended 4060 Ti number (5 x 1.03 = 5.1 GPU-h for a 4060 Ti) is *lower* than the contended V100 full fit, so on equal hardware the premium may be smaller; it was not measured head-to-head. The two .142 full-height fits (`h2h_v100_*`) are the first paired data (see STATE_deploy.md in the main repo).
* **The box bills wall-clock, not GPU-hours** ($4.801/h for all 8 GPUs). So whole-scroll-per-GPU (cheapest per GPU-h) is used while every GPU has a scroll; stripes are used only where a GPU would otherwise idle (the tail), and only if the simulated makespan shortens by >= 3 %.

## Policy
1. **One GPU per scroll, smallest first (SPT):** most scrolls finished per dollar; every GPU busy.
2. **Tail (LPT) z-stripes:** once fewer unstarted jobs remain than GPUs, `planner.tail_split` picks the stripe-height target that minimises the simulated makespan and splits the longer jobs into the fewest stripes under that target. Retried jobs are never split (they would lose their checkpoint).
3. **Stripe height = min(span, VRAM model, 2^24-track limit [LIFTED by default when tracks.py has the chunked multinomial; automatic fallback to the cap: BOX8_NOTES.md], --max-height)**; an OOM re-covers the failed interval at **0.75x** the failed height (grid 100, floor 1,000 slices); the height that succeeded is recorded per scroll (`box8/heights.json`) and wins on a re-run.
4. **Admission up front:** the p90 plan must finish within 80 % (`--plan-frac`) of the soft budget **and** of the max run hours; lowest-priority scrolls are DEFERRED explicitly in the plan (priority: both prizes first, then cheapest; `--priority` overrides). At run time the governor only stops *launching*; it never kills a fit except at the hard cap.
5. **Staging:** scroll inputs are fetched just ahead of the GPUs (`--fetch-ahead` 2 beyond the free GPUs, `--fetch-parallel` 8) only while `base use + held scrolls + this scroll <= high-water` (`--disk-high-water` 0.85 of the volume); inputs are deleted when the scroll's units are pulled+verified (`--free-inputs-on pulled|done`). A blocked stage is announced (`STAGING <s> BLOCKED by disk`).
6. **Route A** (guarded grows) runs on the spare cores by default: slots = physical cores - 2 x busy GPUs - 4 reserve, capped by RAM ((RAM - 30 - busy x 40 GB)/6 GB per grow, 6 GB ASSUMED). `--no-routea`, `--routea-slots N`, `--routea-scrolls`, `--routea-disk-gb`.

## What the default plan says for this box (8x A100 40 GB, V100-equivalent speed, 10 runnable scrolls)
At `--gpu-speed 1.0` only the 3 cheapest prize scrolls pass the 80 %/p90 test; at 1.5 four, at 2.0 five, at 3.0 eight (p90 $36). **The kept set grows quickly with the unmeasured A100 speed: time one stripe on the box in the first minutes and re-plan (`--gpu-speed X --dry-run`).** Idle cost at start is dominated by the first lasagna fetch (~1.15 h at 58 files/s = ~9 idle GPU-h = $5.5); at 3x the fetch rate the same plan is $18.3 instead of $22.0 (p50).

## Known gaps (not hidden)
* Staging is per scroll, so GPUs wait for the first scroll's whole-range lasagna fetch. Fetching only each stripe's z-slab (the manifest already supports a z-window) would let a stripe start after ~11 min instead of ~69 min; not implemented.
* The live scheduler has been exercised with stub jobs (fake GPUs/disks), a 1-GPU pny smoke, and unit tests; multi-GPU box8, a real OOM-ladder step and the hard stop have NOT run on real GPUs.
* Tracks are assumed uniform in z for the 2^24 limit; the detect-early kill (fit.log "loaded N tracks") backs it up.

## Per-stripe staging and time to first fit (v5)
Staging is per stripe: tracks once, then each stripe's lasagna chunks in z order, so a scroll that is split starts its first fit after tracks + 1/k of the lasagna instead of after all of it. The plan prints `time to first fit` and `GPUs busy over time`. With the default 3-scroll plan (8 x 40 GB, 58 files/s per scroll, 108 MB/s link, ASSUMED/quoted rates) the first fit starts at 0.58 h instead of 1.15 h and the 8 GPUs are all busy by 1.48 h; the p50 makespan moves from 4.41 h to 4.67 h because the stripes of one scroll are fetched sequentially (conservative: parallel stripe fetches could be faster if the 192-thread box allows, unmeasured). Idle GPU-h while waiting for data: 8.5 (was 9.2).
