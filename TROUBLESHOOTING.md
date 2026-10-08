# TROUBLESHOOTING

First: read the last 30 lines of the failing stage's log (`routeA_work/logs/`, `routeB_work/bootstrap.log`, `routeB_work/runs/<scroll>/<stripe>/{fit/fit.log,tilework/<tile>/tile.log}`), then look the exact message up below. Shared (both routes):

| symptom | cause | fix |
|---|---|---|
| `BOOTSTRAP FAIL: venv ...` | uv could not create/download Python | disk full? outbound HTTPS to github.com blocked? |
| `BOOTSTRAP FAIL: pip-install` | PyPI / pytorch.org unreachable or pin missing | read the bootstrap log; the lock is exact (`==`) on purpose |
| download stalls / `incomplete (n/m blocks)` | flaky link | just re-run: `.part` + block map resume; verified files are skipped |
| `md5 ... != pinned` (file deleted) | corrupt transfer or upstream changed | re-run once; if it repeats, upstream changed: update `routeB/scrolls/pins.json` after checking |
| `BRANCH SCAN FAILED` | private path/IP/secret/big file crept in | fix the hit; never allow-list a real secret |
| disk full mid-run | tracks 4-13 GB + CT chunks + fit caches | set `ROUTEB_HOME` / `--workdir` to a bigger disk; delete `runs/*/*/tilework/*/layers` (re-renderable) |

## Route B failure catalogue
| symptom (exact text) | cause | fix |
|---|---|---|
| `ROUTEB FAIL preflight: no nvidia-smi` | CPU-only box | Route B needs a CUDA GPU; use `--stages fetch` to pre-stage data only |
| `ROUTEB FAIL preflight: g++ N is too old for C++23` | `vc_spiral` extension needs g++ >= 13 | install g++-13+ (or a newer image, e.g. Ubuntu 24.04+) |
| `ROUTEB FAIL libstdc++: ... GLIBCXX_3.4.29` / `ImportError ... GLIBCXX` from `vc_spiral.*` | torch loads the system libstdc++; the native ext needs 3.4.29+. On v100 this silently degraded the fit and then died at "Packed satisfaction requires vc_spiral.spiral_sampling.PatchSatisfactionAtlas" | newer OS, or put a new libstdc++ on `LD_LIBRARY_PATH` |
| `BOOTSTRAP FAIL: uv-download` / `uv-sha256` | no curl/wget, GitHub blocked, or a changed asset | install curl; check `routeB/pins/uv.env` |
| `BOOTSTRAP FAIL: pip-install` | PyPI/pytorch.org unreachable or a pin was yanked | read `$ROUTEB_HOME/bootstrap.log` |
| `FETCH FAIL ... HTTP 404` on tracks | scroll has no published tracks (846B, 1203, 1218, 1447, 1545, Paris4) | extract tracks first |
| `fetch: ... size N != server Content-Length` | upstream file replaced | re-run `routeB/tools/make_specs.py`, check `pins.json` |
| fit log ends in `CUDA out of memory` and the chain says `chunk N exited rc=1` then resumes | flow-grid memory grows with the z span (fix: `model_flow_voxel_resolution` 32, automatic above 6,000 slices) and the per-iteration leak (FINDINGS 30.16) | the chain restarts from the last autosave; if `NO PROGRESS` appears use a narrower `--stripe-width` or `--fit-overrides` |
| `RuntimeError: number of categories cannot exceed 2^24` | torch.multinomial ceiling on > 16.7 M tracks (PHerc0191 full width) | fixed in `spiral-fitting/tracks.py` (villa b408d54c), present in this branch |
| `fit: spiral_outward_sense for X is unknown` | registry had none | pass `--sense CW|ACW` after an A/B |
| `render: unusable render ... (no CT chunks present for this region?)` | the chunk fetch got 404s (masked/absent chunks) or the tile lies outside the scan | check `assets/.fetched/<S>:volume:chunks:*.json` `absent` count |
| `cuModuleLoadData failed with 222` (render) | NVRTC newer than the driver supports | lock pins `nvidia-cuda-nvrtc-cu12==12.6.*`; do not upgrade |
| `INK FAIL <family>: weights missing` | `models/var/models/<ckpt>` absent | `md5sum -c models/MODELS.md5` |
| `flatten: collapsed flat` | lasagna flatten did not converge on a degenerate tile | tile is skipped, others continue; inspect `tilework/<tile>/tile.log` |
| `PLAN: NOTHING FITS` / `DEFERRED <scroll>: p90 plan ...` | the 80 %-of-budget p90 test; the A100 speed is UNMEASURED (assumed = V100) | time one stripe on the box, re-plan with `--gpu-speed X --dry-run`; or `--plan-frac 1.0` / `--no-plan` knowingly |
| `STAGING <s> BLOCKED by disk` forever, GPUs idle | high-water reached, nothing pulled | run the pull loop; `PULLED.json` releases inputs; or `--free-inputs-on done`, `--disk-high-water 0.9` |
| GPUs idle for the first hour | the whole-range lasagna fetch (241 k objects per scroll, 58 files/s measured on a loaded host) | `grep fetch box8.log`; raise `ROUTEB_FETCH_WORKERS` (128 default; 256 was slower on pny), or restrict with `--z0/--z1` |
| `TAIL SPLIT ...` lines | working as designed: fewer jobs than GPUs, scroll split into z-stripes | none |
| `UNIT PAYLOAD FAILED` | disk full / hardlink across filesystems fails over to copy | free disk; the unit is retried at scroll finalize |
| `Route A NOT started: 0 slots (...)` | cores/RAM formula leaves nothing | `--routea-slots N`, `--routea-ram-per-grow-gb`, `--no-routea` |
| `Route A EXITED rc=...` | kit download (GitHub release) or its env failed | `box8/logs/routeA.log`; Route B is unaffected; `KIT_URL=... ./routeA_run.sh` |
| `budget: REFUSE ... projected $X` | soft cap, or the projected end passes `--max-run-hours` | expected; defers the job (exit 4); raise the cap only with the owner's approval |
| pull: `VERIFICATION FAILED` | partial/corrupt copy | the pull re-syncs with `--checksum` once; run again; unit is never marked pulled |
| `ssh` pull `Permission denied` | the box does not have your key | add your public key to the box; the box never needs a key for our side |
| `CONTROL: gpus file lists GPU N which is not a usable GPU` | typo, or N was skipped at start as foreign-held | `nvidia-smi`; `--force-gpus` at start |
| `REPLAN DEFERRED <scroll>` after shrinking GPUs | the p90 re-plan no longer fits 80 % of the remaining budget/time | expected; `routeB_ctl.sh gpus <more>` re-admits it, or raise `--soft`/`--plan-frac` with the owner's approval |
| `WARNING GPU N holds ... MiB (a foreign process)` | another process (e.g. llama-server) owns VRAM | free it, or `--force-gpus`; the GPU is skipped meanwhile |
| `IDLE: no allowed GPU and nothing running` | `gpus none`, or every GPU drained/killed | `routeB_ctl.sh gpus 0,1,...` or `stop`: the box is still billing |
| planner shows tiny free disk | old builds measured the parent of a not-yet-created home | v4 measures the home itself; see the `PLAN disk: df ...` line |
