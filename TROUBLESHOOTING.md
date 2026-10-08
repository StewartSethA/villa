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
