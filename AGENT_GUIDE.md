# AGENT_GUIDE: picking this branch up out of the box

You are a chatbot/agent handed this repository with no other context. Read this file, then `README.md`, then the route you need. Rules of the house:

1. **Measure, do not guess.** A stage "worked" when a counter/artifact says so (files, byte counts, step counters, GPU utilisation), never because a script exited 0.
2. **Fail loud.** Every stage prints `FAIL <stage>: <why>`; do not add silent fallbacks. A degradation that succeeds is a silent failure -- announce it.
3. **Never add private paths, hostnames, IPs, tokens.** Run `python3 deploy_common/branch_scan.py .` (exit 1 on any hit, any file > 50 MB, absolute symlinks) before every commit.
4. **No `set -e` in batch scripts**: one bad item must not cancel the rest; handle errors per item.
5. **State lives in files** (`routeA_work/`, `routeB_work/`) and is resumable; do not introduce a database.
6. `deploy_common/` is shared by both routes: change it once, re-run both smoke tests.
7. Do not push. The owner pushes: `git push git@github.com:StewartSethA/villa.git routeAB-deploy`.

How to prove a change: `./smoke_test.sh both` (Route A: PHerc0332, 4 seeds, 2 rounds; Route B: PHerc0211 z 9000-9500). Compare the printed wall time / GB with the numbers in TROUBLESHOOTING.md.

## Route B (spiral fits) -- architecture for an agent picking this up

**Goal.** Turn a scroll's published surface *tracks* + lasagna normal fields + our umbilicus into per-winding sheet meshes (`fit_spiral.py`), cut them into
tiles, flatten each tile, render it at 9.5 um/px, run the ink models, export PNGs + a manifest. `./routeB_run.sh` is the only entry point.

**Code map (branch root).**
| path | role |
|---|---|
| `routeB_run.sh` | preflight (Linux/x86_64, nvidia-smi, g++>=13, curl|wget, libstdc++ GLIBCXX_3.4.29) -> `deploy_common/bootstrap_env.sh` (pinned uv + py3.14.7 + `routeB/pins/requirements.lock`) -> builds `spiral-fitting` (vc_spiral C++23 ext, stamp file in env) -> `python -m routeB.cli` |
| `routeB/cli.py` | stage driver (fetch, fit, tiles, ink, export), argument parsing, `--smoke`, failure list/exit status |
| `routeB/common.py` | paths (`ROUTEB_HOME`), `stripes()` (window tiling of z 4500-17500), subprocess `run()` with log tee, stage markers `.done.<stage>.json` |
| `routeB/manifest.py` | which upstream files one (scroll, z-window) needs -> a `deploy_common/fetch_assets.py` manifest; lasagna z-slab selection |
| `routeB/fit.py` | dataset dir (symlinks + generated `spiral-scroll.json`), fit config overrides, checkpoint-restart chain (port of chain142.sh) |
| `routeB/tile_windings.py` | cuts `wNNN_spliced_*` meshes into tiles + `manifest.json` (copy of scripts/routeB/tile_windings.py with `--voxel-um`, `--wind-min/max`) |
| `routeB/tile_chain.py` | per tile: sparse CT chunk fetch -> `sheet_snap.py` -> lasagna flatten -> frame -> `gpu_render/render_gpu.py` -> mask -> `ink_run.py` |
| `routeB/ink_run.py`, `ink/vesuvius_pipeline/` | vendored ink-model modules (unchanged copies) + `models/var/models/*.ckpt` |
| `routeB/scrolls/<S>.json` | per-scroll facts baked from our registry + live listings (volume, tracks URL/size, lasagna prefix, sense + its source, shell + its source, umbilicus file). Rebuild with `routeB/tools/make_specs.py` in the main repo |
| `routeB/scrolls/pins.json` | md5 of upstream files we hold a reference for |
| `deploy_common/` | shared with Route A: `bootstrap_env.sh`, `fetch_assets.py` (http parallel+resumable, s3prefix, s3keys), `branch_scan.py` |
| `spiral-fitting/`, `lasagna/`, `vesuvius/src/vc3d_fiber_format/` | upstream villa code at the commit in `routeB/BUILD_INFO.txt` |

**State** is only files under `$ROUTEB_HOME`: `assets/` (+ `.fetched/*.json` ledger), `runs/<S>/<stripe>/{fit,tiled,tilework}`, `export/`. No database. A stage is skipped when its
marker exists; delete the marker (or the directory) to redo it.

**Verify each stage by counter/artifact, not by exit code.**
- fetch: `assets/.fetched/_totals.json` (`bytes_net`, `failed`), `ls assets/<S>/dataset/lasagna_inputs/*/2 | wc -l`; md5 in the ledger vs `scrolls/pins.json`.
- fit: `runs/.../fit/fit.log` has `PROGRESS` lines and step counters; `fit/out/checkpoint_fitted.ckpt` step (`torch.load(..., mmap=True)['completed_iterations']`); `.done.fit.json: satisfied_track_points` (production 0.58-0.67 at 30000 steps; a 1500-step smoke is lower and proves only that the chain runs); `nvidia-smi` utilisation over time.
- tiles: `tiled/manifest.json` `n_windings`, `n_tiles`, `tile_area_cm2`; each tile dir has x/y/z.tif.
- ink per tile: `tilework/<tile>/tile_result.json` (edge_med_vox, render_scale, recto_is_reversed, render_valid_frac, per-family mean/p99), `ink/*.png` non-black.
- export: `export/<S>/<stripe>/manifest.json` lists tiles, `errors`, model md5s; tar present.

**Known limits (be honest in reports).** No vacuum filter (production drops windings with material_frac < 0.75 using the 19-87 GB surface prediction; here all windings are tiled -- outer windings may be air). No D6 validation against human annotation of the produced sheets. 0.5 mm ML-window rule not checked. Only the recto face is read. Weights `reader_v2_dense_human_cv14k` is a moving target (md5 in the manifest). Fit recipe knobs are copied from the v100 production runs; sense/shell for scrolls without a fit are ASSUMED (spec `*_source` fields say so).

**Tests.** `python -m py_compile` of every module and `bash -n routeB_run.sh` are the floor; the real test is `./routeB_run.sh --scrolls PHerc0211 --smoke` (z 9000-9500) on a GPU box -- watch every stage's log.

## Route A (guarded grow) -- fragment from its owner

## Route A (grow) -- agent guide fragment
1. Architecture: `routeA_run.sh` -> user-space uv/python3.12 venv from `pins/requirements.lock` (hash-locked) -> `vesuvius_pipeline.routea_cloud.run`: kit (`pins/kit.json`, sha256 + per-file md5) -> per-scroll inputs from the PUBLIC bucket (`pins/scrolls.json`) -> DB-free seeds (`seedprop`, umbilicus radius bins) -> `cloud_box.grow_seed` per seed (tracer rounds with in-solve `self_collision`, selfx scrub, degeneracy resume gate between rounds; settings = `pins/settings_snapshot.json`) -> self-verify with `cloud_import.verify`.
2. State (all under `--workdir`, default `routeA_work/`): `.tools/` (uv, python), `venv/` (+`.lock` = lock sha), `kit/` (+`.verified` = tarball sha256), `data/<scroll>/<name>.complete` (object count + bytes of the S3 sync), `seeds/<scroll>.json`, `export/<scroll>/<seg>/{export.json,md5.txt,r*/...}`, `report.json` (phase seconds, GB downloaded, per-segment verdicts, totals). Re-running resumes; delete a marker to force a re-check.
3. Verify each stage by counter: kit -> log line `kit: ... sha256-verified` / `phase kit`; inputs -> `phase inputs` + `downloaded_gb`, `.complete` marker, `meta.json` in the prediction dir; seeds -> `seeds/<scroll>.json` has N entries; grow -> `grown <seg>: <status> rounds=k` lines (status: grown | gate_held | scrub_failed | no_checkpoint | deadline); verify -> `RESULT {segments, area_cm2, cpu_h, cm2_per_cpu_h, verified_pass}`; hub side -> `python -m vesuvius_pipeline.cloud_import scan|verify|import --staging <export dir> [--apply]` (dry-run default).
4. Failure catalogue: `no kit available` -> set KIT_URL / upload the release asset; `sha256 mismatch` -> wrong/corrupt tarball (never bypassed); `kit self-check failed` -> missing system library (glibc >= 2.17); `S3 listing differs from the pin` -> upstream changed, re-pin deliberately (`pins/scrolls.json`); `voxel_um unknown` -> refuse (do not guess); rc 137 during seeds = OOM (coarse budget `MAX_COARSE_VOXELS` in seedprop.py); `gate_held` = degeneracy gate failed (fold-over / normal reversal / close pairs / hairpin / any transverse): exported for quarantine, not an error; uv/python download needs outbound HTTPS to github.com and PyPI.
5. Security: the package never reads or needs a hub token, private key, DB or hostname (`tests/test_routea_cloud.py` scans for them); the hub pulls the export tree and re-verifies it with the FLEET detector (cloud_import); anything unverified is quarantined, never recorded as an artifact.
6. Rebuild the branch tree: `deploy/routeA_cloud/tools/build_branch.sh MAIN_REPO WORKTREE_DIR` (copies library files from the main repo; nothing is edited on the branch). Proof numbers: docs/experiments/cloud_grow_2026-10-08/PUSHBUTTON.md.
