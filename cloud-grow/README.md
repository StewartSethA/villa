# cloud-grow: seed and grow sheets of ONE scroll on one machine (or a rented one), no hub database, no hub token

Cut from the ScrollPrizeTutorial pipeline (fleet release `e1c069a7`, 2026-10-06; see `VENDORED.json`). It does five things:
1. **seeds** (coverage-driven, `cloud_grow/seeding.py`) and **grows** (`runner.py`: tracer `vc_grow_seg_from_seed`, SPACELINE arm,
   normal grids, resume only from a grown checkpoint, D3) with a **file-backed state shim** (`state.py`, one sqlite file) in place of the hub DB;
2. runs the **growth guard** (`growth_guard.py`, vendored unmodified apart from two import lines) with the production policy
   (`config/guard_policy.production.json`: selfcross on, fail closed, min_cells 1, cut_interior 1, stop_inherited 0, hairpin abort 0.66, every `*_enforce` = 1);
3. writes **per-round metrics + a provenance manifest** as new sidecars (`<seg>.export.json`, `rounds.jsonl`, `run_context.json`, `<seg>.md5tree`) and **packs** an export tarball (`pack.py`);
4. **pulls the data** resume-safely with a size preview (`data_fetch.py`, `scripts/fetch_data.sh`) and checks the box (`scripts/preflight.sh`);
5. **hub side:** `hub/import_remote_grow.py` verifies an export, RE-MEASURES it and registers it as append-only sidecar rows.

## Quick start (one scroll, one box)
```
pip install -r cloud-grow/requirements.txt                       # numpy scipy tifffile zarr<3 pytest
cd cloud-grow && python -m pytest                                 # offline tests (see TESTING.md)
scripts/fetch_data.sh PHerc0358 /data/vol --what all              # PREVIEW (sizes, free disk); add --yes to download; re-run to resume
cp config/cloud-grow.example.json run.json  &&  $EDITOR run.json  # paths, voxel_um (from config/scrolls.json), kit_bin
python -m cloud_grow check-tools --kit /opt/vc_kit/bin            # md5 pin gate + `file -L` (tools/BUILD_TOOLS.md)
scripts/preflight.sh --config run.json --grows 8 --passmark-st 2696 --usd-per-hour 1.28
scripts/run_grow.sh run.json 16 8 ./export                        # seed 16, grow 8 at a time, pack
```
Then, **from the hub** (the hub initiates the pull; the box never gets a route or a key to the hub):
`rsync --partial BOX:export/ ./inbox/ && hub/import_remote_grow.py --registry ./remote_registry --voxel-um 9.362 --kit-bin <hub kit>/bin ./inbox/*.tar.gz`
(`--dry-run` verifies and writes nothing).

## Data (what and how big)
Upstream = the public bucket `vesuvius-challenge-open-data` (anonymous, us-east-1; the same layout `vpipe data ensure` uses), listed in `config/scrolls.json`.
| item | needed for | size |
|---|---|---|
| surface prediction (nnU-Net m7, L0 th0.2 zarr) | tracer `-v` (production setting `grow.input.use_prediction=1`), guard ridge_hit/seam/empty_space | PHerc0358 18.9 GB (`prediction_bytes` per scroll in scrolls.json) |
| normal grids (`*.normal-grids/{xy,xz,yz}`) | tracer normal loss | PHerc0358 ~12 GB; size listed at fetch time; or regenerate with `vc_gen_normalgrids` (CPU, not timed) |
| CT levels 1..5 | guard CT sampler = level 1 (32-36 GB per scroll); seeding = level 4 (small) | **level 0 is NOT needed** (PHerc0211: 252 GB) |
Per scroll prediction+grids on the fleet (du 2026-10-06): 0172 5.6 GB, 0332 3.6, 0846A 13, 0826 22, 0211 32, 1218 32, 0358 30, 0257 34, 0813 34, 1203 35, 0125 37, 1447 43, 0191 47, 0800 80, 0268 87, Paris4 422 GB.
Data transfer is the dominant cost of a short rental: preview first. Not measured: our own uplink; the bucket's real download rate to a given provider.
Umbilicus files are NOT shipped (local, not human-vetted); without one the guard SKIPS wrap_spacing and curvature (announced, not scored clean).

## Cost / time
`python -m cloud_grow estimate --passmark-st X --cores N --usd-per-hour Y` prints the vast-benchmark model (EXTRAPOLATED, +-25 %, not validated out of sample; benchmark workload excludes late resume rounds, the 16 GiB cache and the guard; production-equivalent factor 0.2, calibration 0.13..0.33). Full tables: `docs/experiments/cloud_cost_2026-10-06/tables.md` and `vast_benchmark_2026-10-06/` in the fleet repo. Sizing: 1 grow per physical core; RAM >= max(64 GB, 3 GB x grows + 40 GB); `grid_cache_bytes` 256 MB is the BENCHMARK value (resume rounds at that size UNTESTED), production uses 16 GiB.

## What is validated, and what is NOT (D6)
* **Nothing here is validated against human annotation.** The guard criteria, the self-crossing detector (2 human verdicts exist in total), the sheet-skip detectors (TPR 0-80 %, FP 7-14 % on Paris4 n=70-80 clean / 50-60 planted skips) and the grown geometry carry the source fleet's caveats. `claims.validation_status` in every manifest says so.
* "verified_cm2" written on the box is the BOX'S OWN CLAIM (CT level 1 sample, topology on the lattice). The importer re-measures; PASS means "the box's claims reproduce on the hub", not "the sheet is right".
* Bar for equivalence to a hub grow: same tool md5 + same seed + same RNG seed + same inputs gives the same area (source benchmark: byte-identical areas on 5 CPU models). Not re-run here (no binary, no rental).
* Differences from the production loop, all announced in `run_context.json["not_ported"]`: no SPAGHETTI/HOLEY/MUSHY pause gates, no neighbour seeding, no hub neighbour cover (guard `overlap` / `merge_pause` skipped), no lasagna normal sampler (`normal_dev` skipped), flatten_feedback inert; seed verification at CT level 1 (not 0); box-side area scoring at level 1.
* Importer area rule, measured on 80 raw + 30 guarded real checkpoints: guard-written `meta.json` area equals the lattice area exactly (enforced, tol 1 %); the RAW tracer area differs (rel p10/p50/p90 -0.0004/+0.0066/+0.027, min -0.56, max +0.065), so for raw checkpoints the claim is recorded, not enforced, and the registered area is the hub measurement.

## Security
* **Never put the hub token, ssh keys, DB files, labels or any other scroll on a rented box.** The package has no code path that reads `hub.token`; `pack` refuses secret-like files/content. A rented host is untrusted: a fresh key per rental, forced-command rsync/receive-only, hub-initiated pull, destroy the instance and its disk afterwards.
* The prediction zarr and grids are project data, not public-redistributable by default: prefer short rentals; the provider's data-retention policy is NOT verified (user decision).
* The importer extracts tarballs safely (no absolute paths, `..`, links, devices), never modifies an existing meta/result/manifest, and appends only.

## Retention
Keep on the hub: the tarball + its `.sha256` + the registry rows (`imports.jsonl`, `<seg>/*.import.json`). On the box: delete everything after the hub confirms PASS; `cache_root/` (GBs, regenerable) is never shipped.

## Labels / vetting UI (design only, not built)
`schema/label_events.schema.json` + `schema/VETTING_UI_DESIGN.md`: append-only, severable per-channel 3-state vertex labels; unlabelled = unknown = masked, never a penalty; detector output shown as proposals whose accept/reject measures detector precision/recall. Nothing in cloud-grow requires a label.

## Layout
`cloud_grow/` code - `config/` policy, pins, scrolls, example config - `scripts/` preflight, fetch, run - `hub/` importer CLI - `tools/` kit docs + `make_portable_kit.sh` - `schema/` manifest spec + label events - `tests/` - `DEPLOY.md` - `TESTING.md` - `VENDORED.json`.
