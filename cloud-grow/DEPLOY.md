# DEPLOY (one page)

GPU not needed (flatten/render/ink stay on the hub). Linux x86-64, glibc >= 2.14, Python >= 3.10.

| step | command | check it worked (a DIFFERENT observation) |
|---|---|---|
| 1 rent | >= 64 physical cores or fewer (1 grow per core), RAM >= max(64, 3 x grows + 40) GB, NVMe >= inputs + 20 GB, 500 Mbit/s+ down. Price/time: `python -m cloud_grow estimate ...` | `scripts/preflight.sh` prints OK |
| 2 install | `pip install -r requirements.txt`; copy the VC3D kit (`tools/BUILD_TOOLS.md`) | `python -m cloud_grow check-tools --kit <kit>/bin` -> `tool gate: PINNED` |
| 3 data | `scripts/fetch_data.sh SCROLL /data/vol --what all` (preview) then `--yes`; interrupted? run again | `.cache_complete` in each store; `fetch_log.jsonl` has no FAILED; preflight `ct_level_1`/`ct_level_4` PASS |
| 4 run | `scripts/run_grow.sh run.json N_SEEDS PARALLEL ./export` | `rounds.jsonl` per segment; grep stderr for `ALERT`; stdout shows failure rate: `done: N segment(s), K failed (x %)` |
| 5 ship | hub: `rsync --partial BOX:export/ inbox/` | `.sha256` matches; tarball ~0.5-5 MB per segment |
| 6 import | `hub/import_remote_grow.py --registry R --voxel-um V --kit-bin HUBKIT/bin inbox/*.tar.gz` (add `--dry-run` first) | exit 0; `R/imports.jsonl` has PASS rows; a REFUSED row names its reason |
| 7 destroy | delete the instance + disk + the per-rental key | provider console shows none left |

Failure policy: a failure rate above a few per cent is a stop signal (read the `why` strings in `rounds.jsonl` first). A round whose self-crossing check could not run PAUSES that grow and reverts to the last verified surface (never ships unverified cells); `selfx_unverified.json` marks a raw surface the importer refuses.
Resume: re-run `grow`; a segment with earlier rounds resumes from its newest checkpoint (by mtime), never re-grows (D3). Cloud resumes of segments grown elsewhere need the checkpoint copied in (hub push), then `--resume`-style by placing it under `segments/<seg>/r<N>/<ckpt>/`.

## Push-button fleet (provision/, 2026-10-06, DRY-RUN-ONLY, untested live)
Steps 1-7 above for a whole fleet in one command: `provision/deploy.sh PLAN.json --checklist | --dry-run | --smoke --yes | --yes | --abort` (real calls need `--yes` and `FLEET_EXECUTE=1`). Read `provision/README.md` (the honest time/cost answer, what can break, result path) and `provision/RUNBOOK.md`. Run `--smoke` (ONE box, ONE hour) and read the importer's REFUSED reasons (self-crossing rate) before scaling.
