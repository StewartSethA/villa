# Route B pushbutton (spiral-fit sheets -> tiles -> flatten -> render -> ink)

One command, identical on a laptop-with-a-GPU and on a rented box:

```
git clone -b routeAB-deploy --depth 1 https://github.com/StewartSethA/villa.git && cd villa && \
  ./routeB_run.sh --scrolls PHerc0211,PHerc0125 --stripe-width 4500
```

`--stripe-width 4500 | 7500 | 13500 | full` (default `full`; any integer works). Add `--smoke` for the proof-sized run
(z 9000-9500, 1500 fit steps, 1 tile of a few cm2, ~15 min after the downloads). `./routeB_run.sh --help` lists every knob.

## What it needs from the box
Linux x86_64, an NVIDIA GPU with driver >= 525 (16 GB is enough for the quarter-width fits; full width wants 32 GB or
`--fit-overrides '{"model_flow_voxel_resolution": 32}'`, already the default above 6,000 slices), `g++` >= 13 (one C++23 extension is built once, ~25 s),
`curl` or `wget`, and **free disk**: ~8 GB per scroll for the tracks + the lasagna z-slab, + ~1-4 GB of CT chunks per tile set.
No root, no conda, no docker. Everything else is downloaded, hash-checked and cached under `$ROUTEB_HOME` (default `./routeB_work`):

| what | from | verification |
|---|---|---|
| uv (static binary), CPython 3.14.7, ~120 pinned wheels (torch 2.13+cu126) | github.com/astral-sh/uv, python-build-standalone, pypi, download.pytorch.org | uv sha256 pinned; wheels by version pin (`routeB/pins/requirements.lock`) |
| spiral-fit tracks (`.dbm`, 4-13 GB/scroll) | dl.ash2txt.org/datasets/spiral_datasets | size vs Content-Length always; md5 pinned where we hold a reference (`routeB/scrolls/pins.json`), else md5 recorded in `assets/.fetched/` |
| lasagna normals/grad-mag (z-slab of the group-2 zarr) | s3://vesuvius-challenge-open-data (anonymous HTTPS) | per-object size + S3 ETag md5 |
| CT chunks (only the 128^3 chunks a tile touches) | same bucket | per-object ETag md5; 404 = absent chunk = zeros (masked scan) |
| ink weights (4 small U-Nets, 24 MB) | in the branch (`models/var/models`, `models/MODELS.md5`) | md5 listed; written into every export manifest |
| umbilici (our estimates, one small JSON per scroll) | in the branch (`routeB/umbilicus/<scroll>/`) | `routeB/scrolls/<scroll>.json` says which file is used and its provenance |

Downloads are resumable (`.part` + block map) and a re-run downloads nothing that is already verified.

## Stages and what they leave behind (`$ROUTEB_HOME/...`)
1. **fetch** `assets/<scroll>/dataset/{tracks,lasagna_inputs}`; ledger `assets/.fetched/*.json` (+ `_totals.json`: bytes over the network).
2. **fit** `runs/<scroll>/<stripe>/fit/` `fit_spiral.py` (checkpoint-restart chain: a CUDA OOM or crash resumes from the last autosave, stops after a chunk with no progress). Marker `.done.fit.json` holds `satisfied_track_points`.
3. **tiles** `runs/.../tiled/manifest.json` + `tiles/<tile>/{x,y,z}.tif` (per-winding meshes cut to <= 26 cm2 tiles; no database needed).
4. **ink** per tile in `runs/.../tilework/<tile>/`: `snapped/` (recto sheet snap), `flat/` (lasagna flatten), `layers/` (17 centred layers at 9.5 um/px), `ink/<family>.png`, `tile_result.json`.
5. **export** `export/<scroll>/<stripe>/{ink/*.png, manifest.json, FILES.txt}` and `export/<scroll>/routeB_<scroll>_<stripe>.tar`. Pull with `rsync -a box:routeB_work/export/ ./`.

Failures are loud: `ROUTEB FAIL <stage>: <why>` from the shell preflight, `STAGE FAILED ...` / `TILE FAILED ...` from the driver, exit status 1 and a `FAILED:` list
at the end. A failed tile/stripe/scroll never stops the others. Re-running resumes (stage markers, fit checkpoints, cached assets).

Scrolls without published tracks (PHerc0846B, 1203, 1218, 1447, 1545, Paris4) stop at *fetch* with a clear message: extract tracks first (`spiral-fitting/extract_surface_tracks.py`).
See `TROUBLESHOOTING.md` and `AGENT_GUIDE.md`.
