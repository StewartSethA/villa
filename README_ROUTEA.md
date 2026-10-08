# Route A pushbutton grow (routeAB-deploy)

Grow Vesuvius surfaces for one or more scrolls with the **production Route A guard stack**, on any Linux x86-64 box (fleet host or rented), with one command:

```
git clone -b routeAB-deploy --depth 1 https://github.com/StewartSethA/villa.git && cd villa && ./routeA_run.sh --scrolls PHerc0332,PHerc0211 --seeds 8 --hours 6
```
(already cloned: `./routeA_run.sh --scrolls PHerc0332,PHerc0211 --seeds 8 --hours 6`.) If the repository is private, give the box a read-only **deploy key** (fresh per box, GitHub -> Settings -> Deploy keys, read-only): `ssh-keygen -t ed25519 -f deploy_key -N ''`, add `deploy_key.pub` there, then `GIT_SSH_COMMAND='ssh -i deploy_key -o IdentitiesOnly=yes' git clone -b routeAB-deploy --depth 1 git@github.com:StewartSethA/villa.git`. The key lives only on that box and is never part of this package; the package itself needs no credentials at all.

Needs only `bash`, `tar`, one of `curl`/`wget`/`python3`, and outbound HTTPS. No root. Everything else is downloaded, **pinned and verified**, into `./routeA_work/` (or `--workdir DIR`):

| what | from | verification |
|---|---|---|
| `uv` + CPython 3.12 + locked packages (numpy, scipy, tifffile, imagecodecs, zarr, numcodecs) | GitHub release / PyPI | `pins/uv.env` sha256; `pins/requirements.lock` is hash-locked (`--require-hashes`) |
| VC3D kit: patched `libvc_tracer.so` (in-solve self-collision guard, md5 94be1eae), `vc_grow_seg_from_seed` (b747f765), `vc_tifxyz_selfcross` (73800e99) + 76 libs | the GitHub release asset named in `pins/kit.json` (override: `KIT_URL`, see below) | tarball sha256 + per-file md5 (`pins/kit.json`), `--help` self-check |
| per scroll: surface-prediction zarr (the tracer's `-v` volume) + normal grids | public bucket `s3://vesuvius-challenge-open-data/<scroll>/representations/predictions/surfaces/` (anonymous HTTPS) | every object size + md5/ETag, object count/bytes vs `pins/scrolls.json` |
| umbilicus estimates (seed spreading over wraps) | `umbilicus/<scroll>/umbilicus.json` (in this repo) | — |
| guard settings | `pins/settings_snapshot.json` = the production `grow.guard.*` settings + resume-gate / degeneracy / selfcontact tunables | — |

**Kit hosting (one-time, by the owner):** the kit tarball (`routeA-kit-94be1eae.tar.xz`, 100 MB) is too big for git. `pins/kit.json` expects it at `https://github.com/StewartSethA/villa/releases/download/routeA-kit-94be1eae/routeA-kit-94be1eae.tar.xz`: upload once with `gh release create routeA-kit-94be1eae routeA-kit-94be1eae.tar.xz --repo StewartSethA/villa --title routeA-kit --notes 'sha256 9ac4d5cd...'` (any other HTTPS location works: `KIT_URL=https://.../routeA-kit-94be1eae.tar.xz ./routeA_run.sh ...` or `--kit-url`). The sha256 pin makes the host untrusted-safe. Building the kit from source instead: `BUILD_KIT.md`.

Scrolls with public prediction + grids (pin sizes in `pins/scrolls.json`, prediction+grids GB): PHerc0332 3.8, PHerc1203 5.5, PHerc0846A 7.8, PHerc0826 23, PHerc0490B 28, PHerc0358 31, PHerc0211 34, PHerc0125 40, ... PHerc0268 93.

## What runs on the box (all of it, same code as the fleet)
`routea_cloud/run.py` -> `cloud_box.grow_seed` per seed (one core each): tracer rounds with `self_collision` ON (production params), **selfx scrub** (transverse crossings cut to a verified zero) and the **degeneracy resume gate** (fold-over, normal reversal, close pairs, hairpin + any transverse; fail closed: a failing surface is not grown on and is exported for quarantine) between rounds. Flag/stop policy: no segment-level roughness/hairpin abort (= the production `segment_abort_as_flag` flag policy; the box does not need the CT-dependent criteria), a growth ends on rounds exhausted / `--hours` budget / gate hold. Then every export tree is self-verified with `cloud_import.verify` using the kit's own detector.

Results: `routeA_work/export/<scroll>/<seg>/` (`export.json`, `md5.txt`, checkpoints) and `routeA_work/report.json` (phase timings, GB downloaded, cm2, cm2 per CPU-hour, verify verdicts).

## Importing into the hub (pull-based; the box is untrusted)
The box never gets a hub token, ssh key or DB. The hub pulls the export tree (`rsync` from the hub side), then `python -m vesuvius_pipeline.cloud_import scan|verify|import --staging DIR [--apply]` (full repo): manifest + fleet-detector census (0 transverse tolerated) + degeneracy gate; verified -> new segment + provenance sidecar; anything else -> quarantine, never an artifact.

## Tests
`pip install pytest && pytest` (offline; uses a fake S3/HTTP server and synthetic zarr/tifxyz; tests needing the real `vc_tifxyz_selfcross` skip without it).

## Publishing this branch
```
git remote add origin <GITHUB_URL>
git push -u origin routeAB-deploy
```
