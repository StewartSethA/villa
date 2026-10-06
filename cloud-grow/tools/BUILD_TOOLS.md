# Getting the VC3D tools (tracer + self-crossing census)

Needed on the grow box: `vc_grow_seg_from_seed` (tracer) and `vc_tifxyz_selfcross` (guard census). The hub also needs
`vc_tifxyz_selfcross` for the importer's re-check. GPU is not needed.

## Pins (config/tool_pins.json)
| tool | md5 | where it came from |
|---|---|---|
| `vc_grow_seg_from_seed` | `b747f7658a21aca5bbb05e08c95e192b` | the source fleet's portable kit (same md5 on every benchmark host) |
| `vc_tifxyz_selfcross` | prefix `73800e99` (full md5 not recorded in the source docs; run `md5sum` on a kit you trust and add it) | same kit, identical on 9 peers |
| NOT the pin | `06f684441120ae0de85d1dbaf6349574` | the VC3D AppImage tracer: a DIFFERENT build, different areas (1.668 vs 1.638 cm2 on one seed) |

## Route 1 (recommended, reproduces the pinned geometry): copy the kit
`scp` the source fleet's kit directory (`bin/` + `lib/`, 810 MB; produced by `tools/make_portable_kit.sh`) to the box,
then `python -m cloud_grow check-tools --kit <kit>/bin`. Needs GLIBC >= 2.14, libboost 1.84, opencv 4.12 (shipped in `<kit>/lib`;
the runner sets `LD_LIBRARY_PATH=<kit>/lib`). Verify with `md5sum` AND `file -L` (a "binary" can be a shim; D24).

## Route 2: build from upstream
Source: `volume-cartographer/` in https://github.com/ScrollPrize/villa (apps `vc_grow_seg_from_seed`, `vc_tifxyz_selfcross`
are in `apps/CMakeLists.txt`). On Debian/Ubuntu (needs sudo and installs packages, so read it first):
```
cd villa/volume-cartographer && ./build_from_src_debian.sh        # BUILD_DIR, PREFIX, JOBS env vars
tools/make_portable_kit.sh --build <build_dir> --out ~/vc_kit vc_grow_seg_from_seed vc_tifxyz_selfcross
```
(`make_portable_kit.sh` here is a verbatim copy of the fleet's; it resolves the transitive libraries so the kit runs anywhere of the same arch.)
CAVEAT, measured fact only: the commit that produced md5 `b747f765` is NOT recorded in the docs this package was cut from.
A fresh build will almost certainly have a different md5, so `check-tools` will refuse it (use `--allow-unpinned`; it is recorded in the
manifest and the hub importer refuses it unless `--accept-unpinned`). Before trusting a new build, run the determinism check the
fleet used: the same seeds must give the same total area on both builds (benchmark: 4 jobs = 6.49 cm2 on five CPU models).
The source fleet's local tracer carries uncommitted/branch changes (see `docs/experiments/upstream_candidates_2026-09-25.md` in the
fleet repo); none is claimed here to be in the pinned binary.

## Not tested here
No binary was built or run in producing this package (no rented box, no root). The test suite uses fake tools that honour the same CLI.
