# Building the patched kit from source (alternative to KIT_URL)

The tracer patch (in-solve self-collision guard: hard reject of new vertices / rollback of moved vertices that create a transverse triangle-triangle contact; opt-in `params.json` key `self_collision`) is `patches/tracer_selfcollision_94be1eae.diff` against villa `volume-cartographer/` at the base commit `d8c87dd7` (upstream villa `b408d54c` + portable-build fixes: CLI-only cmake, `renameat2` syscall). The kit used in production (libvc_tracer md5 94be1eae) was built from the state `7a35969c` of that patch series. `patches/tracer_selfcollision_minsep_orient.diff` adds the opt-in `min_separation` and orientation-exclusion rules (pilot kits, not deployed).

```
git clone <villa repo> && cd villa && git checkout b408d54c   # + the portable-build fixes (CLI-only cmake, renameat2) from d8c87dd7
git apply <this repo>/patches/tracer_selfcollision_94be1eae.diff
# toolchain used: conda env with gcc/glibc-2.17 sysroot (portable binary); cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DVC_BUILD_APPS=ON -DVC_BUILD_PYTHON=OFF -DVC_TESTING=OFF -DVC_BUILD_FLATBOI=OFF -DCMAKE_INSTALL_RPATH='$ORIGIN/../lib'
cmake --build build --target vc_grow_seg_from_seed vc_tifxyz_selfcross
# kit = bin/{vc_grow_seg_from_seed,vc_tifxyz_selfcross} + lib/ (the libs they load: ldd with LD_LIBRARY_PATH=lib), then:
tar -C KITDIR -cf - bin lib | xz -T8 -6 > routeA-kit-<tag>.tar.xz
python3 tools/pin_kit.py KITDIR routeA-kit-<tag>.tar.xz > pins/kit.json
```
Honest limits: a rebuilt `libvc_tracer.so` will not be byte-identical (toolchain/paths), so its md5 differs from 94be1eae: re-pin with `tools/pin_kit.py`, and the hub's import check re-verifies every OUTPUT with the fleet detector anyway. The prebuilt, sha256-pinned tarball is the supported path.
