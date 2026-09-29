# Progress-prize submission, September 2026: fixes to VC3D and `vesuvius` ink inference

Branch `progress/2026-09/tool-fixes` on `StewartSethA/villa`. It is based on `ScrollPrize/villa`
`main` at **6e53201** (2026-09-29). There is one commit per fix, so each can become its own upstream
PR, and this file is not part of any of them. Every defect was re-checked on 6e53201 and is still
present there.

| # | commit | subproject | fix |
|---|---|---|---|
| 1 | `core: replace segment directories atomically on every filesystem` | volume-cartographer | `QuadSurface::save(force_overwrite)` falls back to `remove_all(final)` then `rename(temp, final)` when `renameat2(RENAME_EXCHANGE)` is unavailable. A failure between those two steps destroys the finished segment. It also throws on NFS, overlayfs and FUSE errno values instead of falling back, skips the parent-directory fsync, and does not build against glibc < 2.28. The new `replaceDirectory()` renames the old tree aside, renames the new one in, and rolls back on failure. |
| 2 | `ink_detection: fix unbounded memory read in occupancy scan fallback` | vesuvius | `compute_nonempty_mask_from_lowres_array` read the whole array when the chosen "low-res" level is the native one (single-level volumes). It now reduces one chunk at a time, with identical output. |
| 3 | `NormalGridVolume: pick the eviction victim in O(1)` | volume-cartographer | Each eviction rebuilt a vector of every key under the exclusive lock and seeded a fresh `mt19937` from `random_device`. It now uses a dense key vector with swap-with-last. The eviction policy is unchanged. |
| 4 | `tracer: skip the space-line distance transform for blocks whose answer is a constant` | volume-cartographer | An all-foreground or all-background 96³ block has a known EDT result (0 or clamped 255), so the transform is skipped for it. The code path is only reached with `space_line_weight > 0`. |
| 5 | `docs: the tracer does not read the -v volume at default loss weights` + `docs: SURFACE_SDT also samples the -v volume` | volume-cartographer docs | The `thresholdedDistance` interpolators in `GrowPatch.cpp` are built and never evaluated. The volume is sampled only by SPACELINE, REFERENCE_RAY and SURFACE_SDT (with `cell_reopt_mode`), and all three are 0 or off by default. |
| 6 | `NormalGridVolume: optional byte budget for the grid cache (opt-in; default unchanged)` | volume-cartographer | `VC_GRID_CACHE_BYTES` sets an opt-in byte budget, with `VC_GRID_CACHE_ENTRIES` as a backstop. Decoded-polyline growth is re-counted on hits. The default stays the 512-entry cap. |

## Evidence recorded earlier (not re-run on 6e53201)

| # | measurement | n | base |
|---|---|---|---|
| 2 | Peak RSS ~187 GB → ~2.2 GB on a 65 × 32,249 × 51,380 surface volume | 1 segment | 2026-08 |
| 3 | User CPU 301.6 → 299.4 s (1.012×); 116 runs md5-identical | 3 paired reps, 1 segment (641 cm²) | 2026-08/09 |
| 4 | 1,085.5 → 954.7 CPU-s (1.137×), second pair 1.138×; `area_cm2` 523.845838 in both; x/y/z/generations.tif md5-identical; field byte-identical on 3 fixtures | 2 paired runs, 1 segment | d5c6f42 |
| 5 | Raw CT, surface prediction and prediction ∧ (CT > 5) as `-v` gave md5-identical tifxyz | 1 segment, 3 arms | b0bcee5 |
| 6 | CPU 70.8 → 44.2 s (1.60×), wall 74.7 → 51.9 s (1.44×), RSS 189 → 1,563 MB; output byte-identical at 6 budgets from 256 MiB to 8 GiB | 1 segment, replicate count not recorded | b0bcee5 |

Timings were not re-measured on 6e53201: the shared build host was under load (load average ~19
on 32 cores) on 2026-09-29. Before quoting them in a PR, re-time with command, input, build type
and p50 over at least 3 reps, as `AGENTS.md` §1.4 requires.

## Validation run on this branch, 2026-09-29

C++ test files were compiled one at a time with conda-forge gcc 15.3.0 (glibc-2.17 sysroot,
`-std=gnu++23 -O1`) against the repo's `test/doctest_compat` shim and the sources each test needs.
The Python test ran under Python 3.14.4, zarr 3.3.0, numpy 2.5.2 and torch 2.13.0.

| # | test | result | mutation, and whether the test went red |
|---|---|---|---|
| 1 | `core/test/test_directory_replace.cpp` + `DirectoryReplace.cpp` | 3/3 pass | Old ordering: `remove_all(target)` in place of rename-aside. **Red**: 2 checks fail (target missing, old data lost). |
| 1 | `QuadSurface.cpp`, `-fsyntax-only` | clean | — |
| 2 | `vesuvius/tests/ink_detection/test_occupancy_scan_memory.py` | 5/5 pass | Whole-array `array[:]` read restored. **Red**: `test_never_reads_more_than_one_chunk_at_a_time` fails. |
| 3 + 6 | `core/test/test_normal_grid_volume.cpp` + `NormalGridVolume.cpp`, `GridStore.cpp`, `MemMap.cpp` | 16/16 pass | Erase no longer releases bytes. **Red**: `CHECK(live >= 2)`. Known limit: corrupting #3's slot bookkeeping is not caught, because it only skews eviction sampling and results stay correct. |
| 3 + 6 | Existing `test_gridstore`, `test_gridstore_branches`, `test_gridstore_malformed` | 16 + 9 + 6 pass | — |
| 4 | `core/test/test_edt_homogeneous_blocks.cpp` (real `libs/edt`, 96³) | 3/3 pass | Black border requested from the transform. **Red**: the 255 case fails. Limit: the test checks the premise (the EDT's answer on uniform blocks), not the skip inside `GrowPatch.cpp`. |
| 4 | `GrowPatch.cpp`, `-fsyntax-only` | clean | — |
| 5 | The docs' own check (`grep -n 'interp_global\|interp(proc_tensor)' core/src/GrowPatch.cpp`) | The three interpolators are declared at lines 3582, 4819 and 4953 and never used. `SURFACE_SDT` also reads the volume, and the second docs commit adds it. | — |

**Not run:** a CMake configure/build of VC3D, `ctest`, the new `vc_add_test` stanzas, macOS,
Windows, or any end-to-end growth or inference on this base. Those are the first steps before
opening upstream PRs.
