# fit_spiral: device memory grows with iteration count (reference cycles)

## Problem
`fit_spiral` builds fresh transform objects every step (`get_slice_to_spiral_transform`,
`IntegratedFlowDiffeomorphism` and its `_cached_integrator` closures). They sit in reference
cycles, so refcounting never frees them. Each holds a full gap-logit lattice copy and a
full-resolution flow copy. Only Python's cyclic collector frees them, whenever it happens to run,
so CUDA memory grows with the iteration rate until a collection fires.

## Evidence (PHerc0343, z 9000-10500, shell 250, 16-voxel flow, V100-SXM2-32GB, same recipe/seed 1)
Live-tensor census, no collection: gap-logit-sized tensors `[1,1,75,145968]` count 7 -> 89 -> 67 at
iterations 13 / 149 / 272 (drops whenever the cyclic GC happens to run); flow-sized copies
`[3,Nz,400,400]` 3 -> 44.

| run | collection | torch max allocated | iterations |
|---|---|---|---|
| m1_z1500_s250_f16 | none | 8.518 GiB | 323 |
| m1gc_z1500_s250_f16 | `gc.collect()` every 0.5 s (side thread) | 3.141 GiB | 320 |

Consequences: an RTX 3090 (24 GB) ran out of memory at iteration 206 without it; a single fit over
the whole of PHerc0343 (z 1200-17500, shell 250, 32-voxel flow grid) peaks at 20.13 GiB torch
(21.6 GB nvidia-smi) on a 32 GB V100 with collection, but ran out of memory at iteration 18 without.

## Fix
`fit_spiral.py`: `gc.collect()` after every optimiser step (`FIT_SPIRAL_GC_EVERY_STEPS`, default 1,
0 disables). Results are unchanged; only unreachable objects are freed. The cleaner fix is to break
the cycles (drop the cached integrator/sampler and pinned logits at the end of `step()`, or hold the
transform weakly in the closures); this patch is the minimal version.

## Honesty notes
- The table was measured with an equivalent external hook (collection from a side thread every
  0.5 s). The in-source per-step variant was only syntax-checked here; a per-step hook
  (`FIT_GC_EVERY_STEP=1`, wrapping `AdamW.step`) was measured to bound memory, but its collection
  time per step was not yet measured.
- One scroll, one seed, ~320 iterations per arm; throughput numbers of the two runs are not
  comparable (different cards busy).

## Reproduce
Run any 1500-slice window twice with `FIT_SPIRAL_GC_EVERY_STEPS=0` and `=1`; read
`torch.cuda.max_memory_allocated()` at iteration ~320.
