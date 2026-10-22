# Dynamic render vs stored render: EXCLUSIVE-DEVICE re-run (2026-10-22)

Supersedes the shared-device timings of the earlier tables in this folder (those stay as they were and are labelled SHARED DEVICE: the GPU and cores were
shared with training jobs, so their wall times are upper bounds). Same three paths and same 66 windows (22 PHerc0139 segments x 3 windows of 1024 x 1024 px,
28 planes), but:
- GPU: one Tesla V100-SXM2-16GB reserved, nothing else on it (state_before.txt: 0 % util, 7 MiB). Benchmark pinned to cores 16-31 (taskset), 16 threads for the CPU renderer. The other GPU of the machine and cores 0-11 were running two training jobs; load average 9.5 of 32 cores, CPU and IO pressure 0.
- 5 repeats per (window, path); repeats 0-1 COLD (file pages of the stored zarr and of the CT chunks the window needs dropped with posix_fadvise DONTNEED before each timed call), repeats 2-4 WARM (after an untimed warm-up of all three paths); path order shuffled within every repeat; one untimed CUDA initialisation call per process.
- Paths: P = read from the upstream pre-rendered surface-volume zarr; C = vc_render_tifxyz (CPU, --flip-normals); G = scripts/gpu_render/render_gpu.py.
- Not recorded: state AFTER the run (the post-run snapshot was not taken), GPU clocks during the run (only the idle value before). Disk cache of the CT volume is dropped only for the chunk files in the window bounding box.

## Wall time per 1024 x 1024 x 28 window (s; p10 / p50 / p90 over calls; 95 % CI of the median over 66 windows)
| path | state | calls | p10 | p50 | p90 | CI95 of median |
|---|---|---|---|---|---|---|
| stored read | cold | 132 | 0.023 | 0.030 | 0.035 | 0.029-0.030 |
| stored read | warm | 198 | 0.024 | 0.029 | 0.034 | 0.029-0.030 |
| CPU render | cold | 132 | 0.807 | 0.898 | 1.013 | 0.880-0.934 |
| CPU render | warm | 198 | 0.774 | 0.865 | 0.964 | 0.840-0.881 |
| GPU render | cold | 132 | 0.311 | 0.330 | 0.360 | 0.329-0.336 |
| GPU render | warm | 198 | 0.290 | 0.304 | 0.327 | 0.302-0.306 |
GPU is 2.81x faster than CPU (median of per-window medians, both states pooled). The stored read is about 10x faster than the GPU render.

## Identity
- Every path is deterministic: 0 of 66 windows had more than one distinct md5 across the 5 repeats, for P, C and G.
- No render is byte-identical to the stored volume: voxel-equal fraction p10 / p50 / p90 = 0.99602 / 0.99675 / 0.99763 (CPU) and 0.99602 / 0.99673 / 0.99763 (GPU); 0 of 297 (CPU) and 0 of 294 (GPU) calls byte-identical; the largest absolute difference is 1 grey level.
- Reproduce: `taskset -c 16-31 env CUDA_VISIBLE_DEVICES=1 python bench_render_excl.py segments.json results_excl.json --repeats 5 --windows 3`; `python analyse_excl.py`.
