> NOTE 2026-10-22: every timing in this README was taken on a SHARED device (GPU and CPU shared with training jobs); see `exclusive_device/README.md` for the exclusive-device re-run, which supersedes them for wall time.

# Dynamic rendering versus pre-rendered layers: disk, CPU time and GPU time

Date: 2026-10-22. Scroll: PHerc0139 only.

## Question

Training an ink model needs, for every segment, a stack of image planes sampled from the CT volume along the surface of the segment (a "surface volume"). Two ways to supply it:

1. **Pre-rendered.** Render the whole segment once, store the planes on disk, read windows from the file.
2. **Dynamic.** Keep only the CT volume and the segment mesh. Render each window when it is asked for, on the CPU or on a GPU.

This study measures, on the same segments and the same windows, what each way costs in disk bytes, wall time and megavoxels per second, and whether the voxels are identical.

## What is compared

For one segment the mesh is a grid of 3-D points (a "tifxyz" folder: x.tif, y.tif, z.tif and meta.json, about 1.6 MB). A rendered plane k at pixel (u, v) is the CT value at

    p(u, v) + (k - c) * s * n(u, v)

where p is the mesh point, n the unit surface normal, s the step (1 voxel) and c the centre index (c = (N-1)/2 for N planes), sampled with trilinear interpolation and rounded to 8 bits. The three paths:

| path | what runs | stored on disk |
|---|---|---|
| P, pre-rendered | read a window out of the stored uncompressed zarr (28 planes, chunks 28 x 128 x 128) | the whole stack of the segment |
| C, CPU dynamic | `vc_render_tifxyz` (Volume Cartographer, OpenMP, 16 threads) with `--crop-*` | nothing |
| G, GPU dynamic | `scripts/gpu_render/render_gpu.py` (the same arithmetic as C on a GPU) with `rect=(y0, y1, x0, x1)` | nothing |

The CT volume (PHerc0139, 20974 x 6621 x 6621 voxels, 9.362 um, uint8, uncompressed, 920 GB) and the mesh are needed by C and G and are not counted as stored output.

## Data

- 22 labelled PHerc0139 segments (20 with dense reader-v2 labels plus 2 of our own training segments), all with a 9.362 um mesh and an upstream pre-rendered surface volume of 28 planes made from the same CT scan (20250728140407). List:
- 20250108000000-w025_2025010863
- 20250108000001-w026_2025010854
- 20250108000002-w027_2025010845
- 20250108000003-w028_2025010836
- 20250108000005-w030_2025010818
- 20250223000000-w059_2025022312
- 20250831000000-w040_2025083102
- 20251226000000-w055_2025122611
- 20260108000000-w041_2026010816
- 20260112000000-w043_2026011217
- 20260115000001-w056_2026011514
- 20260126000000-w045_2026012619
- 20260127000000-w057_2026012713
- 20260130000000-w031_2026013019
- 20260203000000-w032_2026020303
- 20260206000000-w042_2026020613
- 20260206000002-w054_2026020617
- 20260210000000-w058_2026021020
- 20260219000000-w033_2026021913
- 20260219000001-w048_2026021912
- 20260227000000-w052_2026022705
- 20260302000001-w050_2026030220
- 3 random windows of 1024 x 1024 px x 28 planes (29.36 Mvoxel) per segment, chosen where at least 97 % of the pixels are inside the segment (seed 0): 66 windows. A second set of 22 windows (seed 5) was rendered with the GPU first, see "Caveats".
- The pre-rendered stack of each segment is the upstream open-data file `surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr`. Its `.zattrs` records: 28 slices, slice step 1.0, group 0, source CT volume `SCROLLS_HEL_4.681um_113keV_1.2m_binmean_2_PHerc_0139_TA_0001_masked.zarr`.

## Positive control

The upstream surface volumes were produced by this same tool. Rendering the same mesh from the local copy of the CT volume with `vc_render_tifxyz --flip-normals` (plane order reversed relative to the default) must give the same voxels.

| comparison | result |
|---|---|
| CPU render of the **whole canvas** of segment 20250108000000-w025 (28 planes, 1.386 Gvoxel) against the upstream file | 99.741 % of voxels equal, largest difference 1 grey level |
| CPU render of a 1024 x 1024 window of the same segment against the upstream file | 99.703 % equal, largest difference 1 |
| all 66 windows, CPU against upstream | 99.60 / 99.67 / 99.76 % equal (p10 / p50 / p90), largest difference 1 |

So the harness reproduces the upstream render to within 1 grey level on every voxel, but it does **not** reproduce it byte for byte (0 of 66 windows are byte-identical). The differing voxels are spread over the whole canvas (about 98 % of rows contain at least one), the two signs are equally common (0.12 to 0.15 % each), and they sit on slightly brighter than average pixels. This pattern fits rounding to 8 bits in a different build of the renderer; I did not confirm that cause.

## Results

Hardware: AMD EPYC 7282 (16 cores, 32 threads), 499 GB RAM, 2 x Tesla V100-SXM2-16GB (one used), ext4 on a RAID volume, Linux 7.0.0-38, torch 2.2.0+cu121, zarr 3.1.5. The machine was shared with other jobs: load average 13 to 16 during the runs and the GPU used was also running other work. md5 sums and commits: `results/hardware_and_md5.txt`. Raw rows: `results/results_bench.json` (per window, with the md5 of every output), `results_full.json`, `results_cold.json`.

### Wall time and speed per 1024 x 1024 x 28 window (n = 66 windows, 22 segments), p10 / p50 / p90

| path | seconds | Mvoxel per second |
|---|---|---|
| P, read from the stored zarr | 0.0232 / 0.0286 / 0.0313 | 937 / 1027 / 1266 |
| C, CPU render | 0.652 / 0.696 / 0.742 | 39.6 / 42.2 / 45.0 |
| C plus reading the TIFFs it writes | 0.911 / 0.957 / 1.048 | 28.0 / 30.7 / 32.2 |
| G, GPU render (steady call) | 0.282 / 0.330 / 0.427 | 68.8 / 88.9 / 104.1 |
| G, first call of a process | 0.280 / 0.309 / 0.416 | |

Per segment (median of its 3 windows, n = 22 segments): P 0.0245 / 0.0284 / 0.0308 s; C 0.664 / 0.688 / 0.720 s; G 0.289 / 0.339 / 0.423 s.

A GPU call has a fixed cost of about 0.25 s, so small windows are slow per voxel. Rendering a **whole canvas** in one call is much faster (n = 22 segments, p10 / p50 / p90):

| whole canvas, 28 planes (1.37 / 1.50 / 2.10 Gvoxel) | seconds | Mvoxel per second |
|---|---|---|
| CPU, writing TIFFs | 20.8 / 23.5 / 35.3 | 59.6 / 63.2 / 68.0 |
| GPU, no write | 3.7 / 4.9 / 6.7 | 294 / 309 / 405 |
| GPU, plus writing the uncompressed zarr | +4.6 / +5.1 / +7.3 | |

### Disk (bytes per segment, n = 22)

| | p10 / p50 / p90 |
|---|---|
| upstream pre-rendered 28-plane zarr | 1.63 / 1.77 / 2.59 GB |
| our own pre-render written as uncompressed zarr (all-zero chunks omitted) | 1.20 / 1.31 / 1.92 GB |
| mesh needed by C and G | 1.46 / 1.60 / 2.25 MB |

For these 22 segments the stored stacks add up to about 39 GB of upstream files; dynamic rendering stores 35 MB of meshes next to the CT volume that has to be on the machine in any case.

### Identity

Fraction of voxels exactly equal, over 66 windows (1.94 Gvoxel), p10 / p50 / p90:

| pair | equal | largest difference | windows with zero differing voxels |
|---|---|---|---|
| C against P | 99.60 / 99.67 / 99.76 % | 1 | 0 / 66 |
| G against P | 99.60 / 99.67 / 99.76 % | 1 | 0 / 66 |
| G against C | 100.0000 % (5412 voxels differ out of 1.94 G) | 1 | 0 / 66 |

The GPU and the CPU renderer agree on 99.9997 % of voxels; both disagree with the stored upstream file on 0.3 % by exactly 1 grey level. The three paths are therefore pixel-identical only up to a one-grey-level rounding difference.

### One call serves any depth, offset and face (n = 22 segments, one window each)

`render(..., num_slices=N, slice_step=-1, z_offset=z)` places plane k at offset (k - (N-1)/2 + z) steps along the normal (`z_offset` is an optional argument added in this branch; the default 0 leaves every existing call unchanged). A negative step is the same as `--flip-normals`; a positive step walks the normal the other way, i.e. towards the other face of the sheet. Wall time per call, p10 / p50 / p90, and whether the planes equal the matching planes of a 65-plane render of the same window:

| call | seconds | equal to the matching planes of the D65 render |
|---|---|---|
| 17 planes, z = 0 | 0.271 / 0.299 / 0.401 | 22 of 22 |
| 17 planes, z = +3 (depth jitter) | 0.270 / 0.319 / 0.402 | 22 of 22 |
| 17 planes, z = -4 | 0.271 / 0.279 / 0.394 | 22 of 22 |
| 33 planes, z = 0 | 0.289 / 0.316 / 0.408 | 22 of 22 |
| 33 planes, z = +3 | 0.303 / 0.321 / 0.434 | 22 of 22 |
| 33 planes, z = -4 | 0.287 / 0.323 / 0.418 | 22 of 22 |
| 33 planes, step +1 (opposite direction, reversed order) | 0.302 / 0.328 / 0.433 | 22 of 22 equal to the reversed D65 planes |
| 65 planes | 0.339 / 0.387 / 0.507 | reference |

Time hardly depends on the number of planes (0.30 s at 17 planes, 0.39 s at 65), because the fixed cost dominates. Disk cost of every row: zero beyond the CT volume and the mesh.

Where the other face is: the mean grey value along the normal (65-plane render, n = 22 windows) rises to a bright band that is the papyrus sheet. The band (above half height) has its low edge at layer 26 and its high edge at layer 33.5 (p50 of 22 windows; the mesh plane is layer 32), 8.5 layers wide at p50 (about 80 um at 9.362 um per layer; p10 / p90 of the width 6.1 / 19.8 layers). The far face is therefore about 6 layers (56 um) from the mesh plane; a window centred there with `slice_step=+1` is a verso view built from the same mesh and volume. I did not train on it in this study.

### Cost model (median canvas 1.496 Gvoxel, 28 planes)

Time to read every voxel of a canvas once (one epoch), from the window rates above: P 1.5 s, C 35.5 s, G (window calls) 16.8 s, G (whole-canvas calls) 4.9 s. Producing the stored copy once costs 23.5 s on the CPU or about 10 s on the GPU (render plus zarr write) and 1.3 GB. A stored copy pays for itself after fewer than one epoch in time; it never pays for itself in disk. Two limits of this arithmetic: the reads of P come from a warm file cache; and a stored copy fixes the number of planes, their order and the face at render time.

## Limits of stack-backed training in this repository

The segment stacks used for training here (`stack.npy`, channels last) are pre-rendered with exactly **17 planes** around the label plane (and 28 planes for the 9.4 um segments). They cannot serve a model that wants 33 or 65 planes, a different depth offset, or the other face. The call above can.

## Caveats (what this study does not show)

- One scroll, one CT scan, 22 segments, 66 windows. The segments are not independent samples of all scrolls; the percentiles are over windows or over segments, with no confidence intervals.
- Timing was taken on a loaded, shared machine, one run per window. The GPU was shared with other jobs; the CPU used 16 of 32 threads while the load average was 13 to 16. Absolute numbers will differ on an idle machine.
- The page cache was not controlled (no root to drop caches). In the main run the CPU touched each window before the GPU, warming the CT chunks. A second run of 22 windows (seed 5) rendered with the GPU first: GPU first call 0.330 / 0.407 / 0.420 s, steady call 0.322 / 0.405 / 0.416 s, CPU 0.650 / 0.695 / 0.744 s. A true cold-disk read was not measured.
- Window size was one value (1024 x 1024). Windows of 128 to 384 px, the sizes used for training, pay the 0.25 s fixed cost of a GPU call many more times per voxel; this was not measured.
- The identity statement is limited to 1024 x 1024 windows and 28 planes; the residual one-grey-level difference to the upstream file is not explained.
- The 65-plane render is used as the reference for the consistency checks; it is not independent of the 17- and 33-plane calls (same code).

## Setup (all paths come from one config file)

```
cp config.example.json config.json          # edit: paths are relative to the config file unless absolute
```

| key | meaning | default |
|---|---|---|
| `data_dir` | one folder per segment: `<segment>/sv.zarr` (the stored surface volume, uncompressed zarr) and `<segment>/mesh.tifxyz` | `data` |
| `volume` | the CT volume zarr (level 0 uncompressed) the renders are cut from | `volume/PHerc0139.zarr` |
| `cpu_renderer` | the `vc_render_tifxyz` binary (see CPU-only build) | `bin/vc_render_tifxyz` |
| `tmp_dir`, `out_dir` | scratch and output folders | `tmp`, `out` |
| `num_slices` | planes per window | `28` |

Set `STUDY_CONFIG=/path/to/config.json` to use a file elsewhere. Python needs `numpy`, `tifffile`, `zarr`; the GPU path also needs `torch` with CUDA (kernels are compiled at run time with NVRTC from the `nvidia-cuda-nvrtc` wheel, so no `nvcc` is needed).

### CPU-only build of the comparison renderer (no GPU, no CUDA)

The CPU path (`C` in the tables) is `vc_render_tifxyz` from the Volume Cartographer toolkit (VC3D). It needs a C++ compiler, CMake, OpenCV, Boost, libtiff and the libraries listed in that repository's `build_from_src_debian.sh`. No CUDA is involved.

```
sudo apt-get install -y build-essential cmake ninja-build pkg-config libopencv-dev libopencv-contrib-dev \
    libboost-program-options-dev libboost-system-dev libtiff-dev libcgal-dev libceres-dev libblosc-dev \
    libcurl4-openssl-dev libzstd-dev liblz4-dev nlohmann-json3-dev qt6-base-dev zlib1g-dev \
    libgmp-dev libmpfr-dev libopenblas-dev liblapack-dev liblapacke-dev libsuitesparse-dev gfortran
git clone https://github.com/ScrollPrize/volume-cartographer
cmake -S volume-cartographer -B vc-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVC_BUILD_FLATBOI=OFF
cmake --build vc-build --target vc_render_tifxyz
mkdir -p bin && cp "$(find vc-build -name vc_render_tifxyz -type f | head -1)" bin/
```

The measured binary is identified by its md5 in `results/hardware_and_md5.txt`; the source commit it was built from was not recorded, so a different commit may differ by a few grey levels (see Positive control). The `-march=native` option is off by default; the comparison used the default build.

For a comparison that needs no GPU at all (no torch, no CUDA), run `python scripts/cpu_only_check.py`. It renders the windows recorded in `results/results_bench.json` with the CPU renderer and compares them voxel by voxel with the stored surface volume (the `C` against `P` rows), and times both. The GPU rows (`G`) need `scripts/bench_render.py` and a CUDA GPU.

## Reproduce

```
python scripts/s3_pull.py PHerc0139/segments/<segment>/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr <data_dir>/<segment>/sv.zarr --threads 64
python scripts/s3_pull.py PHerc0139/segments/<segment>/mesh/<segment-id>-on-20250728140407-9.362um.tifxyz <data_dir>/<segment>/mesh.tifxyz
python scripts/bench_render.py results/segments.json out/results_bench.json --windows 3     # windows, three paths, identity, flexible calls
python scripts/bench_full.py results/segments.json out/results_full.json                    # cost of pre-rendering a whole canvas
python scripts/cpu_only_check.py                                                            # CPU-only: no GPU, no torch
python scripts/analyse.py out/results_bench.json out/results_full.json                      # the tables above
STUDY_CONFIG=config.json python exclusive_device/bench_render_excl.py results/segments.json out/results_excl.json   # exclusive-device run
python exclusive_device/analyse_excl.py out/results_excl.json out/summary_excl.json
```

The raw results of the runs reported here are in `results/` and `exclusive_device/`. The scripts have not been run end to end on a second machine.
