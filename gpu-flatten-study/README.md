# GPU flatten against CPU flatten on 17 labelled PHerc0139 segments

Date: 2026-10-22. Scroll: PHerc0139 only. No default is changed.

## Question

A "flatten" turns a 3-D surface mesh into a regular 2-D grid (a tifxyz folder: x.tif, y.tif, z.tif, meta.json; one grid vertex per 20 output pixels). Two flatteners are in use:

| name | where it runs | method |
|---|---|---|
| CPU flatten | `vc_flatten` (Volume Cartographer) | ABF++ angle-based flattening, 10 iterations, sparse direct solver on the CPU |
| GPU flatten | `villa/lasagna/fit.py` with `configs/flatten_fast_nofilter.json` | gradient descent (Adam, 4500 steps) on the grid coordinates on the GPU; its loss asks every grid edge to be 20 px long (`flatten_edge_step: 20.0`) and the grid to keep its orientation |

This study runs both on the same meshes and measures time, the geometric distortion of each result, and how well the two grids agree. It stops before the render and ink-map comparison, see "Gate".

## Data

The 17 segments are the PHerc0139 segments of the dense-label set that are not in our 97-segment training split. Input for both flatteners: the open-data mesh of each segment (`mesh/<id>-on-20250728140407-9.362um.tifxyz`, 100 000 to 130 000 valid vertices). List: `results/segments.json` (17 of its 22 names, in the order of `results_flatten.json`).

## Measures

- **Stretch** of a result: length of a grid edge in 3-D divided by 20 (1 = every grid step is exactly 20 voxels), rows and columns separately, per valid quad.
- **Shear**: absolute difference between the angle of the two grid directions and 90 degrees, in degrees.
- **Agreement** of two grids A and B: each vertex of A is matched to the nearest vertex of B in 3-D (all matches within 30 voxels; median 3-D distance 8 to 10 voxels), the grid indices of the matches are related by the best 2-D similarity (rotation or reflection, scale, shift), and the residual of that fit, in output pixels (index x 20), is reported.

**Positive control.** The agreement of a grid with itself gives a residual of 1e-12 px, and the open-data grids give stretch 0.999 (p10 0.997, p90 1.000) and shear 0.46 degrees, as expected for grids made with 20 px spacing.

**Bias to know about.** Stretch is the quantity the GPU flatten optimises directly (edge length 20), so it is not an independent test of that flatten; shear (angle) is the quantity ABF++ optimises. Both are reported for that reason.

## Results (n = 17 segments; p10 / p50 / p90 over segments of the per-segment median)

| | open-data grid | CPU flatten (ABF++) | GPU flatten (lasagna) |
|---|---|---|---|
| wall seconds per segment | | 47 / 75 / 173 | 67 / 91 / 118 |
| stretch, rows | 0.997 / 0.999 / 1.000 | 0.774 / 0.987 / 1.000 | 0.998 / 1.000 / 1.000 |
| stretch, columns | 0.997 / 0.999 / 1.000 | 0.774 / 0.984 / 0.999 | 0.999 / 1.000 / 1.000 |
| spread of stretch inside a segment (p90 / p10 of its stretch) | 1.03 / 1.03 / 1.05 | 1.05 / 1.34 / 5.86 | 1.02 / 1.03 / 1.04 |
| shear, degrees | 0.42 / 0.46 / 0.70 | 0.08 / 0.09 / 0.19 | 0.38 / 0.43 / 0.60 |
| valid vertices kept, against the open-data mesh | | 0.971 / 0.989 / 0.992 | 0.983 / 0.988 / 0.990 |

Agreement (median residual in output pixels over the matched vertices of a segment, then p10 / p50 / p90 over segments):

| pair | median residual px | 90th-percentile residual px |
|---|---|---|
| CPU flatten against open-data grid | 19.6 / 192 / 1080 | 39 / 368 / 2120 |
| GPU flatten against open-data grid | 6.3 / 52 / 131 | 13 / 83 / 200 |
| GPU flatten against CPU flatten | 18.7 / 188 / 938 | 44 / 385 / 1810 |

Reading: the GPU flatten stays within a few tens of pixels of the grid it starts from and keeps every grid step at 20 voxels. ABF++ is more conformal (smaller shear) but its scale varies more inside one segment (stretch p90 / p10: median 1.34 over segments, and 5.9 at the 90th percentile of segments), and its grid differs from the open-data grid by hundreds of pixels. The GPU flatten started from the input grid, so its closeness to the open-data grid is partly by construction.

Time: the GPU flatten was not faster than the CPU flatten here (median 91 s against 75 s). The GPU was a shared V100 with other jobs running and the CPU job ran next to a load average of 13 to 16; one run per segment.

## Gate (CLAUDE.md, D30: a bad flatten quietly makes ink unreadable)

| step | status |
|---|---|
| 1. solve-share measurement | Not reproduced by me. The repository record (docs/BACKLOG.md B3, FINDINGS section 43) gives 6.9x on the linear solve of one real 49 928-row system and 39 to 59 % of an ABF++ iteration spent in the factorisation. I built the GPU BiCGSTAB bench on a V100 host and ran it, but that bench builds its own diagonally dominant lattice (4 iterations), the proxy the record calls invalid, so it says nothing about the real system. |
| 2. typedef and build of the GPU ABF++ solver | Not done. |
| 3. UV comparison on real segments, carried to render and ink map | UV comparison done (above). Render comparison and ink map **not done**. |
| 4. fallback and announcement | In the pipeline code (`stages/render.py: flatten()`): a missing lasagna environment fails the stage with a message, no silent CPU fallback. Not tested here. |

The gate is not passed. This study changes no default and recommends none; "which flatten is better" can only be decided downstream (flatten, render, ink legibility on the same segments), which was not run (docs/BACKLOG.md B6).

## Caveats

One scroll, one scan, 17 meshes whose open-data grid is itself a flatten result; no ground-truth parameterisation exists. Single runs on a shared machine. The two metrics favour different flatteners (see "Bias"). The ABF++ run used the default 10 iterations and default scale option of `vc_flatten`; other settings were not tried.

## Reproduce

```
cp config.example.json config.json     # edit; paths are relative to the config file unless absolute
python scripts/flatten_study.py results/segments.json out/results_flatten.json --gpu 0
```

`config.json` keys: `data_dir` (folder with `<segment>/mesh.tifxyz`; see the dynamic-render study for how the meshes were fetched), `flat_dir` (output folder), `cpu_flattener` (the `vc_flatten` binary from the Volume Cartographer toolkit, built with CMake target `vc_flatten`, CPU only, see the CPU-only build in the dynamic-render study), `lasagna_dir` (the `lasagna` folder of this repository with its Python environment) and `lasagna_python` (the interpreter to run `fit.py` with). `results/segments.json` is `{"segments": [ids]}` (a plain list or a text file of ids also works). The CPU flattener needs no GPU; the GPU flattener needs a CUDA GPU. The script has not been run end to end on a second machine.
