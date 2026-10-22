# What `vc_render_tifxyz` actually does (read from source, not inferred)

Binary in production: `/tmp/appimage_extracted_47cbd7527c064c48bf4f28d26f26d1ad/usr/bin/vc_render_tifxyz`
Source: `volume-cartographer/` (the VC3D toolkit)

## Which code path the production command takes

    vc_render_tifxyz -v <zarr> --scale 1 -g 0 -s flat_final -n 65 --tif-output render_final

`--tif-output` **without** `--zarr-output` selects the **band** path, not the tile path:

- `apps/src/vc_render_tifxyz.cpp:1788` -> `renderBands<uint8_t>()` (definition at line 594)
- `renderTiles()` (line 712, OMP-parallel over output zarr chunk columns) is only used when
  `--zarr-output` is given.

Defaults that matter: `--scale-segmentation 1.0`, `--slice-step 1.0`, `-g 0` -> `ds_scale = 1`,
no affine, no accum offsets, no composite, no rotation.
=> `render_scale = tgt_scale * scale_seg * sA * ds_scale = 1.0`.

## The pipeline per band (bandH = 128 output rows, full output width)

1. `computeCanvasOrigin(fullSize)` -> `u0 = -0.5*(W-1)`, `v0 = -0.5*(H-1) + y0`
2. `genTile()` -> `QuadSurface::gen()` (`core/src/QuadSurface.cpp:1274`)
3. `prepareBaseAndDirs()` (line 276)
4. `readMultiSlice()` -> `readMultiSliceImpl()` (`core/src/Slicing.cpp:698`)
5. `processRawSlices()` (identity when no accum offsets) -> `writeTifBand()`

### `QuadSurface::gen()` — the surface is *upsampled*, not stored at render resolution

The tifxyz grid is at `scale = 0.05` (meta.json), i.e. 1 grid vertex per 20 output pixels.
For a large segment the grid is 356x452 and the render is 9040x7120.

- `ul = internal_loc(offset/scale + _center, ptr, _scale)`, `_center = (cols/2/sx, rows/2/sy, 0)`
- `sx_warp = _scale[0]/render_scale = 0.05`, `ox = ul[0] - 4*sx_warp` (4 px halo)
- coords: `warpBilinearReplicateVec3f` — **bilinear**, replicate border, `fx = fox + dx*fsx`,
  clamp to `[0, cols-1]`, `x0 = int(fx)`, `x1 = min(x0+1, cols-1)`
- validity: `warpNearestConstU8` — **nearest** (`int(lround(fx))`), border 0
- normals: `_normalCache` built once on the *source grid* by `grid_normal_int()`
  (`core/src/Geometry.cpp:29`, central differences + normalised cross product; border rows/cols
  and any cell with a `-1` neighbour are NaN), then warped **nearest** with NaN border
- crop the 4 px halo, then the invalidation pass: `valid==0 -> coords = NaN, normals = NaN`

Consequence: normals are *nearest-neighbour* replicated across each 20x20 output block while
coords are bilinear. That is the single most surprising thing about the geometry stage.

### `prepareBaseAndDirs()`

    base = pts * scale_seg;  dirs = nrm;  [flip] [affine];  normalizeNormals(dirs);  base *= ds_scale;

`normalizeNormals` skips NaN and skips `L2 <= 0`.

### `readMultiSliceImpl()` — the actual inner loop

offsets = `buildOffsetList(65, 1.0, {})` = `-32 .. +32` in level-g voxels.

    #pragma omp parallel
      ChunkSampler<T> s(cache, level);          // 16384-slot direct-mapped hot cache, per thread
      #pragma omp for schedule(dynamic, 16)
      for y in 0..127:                          // <-- band height is 128
        for x in 0..W-1:
          for i in 0..64:
            v = base[y][x] + dirs[y][x] * offsets[i]
            out[i][y][x] = s.inBounds(v) ? clamp_trunc(s.sampleTrilinear(v)) : 0

`ChunkSampler` (`core/src/Slicing.cpp:87`):
- `inBounds(vz,vy,vx)` = `vz>=0 && vy>=0 && vx>=0 && vz<SZ && vy<SY && vx<SX`.
  NaN fails every comparison, so invalid cells fall out here -> 0. This is the sentinel path.
- `sampleTrilinear` = 8 x `sampleInt` + 7 `std::fma`. `sampleInt` bounds-checks **each corner
  independently** and returns 0 out of range, so the volume boundary fades to 0 rather than
  clamping. A missing / all-fill chunk also returns 0.
- uint8 output is **truncated** (`T(v)`), uint16 is **rounded** (`T(v+0.5f)`).

## Two things not in `gen()` that a reimplementation must not miss

1. **`load_quad_from_tifxyz_impl` invalidates by z<=0** (`core/src/QuadSurface.cpp:2466-2470`):
   `if (points(j,i)[2] <= 0.f) points(j,i) = (-1,-1,-1);`
   Undocumented, applies to *every* tifxyz load. Never fires on real PHerc segments (z >= 700)
   but silently changes results for any surface near the z origin.
2. **`main()` replaces the sentinel by testing x only** (`apps/src/vc_render_tifxyz.cpp:1514`):
   `if (raw_points(j,i)[0] == -1) raw_points(j,i) = {NaN,NaN,NaN};`
   whereas `validMask()` tests all three components. After (1) the two agree in practice.

## Parallelism structure (measured consequence)

The band loop is `schedule(dynamic, 16)` over **128** rows => at most **8** OMP work units per
band. `vc_render_tifxyz` starts 119 OMP threads (there is no thread-count option in this binary; only `OMP_NUM_THREADS` controls it) but the band path
cannot use more than 8 of them for sampling; every one of the 119 still constructs and destroys
a 16384-slot `ChunkSampler` (~0.6 MB, with `shared_ptr` members) at every band.
Measured %CPU on this box: **274-357%**, never close to 119 cores.

## Why the volume is cheap to read

`the CT volume's level-0 .zarray` has `"compressor": null` — the chunks
are **raw uncompressed 128^3 uint8 files** (2 MiB each) under `0/<cz>/<cy>/<cx>`. There is no
decompression cost at all; chunk loading is pure file I/O, and a GPU implementation can read the
files directly and DMA them without any codec.
