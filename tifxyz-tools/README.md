# tifxyz-tools

Surface-level tools for **tifxyz** segments (the `x.tif` / `y.tif` / `z.tif` lattices that VC3D and
`vc_grow_seg_from_seed` write; invalid cells are `<= 0`). They were developed on a fleet that grew tens of
thousands of overlapping segments per scroll, and they answer questions about *surfaces*, not volumes.
Pure numpy / scipy (+ tifffile for I/O). No change to the C++ tracer, no service, no database.

This directory is self-contained and deliberately outside `volume-cartographer/` and `vesuvius/`:
it is Python analysis code that consumes segments those projects produce.

## What is here (this branch)

| module | what it does |
|---|---|
| `tifxyz_tools/same_sheet.py` | **Same-sheet estimator.** Is point p of surface P on the *same sheet* as surface T? T covers p when T's surface passes within 4 voxels of p **along T's normal**, T's normal agrees with P's within 20 degrees (orientation-free) and the lateral offset is within 1.5 median lattice edges. `covered_fraction(P, T)` is the fraction of P held by T. |

This branch also has `growth_guard.py` and `fuse3d.py`, stacked on it; other branches carry `seed_dedup.py` and
`spiral-fitting/umbilicus_checks.py`.

| module | what it does |
|---|---|
| `tifxyz_tools/growth_guard.py` | **Per-cell growth guard (PROTOTYPE).** Cuts the FRONTIER of a grown sheet where it has run into vacuum (needs a CT sampler), a hairpin, a crease/crumple swamp, or a sheet another segment already holds; keeps regions enclosed by good surface (a hole, not a cut); reports why each cell went; `regrown_fraction` says whether a resumed tracer is growing back into pruned ground. Writes a new tifxyz directory, never the source. |
| `tifxyz_tools/fuse3d.py` | **3-D-first fusion (PROTOTYPE).** Lays overlapping lattices of one sheet on a geometric frame (cylinder about an umbilicus polyline, or PCA plane) where one sheet is a height field; takes a consensus per node; a second mode more than `jump_tol` away (the next wrap, or a sheet jump) is DROPPED AND COUNTED, never averaged in. Also `graft` (keep the best lattice intact and grow it over the others; nothing inside it moves), `verify` (anchor a member to the surface prediction), `setcover` (which members can be retired without losing surface), `sheet_groups`, `edge_ridge_runs` (does a lattice edge step from one wrap onto the next?). |


## Run the tests

```bash
cd tifxyz-tools
python -m pytest tests -q          # needs numpy, scipy, tifffile, pytest (+ zarr for the CT sampler test); the fuse3d tests take 16-100 s depending on load
```

Tests use synthetic sheets whose right answer is known, and every positive case has a paired negative one
(the next wrap; a crossing sheet). Each was made to fail by mutation (see `VALIDATION.md`).

## Evidence, and how far to trust it

`VALIDATION.md` states, per claim: the number, n, the date, the repository commit it came from, and whether it
was re-run when packaging. Read it before quoting a figure. The short version: the estimator's separation of
"the same sheet grown twice" from "the next wrap" was measured on 3 scrolls (1,500 random bbox-touching pairs
each), but the 4-voxel threshold is a sensitivity, not a valley in the distribution.

## Conventions followed (villa/AGENTS.md)

Correctness first (no numeric change to anything upstream), determinism (per-segment sampling is seeded from
a hash of the segment name), tests included, performance claims carry a command, input and n.
Portability: pure Python, no OS-specific code. Fleet-specific paths, databases and hosts from the code these
tools came from were removed, not parameterised.
