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

Other methods built on it live on stacked branches of this repository (see the branch list in the branch
description): `seed-dedup`, `growth-guard`, `fuse3d`; and, independent of this directory,
`spiral-fitting/umbilicus_checks.py`.

## Run the tests

```bash
cd tifxyz-tools
python -m pytest tests -q          # needs numpy, scipy, tifffile, pytest
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
