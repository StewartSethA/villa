# Validation record: same-sheet estimator

Status legend: **RE-RUN** = reproduced while packaging (2026-09-25); **RECORDED** = transcribed from the source
project's notes, not re-run here; **TESTED** = covered by `tests/`.
Source project: the ScrollPrizeTutorial fleet repo, `src/vesuvius_pipeline/coverage.py` at commit `6b9d0c1`
(2026-09-25; this package's `same_sheet.py` is its estimator, unchanged except that the alignment threshold is a
parameter instead of a module global that a caller had to mutate).

## Estimator behaviour on known geometry (TESTED, 6 tests, synthetic planes)

* the same sheet grown twice (2 voxels apart) is 100 % covered; the adjacent wrap (30 voxels) 0 %; 4.5 voxels
  (just past the band) 0 %; a half-overlapping copy 45-60 %; a sheet crossing at 45 degrees < 10 %;
* mutations that turn the tests red: `SAME_SHEET_VOX` 4 -> 40 (next wrap becomes "covered"); alignment test
  removed (crossing sheet becomes "covered").

## Threshold: does 4 voxels separate a duplicate from the next wrap? (RECORDED)

Per-PAIR median normal distance, 1,500 random bbox-touching segment pairs on each of PHerc0332, PHerc0358 and
PHerc0813 (source script `scripts/coverage/pair_sep.py`, FINDINGS "2026-09-25 same-sheet redundancy census",
n = 3 scrolls x 1,500 pairs):

| population | median separation |
|---|---|
| duplicate-like pairs (> 50 % of points within 4 voxels) | 1.4 / 3.2 / 2.5 voxels (one per scroll) |
| non-overlapping pairs | p10 22-27 voxels, p50 37-40 voxels (about one sheet pitch) |

So 4 voxels sits inside the gap for pairs. **The pooled per-point histogram has no clean valley** (it falls
monotonically from 0 to 60 voxels), so the threshold is a real sensitivity: at 2 / 4 / 8 voxels the count of
not-yet-inked PHerc-scroll segments that would be skipped at 80 % coverage was 3,852 / 7,879 / 10,627. Read
redundancy fractions as +-35 %.

## Census the estimator was used for (RECORDED; not reproducible from this directory)

4,524 inked segments across 23 scrolls, sampled at 3,000 points per segment (750 for the largest scroll):
23.3 % of inked area was already covered by earlier surface (28.5 % on PHerc0211, 4.1 % on PHerc1203 and PHerc0191,
0-6 % on 18 others). Fast mode (coverers pre-filtered to a shared 16-voxel occupancy cell) vs full mode on one
scroll (4,502 segments): max |difference| 0.004 in covered fraction. The census driver needs the source project's
database and is **not** included.

## Not established

* Accuracy against an independent ground truth of "these two segments are the same sheet": none exists here.
  The estimator agrees with a separate overlap study's pair tables (RECORDED, not re-run) and with a second,
  independent duplicate selector on one scroll (PHerc0211: 986 of 1,598 flattened-not-inked segments >= 80 % covered,
  against the other selector's 1,076 of 1,234 duplicates; different denominators, the source notes call this
  agreement). Two methods that share a geometric premise agreeing is weak evidence.
* Scan resolutions other than 7.9-9.4 um voxels, and lattices with tracer steps other than 20.
