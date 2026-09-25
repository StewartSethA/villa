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

# Validation record: seed_dedup (PROTOTYPE)

Source: the fleet repo's `seed_dedup.py` (source commit `f127c30`, 2026-09-25) and its validation script
`scripts/growguard/validate_seed.py`. Ported with the docstring de-fleeted; logic unchanged.

## TESTED (7 tests, synthetic plane): a seed on an existing sheet is refused and the coverer is named; the NEXT WRAP
(30 voxels along the normal) is a legitimate seed; a seed far along the sheet, or 200 voxels beyond its rim, is novel;
only EARLIER segments cover a seed (`before`); a segment never covers its own seed (`own`); points with an undefined
normal never cover. Mutations that turn tests red: dropping the along-normal condition (the next-wrap test fails);
ignoring the normal-validity mask.

## RE-RUN on real lattices (2026-09-25; history as ground truth)

Question: does the seed-only check predict that a grown segment DUPLICATES earlier ones? For each test segment the
seed is the lattice cell with the smallest tracer generation (where the tracer started); truth = fraction of the
segment's lattice on the SAME SHEET (4 voxels, 20 degrees, `same_sheet`) as segments CREATED EARLIER, measured on the
whole lattice independently of the seed point: dup >= 0.6, partial 0.3-0.6, new < 0.3. Verdict = `SurfaceIndex.check`
at the seed with `before` = the segment's creation time. **Both truth and verdict use the same same-sheet premise**, so
this shows the seed point predicts the whole-lattice verdict, not that "same sheet" is right.

Default policy `SeedPolicy(same_sheet_vox=3, lateral_vox=50)`:

| scroll | test segments (random, seed 1) | dup / partial / new | recall on dup | refuses NEW | refuses partial | precision | raw results md5 |
|---|---|---|---|---|---|---|---|
| PHerc0211 | n = 30 | 22 / 5 / 3 | 19/22 = 0.86 | 0/3 | 4/5 | 0.83 | `c491cd49...` |
| PHerc0191 | n = 60 | 2 / 25 / 33 | 0/2 (unmeasurable) | 2/33 = 6 % | 5/25 | 0.00 (no true duplicates to find) | `6bff6e65...` |

The full 3x3 sweep of (same_sheet_vox, lateral_vox) is in the raw files: recall on PHerc0211 dups runs 0.68 to 1.00 and
the cost on new seeds stays at 0/3 (PHerc0211) and 2-9 of 33 (PHerc0191, rising with lateral_vox: 6 % at 50 voxels).
Read the limits:
* **The two scrolls barely overlap in what they can show.** PHerc0211 is 73 % duplicates (few "new" to refuse, n = 3),
  PHerc0191 is 3 % duplicates (recall unmeasurable, n = 2). The source project's own figure, pooled over 359 segments
  on 3 scrolls, was recall 0.76 on later duplicates, 8 % of NEW seeds refused, precision 0.79; the raw file for that
  run was not located, so it is RECORDED and not reproduced. The runs above are consistent with it, not a
  confirmation.
* "refuses partial" is high (4/5, 5/25): a seed lying on a held sheet where the whole lattice is only 30-60 %
  covered is refused. That is the intended behaviour (extend the covering segment) but it is a cost if the partial
  segment would have grown new surface.
