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

# Validation record: growth_guard (PROTOTYPE)

Source: the fleet repo's `growth_guard.py` at commit `a4b680a` (2026-09-25), with the database policy loader, the
neighbour-mergeable pause, the hub neighbour feed and the area-target settings removed (they need the fleet's
database). The alignment threshold used by the overlap criterion is passed as an argument instead of temporarily
overwriting a module global.

## TESTED (20 tests, synthetic lattices with a known answer; each positive has a paired switch-it-off control)

Clean plane untouched by every criterion; vacuum at the frontier cut, and kept when the criterion is off; an
ENCLOSED vacuum is a hole, not a cut; a hairpin tail is dropped; smooth curvature is neither hairpin nor crumple;
a crumpled frontier is cut by the planarity criterion; overlap with a neighbour is cut but a 2-ring seam is kept for
a stitcher; the NEXT WRAP is not overlap; only the largest piece survives; regrowth into pruned ground is measured;
`guard_tifxyz` writes a new directory and the source bytes are unchanged; `guard_round` carries state and blocks a
frontier that regrows; a lattice below `min_keep_cells` that lost nothing is never called "emptied by the guard"
(a real canary run had ended two young lattices that way); the CT sampler reads a level and treats outside as air.
Mutations that turn tests red: cutting every bad region instead of only frontier-touching ones; `regrown_fraction`
ignoring its tolerance.

## RE-RUN on real lattices (2026-09-25, source-project script `scripts/growguard/sweep.py`, one scroll)

Replay of the guard round by round on grown lattices of one scroll (PHerc0191), n = 40 segments = 8 per stratum, strata
chosen by the source fleet's OWN earlier metrics (clean / watch / vacuum / fold / spaghetti), CT read at pyramid level 1,
raw results md5 `c5c962b1646aa23e42c7ab9134b88615`. "useful ratio" = kept area x material fraction / the fleet's
verified area (1.0 = nothing verified lost). Default policy (`fold_half=1`, `fold_radius_um=300`):

| stratum (n = 8 each) | useful ratio p50 (p10) | kept area p50 | material fraction before -> after | fold-cell fraction before -> after |
|---|---|---|---|---|
| clean | 1.00 (1.00) | 1.00 | 1.00 -> 1.00 | 0.001 -> 0.001 |
| watch | 1.00 (0.96) | 0.87 | 0.82 -> 0.94 | 0.009 -> 0.007 |
| vacuum | 0.94 (0.86) | 0.54 | 0.53 -> 0.89 | 0.026 -> 0.007 |
| fold | 0.88 (0.82) | 0.70 | 1.00 -> 1.00 | 0.074 -> 0.046 |
| spaghetti | 0.93 (0.83) | 0.84 | 1.00 -> 1.00 | 0.050 -> 0.039 |

What this does and does not say, without rounding:
* The guard leaves clean lattices untouched (kept area 1.00 in all 8) and removes about half the area of the vacuum
  stratum while raising its material fraction from 0.53 to 0.89.
* **The material gain is not independent evidence**: the guard's vacuum criterion and the material fraction are the
  same CT test, so it improves by construction. The independent quantity is the "useful ratio", i.e. that little
  previously verified area is lost.
* **The fold criterion barely acts here**: the fold stratum's fold-cell fraction only falls 0.074 -> 0.046, and the
  "no fold criterion" arm gives 0.048. Its value is not shown by this run.
* **The threshold sweep is not resolved at n = 8 per stratum**: the 500 um vs 300 um radius, margin, planarity-angle
  and minimum-bad-cells variants give the same medians to two decimals. The calibration comments in the source
  ("500 um over-prunes", "0.6 -> 0.75 loses no verified area") were NOT reproduced here; the defaults are the source
  project's choice, not a validated optimum.
* One scroll, one host, one day. A canary on one fleet host recorded failures 11 % -> 4 % and grown area per
  attempt-hour 2.3 -> 11.3 mm2 before/after with no concurrent control (RECORDED in the source project's status
  notes, not reproducible from here): treat it as an anecdote.
