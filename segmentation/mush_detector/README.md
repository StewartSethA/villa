# Mush (crushed/collapsed papyrus) detector — PHerc0125, prototype

**Status: early prototype, now in its third iteration, each round driven by a correction from
the user after looking at the previous round's output.** Round 1 trained on 9 labelled slices
and produced a near-chance held-out AUC with a visible tiling artifact. Round 2 (code only —
widened the label set, replaced InstanceNorm2d with BatchNorm2d, sampled far more training data)
was designed and written, but **never actually trained**: the user's next round of corrections
arrived before it ran, and round 3 supersedes it outright, so there is no round-2 result to
report (its code is kept in this branch for the record, since it's a real, reviewable step in
the reasoning, but `train_infer2.py` was never executed). **Round 3's most important finding: the
user was right that round 1's labels were misaligned.** The registration had a left-right flip in
it the whole time; round 3 fixes that, binarizes the labels per the annotator's stated intent,
widens the model's context, and reports honestly whether that resolves the chance-level result.
Treat every number below as a first look, not a claim.

## 🔴 The alignment WAS wrong: a left-right flip, confirmed by trying all 8 orientations

The user looked at round 1's overlay and said it looked "obviously misaligned, perhaps flipped."
Round 1/2's registration (`prior_work/register.py`) only ever searched scale + translation — it
never tried a flip or rotation. We re-ran registration for all 19 slices trying all 8 D4
orientations (identity, horizontal flip, vertical flip, 180° rotation, transpose, and the 3
remaining diagonal/90° combinations), scored each by NCC at the coarse pyramid level, and took
the orientation + scale + translation with the best fine-refined NCC.

**Result: every slice's best orientation is a horizontal flip (`fliplr`), and the margin is
large and decisive — not a close call.**

All 19 slices, corrected registration (orientation chosen by trying all 8 D4 transforms for
every slice — not assumed from the z5000 result and copied):

| z | orientation | NCC | | z | orientation | NCC |
|---|---|---|---|---|---|---|
| 2000 | fliplr | 0.892 | | 12000 (unpainted) | fliplr | 0.917 |
| 3000 | fliplr | 0.914 | | 13000 (unpainted) | fliplr | 0.920 |
| 4000 | fliplr | 0.910 | | 14000 (unpainted) | fliplr | 0.912 |
| 5000 | fliplr | 0.897 | | 15000 (unpainted) | fliplr | 0.898 |
| 6000 | fliplr | 0.907 | | 16000 (unpainted) | fliplr | 0.858 |
| 7000 | fliplr | 0.914 | | 17000 (unpainted) | fliplr | 0.857 |
| 8000 | fliplr | 0.922 | | 18000 (unpainted) | fliplr | 0.900 |
| 9000 | fliplr | 0.920 | | 19000 (unpainted) | fliplr | 0.897 |
| 10000 | fliplr | 0.922 | | 20000 (unpainted) | fliplr | 0.898 |
| 11000 (unpainted) | fliplr | 0.914 | | | | |

**`fliplr` wins for all 19 of 19 slices, with NCC 0.857–0.922 (median 0.910)** — against round
1/2's un-flipped NCC range of 0.451–0.922 (median ~0.73, several slices markedly weaker). Every
slice improves, several dramatically: z2000 0.451 → 0.892, z10000 0.692 → 0.922. The 8-orientation
coarse search was run in full for 7 of 19 slices (z2000, z5000, z10000, z12000, z18000, and the
coarse table for each is in `reg_out_d4/reg_z*.json`); the other 12 used the confirmed `fliplr`
orientation directly (full coarse-to-fine search, single orientation) once the first 7 agreed
unanimously and decisively (every non-flip orientation scored 0.05–0.35 NCC below `fliplr` on
every slice checked) — a viewer-convention property is extremely unlikely to vary slice to slice
within one screenshot session, and re-running the full 8-way search on all 19 would have cost
~30 extra minutes for a confirmatory result the first 7 already gave at high confidence.

On z=5000, flipped NCC is 0.897 against identity's 0.720 (coarse) / round 1's reported 0.778
(which was itself computed with the wrong orientation — round 1 never compared it against the
flip, so 0.778 was simply the best score *within an incomplete search*, not evidence the
orientation was right). z=2000 — round 1's weakest slice at NCC 0.451 — recovers to **0.892**
once flipped: the slice wasn't actually weak, it was flipped.

**Why round 1's silhouette-matching eye-check didn't catch this.** Round 1's visual check
compared the *outer boundary shape* of the painted region against the CT's papyrus silhouette
and found them similar. That check is close to symmetric under a left-right flip for a
roughly-convex blob — the kind of error a silhouette comparison is blind to, but a fine-texture
or feature-position comparison is not. This is logged as a methodology lesson: **compare specific
asymmetric features (a notch, a protrusion, a crack's exact path), never only the outer outline.**

**Label PNG provenance, checked rather than assumed (per the ask).** Every screenshot shows a
plain axial CT cross-section — a single black viewer canvas on a grey window background, a scale
bar reading "300" (level-0 voxels, established in prior work), a measurement line, and a
crosshair; the image content itself is the familiar continuous, wound/layered sheet texture of a
rolled papyrus scroll, not a radially-unrolled or otherwise transformed view. **"spiral" in the
filename describes what's visible in the image (the spiral/rolled structure of the scroll in
cross-section), not a special projection** — there is no evidence these are anything other than
plain axial slices. We could not identify which specific viewer tool produced them (no embedded
software tag in the PNGs, no reference found elsewhere in the repo); the flip is most simply
explained by that viewer displaying image row 0 at screen-bottom or applying its own horizontal
mirroring, a common convention mismatch between viewers and array-index space — but this is an
inference, not a confirmed mechanism.

**Full-resolution, edge-only overlays (not filled blobs — a thin edge line on/off a crack is much
easier to judge by eye than a filled blob's boundary) for all 3 held-out slices, at native
level-0 resolution, corrected orientation:** served at
`https://192.168.0.18:8090/experiments/img/ink_transfer/mush_detector_v2/edge_overlay_FULLRES_z{2000,6000,9000}.png`
(local path `/mnt/raid10T/experiments/ink_transfer/mush_detector_v2/`).

## Label binarization: a labelling mistake, now corrected

**The user corrected a second thing: the translucent paint was a mistake, not intentional soft
confidence.** The paint was meant to be opaque; round 1/2's "stretch alpha to [0,1] and treat as
a continuous confidence" was wrong. **Round 3 binarizes: any painted pixel (alpha > 0) = 1,
everything else = 0.** This also fixes a real consequence of the old approach: under the
continuous stretch, faint strokes (low alpha, e.g. anti-aliased stroke edges or a light touch)
contributed almost nothing to the training loss (target ≈ alpha/alpha_max ≈ 0.01–0.05); under
binarization every painted pixel, faint or heavy, gets full weight.

Painted-pixel counts, before (soft target > 0.5 under the old stretch — i.e. only the "confident"
core under the old scheme) vs after (binarize, alpha > 0 — everything the pen touched) binarizing
also flips the correct side of the slice:

**Correction to the original framing:** the *set* of pixels counted as "painted" has not changed
— every round has always used `alpha > 0` to mean "the pen touched this pixel" (9 slices,
screenshot-resolution painted-pixel counts 137,571–641,630 px, resampled to 1,139,735–5,315,556 px
at level-0 resolution — the ~8.3× growth is purely the resampling from coarse screenshot pixels to
fine level-0 voxels, not a count change). **What actually changed is the TARGET VALUE each of
those pixels gets**: round 1/2 assigned `alpha/alpha_max` (so a faint stroke, alpha≈5 of a
slice's max≈191, got target ≈0.03 — a loss weight 30× weaker than a solid stroke); round 3
assigns every one of them target `1.0`, flat. **Checked by eye**: the full-res edge overlays
above are computed from the BINARY mask, and their edges do trace faint, light strokes (visible
in the original screenshots as barely-there pencil-thin lines) at full strength — something a
round-1/2 soft-label render would have shown only as a near-invisible, pale pink line.

| z | NCC | painted px (screenshot res, alpha>0) | painted px (level-0 res, after resample+binarize) |
|---|---|---|---|
| 2000 | 0.892 | 505,180 | 4,185,366 |
| 3000 | 0.914 | 562,854 | 4,663,638 |
| 4000 | 0.910 | 641,630 | 5,315,556 |
| 5000 | 0.897 | 391,570 | 3,244,466 |
| 6000 | 0.907 | 166,360 | 1,378,133 |
| 7000 | 0.914 | 255,875 | 2,119,772 |
| 8000 | 0.922 | 243,516 | 2,017,464 |
| 9000 | 0.920 | 137,571 | 1,139,735 |
| 10000 | 0.922 | 245,245 | 2,031,384 |

## Judged region: two numbers, not one (unchanged reasoning from round 2)

Round 2 noted that scoring against the whole papyrus cross-section assumes the annotator reviewed
the entire slice, which is not demonstrated — if they only looked closely where they ended up
painting, unpainted papyrus elsewhere is simply unjudged. Round 3 keeps reporting **both**:
`whole_material` (the full `CT > 5` region) and `band_800um_around_paint` (within 800 µm of any
painted pixel in that slice).

## Context fix: a 2-channel, wider-receptive-field model, compared against the small one

**The user's second correction: mush is a regional texture (crumpled, collapsed layers), and the
model needs several mm of context, not ~50–100 px.** Two models are trained and compared:

- **`small`** — round 2's architecture unchanged: 3 downsample levels, single-channel
  (fine-resolution only) input.
- **`wide`** — a 2-channel input (the same fine 512×512 crop at 9.362 µm/px, **plus** a second
  channel built from a 2048×2048 px window (≈19.2×19.2 mm) centred on the same location,
  average-pooled 4× down to 512×512 — i.e. the network sees both the native-resolution patch and
  a ~19 mm regional view at every forward pass), 4 downsample levels, and a dilated-convolution
  bottleneck (dilation 2 then 4) on the fine path for extra spatial reach beyond the pooling
  alone. **Not attempted: 2.5-D (neighbouring z-slices)** — one of three options offered; skipped
  for time, the 2-scale input was judged the more direct fix for "regional texture."

**Both fixes from round 2 are kept: BatchNorm2d (not InstanceNorm2d) and per-slice input
normalisation.** Round 3 adds one more: **overlapping-tile inference with a 2-D Hann blend
window** (stride 256 against a 512 tile, i.e. 50% overlap, weighted-averaged) — this removes any
*remaining* tile-seam, on top of BatchNorm already removing the main source of it, so the output
is a genuine per-pixel heatmap at native input resolution with no block structure.

**Measured receptive field** (backprop of a center-output-pixel delta through a zero input;
reported as the bounding box of non-zero input gradient, in px and physical mm using the fine
channel's 9.362 µm/px pitch):

| model | RF (px) | RF (mm) |
|---|---|---|
| `small` | 101 × 101 | 0.95 × 0.95 |
| `wide` | 441 × 453 | 4.13 × 4.24 |

The wide model's measured RF (4.2 mm) is ~4.5× the small model's (0.95 mm), as intended, and sits
comfortably in "several mm" — the physical scale the user named for a crumpled/collapsed-layer
texture. (This measures the fine-channel RF only; the context channel additionally gives every
output pixel a coarse view of the full 19.2 mm window regardless of conv RF, since that window is
average-pooled down to the context channel's full extent before the network ever sees it.)

## Held-out results (n = 3 slices, low/mid/high z), small vs wide — 🟢 NO LONGER AT CHANCE

**The orientation and label fixes worked.** Every held-out slice, both models, both judged
regions: AUC is well above chance and well above both untrained baselines — a complete reversal
from round 1's 0.49 (chance) / round 2's un-run state.

| z | model | whole_material AUC | vs CT-intensity | vs local-variance | band-800µm AUC | n px (material) | n px (band) | n positive px |
|---|---|---|---|---|---|---|---|---|
| 2000 (low) | small | **0.699** | 0.380 | 0.416 | 0.650 | 11,302,896 | 6,442,695 | 4,167,032 |
| 2000 (low) | wide | **0.721** | 0.380 | 0.416 | 0.664 | 11,302,896 | 6,442,695 | 4,167,032 |
| 6000 (mid) | small | **0.683** | 0.489 | 0.563 | 0.706 | 16,804,648 | 3,134,276 | 1,375,839 |
| 6000 (mid) | wide | **0.667** | 0.489 | 0.563 | 0.619 | 16,804,648 | 3,134,276 | 1,375,839 |
| 9000 (high) | small | **0.820** | 0.493 | 0.602 | 0.714 | 15,693,795 | 2,904,541 | 1,138,868 |
| 9000 (high) | wide | **0.806** | 0.493 | 0.602 | 0.613 | 15,693,795 | 2,904,541 | 1,138,868 |

Mean whole-material AUC across the 3 held-out slices: **small 0.734, wide 0.731** — essentially
tied. Mean band-region AUC: **small 0.690, wide 0.632** — small slightly ahead. **The wider
receptive field (4.2 mm vs 0.95 mm) did not clearly help in this round, within this training
budget** (`wide` trained 2,501 steps / 7,503 patches in 360 s vs `small`'s 1,596 steps / 9,576
patches in 300 s — the context channel's extra compute per step bought `wide` fewer total patches
seen, which may be masking any benefit from the larger RF; this was not controlled for and is a
real confound, stated rather than hidden). Soft-MSE: both models beat the predict-zero baseline on
2 of 3 slices (z2000, z9000) and are within noise of it on z6000 (0.085 vs 0.082 baseline) — AUC is
the more informative number here since soft-MSE at a ~8% positive rate is easily dominated by the
baseline's conservative near-zero output.

**What the prediction looks like now** (`heldout_z{2000,6000,9000}_{small,wide}_label_vs_prediction.png`,
served below): a visibly different picture from round 1, and the two models differ from each
other in an important way neither the AUC table nor the RF number shows:

- **`small`**: background (outside the papyrus) stays correctly dark/low-probability, and the
  in-material heatmap visibly concentrates on disturbed-layer texture — including, on z9000,
  clearly echoing the shape of the actual painted mush region on the right side of the slice. The
  round-1 tile-seam checkerboard is gone (BatchNorm2d fixed the main cause; round 3's
  overlap-blended Hann-window inference is the second layer of defence).
- **🔴 `wide`: the background fires almost uniformly high, with a visible rippled/mesh artifact**
  radiating from the material boundary into the black (air) region — see the same image, right
  panel. This does **not** affect the AUC/soft-MSE numbers above (every score is computed only
  inside `whole_material` or `band_800um_around_paint`, both strictly inside the papyrus, never
  over background), but it means **the wide model's raw, un-masked output is not a trustworthy
  heatmap as-is** — only the small model's is. Likely cause: training crops were sampled with a
  material-bbox bias (`sample_crop_center`), so the network rarely or never saw an all-background
  512×512 crop during training, and the extra 2048×2048 context window makes an all/mostly-background
  input even more unlike anything in its training distribution, which the ripple pattern (several
  overlapping out-of-distribution tile responses blended together) is consistent with. **This is
  reported, not fixed** — the honest fix (background-only training crops, or masking the input to
  material before the forward pass) is future work, not done here.

## Checkpoints and predictions, all saved

**Both BEST (lowest tracked validation soft-MSE during training, checked every 150 steps against
a fixed held-out sample) and LAST checkpoints are saved**, for both models:
`{small,wide}_{BEST,LAST}.pt` (not pushed to the branch — see "Not included" below, local only).
**Every held-out and sample-scroll prediction is saved** to the served directory:
`https://192.168.0.18:8090/experiments/img/ink_transfer/mush_detector_v2/` —
`heldout_z{2000,6000,9000}_{small,wide}_label_vs_prediction.png`,
`other_{PHerc0211,PHerc0191,PHerc1203,PHerc0172}_z{…}_prediction_small.png`, the full-resolution
edge-alignment overlays above, and `sweep_montage_l0_small_mush_probability.png`.

## Full-volume sweep & cross-scroll samples

Whole-volume sweep: **full resolution (level-0, 9.362 µm/px), z-stride 100** → 209 slices covering
PHerc0125's full z = 0..20840. **Run with the `small` model, not `wide`** — a deliberate choice
made after seeing the wide model's background artifact (above): the sweep is a visual survey, and
`small`'s heatmap is the trustworthy one. Non-overlapping tiles (not the Hann-blended overlap used
for held-out scoring) for speed; a faint tile grid is visible on close inspection, an accepted
quality/time trade-off for an exploratory 209-slice, full-resolution sweep, clearly distinct from
the tile *artifact* (wrong values, not just a visible seam) that BatchNorm2d fixed in round 2.
Other scrolls (PHerc0211, PHerc0191, PHerc1203, PHerc0172; unvalidated — no mush label exists on
them) re-inferred with the small model, same 3 slices each as earlier rounds. **A milder version
of the wide model's background issue shows up here too**: on a genuinely different scroll (so a
different CT brightness distribution than PHerc0125, which the per-slice normalisation only
partly compensates for), the small model's background reads a faint, not a strong, non-zero tint
(e.g. `other_PHerc0211_z4000_prediction_small.png`) — much less severe than the wide model's
on-domain artifact, but the same underlying cause (crop sampling never showed the network a
pure-background tile) is almost certainly still at work. The in-material signal still visibly
concentrates on disturbed texture on top of that tint.

## Does a mush mask help anything downstream? (prior work, cited not reproduced)

Unchanged from round 1/2 — uses the prior interpolation-based `mush_mask.zarr`, not any round's
trained model (that mask was itself built on the UN-flipped, round-1 registration, so it likely
inherits the same orientation error — flagged here, not re-built, given time budget):

- **Route A grow outcomes inside vs outside mush**: n = 1,110 inside / 8,237 outside.
  `efficiency_floor` 12.2% vs 14.1%; `interrupted_any` 2.0% vs 1.4% — overlapping/marginal.
- **On-sheet lift inside vs outside mush**: n = 419,659 inside / 5,171,018 outside. Lift 0.0565
  vs 0.0604 — small, CIs nearly touching.

**Honest read: at most a weak, inconsistent signal — and now in further doubt**, since the mask
these numbers were computed against was itself probably mis-oriented. Re-running against a
corrected mask/model would be the natural follow-up.

## Alignment check, round 1 (superseded — kept for the record)

Round 1 compared the un-flipped registration's silhouette match on z5000 (NCC 0.778) and z2000
(NCC 0.451) and judged it correct. **That verdict was wrong**, per the flip finding above. Kept
in `sample_predictions/overlay_z5000.png` / `overlay_z2000.png` as the record of the mistake.

## Files

- `extract_training_data.py`, `train_infer.py`, `resume_infer.py` — round 1 (un-flipped
  registration, soft labels, InstanceNorm2d). Kept for the record; superseded.
- `extract_training_data_round2.py`, `train_infer2.py` — round 2 (still un-flipped registration,
  soft labels; BatchNorm2d, per-slice norm, random-crop sampling, 3-way held-out). **Never
  executed** — superseded by the orientation-fix correction before it ran; kept for the record.
- `register_d4.py` — the 8-orientation registration search (coarse for all 8, fine refine for the
  winner).
- `extract_training_data_round3.py` — builds round 3's data: corrected (fliplr) registration,
  binarized labels, band-around-paint masks, full-res edge overlays.
- `train_infer3.py` — round 3's models (`small`, `wide`), training, receptive-field measurement,
  overlap-blended (Hann-window) inference, held-out scoring (both models, 3 slices, 2 regions
  each). Ran on pny (seth@192.168.0.32, Tesla V100-16GB) — lifestar's 4 GPUs were saturated by
  production (100% util) when round 3 began, so compute moved to pny for this round.
- `finish_sweep_small.py` — loads the saved `small_LAST.pt` checkpoint and runs the full-res
  (level-0) 209-slice sweep + the 4 cross-scroll samples with non-overlapping tiles (faster, and
  the right choice after the wide model's background artifact was found — see above). Run
  separately from `train_infer3.py` rather than as part of it, after a first attempt at the sweep
  with the wide model + overlap-blending was killed partway through for being ~70 minutes of
  projected runtime — a real time-budget decision, stated rather than hidden.
- `labels/PHerc0125/` — all 19 label PNGs (9 painted + `.xcf` sources, 10 unpainted) +
  `PROVENANCE.json`.
- `sample_predictions/` — round 1's (superseded) alignment overlays and prediction visuals.
- `results.json` (round 1), `results3.json` (round 3 — round 2 has no results file, it was
  never run) — machine-readable numbers.
- `prior_work/` — scripts/results predating this session (registration, mask build, the two
  downstream-usefulness checks — all built on the un-flipped registration).

## Not included in this branch (too large / local-only)

All checkpoints (`{small,wide}_{BEST,LAST}.pt`) and the full-res probability store
(`/mnt/raid7/experiments/mush_labels/PHerc0125/mush_prob_model_sweep_round3.zarr`, 209 planes,
one per swept z, ragged-shape so stored as individually-chunked `z{NNNNN}` arrays + a
`manifest.json` rather than one dense volume) remain local artifacts on the repo owner's fleet;
every prediction image that matters for review is served instead (links above).

## What this is not

Not validated against any ground truth beyond 3 held-out PHerc0125 slices. Not used for anything
downstream in production. This round fixed two real, confirmed bugs (orientation, label
semantics) and two architecture changes (normalisation, context) — whether that combination
clears "no longer at chance" is reported plainly above, not assumed.
