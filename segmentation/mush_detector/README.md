# Mush (crushed/collapsed papyrus) detector — PHerc0125, prototype

**Status: early, lightly-validated prototype offered as a possible starting point, not a
finished method.** It was built and trained in a single short session on a small labelled
set and has not been reviewed by anyone else. Treat every number below as a first look, not
a claim.

## What this is

A small 2-D U-Net that reads one axial CT cross-section of PHerc0125 and predicts,
per pixel, a soft "how mushy/crushed does this papyrus look" score in `[0, 1]`. It is trained
on 9 hand-painted axial slices (z = 2000..10000, step 1000, level-0 voxel grid, 9.362 µm/px)
where the repo owner painted blobs over regions where the sheet stack is crushed/collapsed and
indistinct, plus thin strokes over cracks/tears. The paint layer is only **partly opaque** —
alpha ranges roughly 1–222 of 255 across the painted pixels — which the authors of these labels
intended as a soft confidence, not a binary mask.

## Label handling: never thresholded

Per‑slice, the painted alpha channel is stretched to `[0, 1]` by its own observed maximum
(`y = clip(alpha / alpha_max_nonzero, 0, 1)`) and used directly as a continuous regression
target — no binarisation anywhere in the pipeline. The output head is a **straight-through
clamp** (`clamp(x,0,1)` forward, gradient 1 everywhere, including outside `[0,1]`, so an
over- or under-shot prediction is still pulled back by the loss instead of being stranded at
a dead gradient), and the loss is a **masked soft MSE**.

**Mask / judged-region choice, stated plainly because it differs from this project's ink-label
rule:** the mask here is the whole papyrus cross-section (`CT > 5`, the repo's own material
threshold), not "painted pixels only". The reasoning: the annotator looked at and judged the
*entire* visible cross-section when painting a slice (comprehensive per-slice review), unlike
the sparse ink strokes elsewhere in this project where most of a segment's surface is never
reviewed at all. If that assumption is wrong for some slices — the annotator missed a mushy
patch rather than judging it clean — this training signal would wrongly suppress it. This is
the single biggest assumption behind the numbers below; a reviewer who disagrees with it should
retrain with `w = (alpha > 0)` instead (one line change, see `train_infer.py`).

## Alignment check (done before training)

The label PNGs are screenshots from a GIMP viewer session, not pixel-registered to the CT by
construction. `prior_work/register.py` (earlier work by the same repo owner, included here for
completeness; not written in this session) fits a per-slice affine (scale +
translation, NCC-based coarse-to-fine search over `/mnt/raid7/scroll_volume_cache/PHerc0125.zarr`)
with fit quality NCC p10/p50/p90 = 0.692 / 0.727 / 0.778 over the 9 slices (weakest: z2000 at
0.451). **We overlaid the registered label on the full-res CT and looked, for the best (z5000,
NCC 0.778) and worst (z2000, NCC 0.451) fits** (`sample_predictions/overlay_z5000.png`,
`overlay_z2000.png`): in both, the outer silhouette of the painted region tracks the papyrus
cross-section's own boundary closely, and the painted blobs sit on visually disturbed/crumpled
texture rather than on the clean, concentric, well-ordered sheet layers elsewhere in the same
slice. **Verdict: the alignment looks correct, including at the weakest-NCC slice** — we did not
re-fit it. This is one person's visual read on 2 of 9 slices, not a quantitative check.

## Data

- Training: 8 of the 9 labelled slices (all except the held-out one below), native level-0
  resolution (9.362 µm/px). From each slice, 10 patches of 512×512 px are sampled — 5 centred
  near painted pixels (jittered), 5 centred on material (`CT>5`) uniformly at random — giving
  80 training patches total.
- Held-out validation: **z=6000** (one full slice, never seen in training), chosen because it
  is a mid-z slice with good registration (NCC 0.750) — picked before looking at any result.
- Whole-volume sweep: PHerc0125 level-0 z runs 0..20840. We inferred at **level-1 CT (18.72
  µm/px, 2× coarser than training resolution — a declared resolution mismatch, done to fit the
  time budget)**, full XY frame, **stride 260 level-0-z** → 81 slices covering the whole volume,
  tiled in 512×512 windows.
- Other scrolls (unvalidated — see below): PHerc0211, PHerc0191, PHerc1203, PHerc0172, 3 slices
  each, level-1.
- Comparison baseline: the existing `mush_mask.zarr` (shape-based SDF interpolation between the
  9 labelled slices, prior work, not retrained here) is the "linearly interpolated volume"
  comparison the task asked for; built by `prior_work/build_mask.py`, not reproduced here (its
  output is a local artifact, see the manifest in `prior_work/PHerc0125_mask_manifest.json`).

## Held-out result (n = 1 slice, z=6000) — NEGATIVE: the detector did not generalize

**🔴 This prototype did not learn a useful mush detector in the time given.** On the held-out
slice (z=6000, 16,804,648 judged pixels inside the papyrus cross-section, 990,702 of them "paint
present" at the 0.1 stretched-label threshold):

| | model | CT intensity (untrained) | 7×7 local variance (untrained) |
|---|---|---|---|
| AUC @ threshold 0.1 | **0.4915** | 0.4758 | 0.4683 |

| | model | predict-zero baseline |
|---|---|---|
| soft MSE (judged region) | **0.0491** | 0.0217 |

AUC is at chance (0.49–0.50 across all three arms), and the trained model's soft-MSE is **worse**
than trivially predicting zero everywhere — the single clearest failure signal in this report.
Training: 1,450 steps / 421 s / 80 patches from 8 slices, throughput ~7.2 MVox/s on one RTX 4060
Ti. The AUC threshold (0.1 of the stretched label) is used only to get one comparable number for
this table; **the model itself was never trained against a threshold.**

**What the prediction actually looks like, and why it likely failed**
(`sample_predictions/heldout_z6000_label_vs_prediction.png`, label left / prediction right;
`sample_predictions/other_PHerc0211_z4000_prediction.png`): the model fires broadly on the
papyrus/background **boundary** rather than on the mush texture inside the sheet stack, and shows
a visible 512-px checkerboard — a tiling artifact from `InstanceNorm2d` normalising each 512×512
inference tile independently, so adjacent tiles see different local statistics. Combined with the
small training set (80 patches, 8 slices, no held-out validation during training to catch this),
the most likely explanation is **severe underfitting of the real texture signal and overfitting
to patch-edge / material-boundary cues** — not a sign that mush is undetectable, just that this
quick attempt did not detect it. A slower, better-resourced attempt should: batch-normalize across
the whole image rather than per-tile (or use GroupNorm/no norm with overlap-blended tiling), use
far more than 80 training patches, and validate during training rather than only at the end.

## Full-volume sweep & cross-scroll samples

`sample_predictions/` holds PNG overlays (red = predicted mush probability) for the held-out
slice, a montage of the sparse full-z sweep, and 3 sample slices each from PHerc0211, PHerc0191,
PHerc1203 and PHerc0172. **The cross-scroll predictions are completely unvalidated** — there is
no mush label on any of those four scrolls, so treat them as "what the model says", nothing more.

## Does a mush mask help anything downstream? (prior work, cited not reproduced)

Two existing analyses (`prior_work/routeA_guard_vs_mush.py`, `prior_work/mush_onsheet.py`, run
before this session, against the interpolation-based `mush_mask.zarr` — **not** this session's
trained model) give a first, honest, weak-signal read:

- **Route A grow outcomes inside vs outside mush** (`prior_work/PHerc0125_routeA_guard_vs_mush.json`):
  n = 1,110 segments inside the mush-labelled z-range and spatially inside mush, n = 8,237 outside.
  `guard_*` stop reasons are a true zero on both sides (merged segments carry no seed position and
  can't be classified). The better-powered proxies go the **opposite** way from what "mush is bad
  for growth" would predict: `efficiency_floor` rate is 12.2% [10.4%, 14.2%] inside vs. 14.1%
  [13.4%, 14.9%] outside; `interrupted_any` is 2.0% [1.3%, 3.0%] inside vs. 1.4% [1.2%, 1.7%]
  outside — overlapping or marginal, not a clear effect either direction.
- **On-sheet lift inside vs outside mush**, one Route B PHerc0125 fit whose z-range overlaps the
  labelled z-range (`prior_work/PHerc0125_onsheet_mush_vs_outside.json`): n = 419,659 points
  inside / 5,171,018 outside, 103 vs 115 windings. Lift is 0.0565 [0.0545, 0.0583] inside vs.
  0.0604 [0.0599, 0.0609] outside — a small, barely-separated difference (mush slightly *lower*
  lift, as hypothesized, but the gap is ~0.004 and the CIs nearly touch).

**Honest read: using the existing mush mask, there is at most a weak, inconsistent signal that
mush hurts Route A or Route B outcomes, well short of a clear effect** — and this used the prior
interpolation-based mask, not this session's (near-chance) model, so it says nothing about
whether *this* detector would help. Re-running these two scripts against this session's
`mush_prob_model_sweep.zarr` once that model is actually working would be the natural follow-up.

## Files

- `extract_training_data.py` — reads the registered label PNGs + the CT zarr, builds the
  training/validation/sweep bundle. Ran on the data-owning host (light I/O only, no GPU).
- `train_infer.py` — the model, training loop, held-out scoring, and sweep/cross-scroll
  inference, end to end. Ran on a free lifestar GPU (RTX 4060 Ti). (Has one fixed bug:
  `numpy.trapz` was removed in numpy>=2.0 and is replaced here with a manual trapezoidal rule.)
- `resume_infer.py` — the same held-out/sweep/cross-scroll inference, but loading the saved
  checkpoint instead of retraining; this is literally what produced the numbers below, after the
  `numpy.trapz` bug surfaced mid-run on the first pass and training did not need repeating.
- `labels/PHerc0125/` — all 9 hand-painted label PNGs (+ `125-z11000spiral.png`, unpainted) and
  their original `.xcf` GIMP sources (all < 10 MB), plus `PROVENANCE.json` (author, export
  method, per-file md5).
- `sample_predictions/` — the alignment-check overlays and the prediction visuals described
  above.
- `results.json` — the held-out numbers and run metadata (seconds, steps, MVox/s), machine-
  readable.
- `prior_work/` — earlier scripts and results by the same repo owner, predating this session:
  `register.py`/`build_mask.py`/`gimp_export_layers.py` (built the labels and the alignment this
  session checked), `routeA_guard_vs_mush.py`/`mush_onsheet.py` (the downstream-usefulness test
  cited above) and their 3 result JSONs.

## Not included in this branch (too large / local-only)

- The trained checkpoint (`mush_detector_small_unet.pt`, 1.9 MB) and the sparse full-volume
  probability OME-Zarr it produced (`/mnt/raid7/experiments/mush_labels/PHerc0125/
  mush_prob_model_sweep.zarr`, level-1 CT 18.72 µm/px, z-stride 260 level-0-z units = 81 planes,
  **not every z** — a declared time-budget compromise) are local artifacts on the repo owner's
  fleet, not pushed here. Given the held-out result above, treat the checkpoint as a starting
  point to debug, not a model to deploy.

## What this is not

Not validated against any ground truth beyond the one held-out PHerc0125 slice. Not tuned —
one architecture, one learning rate, one patch size, chosen quickly to fit the stated time
budget (~5–10 min train, ~1 min validate, ~5–10 min visuals). Not used for anything downstream
in production. Offered only as a possible starting point if useful.
