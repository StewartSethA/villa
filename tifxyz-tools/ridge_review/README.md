# ridge_review: human validation of the growth guard's `ridge_hit` check

`ridge_hit` is the growth guard's largest pruner:
- 22.6 % of grown cells in the A/B's policy D;
- ~14 % of all grown cells in production under D.

An automated CT test suggests the cells it removes are not on a sheet, but that test is weak
(`../GUARDED_GROW.md`, "Is ridge_hit over-pruning?"). This package lets a **person** decide, blind,
for a stratified sample of real `ridge_hit` decisions. It then turns the answers into false-removal
and miss rates, reweighted to the production cut.

Everything needed to review, rescore and change the method is in this directory. No server, no
fleet, no database.

```
ridge_review/
  site/            the review page (index.html, review.js), samples.js (the blind manifest),
                   img/<sample>/*.png (the rendered views), key.js (answers; loaded only after export)
  data/samples/    <sample>.npz, the raw inputs per sample (see "Inputs")
  data/built.json  per-sample metadata (no answers)
  data/key.json    the answers: removed | kept, severity stratum, the round's statistics
  data/prevalence.json   production counts per scroll x stratum (for reweighting)
  data/selection.json    which rounds were drawn, and why; data/shortfall.json: strata too small to fill
  build_inputs.py  builds data/samples from checkpoints + CT + prediction (run where the data live)
  render_pages.py  renders site/ from data/
  score.py         labels -> rates with CIs, reweighted; compares with the automated CT check
```

Size: data 13 MB, site 14 MB, **27 MB in total**, committed directly. That is below the 50–100 MB
target, so neither LFS nor a release asset is needed.

## How to review (start here)

1. Open the page:
   - GitHub Pages or any static host: publish this repository and open `tifxyz-tools/ridge_review/site/index.html`.
   - From a clone: open `site/index.html` in a browser (works from `file://`), or run
     `python -m http.server -d tifxyz-tools/ridge_review/site 8000`.
   - On the source project's hub: `/static/ridge_review/index.html`.
2. Enter your initials.
3. For each sample, label the yellow-ringed cell: **1** sheet, **2** not sheet, **3** unsure.
   - Optionally set a severity (minor / moderate / severe: how bad a wrong decision here would be for
     reading), and write a note.
   - **p** toggles the surface-prediction overlay. It is hidden by default because it is exactly what
     `ridge_hit` reads. Viewing it is allowed and is recorded per sample.
4. When finished, **Export labels (JSON)**. Only then does **Reveal answers** show your labels
   against the guard's decisions. Progress is kept in the browser in between.
5. Score: `python score.py ridge_review_labels_<you>_<date>.json` (several files = several reviewers:
   majority vote per sample, plus pairwise agreement and kappa).

Blindness is procedural: the answers are in `data/key.json` and `site/key.js`, which a reviewer
could open. The page never shows them before export.

## The sample

- Seed 20260929; selection code in the source project, recorded in `data/selection.json`.
- The unit is **one guard round of one segment**. Rounds are stratified by **severity**: the fraction
  of that round's lattice `ridge_hit` removed, as production attributed it.
  - `lt10`: under 10 %;
  - `10to50`: 10–50 %;
  - `gt50_or_emptied`: over 50 %, or the guard emptied the segment.
- Per scroll and stratum, up to 6 rounds give a **removed** cell and up to 6 give a **kept** cell:
  - removed = flagged by `ridge_hit` and absent from the checkpoint the guard wrote (it may also have
    been flagged by another criterion);
  - kept = not flagged and present.
- Sources:
  - production rounds under policy D, 2026-09-29 13:15–~15:00 UTC;
  - for PHerc0211, which production did not grow under D, the A/B's arm-D rounds.

**160 samples** (79 removed, 81 kept):

| scroll | production rounds with ridge_hit computed <10 % / 10–50 % / >50 % | cells removed by ridge_hit, same strata | reviewed removed/kept <10 · 10–50 · >50 |
|---|---|---|---|
| PHerc0125 | 1 / 55 / 0 | 1,296 / 622,719 / 0 | 1/1 · 6/6 · 0/0 |
| PHerc0191 | 62 / 176 / 13 | 22,155 / 234,980 / 17,025 | 6/6 · 6/6 · 6/6 |
| PHerc0211 (A/B arm D) | 0 / 14 / 3 | 0 / 15,847 / 4,035 | 0/0 · 6/6 · 3/3 |
| PHerc0813 | 7 / 22 / 1 | 513 / 47,641 / 6,286 | 2/6 · 6/6 · 1/1 |
| PHerc0826 | 77 / 31 / 181 | 18,824 / 37,991 / 13,635 | 6/4 · 6/6 · 6/6 |
| PHerc0846A | 26 / 99 / 121 | 6,431 / 151,915 / 48,517 | 6/6 · 6/6 · 6/6 |
| PHerc0490B | — | — | none: **ridge_hit never runs there** (no surface prediction exists for it on any host), so there is nothing to review |
| 9 other First Letters scrolls (0175A/B, 0306B, 0343, 0483A/B, 0490A, 0846B, 1545) | 105 / 92 / 32 in total | 44,221 / 80,884 / 11,365 | not sampled (no local CT / prediction on the build host); in `prevalence.json` |

**Prevalence, all production under D** (1,101 rounds, 7.83 M cells grown, 1.37 M removed by ridge_hit):

| stratum | rounds | cells removed by ridge_hit |
|---|---|---|
| <10 % | 25.2 % | 6.8 % |
| 10–50 % | 43.1 % | **86.1 %** |
| >50 % or emptied | 31.6 % | 7.1 % |

So the fleet-wide false-removal rate is dominated by the 10–50 % stratum.

Shortfalls are real, not sampling choices: `data/shortfall.json` lists every stratum with fewer than
6 available rounds (e.g. PHerc0125 had one `<10 %` round and no `>50 %` round). Per-cell CIs on a
cell of n ≤ 6 are wide; the scorer pools across scrolls within a stratum.

## Inputs (per sample, `data/samples/<id>.npz`, uint8 unless noted)

| array | shape | what |
|---|---|---|
| `ct_stack` | 33 × 49 × 49 | CT in the cell's own frame: 33 planes parallel to the surface at normal offsets −16…+16 voxels, each 49 × 49 voxels, 1 voxel/px, nearest-voxel at pyramid level 0. `ct_stack[:, 24, :]` and `[:, :, 24]` are the two sections through the normal; `ct_stack[16]` is the plane at the surface |
| `pred_stack` | 33 × 49 × 49 | the thresholded surface prediction (0/255), same frame: what `ridge_hit` reads |
| `ct_xy`, `pred_xy` | 128 × 128 | the axis-aligned CT / prediction slice at the cell's z |
| `lattice_xyz` (float32), `lattice_valid` | 21 × 21 (×3) | the pre-guard lattice patch around the cell, level-0 voxels |
| `frame` (float32) | 4 × 3 | cell centre (x, y, z), normal n, tangents u, v |

`data/built.json` per sample:
- scroll, segment, round, source, lattice cell, centre;
- voxel size, 9.362 µm for every reviewed scroll (the scroll registry and the volume's `meta.json`
  agree);
- `ridge_hit_vox` = 3;
- the CT volume and prediction names. The prediction is the upstream m7 surface prediction
  `…-surface-20260413222639-surface-m7-L0-th0.2.zarr` for every scroll.

## Outputs

- The page exports `{reviewer, labels: {sample_id: {label, severity, note, pred_viewed, t}}}`.
- `score.py` reports, per severity stratum:
  - **false-removal share** P(sheet | removed): the share of the cut that was sound;
  - **miss share** P(not sheet | kept);
  - **FPR** P(removed | sheet) and **FNR** P(kept | not sheet), computed with production's removed
    and kept cell counts for that stratum.
- It also reports:
  - the production-weighted pooled rates, with bootstrap CIs;
  - per-scroll rates;
  - the automated CT check's agreement (and kappa) with the humans.
- `--weights production_sampled` restricts the weights to the reviewed scrolls. The default uses all
  production scrolls and assumes the unsampled ones share their stratum's rates; say which in any report.

## What `ridge_hit` itself reads and emits

- **Reads:**
  - the lattice (tifxyz x/y/z of the round's checkpoint);
  - per-cell normals from the lattice (`grid_normals`);
  - the thresholded surface prediction (OME-Zarr, level 0);
  - `ridge_hit_vox` (default 3 voxels);
  - for enforcement, the guard's pruning parameters (`min_bad_cells` 12, `reach_rings` 2).
- **Per cell:** samples the prediction at offsets −3…+3 voxels along the normal. The cell is
  **flagged** when no sample is positive.
- **Emits:**
  - the per-cell mask;
  - `frac_unsupported` (flagged / valid cells) and a segment-level `would_stop` when that exceeds
    0.5;
  - under enforcement, the cells removed: flagged regions of ≥ 12 cells that touch the growing edge,
    plus pieces cut off by them;
  - the count charged to `ridge_hit` in `pruned_by`, and a `nothing_left` stop when too little
    remains.

## How this differs from upstream's canonical methods

| upstream | what it does | vs `ridge_hit` |
|---|---|---|
| VC3D `vc_grow_seg_from_seed` with the prediction as `-v` and `space_line_weight` / `sdt_weight` > 0 | **soft residuals inside the Ceres optimiser**: SPACELINE and SURFACE_SDT pull the growing surface toward the prediction while it grows; nothing is deleted | `ridge_hit` is a **hard post-round gate**: it deletes cells that ended up off the prediction. The residuals prevent drift at optimisation cost; the gate removes it after the fact. Growth A/B: arm E (SPACELINE on, prediction as `-v`) plus the gate reached 4.2× vs D's 7.9× without SPACELINE |
| `vc_tifxyz_selfcross` | counts where a surface crosses itself | independent of the prediction; the guard enforces both. `ridge_hit` catches off-sheet surface that does not self-intersect, selfcross catches folds that may sit on sheet |
| `vc_calc_surface_metrics` | `in_surface_frac_valid`, `winding_valid_fraction` against **human point-collection annotations** | the upstream ground-truth route. This package is the same idea: a human says "on sheet", but per cell, blind, sampled from real decisions, with no VC3D annotation session |
| `vc_grow_seg_from_segments --sweep` (8 quality metrics) | scores a finished surface, 5 metrics against a target surface, 3 target-free (bbox fill, enclosed holes, completeness) | a parameter-tuning score for a whole run; it says nothing about which cells are off-sheet |
| Route B `satisfied_track_points` | fraction of surface-track points a spiral fit satisfies | a global-fit objective; blind to a whole-winding error (issue #1621) and not per cell |

## What the expanded tooling adds

- a per-cell, per-round, enforceable prediction-support test;
- its measured prevalence and severity in production;
- an independent automated CT check to compare it with;
- this blind, stratified human review, with reweighting to the production cut.

## Developing the method from here

- Change the test in `tifxyz_tools/growth_guard.py` (`ridge_hit_mask`), then re-run it on
  `data/samples/*.npz`: `pred_stack[:, 24, 24]` is the prediction along the cell's normal, so a new
  rule (a wider window, CT-based support, or a different threshold) can be scored against the human
  labels without the original volumes.
- New samples need the volumes: `build_inputs.py jobs.json OUT/` on a machine that has the
  checkpoints, CT and prediction, then `render_pages.py`.

## Limitations

- Sources: one afternoon of production (6 scrolls reviewed) plus the A/B for PHerc0211.
- Cells are nearest-voxel resampled at level 0 (9.362 µm); no interpolation.
- The CT contrast is stretched per sample (1st–99th percentile), so brightness is not comparable
  across samples.
- "Removed" cells may also have been flagged by another criterion; the key records the round's
  statistics, not per-cell attribution.
- The review question is "on a sheet", not "the right sheet": a surface that jumped to the next
  wrap but lies on papyrus is "sheet" here.

*Tests: `tests/test_ridge_review.py` builds, renders and scores a synthetic sheet with a known answer,
and checks that the shipped page reveals nothing before export.*
