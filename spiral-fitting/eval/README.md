# spiral-fitting/eval — label-free checks for spiral fits

Small, read-only CLIs that check a `fit_spiral.py` result on a scroll with **no verified patches and
no winding annotations** (every 2027 prize scroll). They read the fit's per-winding tifxyz meshes
(`meshes/<name>/wNNN_spliced_*`), the published surface prediction (an OME-Zarr group with arrays
`"0"`, `"1"`, …), and an umbilicus JSON (`control_points: [{z, x, y}]`, level-0 voxels). Each check
has its own null or control. None of them modifies the fit or its inputs.

Dependencies: numpy, zarr, tifffile, scipy (area estimate only). Tested with the repo-external env
used to write them (Python 3.12, zarr 2.18.7, numpy 1.26); **spiral-fitting's own pins
(Python 3.14, zarr 3, numpy 2.5) have not been tried.**

| tool | question | null / control |
|---|---|---|
| `onsheet.py MESHDIR PRED OUT.jsonl [--shift H] [--level 1] [--tol 1]` | Does each fitted winding lie **on** a predicted sheet, or in the gap between two? `on_sheet` = fraction of sampled lattice points with prediction > 0 within ±tol. | Same points moved ±H voxels along the surface normal (H = half the lamina pitch) → `null_shift`; `lift = on_sheet − null_shift`. Both use the same chunk reads, so a wrong prediction store fails both. (`material_frac` cannot answer this: it was 0.9997 against a bbox-occupancy control of 0.987 on PHerc0211 windings.) |
| `outer_gap.py MESHDIR PRED UMB.json Z --voxel-um UM [--ct CT]` | Does the fit reach the outside of the scroll, or is it truncated by `shell_outer_winding_idx`? Per angle bin, outermost fitted radius vs outermost prediction-positive (or CT-body) radius, plus the number of sheet crossings outside the fit. | A fit whose shell is known to be adequate (on PHerc0211 its median gap was 1.5 mm). |
| `window_seams.py MESH_A MESH_B UMB.json Z0 Z1 OUT.json [--pitch P]` | Where two z-window fits overlap, is winding *w* of A the same sheet as winding *w* of B? Median radial residual per offset k ∈ [−5, 5]. | k = 0 with residual well under a pitch = consistent; the runner-up offset is reported as a margin. |
| `reconcile_seams.py UMB.json OUT.json --windows TAG:Z0:Z1:MESHDIR …` | Per-winding matching across every seam. A match is accepted only if its residual is < 0.35 pitch, it beats the runner-up by > 0.3 pitch, and it is mutual. Accepted matches are joined into scroll-long chains. | Unmatched windings are left unmatched, never forced. The pitch is measured from the windings themselves. |
| `scroll_area_est.py CT --z0 --z1 --vox-um UM [--pitch LO HI]` | A physical denominator: CT body volume / winding pitch → sheet-area range, for "what fraction of the scroll does this fit cover". | An upper-bound-ish scale: voids inside the body count as papyrus. Report it as a range. |

Recipe per window: fit → `onsheet` → `outer_gap` (at 2–3 z) → `window_seams` / `reconcile_seams`
against its neighbours.

## Tests

`tests/eval/` builds a synthetic Archimedean spiral (pitch 20 voxels, 9 windings) with a matching
prediction volume. Each test pairs a positive case with a negative one. Run:
`python -m pytest tests/eval -q` → **5 passed** (2026-09-29).

| test | positive / negative | mutation that turns it red |
|---|---|---|
| onsheet | on-sheet windings: on_sheet > 0.9, null < 0.2 / windings half a pitch off: lift < 0 | null shift set to 0 → red |
| outer_gap | full fit: gap < 0.25 pitch / outer 3 windings dropped: gap = 3 pitches ± 0.25 | keep the innermost instead of outermost radius per bin → red |
| window_seams | planted +1 renumbering → best k = +1 / consistent windows → k = 0 | look up `w − k` instead of `w + k` → red |
| reconcile_seams | consistent windings matched at 0; a planted swap is followed (+1/−1) / a winding half a pitch off is left unmatched | accept every best candidate (`ok = True`) → red. **Known limit:** dropping only the mutual-match condition does NOT turn it red (the fixture has no non-mutual case) |
| scroll_area_est | solid cylinder: volume within 2 %, area = volume / pitch within 2 % | multiply by pitch instead of dividing → red |

## Recorded results (from the fits these tools were written for; not re-run for this branch)

Scrolls PHerc0211, PHerc0125, PHerc0191 (9.362 µm volumes), published tracks, published surface
prediction `surface-m7-L0-th0.2`, `input_disable_patches: true`, 30,000-step fits in z-windows with
200-slice overlaps.

- **Window seams, PHerc0211 (5 windows, z 4500–17500).** Mutual per-winding matches at the four seams:
  **15/80, 10/80, 0/90, 0/103**, with best-residual median 0.58–0.65 pitch. A same-window repeat agreed
  to 0.04 pitch. Adjacent window fits largely do **not** put the same sheet under the same winding
  index. This is one scroll, and the windowing choices (overlap, shell size) were not varied.
- **On-sheet lift.** PHerc0125, 5 windows, 2,000 points per winding: on_sheet median 0.30–0.33
  against a half-pitch null of 0.22–0.25; lift > 0 on 68/90, 68/88, 74/95 and 68/88 windings in
  four windows. A refit (shell 125, upstream human umbilicus): 0.315 vs 0.244, lift > 0 on 98/115.
  PHerc0211: 0.30–0.35 vs 0.23–0.27.
- **Outer gap** (one z-slice, 72 angle bins). Median gap: PHerc0211 1.5 mm (control); PHerc0125
  shell 105 → 2.4 mm, shell 125 → +0.08 mm (p90 1.24 mm); PHerc0191 shell 113 → 5.5 mm, shell 141
  → −0.11 mm (p10 −1.33, p90 4.84 mm).
- **Sheet area** (pitch 16–19 L0 voxels): PHerc0125 12,382–14,703 cm²; PHerc0191 14,800–17,574 cm²;
  PHerc0211 10,065–11,952 cm² (its spiral-fit windows sum to 9,909 cm²). Published tracks cover
  z 4500–17500 only, which leaves out ~33 % (PHerc0125) and ~18 % (PHerc0191) of the estimate.

Limitations: most checks are n = 1 scroll or one slice. `onsheet` treats the prediction as truth, so
prediction errors are invisible to it (the null shares them, so it cannot invent lift, but it can
miss some). None of these checks sees two windings on one sheet within a single window. Self-crossing
within a surface is `vc_tifxyz_selfcross`'s job.

## Spiral sense: evidence, not a verdict

Upstream derives `spiral_outward_sense` from the catalog (PR #1899, `surface_orientation.py`):
`"ACW" if z_direction_is_top_to_bottom != left_handed_coordinates else "CW"`. The docstring
states the convention ("every scroll shows the same spiral seen from its top"), and it is a
convention, not a measurement per scroll. We ran fits both ways (same dataset and seed, 30,000
steps) and compared `satisfied_track_points`:

| scroll | catalog z_top_to_bottom / left_handed (metadata.json, fetched 2026-09-29) | catalog-derived | fit A/B higher | Δ satisfied_track_points | same-arm spread (noise floor) |
|---|---|---|---|---|---|
| PHerc0211 | true / false | **ACW** | CW | 0.000283 | 0.000126 (one repeat pair, this scroll; auto umbilicus) |
| PHerc0191 | false / false | **CW** | ACW | 0.000956 | not measured on this scroll (repeat queued); 0211's is 0.000126 |
| PHerc0125 | null / false | underivable | ACW | 0.003779 | not measured on this scroll; 0211's is 0.000126 |

**Both decidable cases point the opposite way from the catalog.**

What was checked about a sign-convention cause: the fit consumes the flag in one place,
`transforms.py` `_get_transform_parts` (CW = no flip; ACW = `AffineTransform` scaling x by −1 in
zyx order). That block is **identical** at our fits' commit (villa `2764e4d`; `fit_spiral.py` md5
`db22aa6a…`) and at `origin/main` (6e53201). Our commit also had a ray-specialised fast path
(`ray_specialized_spiral_to_scroll`, removed upstream in #1871, "Spiral simplification"). It applies
the flip as `x_sign = −1` on the x component, which matches the generic flip. So **the flag's meaning
inside the fit did not change between the code we ran and upstream `main`**. PR #1899 changed
export orientation and added the catalog derivation, not the model's handedness.

That leaves three explanations, none tested yet:
1. The A/B does not measure sense. Issue #1909 reports exactly this for PHerc0826 and PHerc0813
   (both senses fit equally well). Our 0211 effect is only 2.2× a single-pair spread.
2. An axis-order or handedness difference in our fit inputs (umbilicus, tracks) relative to the
   catalog's volume frame.
3. The catalog convention does not hold for these scrolls. Issue #1909 also mentions a community
   PHerc0826 fit that settled on the opposite sense from the catalog, which is the same pattern as ours.

**No recommendation is made from these data.** The catalog value should be used where it exists.
A cheap discriminating test is a 1,500-step CW/ACW pair on PHerc0211 with `origin/main`'s
`fit_spiral.py` and a catalog-derived spec, comparing exported mesh orientation. A second-agent
investigation of this conflict was under way when this branch was written, and nothing from it was
recorded yet.
