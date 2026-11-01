# Sheet-localised 3-D ink head for a reader-v3-style ink model

A 2-D ink U-Net (reader-v3: 3-D depth-fusion stem + 2-D U-Net, 17 input planes at ~9.6 um) says *whether* a pixel carries ink.
This contribution adds a 16-plane **depth head** (a 1x1 conv on the finest decoder stage) that says *on which plane / sheet*, trained **without any 3-D human label**:
the 2-D label is snapped onto the ink-bearing sheet using a teacher's depth profile (Hecate 9.6 um), restricted to +-`film` planes around the teacher's peak, so ink on the
neighbouring sheets is never a target. Cost is about 1/25 of Hecate per mm^2 (reader-v3 ~23 GMAC/mm^2 vs Hecate ~565 GMAC/mm^2).

## What is your contribution?
1. **Scroll data used.** Training / teacher volumes: Scroll 1 (PHercParis4) and Scroll 4 (PHerc1667) open-data segments (11-13 segments, 9.6 um surface renders) plus the unified reader-v3 training store.
   Evaluation: the 8 upstream test segments listed in the reader-v3 README (human-label AUC, equal weights), through the same harness for every row. No eligible-prize-scroll data is used or shown.
2. **Why it raises the chance of reading scrolls.** In overlapped / crushed regions a 2-D model paints ink from the neighbouring sheet onto the sheet being read. A depth head that is trained to
   put ink on the sheet the mesh intends gives a reader-v3-cost model that can (a) be run over whole scrolls, (b) report the depth at which each stroke sits, and (c) be compared against
   other 3-D models with the leakage metrics below. On the upstream test it also scores slightly higher than the stock model (table below).
3. **What it enables that was not possible before.** (i) Per-pixel sheet/depth localisation from a model that costs about the same as the reader-v3 you already run, with no 3-D labels;
   (ii) a controlled ablation of *where* depth supervision should be allowed (+-1 plane around the teacher's peak vs any plane): limiting it to the sheet is better;
   (iii) a leakage measurement (`sheet_leakage_metrics.py`) that compares sheet maps against a laterally shifted null, usable for any 3-D ink output.
4. **Evidence** (all on real data; one training seed per row unless stated; replicate seeds are running and will be appended to this file):

### Upstream test (8 segments, human-label AUC, equal weights, last checkpoint)
| model | P384 | P448 |
|---|---|---|
| stock reader-v3, same harness (positive control; README reference 0.892 at P384) | 0.8893 | 0.892 |
| single-task control (same data and recipe, depth head not trained) | 0.8911 | 0.8941 |
| **depth head, film +-1 plane** (seed 0) | **0.8968** | **0.8994** |
| depth head, film +-1 plane (seed 1) | 0.8947 | 0.8957 |
| depth head, film +-1 plane, 13 segments (seed 2) | 0.8968 | 0.8991 |
| depth head, film +-3 planes | 0.8933 | 0.8953 |
| depth head, any sheet (3-D target = teacher probability on every plane, 2-D = max over depth) | 0.8841 | 0.8871 |

Reading: every film +-1 run is above stock (+0.0054 to +0.0075 AUC at P384) and above the control; the any-sheet variant is *below* stock. n = 8 segments per cell, no confidence interval yet (single seeds):
treat the gain as a lead until the replicates finish.
Earlier versions of this recipe that (a) normalised with (x-150)/25 instead of the reader-v3 robust-MAD recipe and (b) trained on the S1/S4 stream only scored 0.749-0.795 on the same test:
**the store mixing and the optimiser (SGD nesterov, not AdamW) are required**; both are in `train_depth_head.py`.

### Leakage between adjacent sheets (sheet_leakage_metrics.py; Pearson r of per-sheet max-projected ink maps, own vs +-1 sheet, next to a map rolled 97 px = the null)
| model / data | own vs +-1 sheet | null |
|---|---|---|
| official 3-D student, S1, 5 windows of 512 px | -1 side 0.067-0.145 (mean 0.110); +1 side -0.087-0.175 (mean 0.042) | mean 0.027 / 0.012 |
| Hecate 9.6 um forward, S4, 5 segments | 0.23-0.44 | -0.03-0.11 |
| Hecate 9.6 um forward, S1, 5 segments | -0.13-0.09 (null level) | -0.06-0.15 |
Hecate writes ink in a column about 19 um thick (median; p90 38-58 um) on S4 and about 10 um on S1; its input window is 16 planes (154 um) and its gradient receptive field fills the whole 16 x 64 x 64 patch.

## Files
- `depth_head.py` - the head, the snapped target and the masked loss (torch only; tests in `tests/`, synthetic tensors, check the masking and the film restriction).
- `train_depth_head.py` - fine-tuning loop with plug-in net and aux-data factories; writes `datasets.json` (train segments, held-out band, aux source) before step 1.
- `hecate_teacher_volumes.py` - Hecate 9.6 um volumes for a whole surface-render zarr, banded and resumable.
- `sheet_leakage_metrics.py` - the adjacent-sheet coupling metric with its shifted null.

## What is NOT shown
- Tested with the reader-v3 network in our own harness (same architecture as the upstream `vesuvius.ink_detection` model); not yet run through the `vesuvius` package loaders: `--net-factory` is the adapter point.
- Held-out 2-D AUC inside the training segments saturates at stock level and is *not* a transfer measure; use the upstream test.
- One seed per arm for the table above; no confidence intervals.
