# vesuvius-pipeline-ink

Code-only snapshot of the ink-detection part of a private research pipeline (`vesuvius_pipeline`),
shared for reference and reuse. **Weights, labels, predictions, images and data are not included.**
This is an experimental, research-grade drop: it is *not* a standalone installable package.

## What is in here

| Path | What it does |
|---|---|
| `src/vesuvius_pipeline/stages/ink_models/registry.py` | Family registry: one `InkFamily` row per model family (name, window `tile_px` x um/px, input spec, caveats, which scrolls it trained on). `sharp_p2_manifest.json` and `hecate_frames.json` are data tables it reads. |
| `.../reader_v2.py`, `reader_v2_dense.py`, `_reader_v2_villa.py` | Reader v2 (68.2 M-parameter ink_9um architecture) and its dense-output variant. Arms are checkpoint swaps of one module (`CKPTS`): `""` (step 40000), `ft12k`, and `dense_native`. |
| `.../ink9um.py`, `ink9um_student.py`, `grandprize_dense.py`, `hecate.py`, `i3d.py`, `resnet3d_1667.py`, `k1.py`, `colmean.py`, `patchclf.py`, `pixelclf.py`, `stroke_ring.py` | Other model-family adapters (tiling, TTA, blending, per-family loaders). |
| `.../ensemble.py`, `labelloop.py`, `tiling.py`, `_compile.py`, `_progress.py` | Shared helpers: ensembling, label-loop checkpoint handling, tile origins, torch.compile bucketing, progress. |
| `scripts/ink_ensemble/zavg.py` | Z-score-average consensus of finished ink maps (per-map z-score over valid pixels, mean, one p1..p99.5 stretch to uint8). Standalone (numpy + Pillow). |
| `scripts/reader_v2_distill/` | Distillation of Reader v2 into smaller dense students: training (`rv2_distill.py`), eval (`rv2_eval.py`), benches, and the queue shell scripts. |
| `scripts/lasagna_distill/` | Patch building, training and inference for a distilled student of a field-prediction model. |
| `scripts/convnext_ink/{split_guard,leakage_check,submission_valid,pick_submission,splits_apply}.py` | Geometric split guard (excludes training surface that physically overlaps protected/validation surface even when segment names differ), leakage check, and submission-validity/selection helpers. |
| `tests/test_split_guard.py` | The only test that runs standalone (`pytest tests`; needs numpy, pytest). 5 pass. |

## dense_native (Nieuwlaar)

`dense_native` is Erwin Nieuwlaar's fine-tune of ink9um-dense with dense native-scan pseudo-labels
(step 16 000 of 20 000), MIT licensed: https://huggingface.co/Nieuwlaar/ink9um-dense-native .
The release is a bare safetensors state_dict with the same 508 tensors (names and shapes) as Reader v2;
we rebuild a `.pth` with the author's recipe. `reader_v2.py` audits checkpoints against pinned hashes
(`UPSTREAM["md5"]` / `UPSTREAM["sha256"]`, checked on load): `dense_native-016000.pth`
md5 `96e6054f69551914342af5c966259b88`, sha256
`04ce4dd969f6a4d1daa87fd8675245749ce54b842866df7ba7f339c9c7616cb6`. Reader v2 weights come from the
upstream release named in `reader_v2.py` (`UPSTREAM`); that code and weights keep their own licenses
and attribution. Place checkpoints in a `reader_v2/` model dir (env `VPIPE_READER_V2_DIR`).

## What is NOT standalone (plainly)

- The modules import private pipeline modules that are not included: `config`, `frame`
  (`ModelSpec`/`MODELS`), `settings`, `alerts`, `db.pipeline_db`, `remotedb`, `data`, `scrolls`,
  `stages.{ink,finish,memstack,render,input_frame,ink_progress}`. Treat them as readable reference;
  importing `registry` needs those or stubs. Only `zavg.py`, the `convnext_ink` guards and
  `tests/test_split_guard.py` run on their own (numpy/Pillow).
- Scripts expect an environment: `PY` (python interpreter) and `RV2_CKPT` (teacher checkpoint path)
  for the shell queues; `ink_detection_results/<seg>` and `guard_pulled` are placeholder default
  directories (override with `--dir/--out/--work`). Absolute site paths were replaced by these.
- `RV2_TRAIN_SEGMENT_KEYS` (segment ids a checkpoint trained on) is not shipped: set
  `RV2_TRAIN_SEGMENT_KEYS_FILE` to a one-id-per-line file, or derive from the checkpoint's
  `config.datasets`. Without it, segment-level seen/unseen reporting falls back to scroll-level.
- Docstrings refer to internal docs/FINDINGS that are not part of this drop.

## Licenses and attribution

Repository license applies to this code. Model weights are not distributed here; each remains under its
author's license (Reader v2 and ink9um: their authors; dense_native: MIT, E. Nieuwlaar; released
PHerc.1667 ResNet3D iterations: MIT; Grand Prize TimeSformer: Scroll Prize). Ensemble recipe credit:
Nieuwlaar. Verify upstream licenses before redistributing any checkpoint.
