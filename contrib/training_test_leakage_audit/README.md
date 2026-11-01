# Training / test overlap audit for ink models

`leakage_audit.py` compares a model's training manifest with a test list **by physical segment** and exits non-zero when they overlap, so it can gate a training or evaluation job.
It also provides `write_datasets_json`, the record every run should publish before its first step (segments per role, label source, extras).

## What is your contribution?
1. **Scroll data used.** The 99 training representations of reader-v3 (Scrolls PHerc0139, 0500P2, 0009B, 0814, Paris4 (Scroll 1), 1667 (Scroll 4), 0343P, 0172) against the 8 upstream test segments
   (PHerc0009B, PHerc0139 w030 and w045, PHerc0500P2, PHerc0814 46527_2um_try2, PHerc0841 x3). No eligible-prize-scroll data is used or shown.
2. **Why it raises the chance of reading scrolls.** Public ink models are compared on one test list; if a model's training set contains test segments, the table measures memory, not reading.
   The same point was raised in the community as the hard part of any cross-model benchmark. A one-command check and a standard manifest let authors and reviewers say, per model, what is in-sample.
3. **What it enables that was not possible before.** A reproducible, id-robust overlap test (a training representation named `pherc0139-w017` that points at physical segment `20250108000005-w030` is found
   although its name does not contain the id) and a way to attach "this many test segments were seen" to every published number.
4. **Evidence.** Running it on our reconstruction of reader-v3's training list (`example/v3_training_reps.json`, 99 representations, built from the model's `train_config.json` and its label sources;
   `example/upstream_test_ids.json`):

```
$ python leakage_audit.py example/v3_training_reps.json --test example/upstream_test_ids.json
example/v3_training_reps.json: 5 training entries overlap 3 of 8 test segments
   pherc0139-w017   -> 20250108000005 (PHerc0139 w030)   label_kind pseudo
   pherc0139-w029   -> 20260126000000 (PHerc0139 w045)   label_kind pseudo
   pherc0814-46527  -> 46527_2um_try2 (PHerc0814)        label_kind pseudo
   20250108000005-w030_2025010818 -> 20250108000005      label_kind v2native
   20260126000000-w045_2026012619 -> 20260126000000      label_kind v2native
```
So 3 of the 8 test segments (w030, w045, 46527_2um_try2) appear in the training list, both with human ("v2native") and machine pseudo-label representations.
Stock reader-v3 through the upstream-test harness (equal-weight human-label AUC, P384; one run):

| test segment | in the training list | human AUC |
|---|---|---|
| PHerc0009B 20250919125754 | no (other segments of the scroll are) | 0.9385 |
| PHerc0139 w030 | **yes** | 0.9074 |
| PHerc0139 w045 | **yes** | 0.8456 |
| PHerc0500P2 20250825181859 | no | 0.8980 |
| PHerc0814 46527_2um_try2 | **yes** | 0.9369 |
| PHerc0841 20260220213127 | no | 0.8891 |
| PHerc0841 20260220214732 | no | 0.8444 |
| PHerc0841 20260221022814 | no | 0.8548 |

Mean over the 3 overlapping segments 0.8966, over the 5 others 0.8850 (difference +0.012; n = 3 vs 5, so this does not separate in-sample inflation from segment difficulty: w045 is the weakest segment although it is in the list).
We therefore train our own comparison models on the training list with those three segments removed and report both numbers.

## Limits
- The 99-entry list is our reconstruction of reader-v3's `train_config.json`; please run the tool against the original file before relying on the counts.
- Overlap is by physical segment, not by pixel: representations may cover different regions of the same segment. The audit is therefore a conservative flag, not proof that labelled pixels were seen.
- Tests use synthetic ids only; the evidence above is the real-list run.

## Files
`leakage_audit.py`, `tests/test_audit.py`, `example/v3_training_reps.json` (name, scroll, physical segment, label kind: no paths), `example/upstream_test_ids.json`.
