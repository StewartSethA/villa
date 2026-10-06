# TESTING

Run: `cd cloud-grow && python -m pytest` (offline; needs numpy scipy tifffile zarr<3 pytest; ~25 s).
Mutation check: `python -u tests/mutants.py` (~10 min; applies one mutant at a time to a COPY of the tree and requires pytest to go red).

## What the suite covers
| file | covers |
|---|---|
| `test_state_policy_tools.py` | state shim interface (row unpacking idiom, append-only artifacts, pause flags); production guard policy values read back through the unmodified `policy_from_db`; policy typo = error; tool md5/`file` pin gate incl. announced override |
| `test_runner_seeding.py` | 2-round grow with the REAL guard code and fake tools: resume continues generations, D3 re-run resumes (never re-seeds), peak RSS recorded, fail-closed on selfx failure (pause + marker + alert), tracer failure surfaced, missing inputs, newest-checkpoint-by-mtime, config typo; seeding: avoids coverage/edges, separation, gap-seed rejection, provenance |
| `test_export_import.py` | manifest fields, pack (no cache_root, secret refusal), importer PASS/REFUSED: tamper, unlisted file, area overclaim (guard-written), hub selfx contacts, selfx unrunnable (fail closed), skip-selfx never passes, unpinned tool, D3 shrink, tar sha mismatch, path traversal, voxel conflict, `selfx_unverified.json`; dry-run writes nothing; re-import is a no-op; CLI exit codes |
| `test_data_preflight_cost.py` | level planning (no L0), preview before download, resume of a `.part`, skip of verified files, corrupt/truncated remote never kept nor marked complete, meta.json, preflight per-failure flags, cost arithmetic |
| `test_vendored.py` | vendored files unedited (md5), guard API surface, no fleet imports left |

## Seen failing first (D16)
Real defects found by tests going red before the fix: (1) `append_round_record` crashed when the first round failed before the segment dir existed; (2) `write_run_context` crashed on a new workdir; (3) the data preview crashed for a not-yet-existing nested destination; (4) fail-closed rounds wrote no `checkpoint` into `rounds.jsonl`, so a paused export named no final checkpoint. Then `tests/mutants.py` (result block below): every mutant must turn the suite red.

## Mutation result
Final run: **20/20 mutants killed** (importer: D3, selfx density, selfx-unrunnable, md5, unlisted files, tar traversal, guarded-area overclaim, unpinned tool; runner: fail-closed on guard exception, newest-by-mtime, D3 resume; policy typo; tool gate; fetch md5 + completion marker; pack secrets; cache_root exclusion; seeding edge margin + coverage dilation; preflight RAM).
First run was 17/20: three survivors, each a TEST gap, fixed and re-killed: (a) traversal check was redundant with `tarfile`'s own `filter="data"` (mutant now removes both); (b) no test made the guard itself raise (added `test_guard_exception_fails_closed_not_open`); (c) the original coverage mutant was equivalent (covered cells have score 0 anyway), replaced by "coverage not dilated by RADIUS_L4" with a test on the dilation.
Limits: 20 hand-picked mutants on the logic that guards correctness, not a coverage measurement; fakes stand in for the tools.

## NOT tested (explicit)
* No real `vc_grow_seg_from_seed` / `vc_tifxyz_selfcross` was run: the tools are fakes honouring the same CLI. Real geometry, real tracer RSS, real selfcross reports and timing are untested here.
* No rented machine; no download from the real bucket: the HTTP/XML listing + ranged-GET path (`HttpS3Source`) is untested, the engine is tested over a local directory source.
* No root: package installs, the kit build (`build_from_src_debian.sh`), cgroup/systemd limits, firewall are untested.
* `python -m cloud_grow seed|grow|pack` was smoke-run ONCE by hand with the fake tools on a synthetic CT zarr (2 seeds, 2 segments grown, 2 tarballs of ~0.01 MB); it is not a pytest and seeding on real level-4 arrays is untested.
* Cross-box determinism (same area on another CPU) and resume rounds at a 256 MB grid cache: not measured.
* Hub integration: registration is JSONL + sidecar files (+ optional INSERT-only sqlite table). Wiring those rows into the fleet's `pipeline_db`/scheduler is NOT done (it lives in the fleet repo).
