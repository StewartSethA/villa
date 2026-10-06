#!/usr/bin/env bash
# One-box orchestration: preflight -> seed -> grow -> pack. No `set -e` (D28): each stage reports its own exit code and
# the next stage only runs when its prerequisite succeeded (explicit checks, not a global abort).
# Usage: run_grow.sh run.json N_SEEDS PARALLEL OUT_DIR
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="$1"; N="${2:-8}"; PAR="${3:-1}"; OUT="${4:-./export}"
[ -f "$CFG" ] || { echo "usage: $0 run.json [n_seeds] [parallel] [out_dir]" >&2; exit 2; }
PY="${PYTHON:-python3}"; export PYTHONPATH="$HERE"
cd "$HERE" || exit 2
"$PY" -m cloud_grow.preflight --config "$CFG" --grows "$PAR"; rc=$?
[ $rc -eq 0 ] || { echo "preflight FAILED ($rc): fix before spending rental time" >&2; exit $rc; }
"$PY" -m cloud_grow seed --config "$CFG" --count "$N" > "$OUT.seeds.jsonl"; rc=$?
[ $rc -eq 0 ] || { echo "seeding produced no seeds ($rc)" >&2; exit $rc; }
"$PY" -m cloud_grow grow --config "$CFG" --parallel "$PAR"; grc=$?
echo "grow exit code $grc (nonzero = at least one segment failed; the rest are still packed)"
"$PY" -m cloud_grow pack --config "$CFG" --out "$OUT"; prc=$?
echo "pack exit code $prc; next: pull $OUT from the hub (hub-initiated), then hub/import_remote_grow.py"
exit $(( grc != 0 ? grc : prc ))
