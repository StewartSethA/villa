#!/usr/bin/env bash
# On-box loop: seed batch -> grow -> pack -> upload; repeat until /data/cloud-grow/STOP. No `set -e` (D28).
# Env (from /etc/cloud-grow.env + unit): CG BOX, UPLOAD_KIND (s3|rsync), UPLOAD_DEST, SEEDS_PER_BATCH, PARALLEL, ZMIN, ZMAX, RNG_SEED.
CFG=/data/cloud-grow/run.json; OUT=/data/cloud-grow/export; STOP=/data/cloud-grow/STOP
cd "$CG" || exit 2; export PYTHONPATH="$CG"
upload() {
  # write-only credential; objects are immutable names (<seg>.tar.gz + .sha256), re-upload is idempotent
  case "$UPLOAD_KIND" in
    s3) aws s3 sync "$OUT" "$UPLOAD_DEST/$BOX/" --only-show-errors --exclude "*" --include "*.tar.gz" --include "*.sha256" ;;
    rsync) rsync -a --partial -e "ssh -i /etc/cloud-grow-upload.key -o StrictHostKeyChecking=yes" "$OUT"/ "$UPLOAD_DEST/$BOX/" ;;
  esac
  echo "[loop] upload rc=$?"
}
batch=0
while [ ! -e "$STOP" ]; do
  batch=$((batch+1)); echo "[loop] $(date -u +%FT%TZ) batch $batch"
  ZARGS=""; [ -n "$ZMIN" ] && ZARGS="$ZARGS --zmin $ZMIN"; [ -n "$ZMAX" ] && ZARGS="$ZARGS --zmax $ZMAX"
  python3 -m cloud_grow seed --config "$CFG" --count "${SEEDS_PER_BATCH:-16}" --rng-seed $((RNG_SEED+batch)) $ZARGS > "$OUT.seeds.$batch.jsonl"; src=$?
  if [ $src -ne 0 ]; then echo "[loop] seeding returned $src (no seeds left in this shard?): sleeping 300 s, then stop-check"; sleep 300; continue; fi
  python3 -m cloud_grow grow --config "$CFG" --parallel "${PARALLEL:-1}"; echo "[loop] grow rc=$? (nonzero = some segment failed; failure rate is in the 'done:' line)"
  python3 -m cloud_grow pack --config "$CFG" --out "$OUT"; echo "[loop] pack rc=$?"
  upload
done
upload; echo "[loop] STOP seen: final upload done"
