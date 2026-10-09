#!/bin/bash
# Reader v2 teacher targets for every distillation segment, on the SAME pixels as the ink_9um targets (x.npy ->
# 17-layer stack -> ink9um_teacher_targets.py --ckpt reader-v2). Resumable; re-scans <data> until <expect>
# segments have targets or a pass finds no new work twice in a row (data still arriving -> keeps polling).
# No set -e: one bad segment is reported by the teacher script and skipped. 2026-10-06.
# Usage: rv2_targets_loop.sh <base> <expect> <ckpt>      (base holds code/, data_ink9/, data_rv2/, scratch/)
B=$1; EXPECT=$2; CKPT=$3
PY="${PY:-python}"
C=$B/code
idle=0
while true; do
  $PY $C/scripts/reader_v2_distill/rv2_layers_from_x.py $B/data_ink9 $B/scratch/layers
  before=$(ls $B/data_rv2/*/meta.json 2>/dev/null | wc -l)
  CUDA_VISIBLE_DEVICES=0 $PY $C/scripts/hires_ink/ink9um_teacher_targets.py $B/scratch/layers/list.tsv $B/data_rv2 \
      --repo $C --ckpt $CKPT --scratch /dev/shm/rv2_teacher --batch 16 --workers 6 2>&1 | grep -E "^\[teacher\]"
  # free the layer copies of finished segments; carry any loss mask (S4 val boxes cut out) across
  for d in $B/data_rv2/*/; do
    s=$(basename $d)
    [ -f $d/meta.json ] && rm -rf $B/scratch/layers/$s
    [ -f $B/data_ink9/$s/lossmask.npy ] && [ ! -f $d/lossmask.npy ] && cp $B/data_ink9/$s/lossmask.npy $d/
  done
  after=$(ls $B/data_rv2/*/meta.json 2>/dev/null | wc -l)
  echo "$(date -u +%FT%TZ) targets $after / $EXPECT (this pass +$((after-before)))"
  [ $after -ge $EXPECT ] && break
  if [ $after -eq $before ]; then idle=$((idle+1)); sleep 120; else idle=0; fi
  [ $idle -ge 30 ] && { echo "no progress for 30 passes: stop"; break; }
done
echo "$(date -u +%FT%TZ) TARGETS DONE"
