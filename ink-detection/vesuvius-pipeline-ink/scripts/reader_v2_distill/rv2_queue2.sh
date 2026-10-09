#!/bin/bash
# Follow-up arm after rv2_queue.sh (added 2026-10-06 once the paired baseline showed the 59 px student far below the
# teacher): rv2_n47 = 2-level net + 9/17/47 px normaliser (support 17 + 46 = 63 px <= 64 cap) -- is the missing
# accuracy the normaliser's context or the net's depth? Then ONE eval over every arm (eval/rv2_eval_all.json).
# No set -e. Usage: rv2_queue2.sh <base> [steps]
B=$1; STEPS=${2:-12000}
PY="${PY:-python}"
C=$B/code
log() { echo "$(date -u +%FT%TZ) $*"; }
while ! grep -q "QUEUE DONE" $B/logs/queue.log; do sleep 60; done
RV2="--targets-desc reader_v2-step040000 --teacher-id reader_v2:${RV2_CKPT:-reader-v2-step040000.pth}:md5=7f261ac1b55e9aa19c4ad2fefe22a996"
arm=rv2_n47
if [ ! -f $B/runs/$arm/.done ]; then
  log "train $arm"
  CUDA_VISIBLE_DEVICES=0 $PY $C/scripts/reader_v2_distill/rv2_distill.py --data $B/data_rv2 --src $C/src \
    --out $B/runs/$arm --steps $STEPS --val-every 2000 --val-max 12 --workers 6 --batch 16 --crop 256 \
    --hot-frac 0.5 --rescan-every 0 --label-subj $B/subj --run-prefix rv2dense_$arm \
    --widths 48,64 --norm ms9-17-47 --loss softbce+logit $RV2 > $B/runs/$arm.log 2>&1
  rc=$?; log "train $arm rc=$rc"
  [ $rc -eq 0 ] && [ -f $B/runs/$arm/student_best.ckpt ] && touch $B/runs/$arm/.done
fi
CK=()
for a in ink9_base rv2_base rv2_edge rv2_sharpen2 rv2_rf33 rv2_n47; do
  [ -f $B/runs/$a/student_best.ckpt ] && CK+=(--ckpt $a=$B/runs/$a/student_best.ckpt)
done
log "eval ${CK[*]}"
CUDA_VISIBLE_DEVICES=0 $PY $C/scripts/reader_v2_distill/rv2_eval.py $B/subj --src $C/src "${CK[@]}" \
  --ref teacher_rv2=rv2maps:$B/ref_rv2 --ref ink9um_student=arm:$B/ref_sharp:arm0 \
  --out $B/eval/rv2_eval_all.json --save-maps $B/maps > $B/eval/rv2_eval_all.log 2>&1
log "eval rc=$?"
log "QUEUE2 DONE"
