#!/bin/bash
# Reader v2 dense distillation study queue (host V100, 2026-10-06). Arms train one after another on GPU 0, each
# resumable (an arm with student_best.ckpt + a .done marker is skipped), then one eval over every arm + references.
# The paired baseline (ink9_base: SAME arch/recipe/segments, ink_9um targets) starts as soon as the ink9 data copy
# is complete; the Reader v2 arms wait for the Reader v2 targets (logs/targets.log "TARGETS DONE").
# No set -e: an arm that fails is logged and the queue continues.
# Usage: rv2_queue.sh <base> [steps]
B=$1; STEPS=${2:-12000}
PY="${PY:-python}"
C=$B/code
log() { echo "$(date -u +%FT%TZ) $*"; }
wait_for() { while ! eval "$1"; do sleep 60; done; }

train() {   # train <arm> <data> <args...>; a killed run (oomd, crash) is RESUMED from its last checkpoint, <= 3 tries
  local arm=$1 data=$2; shift 2
  [ -f $B/runs/$arm/.done ] && { log "train $arm: done already"; return 0; }
  local try rc res
  for try in 1 2 3; do
    res=(); [ -f $B/runs/$arm/student.ckpt ] && res=(--resume $B/runs/$arm/student.ckpt)
    log "train $arm try $try ${res[*]} $*"
    CUDA_VISIBLE_DEVICES=0 $PY $C/scripts/reader_v2_distill/rv2_distill.py --data $data --src $C/src \
      --out $B/runs/$arm --steps $STEPS --val-every 2000 --val-max 12 --workers 6 --batch 16 --crop 256 \
      --hot-frac 0.5 --rescan-every 0 --label-subj $B/subj --run-prefix rv2dense_$arm "${res[@]}" "$@" >> $B/runs/$arm.log 2>&1
    rc=$?
    log "train $arm try $try rc=$rc"
    [ $rc -eq 0 ] && [ -f $B/runs/$arm/student_best.ckpt ] && { touch $B/runs/$arm/.done; return 0; }
    log "ALERT train $arm FAILED rc=$rc (try $try): $(grep -E 'Error|Killed' $B/runs/$arm.log | tail -1)"
  done
}

W3="--widths 48,64,128 --norm ms9-17"
RV2="--targets-desc reader_v2-step040000 --teacher-id reader_v2:${RV2_CKPT:-reader-v2-step040000.pth}:md5=7f261ac1b55e9aa19c4ad2fefe22a996"

wait_for "[ -f $B/data_ink9/.complete ]"
train ink9_base $B/data_ink9 $W3 --loss softbce+logit --targets-desc ink9um
wait_for "grep -q 'TARGETS DONE' $B/logs/targets.log"
train rv2_base $B/data_rv2 $W3 --loss softbce+logit $RV2
train rv2_edge $B/data_rv2 $W3 --loss softbce+edge $RV2
train rv2_sharpen2 $B/data_rv2 $W3 --loss softbce+logit --target-sharpen 2:1.0 $RV2
train rv2_rf33 $B/data_rv2 --widths 48,64 --norm ms9-17 --loss softbce+logit $RV2

CK=()
for a in ink9_base rv2_base rv2_edge rv2_sharpen2 rv2_rf33; do
  [ -f $B/runs/$a/student_best.ckpt ] && CK+=(--ckpt $a=$B/runs/$a/student_best.ckpt)
done
log "eval ${CK[*]}"
CUDA_VISIBLE_DEVICES=0 $PY $C/scripts/reader_v2_distill/rv2_eval.py $B/subj --src $C/src "${CK[@]}" \
  --ref teacher_rv2=rv2maps:$B/ref_rv2 --ref ink9um_student=arm:$B/ref_sharp:arm0 \
  --out $B/eval/rv2_eval.json --save-maps $B/maps > $B/eval/rv2_eval.log 2>&1
log "eval rc=$?"
log "QUEUE DONE"
