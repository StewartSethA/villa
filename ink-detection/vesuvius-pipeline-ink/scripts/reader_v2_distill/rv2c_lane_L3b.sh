#!/bin/bash
# Takes over lane L3 after arm0c (2026-10-06): waits for the running arm0c trainer to exit, evaluates it, then runs
# the warp-decomposition arms BEFORE the remaining sharp arms (priority: the user's warp order), via rv2c_queue2.sh.
B=$1
C=$B/code_v3/scripts/reader_v2_distill
PY="${PY:-python}"
LOG=$B/logs/conv_L3b.log
log() { echo "$(date -u +%FT%TZ) [L3b] $*" >> $LOG; }
while pgrep -f "[r]uns_conv/arm0c " > /dev/null; do sleep 60; done
if grep -q '"converged": true' $B/runs_conv/arm0c/convergence.json 2>/dev/null; then
  touch $B/runs_conv/arm0c/.done; log "arm0c converged; eval"
  systemctl --user reset-failed conv-eval-arm0c.scope 2>/dev/null
  systemd-run --user --scope -p MemoryMax=60G --unit=conv-eval-arm0c -E PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY $C/rv2_eval.py $B/subj --src $B/code_v3/src --ckpt conv_arm0c=$B/runs_conv/arm0c/student_best.ckpt \
    --ckpt conv_arm0c_last=$B/runs_conv/arm0c/student.ckpt \
    --ref teacher_rv2=rv2maps:$B/ref_rv2 --ref ink9um_student=arm:$B/ref_sharp:arm0 \
    --out $B/eval_conv/arm0c.json --save-maps $B/maps_conv > $B/eval_conv/arm0c.log 2>&1
  log "eval arm0c rc=$?"
else
  log "ALERT arm0c trainer exited without convergence.json converged=true; rv2c_queue2 will resume it"
fi
bash $C/rv2c_queue2.sh $B L3b arm0c rv2c_sharpen2_warpxy rv2c_sharpen2_warpz arm1_edge_c base_repro_c arm2c
