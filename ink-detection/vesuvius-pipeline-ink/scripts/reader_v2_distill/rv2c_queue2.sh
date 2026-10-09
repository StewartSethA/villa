#!/bin/bash
# v2 of rv2c_queue.sh (+ warpxy / warpz arms; a running bash lane must never have its script edited underneath it)
# Convergence retrain (2026-10-06, user: "undertrained ... did not converge"): one LANE of arms, run in sequence on
# host's V100, each in its own memory-capped systemd scope, relaunched with --auto-resume on failure (max 4 tries),
# then scored with rv2_eval.py. No set -e (one bad arm must not cancel the lane).
# Usage: rv2c_queue.sh <base> <lane-name> <arm> [<arm> ...]       arms are defined below.
# Log: <base>/logs/conv_<lane>.log  (ALERT lines on every failure / give-up)
B=$1; LANE=$2; shift 2
PY="${PY:-python}"
C=$B/code_v3
LOG=$B/logs/conv_$LANE.log
mkdir -p $B/logs $B/runs_conv $B/eval_conv $B/maps_conv
log() { echo "$(date -u +%FT%TZ) [$LANE] $*" >> $LOG; }
RV2="--targets-desc reader_v2-step040000 --teacher-id reader_v2:${RV2_CKPT:-reader-v2-step040000.pth}:md5=7f261ac1b55e9aa19c4ad2fefe22a996"
# convergence regime shared by every arm: plateau schedule on held-out teacher r, EMA weights, save best AND last
CONV="--sched plateau --ema 0.9995 --warmup 2000 --val-every 4000 --val-max 12 --plateau-patience 4 --plateau-eps 0.002 --plateau-factor 0.3 --plateau-max-red 2 --rescan-every 0 --label-subj $B/subj --workers 8 --auto-resume"
EDGE="--outside-w 0.5 --edge-frac 0.15"
RV2NET="--widths 48,64,128 --norm ms9-17 --loss softbce+logit --hot-frac 0.5 --crop 256"
SHARP="--widths 48,64,128,192,256 --norm ms33-129-257 --rf-cap 0 --hot-frac 0.5 --crop 256 --batch 16"

args_for() {
  case $1 in
    rv2c_sharpen2)      echo "--data $B/data_rv2 $RV2NET --target-sharpen 2:1.0 --batch 32 --lr 2e-3 --steps 200000 --min-steps 40000 $EDGE $RV2 --run-prefix rv2c_sharpen2" ;;
    rv2c_sharpen2_warp) echo "--data $B/data_rv2 $RV2NET --target-sharpen 2:1.0 --batch 32 --lr 2e-3 --steps 200000 --min-steps 40000 $EDGE --warp3d 0.7 --domain-aug 0.7 $RV2 --run-prefix rv2c_sharpen2_warp" ;;
    # paired warp-decomposition arms (coordinator 2026-10-06, after the ensemble TTA study): in-plane only (coarse
    # field, same shift in every layer, 0-2 px rms) and depth only (depth fields + stretch/pinch/bulge); no domain aug
    rv2c_sharpen2_warpxy) echo "--data $B/data_rv2 $RV2NET --target-sharpen 2:1.0 --batch 32 --lr 2e-3 --steps 200000 --min-steps 40000 $EDGE --warp3d 0.7 --warp-mode inplane $RV2 --run-prefix rv2c_sharpen2_warpxy" ;;
    rv2c_sharpen2_warpz)  echo "--data $B/data_rv2 $RV2NET --target-sharpen 2:1.0 --batch 32 --lr 2e-3 --steps 200000 --min-steps 40000 $EDGE --warp3d 0.7 --warp-mode depth $RV2 --run-prefix rv2c_sharpen2_warpz" ;;
    rv2c_base)          echo "--data $B/data_rv2 $RV2NET --batch 32 --lr 2e-3 --steps 200000 --min-steps 40000 $EDGE $RV2 --run-prefix rv2c_base" ;;
    rv2c_wide_sharpen2) echo "--data $B/data_rv2 --widths 96,128,256 --norm ms9-17 --loss softbce+logit --hot-frac 0.5 --crop 256 --target-sharpen 2:1.0 --batch 32 --lr 2e-3 --steps 200000 --min-steps 40000 $EDGE $RV2 --run-prefix rv2c_wide_sharpen2" ;;
    # sharp ablation family (ink_9um targets, production ink9um_student architecture, RF NOT capped: these are
    # comparisons with the production student, never deployed)
    arm0c)       echo "--data $B/data_ink9 $SHARP --loss softbce+logit --lr 5e-4 --steps 180000 --min-steps 80000 --resume $B/ref_prod_ink9um_student.ckpt $EDGE --run-prefix arm0c_from_prod60k" ;;
    base_repro_c) echo "--data $B/data_ink9 $SHARP --loss softbce+logit --lr 2e-3 --steps 200000 --min-steps 40000 $EDGE --run-prefix base_repro_c" ;;
    arm1_edge_c) echo "--data $B/data_arm1 $SHARP --loss softbce+edge --lr 2e-3 --steps 200000 --min-steps 40000 $EDGE --targets-desc ink9um_stride32_centrecrop --run-prefix arm1_edge_c" ;;
    *) echo "" ;;
  esac
}

for arm in "$@"; do
  RD=$B/runs_conv/$arm
  if [ -f $RD/.done ]; then log "$arm: done already"; continue; fi
  if [ "$arm" = "arm2c" ]; then
    # label fine-tune of the converged arm0c (ring buffer-zone supervision on S1/S5 ctl labels -> contaminates
    # S1/S5 eval; S4, PHerc0841, fragments stay clean). Waits for arm0c.
    while [ ! -f $B/runs_conv/arm0c/.done ]; do sleep 300; done
    for try in 1 2 3; do
      log "train $arm try $try (init arm0c best)"
      systemctl --user reset-failed conv-$arm-$try.scope 2>/dev/null
      systemd-run --user --scope -p MemoryMax=60G --unit=conv-$arm-$try -E PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        $PY $C/scripts/hires_ink/ink9um_arm2_finetune.py --subj $B/subj --init $B/runs_conv/arm0c/student_best.ckpt \
        --out $RD --src $C/src --steps 30000 --eval-every 1000 --patience 8 --crop 384 --lr 1e-4 >> $B/runs_conv/$arm.log 2>&1
      rc=$?; log "train $arm try $try rc=$rc"
      [ $rc -eq 0 ] && { touch $RD/.done; break; }
      log "ALERT $arm failed rc=$rc (try $try): $(grep -E 'Error|error|Killed' $B/runs_conv/$arm.log | tail -1)"
    done
    continue
  fi
  A=$(args_for $arm)
  if [ -z "$A" ]; then log "ALERT unknown arm $arm"; continue; fi
  for try in 1 2 3 4; do
    log "train $arm try $try: $A"
    systemctl --user reset-failed conv-$arm-$try.scope 2>/dev/null   # a failed unit of the same name blocks systemd-run
    systemd-run --user --scope -p MemoryMax=70G --unit=conv-$arm-$try -E PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      $PY $C/scripts/reader_v2_distill/rv2_distill.py --src $C/src --out $RD $CONV $A >> $B/runs_conv/$arm.log 2>&1
    rc=$?; log "train $arm try $try rc=$rc"
    if [ $rc -eq 0 ] && [ -f $RD/student_best.ckpt ]; then touch $RD/.done; break; fi
    log "ALERT $arm failed rc=$rc (try $try; relaunch resumes from $RD/resume.pt): $(grep -E 'Error|error|Killed|oom' $B/runs_conv/$arm.log | tail -1)"
    sleep 120
  done
  [ -f $RD/.done ] || { log "ALERT $arm GAVE UP after 4 tries"; continue; }
  log "eval $arm"
  systemd-run --user --scope -p MemoryMax=60G --unit=conv-eval-$arm -E PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY $C/scripts/reader_v2_distill/rv2_eval.py $B/subj --src $C/src --ckpt conv_$arm=$RD/student_best.ckpt \
    --ckpt conv_${arm}_last=$RD/student.ckpt \
    --ref teacher_rv2=rv2maps:$B/ref_rv2 --ref ink9um_student=arm:$B/ref_sharp:arm0 \
    --out $B/eval_conv/$arm.json --save-maps $B/maps_conv > $B/eval_conv/$arm.log 2>&1
  rc=$?; log "eval $arm rc=$rc"
  [ $rc -eq 0 ] || log "ALERT eval $arm rc=$rc: $(tail -1 $B/eval_conv/$arm.log)"
done
log "LANE DONE"
