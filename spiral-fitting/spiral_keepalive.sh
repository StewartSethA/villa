#!/bin/bash
# Auto-restart wrapper working around a real, unbounded GPU-memory growth bug
# in fit_spiral.py's training loop (confirmed: memory climbs steadily with
# iteration count on both a P40 and a V100, rather than plateauing after
# setup, eventually OOMing regardless of z-range size). Rather than chase the
# leak's root cause, this keeps the GPU continuously busy by resuming from
# the most recent autosave checkpoint every time the process dies.
#
# Usage: spiral_keepalive.sh <cuda_visible_devices> <run_tag> <out_dir> <cache_dir> <z_begin> <z_end> <extra_config_json>
set -uo pipefail

CVD="$1"; TAG="$2"; OUT_DIR="$3"; CACHE_DIR="$4"; ZB="$5"; ZE="$6"; EXTRA_JSON="$7"

mkdir -p "$OUT_DIR" "$CACHE_DIR"
LOG="$OUT_DIR/keepalive.log"
ATTEMPT=0

BASE_JSON="\"z_begin\":$ZB,\"z_end\":$ZE,\"optimizer_num_training_steps\":30000,\"input_use_tracks\":false,\"dense_spacing_mode\":\"grad_mag\",\"loss_weight_dense_spacing_density\":0.0,\"loss_weight_dense_normals\":0.0,\"loss_weight_dense_spacing\":0.0,\"model_flow_field_direct_lr\":false"
if [ -n "$EXTRA_JSON" ]; then
  CONFIG_JSON="{${BASE_JSON},${EXTRA_JSON}}"
else
  CONFIG_JSON="{${BASE_JSON}}"
fi
if ! echo "$CONFIG_JSON" | .venv/bin/python -c "import json,sys; json.loads(sys.stdin.read())" 2>>"$LOG"; then
  echo "[$(date -Is)] FATAL: FIT_SPIRAL_CONFIG_OVERRIDES is not valid JSON, aborting: $CONFIG_JSON" >> "$LOG"
  exit 1
fi

echo "[$(date -Is)] keepalive starting for $TAG (gpu $CVD, z=$ZB-$ZE)" >> "$LOG"

while true; do
  ATTEMPT=$((ATTEMPT + 1))
  CKPT=$(find "$OUT_DIR" -maxdepth 2 -name "checkpoint_fitted.ckpt" 2>/dev/null | head -1)
  if [ -n "$CKPT" ] && [ -f "$CKPT" ]; then
    echo "[$(date -Is)] attempt $ATTEMPT: resuming from $CKPT" >> "$LOG"
    export FIT_SPIRAL_RESUME_PATH="$CKPT"
  else
    echo "[$(date -Is)] attempt $ATTEMPT: starting fresh (no checkpoint yet)" >> "$LOG"
    unset FIT_SPIRAL_RESUME_PATH
  fi

  CUDA_VISIBLE_DEVICES="$CVD" \
  FIT_SPIRAL_OUT_DIR="$OUT_DIR" \
  FIT_SPIRAL_CACHE_DIR="$CACHE_DIR" \
  FIT_SPIRAL_RUN_TAG="$TAG" \
  FIT_SPIRAL_AUTOSAVE_INTERVAL=50 \
  FIT_SPIRAL_SETUP_TIMING=1 \
  FIT_SPIRAL_PATCH_LOAD_WORKERS="${FIT_SPIRAL_PATCH_LOAD_WORKERS:-5}" \
  WANDB_MODE=disabled \
  FIT_SPIRAL_CONFIG_OVERRIDES="$CONFIG_JSON" \
  FIT_SPIRAL_TRITON=0 \
  .venv/bin/torchrun --nproc-per-node=1 fit_spiral.py \
    --dataset /mnt/raid10T/spiral_datasets/PHercParis4 \
    --cache "$CACHE_DIR" \
    >> "$OUT_DIR/run_$ATTEMPT.log" 2>&1
  RC=$?

  LAST_LINE=$(tail -1 "$OUT_DIR/run_$ATTEMPT.log" 2>/dev/null)
  echo "[$(date -Is)] attempt $ATTEMPT exited rc=$RC last_line=\"$LAST_LINE\"" >> "$LOG"

  if [ "$RC" -eq 0 ]; then
    echo "[$(date -Is)] completed successfully, stopping keepalive" >> "$LOG"
    break
  fi

  sleep 5
done
