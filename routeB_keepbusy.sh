#!/usr/bin/env bash
# routeB_keepbusy.sh -- keep the rented box busy: when the MAIN box8 run has nothing queued and a GPU sits idle, start an ALTERNATIVE pass at a NARROWER stripe
# height on the scrolls the main run has completed (user 2026-10-08: "The box should automatically start narrower stripes if idle").  Reuses the tested box8 scheduler:
# each pass is an ordinary `routeB_run.sh --mode box8` in its OWN home (<main>_alt_h<H>) with the main run's environment symlinked in, so nothing in the main run is touched.
# Mutual exclusion with the main run is box8's own foreign-VRAM check: a GPU that holds another process's fit is skipped (and the main run skips alt's).
#   env: ROUTEB_HOME (main, default /workspace/routeB)  KB_HEIGHTS ("4500 2800")  KB_IDLE_S (180)  KB_POLL (30)  KB_MAX_HOURS (8, per pass)  KB_SOFT/KB_HARD (170/190)
#        KB_DEADLINE_H (24: no new pass after this many hours since BUDGET_BOX_START)  KB_SCROLLS (default: every scroll the main STATUS.json calls complete)  KB_DRY=1 KB_ONCE=1
#   stop:  touch <main>_alt/keepbusy.STOP      status:  bash routeB_keepbusy.sh status
# No `set -e` (repo convention): each step is checked.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MAIN=${ROUTEB_HOME:-/workspace/routeB}
ALT=${KB_HOME:-${MAIN}_alt}
HEIGHTS=${KB_HEIGHTS:-"4500 2800"}
IDLE_S=${KB_IDLE_S:-180}
POLL=${KB_POLL:-30}
MAXH=${KB_MAX_HOURS:-8}
SOFT=${KB_SOFT:-170}
HARD=${KB_HARD:-190}
DEADLINE_H=${KB_DEADLINE_H:-24}
START=${BUDGET_BOX_START:-$(date +%s)}
log() { echo "[keepbusy $(date -u +%H:%M:%S)] $*"; }
mkdir -p "$ALT" 2>/dev/null

main_state() {   # prints: pending running complete_scrolls_csv   (pending counts 'pending'+'deferred'? no: only 'pending')
  python3 - "$MAIN/out/STATUS.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("? ? -"); raise SystemExit
j = d.get("jobs") or {}
sc = ",".join(sorted(k for k, v in (d.get("scrolls") or {}).items() if v == "complete")) or "-"
print(j.get("pending", 0), j.get("running", 0), sc)
PY
}
idle_gpus() {    # number of GPUs holding < 1500 MiB (no fit on them)
  if [ -n "$KB_FAKE_IDLE" ]; then echo "$KB_FAKE_IDLE"; return; fi
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '$1 < 1500 {n++} END {print n + 0}'
}

if [ "$1" = status ]; then
  read -r P R SC <<< "$(main_state)"
  log "main: pending=$P running=$R complete=$SC | idle GPUs: $(idle_gpus) | alt homes: $(ls -d ${ALT}_h* 2>/dev/null | tr '\n' ' ')"
  for h in ${ALT}_h*; do [ -f "$h/out/STATUS.json" ] && log "$h: $(python3 -c "import json;d=json.load(open('$h/out/STATUS.json'));print(d['jobs'],d['scrolls'])" 2>/dev/null | cut -c1-200)"; done
  exit 0
fi

log "armed: main=$MAIN heights=[$HEIGHTS] idle>=${IDLE_S}s caps soft \$$SOFT hard \$$HARD, <= ${MAXH} h per pass, no new pass after ${DEADLINE_H} h from box start"
for H in $HEIGHTS; do
  HOMEH="${ALT}_h${H}"
  idle_for=0
  while true; do
    [ -e "$ALT/keepbusy.STOP" ] && { log "STOP file: leaving"; exit 0; }
    el=$(( $(date +%s) - START ))
    if [ "$el" -gt $(( DEADLINE_H * 3600 )) ]; then log "deadline ${DEADLINE_H} h since box start passed: no new pass"; exit 0; fi
    read -r P R SC <<< "$(main_state)"
    IDLE=$(idle_gpus)
    isidle=0
    if [ "$P" = "0" ] && [ "$IDLE" -ge 1 ] 2>/dev/null; then
      isidle=1
      idle_for=$(( idle_for + POLL ))
      [ $(( idle_for % 120 )) -lt "$POLL" ] && log "main has nothing queued and $IDLE GPU(s) idle for ${idle_for}s (need ${IDLE_S}s)"
    else
      [ "$idle_for" -gt 0 ] && log "idle streak broken (pending=$P idle=$IDLE)"
      idle_for=0
    fi
    [ "$isidle" = 1 ] && [ "$idle_for" -ge "$IDLE_S" ] && break
    [ -n "$KB_ONCE" ] && { log "not idle (pending=$P idle=$IDLE): nothing to do"; exit 0; }
    sleep "$POLL"
  done
  SCR=${KB_SCROLLS:-$SC}
  [ -z "$SCR" ] || [ "$SCR" = "-" ] && { log "no completed scrolls to re-run: nothing to do"; exit 0; }
  mkdir -p "$HOMEH"
  for d in env_cu126 env_cu128 env_cu129 env_cu128.tools env_cu126.tools env_cu129.tools; do
    [ -e "$MAIN/$d" ] && [ ! -e "$HOMEH/$d" ] && ln -s "$MAIN/$d" "$HOMEH/$d"
  done
  CMD="./routeB_run.sh --mode box8 --scrolls $SCR --order spt --max-height $H --no-tail-split --no-routea --free-inputs-on done --soft $SOFT --hard $HARD --max-run-hours $MAXH"
  [ -n "$KB_SPEED" ] && CMD="$CMD --gpu-speed $KB_SPEED"
  log "LAUNCH alternative pass: height $H on [$SCR], home $HOMEH"
  log "  cmd: ROUTEB_HOME=$HOMEH $CMD"
  if [ -n "$KB_DRY" ]; then log "(dry run: not launching)"; continue; fi
  ( cd "$HERE" && env ROUTEB_HOME="$HOMEH" BUDGET_BOX_START="$START" $CMD > "$HOMEH/keepbusy_pass.log" 2>&1 )
  log "pass at height $H ended (rc=$?); log $HOMEH/keepbusy_pass.log; units under $HOMEH/out/ (PULL THEM: --remote-home $HOMEH)"
done
log "all heights done"
