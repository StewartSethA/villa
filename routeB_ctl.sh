#!/bin/bash
# Runtime control of a running box8 (re-read every ~10 s by the scheduler; no restart).
#   routeB_ctl.sh [--home DIR] status        allowed GPUs, jobs, budget, REPLAN, control events
#   routeB_ctl.sh [--home DIR] gpus 0,1,2    allow ONLY these GPUs (re-allows drained/killed named)
#                                            'gpus none' allows nothing
#   routeB_ctl.sh [--home DIR] drain N       finish GPU N's current fit, then stop using it
#   routeB_ctl.sh [--home DIR] kill N        SIGTERM GPU N's fit; interval re-queued and resumes
#                                            from its last autosave (never lost); GPU N stays out
#   routeB_ctl.sh [--home DIR] stop          graceful: no new launches, running fits finish
#   routeB_ctl.sh [--home DIR] pause | resume
# --home defaults to $ROUTEB_HOME, then ~/.routeb_home.
# The hard stop (kills running fits) is: touch <home>/box8/STOP
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HOME_DIR=${ROUTEB_HOME:-}
if [ "$1" = "--home" ]; then HOME_DIR=$2; shift 2; fi
if [ -z "$HOME_DIR" ] && [ -f "$HOME/.routeb_home" ]; then HOME_DIR=$(cat "$HOME/.routeb_home"); fi
[ -n "$HOME_DIR" ] || HOME_DIR=$HERE/routeB_work
C="$HOME_DIR/box8/control"
if [ ! -d "$HOME_DIR/box8" ]; then
  echo "routeB_ctl: no $HOME_DIR/box8 (is --home / ROUTEB_HOME right?)" >&2
  exit 2
fi
mkdir -p "$C"
need_n() {
  case "$1" in
    ''|*[!0-9]*) echo "routeB_ctl: GPU index required (got '$1')" >&2; exit 2 ;;
  esac
}
cmd=$1
shift
case "$cmd" in
  gpus)
    L=$1
    [ "$L" = none ] && L=""
    for g in $(echo "$L" | tr ',' ' '); do need_n "$g"; rm -f "$C/drain.$g"; done
    echo "$L" > "$C/gpus"
    echo "allowed GPUs -> '${L:-none}' (effective within ~10 s; fits on removed GPUs finish)" ;;
  drain)
    need_n "$1"
    echo draining > "$C/drain.$1"
    echo "GPU $1 will be drained after its current fit" ;;
  kill)
    need_n "$1"
    echo kill > "$C/kill.$1"
    echo "GPU $1: fit SIGTERMed and re-queued from its last autosave"
    echo "GPU $1 stays out until 'gpus' names it again" ;;
  stop) echo stop > "$C/STOP"; echo "graceful stop requested" ;;
  pause) echo pause > "$C/PAUSE"; echo "paused" ;;
  resume) echo resume > "$C/RESUME"; echo "resume requested (clears PAUSE/STOP)" ;;
  status)
    export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
    python3 -m routeB.watch --ctl-status --home "$HOME_DIR"
    Q=index,utilization.gpu,memory.used
    command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=$Q --format=csv,noheader ;;
  *) sed -n 2,14p "$0"; exit 2 ;;
esac
