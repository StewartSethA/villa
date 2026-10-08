#!/usr/bin/env bash
# box_bootstrap.sh -- ONE command on a fresh rented GPU box (Ubuntu 20.04/22.04/24.04, Linux x86_64, NVIDIA driver present):
#   curl -fsSL https://raw.githubusercontent.com/StewartSethA/villa/routeAB-deploy-v3/box_bootstrap.sh | bash -s -- shakedown
#   curl -fsSL https://raw.githubusercontent.com/StewartSethA/villa/routeAB-deploy-v3/box_bootstrap.sh | bash -s -- run --scrolls PHerc0211,PHerc0125
# It (1) installs every OS dependency (git, rsync, curl, build tools, g++>=13, tmux), (2) clones this branch, (3) runs the
# Route B pushbutton (python env, native build and all assets are fetched by routeB_run.sh itself), detached, logging to
# $DEST/boxrun.log so an ssh disconnect does not kill it.  Modes:  shakedown | run <routeB_run.sh args...> | status | update
# Env overrides: ROUTEB_REPO ROUTEB_BRANCH ROUTEB_DIR  BUDGET_* (see BOX8_NOTES.md; defaults are for the 8xA100 box)  BUDGET_MAX_RUN_HOURS (10)
# No `set -e` (repo convention): every step is checked and fails LOUD.
REPO=${ROUTEB_REPO:-https://github.com/StewartSethA/villa.git}
BRANCH=${ROUTEB_BRANCH:-routeAB-deploy-v3}
DEST=${ROUTEB_DIR:-$HOME/routeAB}
export BUDGET_MAX_RUN_HOURS=${BUDGET_MAX_RUN_HOURS:-10}   # box8 prices compute $4.276/h + disk $0.009/16GB/h itself (effective ~$4.80/h)
say(){ echo "[box_bootstrap $(date +%H:%M:%S)] $*"; }
fail(){ echo "[box_bootstrap FAIL] $1" >&2; exit 2; }
MODE=${1:-shakedown}; [ $# -gt 0 ] && shift
[ "$(uname -s)" = Linux ] && [ "$(uname -m)" = x86_64 ] || fail "needs Linux x86_64 (got $(uname -sm))"
SUDO=""; [ "$(id -u)" != 0 ] && { command -v sudo >/dev/null && SUDO=sudo || fail "not root and no sudo"; }

if [ "$MODE" = status ]; then
  [ -f "$DEST/boxrun.log" ] && tail -40 "$DEST/boxrun.log" || echo "no log yet at $DEST/boxrun.log"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null
  df -h "$DEST" | tail -1; uptime; exit 0
fi

command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 || fail "no working NVIDIA driver (nvidia-smi failed)"
say "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tr '\n' ';')  cores=$(nproc)  RAM=$(free -g | awk '/Mem:/{print $2}')G  free disk=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)G"

say "installing OS packages"
export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update -y >/tmp/box_apt.log 2>&1 || say "apt-get update warned (see /tmp/box_apt.log)"
$SUDO apt-get install -y --no-install-recommends git rsync curl wget ca-certificates build-essential pkg-config cmake \
    tmux util-linux xz-utils unzip zstd python3 python3-venv software-properties-common gnupg libgl1 libglib2.0-0 >>/tmp/box_apt.log 2>&1 \
    || { tail -15 /tmp/box_apt.log >&2; fail "apt-get install failed"; }
gxx_major(){ g++ -dumpversion 2>/dev/null | cut -d. -f1; }
if [ "${GV:-$(gxx_major)}" -lt 13 ] 2>/dev/null || ! command -v g++ >/dev/null; then
  say "g++ $(gxx_major) too old for C++23: installing g++-13"
  $SUDO apt-get install -y g++-13 >>/tmp/box_apt.log 2>&1 || {
    $SUDO add-apt-repository -y ppa:ubuntu-toolchain-r/test >>/tmp/box_apt.log 2>&1 && $SUDO apt-get update -y >>/tmp/box_apt.log 2>&1 \
      && $SUDO apt-get install -y g++-13 gcc-13 >>/tmp/box_apt.log 2>&1; }
  if command -v g++-13 >/dev/null; then
    $SUDO update-alternatives --install /usr/bin/g++ g++ "$(command -v g++-13)" 130 >/dev/null 2>&1
    $SUDO update-alternatives --install /usr/bin/gcc gcc "$(command -v gcc-13)" 130 >/dev/null 2>&1
  fi
fi
[ "$(gxx_major)" -ge 13 ] 2>/dev/null || { tail -15 /tmp/box_apt.log >&2; fail "g++ >= 13 not available (have $(gxx_major))"; }
say "g++ $(g++ -dumpversion) OK"

if [ -d "$DEST/.git" ]; then
  say "updating $DEST ($BRANCH)"; git -C "$DEST" fetch -q --depth 1 origin "$BRANCH" && git -C "$DEST" checkout -q -B "$BRANCH" FETCH_HEAD || fail "git update failed"
else
  say "cloning $REPO ($BRANCH) -> $DEST"; git clone -q --depth 1 -b "$BRANCH" "$REPO" "$DEST" || fail "git clone failed (is the branch public?)"
fi
cd "$DEST" || fail "cd $DEST"
chmod +x routeA_run.sh routeB_run.sh routeB_shakedown.sh routeB_pull.sh smoke_test.sh 2>/dev/null
[ "$MODE" = update ] && { say "updated to $(git rev-parse --short HEAD)"; exit 0; }

case "$MODE" in
  shakedown) CMD="./routeB_shakedown.sh $*" ;;
  run)       CMD="./routeB_run.sh --mode box8 $*" ;;
  *)         fail "unknown mode '$MODE' (shakedown | run ARGS | status | update)" ;;
esac
say "starting detached: $CMD   (commit $(git rev-parse --short HEAD); max ${BUDGET_MAX_RUN_HOURS} h)"
nohup setsid bash -c "cd '$DEST' && $CMD" >"$DEST/boxrun.log" 2>&1 < /dev/null &
sleep 3
say "running. Watch:   tail -f $DEST/boxrun.log     status: bash $DEST/box_bootstrap.sh status"
say "Pull results from YOUR machine (resumable, checksummed):  see $DEST/routeB_pull.sh --help   (payloads under \$ROUTEB_HOME)"
