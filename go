#!/usr/bin/env bash
# box_bootstrap.sh (also served as `go`): ONE command on a fresh rented GPU box
# (Ubuntu 20.04-24.04, x86_64, NVIDIA driver).
#   curl -fsSL https://raw.githubusercontent.com/StewartSethA/villa/routeAB-deploy-v5/go|bash -s run
# Order (nothing that costs money-time runs before the link is known):
#   0. LINK CHECK: parallel range reads from the real data host and a CDN (~13 s), verdict box, gate
#   1. OS packages (git rsync curl tmux g++>=13 ...)   2. clone this branch
#   3. start the run DETACHED in tmux session `routeb` (env passed inside the tmux command)
#   4. attach the live dashboard (routeB_watch.sh); Ctrl-C detaches only the watcher
# Modes: run [routeB_run.sh args] | watch | snapshot | attach | status | update | shakedown
# Flags understood here (all are also passed on to the run): --accept-slow-link  --min-link-mb-s N
# Env: ROUTEB_HOME (default /workspace/routeB or ~/routeB)  ROUTEB_BRANCH  ROUTEB_REPO  ROUTEB_DIR
#      ROUTEB_PLAN_GB (GB the link check assumes, default 40)  ROUTEB_NO_WATCH=1  BUDGET_BOX_START
# No set -e (repo convention): every step is checked and fails LOUD.
REPO=${ROUTEB_REPO:-https://github.com/StewartSethA/villa.git}
BRANCH=${ROUTEB_BRANCH:-routeAB-deploy-v5}
DEST=${ROUTEB_DIR:-$HOME/routeAB}
T_START=$(date +%s)
DATA_BASE=https://dl.ash2txt.org/datasets/spiral_datasets/PHerc0826/20250821151701/tracks
DATA_FILE=PHerc0826_20250821151701_surface_m7_L0_th0.2.dbm
DATA_URL=${ROUTEB_DATA_URL:-$DATA_BASE/$DATA_FILE}
CDN_URL=${ROUTEB_CDN_URL:-https://speed.cloudflare.com/__down?bytes=16000000}
MIN_MBS=20
ACCEPT=0
PREV=""
for A in "$@"; do
  [ "$A" = "--accept-slow-link" ] && ACCEPT=1
  [ "$PREV" = "--min-link-mb-s" ] && MIN_MBS=$A
  PREV=$A
done
say() { echo "[box_bootstrap $(date +%H:%M:%S)] $*"; }
fail() { echo "[box_bootstrap FAIL] $1" >&2; exit "${2:-2}"; }
MODE=${1:-run}
[ $# -gt 0 ] && shift

# ---------------------------------------------------------------- step 0: the link check
MBS_AWK='{s=0; for(i=1;i<=NF-2;i++) s+=$i; d=$NF-$(NF-1);
  if(d<0.5) d=0.5; printf "%.1f", s/1e6/d}'
probe_range() {
  local url=$1 secs=$2 i lo t0 t1 tot
  t0=$(date +%s.%N)
  tot=$(for i in 0 1 2 3 4 5 6 7; do
    lo=$((i * 8388608))
    curl -s -m "$secs" -r "$lo-$((lo + 8388607))" -o /dev/null -w '%{size_download}\n' "$url" &
  done; wait)
  t1=$(date +%s.%N)
  echo $tot $t0 $t1 | awk "$MBS_AWK"
}
probe_cdn() {
  local url=$1 secs=$2 i t0 t1 tot
  t0=$(date +%s.%N)
  tot=$(for i in 0 1 2 3 4 5 6 7; do
    curl -s -m "$secs" -o /dev/null -w '%{size_download}\n' "$url" &
  done; wait)
  t1=$(date +%s.%N)
  echo $tot $t0 $t1 | awk "$MBS_AWK"
}
row() { printf '| %-94s |\n' "$1"; }
link_check() {
  local d c hours usd verdict diag plan_gb=${ROUTEB_PLAN_GB:-40}
  command -v curl >/dev/null || fail "curl is required" 2
  say "LINK CHECK first (about 13 s): data host, then a CDN"
  d=$(probe_range "$DATA_URL" 8)
  c=$(probe_cdn "$CDN_URL" 5)
  hours=$(awk -v g="$plan_gb" -v m="$d" 'BEGIN{ if(m<=0) m=0.01; printf "%.2f", g*1000/m/3600 }')
  usd=$(awk -v h="$hours" 'BEGIN{ printf "%.2f", h*4.80 }')
  verdict=$(awk -v m="$d" -v mn="$MIN_MBS" -v h="$hours" \
    'BEGIN{ if(m<mn) print "BAD"; else if(m<2*mn || h>1.5) print "MARGINAL"; else print "GOOD" }')
  diag=$(awk -v d="$d" -v c="$c" -v mn="$MIN_MBS" 'BEGIN{
    if(d>=mn) print "link and source look fine";
    else if(c>=3*d) print "SOURCE (dl.ash2txt.org) slow: CDN is much faster, box link fine";
    else if(c<mn) print "BOX LINK slow: both independent hosts are slow";
    else print "data host slow, CDN not much faster: link or source, re-probe later" }')
  echo "+$(printf '%0.s-' $(seq 1 96))+"
  row "LINK CHECK   VERDICT: $verdict   (gate: data host >= $MIN_MBS MB/s)"
  row "data host  dl.ash2txt.org   : $d MB/s   (8 parallel range reads, 8 s)"
  row "2nd host   speed.cloudflare : $c MB/s   (8 parallel reads, 5 s)"
  row "diagnosis: $diag"
  row "~$plan_gb GB to fetch -> $hours h of pure transfer; box time idling for it: about \$$usd"
  echo "+$(printf '%0.s-' $(seq 1 96))+"
  if [ "$verdict" = BAD ]; then
    if [ "$ACCEPT" = 1 ]; then
      say "SLOW LINK ACCEPTED (--accept-slow-link): continuing; the planner will shrink --gpus"
    else
      echo "*** DESTROY THIS BOX, or re-run with --accept-slow-link. Nothing was installed. ***" >&2
      exit 5
    fi
  fi
}

# ---------------------------------------------------------------- helpers
gpu_step() {
  local q=index,name,memory.total,memory.used,driver_version,compute_cap
  command -v nvidia-smi >/dev/null || fail "no nvidia-smi: no working NVIDIA driver"
  nvidia-smi -L >/dev/null 2>&1 || fail "nvidia-smi cannot talk to the driver"
  say "GPUs (index, name, VRAM, used, driver, compute capability):"
  nvidia-smi --query-gpu=$q --format=csv,noheader | sed 's/^/    /'
  local cap drv foreign tc
  cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -rn | head -1)
  drv=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)
  foreign=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | awk '$1+0 > 1500 {n++} END{print n+0}')
  tc=cu126
  case "$cap" in 12.*|13.*) tc=cu128 ;; esac
  case "$drv" in 12.[89]|12.[1-9][0-9]|13.*) tc=cu128 ;; esac
  say "torch wheel index chosen from the hardware: $tc (compute cap $cap, driver CUDA $drv)"
  if [ "$tc" != cu126 ]; then
    say "$tc is UNVALIDATED numerically: spot-check a stripe on a known GPU (D6)"
  fi
  if [ "$foreign" -gt 0 ]; then
    say "WARNING: $foreign GPU(s) hold >1500 MiB (foreign process): they are skipped"
  fi
}
box_home() {
  if [ -n "$ROUTEB_HOME" ]; then echo "$ROUTEB_HOME"; return; fi
  if [ -d /workspace ] && [ -w /workspace ]; then
    echo /workspace/routeB
  else
    echo "$HOME/routeB"
  fi
}
box_start() {
  local b
  if [ -n "$BUDGET_BOX_START" ]; then echo "$BUDGET_BOX_START"; return; fi
  b=$(stat -c %Y /proc/1 2>/dev/null || echo "$T_START")
  if [ "$b" -lt "$T_START" ] && [ $((T_START - b)) -lt 172800 ]; then
    echo "$b"
  else
    echo "$T_START"
  fi
}
need_root() {
  SUDO=""
  if [ "$(id -u)" != 0 ]; then
    command -v sudo >/dev/null && SUDO=sudo || fail "not root and no sudo"
  fi
}
gxx_major() { g++ -dumpversion 2>/dev/null | cut -d. -f1; }
install_os() {
  export DEBIAN_FRONTEND=noninteractive
  $SUDO apt-get update -y >/tmp/box_apt.log 2>&1 || say "apt-get update warned (/tmp/box_apt.log)"
  $SUDO apt-get install -y --no-install-recommends git rsync curl wget ca-certificates \
    build-essential pkg-config cmake tmux util-linux xz-utils unzip zstd python3 python3-venv \
    software-properties-common gnupg libgl1 libglib2.0-0 >>/tmp/box_apt.log 2>&1 \
    || { tail -15 /tmp/box_apt.log >&2; fail "apt-get install failed"; }
  if ! command -v g++ >/dev/null || [ "$(gxx_major)" -lt 13 ] 2>/dev/null; then
    say "g++ $(gxx_major) is too old for C++23: installing g++-13"
    $SUDO apt-get install -y g++-13 >>/tmp/box_apt.log 2>&1 || {
      $SUDO add-apt-repository -y ppa:ubuntu-toolchain-r/test >>/tmp/box_apt.log 2>&1
      $SUDO apt-get update -y >>/tmp/box_apt.log 2>&1
      $SUDO apt-get install -y g++-13 gcc-13 >>/tmp/box_apt.log 2>&1; }
    if command -v g++-13 >/dev/null; then
      GX=$(command -v g++-13)
      GC=$(command -v gcc-13)
      $SUDO update-alternatives --install /usr/bin/g++ g++ "$GX" 130 >/dev/null 2>&1
      $SUDO update-alternatives --install /usr/bin/gcc gcc "$GC" 130 >/dev/null 2>&1
    fi
  fi
  if ! [ "$(gxx_major)" -ge 13 ] 2>/dev/null; then
    tail -15 /tmp/box_apt.log >&2
    fail "g++ >= 13 missing"
  fi
  say "g++ $(g++ -dumpversion) OK"
}
fetch_tree() {
  if [ -d "$DEST/.git" ]; then
    say "updating $DEST ($BRANCH)"
    git -C "$DEST" fetch -q --depth 1 origin "$BRANCH" || fail "git fetch failed"
    git -C "$DEST" checkout -q -B "$BRANCH" FETCH_HEAD || fail "git checkout failed"
  else
    say "cloning $REPO ($BRANCH) -> $DEST"
    git clone -q --depth 1 -b "$BRANCH" "$REPO" "$DEST" || fail "git clone failed (public branch?)"
  fi
  chmod +x "$DEST"/*.sh 2>/dev/null
}
start_run() {
  local H T Q CMD MIN_BEFORE
  H=$(box_home)
  T=$(box_start)
  mkdir -p "$H" || fail "cannot create $H"
  echo "$H" > "$HOME/.routeb_home"
  say "ROUTEB_HOME = $H"
  MIN_BEFORE=$(( (T_START - T) / 60 ))
  say "budget clock starts at $(date -d "@$T" +%H:%M:%S) ($MIN_BEFORE min before this script)"
  if tmux has-session -t routeb 2>/dev/null; then
    say "tmux session routeb already exists: NOT starting a second run"
    say "(raw console: tmux attach -t routeb)"
    return 0
  fi
  Q=$(printf '%q ' "$@")
  CMD="cd $DEST && ./routeB_run.sh --mode box8 $Q 2>&1 | tee -a $H/box8.log"
  tmux new-session -d -s routeb "env ROUTEB_HOME=$H BUDGET_BOX_START=$T bash -c '$CMD'" \
    || fail "tmux could not start the session"
  say "run started detached in tmux session routeb"
  say "commit $(git -C "$DEST" rev-parse --short HEAD)"
}
watch_now() {
  local H
  H=$(box_home)
  if [ -t 1 ] && [ -z "$ROUTEB_NO_WATCH" ]; then
    say "attaching the dashboard: Ctrl-C detaches only the watcher"
    say "reattach any time: bash $DEST/routeB_watch.sh"
    ROUTEB_HOME=$H exec bash "$DEST/routeB_watch.sh"
  fi
  ROUTEB_HOME=$H bash "$DEST/routeB_watch.sh" --once
}

# ---------------------------------------------------------------- modes
[ "$(uname -s)" = Linux ] && [ "$(uname -m)" = x86_64 ] || fail "needs Linux x86_64"
case "$MODE" in
  linkcheck) link_check; exit 0 ;;
  watch) ROUTEB_HOME=$(box_home) exec bash "$DEST/routeB_watch.sh" "$@" ;;
  snapshot) ROUTEB_HOME=$(box_home) exec bash "$DEST/routeB_watch.sh" --once ;;
  attach) exec tmux attach -t routeb ;;
  status) ROUTEB_HOME=$(box_home) exec bash "$DEST/routeB_watch.sh" --once ;;
esac
link_check
gpu_step
need_root
say "cores=$(nproc) RAM=$(free -g | awk '/Mem:/{print $2}')G"
say "installing OS packages"
install_os
fetch_tree
case "$MODE" in
  update) say "updated to $(git -C "$DEST" rev-parse --short HEAD)"; exit 0 ;;
  shakedown) cd "$DEST" && exec ./routeB_shakedown.sh "$@" ;;
  run) start_run "$@"; watch_now ;;
  *) fail "unknown mode '$MODE' (run | watch | snapshot | attach | status | update | shakedown)" ;;
esac
