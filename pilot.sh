#!/usr/bin/env bash
# routeAB PILOT: a ~5-10 minute CUT / NO-CUT test for a rented GPU box. Run it first, before anything else is installed.
#   curl -fsSL https://raw.githubusercontent.com/StewartSethA/villa/routeAB-pilot/pilot.sh | bash
# What it does, in this order (it never installs anything big before the link and GPUs are known):
#   1. GPUs: names, VRAM, driver, compute capability (nvidia-smi).  A foreign process holding VRAM is flagged.
#   2. LINK, started immediately: 8 parallel range reads of a real tracks file from the data host, plus a second independent host.
#   3. TORCH, flexible (not pinned): the CUDA wheel is chosen from the hardware (cu128 for Blackwell / driver CUDA >= 12.8, else cu126), installed
#      in a throwaway venv, while the link test result is already on screen.
#   4. KERNEL SMOKE TEST + SPEED on EVERY GPU in parallel: real matmul (fp32 strict and tf32), free VRAM, arch list, memory-bandwidth copy.
#   5. VERDICT: GO or CUT with the reasons, the longest stripe each card can hold (VRAM model from the planner), and a --gpu-speed hint.
# No set -e (one failing step must not hide the rest).  Env: PILOT_DIR (default $HOME/pilot), PILOT_MIN_MBS (default 30), PILOT_SKIP_TORCH=1.
W=${PILOT_DIR:-$HOME/pilot}
MIN=${PILOT_MIN_MBS:-30}
mkdir -p "$W/dl" 2>/dev/null
cd "$W" || exit 1
T0=$(date +%s)
say() { echo "[$(date +%H:%M:%S) +$(( $(date +%s) - T0 ))s] $*"; }
bad=0
why=""

say "=== 1. GPUs ==="
if ! command -v nvidia-smi >/dev/null 2>&1; then say "NO nvidia-smi: no usable GPU driver"; echo "VERDICT: CUT (no GPU driver)"; exit 2; fi
nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version,compute_cap --format=csv,noheader 2>/dev/null | tee gpus.csv
if [ ! -s gpus.csv ]; then nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv,noheader | tee gpus.csv; fi
NG=$(wc -l < gpus.csv)
CAP=$(head -1 gpus.csv | awk -F', ' '{print $6}')
DRV_CUDA=$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: *[0-9.]*' | head -1 | grep -o '[0-9.]*$')
say "GPUs: $NG   compute_cap(first): ${CAP:-unknown}   driver CUDA: ${DRV_CUDA:-unknown}"
FOREIGN=$(awk -F', ' '{gsub(/ MiB/,"",$4); if ($4+0 > 1500) n++} END{print n+0}' gpus.csv)
[ "$FOREIGN" -gt 0 ] && { say "WARNING: $FOREIGN GPU(s) already hold >1500 MiB (a foreign process, e.g. llama-server)"; }

say "=== 1b. CPU / RAM / DISK ==="
CPUMODEL=$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | sed 's/.*: *//')
NPROC=$(nproc 2>/dev/null || echo 1)
PHYS=$(lscpu -p=CORE,SOCKET 2>/dev/null | grep -v '^#' | sort -u | wc -l)
CGCPU=""
if [ -r /sys/fs/cgroup/cpu.max ]; then CGCPU=$(awk '{ if ($1=="max") print "none"; else printf "%.1f", $1/$2 }' /sys/fs/cgroup/cpu.max); fi
RAMGB=$(awk '/MemTotal/{printf "%d", $2/1048576}' /proc/meminfo)
RAMAV=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
CGMEM=""
if [ -r /sys/fs/cgroup/memory.max ]; then CGMEM=$(awk '{ if ($1=="max") print "none"; else printf "%d", $1/1073741824 }' /sys/fs/cgroup/memory.max); fi
DISKDIR=${ROUTEB_HOME:-$W}
mkdir -p "$DISKDIR" 2>/dev/null
DFLINE=$(df -P -BG "$DISKDIR" 2>/dev/null | tail -1)
DISKFREE=$(echo "$DFLINE" | awk '{gsub(/G/,"",$4); print $4+0}')
DISKTOT=$(echo "$DFLINE" | awk '{gsub(/G/,"",$2); print $2+0}')
say "CPU: ${CPUMODEL:-unknown}; usable threads (nproc) $NPROC; physical cores visible ~${PHYS:-?}; container CPU limit: ${CGCPU:-n/a}"
say "RAM: ${RAMGB} GiB total, ${RAMAV} GiB available now; container memory limit: ${CGMEM:-n/a} GiB"
say "DISK: $DISKDIR is on $(echo "$DFLINE" | awk '{print $1}'): ${DISKTOT} GB total, ${DISKFREE} GB free"
EFFCPU=$NPROC
if [ -n "$CGCPU" ] && [ "$CGCPU" != none ]; then EFFCPU=$(awk -v a="$CGCPU" -v b="$NPROC" 'BEGIN{printf "%d", (a<b?a:b)}'); fi
EFFRAM=$RAMGB
if [ -n "$CGMEM" ] && [ "$CGMEM" != none ]; then EFFRAM=$(awk -v a="$CGMEM" -v b="$RAMGB" 'BEGIN{printf "%d", (a<b?a:b)}'); fi
NEED_RAM=$(( 30 + 40 * NG ))
NEED_CPU=$(( 2 * NG + 8 ))
NEED_DISK=${PILOT_MIN_DISK_GB:-400}
RASLOTS=$(awk -v c="$EFFCPU" -v g="$NG" -v r="$EFFRAM" 'BEGIN{a=c-2*g-4; b=(r-30-g*40)/6; m=(a<b?a:b); if (m<0) m=0; printf "%d", m}')
say "needs for $NG GPU(s): RAM >= ${NEED_RAM} GiB (30 + 40/fit), threads >= ${NEED_CPU}, free disk >= ${NEED_DISK} GB (plan peak ~330 GB for 5 scrolls)"
say "Route A slots the planner would pick: min(threads - 2 x GPUs - 4 = $(( EFFCPU - 2*NG - 4 )), RAM-bound) = $RASLOTS"
if [ "$EFFRAM" -lt "$NEED_RAM" ]; then bad=1; why="$why RAM ${EFFRAM} GiB < ${NEED_RAM};"; fi
if [ "$EFFCPU" -lt "$NEED_CPU" ]; then bad=1; why="$why threads $EFFCPU < ${NEED_CPU};"; fi
if [ "$DISKFREE" -lt "$NEED_DISK" ]; then bad=1; why="$why free disk ${DISKFREE} GB < ${NEED_DISK};"; fi

say "=== 2. LINK (started now) ==="
URL=https://dl.ash2txt.org/datasets/spiral_datasets/PHerc0191/20250821151635/tracks/PHerc0191_20250821151635_surface_m7_L0_th0.2.dbm
URL2=https://github.com/StewartSethA/villa/releases/download/routeA-kit-94be1eae/routeA-kit-94be1eae.tar.xz
probe() { # streams, bytes per stream -> prints MB/s (aggregate bytes / wall seconds)
  rm -f dl/p* 2>/dev/null
  local n=$1 len=$2 t1 t2 b i S E
  t1=$(date +%s.%N)
  for i in $(seq 0 $(( n - 1 ))); do
    S=$(( i * 120000000 )); E=$(( S + len - 1 ))
    curl -s -r "$S-$E" --max-time 20 -o "dl/p$i" "$URL" &
  done
  wait
  t2=$(date +%s.%N)
  b=$(cat dl/p* 2>/dev/null | wc -c)
  rm -f dl/p* 2>/dev/null
  awk -v b="$b" -v a="$t1" -v c="$t2" 'BEGIN{d=c-a; if (d<=0) d=1; printf "%.1f", b/1e6/d}'
}
M8=$(probe 8 100000000)
say "data host (dl.ash2txt.org), 8 parallel ranges: $M8 MB/s"
M64=$(probe 64 40000000)
say "data host (dl.ash2txt.org), 64 parallel ranges (the real fetcher uses 128 connections): $M64 MB/s"
MBS=$(awk -v a="$M8" -v b="$M64" 'BEGIN{printf "%.1f", (a>b?a:b)}')
B2=$(curl -sL -o /dev/null -w "%{speed_download}" --max-time 12 "$URL2")  # one stream: informational only (github is slow on many routes)
MBS2=$(awk -v b="$B2" 'BEGIN{printf "%.1f", b/1e6}')
say "second host (github release, 1 stream): $MBS2 MB/s"
if awk -v m="$MBS" -v min="$MIN" 'BEGIN{exit !(m+0 < min+0)}'; then bad=1; why="$why link $MBS MB/s < $MIN;"; fi

if [ "${PILOT_SKIP_TORCH:-0}" = 1 ]; then say "PILOT_SKIP_TORCH=1: skipping torch + GPU tests"; else
say "=== 3. TORCH (flexible, throwaway venv) ==="
CU=cu126
case "$CAP" in 12.*|13.*) CU=cu128 ;; esac
case "$DRV_CUDA" in 12.[89]|12.[1-9][0-9]|13.*) CU=cu128 ;; esac
say "wheel index chosen from hardware: $CU (cap ${CAP:-?}, driver CUDA ${DRV_CUDA:-?})"
PY=python3
$PY -m venv venv >/dev/null 2>&1 || { say "python3 -m venv failed; trying virtualenv/uv"; command -v uv >/dev/null 2>&1 && uv venv venv >/dev/null 2>&1; }
if [ -x venv/bin/python ]; then PIP="venv/bin/python -m pip"; command -v uv >/dev/null 2>&1 && PIP="uv pip install --python venv/bin/python"; else PIP="$PY -m pip"; fi
T3=$(date +%s)
IDX=https://download.pytorch.org/whl/$CU
tryinst() { say "torch install attempt: $*"; "$@" >> torch_install.log 2>&1; venv/bin/python -c "import torch" >/dev/null 2>&1; }
: > torch_install.log
OKT=0
if command -v uv >/dev/null 2>&1; then
  tryinst uv pip install --python venv/bin/python torch --index-url "$IDX" && OKT=1
  [ "$OKT" = 0 ] && tryinst uv pip install --native-tls --python venv/bin/python torch --index-url "$IDX" && OKT=1
  [ "$OKT" = 0 ] && tryinst uv pip install --system-certs --python venv/bin/python torch --index-url "$IDX" && OKT=1
fi
[ "$OKT" = 0 ] && [ -x venv/bin/python ] && tryinst venv/bin/python -m pip install -q torch --index-url "$IDX" && OKT=1
[ "$OKT" = 0 ] && tryinst $PY -m pip install -q --break-system-packages torch --index-url "$IDX" && { OKT=1; PYV=$PY; }
say "torch install took $(( $(date +%s) - T3 )) s, success=$OKT (also a throughput test: ~2.5 GB)"
if [ "$OKT" = 0 ]; then
  say "TORCH INSTALL FAILED (this is NOT a GPU result). Last lines of torch_install.log:"; tail -12 torch_install.log
  echo "VERDICT: UNKNOWN for GPUs (torch would not install: see $W/torch_install.log). Link: ${MBS} MB/s (8 streams ${M8}, 64 streams ${M64}). total $(( $(date +%s) - T0 )) s"
  exit 3
fi
[ -n "$PYV" ] || PYV=$PWD/venv/bin/python; [ -x "$PYV" ] || PYV=$PY
cat > smoke.py <<'PYEOF'
import sys, time, json
import torch
i = int(sys.argv[1])
out = {"gpu": i}
try:
    out.update(torch=torch.__version__, cuda=torch.version.cuda, name=torch.cuda.get_device_name(0), cap=list(torch.cuda.get_device_capability(0)))
    out["arch_list"] = torch.cuda.get_arch_list()
    free, total = torch.cuda.mem_get_info()
    out["vram_total_gib"] = round(total / 2**30, 1); out["vram_free_gib"] = round(free / 2**30, 1)
    a = torch.randn(4096, 4096, device="cuda"); b = torch.randn(4096, 4096, device="cuda")
    ok = float((torch.ones(8, 8, device="cuda") @ torch.ones(8, 8, device="cuda")).sum().item()) == 512.0
    out["kernel_ok"] = ok
    def tf(mode, n=20):
        torch.backends.cuda.matmul.allow_tf32 = (mode == "tf32")
        torch.cuda.synchronize(); (a @ b); torch.cuda.synchronize()
        t = time.time()
        for _ in range(n): c = a @ b
        torch.cuda.synchronize(); dt = time.time() - t
        return round(2 * 4096**3 * n / dt / 1e12, 1)
    out["fp32_tflops"] = tf("fp32"); out["tf32_tflops"] = tf("tf32")
    x = torch.empty(1 << 28, device="cuda"); torch.cuda.synchronize(); t = time.time()
    for _ in range(10): y = x.clone()
    torch.cuda.synchronize(); dt = time.time() - t
    out["copy_GBps"] = round(10 * 2 * x.numel() * 4 / dt / 1e9)
except Exception as e:
    out["error"] = (type(e).__name__ + ": " + str(e))[:300]
print(json.dumps(out))
PYEOF
say "=== 4. KERNEL SMOKE TEST + SPEED, all GPUs in parallel ==="
for g in $(seq 0 $(( NG - 1 ))); do CUDA_VISIBLE_DEVICES=$g "$PYV" smoke.py $g > smoke_$g.json 2> smoke_$g.err & done
wait
OKN=0; FP=0
for g in $(seq 0 $(( NG - 1 ))); do
  say "gpu$g: $(cat smoke_$g.json 2>/dev/null | head -c 400)"
  grep -q '"kernel_ok": true' smoke_$g.json 2>/dev/null && OKN=$(( OKN + 1 ))
done
if [ "$OKN" -lt "$NG" ]; then bad=1; why="$why only $OKN of $NG GPUs pass the kernel test (see smoke_*.json/err; 'no kernel image' = torch lacks this arch);"; fi
FP=$("$PYV" -c "import json;d=json.load(open('smoke_0.json'));print(d.get('fp32_tflops',0))" 2>/dev/null)
VR=$("$PYV" -c "import json;d=json.load(open('smoke_0.json'));print(d.get('vram_total_gib',0))" 2>/dev/null)
SPD=$(awk -v f="${FP:-0}" 'BEGIN{printf "%.1f", f/19.5}')
HMAX=$(awk -v v="${VR:-0}" 'BEGIN{h=(v*0.97-3.97)/0.00208; if (h>13000) h=13000; if (h<0) h=0; printf "%d", h}')
say "card 0: fp32 matmul ${FP:-?} TFLOPs (A100 ref 19.5 -> x$SPD), VRAM ${VR:-?} GiB -> longest stripe by the planner model ~ $HMAX slices"
fi

say "=== 5. VERDICT ==="
if [ "$bad" = 0 ]; then
  say "GO: link $MBS MB/s >= $MIN, $NG GPU(s) usable. Hints for the main run:  --gpu-speed ${SPD:-1.0}   --max-height ${HMAX:-13000}"
  [ "$MBS" != "" ] && awk -v m="$MBS" 'BEGIN{ if (m<60) print "NOTE: link under 60 MB/s: use --gpus " (m<30?"2":"4") " rather than all cards"}'
  echo "VERDICT: GO   (link ${MBS} MB/s, ${NG} GPUs, ${EFFCPU} threads, ${EFFRAM} GiB RAM, ${DISKFREE} GB free, Route A slots ~${RASLOTS}, torch wheel ${CU:-skipped}, total $(( $(date +%s) - T0 )) s)"
  exit 0
else
  echo "VERDICT: CUT   (${why}) -- destroy this box. total $(( $(date +%s) - T0 )) s"
  exit 1
fi
