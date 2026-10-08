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
DRV_CUDA=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)
say "GPUs: $NG   compute_cap(first): ${CAP:-unknown}   driver CUDA: ${DRV_CUDA:-unknown}"
FOREIGN=$(awk -F', ' '{gsub(/ MiB/,"",$4); if ($4+0 > 1500) n++} END{print n+0}' gpus.csv)
[ "$FOREIGN" -gt 0 ] && { say "WARNING: $FOREIGN GPU(s) already hold >1500 MiB (a foreign process, e.g. llama-server)"; }

say "=== 2. LINK (started now) ==="
URL=https://dl.ash2txt.org/datasets/spiral_datasets/PHerc0191/20250821151635/tracks/PHerc0191_20250821151635_surface_m7_L0_th0.2.dbm
URL2=https://github.com/StewartSethA/villa/releases/download/routeA-kit-94be1eae/routeA-kit-94be1eae.tar.xz
T1=$(date +%s.%N)
for i in 0 1 2 3 4 5 6 7; do
  S=$(( i * 1000000000 )); E=$(( S + 99999999 ))
  curl -s -r "$S-$E" --max-time 20 -o "dl/p$i" "$URL" &
done
wait
T2=$(date +%s.%N)
BYTES=$(cat dl/p* 2>/dev/null | wc -c)
SEC=$(awk -v a="$T1" -v b="$T2" 'BEGIN{printf "%.2f", b-a}')
MBS=$(awk -v b="$BYTES" -v s="$SEC" 'BEGIN{printf "%.1f", b/1e6/(s>0?s:1)}')
say "data host (dl.ash2txt.org): ${BYTES} bytes in ${SEC}s with 8 parallel ranges = $MBS MB/s"
rm -f dl/p*
B2=$(curl -sL -o /dev/null -w "%{speed_download}" --max-time 12 "$URL2")
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
case "$PIP" in uv*) $PIP torch --index-url https://download.pytorch.org/whl/$CU 2>&1 | tail -2 ;; *) $PIP install -q torch --index-url https://download.pytorch.org/whl/$CU 2>&1 | tail -2 ;; esac
say "torch install took $(( $(date +%s) - T3 )) s (it is also a throughput test: ~2.5 GB)"
PYV=$PWD/venv/bin/python; [ -x "$PYV" ] || PYV=$PY
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
  echo "VERDICT: GO   (link ${MBS} MB/s, ${NG} GPUs, torch wheel ${CU:-skipped}, total $(( $(date +%s) - T0 )) s)"
  exit 0
else
  echo "VERDICT: CUT   (${why}) -- destroy this box. total $(( $(date +%s) - T0 )) s"
  exit 1
fi
