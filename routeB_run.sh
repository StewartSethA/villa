#!/bin/bash
# Route B pushbutton:  ./routeB_run.sh --scrolls PHerc0211,PHerc0125 [--stripe-width 4500|7500|13500|full] [--smoke] [more options: --help]
# Same code path locally and on a rented box.  Everything is fetched into $ROUTEB_HOME (default ./routeB_work): python env, native build, assets.
# No `set -e` (batch convention): every stage is checked explicitly and fails LOUD with 'ROUTEB FAIL <stage>: <why>'.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export ROUTEB_HOME=${ROUTEB_HOME:-$HERE/routeB_work}
mkdir -p "$ROUTEB_HOME" || { echo "ROUTEB FAIL preflight: cannot create $ROUTEB_HOME" >&2; exit 2; }
fail(){ echo "ROUTEB FAIL $1: $2" >&2; exit "${3:-2}"; }
case "$1" in -h|--help|"") ;; esac
# ---- preflight (cheap, explicit)
[ "$(uname -s)" = Linux ] && [ "$(uname -m)" = x86_64 ] || fail preflight "needs Linux x86_64 (got $(uname -sm))"
command -v nvidia-smi >/dev/null || fail preflight "no nvidia-smi: a CUDA GPU + driver >= 525 is required for fit/flatten/render/ink (use --stages fetch to only download)"
nvidia-smi -L >/dev/null 2>&1 || fail preflight "nvidia-smi cannot talk to the driver"
command -v g++ >/dev/null || fail preflight "no g++ (needed once to build the vc_spiral C++23 extension): apt-get install g++ (>= 13)"
GV=$(g++ -dumpversion | cut -d. -f1); [ "${GV:-0}" -ge 13 ] 2>/dev/null || fail preflight "g++ $GV is too old for C++23 (need >= 13)"
command -v curl >/dev/null || command -v wget >/dev/null || fail preflight "need curl or wget"
LIBSTD=$(g++ -print-file-name=libstdc++.so.6)
FREE=$(df -BG --output=avail "$ROUTEB_HOME" | tail -1 | tr -dc 0-9)
echo "routeB: host=$(hostname) gpu=[$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tr '\n' ';')] free=${FREE}G in $ROUTEB_HOME g++=$GV"
# ---- LINK CHECK FIRST (box8 mode): measure ingress from the data host and a CDN before anything that costs money-time (env, compile, fetch)
case " $* " in
  *" --mode box8 "*)
    if [ "${ROUTEB_SKIP_LINKCHECK:-0}" != 1 ] && command -v python3 >/dev/null; then
      PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" python3 -m routeB.linkcheck "$@" \
        || fail link "link check gate (see the verdict box above); nothing was built" 5
    fi;;
esac
# ---- EARLY PREFETCH (box8): start downloading the first scroll's stripes now, while the env builds
PF_PID=""
case " $* " in
  *" --mode box8 "*)
    if [ "${ROUTEB_SKIP_PREFETCH:-0}" != 1 ] && command -v python3 >/dev/null; then
      PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" python3 -m routeB.prefetch "$@" \
        > "$ROUTEB_HOME/prefetch.log" 2>&1 &
      PF_PID=$!
      echo "routeB: early prefetch running in the background (pid $PF_PID, log $ROUTEB_HOME/prefetch.log)"
    fi;;
esac
# ---- python env (pinned lock) + native build
# ---- torch build: cu126 (validated) unless --torch-cuda says otherwise or a Blackwell card (compute capability >= 12) needs sm_120 kernels
TC=""
PREV=""
for A in "$@"; do
  [ "$PREV" = "--torch-cuda" ] && TC=$A
  PREV=$A
done
if [ -z "$TC" ] || [ "$TC" = auto ]; then
  MAXCC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -rn | head -1)
  DRVC=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)
  TC=cu126
  if [ "${MAXCC%%.*}" -ge 12 ] 2>/dev/null; then TC=cu128; fi
  case "$DRVC" in 12.[89]|12.[1-9][0-9]|13.*) TC=cu128 ;; esac
  echo "routeB: torch build chosen from the hardware: $TC (compute capability ${MAXCC:-?}, driver CUDA ${DRVC:-?})"
fi
case "$TC" in
  cu126) LOCK="$HERE/routeB/pins/requirements.lock"; ENVD="$ROUTEB_HOME/env" ;;
  cu128|cu129) LOCK="$HERE/routeB/pins/requirements.$TC.lock"; ENVD="$ROUTEB_HOME/env_$TC"
    echo "routeB: *** torch build $TC is UNVALIDATED numerically (the validated build is cu126): spot-check"
    echo "routeB: *** one stripe on a known GPU before trusting these fits (D6) ***" ;;
  *) fail args "--torch-cuda must be auto|cu126|cu128|cu129 (got $TC)" ;;
esac
export ROUTEB_TORCH_CUDA=$TC
bash "$HERE/deploy_common/bootstrap_env.sh" "$ENVD" "$LOCK" --uv-env "$HERE/routeB/pins/uv.env" \
     --extra-index "https://download.pytorch.org/whl/$TC" >"$ROUTEB_HOME/bootstrap.log" 2>&1 \
  || { tail -20 "$ROUTEB_HOME/bootstrap.log" >&2; fail env "bootstrap_env.sh failed (log $ROUTEB_HOME/bootstrap.log)"; }
tail -1 "$ROUTEB_HOME/bootstrap.log"
SFSTAMP="$ENVD/.spiral.$(md5sum "$HERE"/spiral-fitting/cpp/*.cpp "$HERE"/spiral-fitting/pyproject.toml | md5sum | cut -c1-12)"
if [ ! -f "$SFSTAMP" ]; then
  echo "routeB: building the vc_spiral native extension (once)"
  UV_CACHE_DIR="$ENVD.tools/cache" UV_PYTHON_PREFERENCE=only-managed "$ENVD.tools/uv" pip install --python "$ENVD/bin/python" --no-deps "$HERE/spiral-fitting" \
      >"$ROUTEB_HOME/build_spiral.log" 2>&1 || { tail -30 "$ROUTEB_HOME/build_spiral.log" >&2; fail native-build "uv pip install spiral-fitting failed (log $ROUTEB_HOME/build_spiral.log)"; }
  touch "$SFSTAMP"
fi
# SHADOWING gotcha (found 2026-10-08 by the pny dev run: every fit "finished" then died in satisfaction with "rebuild the Spiral native extensions"):
# fit_spiral.py runs with cwd=spiral-fitting, whose source dir vc_spiral/ (only __init__.py) shadows the installed vc_spiral package, so
# `import vc_spiral.spiral_sampling` raised ImportError -> load_spiral_sampling() returned None -> silent pure-python fallback, then a crash at the end.
# Fix: put the built extensions next to the source package too, then PROVE the import from the cwd the fit uses.
SPSO=$(ls "$ENVD"/lib/python*/site-packages/vc_spiral/*.so 2>/dev/null)
[ -n "$SPSO" ] || fail native-build "no vc_spiral/*.so installed in $ENVD (see $ROUTEB_HOME/build_spiral.log)"
cp -f $SPSO "$HERE/spiral-fitting/vc_spiral/" || fail native-build "cannot copy the vc_spiral extensions into $HERE/spiral-fitting/vc_spiral"
# libstdc++ gotcha (GLIBCXX_3.4.29+ needed by the native extensions under torch)
SYSLIB=$(ldconfig -p 2>/dev/null | awk '/libstdc\+\+\.so\.6 /{print $NF; exit}')
if ! strings "$SYSLIB" 2>/dev/null | grep -q GLIBCXX_3.4.29; then
  [ -f "$LIBSTD" ] && strings "$LIBSTD" | grep -q GLIBCXX_3.4.29 && export LD_LIBRARY_PATH="$(dirname "$LIBSTD")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" && echo "routeB: using $(dirname "$LIBSTD") for libstdc++ (system one lacks GLIBCXX_3.4.29)" \
    || fail libstdc++ "system libstdc++ lacks GLIBCXX_3.4.29 and g++'s does not provide it either"
fi
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" ROUTEB_PYTHON="$ENVD/bin/python"
export PYTHONWARNINGS=ignore
# the vesuvius/src path makes vc3d_fiber_format importable for spiral-fitting/lasagna (their adapters search <parent>/vesuvius/src)
"$ENVD/bin/python" -c "import torch, importlib; importlib.import_module('vc_spiral.spiral_sampling'); print('routeB: torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'vc_spiral OK')" \
   || fail env-check "torch/vc_spiral do not import in $ENVD (see TROUBLESHOOTING.md: libstdc++, driver)"
( cd "$HERE/spiral-fitting" && "$ENVD/bin/python" -c "import vc_spiral.spiral_sampling as s; assert hasattr(s,'PatchSatisfactionAtlas'), 'PatchSatisfactionAtlas missing'; print('routeB: native vc_spiral importable from the fit cwd, PatchSatisfactionAtlas present')" ) \
   || fail native-shadow "vc_spiral.spiral_sampling is not importable from $HERE/spiral-fitting (the cwd fit_spiral.py runs in): the fit would fall back to python and crash at the end"
# ---- record exactly what was installed (downloadable with the results)
mkdir -p "$ROUTEB_HOME/box8"
"$ENVD/bin/python" -c "import json,sys,torch,importlib.metadata as m; d={'torch':torch.__version__,'cuda':torch.version.cuda,'arch_list':torch.cuda.get_arch_list(),'build':sys.argv[1],'lock':sys.argv[2],'torchvision':m.version('torchvision'),'triton':m.version('triton')}; print(json.dumps(d))" "$TC" "$(basename "$LOCK")" > "$ROUTEB_HOME/box8/env_installed.json" 2>/dev/null
echo "routeB: installed: $(cat "$ROUTEB_HOME/box8/env_installed.json" 2>/dev/null)"
# ---- GPU kernel smoke test (box8 mode): every allowed GPU must run real kernels BEFORE any fetch or fit
case " $* " in
  *" --mode box8 "*)
    if [ "${ROUTEB_SKIP_GPUSMOKE:-0}" != 1 ]; then
      GL=""
      PREV=""
      for A in "$@"; do
        [ "$PREV" = "--gpus" ] && GL=$A
        PREV=$A
      done
      "$ENVD/bin/python" -m routeB.gpusmoke ${GL:+--gpus "$GL"} \
        || fail gpu-smoke "a GPU cannot run this torch build ($TC); see the table above. Nothing was fetched." 6
    fi;;
esac
if [ -n "$PF_PID" ]; then
  kill -TERM "$PF_PID" 2>/dev/null
  wait "$PF_PID" 2>/dev/null
  echo "routeB: prefetch handed over to the scheduler: $(tail -1 "$ROUTEB_HOME/prefetch.log" 2>/dev/null)"
fi
exec "$ENVD/bin/python" -m routeB.cli "$@"
