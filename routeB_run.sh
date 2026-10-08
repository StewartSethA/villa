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
# ---- python env (pinned lock) + native build
ENVD="$ROUTEB_HOME/env"
bash "$HERE/deploy_common/bootstrap_env.sh" "$ENVD" "$HERE/routeB/pins/requirements.lock" --uv-env "$HERE/routeB/pins/uv.env" \
     --extra-index https://download.pytorch.org/whl/cu126 >"$ROUTEB_HOME/bootstrap.log" 2>&1 || { tail -20 "$ROUTEB_HOME/bootstrap.log" >&2; fail env "bootstrap_env.sh failed (log $ROUTEB_HOME/bootstrap.log)"; }
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
exec "$ENVD/bin/python" -m routeB.cli "$@"
