#!/bin/bash
# Idempotent, user-space, fail-loud Python environment bootstrap shared by the deploy branches.
#
# usage: bootstrap_env.sh <env_dir> <lockfile> [--python 3.14] [--uv-env pins/uv.env] [--extra-index URL]
#   <env_dir>   created if absent; <env_dir>/bin/python is the result; <env_dir>/.uv holds the pinned uv
#   <lockfile>  `name==version` lines (uv pip freeze style); torch lines may carry +cuXXX local versions
#   --uv-env    file with UV_URL=, UV_SHA256= (pinned uv static binary, sha256-checked) -- default: next to <lockfile>/uv.env
#   --extra-index  e.g. https://download.pytorch.org/whl/cu126 (needed for +cuXXX torch wheels)
# Prints "BOOTSTRAP OK python=<path>" on success; every failure prints "BOOTSTRAP FAIL: <stage>: <why>" and exits non-zero.
# No `set -e` (repo convention): each stage is checked explicitly so the message names the stage.
ENVD=${1:?env_dir}; LOCK=${2:?lockfile}; shift 2
PYV=3.14.7; UVENV=""; EXTRA=""
while [ $# -gt 0 ]; do case "$1" in
  --python) PYV=$2; shift 2;; --uv-env) UVENV=$2; shift 2;; --extra-index) EXTRA=$2; shift 2;;
  *) echo "BOOTSTRAP FAIL: args: unknown option $1"; exit 2;; esac; done
fail(){ echo "BOOTSTRAP FAIL: $1: $2" >&2; exit "${3:-1}"; }
[ -f "$LOCK" ] || fail lockfile "missing $LOCK"
[ -n "$UVENV" ] || UVENV="$(dirname "$LOCK")/uv.env"
mkdir -p "$ENVD" || fail mkdir "cannot create $ENVD"
ENVD=$(cd "$ENVD" && pwd)
TOOLS="$ENVD.tools"          # pinned uv + managed pythons + cache live BESIDE the env (uv venv refuses a non-empty dir)
mkdir -p "$TOOLS" || fail mkdir "cannot create $TOOLS"
UV=""
if [ -x "$TOOLS/uv" ]; then UV="$TOOLS/uv"
else
  [ -f "$UVENV" ] || fail uv-pin "missing $UVENV (needs UV_URL, UV_SHA256)"
  . "$UVENV"
  [ -n "$UV_URL" ] && [ -n "$UV_SHA256" ] || fail uv-pin "$UVENV lacks UV_URL/UV_SHA256"
  mkdir -p "$TOOLS"
  T="$TOOLS/uv.tar.gz"
  for i in 1 2 3 4; do
    if command -v curl >/dev/null; then curl -fsSL --retry 3 -o "$T" "$UV_URL" && break
    elif command -v wget >/dev/null; then wget -q -O "$T" "$UV_URL" && break
    else fail uv-download "neither curl nor wget is installed (apt-get install curl)"; fi
    sleep $((i*3))
  done
  [ -s "$T" ] || fail uv-download "could not fetch $UV_URL"
  GOT=$(sha256sum "$T" | awk '{print $1}')
  [ "$GOT" = "$UV_SHA256" ] || { rm -f "$T"; fail uv-sha256 "got $GOT expected $UV_SHA256"; }
  tar -xzf "$T" -C "$TOOLS" --strip-components=1 || fail uv-extract "tar failed"
  UV="$TOOLS/uv"; [ -x "$UV" ] || fail uv-extract "no uv binary in the archive"
fi
echo "bootstrap: uv $("$UV" --version) ; python $PYV ; lock $LOCK ($(grep -c '==' "$LOCK") pins)"
export UV_PYTHON_INSTALL_DIR="$TOOLS/pythons" UV_CACHE_DIR="$TOOLS/cache" UV_LINK_MODE=copy UV_PYTHON_PREFERENCE=only-managed
if [ ! -x "$ENVD/bin/python" ]; then
  "$UV" venv --python "$PYV" "$ENVD" >&2 || fail venv "uv venv --python $PYV failed (python download blocked? disk?)"
fi
STAMP="$ENVD/.lock.sha256"; WANT=$(sha256sum "$LOCK" | awk '{print $1}')
if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$WANT" ]; then
  echo "bootstrap: lock unchanged, env already installed"
else
  IDX=(); [ -n "$EXTRA" ] && IDX=(--extra-index-url "$EXTRA")
  "$UV" pip install --python "$ENVD/bin/python" --index-strategy unsafe-best-match "${IDX[@]}" -r "$LOCK" >&2 \
    || fail pip-install "uv pip install -r $LOCK failed (see output above)"
  echo "$WANT" > "$STAMP"
fi
"$ENVD/bin/python" -c "import sys; print('python', sys.version.split()[0])" || fail verify "env python does not run"
echo "BOOTSTRAP OK python=$ENVD/bin/python"
