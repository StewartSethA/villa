#!/usr/bin/env bash
# Route A pushbutton: grow surfaces for the given scrolls. Same command on a fleet host and on a rented box. Needs only: bash, tar, one of curl/wget/python3 and outbound HTTPS.
#   ./routeA_run.sh --scrolls PHerc0332,PHerc0211 [--seeds N] [--hours H] [--workdir DIR] [--kit-url URL | KIT_URL=URL] [--stage-only]
# Everything else is downloaded and verified (pins/): uv + python 3.12 + locked packages (user space), the VC3D kit (sha256), per-scroll surface prediction + normal grids (public open-data bucket).
# No root, no hub token, no ssh key, no database. Any missing piece stops the run with a message (no silent fallback). Re-running resumes.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
die() { echo "routeA_run: ERROR: $*" >&2; exit 2; }

# --- workdir (parsed early, passed through unchanged) ---
WORK="${ROUTEA_WORK:-$HERE/routeA_work}"
prev=""; for a in "$@"; do if [ "$prev" = "--workdir" ]; then WORK="$a"; fi; prev="$a"; done
case "$WORK" in /*) ;; *) WORK="$PWD/$WORK";; esac
mkdir -p "$WORK/.tools" "$WORK/downloads" || die "cannot create $WORK"

# --- helpers: fetch / sha256 without assuming any one tool ---
fetch() { # url dst
  if command -v curl >/dev/null 2>&1; then curl -fsSL --retry 5 --retry-delay 3 -o "$2" "$1"
  elif command -v wget >/dev/null 2>&1; then wget -q -t 5 -O "$2" "$1"
  elif command -v python3 >/dev/null 2>&1; then python3 - "$1" "$2" <<'PY'
import sys, urllib.request
urllib.request.urlretrieve(sys.argv[1], sys.argv[2])
PY
  else die "need curl, wget or python3 to download"; fi
}
sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1
  else die "need sha256sum or shasum"; fi
}

# --- 1. uv (pinned, verified) ---
# shellcheck disable=SC1091
. "$HERE/pins/uv.env"
UV="$WORK/.tools/uv"
if [ ! -x "$UV" ]; then
  echo "routeA_run: fetching uv $UV_VERSION"
  fetch "$UV_URL" "$WORK/downloads/uv.tgz" || die "uv download failed ($UV_URL)"
  [ "$(sha256 "$WORK/downloads/uv.tgz")" = "$UV_SHA256" ] || die "uv sha256 mismatch (expected $UV_SHA256)"
  tar -xzf "$WORK/downloads/uv.tgz" -C "$WORK/.tools" --strip-components=1 uv-x86_64-unknown-linux-gnu/uv || die "uv extract failed"
  chmod +x "$UV"
fi

# --- 2. python env from the hash-locked requirements (user space) ---
LOCKSUM="$(sha256 "$HERE/pins/requirements.lock")-py$PYTHON_VERSION"
if [ ! -x "$WORK/venv/bin/python" ] || [ "$(cat "$WORK/venv/.lock" 2>/dev/null)" != "$LOCKSUM" ]; then
  echo "routeA_run: creating python $PYTHON_VERSION venv + installing locked packages"
  rm -rf "$WORK/venv"
  UV_PYTHON_INSTALL_DIR="$WORK/.tools/python" "$UV" venv --python "$PYTHON_VERSION" "$WORK/venv" || die "uv venv failed (needs outbound HTTPS to github.com for the standalone python)"
  UV_CACHE_DIR="$WORK/.tools/uv-cache" "$UV" pip install --python "$WORK/venv/bin/python" --require-hashes --no-deps -r "$HERE/pins/requirements.lock" || die "package install failed"
  echo "$LOCKSUM" > "$WORK/venv/.lock"
fi
"$WORK/venv/bin/python" - <<'PY' || die "python environment check failed"
import numpy, scipy, tifffile, zarr, numcodecs  # noqa: F401
PY

# --- 3. the driver (kit, inputs, seeds, grows, verify) ---
export ROUTEA_ROOT="$HERE"
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$WORK/venv/bin/python" -m vesuvius_pipeline.routea_cloud.run "$@"
