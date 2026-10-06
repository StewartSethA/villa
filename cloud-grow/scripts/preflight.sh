#!/usr/bin/env bash
# Preflight for a cloud-grow box. Usage: preflight.sh --config run.json [--grows N] [--passmark-st X --usd-per-hour Y]
# No `set -e` (repo convention D28): every check runs and reports. Exit code is the python result.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
echo "== host =="; uname -a; grep -m1 'model name' /proc/cpuinfo; nproc; free -g | sed -n 1,2p; df -h . | tail -1
echo "== glibc (tracer needs >= 2.14) =="; ldd --version 2>&1 | head -1
echo "== cloud-grow preflight =="
cd "$HERE/.." && PYTHONPATH="$HERE/.." exec "$PY" -m cloud_grow.preflight "$@"
