#!/usr/bin/env bash
# Resume-safe data pull with a size preview. Re-run the same command after any interruption.
# Usage: fetch_data.sh SCROLL DEST [--what ct|prediction|grids|all] [--yes] ...   (no --yes = preview only)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ $# -ge 2 ] || { echo "usage: $0 SCROLL DEST [--yes] [--what all] [--levels 1,2,3,4,5]" >&2; exit 2; }
S="$1"; D="$2"; shift 2
cd "$HERE/.." && PYTHONPATH="$HERE/.." exec "${PYTHON:-python3}" -m cloud_grow fetch --scroll "$S" --dest "$D" "$@"
