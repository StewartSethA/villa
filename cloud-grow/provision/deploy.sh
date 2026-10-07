#!/usr/bin/env bash
# One button: ./deploy.sh PLAN.json [--checklist|--dry-run|--smoke --yes|--yes|--abort]
# Dry-run by default: real provider calls need --yes AND FLEET_EXECUTE=1. No `set -e` (D28).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$HERE/deploy.py" "$@"
