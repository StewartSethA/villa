#!/bin/bash
# Run on OUR machine.  usage: ./routeB_pull.sh --host user@box [--port N] [-i key] --remote-home <box ROUTEB_HOME> --dest <local dir> [--final|--once] [--remote-tree <box checkout>]
exec python3 "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pull_box8.py" "$@"
