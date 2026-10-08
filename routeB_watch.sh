#!/bin/bash
# Live dashboard of a running box8 (read-only; Ctrl-C only detaches the watcher).
#   routeB_watch.sh [--home DIR] [--interval 5] [--plain] [--once]
# --once prints ONE snapshot block to paste back for diagnosis.
# Reattach any time. Raw console of the run: tmux attach -t routeb
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ -z "$ROUTEB_HOME" ] && [ -f "$HOME/.routeb_home" ]; then
  ROUTEB_HOME=$(cat "$HOME/.routeb_home")
fi
export ROUTEB_HOME=${ROUTEB_HOME:-$HERE/routeB_work}
PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" exec python3 -m routeB.watch "$@"
