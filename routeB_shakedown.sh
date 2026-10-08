#!/bin/bash
# <= $5 shakedown of every Route B stage on the rental box (about 1-2 h of box time).  Run it FIRST on the rented box:
#   ./routeB_shakedown.sh [--scrolls PHerc0211,PHerc0826] [--steps 1500] [--wait-pull SECONDS]
# 1) env + native build + budget plan (dry run)   2) box8 mode --smoke on 2 scrolls in parallel: fetch -> fit -> tiles -> payload (DONE marker)
# 3) pipeline mode --smoke --stages ink,export on the same dirs: CT chunk fetch -> sheet snap -> flatten -> render -> 4 ink families -> export
# 4) pull-back: run the printed command from OUR machine (the box never connects out to us); this script then sees PULLED.json.
# Then it prints PASS/FAIL/PENDING per stage with seconds and GB, from ARTIFACTS (not exit codes).  The budget governor runs with soft $4.0 / hard $4.8.
# No set -e: every stage runs and is reported even if an earlier one failed.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCROLLS=PHerc0211,PHerc0826; STEPS=1500; WAITPULL=0; WITHINK=0
while [ $# -gt 0 ]; do case "$1" in --scrolls) SCROLLS=$2; shift 2;; --steps) STEPS=$2; shift 2;; --wait-pull) WAITPULL=$2; shift 2;; --with-ink) WITHINK=1; shift;; *) echo "SHAKEDOWN FAIL args: unknown $1" >&2; exit 2;; esac; done
export ROUTEB_HOME=${ROUTEB_HOME:-$HERE/routeB_shakedown_work}
export BUDGET_SOFT_USD=${BUDGET_SOFT_USD:-5.0} BUDGET_HARD_USD=${BUDGET_HARD_USD:-5.8}
export BUDGET_BOX_START=${BUDGET_BOX_START:-$(date +%s)}
mkdir -p "$ROUTEB_HOME/shakedown" || { echo "SHAKEDOWN FAIL: cannot create $ROUTEB_HOME" >&2; exit 2; }
LOG=$ROUTEB_HOME/shakedown/stages.log; : > "$LOG"
T0=$(date +%s)
echo "== shakedown $(date -u +%FT%TZ) host=$(hostname) ROUTEB_HOME=$ROUTEB_HOME scrolls=$SCROLLS steps=$STEPS budget soft=\$$BUDGET_SOFT_USD hard=\$$BUDGET_HARD_USD"
echo "== GPUs:"; nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader 2>&1 | sed 's/^/   /'
echo "== disk:"; df -h "$ROUTEB_HOME" | sed 's/^/   /'; echo "== RAM:"; free -g | sed 's/^/   /'
stage(){ local n=$1; shift; local t=$(date +%s); echo "---- stage $n: $*"; "$@"; local rc=$?; echo "stage $n rc=$rc seconds=$(( $(date +%s) - t ))" | tee -a "$LOG"; return $rc; }
stage 1-env-plan "$HERE/routeB_run.sh" --mode box8 --scrolls "$SCROLLS" --smoke --steps "$STEPS" --dry-run
stage 2-box8-smoke "$HERE/routeB_run.sh" --mode box8 --scrolls "$SCROLLS" --smoke --steps "$STEPS"
if [ "$WITHINK" = 1 ]; then stage 3-ink-export "$HERE/routeB_run.sh" --scrolls "$SCROLLS" --smoke --steps "$STEPS" --stages ink,export; else echo "stage 3 (flatten/render/ink) skipped: the cloud run is segment production only (use --with-ink to include it)"; fi
echo "---- stage 4: PULL-BACK (run on OUR machine; the box has no credentials for our network):"
echo "     python3 pull_box8.py --host <user@box> --remote-home $ROUTEB_HOME --remote-tree $HERE --dest <local dir> --final"
if [ "$WAITPULL" -gt 0 ] 2>/dev/null; then
  for i in $(seq 1 $(( WAITPULL / 10 ))); do n=$(ls "$ROUTEB_HOME"/out/*/*/PULLED.json 2>/dev/null | wc -l); [ "$n" -ge "$(echo "$SCROLLS" | tr ',' '\n' | wc -l)" ] && break; sleep 10; done
fi
echo "---- report"
"$ROUTEB_HOME/env/bin/python" -m routeB.shakedown_report $([ "$WITHINK" = 1 ] && echo --with-ink) --scrolls "$SCROLLS" --json "$ROUTEB_HOME/shakedown/report.json" ; RC=$?
W=$(( $(date +%s) - T0 ))
python3 - <<PY
w=$W; gb=0.0
import json,glob
for p in glob.glob("$ROUTEB_HOME/assets/.fetched/_totals.json"):
    gb=sum(x["bytes_net"] for x in json.load(open(p)))/1e9
import sys; sys.path.insert(0, "$HERE/deploy_common"); import budget; R = budget.Rates.from_env().eff_hour_usd
print(f"shakedown wall {w} s = {w/3600:.2f} h -> \${w/3600*R:.2f} box time (machine+disk \${R:.3f}/h) + \${gb*2.70/1000:.3f} for {gb:.2f} GB fetched (box download) ; budget ceiling \$5")
PY
exit $RC
