#!/bin/bash
# Prove the pushbuttons on this box with small cases.  usage: ./smoke_test.sh A|B|both   (prints wall time and GB downloaded)
# Not `set -e`: both routes are attempted; the exit status is non-zero if any failed.
HERE=$(cd "$(dirname "$0")" && pwd); rc=0
t0=$(date +%s)
if [ "$1" = A ] || [ "$1" = both ]; then
  ROUTEA_WORK=${ROUTEA_WORK:-$HERE/routeA_work_smoke} "$HERE/routeA_run.sh" --scrolls PHerc0332 --seeds 4 --rounds 2 --workdir "${ROUTEA_WORK:-$HERE/routeA_work_smoke}" || { echo "SMOKE A FAILED" >&2; rc=1; }
fi
if [ "$1" = B ] || [ "$1" = both ]; then
  ROUTEB_HOME=${ROUTEB_HOME:-$HERE/routeB_work_smoke} "$HERE/routeB_run.sh" --scrolls PHerc0211 --smoke || { echo "SMOKE B FAILED" >&2; rc=1; }
  python3 - <<PY
import json,glob
for p in glob.glob("${ROUTEB_HOME:-$HERE/routeB_work_smoke}/assets/.fetched/_totals.json"):
    print("routeB downloaded GB:", round(sum(x["bytes_net"] for x in json.load(open(p)))/1e9,3))
PY
fi
echo "smoke wall: $(( $(date +%s) - t0 )) s, rc=$rc"
exit $rc
