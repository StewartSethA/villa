#!/bin/bash
# Runtime control of a running box8 (re-read by the scheduler every ~10 s; no restart).  usage:
#   ./routeB_ctl.sh [--home DIR] status                 allowed GPUs, jobs, budget, the latest REPLAN, recent control events
#   ./routeB_ctl.sh [--home DIR] gpus 0,1,2             allow ONLY these GPUs (also re-allows drained/killed ones named here); 'gpus none' allows nothing
#   ./routeB_ctl.sh [--home DIR] drain N                finish GPU N's current fit, then stop using it
#   ./routeB_ctl.sh [--home DIR] kill N                 SIGTERM GPU N's fit now; the interval is re-queued and resumes from its last autosave (never lost); GPU N stays out
#   ./routeB_ctl.sh [--home DIR] stop                   graceful: no new launches, running fits finish, the run then ends (re-run to resume)
#   ./routeB_ctl.sh [--home DIR] pause | resume         pause/resume launching (resume also clears a pending stop)
# --home defaults to $ROUTEB_HOME. The hard stop (kills running fits) is `touch <home>/box8/STOP`.
HOME_DIR=${ROUTEB_HOME:-}
if [ "$1" = "--home" ]; then HOME_DIR=$2; shift 2; fi
[ -n "$HOME_DIR" ] || HOME_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/routeB_work
C="$HOME_DIR/box8/control"
[ -d "$HOME_DIR/box8" ] || { echo "routeB_ctl: no $HOME_DIR/box8 (is --home / ROUTEB_HOME right?)" >&2; exit 2; }
mkdir -p "$C"
need_n(){ case "$1" in ''|*[!0-9]*) echo "routeB_ctl: GPU index required (got '$1')" >&2; exit 2;; esac; }
cmd=$1; shift
case "$cmd" in
  gpus)
    L=$1; [ "$L" = none ] && L=""
    if [ -n "$L" ]; then for g in $(echo "$L" | tr ',' ' '); do need_n "$g"; rm -f "$C/drain.$g"; done; fi
    echo "$L" > "$C/gpus"; echo "allowed GPUs -> '${L:-none}' (taking effect within ~10 s; running fits on removed GPUs finish)";;
  drain) need_n "$1"; echo "draining" > "$C/drain.$1"; echo "GPU $1 will be drained after its current fit";;
  kill)  need_n "$1"; echo "kill" > "$C/kill.$1"; echo "GPU $1: fit will be SIGTERMed and re-queued from its last autosave; GPU stays out until 'gpus' names it";;
  stop)  echo stop > "$C/STOP"; echo "graceful stop requested";;
  pause) echo pause > "$C/PAUSE"; echo "paused";;
  resume) echo resume > "$C/RESUME"; echo "resume requested (clears PAUSE/STOP)";;
  status)
    python3 - "$HOME_DIR" <<'PY'
import json, sys, glob, os
h = sys.argv[1]; c = os.path.join(h, "box8", "control")
def rd(p):
    try: return open(p).read().strip()
    except OSError: return None
print("control dir:", c)
print("  gpus file :", repr(rd(c + "/gpus")), "| drain:", sorted(os.path.basename(x)[6:] for x in glob.glob(c + "/drain.*")),
      "| STOP" if os.path.exists(c + "/STOP") else "", "| PAUSE" if os.path.exists(c + "/PAUSE") else "", "| hard STOP file!" if os.path.exists(h + "/box8/STOP") else "")
try:
    st = json.load(open(h + "/out/STATUS.json"))
    b = st["budget"]
    print(f"status @ {st['t']}: jobs {st['jobs']} busy GPUs {st['busy_gpus']}; spent ${b['spent']:.2f} projected ${b['projected_total']:.2f} (soft {b['soft']}, hard {b['hard']}) at {b['hours']:.2f} h; stop={st['stop']}")
    print("  scrolls:", st["scrolls"])
except Exception as e:
    print("no out/STATUS.json yet:", e)
p = rd(c + "/PLAN.txt")
print("latest REPLAN:\n  " + (p.replace("\n", "\n  ") if p else "(none yet)"))
ev = h + "/box8/events.jsonl"
if os.path.exists(ev):
    rows = [json.loads(l) for l in open(ev) if '"control"' in l or '"replan"' in l or 'ctl_kill' in l]
    for r in rows[-5:]: print("  event", r.get("t"), r.get("kind"), r.get("what") or r.get("reason") or r.get("job") or "")
PY
    command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | sed 's/^/  gpu /';;
  *) sed -n 2,12p "$0"; exit 2;;
esac
