"""routeB_keepbusy.sh: start narrower-stripe passes by itself when the box goes idle (user 2026-10-08)."""
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _run(tmp_path, jobs, scrolls, extra_env=None, args=()):
    main = tmp_path / "routeB"
    (main / "out").mkdir(parents=True, exist_ok=True)
    (main / "out" / "STATUS.json").write_text(json.dumps({"jobs": jobs, "scrolls": scrolls}))
    env = {**os.environ, "ROUTEB_HOME": str(main), "KB_DRY": "1", "KB_IDLE_S": "0", "KB_POLL": "1", "KB_FAKE_IDLE": "2", "KB_ONCE": "1", **(extra_env or {})}
    r = subprocess.run(["bash", str(ROOT / "routeB_keepbusy.sh"), *args], capture_output=True, text=True, env=env, timeout=60)
    return r.stdout + r.stderr


def test_idle_and_empty_queue_launches_narrower_passes_on_complete_scrolls(tmp_path):
    out = _run(tmp_path, {"done": 7, "running": 2}, {"PHerc0125": "complete", "PHerc0191": "complete", "PHerc0800": "fetched"})
    assert "LAUNCH alternative pass: height 4500 on [PHerc0125,PHerc0191]" in out and "LAUNCH alternative pass: height 2800" in out
    assert "--max-height 4500 --no-tail-split --no-routea --free-inputs-on done" in out and "PHerc0800" not in out.split("LAUNCH")[1]      # unfinished scrolls are not re-run
    assert "all heights done" in out


def test_queued_work_means_not_idle(tmp_path):
    out = _run(tmp_path, {"done": 3, "running": 2, "pending": 2}, {"PHerc0125": "complete"})
    assert "LAUNCH" not in out and "not idle" in out


def test_no_idle_gpu_means_not_idle(tmp_path):
    out = _run(tmp_path, {"done": 3, "running": 4}, {"PHerc0125": "complete"}, {"KB_FAKE_IDLE": "0"})
    assert "LAUNCH" not in out


def test_stop_file_and_deadline_and_no_completed_scrolls(tmp_path):
    (tmp_path / "routeB_alt").mkdir()
    (tmp_path / "routeB_alt" / "keepbusy.STOP").write_text("")
    assert "STOP file" in _run(tmp_path, {"done": 1}, {"PHerc0125": "complete"})
    (tmp_path / "routeB_alt" / "keepbusy.STOP").unlink()
    old = str(int(time.time()) - 5 * 3600)
    assert "deadline" in _run(tmp_path, {"done": 1}, {"PHerc0125": "complete"}, {"KB_DEADLINE_H": "1", "BUDGET_BOX_START": old})
    assert "no completed scrolls" in _run(tmp_path, {"done": 0}, {"PHerc0125": "fetched"})


def test_status_mode(tmp_path):
    out = _run(tmp_path, {"done": 1, "running": 1}, {"PHerc0125": "complete"}, args=("status",))
    assert "pending=0 running=1 complete=PHerc0125" in out
