"""The one place a fleet-wide alert is spelled.

`stages/render.py` grew an `_alert()` for flatten faults (user, 2026-09-15: "a fallback or a
host fault must never be silent"). The ink-model families never got it: each printed its skip
to plain stdout instead, so a host missing the Grand Prize venv logged **5,697**
`[grandprize_dense] skipped: no Grand Prize venv on this host` lines into one segment's task
log and raised nothing anywhere. A skip SUCCEEDS from the scheduler's point of view -- it
leaves no mark on queue depth, worker count or task outcome -- so nearly six thousand
discarded tasks were invisible to every metric anyone was watching while eight V100s sat at
0 % (FINDINGS 111.1).

WHAT EARNS AN ALERT, and the distinction is the whole point. `available()` answers "are the
weights here", never "should this run" (CLAUDE.md, "Adding an ink model"). So a false from it
is an ENVIRONMENT FAULT -- this host is misconfigured and a person must fix it -- and that is
worth the marker. A transient ("card full", a busy GPU) is not: it resolves itself, and
alerting on it would spend the reader's attention on work that is already being retried.
CLAUDE.md is explicit that a marker used for something merely interesting stops meaning
anything, so the families route only the `available()`-false path here and keep ordinary
skips on stdout.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone


def alert(msg: str) -> None:
    """Mark an environment fault on stderr and in the durable fleet-wide record.

    stderr lands in the task's own log, which the hub already surfaces per segment; the file
    is what makes a fault greppable across hosts and survives the log rotating. Both are
    best-effort: alerting must never raise into the caller's path, because the caller is
    already handling a fault.
    """
    print(f"\U0001F534 ALERT: {msg}", file=sys.stderr, flush=True)
    try:
        import os
        from pathlib import Path
        from . import config
        # VPIPE_ALERTS_LOG redirects the durable record: the test suite sets it, because its
        # fake worlds publish deliberately degenerate maps and 256 of those test alerts
        # (paths under /tmp/pytest-of-*) had landed in the hub's real ALERTS.log -- the file
        # /api/alerts reads -- burying the real ones (2026-09-25).
        p = Path(os.environ["VPIPE_ALERTS_LOG"]) if os.environ.get("VPIPE_ALERTS_LOG") \
            else config.repo_root() / "var" / "log" / "ALERTS.log"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()} {msg}\n")
    except (OSError, ImportError):
        pass
