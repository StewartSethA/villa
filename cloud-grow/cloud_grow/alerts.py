"""Alert sink for the vendored guard: stderr plus an append-only JSONL file (CLOUD_GROW_ALERT_LOG).

The fleet's `alerts.alert()` posts to the hub. A cloud box has no hub; an alert must still be loud and must
survive in the export, so it is written to stderr and appended to a JSONL that `pack` ships as a sidecar.
"""
from __future__ import annotations

import json
import os
import sys
import time


def alert(msg: str, level: str = "MAJOR") -> None:
    line = f"[cloud-grow ALERT {level}] {msg}"
    print(line, file=sys.stderr, flush=True)
    path = os.environ.get("CLOUD_GROW_ALERT_LOG")
    if path:
        try:
            with open(path, "a") as fh:
                fh.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "level": level,
                                     "msg": str(msg)}) + "\n")
        except OSError as e:
            print(f"[cloud-grow] alert log unwritable: {e}", file=sys.stderr, flush=True)
