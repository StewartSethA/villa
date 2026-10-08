"""The production guard settings snapshot (pins/settings_snapshot.json) -> GuardPolicy for the box. No DB: the snapshot IS the settings."""
from __future__ import annotations

import json
import os

from . import bootstrap


def policy(kit_dir, threads: int = 2, root=None):
    from .. import growth_guard as GG
    snap = bootstrap.load_pin("settings_snapshot.json", root)
    over = {}
    for k, v in snap["grow_settings"].items():
        if k.startswith("grow.guard."):
            over[k[len("grow.guard."):]] = v
    base = GG.GuardPolicy(selfcross=True, selfcross_bin=os.path.join(str(kit_dir), "bin", "vc_tifxyz_selfcross"),
                          selfcross_env={"LD_LIBRARY_PATH": os.path.join(str(kit_dir), "lib")}, selfcross_threads=threads)
    for k, v in (_control().get("policy") or {}).items():        # in-flight guard policy tuning: applies to seeds that START after the edit
        over[str(k)] = v
    pol = GG.apply_segment_overrides(base, json.dumps(over))
    from dataclasses import replace
    return replace(pol, selfcross=True, selfcross_bin=base.selfcross_bin, selfcross_env=base.selfcross_env, selfcross_threads=threads)


def _control() -> dict:
    """<work>/control/guards.json (ROUTEA_WORK is set by routea_cloud.run): in-flight guard tuning, see routea_cloud/status.py."""
    w = os.environ.get("ROUTEA_WORK")
    if not w:
        return {}
    try:
        d = json.loads(open(os.path.join(w, "control", "guards.json")).read())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def gate_override(root=None) -> dict:
    """Pinned resume_gate thresholds, with the control file's `gate` values on top.  Called at EVERY round of every seed, so an edit applies to the next gate check."""
    from . import status as _st
    base = dict(bootstrap.load_pin("settings_snapshot.json", root).get("resume_gate") or {})
    over, _bad = _st.clean_gate(_control().get("gate") or {})
    base.update(over)
    return base


def self_collision_on(root=None) -> bool:
    return str(bootstrap.load_pin("settings_snapshot.json", root)["grow_settings"].get("grow.self_collision", "1")) in ("1", "true", "True")
