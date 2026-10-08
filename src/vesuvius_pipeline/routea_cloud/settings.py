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
    pol = GG.apply_segment_overrides(base, json.dumps(over))
    from dataclasses import replace
    return replace(pol, selfcross=True, selfcross_bin=base.selfcross_bin, selfcross_env=base.selfcross_env, selfcross_threads=threads)


def gate_override(root=None) -> dict:
    return dict(bootstrap.load_pin("settings_snapshot.json", root).get("resume_gate") or {})


def self_collision_on(root=None) -> bool:
    return str(bootstrap.load_pin("settings_snapshot.json", root)["grow_settings"].get("grow.self_collision", "1")) in ("1", "true", "True")
