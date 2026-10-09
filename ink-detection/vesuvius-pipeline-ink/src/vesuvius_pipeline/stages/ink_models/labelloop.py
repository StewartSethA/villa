"""`labelloop:<checkpoint>` -- run any checkpoint the continual fine-tune loop registered.

The loop's exports are ordinary dense / narrow heads: the same architecture, the same
input contract and the same inference code as `grandprize_dense`. What differs is only
WHICH file, and where the answer to that lives -- the `labelloop_ckpt` registry, so a
checkpoint is runnable from the prospecting tab the moment it is registered and nobody has
to copy a path into a config.

    labelloop:dense              the current export of the dense head
    labelloop:narrow32           the current export of the ~32x32x13 head (FINDINGS 66)
    labelloop:narrow32.s01200    an intermediate checkpoint, by step

A name that is not in the registry is UNAVAILABLE, not an error: a sweep that names a
checkpoint this host has not received skips it with a reason, exactly as every other
optional family does.
"""
from __future__ import annotations

import os

from ... import alerts as _alerts

from . import grandprize_dense as GD

SPEC = dict(GD.SPEC)


def training_frame():
    return GD.training_frame()


def check_frame(frame, tol=None):
    return GD.check_frame(frame) if tol is None else GD.check_frame(frame, tol=tol)


def enabled() -> bool:
    """Never a default production family. `labelloop:<name>` is run by being NAMED -- in a
    prospect sweep or by the loop's own inference pass -- so it must not join the finish
    stage's set merely by having been imported."""
    return False


def _registry_lookup(name: str) -> dict:
    # THE HUB'S registry, through connect_for: on a peer, pipeline_db().connect() opened (and
    # created) an EMPTY local var/pipeline.db, so every peer read "not registered" for a
    # checkpoint the hub had registered -- host, 2026-09-24.
    try:
        from ... import config
        from ...remotedb import connect_for, hub_token
        from .. import labelloop as LL
        fleet = config.load()
        me = config.this_host(fleet)
        db = connect_for(fleet, hub_token(fleet))
        if me is not None and me.name == fleet.hub_host:
            LL.ensure_schema(db)
        return LL.checkpoint(db, f"labelloop:{name}") or LL.checkpoint(db, name)
    except Exception as e:           # noqa: BLE001 - reported, then treated as unavailable
        _alerts.alert(f"labelloop registry lookup for {name!r} failed: {type(e).__name__}: {e}")
        return {}


def checkpoint_path(arm: str | None = None) -> str | None:
    if not arm:
        return None
    env = os.environ.get(f"VPIPE_LABELLOOP_CKPT_{arm.replace('.', '_').upper()}", "")
    if env and os.path.exists(env):
        return os.path.abspath(env)
    rec = _registry_lookup(arm)
    p = rec.get("path")
    if not p:
        return None
    if os.path.exists(p):
        return os.path.abspath(p)
    return _localise(p, rec.get("sha"))


def _localise(p: str, sha: str | None) -> str | None:
    """The registry records the path on the host that TRAINED it (the hub). On a peer that path
    does not exist -- a peer's repo is at a different path, so job 769 reported
    "labelloop:dense is not registered on this host" and produced nothing (2026-09-24). Rebase
    it onto this host's own repo `var/`, and when the file is not there either, pull it from the
    hub over the peer's own ssh setup and verify the registered sha before using it."""
    import subprocess
    from pathlib import Path
    from ... import config
    if "/var/" not in p:
        return None
    rel = p.split("/var/", 1)[1]
    fleet = config.load()
    local = Path(fleet.root or config.repo_root()) / "var" / rel
    if not local.exists():
        hub = fleet.hosts.get(fleet.hub_host) if getattr(fleet, "hub_host", None) else None
        if hub is None:
            _alerts.alert(f"labelloop checkpoint {p}: not on this host and no hub to pull it from")
            return None
        from ...data import peer_addr
        local.parent.mkdir(parents=True, exist_ok=True)
        try:
            r = subprocess.run(["rsync", "-a", "--partial", f"{peer_addr(hub)}:{p}", str(local)],
                               capture_output=True, text=True, timeout=1800)
        except (OSError, subprocess.SubprocessError) as e:
            _alerts.alert(f"labelloop checkpoint pull from {fleet.hub_host} failed: {e}")
            return None
        if r.returncode != 0 or not local.exists():
            _alerts.alert(f"labelloop checkpoint pull from {fleet.hub_host} failed: {r.stderr[-300:]}")
            return None
    if sha:
        from .. import labelloop as LL
        got = LL.sha_of(local)
        if got != sha:
            _alerts.alert(f"labelloop checkpoint {local}: sha {got} != registered {sha}; not used")
            return None
    return str(local)


def describe(arm: str | None = None) -> dict:
    c = _registry_lookup(arm or "")
    return {k: c.get(k) for k in ("name", "arch", "receptive_field", "step", "auc",
                                  "boundary_f1", "false_stroke", "labels_px", "sha", "parent")} if c else {}


def available(fleet=None, arm: str | None = None) -> tuple[bool, str]:
    if not arm:
        return False, "labelloop needs a checkpoint name: labelloop:dense, labelloop:narrow32"
    p = checkpoint_path(arm)
    if p is None:
        return False, f"labelloop:{arm} is not registered on this host"
    return True, p


def predict(layers_dir: str, mask_png: str, out_png: str, gpu: int = 0, arm: str | None = None,
            **spec) -> bool:
    ok, why = available(arm=arm)
    if not ok:
        _alerts.alert(f"labelloop on this host: {why}")
        return False
    # one inference implementation, selected by checkpoint: the arch is read from the file
    # by the dense loader, so a narrow32 export needs no separate path here.
    # EVERY variable is restored afterwards: a prospect worker is one long-lived process, and a
    # VPIPE_GP_DENSE_CKPT left pointing at this export made the next plain `grandprize_dense`
    # job in that worker silently run the labelloop weights under the production name.
    keys = ("VPIPE_GP_DENSE", GD.ENV_CKPT, "VPIPE_GP_DENSE_CKPT_" + str(arm).upper())
    saved = {k: os.environ.get(k) for k in keys}
    os.environ["VPIPE_GP_DENSE"] = "1"
    os.environ[GD.ENV_CKPT] = why
    os.environ.pop(keys[2], None)
    try:
        return GD.predict(layers_dir, mask_png, out_png, gpu=gpu, arm=None, **spec)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
