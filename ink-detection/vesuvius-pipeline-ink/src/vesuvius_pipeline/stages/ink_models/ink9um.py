"""ink9um (the team's ink_9um hybrid_3d2d, checkpoint step-075000): the LEGACY family, given a memstack reader.

The family has no registry row: stages/ink.py runs it through `run_ink9um` (ink_dashboard/run_ink9um.py -> villa's
`vesuvius.ink_detection.inference.infer`). In NATIVE-ONLY mode (finish.native_render_enabled, user directive 2026-09-29:
no resampled stacks on disk) the dispatcher hands an in-memory view key and refused it, because that subprocess reads layer
FILES ("env-fault: native-only mode ... ink9um ... cannot take an in-memory view", 2026-10-08, one segment).

This module is the same move reader_v2 makes (same architecture, same CLI, same 17 centred planes): MATERIALISE the CLI's own
17 planes of the view as TIFFs in a temp dir beside out_png (the work disk, never /dev/shm, D18), run the SAME wrapper on that
directory with the DEFAULT checkpoint (no INK9UM_CKPT), delete the directory. Single-source: the plane selection and the
materialiser are reader_v2's (`centred`, `_materialise`); nothing here re-implements them. Forward face only: the dispatcher
reverses the view for the `_reversed` variants before calling predict, so predict always runs `--direction forward`.

`enabled()` is False on purpose: the family is planned by ink.py's own list, not by the zoo's environment switches.
"""
from __future__ import annotations

import os

MEMSTACK_READY = True
TARGET_UM = 9.5
DEPTH = 17

SPEC = {"layers": DEPTH, "um_per_px": TARGET_UM, "arch": "vesuvius_unet_3d_stem_2d (= ink_9um hybrid_3d2d)",
        "patch": [DEPTH, 128, 128], "checkpoint": "default (checkpoints/ink_9um/hybrid_3d2d-seed42/step-075000.pth)"}


def enabled() -> bool:
    return False


def available(fleet=None, arm: str | None = None) -> tuple[bool, str]:
    """The wrapper resolves its own default checkpoint; availability is the wrapper's, answered at run time."""
    return True, "default ink_9um checkpoint (resolved by run_ink9um)"


def training_frame():
    from ...frame import ModelSpec
    return ModelSpec("ink9um", TARGET_UM, DEPTH, tile_px=128).frame(source_id="model:ink9um")


def predict(layers_dir: str, mask_png: str | None, out_png: str, gpu: int = 0, arm: str | None = None,
            fleet=None, log: str | None = None, **spec) -> bool:
    import shutil
    from ... import config
    from .. import ink as INK
    from .. import memstack as MS
    from . import reader_v2 as RV2
    if fleet is None:
        fleet = config.load()
    seg = "ink9um_" + os.path.basename(os.path.dirname(os.path.abspath(out_png)))
    log = log or os.path.splitext(out_png)[0] + ".log"
    tmp, src, um = None, layers_dir, None
    try:
        if MS.list_names(layers_dir) is not None:
            tmp, um = RV2._materialise(layers_dir, os.path.dirname(os.path.abspath(out_png)))
            src = tmp
        ok, _detail = INK.run_ink9um(fleet, seg, src, out_png, str(gpu), TARGET_UM, "forward", log, um_per_px=um)
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    return bool(ok)
