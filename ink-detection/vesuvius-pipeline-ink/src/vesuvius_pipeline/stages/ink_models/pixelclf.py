"""pixelclf: per-pixel ink classifiers with a bounded receptive field, named `pixelclf:<W>x<D>`.

The patch-size study (`scripts/patchsize/`) measured the cross-scroll optimum at about
32x32x13 and 64x64x4 -- small windows generalise, wide deep ones memorise stroke texture
and collapse across scrolls (FINDINGS 70). Wrapping "the study's best classifier" was the
obvious thing to do and IT CANNOT BE DONE AS SUCH: the grid trains and scores each cell and
never writes weights -- `grep -c torch.save scripts/patchsize/*.py` is 0, and no `.pt`
exists under the campaign's output on either box. There is nothing to load.

So this family is that classifier REBUILT as a dense head and trained here:
`scripts/hires_ink/narrow_head.py` is a small 3D stem over 13 depth layers plus a counted
stack of 3x3 convolutions, one logit per pixel, with an in-plane receptive field MEASURED
by backprop (`measure_receptive_field`) rather than asserted -- 31 px for `pixelclf:32x13`
and 17 px for `pixelclf:16x13`. Same 13-layer depth as the study's optimum; the window is
the receptive field rather than a cropped tile, which is what makes the output per-pixel
instead of per-window.

Checkpoints are the `narrow32` / `narrow16` arms exported by
`scripts/hires_ink/export_models.py`, and they are PRELIMINARY -- the step count is in the
file name and in the `.metrics.json` beside the weights.

DEFAULT OFF: `VPIPE_PIXELCLF=1`.
"""
from __future__ import annotations

import os

from . import grandprize_dense as GD
from ... import alerts as _alerts

# name -> the exported arm that implements it
ARMS = {"32x13": "narrow32", "16x13": "narrow16"}
SPEC = dict(GD.SPEC, arch="narrow_conv_per_pixel", output_stride_px=1,
            note="receptive field measured by backprop: 31 px (32x13) / 17 px (16x13), 13 layers")


def training_frame():
    return GD.training_frame()


def check_frame(frame, tol: float = 0.05) -> None:
    GD.check_frame(frame, tol=tol)


def enabled() -> bool:
    return os.environ.get("VPIPE_PIXELCLF", "0") == "1"


def _arm(name: str | None) -> str | None:
    if name is None:
        return None
    return ARMS.get(name, name if name in ARMS.values() else None)


def available(fleet=None, arm: str | None = None) -> tuple[bool, str]:
    """Are the weights here -- see grandprize_dense.available() for why this no longer
    fails closed on VPIPE_PIXELCLF. The finish stage gates on
    `finish.pixelclf_<arm>_enabled` in the hub DB; a prospect job gates on being named."""
    a = _arm(arm)
    if a is None:
        return False, f"pixelclf arm {arm!r} unknown; known: {sorted(ARMS)}"
    p = GD.checkpoint_path(a)
    if p is None:
        return False, (f"pixelclf:{arm} checkpoint not on this host "
                       f"(expected grandprize_dense_{a}.ckpt in var/models)")
    return True, p


def describe(arm: str | None = None) -> dict:
    return GD.describe(_arm(arm))


def predict(layers_dir: str, mask_png: str, out_png: str, gpu: int = 0, arm: str | None = None,
            **spec) -> bool:
    ok, why = available(arm=arm)
    if not ok:
        _alerts.alert(f"pixelclf on this host: {why}")
        return False
    os.environ.setdefault("VPIPE_GP_DENSE", "1")     # the loader is the dense-head one
    return GD.predict(layers_dir, mask_png, out_png, gpu, arm=_arm(arm), **spec)
