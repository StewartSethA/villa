"""Ink-detection model families that are neither the Grand Prize TimeSformer nor unet3d.

Every module here exposes the same two things, so the finish stage and the Manual Ink
Prospecting sweep can treat a family as data rather than as code:

    SPEC     -- the input contract the checkpoint was trained under (layer count and
                which layers, um/px, normalisation, tile/stride). It lives BESIDE the
                weights, not in the caller, because a checkpoint run at the wrong
                layer window or the wrong scale still produces a plausible-looking
                image -- the failure is silent (FINDINGS 36: 18 % off scale cost
                0.02-0.03 AUC and nothing errored).
    predict(layers_dir, mask_png, out_png, gpu, **spec) -> bool

and a pair of gate functions mirroring stages/ink3d.py's `*_enabled` / `*_available`,
so a family with no weights on this host is SKIPPED with a reason rather than failing.

Registered families:
    grandprize_dense -- the Grand Prize TimeSformer backbone with a dense pixel-shuffle
           head in place of the shipped 4x4 CLS-token head: output stride 1 px instead
           of 16, at roughly the shipped cost. DEFAULT OFF (VPIPE_GP_DENSE=1).
    pixelclf -- a from-scratch per-pixel dense head built to approximate the patch-size
           study (`pixelclf:32x13`, `pixelclf:16x13`); its checkpoints never converged
           (narrow32 chance, narrow16 weak -- pixelclf recovery, 2026-09-30). DEFAULT OFF
           (VPIPE_PIXELCLF=1).
    patchclf -- the patch-size study's ACTUAL window classifier (`scripts/patchsize/
           models.py`), run convolutionally / sliding-window at a measured stride
           (`patchclf:32x13`, `patchclf:16x13`). Reproduces FINDINGS section 66's published
           numbers (0.739 within-scroll, 0.629 cross-scroll LOSO at 32x13) because it is
           now trained WITH `--save-models`, unlike pixelclf's checkpoints. DEFAULT OFF
           (VPIPE_PATCHCLF=1).
    colmean -- UNTRAINED per-pixel scores over the rendered depth column (`colmean:pm8`,
           `colmean:pm16`, `colstd:std_pm16`, `coldiffabs:diffabs_pm4`, `surface`). No
           checkpoint exists: each is a closed form, so its held-out AUC on a scroll it
           has never seen is not a transfer loss but the same measurement. CPU numpy,
           seconds per cm2, shares whatever render it is handed.
    ink9um_student -- a ~3.7 M-parameter 2-D U-Net distilled from the ink_9um teacher on its own
           17-layer, ~9.5 um/px input; tiles of 2048 px with a halo instead of a 50 %-overlap
           grid. Reads the ink9um render. DEFAULT OFF.
    k1 -- UNTRAINED consensus of C1 (stroke_ring's own band/scale/ring geometry, median/MAD
           standardised) and an edge-aligned band (7.2 um inside the ink-side half-max sheet
           edge). First untrained arm measured to beat C1 on unseen objects (FINDINGS Sec
           122.1 part 3). No checkpoint. DEFAULT OFF pending production-code reproduction.
    hecate -- the upstream staff model scrollprize/hecate (ResNet-152 3-D + learned depth attention), run
           through the unmodified upstream hecate.py beside its weights; 9.6 um isotropic input (our
           one-voxel planes resampled in depth). DEFAULT OFF.
    reader_v2 -- DomRusso2's Reader v2 (MIT): an ink_9um-architecture checkpoint run through the ink9um
           family's own wrapper with only the checkpoint swapped (byte-identical to the author's reference
           CLI on a test tile). DEFAULT OFF.
    reader_v2_dense -- Reader v2 distilled into a small-RF (59 px support) dense U-Net student. DEFAULT OFF.
    i3d -- InceptionI3d + conv decoder, the Grand Prize winner repo's
           `inference_resnet3d.py` path (enc='i3d'). Our Scroll 4 fine-tunes are this
           architecture, NOT the TimeSformer: their state_dict keys are
           `backbone.Conv3d_1a_7x7.*` / `decoder.logit.*`, which the TimeSformer
           wrapper cannot load.
"""
from __future__ import annotations

from .registry import (INK_BY_NAME, INK_FAMILIES, InkFamily,  # noqa: F401
                       ink_family)
from . import i3d               # noqa: F401
from . import resnet3d_1667    # noqa: F401
from . import grandprize_dense  # noqa: F401
from . import pixelclf        # noqa: F401
from . import patchclf       # noqa: F401
from . import labelloop      # noqa: F401
from . import colmean       # noqa: F401
from . import stroke_ring  # noqa: F401
from . import ink9um_student  # noqa: F401
from . import k1            # noqa: F401
from . import hecate        # noqa: F401
from . import reader_v2     # noqa: F401
from . import reader_v2_dense  # noqa: F401
from . import ink9um        # noqa: F401  (legacy family's memstack reader)

FAMILIES = {"i3d": i3d, "resnet3d_1667": resnet3d_1667, "grandprize_dense": grandprize_dense,
            "pixelclf": pixelclf, "patchclf": patchclf, "labelloop": labelloop, "colmean": colmean,
            "stroke_ring": stroke_ring, "ink9um_student": ink9um_student, "k1": k1,
            "hecate": hecate, "reader_v2": reader_v2, "reader_v2_dense": reader_v2_dense, "ink9um": ink9um}

def ink_family(name: str) -> InkFamily | None:
    """The registry row for a planned model name, or None when it is not a zoo family
    (`grandprize`, `ink9um`, `unet3d`, `warp*` are handled in stages/ink.py)."""
    return INK_BY_NAME.get(name)


def training_frame(name: str):
    """The Frame family `name` was trained in -- its scale contract, declared beside its
    weights rather than reconstructed by the caller."""
    return FAMILIES[name].training_frame()


def available(name: str, fleet) -> tuple[bool, str]:
    # `family:arm` selects a named variant (grandprize_dense:narrow32); the bare family name
    # keeps working, so nothing that already calls this changes.
    base, _, arm = name.partition(":")
    m = FAMILIES.get(base)
    if m is not None and arm:
        try:
            return m.available(fleet, arm=arm)
        except TypeError:
            return False, f"{base} does not take a named arm ({arm!r})"
    name = base
    if m is None:
        return False, f"unknown ink model family {name!r}"
    return m.available(fleet)


def enabled_families() -> tuple[str, ...]:
    return tuple(n for n, m in FAMILIES.items() if m.enabled())
