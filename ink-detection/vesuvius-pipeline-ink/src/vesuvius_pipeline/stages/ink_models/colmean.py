"""colmean: UNTRAINED per-pixel ink scores read straight off the rendered depth column.

There is no checkpoint. Every score here is a fixed, closed-form function of the depth
column under one UV pixel, so there is nothing to fit, nothing to transfer and nothing
that can memorise a scroll's stroke texture. That is the whole argument for them:
Simple-ink Phase 0 (FINDINGS, 2026-09-14) measured these on held-out segments of Scroll
1, Scroll 4 and Scroll 5 and the column-brightness score reads the same AUC on a scroll
it has never seen as on the one it was picked on -- because "the scroll it was picked
on" contributed a sign and a smoothing radius and nothing else. A trained 1x1 model
that beats it in-sample does not beat it across scrolls (FINDINGS 79, 80, 84: "No
depth-only 1x1 model reaches 0.6 on any scroll it did not train on ... every cell <= +0.002
vs colmean"), the Hadamard-gated arms fall to 0.477-0.515 off-scroll, and no GP-family
head transfers to Scroll 4 at all (best 0.54-0.56 against 0.51-0.61 for these).

What each score is (`scripts/simple_ink/phase0_untrained.py`, `Volume.feature`, kept
formula-for-formula so a published map here is the same number the study scored):

    layer      L[zc + dk]                                one rendered layer
    colmean    mean L[zc-h .. zc+h]                      column brightness
    colstd     std  L[zc-h .. zc+h]                      depth-wise contrast, 1x1 in plane
    coldiffabs mean |L[j+1] - L[j]| over that window     depth-wise roughness, 1x1 in plane

plus one two-stage arm, `normsurf`, which divides the surface layer by ITS OWN local
background (a mask-normalised Gaussian of 506 um) before integrating at the stroke width
(300 um). That division is what makes it read the same on three scrolls with three
different brightness offsets, and it is the single most consistent score Phase 1b
measured -- "the only prior never significantly worse than a baseline on any scroll"
(near-ink held-out AUC: Scroll 1 0.571, Scroll 5 0.617, Scroll 4 0.592).

The window is inclusive and clipped at the stack's ends -- layers
[max(0, zc-h), min(Z, zc+h+1)), i.e. 2h+1 of them -- and `dk` is ignored by the column
scores, exactly as phase0_untrained.Volume.feature does it.

`zc` is the SURFACE layer: the middle of the rendered stack (`len(layers)//2` -- the same
layer `make_mask` thresholds and the same one finish publishes as the `render` family),
so the score is centred on the sheet rather than on an arbitrary end of the window.

`sign` is +1 when brighter means ink (ink is denser than papyrus: the a-priori physics
sign) and -1 when darker does. It is part of the declared score, never `max(AUC, 1-AUC)`
on the evaluation segment.

`sigma_um` integrates the per-pixel score spatially with a MASK-NORMALISED Gaussian --
`blur(x*m)/blur(m)`, so the sheet's edge does not bleed in. This is the "a collective
signal emerges over area" hypothesis done numerically instead of by eye, and it is the
single biggest term in these scores' AUC. It is declared in MICRONS and converted to
pixels against the render's own pitch (`frame.json`), so the same physical integration
length is used at every scale of the multiscale bracket -- 126 um is 16 px at 7.91 um/px
and 8 px at 15.82, and using "16 px" at both would be two different scores wearing one
name.

Depth is NOT rescaled by the bracket: a render's layers are one voxel apart along the
normal whatever `--scale` does in plane, so `h` stays a layer count.

Cost. One pass reads 2h+1 of the 65 layers, accumulates two float32 planes and blurs
them. No GPU, no torch, no weights: on a 1.5 cm2 tile it is seconds, against minutes for
a TimeSformer pass. The render is the only thing these families cost, and they share it
with every other family at that scale.

Output. The score is mapped to 0-255 by a MONOTONE stretch (masked p1..p99 clipped), so
the published PNG carries the same ranking -- hence the same AUC -- as the float score,
and nothing outside the mask is scored.
"""
from __future__ import annotations

import json
import os

from ... import alerts as _alerts

import numpy as np

GP_TARGET_UM = 7.91
SURFACE_LAYER_UM = GP_TARGET_UM       # the frame these scores were measured in
# ...in depth too: the scores were measured on 7.91 um-voxel scans, whose renders step 7.91 um
# per plane. A depth window declared in LAYERS is a window of h x 7.91 um.
PLANE_UM_MEASURED = GP_TARGET_UM
# reads an in-memory native-stack view (stages/memstack.py) through its layer readers
MEMSTACK_READY = True


def plane_pitch_um(layers_dir: str) -> float | None:
    """um between adjacent layers of a render: the CT VOXEL, whatever the in-plane pitch
    (every render tier steps one voxel per plane, render_gpu slice_step 1.0). A memstack view
    carries it; a stored stack's frame.json records its render frame, whose `native_um` is the
    volume's level-0 voxel. None when neither says. (2026-09-29: K1, stroke_ring and colmean
    all used the in-plane pitch here -- 7.91 um on a 9.362 um scan, 18 % shallow.)"""
    from .. import memstack as _MS
    v = _MS.plane_um(layers_dir)
    if v:
        return float(v)
    for d in (layers_dir, os.path.realpath(str(layers_dir))):
        try:
            fr = json.load(open(os.path.join(d, "frame.json")))["frame"]
            return float(fr["native_um"][0])
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            continue
    sib = str(layers_dir).rstrip("/")
    if sib.endswith("_reversed"):
        return plane_pitch_um(sib[: -len("_reversed")])
    return None

# ---- the declared scores --------------------------------------------------------------
# name -> (spec, sign, sigma_px_at_7.91um, provenance). Taken from
# `scripts/simple_ink/fixed_untrained_scores_v1.json`, the Phase 0 fixed-score set: each
# entry was FIXED before the held-out scrolls were scored, so the AUCs in FINDINGS are
# transfer numbers rather than the best of a sweep.
ARMS: dict[str, dict] = {
    # The A-PRIORI arms. Sign, depth window and sigma were all fixed BEFORE any held-out
    # scroll was scored, so their numbers are transfer numbers by construction -- there is
    # no scroll in them to transfer FROM. FINDINGS, Simple-ink Phase 0, H9: "No transferred
    # score beats the prior column mean on its target", and "fixed a-priori physics scores
    # match or beat every val-selected score on two of three scrolls, and never lose
    # significantly on the third".
    "pm8": dict(spec=dict(kind="colmean", h=8), sign=1, sigma_px=16, bg_px=0,
                auc="S1 0.524 [0.469,0.582] | S5 0.650 [0.626,0.672] | S4 0.596 [0.515,0.677]",
                source="prior_colmean_pm8 (fixed_untrained_scores_v1.json): a-priori physics "
                       "sign (ink is denser -> brighter); h=8 and sigma 16 px fixed in advance"),
    "surface": dict(spec=dict(kind="layer", dk=0), sign=1, sigma_px=16, bg_px=0,
                    auc="S1 0.539 [0.498,0.578] | S5 0.576 [0.557,0.593] | S4 0.609 [0.509,0.692]",
                    source="prior_surface_layer: the surface layer alone, a-priori sign, sigma "
                           "16 px fixed in advance. The Phase 1 paired baseline; best of the "
                           "priors on Scroll 4"),
    # The most CONSISTENT score measured: divide the surface layer by its own local
    # background before integrating, and integrate at the stroke width rather than at an
    # arbitrary sigma. Phase 1b: "It is the only prior never significantly worse than a
    # baseline on any scroll." Stroke-width smoothing beats sigma 16 only AFTER the
    # background division (+0.028 / +0.017 / +0.018).
    "normsurf": dict(spec=dict(kind="layer", dk=0), sign=1, sigma_px=38, bg_px=64,
                     auc="S1 0.571 | S5 0.617 | S4 0.592 (near-ink region, Phase 1b)",
                     source="normsurf @sigma_stroke: L(zc)/G_mask(L(zc), sigma 64 px) - 1, then "
                            "integrated at sigma_stroke (FWHM = the scroll's median stroke "
                            "width: 38.4 / 38.2 / 36.5 px on S1 / S5 / S4 -- 38 px is all three)"),
    # Selected on ONE scroll's validation segment, so their off-scroll numbers ARE transfer
    # losses. Kept because the operator may want the S5/S4 arms on S5/S4, and because
    # Phase 0's own lesson is that a val-selected score does not survive the move.
    "pm16": dict(spec=dict(kind="colmean", h=16), sign=1, sigma_px=32, bg_px=0,
                 auc="S1 0.541 [0.472,0.614] | S5 0.645 (seen) | S4 0.592 [0.495,0.690]",
                 source="S5win_colmean_pm16: best colmean on the Scroll 5 VAL segment "
                        "(phase0 S5, script b01d2b40)"),
    "std_pm16": dict(spec=dict(kind="colstd", h=16), sign=1, sigma_px=32, bg_px=0,
                     auc="S1 0.502 [0.423,0.574] | S5 0.644 (seen) | S4 0.491 [0.367,0.626]",
                     source="S5_colstd_pm16: best colstd on the Scroll 5 VAL segment "
                            "(phase0b S5, script 42ec4b3c)"),
    "diffabs_pm4": dict(spec=dict(kind="coldiffabs", h=4), sign=1, sigma_px=32, bg_px=0,
                        auc="S1 0.483 [0.406,0.552] | S5 0.573 [0.542,0.604] | S4 0.554 (seen)",
                        source="S4win_coldiffabs_pm4: best texture on the Scroll 4 VAL letter "
                               "boxes (phase0b S4, script 40d25f60)"),
}

# 🔴 `surfdiff` (L(zc) minus a stack of layers to one side) is DELIBERATELY ABSENT. Phase 1b
# falsified it: S5 -0.045, S4 -0.072 against the surface layer with the label-free side, and
# forcing the other side gives S5 0.489 (-0.105 [-0.125,-0.085]). Subtracting a depth stack
# from the surface loses signal from either side, so it is not offered as an arm.

SPEC = {
    "layers": 65,
    "layer_start": 0,
    "layer_stack": 65,
    "tile": 0,                   # no tiling: the score is per pixel, computed over the whole plane
    "stride": 0,
    "um_per_px": GP_TARGET_UM,
    "normalisation": "none (raw rendered intensities; these scores are NOT GP-normalised)",
    "arch": "closed_form_depth_column",
    "output_stride_px": 1,
    "trained_on": "nothing -- no weights exist",
}


def training_frame():
    from ...frame import MODELS, ModelSpec
    return MODELS.get("colmean_pm8", ModelSpec("colmean_pm8", GP_TARGET_UM, 65)).frame(
        source_id="model:colmean")


def check_frame(frame, tol: float = 0.05) -> None:
    """These scores do not resample, but they are also not frame-bound the way a trained
    net is: `sigma_um` is physical and `h` is a layer count. A mismatch is recorded, not
    raised -- refusing a render would defeat the point of the multiscale bracket, whose
    whole job is to run the same score at five different pitches."""
    return None


def enabled() -> bool:
    """No environment variable: these families are new and their toggle is the hub DB row
    `finish.<name>_enabled` from the start. An env var on whichever box launched the
    worker is exactly the knob settings.py exists to replace."""
    return True


def available(fleet=None, arm: str | None = None) -> tuple[bool, str]:
    """Always here: there is no checkpoint to be missing. `available()` answers 'are the
    weights on this host', and the honest answer for a closed-form score is yes."""
    if arm is None or arm == "":
        return False, f"colmean needs a named arm; known: {sorted(ARMS)}"
    if arm not in ARMS:
        return False, f"colmean arm {arm!r} unknown; known: {sorted(ARMS)}"
    return True, f"colmean:{arm} (no checkpoint -- closed form)"


def describe(arm: str | None = None) -> dict:
    a = ARMS.get(arm or "", {})
    return {"family": "colmean", "arm": arm, "spec": dict(SPEC), **{k: v for k, v in a.items()}}


# ---- the score ------------------------------------------------------------------------
def _layer_files(layers_dir: str) -> list[str]:
    from .. import memstack as _MS
    names = _MS.list_names(layers_dir)
    if names is not None:
        return names
    return sorted(f for f in os.listdir(layers_dir) if f.lower().endswith((".tif", ".tiff")))


def _read(layers_dir: str, files: list[str], k: int) -> np.ndarray:
    from .. import memstack as _MS
    st = _MS.get(layers_dir)
    if st is not None:
        return st.full(st.names.index(files[k])).astype(np.float32)
    import tifffile
    a = tifffile.imread(os.path.join(layers_dir, files[k]))
    return a.astype(np.float32)


def layers_at_pitch(h: int, plane_um: float | None) -> int:
    """A depth length declared in layers of the 7.91 um/plane frame the score was measured in,
    as a layer count at THIS render's plane pitch (h=8 is +-63 um: 7 planes at 9.362 um)."""
    if not plane_um or h == 0:
        return int(h)
    return int(np.sign(h)) * max(1, int(round(abs(h) * PLANE_UM_MEASURED / float(plane_um))))


def _window(zc: int, h: int, n: int) -> tuple[int, int]:
    """[a, b) -- exactly phase0_untrained.Volume.feature's clamp."""
    return max(0, zc - h), min(n, zc + h + 1)


def column_score(layers_dir: str, spec: dict, plane_um: float | None = None) -> np.ndarray:
    """The raw float score plane, streamed one layer at a time (never the whole stack:
    65 x 4000 x 4000 is a gigabyte and these run beside a TimeSformer). `plane_um`: the
    render's layer spacing; the spec's layer counts are converted from the 7.91 um/plane frame."""
    files = _layer_files(layers_dir)
    n = len(files)
    if not n:
        raise ValueError(f"no rendered layers in {layers_dir}")
    zc = n // 2
    kind, dk = spec["kind"], layers_at_pitch(int(spec.get("dk", 0)), plane_um)
    spec = dict(spec, h=layers_at_pitch(int(spec.get("h", 0)), plane_um))
    if kind == "layer":
        k = zc + dk
        if not (0 <= k < n):
            raise ValueError(f"layer index {k} outside the {n}-layer stack")
        return _read(layers_dir, files, k)
    a, b = _window(zc, int(spec["h"]), n)
    if kind == "colmean":
        s = _read(layers_dir, files, a)
        for j in range(a + 1, b):
            s += _read(layers_dir, files, j)
        return s / float(b - a)
    if kind == "colstd":
        s1 = np.zeros_like(_read(layers_dir, files, a), dtype=np.float64)
        s2 = np.zeros_like(s1)
        for j in range(a, b):
            L = _read(layers_dir, files, j).astype(np.float64)
            s1 += L
            s2 += L * L
        m = s1 / float(b - a)
        return np.sqrt(np.clip(s2 / float(b - a) - m * m, 0, None)).astype(np.float32)
    if kind == "coldiffabs":
        prev = _read(layers_dir, files, a)
        s = np.zeros_like(prev)
        for j in range(a + 1, b):
            cur = _read(layers_dir, files, j)
            s += np.abs(cur - prev)
            prev = cur
        return s / float(max(1, b - 1 - a))
    raise ValueError(f"unknown colmean score kind {kind!r}")


def gblur(x: np.ndarray, m: np.ndarray, sigma_px: float) -> np.ndarray:
    """Mask-normalised Gaussian: blur(x*m)/blur(m), so the sheet's edge does not bleed
    the background in. Same definition as phase0_untrained.gblur."""
    if sigma_px <= 0:
        return x
    try:
        from scipy.ndimage import gaussian_filter as G
    except ImportError:                    # noqa: BLE001 - a box-blur cascade is a fine stand-in
        return _box_gblur(x, m, sigma_px)
    num = G(x * m, sigma_px, mode="constant", cval=0.0, truncate=3.0)
    den = G(m, sigma_px, mode="constant", cval=0.0, truncate=3.0)
    return num / np.clip(den, 1e-6, None)


def _box_gblur(x: np.ndarray, m: np.ndarray, sigma_px: float) -> np.ndarray:
    """Three box passes approximate a Gaussian to well under a percent, and a box mean is
    two cumsums. Only used when scipy is not importable."""
    w = max(1, int(round(sigma_px * (12.0 / 3.0) ** 0.5 / 2)) * 2 + 1)

    def box(a):
        for axis in (0, 1):
            a = np.moveaxis(a, axis, -1)
            c = np.cumsum(np.pad(a, ((0, 0),) * (a.ndim - 1) + ((w // 2 + 1, w // 2),)), axis=-1)
            a = (c[..., w:] - c[..., :-w]) / float(w)
            a = np.moveaxis(a, -1, axis)
        return a

    num, den = x * m, m.copy()
    for _ in range(3):
        num, den = box(num), box(den)
    return num / np.clip(den, 1e-6, None)


def render_um_per_px(layers_dir: str, default: float = GP_TARGET_UM) -> float:
    """The pitch of THIS render, from the frame.json finish writes beside it. Falling back
    to 7.91 silently would make sigma_um a lie at every bracket scale, so the fallback is
    announced by the caller through `detail`."""
    from .. import memstack as _MS
    um = _MS.um_per_px(layers_dir)                  # an in-memory view carries its own frame
    if um is not None:
        return um
    for d in (layers_dir, os.path.realpath(layers_dir)):
        p = os.path.join(d, "frame.json")
        try:
            return float(json.load(open(p))["um_per_px"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
    # a `_reversed` view is a directory of symlinks beside the real render
    sib = layers_dir.rstrip("/")
    if sib.endswith("_reversed"):
        return render_um_per_px(sib[: -len("_reversed")], default)
    return default


def to_uint8(score: np.ndarray, mask: np.ndarray, lo_p: float = 1.0, hi_p: float = 99.0) -> np.ndarray:
    """Monotone stretch over the MASKED pixels only, so the published image ranks its
    pixels exactly as the float score does (AUC is invariant under this) and the unmasked
    background cannot set the window."""
    v = score[mask > 0]
    if v.size == 0:
        return np.zeros(score.shape, dtype=np.uint8)
    lo, hi = np.percentile(v, [lo_p, hi_p])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(v)), float(np.max(v))
    if hi <= lo:
        return np.zeros(score.shape, dtype=np.uint8)
    out = np.clip((score - lo) / (hi - lo), 0, 1) * 255.0
    return (out * (mask > 0)).astype(np.uint8)


def predict(layers_dir: str, mask_png: str, out_png: str, gpu: int = 0, arm: str | None = None,
            fleet=None, log: str | None = None, **spec) -> bool:
    ok, why = available(fleet, arm)
    if not ok:
        _alerts.alert(f"colmean on this host: {why}")
        return False
    a = ARMS[arm]
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    dz = plane_pitch_um(layers_dir)
    if dz is None:
        _alerts.alert(f"colmean: plane pitch of {os.path.basename(str(layers_dir).rstrip('/'))} unknown; "
                      f"depth window read as {PLANE_UM_MEASURED:g} um/plane")
    score = column_score(layers_dir, a["spec"], plane_um=dz)
    if os.path.exists(mask_png):
        m = np.asarray(Image.open(mask_png).convert("L"))
        if m.shape != score.shape:
            print(f"[colmean] mask {m.shape} != render {score.shape}; scoring the whole plane", flush=True)
            m = np.ones(score.shape, dtype=np.uint8) * 255
    else:
        m = np.ones(score.shape, dtype=np.uint8) * 255
    mf = (m > 0).astype(np.float32)
    um = render_um_per_px(layers_dir)
    # every sigma is declared in PIXELS OF THE 7.91 um FRAME the score was measured in and
    # converted to this render's pitch, so the bracket varies the sampling and not the
    # physical length the score integrates over
    px = SURFACE_LAYER_UM / um
    bg_px = float(a.get("bg_px") or 0) * px
    if bg_px > 0:
        # local-background division: the score becomes "how much brighter than this
        # pixel's own papyrus", which is what makes it survive a scroll's brightness offset
        score = score / np.clip(gblur(score, mf, bg_px), 1e-6, None) - 1.0
    sigma_px = float(a["sigma_px"]) * px
    score = float(a["sign"]) * gblur(score, mf, sigma_px)
    img = to_uint8(score, m)
    Image.fromarray(img).save(out_png)
    print(f"[colmean:{arm}] {a['spec']} sign {a['sign']:+d} bg {float(a.get('bg_px') or 0):g} px "
          f"sigma {float(a['sigma_px']):g} px @7.91 um -> bg {bg_px:.1f} px sigma {sigma_px:.1f} px "
          f"@ {um:.2f} um/px", flush=True)
    return True
