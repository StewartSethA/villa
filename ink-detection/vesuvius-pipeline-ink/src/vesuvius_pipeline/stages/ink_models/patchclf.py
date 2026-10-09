"""patchclf: FINDINGS section 66's patch-size study window classifier, run convolutionally.

`pixelclf.py` already documents why it could not simply wrap "the study's best classifier":
`scripts/patchsize/run_grid.py` trains and scores every (W, D) grid cell and never called
`torch.save` -- there was nothing to load. `pixelclf.py` is instead a from-scratch per-pixel
dense head (`narrow_head.Narrow`) that never reached the study's quality (narrow32 AUC 0.477 =
chance at its final export step, narrow16 0.537 -- pixelclf recovery writeup,
docs/experiments/pixelclf_recovery/STATE.md, 2026-09-30).

This family is the thing the study actually measured: a small per-CENTRE-PIXEL classifier
(`scripts/patchsize/models.py::build`, a plain Conv2d/BatchNorm/MaxPool stack reading D depth
layers as channels and a W x W in-plane crop, global-average-pooled to one logit) -- trained
with `run_grid.py --save-models` (added 2026-09-30 for this recovery) so the weights now exist.
Published numbers it is reproducing (FINDINGS section 66, `docs/experiments/
sensitivity_11_patchsize.json`): W=32,D=13 within-scroll (Scroll 5) 0.7390, cross-scroll
(trained Scroll 5+1, held out Scroll 4, the LOSO arm) 0.6294; W=16,D=13 0.6987 / untested LOSO.

THE OUTPUT IS A SLIDING-WINDOW CLASSIFICATION, not a dense segmentation head: the trained
network only ever says "is the CENTRE pixel of this W x W x D crop ink", so producing a map
means running it at many centres and painting each one's answer over `stride` px -- the same
thing the shipped Grand Prize family does at a 16 px cell (`grandprize.py`'s own 4x4-from-CLS-
token head), just at a finer, measured stride. This is NOT the dense per-pixel supervision
`grandprize_dense` trains against; every pixel inside one stride cell gets the SAME score.

DEFAULT OFF: `VPIPE_PATCHCLF=1`. `available()` refuses when the checkpoint is not on the
host, same convention as every other family here.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from ... import alerts as _alerts

IN_UM_PER_PX = 7.91
N_LAYERS = 65                      # the rendered stack this was trained against (patchsize.data.N_LAYERS)

# arm -> (W, D, output stride in px). Stride is a quarter of the window: finer than the
# shipped Grand Prize's 16 px cell at W=32, and the cheapest stride that does not paint most
# of the window at one answer when W=16.
ARMS = {"32x13": {"W": 32, "D": 13, "stride": 8},
        "16x13": {"W": 16, "D": 13, "stride": 4}}

# Checkpoints are the {tag}_W{W}_D{D}_s{seed}.pt files `run_grid.py --save-models` writes.
# `TAG` picks the JOINT (Scroll 1 + Scroll 5 trained, Scroll 4 held out) run by default: it is
# the one arm with an honest CROSS-scroll number (0.6294 at 32x13) rather than only a
# within-scroll one, which is what a production family run on an unseen scroll needs.
TAG = os.environ.get("VPIPE_PATCHCLF_TAG", "joint_loso_scroll4_recovery")
MODEL_ROOTS = ("var/models",
               "/dev/shm/vpipe/ScrollPrizeTutorial/current/var/models",
               "/dev/shm/vpipe/var/models")

SPEC = {
    "layers": None,                 # set per arm (D) in checkpoint_spec()
    "um_per_px": IN_UM_PER_PX,
    "normalisation": "patchsize_global_mu_sd",   # carried IN the checkpoint, not asserted
    "arch": "patchsize_convnet_centre_classifier",
    "output_stride_px": None,       # set per arm
    "trained_on": "PHercParis4 (Scroll 1) + PHerc0172 (Scroll 5), mesh-disjoint, LOSO Scroll 4 held out",
    "source": "scripts/patchsize/run_grid.py --save-models (FINDINGS section 66)",
}


def enabled() -> bool:
    return os.environ.get("VPIPE_PATCHCLF", "0") == "1"


def _candidates(arm: str, seed: int) -> list[str]:
    stem = f"{TAG}_W{ARMS[arm]['W']}_D{ARMS[arm]['D']}_s{seed}"
    cands = [os.environ.get(f"VPIPE_PATCHCLF_CKPT_{arm.replace('x', '_').upper()}", "")]
    cands += [os.path.join(r, f"patchclf_{stem}.pt") for r in MODEL_ROOTS]
    return [c for c in cands if c]


def checkpoint_paths(arm: str, seeds=(0, 1, 2)) -> list[str]:
    """Every seed's checkpoint on this host for `arm`, in seed order -- an ensemble when
    more than one is present, same as a single seed when only one was installed."""
    out = []
    for s in seeds:
        for c in _candidates(arm, s):
            if os.path.exists(c):
                out.append(os.path.abspath(c))
                break
    return out


def available(fleet=None, arm: str | None = None) -> tuple[bool, str]:
    if arm not in ARMS:
        return False, f"patchclf arm {arm!r} unknown; known: {sorted(ARMS)}"
    ps = checkpoint_paths(arm)
    if not ps:
        return False, f"patchclf:{arm} checkpoint not on this host (expected patchclf_{TAG}_W{ARMS[arm]['W']}_D{ARMS[arm]['D']}_s<seed>.pt in var/models)"
    return True, ",".join(ps)


def describe(arm: str | None = None) -> dict:
    ps = checkpoint_paths(arm) if arm in ARMS else []
    if not ps:
        return {}
    m = ps[0].replace(".pt", ".metrics.json")
    try:
        return json.load(open(m))
    except Exception:                                   # noqa: BLE001 - a missing sidecar is not fatal
        return {"checkpoints": ps, "preliminary": True, "caption": "PRELIMINARY (no metrics sidecar)"}


def _build_model(W: int, D: int, seed: int):
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    sys.path.insert(0, os.path.join(repo, "scripts"))
    from patchsize import models as PM           # noqa: E402
    return PM.build(W, D, seed)


def _layer_stack(layers_dir: str, n: int = N_LAYERS) -> "np.ndarray":
    """(n, H, W) uint8 -- the rendered stack, same reader every family here uses."""
    from .. import memstack as _MS
    all_files = _MS.list_names(layers_dir)
    if all_files is None:
        all_files = sorted(f for f in os.listdir(layers_dir) if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")))
    if len(all_files) < n:
        raise RuntimeError(f"{layers_dir}: only {len(all_files)} layers, need {n}")
    import tifffile
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    planes = []
    for f in all_files[:n]:
        fp = os.path.join(layers_dir, f)
        lz = _MS.read_layer(fp)
        if lz is None:
            if fp.lower().endswith((".tif", ".tiff")):
                try:
                    lz = np.asarray(tifffile.memmap(fp, mode="r"))
                except (ValueError, OSError):
                    lz = tifffile.imread(fp)
            else:
                lz = np.asarray(Image.open(fp).convert("L"))
        planes.append(lz)
    return np.stack(planes, axis=0)


def predict(layers_dir: str, mask_png: str, out_png: str, gpu: int = 0, arm: str | None = None,
            fleet=None, log: str | None = None, timeout: int = 3600, **spec) -> bool:
    ok, why = available(arm=arm)
    if not ok:
        _alerts.alert(f"patchclf on this host: {why}")
        return False
    import torch
    from PIL import Image
    cfg = ARMS[arm]
    W, Dp, stride = cfg["W"], cfg["D"], cfg["stride"]
    dev = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    ckpts = checkpoint_paths(arm)
    models = []
    mu = sd = depth_centre = None
    for cp in ckpts:
        state = torch.load(cp, map_location=dev)
        m = _build_model(W, Dp, int(state.get("seed", 0))).to(dev)
        m.load_state_dict(state["state_dict"])
        m.eval()
        models.append(m)
        mu, sd, depth_centre = state["mu"], state["sd"], state["depth_centre"]
    vol = _layer_stack(layers_dir)                      # (65, H, W_img) uint8
    Hh, Ww = vol.shape[1], vol.shape[2]
    d0 = depth_centre - (Dp - 1) // 2
    d1 = d0 + Dp
    v = vol[max(d0, 0):min(d1, vol.shape[0])].astype(np.float32)
    if v.shape[0] < Dp:                                  # pad at the stack's edge, same as training's clamp_centre
        pad = Dp - v.shape[0]
        v = np.pad(v, ((0, pad), (0, 0), (0, 0)))
    v = (v - mu) / sd
    o = (W - 1) // 2
    ys = list(range(o, Hh - o, stride)) or [min(o, Hh - 1)]
    xs = list(range(o, Ww - o, stride)) or [min(o, Ww - 1)]
    prob = np.zeros((Hh, Ww), dtype=np.float32)
    batch = 512
    vt = torch.from_numpy(v).to(dev)
    with torch.no_grad():
        cells = [(y, x) for y in ys for x in xs]
        for i in range(0, len(cells), batch):
            chunk = cells[i:i + batch]
            crops = torch.stack([vt[:, y - o:y - o + W, x - o:x - o + W] for y, x in chunk])
            s = torch.zeros(len(chunk), device=dev)
            for m in models:
                with torch.amp.autocast("cuda", enabled=dev.startswith("cuda")):
                    s += torch.sigmoid(m(crops).float())
            s = (s / len(models)).cpu().numpy()
            for (y, x), p in zip(chunk, s, strict=True):
                y0, y1 = max(0, y - stride // 2), min(Hh, y + stride - stride // 2)
                x0, x1 = max(0, x - stride // 2), min(Ww, x + stride - stride // 2)
                prob[y0:y1, x0:x1] = p
    valid = None
    if mask_png and os.path.exists(mask_png):
        mk = np.asarray(Image.open(mask_png).convert("L"))
        h, w = min(mk.shape[0], Hh), min(mk.shape[1], Ww)
        valid = np.zeros(prob.shape, dtype=bool)
        valid[:h, :w] = mk[:h, :w] > 127
        prob *= valid
    u8 = (np.clip(prob, 0, 1) * 255.0).astype(np.uint8)
    from .grandprize_dense import check_output_diversity
    check_output_diversity(u8, valid, family=f"patchclf:{arm}", ckpt=",".join(ckpts), out_png=out_png)
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    Image.fromarray(u8).save(out_png)
    side = os.path.splitext(out_png)[0] + ".patchclf.json"
    json.dump({"model": f"patchclf:{arm}", "checkpoints": ckpts, "W": W, "D": Dp, "stride": stride,
               "n_seeds": len(models)}, open(side, "w"))
    return True


def training_frame():
    from ...frame import MODELS, ModelSpec
    return MODELS.get("patchclf", ModelSpec("patchclf", IN_UM_PER_PX, 13)).frame(source_id="model:patchclf")


def check_frame(frame, tol: float = 0.05) -> None:
    training_frame().assert_compatible(frame, tol=tol, what="patchclf input")
