"""reader_v2_dense: a small DENSE student distilled from Reader v2 (stages/ink_models/reader_v2.py).

Teacher: Reader v2 (DomRusso2, HF domenicor046/reader-v2 @ 72633d8a, reader-v2-step040000.pth, md5 7f261ac1...),
the ink_9um architecture (68.2 M parameters, 17 x 128 x 128 patches, per-patch robust-MAD normalisation, stride-64
Hann blend). Distilled by `scripts/reader_v2_distill/rv2_distill.py` (the ink9um_student recipe: soft BCE + 0.1 logit
MSE to the teacher's probability on BOTH faces, never thresholded, 50 % of crops on the teacher's hottest 32-px blocks,
scroll-balanced sampling, whole segments held out, best checkpoint by held-out teacher agreement) into the
ink9um_student U-Net (`ink9um_student.build`) with a deliberately SMALL receptive field:

  * normalisation is a stack of small local standardisations (`ms9-17` = 9 px and 17 px windows), not the production
    student's 33/129/257 px boxes -- an in-model local-contrast box widens the receptive field by k - 1;
  * 2-3 U-Net levels, so the architectural support (net extent + largest norm window - 1) stays under the 64 px cap
    (user rule 2026-09-08) and the 99 %-gradient-mass window under 32 px. Both are MEASURED by backprop through
    normaliser + net (`measure_total_rf`) and carried in the checkpoint (`rf_total`).

Output: one logit per input pixel (stride 1, ~9.5 um/px), 2048 px tiles with a halo, no overlap blending.
Input contract = ink9um_student's / Reader v2's: the 17 centred layers of the ~9.5 um ink9um render, uint8; the
validity mask is the middle layer > 0. A student inherits its teacher's training set: Reader v2 trained on S1
(PHercParis4), S4 (PHerc1667) and S5 (PHerc0172), so only PHerc0841 and the Kaggle fragments are unseen.

OFF by default (VPIPE_READER_V2_DENSE=1 to enable); weights var/models/reader_v2_dense[_<arm>].ckpt.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

from ... import alerts as _alerts
from . import ink9um_student as S

IN_CHANS = 17
UM_PER_PX = 9.5
TILE = 2048
# reads an in-memory native-stack view (stages/memstack.py) through ink9um_student.layer_files and
# grandprize_dense._layer_array, both memstack-aware -- so native-render mode may hand it a view key (2026-10-06)
MEMSTACK_READY = True
CKPT_NAME = "reader_v2_dense.ckpt"
ENV_CKPT = "VPIPE_READER_V2_DENSE_CKPT"
MODEL_ROOTS = S.MODEL_ROOTS
SPEC = {
    "layers": IN_CHANS,
    "layer_select": "centred (upstream select_layer_indices)",
    "um_per_px": UM_PER_PX,
    "normalisation": "stack of local standardisations, window(s) named in the checkpoint (default ms9-17)",
    "arch": "unet2d_17ch_bnfolded (ink9um_student.build), 2-3 levels",
    "tile": TILE,
    "output_stride_px": 1,
    "teacher": "reader_v2 (HF domenicor046/reader-v2 @ 72633d8a, step040000)",
}


# ---------------------------------------------------------------------------------------------
# normalisation: 'local<k>' or 'ms<a>-<b>[-<c>...]' (each an odd box window in px)
# ---------------------------------------------------------------------------------------------
def norm_windows(norm: str) -> tuple[int, ...]:
    m = re.fullmatch(r"local(\d+)", norm)
    if m:
        w = (int(m.group(1)),)
    else:
        m = re.fullmatch(r"ms(\d+(?:-\d+)+)", norm)
        if not m:
            raise ValueError(f"reader_v2_dense: unknown norm {norm!r} (use local<k> or ms<a>-<b>...)")
        w = tuple(int(v) for v in m.group(1).split("-"))
    if any(k < 3 or k % 2 == 0 for k in w):
        raise ValueError(f"reader_v2_dense: norm windows must be odd and >= 3, got {w}")
    return w


def norm_channels(norm: str) -> int:
    return len(norm_windows(norm))


def norm_reach(norm: str) -> int:
    return max(norm_windows(norm)) // 2


def apply_norm(x, norm: str):
    import torch
    ws = norm_windows(norm)
    return torch.cat([S.local_normalise(x, w) for w in ws], 1) if len(ws) > 1 else S.local_normalise(x, ws[0])


def build(widths, in_ch: int):
    return S.build(tuple(widths), in_ch=in_ch)


def measure_total_rf(model, norm: str, dev: str = "cpu", size: int = 256) -> dict:
    """Receptive field of the centre output pixel THROUGH the normaliser and the net, by backprop
    on raw grey input: full extent of nonzero gradient (rel. 1e-6), the 99 %-mass window, and the
    architectural support (net-only extent + largest norm window - 1, the bound a trained net can reach)."""
    import copy
    import torch
    m = copy.deepcopy(model).float().to(dev).eval()
    g0 = torch.Generator().manual_seed(0)
    x = (torch.rand(1, IN_CHANS, size, size, generator=g0) * 120 + 60).to(dev).requires_grad_(True)
    y = m(apply_norm(x, norm))
    y[0, 0, size // 2, size // 2].backward()
    g = x.grad.abs().sum(1)[0].detach().cpu().numpy()
    nz = np.argwhere(g > g.max() * 1e-6)
    ext = int(max(np.ptp(nz[:, 0]), np.ptp(nz[:, 1])) + 1)
    yy, xx = np.mgrid[:size, :size]
    r = np.maximum(np.abs(yy - size // 2), np.abs(xx - size // 2)).ravel()
    o = np.argsort(r)
    cm = np.cumsum(g.ravel()[o]) / g.sum()
    r99 = int(r[o][np.searchsorted(cm, 0.99)])
    # net only, on unit-variance input of the net's own channel count
    xn = torch.randn(1, m.enc[0][0][0].in_channels, size, size, generator=g0).to(dev).requires_grad_(True)
    m(xn)[0, 0, size // 2, size // 2].backward()
    gn = xn.grad.abs().sum(1)[0].detach().cpu().numpy()
    nzn = np.argwhere(gn > gn.max() * 1e-6)
    net_ext = int(max(np.ptp(nzn[:, 0]), np.ptp(nzn[:, 1])) + 1)
    return {"total_extent_px": ext, "total_r99_window_px": 2 * r99 + 1, "net_extent_px": net_ext,
            "arch_support_px": net_ext + max(norm_windows(norm)) - 1, "norm": norm, "um_per_px": UM_PER_PX}


# ---------------------------------------------------------------------------------------------
# load / predict
# ---------------------------------------------------------------------------------------------
def load(path: str, device: str = "cuda", fp16: bool = True):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = build(ck["config"]["widths"], int(ck["config"]["in_ch"]))
    m.load_state_dict(ck["state_dict"])
    m.eval()
    m = S.fold_bn(m).to(device)
    if device.startswith("cuda"):
        m = m.to(memory_format=torch.channels_last)
        if fp16:
            m = m.half()
    return m, ck


def predict_array(model, arrs: list, norm: str, dev: str = "cuda", tile: int = TILE, halo: int | None = None,
                  fp16: bool = True, shift_tta: int = 1) -> np.ndarray:
    """Logits (H, W) float32 for a list of 17 2-D uint8 arrays: tiles of `tile` px with a `halo`
    margin, no overlap blending (each output pixel from the one tile whose interior it is).

    shift_tta = N > 1: average the logits of N diagonal input shifts (0,0),(1,1),..,(N-1,N-1) px (top-left edge padding, cropped back). The stride-4
    decoder leaves a 4 px grid in a single pass (boxy_diag.py); N = 4 spans the 4 px phase and removed it (P=4 depth 9.3x -> 0.70x), cost xN."""
    if shift_tta and int(shift_tta) > 1:
        H0, W0 = arrs[0].shape[:2]
        acc = None
        for k in range(int(shift_tta)):
            ak = arrs if k == 0 else [np.pad(np.asarray(a)[:H0, :W0], ((k, 0), (k, 0)), mode="edge") for a in arrs]
            o = predict_array(model, ak, norm, dev, tile, halo, fp16, shift_tta=1)[k:k + H0, k:k + W0]
            acc = o.astype(np.float32) if acc is None else acc + o
        return acc / int(shift_tta)
    import torch
    H, W = arrs[0].shape[:2]
    halo = int(halo if halo is not None else norm_reach(norm) + 64)
    out = np.empty((H, W), np.float32)
    mult = 2 ** (len(getattr(model, "widths", (0, 0, 0))) - 1)
    dt = torch.float16 if (fp16 and dev.startswith("cuda")) else torch.float32
    for r0 in range(0, H, tile):
        r1 = min(H, r0 + tile)
        a0, a1 = max(0, r0 - halo), min(H, r1 + halo)
        band = torch.from_numpy(np.stack([np.asarray(a[a0:a1, :W]) for a in arrs])[None]).to(dev)
        for c0 in range(0, W, tile):
            c1 = min(W, c0 + tile)
            b0, b1 = max(0, c0 - halo), min(W, c1 + halo)
            x = apply_norm(band[:, :, :, b0:b1].to(torch.float32), norm)
            ph, pw_ = (-x.shape[2]) % mult, (-x.shape[3]) % mult
            if ph or pw_:
                x = torch.nn.functional.pad(x, (0, pw_, 0, ph), mode="replicate")
            x = x.to(dt).contiguous(memory_format=torch.channels_last)
            with torch.no_grad():
                y = model(x)[0, 0].float()
            out[r0:r1, c0:c1] = y[r0 - a0:r0 - a0 + (r1 - r0), c0 - b0:c0 - b0 + (c1 - c0)].cpu().numpy()
        del band
    return out


def checkpoint_path(arm: str | None = None) -> str | None:
    name = f"reader_v2_dense_{arm}.ckpt" if arm else CKPT_NAME
    cands = [os.environ.get(ENV_CKPT, "")] if not arm else []
    cands += [os.path.join(r, name) for r in MODEL_ROOTS]
    for c in cands:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return None


def training_frame():
    from ...frame import ModelSpec
    return ModelSpec("reader_v2_dense", UM_PER_PX, IN_CHANS).frame(source_id="model:reader_v2_dense")


def enabled() -> bool:
    return os.environ.get("VPIPE_READER_V2_DENSE", "0") == "1"


def available(fleet=None, arm: str | None = None) -> tuple[bool, str]:
    """Are the weights here. NOT whether the family should run."""
    p = checkpoint_path(arm)
    if p is None:
        return False, f"reader_v2_dense checkpoint not on this host (var/models/{CKPT_NAME}, or ${ENV_CKPT})"
    return True, p


def predict(layers_dir: str, mask_png: str | None, out_png: str, gpu: int = 0, arm: str | None = None,
            fleet=None, log: str | None = None, **spec) -> bool:
    """One rendered stack (already in the face the dispatcher wants) -> one probability PNG."""
    ok, ck = available(fleet, arm)
    if not ok:
        _alerts.alert(f"reader_v2_dense on this host: {ck}")
        return False
    try:
        import torch
    except ImportError:
        _alerts.alert("reader_v2_dense on this host: torch does not import in this interpreter")
        return False
    import time
    from PIL import Image
    from .grandprize_dense import _layer_array
    Image.MAX_IMAGE_PIXELS = None
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    files = S.layer_files(layers_dir)
    idx = S.select_layers(len(files))
    arrs = [_layer_array(files[i]) for i in idx]
    H, W = arrs[0].shape[:2]
    model, ckd = load(ck, dev)
    norm = ckd["config"]["norm"]
    t1 = time.time()
    n_shift = max(1, int(spec.get("shift_tta", 1) or 1))
    lg = predict_array(model, arrs, norm, dev, halo=int(ckd["config"].get("halo", norm_reach(norm) + 64)), shift_tta=n_shift)
    if dev == "cuda":
        torch.cuda.synchronize()
    t2 = time.time()
    # optional operating-point shift stored in the checkpoint (sharp_dense: BCE on ink-centred crops puts the whole map high;
    # p = sigmoid(logit - bias) with bias = logit of the pooled 5 % confirmed-negative-FPR threshold, so 0.5 = that point).
    # A monotone shift: AUC and edge sharpness are unchanged. 0 for every checkpoint that does not carry one.
    bias = float(ckd["config"].get("logit_bias", 0.0))
    prob = 1.0 / (1.0 + np.exp(-(lg - bias)))
    prob *= (np.asarray(_layer_array(files[len(files) // 2])[:H, :W]) > 0)
    if mask_png and os.path.exists(mask_png):
        mk = np.asarray(Image.open(mask_png).convert("L"))
        h, w = min(mk.shape[0], H), min(mk.shape[1], W)
        prob[:h, :w] *= (mk[:h, :w] > 127)
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    Image.fromarray(np.clip(prob * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(out_png)
    stats = {"wall_s": round(time.time() - t0, 3), "gpu_s": round(t2 - t1, 3), "shape": [H, W],
             "MVox_per_s": round(H * W * IN_CHANS / 1e6 / max(t2 - t1, 1e-9), 1), "norm": norm,
             "layers": idx, "device": dev, "logit_bias": bias, "shift_tta": n_shift}
    with open(os.path.splitext(out_png)[0] + ".reader_v2_dense.json", "w") as fh:
        json.dump({"model": "reader_v2_dense" + (f":{arm}" if arm else ""), "checkpoint": ck, "spec": SPEC,
                   "rf_total": ckd.get("rf_total"), "teacher": ckd.get("teacher"), "stats": stats}, fh, indent=1)
    return True
