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
    if ck["config"].get("sharp_spec"):                       # a sharp_train student (ConvNeXt / dilated / 3-D-stem ... blocks), wrapped by scripts/sharp_strategy/sharp_student_to_prod.py
        import importlib.util
        mp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "scripts", "sharp_strategy", "sharp_models.py")
        spc = importlib.util.spec_from_file_location("sharp_models_prod", os.path.abspath(mp)); SM = importlib.util.module_from_spec(spc); spc.loader.exec_module(SM)
        m = SM.build(ck["config"]["sharp_spec"], int(ck["config"]["in_ch"]))
        m.load_state_dict(ck["state_dict"]); m.eval().to(device)
        if device.startswith("cuda"):
            for mod in m.modules():
                if isinstance(mod, torch.nn.Conv2d):
                    mod.to(memory_format=torch.channels_last)
            if fp16:
                m = m.half()
        return m, ck
    m = build(ck["config"]["widths"], int(ck["config"]["in_ch"]))
    m.load_state_dict(ck["state_dict"])
    m.eval()
    m = S.fold_bn(m).to(device)
    if device.startswith("cuda"):
        m = m.to(memory_format=torch.channels_last)
        if fp16:
            m = m.half()
    return m, ck


def predict_blend(model, arrs: list, norm: str, dev: str = "cuda", patch: int = 128, stride: int = 63, batch: int = 96, fp16: bool = True,
                  floor: float = 0.05, band: int = 2048, margin: int = 16, jitter: int = 0, dual: bool = False) -> np.ndarray:
    """Overlapped PATCH inference with a floored 2-D Hann blend (reader_v2's way: 128 px patches), logits (H, W) float32, all on the GPU:
    the input is normalised ONCE in bands of `band` px (with a norm-reach halo, as predict_array does), patches are sliced from the normalised tensor, run in batches of `batch`,
    weighted and accumulated on the GPU, and only the final map is copied back (the first version normalised and copied every patch separately: 3.5x slower than shift x4).
    `margin` (default 16 px): the outer border of every patch gets ZERO weight. reader_v2_dense responds at the (replicate-padded) border of its input tensor like at a segment edge, which
    painted a bright lattice at the patch positions when blended with margin 0 (relative lattice depth at the stride period 4.0-4.3x on Frag1 / PHerc0172 tiles; sharp_dense 1.7-1.9x);
    margin 16 -> 1.5-1.6x for both (margin 8 is worse, 5.8-7x; 12 is 1.7-2.9x; 20 is 1.5-1.7x; a larger margin shrinks the interior below the stride).
    The stride must NOT be a multiple of the decoder period (4 px): every patch then has a different phase relative to the decoder grid and the average removes it.
    Measured (sharp_dense, 54 tiles): stride 63 -> grid depth 9.3x -> 0.71x at 4.1 passes; strides 64 and 32 leave the grid (9.4x, 9.3x)."""
    import torch
    H, W = arrs[0].shape[:2]
    P, S = int(patch), int(stride)
    reach = norm_reach(norm)
    dt = torch.float16 if (fp16 and dev.startswith("cuda")) else torch.float32
    Hp, Wp = max(H, P), max(W, P)
    C = norm_channels(norm) * len(arrs)
    xn = torch.zeros((1, C, Hp, Wp), dtype=dt, device=dev)
    full = np.stack([np.asarray(a[:H, :W]) for a in arrs])
    for r0 in range(0, H, band):
        r1 = min(H, r0 + band); a0, a1 = max(0, r0 - reach), min(H, r1 + reach)
        rows = torch.from_numpy(full[:, a0:a1])
        for c0 in range(0, W, band):
            c1 = min(W, c0 + band); b0, b1 = max(0, c0 - reach), min(W, c1 + reach)
            t = apply_norm(rows[:, :, b0:b1][None].to(dev).float(), norm)
            xn[:, :, r0:r1, c0:c1] = t[:, :, r0 - a0:r0 - a0 + (r1 - r0), c0 - b0:c0 - b0 + (c1 - c0)].to(dt)
    if Hp > H or Wp > W:                                           # image smaller than one patch: replicate-pad
        xn[:, :, H:, :W] = xn[:, :, H - 1:H, :W]; xn[:, :, :, W:] = xn[:, :, :, W - 1:W]
    rng = np.random.default_rng(0)
    def pos(n, off=0):
        if n <= P:
            return [0]
        p = list(range(off, n - P + 1, S))
        if jitter:                                               # break the regular stride lattice: patch origins jittered by +-jitter px (fixed seed), coverage kept
            p = sorted({int(np.clip(v + rng.integers(-jitter, jitter + 1), 0, n - P)) for v in p})
        if p[-1] != n - P:
            p.append(n - P)
        if p[0] != 0:
            p.insert(0, 0)
        return p
    ys, xs = pos(Hp), pos(Wp)
    if dual:                                                     # second copy of the patch lattice shifted by S // 2 on both axes: cancels the fundamental of the stride-period seam residue
        ys2, xs2 = pos(Hp, S // 2), pos(Wp, S // 2)
    m_ = int(margin)
    h = np.hanning(P - 2 * m_ + 2)[1:-1].astype(np.float32)
    win_np = np.zeros((P, P), np.float32); win_np[m_:P - m_, m_:P - m_] = np.maximum(np.outer(h, h), floor)      # `margin` px of every patch border get ZERO weight (models respond at zero-padded tensor borders)
    win = torch.from_numpy(win_np).to(dev)
    acc = torch.zeros((Hp, Wp), dtype=torch.float32, device=dev); ws = torch.zeros((Hp, Wp), dtype=torch.float32, device=dev)
    jobs = [(y, x) for y in ys for x in xs]
    if dual:
        jobs += [(y, x) for y in ys2 for x in xs2]
    for i0 in range(0, len(jobs), batch):
        chunk = jobs[i0:i0 + batch]
        xb = torch.stack([xn[0, :, y:y + P, x:x + P] for (y, x) in chunk]).contiguous(memory_format=torch.channels_last)
        with torch.no_grad():
            yb = model(xb)[:, 0].float()
        for k, (y, x) in enumerate(chunk):
            acc[y:y + P, x:x + P] += yb[k] * win; ws[y:y + P, x:x + P] += win
    return (acc / ws.clamp_min(1e-6))[:H, :W].cpu().numpy()


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
    fixed = getattr(model, "_orig_mod", None) is not None          # compiled model: ONE static input shape (all tiles padded to tile + 2 halo)
    S_fix = ((tile + 2 * halo + mult - 1) // mult) * mult
    dt = torch.float16 if (fp16 and dev.startswith("cuda")) else torch.float32
    for r0 in range(0, H, tile):
        r1 = min(H, r0 + tile)
        a0, a1 = max(0, r0 - halo), min(H, r1 + halo)
        band = torch.from_numpy(np.stack([np.asarray(a[a0:a1, :W]) for a in arrs])[None]).to(dev)
        for c0 in range(0, W, tile):
            c1 = min(W, c0 + tile)
            b0, b1 = max(0, c0 - halo), min(W, c1 + halo)
            x = apply_norm(band[:, :, :, b0:b1].to(torch.float32), norm)
            if fixed:
                from ._compile import bucket as _bucket
                ph, pw_ = _bucket(x.shape[2], mult) - x.shape[2], _bucket(x.shape[3], mult) - x.shape[3]
            else:
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
    n_shift = max(1, int(spec.get("shift_tta", 1) or 1))
    from . import _compile as _C
    use_c = _C.decision(os.path.abspath(ck), n_shift, max(H, W), spec.get("torch_compile", "auto"))
    if use_c:
        model = _C.maybe_compile(model, (os.path.abspath(ck), os.path.getmtime(ck), dev), True, dev, int(ckd["config"]["in_ch"]))
    t1 = time.time()
    if spec.get("blend_stride"):
        lg = predict_blend(model, arrs, norm, dev, patch=int(spec.get("blend_patch", 128)), stride=int(spec["blend_stride"]), margin=int(spec.get("blend_margin", 16)))
    else:
        lg = predict_array(model, arrs, norm, dev, tile=int(spec.get("tile", TILE)), halo=int(ckd["config"].get("halo", norm_reach(norm) + 64)), shift_tta=n_shift)
    if dev == "cuda":
        torch.cuda.synchronize()
    t2 = time.time()
    # optional operating-point shift stored in the checkpoint (sharp_dense: BCE on ink-centred crops puts the whole map high;
    # p = sigmoid(logit - bias) with bias = logit of the pooled 5 % confirmed-negative-FPR threshold, so 0.5 = that point).
    # A monotone shift: AUC and edge sharpness are unchanged. 0 for every checkpoint that does not carry one.
    bias = float(ckd["config"].get("logit_bias", 0.0))
    scale = float(ckd["config"].get("logit_scale", 1.0))      # temperature (monotone): reader_v2_dense's confirmed-negative logits span only 1.35 logits (sharp_dense 5.6)
    prob = 1.0 / (1.0 + np.exp(-(lg - bias) * scale))
    prob *= (np.asarray(_layer_array(files[len(files) // 2])[:H, :W]) > 0)
    if mask_png and os.path.exists(mask_png):
        mk = np.asarray(Image.open(mask_png).convert("L"))
        h, w = min(mk.shape[0], H), min(mk.shape[1], W)
        prob[:h, :w] *= (mk[:h, :w] > 127)
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    Image.fromarray(np.clip(prob * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(out_png)
    stats = {"wall_s": round(time.time() - t0, 3), "gpu_s": round(t2 - t1, 3), "shape": [H, W],
             "MVox_per_s": round(H * W * IN_CHANS / 1e6 / max(t2 - t1, 1e-9), 1), "norm": norm,
             "layers": idx, "device": dev, "logit_bias": bias, "logit_scale": scale, "shift_tta": n_shift, "blend_stride": spec.get("blend_stride"), "blend_patch": spec.get("blend_patch"), "torch_compile": type(model).__name__ == "OptimizedModule"}
    with open(os.path.splitext(out_png)[0] + ".reader_v2_dense.json", "w") as fh:
        json.dump({"model": "reader_v2_dense" + (f":{arm}" if arm else ""), "checkpoint": ck, "spec": SPEC,
                   "rf_total": ckd.get("rf_total"), "teacher": ckd.get("teacher"), "stats": stats}, fh, indent=1)
    return True
