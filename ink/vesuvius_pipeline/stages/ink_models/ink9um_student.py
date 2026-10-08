"""ink9um_student: a small fully-convolutional student distilled from the ink_9um teacher.

The teacher (`ink_9um` hybrid_3d2d, 68.2 M parameters, `checkpoints/ink_9um/...step-075000.pth`)
reads 17 centred layers of the ~9.5 um/px render in 128 x 128 patches, normalises EVERY PATCH by
its own robust median/MAD, and blends patches on a 50 %-overlap Hann grid (stride 64 px). Each
output pixel is therefore computed by four forward passes of a 68 M-parameter net, and the
per-patch statistics forbid running it on a bigger tile.

The student reads the SAME input (the same render, the same 17 centred layers, the same order --
so `ink9um_student_reversed` is the teacher's `ink9um_reversed` face), but:
  * normalises DENSELY instead of per patch: every pixel by the mean/std of all 17 layers over
    the 129 x 129 window centred on it (`local_normalise`) -- the teacher standardises each
    128-px patch as a whole, and a pilot with one global per-segment normalisation could not
    follow it (held-out pixel r 0.21 after 3,000 steps). A pure convolution, so a tile's
    interior does not depend on where the tile was cut;
  * is a 5-level 2-D U-Net over the 17 layers as channels (~1.4 M parameters, BatchNorm folded at
    load), so it runs on 2048 px tiles with a halo of `HALO` px and NO overlap blending -- the
    stride is the tile;
  * runs in fp16, channels_last.

Trained by `scripts/hires_ink/ink9um_distill.py` against the teacher's own probability maps
(soft targets, never thresholded, BCE against the teacher probability), whole segments held out.
The checkpoint carries its config, its training provenance and its held-out numbers.

Input contract (identical to the teacher's, which is why it can share the ink9um render):
  * a >= 17-layer render at ~9.5 um/px; the centred 17 layers (upstream `select_layer_indices`);
  * uint8; the validity mask is the middle rendered layer > 0 (as run_ink9um.py builds it).
"""
from __future__ import annotations

import json
import os

import numpy as np

from ... import alerts as _alerts

# reads an in-memory native-stack view (stages/memstack.py) through its layer readers
MEMSTACK_READY = True

IN_CHANS = 17
UM_PER_PX = 9.5
TILE = 2048
HALO = 160              # >= LOCAL_WIN//2 + half the receptive field (checkpoint `rf_trained`)
LOCAL_WIN = 129         # the local-normalisation window: the teacher normalises per 128-px patch
CKPT_NAME = "ink9um_student.ckpt"
MODEL_ROOTS = ("var/models",
               "/dev/shm/vpipe/ScrollPrizeTutorial/current/var/models",
               "/dev/shm/vpipe/var/models")
ENV_CKPT = "VPIPE_INK9UM_STUDENT_CKPT"

SPEC = {
    "layers": IN_CHANS,
    "layer_select": "centred (upstream select_layer_indices)",
    "um_per_px": UM_PER_PX,
    "normalisation": "local: (x - mean) / std over all 17 layers in a 129 x 129 px window, clip +-6",
    "arch": "unet2d_17ch_bnfolded",
    "tile": TILE,
    "halo": HALO,
    "output_stride_px": 1,
    "teacher": "ink_9um hybrid_3d2d-seed42 step-075000",
}


# ---------------------------------------------------------------------------------------------
# the network -- torch imported lazily so the registry/settings path never pulls it in
# ---------------------------------------------------------------------------------------------
def build(widths=(32, 48, 96, 128, 192), in_ch: int = IN_CHANS, blocks=None,
          skip_off=None, upscale: int = 1):
    """`blocks[i]` = number of residual 3x3 pairs added at encoder level i (default none): the
    depth goes where it is cheap, at the coarse levels, the way the teacher's encoder is built
    (1, 3, 4, 6, 6, 6 blocks per stage).

    `skip_off` (sharp_student ablation C, 2026-10-01): a set/iterable of DECODER step indices
    (0 = the finest, the skip from `enc[0]`) whose skip connection is dropped -- the decoder at
    that step gets only the upsampled coarser feature (zero-filled skip), never the full-res
    encoder activation. Tests whether the model's own full-res skips are what blur the output
    (CLAUDE.md problem statement's own architecture hypothesis), independent of the targets.

    `upscale` (ablation C, PixelShuffle head): when > 1, the head is `Conv -> PixelShuffle(u)`,
    producing a (u*H, u*W) output on a grid FINER than the input pitch. Per the physics note
    (no separable ink signal above ~1/20 um^-1 in the CT, FINDINGS), this buys a sharper-LOOKING
    grid only if the net can invent genuine sub-input-pixel shape from a learned prior (sharp
    labels) -- it is trained by average-pooling the upscale x upscale output back to the native
    grid for the loss (same supervision as upscale=1), so nothing at the native resolution
    changes; the up-scaled map itself is inspected/measured separately, in its own finer um/px."""
    import torch
    from torch import nn
    import torch.nn.functional as F

    def cbr(i, o, s=1):
        return nn.Sequential(nn.Conv2d(i, o, 3, stride=s, padding=1, bias=False),
                             nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    class Res(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.body = nn.Sequential(cbr(c, c), nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c))

        def forward(self, x):
            return torch.relu(x + self.body(x))

    nb = list(blocks) if blocks else [0] * len(widths)
    skip_off = set(skip_off or ())

    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            w = list(widths)
            self.widths = tuple(w)
            self.skip_off = tuple(sorted(skip_off))
            self.upscale = int(upscale)
            self.enc = nn.ModuleList()
            self.enc.append(nn.Sequential(cbr(in_ch, w[0]), cbr(w[0], w[0]), *[Res(w[0]) for _ in range(nb[0])]))
            for k, (a, b) in enumerate(zip(w[:-1], w[1:], strict=True), start=1):
                self.enc.append(nn.Sequential(cbr(a, b, 2), cbr(b, b), *[Res(b) for _ in range(nb[k])]))
            self.dec = nn.ModuleList()
            for a, b in zip(w[::-1][:-1], w[::-1][1:], strict=True):      # (deep, skip) pairs
                self.dec.append(nn.Sequential(cbr(a + b, b), cbr(b, b)))
            if self.upscale > 1:
                self.head = nn.Sequential(nn.Conv2d(w[0], self.upscale * self.upscale, 1),
                                          nn.PixelShuffle(self.upscale))
            else:
                self.head = nn.Conv2d(w[0], 1, 1)

        def forward(self, x):
            skips = []
            for e in self.enc:
                x = e(x)
                skips.append(x)
            x = skips.pop()
            for i, d in enumerate(self.dec):
                s = skips.pop()
                x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
                if i in self.skip_off:
                    s = torch.zeros_like(s)
                x = d(torch.cat([x, s], 1))
            return self.head(x)

    return Student()


def fold_bn(model):
    """Fold every Conv2d(bias=False) + BatchNorm2d pair into one biased conv (inference)."""
    import torch
    from torch import nn

    def _fold(seq):
        mods = list(seq.children())
        out = []
        i = 0
        while i < len(mods):
            m = mods[i]
            if isinstance(m, nn.Conv2d) and i + 1 < len(mods) and isinstance(mods[i + 1], nn.BatchNorm2d):
                bn = mods[i + 1]
                c = nn.Conv2d(m.in_channels, m.out_channels, m.kernel_size, m.stride, m.padding, bias=True)
                sc = bn.weight / torch.sqrt(bn.running_var + bn.eps)
                c.weight.data = m.weight.data * sc[:, None, None, None]
                b0 = m.bias.data if m.bias is not None else torch.zeros_like(bn.running_mean)
                c.bias.data = (b0 - bn.running_mean) * sc + bn.bias
                out.append(c)
                i += 2
            else:
                out.append(_fold(m) if isinstance(m, nn.Sequential) else m)
                i += 1
        return nn.Sequential(*out)

    def _walk(mod):
        for name, ch in list(mod.named_children()):
            if isinstance(ch, nn.Sequential):
                setattr(mod, name, _fold(ch))
                _walk(getattr(mod, name))
            elif isinstance(ch, nn.ModuleList):
                for k in range(len(ch)):
                    if isinstance(ch[k], nn.Sequential):
                        ch[k] = _fold(ch[k])
                    _walk(ch[k])
            else:
                _walk(ch)
    _walk(model)
    return model


def load(path: str, device: str = "cuda", fp16: bool = True, compile_: bool = False):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = build(tuple(ck["config"]["widths"]), blocks=ck["config"].get("blocks"),
              in_ch=int(ck["config"].get("in_ch", IN_CHANS)),
              skip_off=ck["config"].get("skip_off"), upscale=int(ck["config"].get("upscale", 1)))
    m.load_state_dict(ck["state_dict"])
    m.eval()
    m = fold_bn(m).to(device)
    if device.startswith("cuda"):
        m = m.to(memory_format=torch.channels_last)
        if fp16:
            m = m.half()
    if compile_:
        m = torch.compile(m, dynamic=False)
    return m, ck


# ---------------------------------------------------------------------------------------------
# input handling shared with the trainer
# ---------------------------------------------------------------------------------------------
def select_layers(n: int, depth: int = IN_CHANS) -> list[int]:
    """Upstream `select_layer_indices` for layer_start/end None, direction forward: the centred
    `depth` layers, upper centre for an even excess."""
    if n < depth:
        raise RuntimeError(f"only {n} layers in the stack, need {depth}")
    s = n // 2 - depth // 2
    return list(range(s, s + depth))


def layer_files(layers_dir: str) -> list[str]:
    from .. import memstack as _MS
    fs = _MS.list_names(layers_dir)                 # an in-memory view of the native stack
    if fs is None:
        fs = sorted(f for f in os.listdir(layers_dir) if f.lower().endswith((".tif", ".tiff")))
    return [os.path.join(layers_dir, f) for f in fs]


def robust_stats(sample: np.ndarray) -> dict:
    """p1/p99 clip, median and 1.4826*MAD of a 1-D uint8 sample of MATERIAL voxels."""
    s = sample.astype(np.float32)
    if s.size < 1000:
        return {"lo": 0.0, "hi": 255.0, "med": 128.0, "mad": 32.0, "n": int(s.size)}
    lo, hi = np.percentile(s, [1, 99])
    c = np.clip(s, lo, hi)
    med = float(np.median(c))
    mad = float(np.median(np.abs(c - med)) * 1.4826)
    if not np.isfinite(mad) or mad < 1e-3:
        mad = max(float(c.std()), 1.0)
    return {"lo": float(lo), "hi": float(hi), "med": med, "mad": mad, "n": int(s.size)}


def segment_stats(arrs: list, step: int = 7) -> dict:
    """Stats over every `step`-th pixel of the layers, MATERIAL only (middle layer > 0)."""
    mid = np.asarray(arrs[len(arrs) // 2][::step, ::step])
    m = mid > 0
    return robust_stats(np.concatenate([np.asarray(a[::step, ::step])[m] for a in arrs]))


def normalise_(x, st: dict):
    """In place on a float tensor of raw grey values."""
    return x.clamp_(st["lo"], st["hi"]).sub_(st["med"]).div_(st["mad"])


def local_normalise(x, win: int = LOCAL_WIN, eps: float = 1.0):
    """Dense stand-in for the teacher's PER-PATCH normalisation.

    The teacher standardises each 17 x 128 x 128 patch by ONE robust centre and spread taken
    over the whole patch -- raw zeros outside the surface included -- and blends patches on a
    64-px grid, so its output depends on the local 128-px statistics. Here every pixel gets the
    mean and std of all 17 layers over the `win` x `win` window centred on it (separable box
    filter, fp32): (x - mean) / std, clipped to +-6. Pixels within win//2 of a tile edge see a
    replicate-padded window, which is why the inference halo exceeds win//2."""
    p = win // 2
    a = x.mean(1, keepdim=True)
    a2 = (x * x).mean(1, keepdim=True)
    box = _box_cumsum if os.environ.get("VPIPE_STUDENT_BOX", "cumsum") == "cumsum" else _box_pool
    m = box(a, win, p)
    sd = (box(a2, win, p) - m * m).clamp_min(eps).sqrt_()
    return ((x - m) / sd).clamp_(-6.0, 6.0)


def _box_pool(t, win: int, p: int):
    """The original separable box mean: avg_pool2d over replicate padding, O(win) taps per pixel
    per axis (the measured normaliser ceiling, progdistil 2026-09-29: ~36 MPix/s with a null net)."""
    import torch.nn.functional as F
    t = F.avg_pool2d(F.pad(t, (p, p, 0, 0), mode="replicate"), (1, win), stride=1)
    return F.avg_pool2d(F.pad(t, (0, 0, p, p), mode="replicate"), (win, 1), stride=1)


def _box_cumsum(t, win: int, p: int):
    """The same box mean by running sums: O(1) per pixel whatever the window. Accumulated in
    float64 (a 2.5k-px row of squared uint8 greys sums past 1e8, where float32 loses whole units),
    returned in the input dtype. Replicate padding exactly as `_box_pool`."""
    import torch
    import torch.nn.functional as F
    dt = t.dtype
    u = F.pad(t, (p, p, 0, 0), mode="replicate").double()
    c = torch.nn.functional.pad(u.cumsum(3), (1, 0))
    u = (c[..., win:] - c[..., :-win]) / win
    u = F.pad(u, (0, 0, p, p), mode="replicate")
    c = torch.nn.functional.pad(u.cumsum(2), (0, 0, 1, 0))
    return ((c[..., win:, :] - c[..., :-win, :]) / win).to(dt)


# ---------------------------------------------------------------------------------------------
# the finish-stage contract
# ---------------------------------------------------------------------------------------------
def checkpoint_path(arm: str | None = None) -> str | None:
    name = f"ink9um_student_{arm}.ckpt" if arm else CKPT_NAME
    cands = [os.environ.get(ENV_CKPT, "")] if not arm else []
    cands += [os.path.join(r, name) for r in MODEL_ROOTS]
    for c in cands:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return None


def training_frame():
    from ...frame import MODELS, ModelSpec
    return MODELS.get("ink9um_student", ModelSpec("ink9um_student", UM_PER_PX, IN_CHANS)).frame(
        source_id="model:ink9um_student")


def check_frame(frame, tol: float = 0.05) -> None:
    """Refuse a render whose pitch is not this model's. Raises FrameError."""
    training_frame().assert_compatible(frame, tol=tol, what="ink9um_student input")


def enabled() -> bool:
    return os.environ.get("VPIPE_INK9UM_STUDENT", "0") == "1"


def available(fleet=None, arm: str | None = None) -> tuple[bool, str]:
    """Are the weights here. NOT whether it should run (the settings row decides that)."""
    p = checkpoint_path(arm)
    if p is None:
        return False, f"ink9um_student checkpoint not on this host (var/models/{CKPT_NAME}, or ${ENV_CKPT})"
    return True, p


def describe(arm: str | None = None) -> dict:
    p = checkpoint_path(arm)
    if not p:
        return {}
    try:
        return json.load(open(p.replace(".ckpt", ".metrics.json")))
    except Exception:                                    # noqa: BLE001 - a missing sidecar is not fatal
        return {"checkpoint": p, "caption": "no metrics sidecar"}


def _torch_python(fleet) -> str | None:
    from ... import config
    root = (getattr(fleet, "root", None) if fleet is not None else None) or config.repo_root()
    for p in (os.environ.get("VPIPE_INK9UM_STUDENT_PY"), os.environ.get("VILLA_PY"),
              str(root / "villa" / "vesuvius" / ".venv" / "bin" / "python"),
              str(root / "Vesuvius-Grandprize-Winner" / ".venv" / "bin" / "python")):
        if p and os.path.exists(p):
            return p
    return None


def predict(layers_dir: str, mask_png: str, out_png: str, gpu: int = 0, arm: str | None = None,
            fleet=None, log: str | None = None, timeout: int = 3600, **spec) -> bool:
    """One rendered stack -> one per-pixel probability PNG. In-process when torch imports here,
    else handed to a torch interpreter (the worker venv on the peers has none)."""
    ok, why = available(arm=arm)
    if not ok:
        _alerts.alert(f"ink9um_student on this host: {why}")
        return False
    try:
        import torch  # noqa: F401
        return predict_inproc(layers_dir, mask_png, out_png, gpu=gpu, arm=arm, **spec) is not None
    except ImportError:
        pass
    from ... import config
    from ..finish import _run
    if fleet is None:
        fleet = config.load()
    py = _torch_python(fleet)
    if not py:
        _alerts.alert("ink9um_student on this host: no torch interpreter (villa/vesuvius/.venv)")
        return False
    src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    cmd = [py, "-m", "vesuvius_pipeline.stages.ink_models.ink9um_student", "--layers", layers_dir,
           "--out", out_png, "--gpu", "0"] + (["--mask", mask_png] if mask_png and os.path.exists(mask_png) else []) \
        + (["--arm", arm] if arm else [])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
               PYTHONPATH=src + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""))
    rc = _run(cmd, log or os.path.splitext(out_png)[0] + ".log", timeout, env=env)
    return rc == 0 and os.path.exists(out_png) and os.path.getsize(out_png) > 0


NORM_CHANNELS = {"segment": 1, "local129": 1, "ms33-129-257": 3, "local33": 1, "ms9-33-129": 3}
NORM_REACH = {"segment": 0, "local129": 64, "ms33-129-257": 128, "local33": 16, "ms9-33-129": 64}   # px of context the normaliser reads
# sharp_student ablation (2026-10-02): `local33` / `ms9-33-129` shrink the normaliser window, whose
# 129/257 px box means are a second source of spatial blur beside the net's own receptive field


def apply_norm(x, norm: str, st: dict | None = None):
    """The input transform named in the checkpoint's config. `ms33-129-257` stacks three
    local standardisations (33, 129 and 257 px windows) so the net can build its own version of
    the teacher's per-128-px-patch statistics instead of being handed one."""
    import torch
    if norm == "local129":
        return local_normalise(x)
    if norm == "ms33-129-257":
        return torch.cat([local_normalise(x, 33), local_normalise(x, 129), local_normalise(x, 257)], 1)
    if norm == "local33":
        return local_normalise(x, 33)
    if norm == "ms9-33-129":
        return torch.cat([local_normalise(x, 9), local_normalise(x, 33), local_normalise(x, 129)], 1)
    return normalise_(x, st)


_PINNED: dict = {}


def _to_dev_u8(a, r0: int, r1: int, W: int, dev: str):
    """Rows [r0, r1) of one uint8 layer on `dev`, as uint8 (4x fewer bytes than float32).

    A LazyLayer view that resamples on the card hands back a device tensor directly (no host
    round trip); a memmap/array goes through ONE reused pinned staging buffer with an async copy."""
    import torch
    rows = getattr(a, "rows_device", None)
    if rows is not None and str(dev).startswith("cuda"):
        t = rows(r0, r1)
        if t is not None:
            return t[:, :W]
    src = np.asarray(a[r0:r1, :W])
    if not str(dev).startswith("cuda"):
        return torch.from_numpy(np.ascontiguousarray(src))
    n = src.size
    buf = _PINNED.get("buf")
    if buf is None or buf.numel() < n:
        buf = _PINNED["buf"] = torch.empty(max(n, 1 << 24), dtype=torch.uint8).pin_memory()
    torch.cuda.current_stream().synchronize()             # the last async copy out of buf is done
    view = buf[:n].view(src.shape)
    view.numpy()[...] = src
    return view.to(dev, non_blocking=True)


def predict_array(model, arrs: list, st: dict, dev: str = "cuda", tile: int = TILE, halo: int = HALO,
                  fp16: bool = True, pw=None, norm: str = "local129", timing: dict | None = None) -> np.ndarray:
    """Logits (H, W) float32 for a list of IN_CHANS 2-D uint8 arrays (memmaps are fine).

    Tiles of `tile` px with a `halo` px margin on every side, NO overlap blending: each output
    pixel is computed once, from a tile whose interior it is. Row bands are staged to the card as
    uint8 and normalised there."""
    import torch
    H, W = arrs[0].shape[:2]
    out = np.empty((H, W), np.float32)
    base = getattr(model, "_orig_mod", model)
    mult = 2 ** (len(getattr(base, "widths", (0,) * 6)) - 1)
    dt = torch.float16 if (fp16 and dev.startswith("cuda")) else torch.float32
    for r0 in range(0, H, tile):
        r1 = min(H, r0 + tile)
        a0, a1 = max(0, r0 - halo), min(H, r1 + halo)
        import time as _t
        _t0 = _t.time()
        band = torch.empty((1, len(arrs), a1 - a0, W), dtype=torch.uint8, device=dev)
        for j, a in enumerate(arrs):
            band[0, j] = _to_dev_u8(a, a0, a1, W, dev)
        if timing is not None:
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            timing["read_h2d_s"] = timing.get("read_h2d_s", 0.0) + _t.time() - _t0
        for c0 in range(0, W, tile):
            c1 = min(W, c0 + tile)
            b0, b1 = max(0, c0 - halo), min(W, c1 + halo)
            x = band[:, :, :, b0:b1].to(torch.float32)
            x = apply_norm(x, norm, st)
            # pad to ONE fixed tile shape (tile + 2 halo, rounded to 2**(levels-1)) whenever the
            # canvas is that big, so cuDNN autotunes one shape instead of one per edge tile
            # (measured: per-shape autotuning made a 14 Mpx canvas cost 12 s of GPU time);
            # replicate padding sits outside the canvas and is cropped away
            fh = min(-(-(tile + 2 * halo) // mult) * mult, -(-(a1 - a0) // mult) * mult if H <= tile else 10 ** 9)
            fw = min(-(-(tile + 2 * halo) // mult) * mult, -(-(b1 - b0) // mult) * mult if W <= tile else 10 ** 9)
            ph, pw_ = max(0, fh - x.shape[2]), max(0, fw - x.shape[3])
            ph += (-(x.shape[2] + ph)) % mult
            pw_ += (-(x.shape[3] + pw_)) % mult
            if ph or pw_:
                x = torch.nn.functional.pad(x, (0, pw_, 0, ph), mode="replicate")
            x = x.to(dt).contiguous(memory_format=torch.channels_last)
            with torch.no_grad():
                y = model(x)[0, 0].float()
            out[r0:r1, c0:c1] = y[r0 - a0:r0 - a0 + (r1 - r0), c0 - b0:c0 - b0 + (c1 - c0)].cpu().numpy()
        del band
        if timing is not None:
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            timing["band_total_s"] = timing.get("band_total_s", 0.0) + _t.time() - _t0
        if pw is not None:
            try:
                pw.update(r1, None, batch=r1)
            except Exception:                            # noqa: BLE001, S110 - progress is never a gate
                pass
    return out


# D4 TEST-TIME AUGMENTATION (2026-09-30 shift/D4 study, FINDINGS): on the CURRENTLY deployed
# checkpoint (v5_ms, step 60000, ms33-129-257 norm) -- not the superseded v5 (step 40000,
# single-scale local129) the first TTA study scored -- D4 (8-view dihedral mean of
# probabilities) reads +0.008 to +0.023 AUC, CI excluding 0, on 3 of 6 scoreable groups
# (unseen open-data scrolls n=5, S5 n=11, S1 n=6), flat elsewhere, never significantly
# negative on a multi-segment group. A 4x4 px SHIFT ensemble (offsets {0,8,16,24}^2, the
# other arm that study measured) does NOT help this checkpoint -- it is flat to
# significantly NEGATIVE in 4 of 6 groups (CI excluding 0) -- so no shift option is shipped;
# the dense local normalisation (`local_normalise`, a sliding box filter) is already
# shift-covariant for interior pixels, unlike the OLDER architecture the shift idea was
# proposed against. D4 costs 8x a plain pass; default OFF (`VPIPE_INK9UM_STUDENT_TTA=d4`).
D4_VIEWS = tuple((k, f) for f in (0, 1) for k in range(4))   # (quarter-turns, flip-W-first), 8 distinct


def _d4_view(a, t):
    k, f = t
    if f:
        a = np.flip(a, axis=1)
    if k:
        a = np.rot90(a, k, axes=(0, 1))
    return np.ascontiguousarray(a)


def _d4_invert(p, t):
    k, f = t
    if k:
        p = np.rot90(p, -k, axes=(0, 1))
    if f:
        p = np.flip(p, axis=1)
    return np.ascontiguousarray(p)


def predict_array_d4(model, arrs: list, st: dict, dev: str = "cuda", **kw) -> np.ndarray:
    """Mean PROBABILITY (not logits -- the 8 views are not on one linear scale after a
    rotation/flip changes which replicate-padded edge a tile sees) over the D4 orbit.
    `arrs` must already be concrete per-layer 2-D uint8 arrays (not a lazy/remote view): a
    rotation materialises a full copy, which `predict_array`'s banded reader is built to
    avoid for the plain path, so D4 is accepted as the cost of the 8x it already is."""
    acc = None
    for t in D4_VIEWS:
        tarrs = [_d4_view(np.asarray(a), t) for a in arrs]
        lg = predict_array(model, tarrs, st, dev, **kw)
        p = _d4_invert(1.0 / (1.0 + np.exp(-lg.astype(np.float64))), t)
        acc = p if acc is None else acc + p
    return (acc / len(D4_VIEWS)).astype(np.float32)


def predict_inproc(layers_dir: str, mask_png: str | None, out_png: str, gpu: int = 0,
                   arm: str | None = None, compile_: bool = False, tta: str | None = None,
                   **spec) -> dict | None:
    """Write the probability PNG; return the stats dict (None on failure)."""
    import time
    ok, ckpt = available(arm=arm)
    if not ok:
        _alerts.alert(f"ink9um_student on this host: {ckpt}")
        return None
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu))
    import torch
    from PIL import Image
    from .grandprize_dense import _layer_array
    Image.MAX_IMAGE_PIXELS = None
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    files = layer_files(layers_dir)
    idx = select_layers(len(files))
    arrs = [_layer_array(files[i]) for i in idx]
    H, W = arrs[0].shape[:2]
    model, ck = load(ckpt, dev, compile_=compile_)
    norm = ck["config"].get("norm", "segment")
    st = segment_stats(arrs) if norm == "segment" else None
    from . import _progress
    pw = _progress.writer(H, (H, W), model="ink9um_student")
    t1 = time.time()
    timing: dict = {}
    halo = int(ck["config"].get("halo", HALO))
    tta = (tta or os.environ.get("VPIPE_INK9UM_STUDENT_TTA", "none")).lower()
    if tta == "d4":
        prob = predict_array_d4(model, arrs, st, dev, pw=pw, norm=norm, timing=timing, halo=halo)
    elif tta not in ("none", "", "0"):
        raise ValueError(f"ink9um_student tta={tta!r} not implemented (only 'none'/'d4' -- "
                         f"a shift ensemble was measured 2026-09-30 and does NOT help this "
                         f"checkpoint, so it was never wired in; see FINDINGS)")
    else:
        lg = predict_array(model, arrs, st, dev, pw=pw, norm=norm, timing=timing, halo=halo)
        prob = np.ascontiguousarray(lg, dtype=np.float32)
        np.negative(prob, out=prob)
        np.exp(prob, out=prob)
        prob += 1.0
        np.reciprocal(prob, out=prob)
    if dev == "cuda":
        torch.cuda.synchronize()
    t2 = time.time()
    # validity: the middle rendered layer > 0, exactly as run_ink9um.py masks the teacher, and
    # the finish stage's mask PNG on top when one is given
    mid = _layer_array(files[len(files) // 2])
    prob *= (np.asarray(mid[:H, :W]) > 0)
    if mask_png and os.path.exists(mask_png):
        mk = np.asarray(Image.open(mask_png).convert("L"))
        h, w = min(mk.shape[0], H), min(mk.shape[1], W)
        prob[:h, :w] *= (mk[:h, :w] > 127)
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    if pw is not None:
        try:
            pw.finish(H, mp=prob)
        except Exception:                                # noqa: BLE001, S110 - progress is never a gate
            pass
    Image.fromarray(np.clip(prob * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(out_png)
    stats = {"wall_s": round(time.time() - t0, 3), "gpu_s": round(t2 - t1, 3), "shape": [H, W],
             "read_h2d_s": round(timing.get("read_h2d_s", 0.0), 3),
             "compute_s": round(timing.get("band_total_s", 0.0) - timing.get("read_h2d_s", 0.0), 3),
             "setup_s": round(t1 - t0, 3),
             "mpx": round(H * W / 1e6, 3), "layers": idx, "norm": norm, "tile": TILE, "halo": halo,
             "device": dev, "tta": tta}
    with open(os.path.splitext(out_png)[0] + ".ink9um_student.json", "w") as fh:
        json.dump({"model": f"ink9um_student:{arm}" if arm else "ink9um_student", "checkpoint": ckpt,
                   "spec": SPEC, "metrics": describe(arm), "stats": stats}, fh, indent=1)
    return stats


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mask", default=None)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--arm", default=None)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--tta", default=None, help="'none' (default) or 'd4'; see predict_array_d4")
    a = ap.parse_args()
    raise SystemExit(0 if predict_inproc(a.layers, a.mask, a.out, gpu=a.gpu, arm=a.arm,
                                         compile_=a.compile, tta=a.tta) else 1)
