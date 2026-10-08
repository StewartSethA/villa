"""grandprize_dense: the Grand Prize TimeSformer with a dense decoder head.

The shipped Grand Prize model emits SIXTEEN numbers per 64x64 tile -- and they are a
linear map of the CLS token alone (`timesformer_pytorch.TimeSformer.forward`:
`self.to_out(x[:, 0])`), reshaped to 4x4 and bilinearly upsampled x16
(`inference_timesformer.py:328`). One output cell is therefore 16 px, 0.127 mm at
7.91 um/px, and the 26 x 4 x 4 patch tokens the backbone computes are discarded.

This family keeps that backbone and reads the patch tokens instead: mean and std over
the 26 depth frames give (B, 1024, 4, 4), four pixel-shuffle stages give (B, 1, 64, 64).
Output stride is 1 px instead of 16, at roughly the cost of the shipped model -- the
alternatives that buy resolution by tiling (stride 4, or 4x4 sub-cell phase averaging)
cost 16-27x.

Training: scripts/hires_ink/dense_head.py, on the transfer study's mesh-disjoint
manifest, Dice + SoftBCE against the full-resolution label with a 16x16 auxiliary head,
EMA. The checkpoint carries its own config and its held-out numbers.

Input contract, identical to the Grand Prize family it is fine-tuned from:
  * 26 layers starting at index 17 of a 65-layer surface volume;
  * 7.91 um/px;
  * intensity normalised to mean 150 / std 25 (FINDINGS: the GP model needs this);
  * 64 px tiles; stride is free -- 32 is plenty, since the output is already per-pixel.

DEFAULT OFF. `VPIPE_GP_DENSE=1` enables it, and `available()` refuses when the
checkpoint is not on the host, so a box without the weights SKIPS with a reason.
"""
from __future__ import annotations

import json
import os

import numpy as np
from ... import alerts as _alerts

# reads an in-memory native-stack view (stages/memstack.py) through its layer readers
MEMSTACK_READY = True

IN_CHANS = 26
LAYER_START = 17
TILE = 64
STRIDE = 32

SPEC = {
    "layers": IN_CHANS,
    "layer_start": LAYER_START,
    "tile": TILE,
    "stride": STRIDE,
    "um_per_px": 7.91,
    "normalisation": "gp_mean150_std25",
    "arch": "timesformer_dense_pixelshuffle",
    "output_stride_px": 1,             # the shipped Grand Prize head is 16
    "trained_on": "PHercParis4 (Scroll 1) + PHerc0172 (Scroll 5), mesh-disjoint split",
    "license": "same as the Grand Prize weights it fine-tunes",
}

ENV_CKPT = "VPIPE_GP_DENSE_CKPT"
# Named arms: `grandprize_dense:<arm>` resolves to grandprize_dense_<arm>.ckpt in the first
# model root that has it. The arms are exported mid-training by
# scripts/hires_ink/export_models.py and are PRELIMINARY -- each ships a .metrics.json
# recording its step count, held-out numbers and sha256, and `describe()` returns it so a
# caller can print the caption rather than presenting a half-trained head as production.
# `unet32` is NOT a Grand Prize fine-tune -- it carries no TimeSformer backbone at all
# (135 k parameters, a 3D->2D U-Net with a MEASURED 28 px receptive field). It is dispatched
# from this family only because `dense_head.load` reads the arch out of the checkpoint and
# every arm shares one input contract; `describe()` reports its own arch so a caller is not
# told it is running a TimeSformer.
ARMS = ("dense64", "narrow32", "narrow16", "distill", "unet32")
MODEL_ROOTS = ("var/models",
               "/dev/shm/vpipe/ScrollPrizeTutorial/current/var/models",
               "/dev/shm/vpipe/var/models")
DEFAULT_CKPTS = tuple(os.path.join(r, "grandprize_dense.ckpt") for r in MODEL_ROOTS)


def training_frame():
    from ...frame import MODELS, ModelSpec
    return MODELS.get("grandprize_dense",
                      ModelSpec("grandprize_dense", SPEC["um_per_px"], SPEC["layers"])).frame(
        source_id="model:grandprize_dense")


def check_frame(frame, tol: float = 0.05) -> None:
    training_frame().assert_compatible(frame, tol=tol, what="grandprize_dense input")


def enabled() -> bool:
    return os.environ.get("VPIPE_GP_DENSE", "0") == "1"


def checkpoint_path(arm: str | None = None) -> str | None:
    """The checkpoint for `grandprize_dense` or for a named arm `grandprize_dense:<arm>`."""
    if os.environ.get(ENV_CKPT) and not arm:
        cands = [os.environ[ENV_CKPT]]
    elif arm:
        cands = [os.environ.get(f"VPIPE_GP_DENSE_CKPT_{arm.upper()}", "")] + \
                [os.path.join(r, f"grandprize_dense_{arm}.ckpt") for r in MODEL_ROOTS]
    else:
        cands = list(DEFAULT_CKPTS)
    for c in cands:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return None


def split_name(name: str) -> tuple[str, str | None]:
    """'grandprize_dense:narrow32' -> ('grandprize_dense', 'narrow32')."""
    fam, _, arm = name.partition(":")
    return fam, (arm or None)


def describe(arm: str | None = None) -> dict:
    """The exported metrics beside the weights -- step count, held-out numbers, sha256 and
    the PRELIMINARY caption. Empty dict when the sidecar is missing."""
    p = checkpoint_path(arm)
    if not p:
        return {}
    m = p.replace(".ckpt", ".metrics.json")
    try:
        return json.load(open(m))
    except Exception:                                    # noqa: BLE001 - a missing sidecar is not fatal
        return {"checkpoint": p, "preliminary": True, "caption": "PRELIMINARY (no metrics sidecar)"}


def available(fleet=None, arm: str | None = None) -> tuple[bool, str]:
    """CAN this host run the family -- are the weights here. NOT "should it run".

    `available()` used to fail closed on `enabled()`, i.e. on VPIPE_GP_DENSE. That made the
    family unreachable from the hub database: the Workflow tab could tick
    `finish.grandprize_dense_enabled`, the plan would carry it, and the availability check
    in stages/finish.py then dropped it with "disabled (set VPIPE_GP_DENSE=1)" -- a
    checkbox that ran nothing, and the prospect path already worked around it by setting
    the variable itself before calling predict(). Whether a family SHOULD run is the
    caller's question: the finish stage asks the settings row, a prospect job asks whether
    it was named. `enabled()` stays as the env flag those callers and the tests read."""
    p = checkpoint_path(arm)
    if p is None:
        which = f"grandprize_dense:{arm}" if arm else "grandprize_dense"
        return False, f"{which} checkpoint not on this host (set {ENV_CKPT}, or rsync it into var/models)"
    return True, p


def predict(layers_dir: str, mask_png: str, out_png: str, gpu: int = 0, arm: str | None = None,
            fleet=None, log: str | None = None, timeout: int = 3600, **spec) -> bool:
    """Run the dense head over one rendered stack. The worker's own interpreter (the
    peer's deps-only venv) has no torch, so unless torch is importable here the work is
    handed to the Grand Prize venv the way i3d/resnet3d_1667 do (2026-09-08: every dense
    pass on the V100 workers died with "No module named 'torch'")."""
    try:
        import torch  # noqa: F401
        return _predict_inproc(layers_dir, mask_png, out_png, gpu=gpu, arm=arm, **spec)
    except ImportError:
        pass
    from . import i3d
    from ... import config
    from ..finish import _run
    if fleet is None:
        fleet = config.load()
    py = i3d.python(fleet)
    if not py:
        _alerts.alert("grandprize_dense on this host: no Grand Prize venv -- run `vpipe setup models`")
        return False
    ok, why = available(arm=arm)
    if not ok:
        _alerts.alert(f"grandprize_dense on this host: {why}")
        return False
    cmd = [py, "-m", "vesuvius_pipeline.stages.ink_models.grandprize_dense", "--layers", layers_dir,
           "--out", out_png, "--gpu", str(gpu)] + (["--mask", mask_png] if mask_png and os.path.exists(mask_png) else []) \
          + (["--arm", arm] if arm else [])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
               PYTHONPATH=str(i3d._root(fleet) / "src") + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""))
    rc = _run(cmd, log or os.path.splitext(out_png)[0] + ".log", timeout, env=env)
    return rc == 0 and os.path.exists(out_png) and os.path.getsize(out_png) > 0



def _layer_files(layers_dir: str, start: int, n: int) -> list:
    """The model's `n`-layer window, starting at `start` when the stack is deep enough.

    When it is not, the window is CENTRED on the stack instead of failing -- as long as the
    stack has at least `n` layers at all. The scaled renders are shallower than the 65-layer
    default (a 2x render measured 41 layers), so `start=17` ran off the end: "24 layers from
    index 17, need 26" failed every variant on 68 finishes in one day while 41 layers were
    sitting there. Centring keeps the same physical intent -- a window about the surface --
    and the choice is announced, never silent. A stack with fewer than `n` layers still
    refuses, because a model fed fewer layers than it was trained on is reading nonsense.
    """
    from .. import memstack as _MS
    all_files = _MS.list_names(layers_dir)          # an in-memory view of the native stack
    if all_files is None:
        all_files = sorted(f for f in os.listdir(layers_dir) if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")))
    if len(all_files) < n:
        raise RuntimeError(f"{layers_dir}: only {len(all_files)} layers in the stack, need {n}")
    if start + n > len(all_files):
        centred = (len(all_files) - n) // 2
        print(f"[grandprize_dense] {layers_dir}: {len(all_files)} layers, start {start} would run "
              f"off the end -- using the centred window {centred}..{centred + n - 1}", flush=True)
        start = centred
    return [os.path.join(layers_dir, f) for f in all_files[start:start + n]]


def _layer_array(fp: str) -> "np.ndarray":
    """One rendered layer as a 2-D uint8 array WITHOUT a host copy where possible: the
    renderer writes one uncompressed strip per TIFF, which tifffile.memmap maps straight
    from the page cache (0.0 s against 5.2 s to decode and 24 s to copy under load)."""
    from .. import memstack as _MS
    lz = _MS.read_layer(fp)          # resampled in memory from the native stack, never on disk
    if lz is not None:
        return lz
    import tifffile
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    if fp.lower().endswith((".tif", ".tiff")):
        try:
            a = tifffile.memmap(fp, mode="r")
        except (ValueError, OSError):
            a = tifffile.imread(fp)
    else:
        a = np.asarray(Image.open(fp).convert("L"))
    if a.ndim == 3:
        a = a[..., 0]
    if a.dtype != np.uint8:
        a = np.clip(a.astype(np.float32) * ((255.0 / 65535.0) if a.max() > 255 else 1.0), 0, 255).astype(np.uint8)
    return a


# Bounded-memory input assembly for _predict_inproc. Both helpers stream the rendered
# layers from the page cache (tifffile.memmap) a row band at a time and never hold a
# whole-canvas float array on the host or on the card.
BAND_VRAM_FRACTION = float(os.environ.get("VPIPE_GP_DENSE_BAND_FRAC", "0.40"))
BAND_MIN_ROWS = 256          # a band below this costs more in halo than it saves in VRAM


def _is_oom(exc: BaseException) -> bool:
    """Is this exception the card being full?

    Not every out-of-memory arrives as `torch.cuda.OutOfMemoryError`. When VRAM is exhausted
    before cuBLAS has its workspace, the first failure is
    `RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate(handle)`
    -- a plain RuntimeError, from inside the first Linear of the TimeSformer. Catching only
    the typed error let that one through as a traceback, which the finish stage can report
    only as a dead variant (measured 2026-09-20 on the hub's 16 GB cards).
    """
    import torch
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc)
    return isinstance(exc, RuntimeError) and (
        "CUBLAS_STATUS_ALLOC_FAILED" in msg or "out of memory" in msg.lower()
        or "CUDA error: out of memory" in msg)


def _material_stats(mid_file: str, H: int, W: int, dev: str = "cpu", chunk: int = 2048):
    """(mean, std) of the MATERIAL pixels (>0) of the middle layer, over the WHOLE canvas.

    The normalisation has to be global: computing it per band would give each band its own
    affine and put visible steps across the map.

    ON THE CPU, DELIBERATELY. This is a mean and a variance over one uint8 layer -- numpy
    over the memmap does it without a single byte of VRAM. It used to stage each row chunk
    on the GPU, and on a card shared with a resident ink server and a prospect worker that
    120 MB was the allocation that failed: the variant died in the STATS pass, before the
    band loop that was carefully written to survive a full card (measured 2026-09-20 on the
    hub's 16 GB cards, 91 MiB free). `dev` is accepted and ignored so callers need not care.

    float64 accumulation over uint8 inputs, so the sums are exact.
    """
    a = _layer_array(mid_file)
    n = 0
    s_ = s2 = 0.0
    for r0 in range(0, H, chunk):
        x = np.asarray(a[r0:min(H, r0 + chunk), :W], dtype=np.uint8)
        m = x > 0
        k = int(m.sum())
        if k:
            v = x[m].astype(np.float64)
            n += k
            s_ += float(v.sum())
            s2 += float((v * v).sum())
            del v
        del x, m
    if n < 1000:
        return float("nan"), float("nan")
    mu = s_ / n
    return mu, max(0.0, s2 / n - mu * mu) ** 0.5


def _band_rows(H: int, W: int, dev: str, stride: int) -> int:
    """How many OUTPUT rows one band may cover, from the free VRAM right now.

    Sized for the input volume plus core.predict's two (Hp, Wp) float32 accumulators;
    core.predict sizes its own tile batch from free VRAM and halves it on OOM, so the
    batch is not budgeted here. Aligned to `stride` so the band's tile grid is the global
    one (see the comment at the call site)."""
    import torch
    # torch's pad/unfold kernels index with int32: a band tensor of more than 2**31 elements
    # raises "input tensor must fit into 32-bit index math" however much VRAM is free. A
    # 32 GB V100 sized the _s200 arm (~10k px wide, IN_CHANS layers) at the whole segment
    # and every _s200 variant on .142 failed that way (2026-09-24). Cap rows by elements.
    idx_rows = max(stride, (((2**31 - 1) // 2) // max(IN_CHANS * (W + 2 * TILE), 1) // stride) * stride)
    forced = int(os.environ.get("VPIPE_GP_DENSE_BAND_ROWS", "0"))   # tests and operators
    if forced > 0:
        return min(H, idx_rows, max(stride, (forced // stride) * stride))
    if not str(dev).startswith("cuda"):
        return min(H, idx_rows)
    try:
        free, _ = torch.cuda.mem_get_info()
    except Exception:                                  # noqa: BLE001 - no CUDA context yet
        return H
    wp = W + TILE
    per_row = IN_CHANS * wp * 2 + wp * 8 + 4096        # fp16 volume + acc + cnt + slack
    rows = int((free * BAND_VRAM_FRACTION) // max(per_row, 1))
    rows = (rows // stride) * stride
    # BAND_MIN_ROWS is a floor on EFFICIENCY, not on memory: below it the halo costs more
    # than the band saves. It must not be able to push the band ABOVE what fits, which is
    # what it did on a card with 145 MiB free running the 2x-native arm (twice the canvas
    # width, so twice the bytes per row): the floor forced a 294 MiB allocation that could
    # not succeed, and every _s200 variant failed on the hub's 16 GB cards.
    if rows < BAND_MIN_ROWS:
        rows = max(stride, rows) if rows > 0 else stride
    else:
        rows = max(BAND_MIN_ROWS, rows)
    return min(H, idx_rows, max(rows, stride))


def _band_volume(files: list, in0: int, in1: int, W: int, dev: str, fdt, mu: float, sd: float):
    """(1, IN_CHANS, in1-in0, W) on `dev`, normalised with the GLOBAL affine.

    Zero host copies: each layer goes memmap -> GPU as a page-cache view and the affine
    runs on the card, which is what the whole-canvas version did and the reason it is kept."""
    import torch
    import warnings
    h = in1 - in0
    vol = torch.empty((1, IN_CHANS, h, W), dtype=fdt, device=dev)
    scale_ok = np.isfinite(mu) and np.isfinite(sd) and sd >= 1e-3
    for j in range(IN_CHANS):
        a = _layer_array(files[j])
        xg = None
        rd = getattr(a, "rows_device", None)          # an in-memory view resampling on the card
        if rd is not None and str(dev).startswith("cuda"):
            xg = rd(in0, in1)
            xg = None if xg is None else xg[:, :W]
        if xg is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                xg = torch.from_numpy(np.asarray(a[in0:in1, :W])).to(dev)
        xf = xg.float()
        if scale_ok:
            xf = ((xf - mu) / sd * 25.0 + 150.0).clamp_(1.0, 255.0) * (xf > 0)
        vol[0, j] = (xf / 255.0).to(fdt)
        del xg, xf, a
    return vol


def _predict_inproc(layers_dir: str, mask_png: str, out_png: str, gpu: int = 0, arm: str | None = None,
                    **spec) -> bool:
    """Write a per-pixel ink probability PNG for one rendered layer stack.

    Returns False rather than raising when the family is unavailable, so the finish
    stage skips it the way it skips every other optional family."""
    ok, why = available(arm=arm)
    if not ok:
        _alerts.alert(f"grandprize_dense on this host: {why}")
        return False
    ckpt = why
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu))
    import sys
    import torch
    from PIL import Image
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    sys.path.insert(0, os.path.join(repo, "scripts", "hires_ink"))
    sys.path.insert(0, os.path.join(repo, "scripts"))
    import core                      # noqa: E402
    import dense_head                # noqa: E402
    # A LONG-RUNNING WORKER MUST NOT KEEP STALE MODEL CODE. `import` caches the first version
    # for the life of the process, so the hub's prospect worker (up 5.4 h) went on
    # double-normalising after the fix (64a0436) had shipped: its job 1867 produced 5 white
    # dense64 maps while fleet hosts, restarted onto the fix, produced healthy ones
    # (2026-09-24). The model code is reloaded whenever its file is newer than the loaded copy,
    # and a copy imported from outside this repo is named loudly.
    import importlib
    for _m in (core, dense_head):
        _f = getattr(_m, "__file__", None) or ""
        if not os.path.abspath(_f).startswith(repo):
            print(f"[grandprize_dense] WARNING: {_m.__name__} imported from {_f}, not {repo}", flush=True)
        try:
            _mt = os.path.getmtime(_f)
        except OSError:
            continue
        if getattr(_m, "_vp_loaded_mtime", None) is not None and _mt > _m._vp_loaded_mtime:
            print(f"[grandprize_dense] {_m.__name__} changed on disk: reloading", flush=True)
            _m = importlib.reload(_m)
        _m._vp_loaded_mtime = _mt
    core, dense_head = sys.modules["core"], sys.modules["dense_head"]
    Image.MAX_IMAGE_PIXELS = None
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    start = int(spec.get("layer_start", LAYER_START))
    stride = int(spec.get("stride", STRIDE))
    # THE INPUT WINDOW ONLY, AS UINT8, VIA MEMMAP. It used to be transfer.gp_model.read_stack:
    # all 65 layers decoded into float32 (19.8 GB for a 28 cm2 sheet), then intensity_affine
    # and `/ 255.0` making float64 and float32 copies of the 26-layer slice -- ~40 GB of fresh
    # pages per pass on a box whose page-fault path was measured at 24 s per 305 MB layer under
    # load (2026-09-15). Every V100 finish sat 1-2 h in that CPU-side prep, single-core, before
    # its first CUDA call, with all eight cards idle. Now: 26 layers memmapped (no decode, no
    # copy), padded once in uint8 so core.predict pads nothing on the card, and the affine
    # normalisation runs on the GPU a layer at a time into an fp16 volume (3.9 GB at 76 MP).
    # ZERO HOST COPIES. Measured on the V100 box under its grow load (2026-09-15): a fresh
    # 300 MB host allocation costs 28 s and a 1.2 GB float32 copy 108 s (the kernel is in
    # direct reclaim: 680 GB of tmpfs + 1 TB of page cache, Committed_AS above the commit
    # limit), so a 26-layer uint8 stack + a padded copy was still minutes of CPU. Each layer
    # goes memmap -> GPU as a page-cache view; the material stats, the affine and the
    # replicate padding all happen on the card.
    files = _layer_files(layers_dir, start, IN_CHANS)
    a0 = _layer_array(files[0]); H, W = a0.shape[:2]
    fdt = torch.float16 if dev == "cuda" else torch.float32
    # THE MODEL FIRST, then the memory budget: the weights have to be in VRAM before
    # `free` means anything for sizing the input volume below.
    try:
        model = dense_head.load(ckpt, dev)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if not _is_oom(e):
            raise
        # Nothing to shrink here -- the weights are the weights. A card this full cannot
        # host this family right now; say so and skip, rather than raise a traceback the
        # finish stage can only report as "0 variants".
        print(f"[grandprize_dense] skipped: cannot load the model, card full "
              f"({torch.cuda.mem_get_info()[0] / 1e6:.0f} MB free)", flush=True)
        return False
    # This family's tiling lives inside core.predict, so there is no per-batch hook to
    # publish from. It still reports itself LIVE -- one frame at the start, one at the end.
    from . import _progress
    pw = _progress.writer(1, (H, W), model="grandprize_dense")
    if pw is not None:
        pw.update(0, None, batch=0)

    def tile_fn(t):
        with torch.no_grad():
            return model.forward(t.float())[0]

    # BANDED, NOT WHOLE-CANVAS. `vol` used to be one (1,26,H,W) fp16 tensor: 2.5 GiB for a
    # 7010 x 7323 render, which is more than the FREE VRAM on a card that also carries a
    # resident ink server (8.6 GB) and a prospect worker (7.2 GB). Every one of the four
    # grandprize_dense variants therefore raised OutOfMemoryError on the hub's 16 GB V100s
    # for days, each finish published 3 of 7 variants, and `scheduler.finish_candidates`
    # -- which re-queues a segment until EVERY planned variant is newer than its surface --
    # re-finished the same three segments on a loop while 200 candidates waited (found
    # 2026-09-19: 187 "ok" finishes in 24 h, all of them repeats of three segments).
    # Peak VRAM is now bounded by the band, not by the segment's area.
    #
    # The band is EXACT, not an approximation. A tile at origin y covers rows y..y+TILE-1,
    # so output rows [r0, r1) are complete once the band carries input rows
    # [floor_to_stride(r0 - TILE) .. r1 + TILE - 1) -- every tile that touches the interior
    # exists inside the band. Band origins are aligned to `stride`, so the band's local tile
    # grid IS the global one, and core.predict normalises by its own per-pixel weight sum
    # (`acc / cnt`), so an interior pixel sees the same tiles and the same weights it would
    # in a whole-canvas pass. Replicate padding at the canvas edges is unchanged.
    mu, sd = _material_stats(files[IN_CHANS // 2], H, W, dev)
    band = _band_rows(H, W, dev, stride)
    m = np.empty((H, W), dtype=np.float32)
    stats: dict = {}
    nb = 0                       # counted, not predicted: the band shrinks on OOM
    r0 = 0
    while r0 < H:
        r1 = min(H, r0 + band)
        in0 = max(0, ((r0 - core.TILE) // stride) * stride) if r0 else 0
        in1 = min(H, r1 + core.TILE - 1)
        vol = mb = None
        try:
            vol = _band_volume(files, in0, in1, W, dev, fdt, mu, sd)
            mb, st = core.predict(None, vol, stride=stride, blend="gauss", batch=None, tile_fn=tile_fn)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if not _is_oom(e):
                raise
            # Another tenant took the card between sizing and allocating -- these cards are
            # shared with a resident ink server and a prospect worker, and free VRAM at
            # model-load time is a snapshot, not a reservation. Halve and retry; give up
            # only when a band of one tile row still will not fit, and say so rather than
            # emitting a partial map.
            del vol, mb                       # drop the partial band before reclaiming
            vol = mb = None
            torch.cuda.empty_cache()
            if band <= stride:
                print(f"[grandprize_dense] out of memory at the smallest band ({stride} rows); "
                      f"card too full for this canvas ({H}x{W})", flush=True)
                return False
            band = max(stride, (band // 2 // stride) * stride)
            print(f"[grandprize_dense] out of memory; band -> {band} rows and retrying", flush=True)
            continue
        m[r0:r1] = mb[r0 - in0:r1 - in0, :W]
        del vol, mb
        if dev == "cuda":
            torch.cuda.empty_cache()
        nb += 1
        bi = nb
        r0 = r1
        for k, v in st.items():                   # summed over bands where that makes sense
            if isinstance(v, (int, float)) and k in ("n_tiles", "wall_s", "oom_retries", "gpu_s"):
                stats[k] = (stats.get(k) or 0) + v
            else:
                stats[k] = v
        if pw is not None:
            # A LIVE FRAME PER BAND (user, 2026-09-24: "make sure that it shows animated at the
            # top in real-time"). This family used to publish one frame at the start and one at
            # the end, so the live view sat empty and then popped. Each finished band is now
            # published as the probability map so far (<= 1024 px, rows not yet run left empty)
            # with its coverage, so the glow lands band by band as the card produces it.
            try:
                step = max(1, int(np.ceil(max(H, W) / 1024.0)))
                done_v = m[:r1:step, ::step]
                full = np.zeros(((H + step - 1) // step, (W + step - 1) // step), np.float32)
                full[:done_v.shape[0], :done_v.shape[1]] = 1.0 / (1.0 + np.exp(-np.clip(done_v, -30, 30)))
                cov = np.zeros_like(full)
                cov[:done_v.shape[0], :] = 1.0
                pw.tiles_total = max(int(getattr(pw, "tiles_total", 0) or 0), -(-H // max(1, band)))
                pw.update(nb, full, bbox=(r0 // step, 0, -(-r1 // step), full.shape[1]), batch=nb,
                          coverage=cov)
            except Exception:                  # noqa: BLE001 - a dropped frame, never a failed run
                pw.update(nb, None, batch=bi + 1)
    stats["bands"] = int(nb)
    stats["band_rows"] = int(band)
    del a0
    if dev == "cuda":
        stats["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    del model
    prob = np.ascontiguousarray(m, dtype=np.float32)          # sigmoid IN PLACE: no float64 temporaries
    np.negative(prob, out=prob); np.exp(prob, out=prob); prob += 1.0; np.reciprocal(prob, out=prob)
    valid = None
    if mask_png and os.path.exists(mask_png):
        mk = np.asarray(Image.open(mask_png).convert("L"))
        h, w = min(mk.shape[0], prob.shape[0]), min(mk.shape[1], prob.shape[1])
        valid = np.zeros(prob.shape, dtype=bool)
        valid[:h, :w] = mk[:h, :w] > 127
        prob[:h, :w] *= valid[:h, :w]
        prob[h:, :] = 0.0; prob[:, w:] = 0.0
        del mk
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    if pw is not None:
        pw.finish(1, mp=prob)
    np.multiply(prob, 255.0, out=prob)
    u8 = prob.astype(np.uint8)
    check_output_diversity(u8, valid, family=f"grandprize_dense:{arm}" if arm else "grandprize_dense",
                            ckpt=ckpt, out_png=out_png)
    Image.fromarray(u8).save(out_png)
    side = os.path.splitext(out_png)[0] + ".grandprize_dense.json"
    with open(side, "w") as fh:
        json.dump({"model": f"grandprize_dense:{arm}" if arm else "grandprize_dense",
                   "checkpoint": ckpt, "spec": SPEC, "metrics": describe(arm),
                   "layer_start": start, "stats": stats, "shape": list(prob.shape)}, fh, indent=1)
    return True


# A FAILURE IS NEVER SILENT (CLAUDE.md): a checkpoint that collapsed during training --
# dead ReLUs, a mid-run export, a normalisation mismatch -- produces a map that is
# essentially constant everywhere, whatever the input. Nothing about that FAILS: `predict()`
# returns True, the PNG exists and is non-empty, `scheduler.finish_candidates` is satisfied,
# and the family looks healthy on every dashboard that counts files rather than reading them.
# This is exactly the §2026-09-15 pixelclf_32x13 incident: every production map (25 + 13 +
# 239 + 166 + ... of them) carried only 3 distinct grey levels (~102-104/255) because the
# deployed checkpoint was a mid-run, non-converged export (its OWN .metrics.json says so:
# "PRELIMINARY ... not converged, not a production model"), confirmed directly against the
# weights (held-out AUC 0.477, chance) and reproduced here with THREE unrelated synthetic
# inputs (uniform noise at two scales, a smooth gradient) that all come back std <= 0.004 in
# probability, 9 distinct uint8 levels -- the model is not reading its input. No wrapper
# preprocessing bug reproduces or fixes this; the fix is to never train, export or ship a
# checkpoint whose output cannot clear this floor.
MIN_OUTPUT_LEVELS = 20


def check_output_diversity(u8_map: "np.ndarray", valid: "np.ndarray | None" = None,
                           *, family: str = "", ckpt: str = "", out_png: str = "",
                           min_levels: int = MIN_OUTPUT_LEVELS) -> int:
    """Count distinct uint8 levels in the VALID region of a published probability map and
    alert loudly (never raise -- a degenerate map is still a map the pipeline can inspect)
    when it is at or below `min_levels`. Returns the count, so a caller or a test can assert
    on it directly instead of grepping for the alert string."""
    region = u8_map if valid is None else u8_map[valid.astype(bool)]
    if region.size == 0:
        return 0
    n = int(np.unique(region).size)
    if n <= min_levels:
        _alerts.alert(f"{family or 'grandprize_dense'}: published map has only {n} distinct "
                      f"grey level(s) over {region.size} valid px (checkpoint {ckpt!r}, "
                      f"{out_png!r}) -- the model is very likely not reading its input "
                      f"(dead/undertrained checkpoint); see grandprize_dense.check_output_diversity")
    return n


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--mask", default=None); ap.add_argument("--gpu", default="0"); ap.add_argument("--arm", default=None)
    a = ap.parse_args()
    ok = _predict_inproc(a.layers, a.mask, a.out, gpu=a.gpu, arm=a.arm)
    raise SystemExit(0 if ok else 1)
