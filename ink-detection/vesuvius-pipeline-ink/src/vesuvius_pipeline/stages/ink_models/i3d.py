"""i3d: InceptionI3d + conv decoder, the Grand Prize winner repo's `inference_resnet3d.py`
family (RegressionPLModel with enc='i3d'), as an ink family over a RENDERED layer stack.

Why this exists. Our Scroll 4 (PHerc1667) fine-tunes (the snapshot directory named in
docs/wiki/ink_models.md) are Lightning checkpoints whose hyper_parameters read {'enc': 'i3d', 'size': 64, 'with_norm': False} and whose
state_dict keys are `backbone.Conv3d_1a_7x7.*` / `decoder.logit.*`. That is a
DIFFERENT NETWORK from the Grand Prize TimeSformer, so dropping one into
var/models/grandprize_<tag>.ckpt cannot work -- the TimeSformer wrapper would fail to
load it, or worse, load it `strict=False` and run an untrained decoder that still
emits a plausible grey image. Hence its own family with its own wrapper.

THE INPUT CONTRACT IS THE POINT OF THIS FILE. Recovered from the repo's own
inference script rather than guessed, and carried in SPEC so the sweep honours it:

  * 30 layers, indices 17..46 of a 65-layer stack -- `start_f = 17`,
    `end_f = start_f + in_chans` at inference_resnet3d.py:677. NOT all 65, and not
    centred by arithmetic: 17 is what the released runs used.
  * each layer read as 8-bit grey and CLIPPED to [0, 200] before stacking
    (read_image_mask), then scaled by 1/255 -- albumentations `Normalize(mean=[0]*30,
    std=[1]*30)` is a divide-by-255 and nothing else, so there is no mean/std
    standardisation here. This is the opposite of the Grand Prize contract, which
    wants the stack normalised to mean 150 / std 25; feeding a GP-normalised stack to
    this net is out of contract.
  * tiles 64 x 64, stride 64 // 3 = 21, only where the mask is fully set;
  * the decoder emits 16 x 16 per tile, bilinearly upsampled x4 to 64 and accumulated
    with a Gaussian window (gkern(64, 1) / max) over a plain count, so overlapping
    tiles blend rather than seam;
  * sigmoid, then divide by the count. Output is at the layer stack's own resolution.

Voxel scale. These checkpoints were fine-tuned on Scroll 4 renders in the ~7.91 um
Grand Prize frame; the family does not resample, it records `um_per_px` in SPEC and
the caller is responsible for handing it a stack at that pitch (the same discipline
stages/ink3d.py applies by resampling).

Frames and process split. Module-level imports are numpy only: this file is the stage
module under the repo venv AND the inference worker under the Grand Prize repo's torch
interpreter (`python -m vesuvius_pipeline.stages.ink_models.i3d`), because torch does
not live in the repo venv. The net's definition is imported from the Grand Prize repo
checkout (`Vesuvius-Grandprize-Winner/models/i3dallnl.py`) rather than vendored, so
there is exactly one copy of it on disk.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

# ---- the input contract ---------------------------------------------------------------
LAYER_START = 17            # inference_resnet3d.py:677  start_f = 17
IN_CHANS = 30               # CFG.in_chans
TILE = 64                   # CFG.tile_size == CFG.size
STRIDE = TILE // 3          # 21
CLIP_MAX = 200              # read_image_mask: np.clip(image, 0, 200)
UPSCALE = 4                 # decoder 16 -> 64
GP_TARGET_UM = 7.91

SPEC = {
    "layers": IN_CHANS,
    "layer_start": LAYER_START,
    "layer_stack": 65,
    "tile": TILE,
    "stride": STRIDE,
    "um_per_px": GP_TARGET_UM,
    "normalisation": "clip0_200_div255",   # NOT the Grand Prize mean150/std25
    "arch": "i3d",
}


# ---- gating (mirrors stages/ink3d.py) ---------------------------------------------------

def training_frame():
    """The Frame this checkpoint was trained in. A stack handed to `predict` must match
    it (`Frame.assert_compatible`); this family does NOT resample, so a mismatch is a
    silent wrong answer rather than an error."""
    from ...frame import MODELS, ModelSpec
    return MODELS.get("i3d", ModelSpec("i3d", SPEC["um_per_px"], SPEC["layers"])).frame(
        source_id="model:i3d")


def check_frame(frame, tol: float = 0.05) -> None:
    """Refuse a render whose pitch is not this model's. Raises FrameError."""
    training_frame().assert_compatible(frame, tol=tol, what="i3d input")


def enabled() -> bool:
    """Off in production until scored; VPIPE_I3D=1 enables the family everywhere."""
    return os.environ.get("VPIPE_I3D", "0") == "1"


def _root(fleet):
    from ... import config
    return fleet.root or config.repo_root()


def checkpoints(fleet) -> dict[str, str]:
    """tag -> path, over var/models/i3d_*.ckpt plus $VPIPE_I3D_CKPT (tag `env`)."""
    out: dict[str, str] = {}
    env = os.environ.get("VPIPE_I3D_CKPT")
    if env:                       # an explicit checkpoint is an override, not an addition
        return {"env": env} if os.path.exists(env) else {}
    for p in sorted(glob.glob(str(_root(fleet) / "var" / "models" / "i3d_*.ckpt"))):
        out[os.path.basename(p)[len("i3d_"):-len(".ckpt")]] = p
    return out


def code_dir(fleet) -> str | None:
    """The Grand Prize repo checkout holding models/i3dallnl.py."""
    for p in (os.environ.get("VPIPE_I3D_CODE"),
              str(_root(fleet) / "Vesuvius-Grandprize-Winner")):
        if p and os.path.isfile(os.path.join(p, "models", "i3dallnl.py")):
            return p
    return None


def python(fleet) -> str | None:
    """torch lives in the Grand Prize repo's venv, not in the repo venv."""
    for p in (os.environ.get("VPIPE_I3D_PY"),
              str(_root(fleet) / "Vesuvius-Grandprize-Winner" / ".venv" / "bin" / "python")):
        if p and os.path.exists(p):
            return p
    return None


def available(fleet) -> tuple[bool, str]:
    if not checkpoints(fleet):
        return False, "i3d: no checkpoint (var/models/i3d_*.ckpt)"
    if code_dir(fleet) is None:
        return False, "i3d: models/i3dallnl.py missing (Vesuvius-Grandprize-Winner checkout)"
    if python(fleet) is None:
        return False, "i3d: no torch interpreter (Vesuvius-Grandprize-Winner/.venv)"
    return True, "ok"


# ---- the family's entry point -----------------------------------------------------------
def _headroom(fleet, gpu) -> bool:
    """Is there room to spawn our own process on this card? Waits if not.

    Spawning into a full card is how a refused request became a crashed one."""
    if os.environ.get("INK_SERVER", "1") == "0":
        return True
    try:
        import sys as _s
        _s.path.insert(0, str(_root(fleet) / "ink_dashboard"))
        import ink_client
    except ImportError:
        return True
    return ink_client.wait_for_headroom(
        gpu, float(os.environ.get("INK_SERVER_MIN_FREE_GB", "6")),
        max_wait_s=float(os.environ.get("INK_WAIT_S", "600")))


def _serve(fleet, gpu, family, layers_dir, mask_png, out_png, model_path, code, s, spec):
    """Ask this card's resident ink server. -> True if it produced `out_png`.

    Soft on every path: no server, a refusal, a timeout or a missing output all
    return False and the caller spawns. The server runs the family's OWN worker,
    so a served map is the spawned map (verified byte-identical)."""
    if os.environ.get("INK_SERVER", "1") == "0":
        return False
    try:
        import sys as _s
        _s.path.insert(0, str(_root(fleet) / "ink_dashboard"))
        import ink_client
    except ImportError:
        return False
    import tempfile
    out_dir = tempfile.mkdtemp(prefix="served_", dir=os.path.dirname(os.path.abspath(out_png)))
    req = {"family": family, "segment_id": os.path.basename(out_png)[:-4] or family,
           "layers_dir": layers_dir, "mask_png": mask_png or "", "model_path": model_path,
           "code": code, "out_path": out_dir, "segment_path": os.path.dirname(layers_dir),
           "stride": s.get("stride"), "tile": s.get("tile"),
           "layer_start": s.get("layer_start"), "in_chans": s.get("layers"),
           "bs": int(os.environ.get("VPIPE_I3D_BS", "64")),
           "reverse": bool(spec.get("reverse"))}
    if s.get("resample"):
        req["resample"] = s["resample"]
    req = {k: v for k, v in req.items() if v is not None}
    r = ink_client.predict(gpu, req)
    ok = bool(r) and os.path.exists(r.get("out") or "")
    if ok:
        os.replace(r["out"], out_png)
        js = os.path.splitext(r["out"])[0] + ".json"
        if os.path.exists(js):
            os.replace(js, os.path.splitext(out_png)[0] + ".json")
    import shutil; shutil.rmtree(out_dir, ignore_errors=True)
    return ok


def predict(layers_dir: str, mask_png: str, out_png: str, gpu: str, fleet=None, ckpt: str | None = None,
            log: str | None = None, timeout: int = 3600, **spec) -> bool:
    """Run one checkpoint over one rendered layer stack; True when out_png was written.

    `spec` overrides SPEC (layer_start, layers, tile, stride) for a checkpoint trained
    under a different window; anything not overridden comes from SPEC, so a caller that
    passes nothing gets the released contract.
    """
    from ... import config
    from ..finish import _run
    if fleet is None:
        fleet = config.load()
    ok, why = available(fleet)
    if not ok:
        raise RuntimeError(why)
    cks = checkpoints(fleet)
    path = ckpt if (ckpt and os.path.exists(ckpt)) else cks.get(ckpt or "", next(iter(cks.values())))
    s = dict(SPEC, **{k: v for k, v in spec.items() if k in SPEC})
    cmd = [python(fleet), "-m", "vesuvius_pipeline.stages.ink_models.i3d",
           "--layers", layers_dir, "--out", out_png, "--ckpt", path, "--code", code_dir(fleet),
           "--layer-start", str(s["layer_start"]), "--in-chans", str(s["layers"]),
           "--tile", str(s["tile"]), "--stride", str(s["stride"]),
           "--bs", os.environ.get("VPIPE_I3D_BS", "0")]   # 0 = size it from the card
    if spec.get("reverse"):
        cmd.append("--reverse")
    if mask_png and os.path.exists(mask_png):
        cmd += ["--mask", mask_png]
    # Prefer this card's resident server: it already holds torch, the CUDA context
    # and this checkpoint, so an arm pays only the stack read (and with the .npy
    # handoff, barely that). Falls back to spawning on any failure, so a card
    # without a server -- a training card, by design -- still runs the family.
    served = _serve(fleet, gpu, "i3d", layers_dir, mask_png, out_png, path, code_dir(fleet), s, spec)
    if served:
        return True
    if not _headroom(fleet, gpu):
        print(f"i3d: card {gpu} has no headroom for a local run", file=sys.stderr)
        return False
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
               PYTHONPATH=str(_root(fleet) / "src") + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""))
    rc = _run(cmd, log or os.path.splitext(out_png)[0] + ".log", timeout, env=env)
    return rc == 0 and os.path.exists(out_png) and os.path.getsize(out_png) > 0


# ---- the worker (runs under the Grand Prize venv's torch) --------------------------------
def _gkern(n: int, nsig: float = 1.0) -> "np.ndarray":
    """gkern(64, 1) normalised by its max -- inference_resnet3d.py:32 and :630-631.
    Reproduced with erf so the worker does not need scipy."""
    from math import erf, sqrt
    x = np.linspace(-nsig, nsig, n + 1)
    cdf = np.array([0.5 * (1 + erf(v / sqrt(2))) for v in x])
    k1 = np.diff(cdf)
    k2 = np.outer(k1, k1)
    k2 = k2 / k2.sum()
    return k2 / k2.max()


_BUILD_CACHE: dict = {}


def _build(code: str, ckpt_path: str, device):
    """Memoised on (code, checkpoint, device).

    A process that runs ONE segment pays this once either way; ink_server runs
    many in one process, and rebuilding the network and re-reading the state dict
    per segment is pure overhead. The cache key includes the checkpoint path, so
    two checkpoints of the same architecture do not share an entry -- serving the
    wrong weights is the one failure this must not be able to cause."""
    key = (code, os.path.abspath(ckpt_path), str(device))
    if key in _BUILD_CACHE:
        return _BUILD_CACHE[key]
    import time as _t
    _t0 = _t.perf_counter()
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    # i3dallnl.py does `from non_local_helper import NLBlockND`, a sibling-relative import
    # that only resolves if models/ is itself on sys.path -- so both go on.
    for d in (code, os.path.join(code, "models")):
        if d not in sys.path:
            sys.path.insert(0, d)
    from models.i3dallnl import InceptionI3d          # noqa: E402

    class Decoder(nn.Module):
        """inference_resnet3d.py Decoder, verbatim in structure so the keys match."""
        def __init__(self, dims, upscale):
            super().__init__()
            self.convs = nn.ModuleList([
                nn.Sequential(nn.Conv2d(dims[i] + dims[i - 1], dims[i - 1], 3, 1, 1, bias=False),
                              nn.BatchNorm2d(dims[i - 1]), nn.ReLU(inplace=True))
                for i in range(1, len(dims))])
            self.logit = nn.Conv2d(dims[0], 1, 1, 1, 0)
            self.up = nn.Upsample(scale_factor=upscale, mode="bilinear")

        def forward(self, fm):
            for i in range(len(fm) - 1, 0, -1):
                f = torch.cat([fm[i - 1], F.interpolate(fm[i], scale_factor=2, mode="bilinear")], dim=1)
                fm[i - 1] = self.convs[i - 1](f)
            return self.up(self.logit(fm[0]))

    class Net(nn.Module):
        """The Lightning module's forward without Lightning: backbone -> max over depth -> decoder."""
        def __init__(self):
            super().__init__()
            self.backbone = InceptionI3d(in_channels=1, num_classes=512, non_local=True)
            # The channel widths come from a probe forward. Run it on the TARGET DEVICE:
            # on CPU it is a 3-D CNN over 1x1x20x256x256 and, on a loaded box with
            # OMP_NUM_THREADS bounded, it takes longer than the inference it is setting up
            # (measured: over an hour at load 139, printing nothing -- it looks like a hang).
            self.backbone.to(device)
            with torch.no_grad():
                dims = [x.size(1) for x in self.backbone(torch.rand(1, 1, 20, 256, 256, device=device))]
            self.decoder = Decoder(dims, upscale=1)

        def forward(self, x):
            if x.ndim == 4:
                x = x[:, None]
            fm = [torch.max(f, dim=2)[0] for f in self.backbone(x)]
            return self.decoder(fm)

    # STAGE TIMING (FINDINGS 2026-09-09): job 181 pinned one CPU core at ~93% with 0% GPU
    # for 20+ minutes with no child process -- exactly the resident ink_server running
    # this build/inference in-process. `net = Net()` below runs the probe forward
    # described above; instrument it separately from torch.load/load_state_dict/device
    # transfer so the next stall's log names the actual stage instead of "somewhere in
    # _build". Printed to stderr (goes to the job's own log) whichever device is chosen,
    # and DEVICE is logged explicitly -- if `device` is ever "cpu" while a GPU was
    # requested, that silent fallback is precisely the failure this family's own comment
    # already warned about.
    print(f"i3d._build: device={device} cuda_available={torch.cuda.is_available()} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} "
          f"(import+setup {_t.perf_counter() - _t0:.2f}s)", file=sys.stderr, flush=True)
    _t1 = _t.perf_counter()
    net = Net()                    # includes the probe forward (see Net.__init__ above)
    _t2 = _t.perf_counter()
    print(f"i3d._build: Net() [backbone ctor + probe forward on {device}] {_t2 - _t1:.2f}s",
          file=sys.stderr, flush=True)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _t3 = _t.perf_counter()
    print(f"i3d._build: torch.load({os.path.basename(ckpt_path)}) {_t3 - _t2:.2f}s",
          file=sys.stderr, flush=True)
    sd = ck.get("state_dict", ck)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    # An i3d checkpoint loaded into this net must cover every weight. A silent
    # partial load is exactly the failure this family exists to prevent, so refuse it.
    hard = [k for k in missing if not k.endswith("num_batches_tracked")]
    if hard:
        raise SystemExit(f"i3d: checkpoint does not fit the net: {len(hard)} missing, e.g. {hard[:3]}")
    net.to(device).eval()
    _t4 = _t.perf_counter()
    print(f"i3d._build: load_state_dict + .to(device).eval() {_t4 - _t3:.2f}s -- "
          f"TOTAL _build {_t4 - _t0:.2f}s", file=sys.stderr, flush=True)
    _BUILD_CACHE[key] = (net, len(sd), len(unexpected))
    return _BUILD_CACHE[key]


def _auto_batch(chans: int, tile: int, device, cap: int = 256, floor: int = 1, frac: float = 0.30) -> int:
    """A tile batch that fits the FREE VRAM right now -- see resnet3d_1667._auto_batch."""
    import torch
    try:
        free, _ = torch.cuda.mem_get_info(device) if str(device).startswith("cuda") else (0, 0)
    except Exception:                                   # noqa: BLE001 - CPU, or no CUDA
        free = 0
    if not free:
        return 32
    per = chans * tile * tile * 4 * 12                  # input + activations, conservative
    return int(max(floor, min(cap, (free * frac) // max(per, 1))))


def _main(argv=None) -> int:
    import time as _t
    _t_main0 = _t.perf_counter()
    import cv2
    import torch
    import torch.nn.functional as F
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", required=True)
    ap.add_argument("--mask")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--layer-start", type=int, default=LAYER_START)
    ap.add_argument("--in-chans", type=int, default=IN_CHANS)
    ap.add_argument("--tile", type=int, default=TILE)
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--bs", type=int, default=0,
                    help="tiles per forward; 0 (the default) sizes it from the free VRAM at "
                         "call time and halves on OOM.")
    ap.add_argument("--reverse", action="store_true",
                    help="read the layer stack outermost-first. Which end of a render is "
                         "the recto is a property of the render, not of the model, and it is "
                         "worth 0.52 -> 0.92 AUC on Scroll 1 (project memory: 'the published "
                         "face is our reversed order'). Always score both faces.")
    a = ap.parse_args(argv)

    files = sorted(glob.glob(os.path.join(a.layers, "*.tif")))
    if len(files) < a.layer_start + a.in_chans:
        print(f"i3d: {len(files)} layers, need {a.layer_start + a.in_chans}", file=sys.stderr)
        return 2
    if a.reverse:
        files = files[::-1]
    # 🔴 FINDINGS 2026-09-09 (job 181): this loop was the ~93%-of-one-core, 0%-GPU stall
    # a caller sees as a hang. `cv2.imread` of a ~12.7 MB uncompressed layer TIFF is a
    # single-threaded C decode+copy, and under fleet disk contention on the shared RAID
    # it climbed from ~1s to ~7.6s per file over just 30 reads (measured directly against
    # this job's own layer stack: 165.9s serial). It happens entirely BEFORE the model or
    # GPU are touched, so nvidia-smi reads 0% the whole time -- indistinguishable from a
    # hang unless you know to look here rather than at the network.
    # cv2.imread releases the GIL for its C decode, so a thread pool is real parallelism
    # (verified: same 30 files, same host, 19.6s threaded -- 8.5x). Order is preserved by
    # writing each result into its own slot rather than relying on completion order.
    from concurrent.futures import ThreadPoolExecutor
    want = files[a.layer_start:a.layer_start + a.in_chans]
    imgs: list = [None] * len(want)
    bad: list = []

    def _read_layer(i_f):
        i, f = i_f
        im = cv2.imread(f, 0)
        if im is None:
            bad.append(f)
            return
        imgs[i] = np.clip(im, 0, CLIP_MAX)

    with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 1, len(want))) as ex:
        list(ex.map(_read_layer, enumerate(want)))
    if bad:
        print(f"i3d: unreadable layer {bad[0]}", file=sys.stderr)
        return 2
    h0, w0 = imgs[0].shape
    pad0, pad1 = (256 - h0 % 256) % 256, (256 - w0 % 256) % 256
    vol = np.stack([np.pad(i, [(0, pad0), (0, pad1)]) for i in imgs], axis=2)   # H,W,C
    H, W = vol.shape[:2]
    if a.mask:
        m = cv2.imread(a.mask, 0)
        m = np.pad(m, [(0, pad0), (0, pad1)])
    else:
        m = np.full((H, W), 255, np.uint8)
    _t_read = _t.perf_counter()
    print(f"i3d: read+pad {a.in_chans} layers ({h0}x{w0}) {_t_read - _t_main0:.2f}s",
          file=sys.stderr, flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, n_sd, n_unexp = _build(a.code, a.ckpt, dev)
    _t_build = _t.perf_counter()

    # Device-side stack, and the windows sliced PER BATCH rather than unfolded in one go:
    # at 3806x3334 with tile 64 / stride 21 one unfold is ~30,000 patches of 30x64x64
    # float32 = 13.84 GiB materialised before a single forward, which is the allocation
    # that OOM'd job 129 on a card with 1.36 GiB free. Same fault and same fix as
    # resnet3d_1667 and hires_ink: slice per batch, size the batch from the FREE VRAM at
    # call time, halve and retry on OOM instead of failing the job.
    from .tiling import tile_origins
    from . import _progress
    dvol = torch.from_numpy(np.ascontiguousarray(vol.transpose(2, 0, 1))).to(dev).float().div_(255.0)
    origins = tile_origins(H, W, a.tile, a.stride)
    _t_h2d = _t.perf_counter()
    keep = [i for i, (y1, x1) in enumerate(origins) if m[y1:y1 + a.tile, x1:x1 + a.tile].all()]
    _t_keep = _t.perf_counter()
    print(f"i3d: h2d+origins ({len(origins)} tiles) {_t_h2d - _t_build:.2f}s, "
          f"keep-mask filter ({len(keep)} kept) {_t_keep - _t_h2d:.2f}s", file=sys.stderr, flush=True)
    kern = torch.from_numpy(_gkern(a.tile, 1.0).astype(np.float32)).to(dev)
    pred = torch.zeros((H, W), device=dev); cnt = torch.zeros((H, W), device=dev)
    pw = _progress.writer(len(keep), (h0, w0), model="i3d",
                          origins=[origins[k] for k in keep], tile=a.tile,
                          stride=a.stride)
    bs = a.bs if a.bs > 0 else _auto_batch(a.in_chans, a.tile, dev)
    print(f"i3d: batch size {bs}, {len(keep)} tiles -> {-(-len(keep) // max(bs, 1))} batches",
          file=sys.stderr, flush=True)
    i, oom = 0, 0
    _t_loop0 = _t.perf_counter()
    _t_slice_total = _t_fwd_total = _t_blend_total = 0.0
    n_batches = 0
    while i < len(keep):
        idx = keep[i:i + bs]
        try:
            _bt0 = _t.perf_counter()
            x = torch.stack([dvol[:, y1:y1 + a.tile, x1:x1 + a.tile]
                             for y1, x1 in (origins[k] for k in idx)]).unsqueeze(1)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            _bt1 = _t.perf_counter()
            with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                y = net(x)
            y = torch.sigmoid(y.float())
            y = F.interpolate(y, scale_factor=UPSCALE, mode="bilinear").squeeze(1)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            _bt2 = _t.perf_counter()
            for j, k in enumerate(idx):
                y1, x1 = origins[k]
                pred[y1:y1 + a.tile, x1:x1 + a.tile] += y[j] * kern
                # kern, NOT 1.0. The released script weights the numerator by the Gaussian
                # and divides by a flat COUNT, which is not a weighted average at all: it
                # attenuates every tile towards its edges and leaves a dark lattice over
                # the map. Dividing by the same weights is what makes it a mean.
                cnt[y1:y1 + a.tile, x1:x1 + a.tile] += kern
            del x, y
            if dev.type == "cuda":
                torch.cuda.synchronize()
            _bt3 = _t.perf_counter()
            _t_slice_total += _bt1 - _bt0
            _t_fwd_total += _bt2 - _bt1
            _t_blend_total += _bt3 - _bt2
            n_batches += 1
            i += len(idx)
        except torch.cuda.OutOfMemoryError:
            if bs <= 1:
                raise
            bs = max(1, bs // 2); oom += 1
            torch.cuda.empty_cache()
            continue
        if pw is not None and pw.due():
            _progress.publish(pw, min(i, len(keep)), pred, cnt, h0, w0,
                              [origins[k] for k in idx], a.tile, i // max(bs, 1),
                              running=(i - len(idx), i))
    _t_loop1 = _t.perf_counter()
    print(f"i3d: batch loop {_t_loop1 - _t_loop0:.2f}s over {n_batches} batches -- "
          f"per-batch mean: slice/stack {1000 * _t_slice_total / max(n_batches, 1):.1f}ms, "
          f"forward+sigmoid+upsample {1000 * _t_fwd_total / max(n_batches, 1):.1f}ms, "
          f"blend-accumulate {1000 * _t_blend_total / max(n_batches, 1):.1f}ms "
          f"(totals: slice {_t_slice_total:.2f}s, forward {_t_fwd_total:.2f}s, "
          f"blend {_t_blend_total:.2f}s)", file=sys.stderr, flush=True)
    if pw is not None:
        _progress.publish(pw, len(keep), pred, cnt, h0, w0, origins, a.tile, -1, done=True)
    pred = pred.cpu().numpy().astype(np.float64); cnt = cnt.cpu().numpy().astype(np.float64)
    if cnt.max() <= 0:
        print("i3d: no tile fell entirely inside the mask", file=sys.stderr)
        return 3
    out = np.divide(pred, cnt, out=np.zeros_like(pred), where=cnt > 1e-6)[:h0, :w0]
    out = np.clip(np.nan_to_num(out), 0, 1)
    cv2.imwrite(a.out, (out * 255).astype(np.uint8))
    json.dump({"ckpt": a.ckpt, "reverse": bool(a.reverse), "sha_note": "see docs/wiki/ink_models.md", "spec": SPEC,
               "layer_start": a.layer_start, "in_chans": a.in_chans, "tiles": int((cnt > 1e-6).sum()), "batch": int(bs), "oom_retries": int(oom),
               "state_dict_tensors": n_sd, "unexpected_keys": n_unexp,
               "shape": [int(h0), int(w0)], "mean": float(out.mean()), "max": float(out.max())},
              open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    print(f"i3d: TOTAL {_t.perf_counter() - _t_main0:.2f}s "
          f"(read {_t_read - _t_main0:.2f}s, build {_t_build - _t_read:.2f}s, "
          f"tiling {_t_keep - _t_build:.2f}s, loop {_t_loop1 - _t_loop0:.2f}s, "
          f"write {_t.perf_counter() - _t_loop1:.2f}s)", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
