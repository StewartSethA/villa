"""resnet3d_1667: the released PHerc.1667 (Scroll 4) ink models, six of them.

`scrollprize/PHerc.1667-iteration-{0..5}` on Hugging Face, MIT, released 2026-06-30
alongside the end-to-end reading of Scroll 4. These are the only PUBLIC ink
checkpoints trained on Scroll 4 that we have found:

  iteration-0  cross-segment baseline -- NO l_2 labels; trained on 500p2a, 658,
               two segments (ids omitted)  (20,075 tiles).
               This is the honest one to score a Scroll 4 segment with, because the
               other five saw l_2.
  iteration-1..5  label ablations on segment l_2, increasing coverage;
               iteration-5 is the densest (33,061 tiles) and defines the 12,396-step
               budget all six share.

Architecture: ResNet3D-50 (Hara et al. 2018, Kinetics-700 init with conv1 summed to
one grey channel) over (B, 1, 62, 256, 256); each of the four stages collapsed along z
with torch.max; a 2-D U-Net decoder with skips; a 1x1 head giving one logit channel at
QUARTER resolution (B, 1, 64, 64), bilinearly upsampled x4 to the tile.

So it is the same shape of model as the `i3d` family -- 3-D backbone, max over depth,
2-D decoder -- with a different backbone, 62 layers instead of 30 and a 256 tile
instead of 64. That is exactly why they are separate families rather than one with a
flag: the layer window and tile size ARE the model, and a checkpoint run under the
other family's numbers still emits a plausible grey image.

Input contract, from the model card (not inferred):
  * 62 z-layers, 256 x 256 windows, stride 128 (2x oversample; 64 for 8x);
  * raw uint8 layers CLIPPED to [0, 200] then Normalize(mean=0, std=1), i.e. a plain
    divide by 255 -- the card's "already z-score normalised" is contradicted by its own
    "clipped raw uint8 layers to [0,200] then applied Normalize(mean=0, std=1) which
    keeps the magnitude small". The second is what the training pipeline did, and it is
    what this wrapper does. NOT the Grand Prize mean-150/std-25 contract.
  * a tile is used only where the fragment mask is entirely set;
  * sigmoid, then accumulate over overlaps with a GAUSSIAN window and divide by the sum
    of those same weights. The card's own snippet uses a flat count, which leaves a hard
    `stride` grid wherever coverage changes (4 tiles -> 2 -> 1); the weighted sum is the
    same estimator without the steps.

Which 62 of a 65-layer stack. The card does not say, so this wrapper CENTRES the window
(`layer_start = (n - 62) // 2`) and records `layer_start` in the sidecar JSON, and the
`--reverse` face control is available as for i3d. An undocumented window is a real
uncertainty; recording it is what lets a later run move it and see the difference.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

IN_CHANS = 62
TILE = 256
STRIDE = 128
CLIP_MAX = 200
UPSCALE = 4
REPOS = tuple(f"PHerc.1667-iteration-{i}" for i in range(6))

SPEC = {
    "layers": IN_CHANS,
    "layer_start": None,          # centred on the stack unless overridden
    "tile": TILE,
    "stride": STRIDE,
    # The card states NO pitch -- not the README, not config.json -- and 7.91 was asserted
    # here without evidence, which was wrong: Scroll 4 publishes 3.24 um and 7.91 um
    # volumes and a separate 2 um scan is in circulation, so the training frame could have
    # been any of three. It is MEASURED instead (FINDINGS 61.6): one held-out segment,
    # the same scored pixels, the render resampled to each candidate; the winner is here.
    "um_per_px": 7.91,
    "um_per_px_provenance": "DECLARED default, UNSUPPORTED: the resampling sweep that claimed to measure 7.91 (FINDINGS 61.6) was retracted in 61.9 (scored against an invalid Scroll 4 label); training pitch unknown",
    "normalisation": "clip0_200_div255",
    "arch": "resnet3d50_unet2d",
    "trained_on": "PHerc1667 (Scroll 4)",
    "license": "mit",
}



def training_frame():
    """The Frame this checkpoint was trained in. A stack handed to `predict` must match
    it (`Frame.assert_compatible`); this family does NOT resample, so a mismatch is a
    silent wrong answer rather than an error."""
    from ...frame import MODELS, ModelSpec
    return MODELS.get("resnet3d_1667", ModelSpec("resnet3d_1667", SPEC["um_per_px"], SPEC["layers"])).frame(
        source_id="model:resnet3d_1667")


def check_frame(frame, tol: float = 0.05) -> None:
    """Refuse a render whose pitch is not this model's. Raises FrameError."""
    training_frame().assert_compatible(frame, tol=tol, what="resnet3d_1667 input")


def enabled() -> bool:
    return os.environ.get("VPIPE_RESNET3D_1667", "0") == "1"


def _root(fleet):
    from ... import config
    return fleet.root or config.repo_root()


def model_dirs(fleet) -> dict[str, str]:
    """iteration tag -> local snapshot dir, resolved from the fleet's `[models]` table.

    The six repos are declared as `resnet3d_1667_it<N>` with repo-relative `files`, so
    this names no machine: `vpipe setup models` puts the bytes where the table says and
    the wrapper looks exactly there. A snapshot counts only when BOTH the weights and the
    config are present -- half a download is not a model.
    """
    from ... import config
    root = fleet.root or config.repo_root()
    env = os.environ.get("VPIPE_INK_MODELS")
    out: dict[str, str] = {}
    if env:
        # An explicit VPIPE_INK_MODELS is an OVERRIDE, not another candidate: pointing the
        # family at a directory and still silently loading a declared checkpoint elsewhere
        # is how a run reports one model and executes another.
        for name in REPOS:
            d = os.path.join(env, name)
            if os.path.isfile(os.path.join(d, "model.safetensors")) and os.path.isfile(os.path.join(d, "config.json")):
                out[name.split("iteration-")[1]] = d
        return out
    for tag, m in (getattr(fleet, "models", None) or {}).items():
        if not tag.startswith("resnet3d_1667_it"):
            continue
        files = [str(root / f) for f in m.get("files", [])]
        w = next((f for f in files if f.endswith("model.safetensors")), None)
        if w and os.path.isfile(w) and os.path.isfile(os.path.join(os.path.dirname(w), "config.json")):
            out[tag[len("resnet3d_1667_it"):]] = os.path.dirname(w)
    return out


def python(fleet) -> str | None:
    for p in (os.environ.get("VPIPE_I3D_PY"),
              str(_root(fleet) / "Vesuvius-Grandprize-Winner" / ".venv" / "bin" / "python")):
        if p and os.path.exists(p):
            return p
    return None


def available(fleet) -> tuple[bool, str]:
    if not model_dirs(fleet):
        return False, "resnet3d_1667: no snapshot present; run `vpipe setup models` for resnet3d_1667_it0..5"
    py = python(fleet)
    if py is None:
        return False, "resnet3d_1667: no torch interpreter"
    return True, "ok"


def predict(layers_dir: str, mask_png: str, out_png: str, gpu: str, fleet=None, ckpt: str = "5",
            log: str | None = None, timeout: int = 3600, render_frame=None, **spec) -> bool:
    """`ckpt` is an iteration tag ('0'..'5') or a path to a snapshot directory.

    `render_frame` is the Frame of the layer stack being read. Given one, the resample
    onto the model's trained pitch is derived from it; without one the stack is assumed to
    be already in the model's frame, which is what the Grand Prize render is (7.91 um/px).
    """
    from ... import config
    from ..finish import _run
    if fleet is None:
        fleet = config.load()
    ok, why = available(fleet)
    if not ok:
        raise RuntimeError(why)
    dirs = model_dirs(fleet)
    d = ckpt if os.path.isdir(str(ckpt)) else dirs.get(str(ckpt))
    if d is None:
        raise RuntimeError(f"resnet3d_1667: iteration {ckpt!r} not present; have {sorted(dirs)}")
    s = dict(SPEC, **{k: v for k, v in spec.items() if k in SPEC})
    cmd = [python(fleet), "-m", "vesuvius_pipeline.stages.ink_models.resnet3d_1667",
           "--layers", layers_dir, "--out", out_png, "--model", d,
           "--in-chans", str(s["layers"]), "--tile", str(s["tile"]), "--stride", str(s["stride"]),
           "--bs", os.environ.get("VPIPE_1667_BS", "0")]   # 0 = size it from the card
    if s.get("layer_start") is not None:
        cmd += ["--layer-start", str(s["layer_start"])]
    # The resample is DERIVED, never hand-set: build the render's Frame and ask it what it
    # takes to present this model its own pixels. What crosses to the worker is the
    # dimensionless factor -- a raw um/px handed across a process boundary is a ruler
    # nobody downstream can check, which is what test_no_stage_publishes_a_pitch_without_a_frame
    # exists to stop.
    plan = None
    if render_frame is not None:
        from ...frame import Frame                       # noqa: F401  (the ruler this factor comes from)
        plan = render_frame.to_model("resnet3d_1667")
        if not plan.in_tolerance:
            cmd += ["--resample", f"{plan.resample_factor:.6f}"]
    if spec.get("reverse"):
        cmd.append("--reverse")
    if mask_png and os.path.exists(mask_png):
        cmd += ["--mask", mask_png]
    # This card's resident server first: it holds torch, the CUDA context and this
    # iteration's weights already. The resample factor travels with the request --
    # serving without it would score at the wrong pitch, silently.
    from .i3d import _serve
    extra = dict(s)
    if plan is not None and not plan.in_tolerance:
        extra["resample"] = f"{plan.resample_factor:.6f}"
    if _serve(fleet, gpu, "resnet3d_1667", layers_dir, mask_png, out_png, d, "", extra, spec):
        return True
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
               PYTHONPATH=str(_root(fleet) / "src") + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""))
    rc = _run(cmd, log or os.path.splitext(out_png)[0] + ".log", timeout, env=env)
    return rc == 0 and os.path.exists(out_png) and os.path.getsize(out_png) > 0


_MODEL_CACHE: dict = {}


def _auto_batch(chans: int, tile: int, device, cap: int = 32, floor: int = 1, frac: float = 0.30) -> int:
    """A tile batch that fits the FREE VRAM right now.

    a shared GPU host's cards are shared -- other tenants routinely hold 25 GB of 32 -- so a fixed
    batch is a job that fails whenever someone else is busy. One tile costs
    chans*tile*tile*4 bytes of input and the ResNet3D forward's activations are several
    times that, so a conservative multiple is used and the caller halves on OOM anyway.
    """
    import torch
    try:
        free, _ = torch.cuda.mem_get_info(device) if str(device).startswith("cuda") else (0, 0)
    except Exception:                                   # noqa: BLE001 - CPU, or no CUDA
        free = 0
    if not free:
        return 8
    per = chans * tile * tile * 4 * 12                  # input + activations, conservative
    return int(max(floor, min(cap, (free * frac) // max(per, 1))))


def _main(argv=None) -> int:
    import cv2
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", required=True)
    ap.add_argument("--mask")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--in-chans", type=int, default=IN_CHANS)
    ap.add_argument("--layer-start", type=int, default=-1)
    ap.add_argument("--tile", type=int, default=TILE)
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--bs", type=int, default=0,
                    help="tiles per forward; 0 (the default) sizes it from the free VRAM at "
                         "call time and halves on OOM. A fixed batch is a job that fails "
                         "whenever another tenant is busy on the card.")
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--resample", type=float, default=1.0,
                    help="linear factor from the render's frame onto the model's, as computed "
                         "by Frame.to_model().resample_factor. A factor, not a pitch: the "
                         "worker is handed a ratio it can apply, not a ruler it cannot check.")
    a = ap.parse_args(argv)
    # f > 1: the source is COARSER than the model's frame, so a feature must be magnified
    # to reach the net at the size it was trained on. Rather than magnifying the whole
    # stack (2048^2 x 62 at f=4 is 16 GB, and the unfold of it 60 GB), tiles are cut at
    # round(tile / f) SOURCE pixels and resized to `tile`: the same arithmetic at the
    # source's memory cost, and every arm scores the identical pixels.
    f = 1.0 / a.resample if a.resample > 0 else 1.0

    files = sorted(glob.glob(os.path.join(a.layers, "*.tif")))
    if a.reverse:
        files = files[::-1]
    if len(files) < a.in_chans:
        print(f"resnet3d_1667: {len(files)} layers, need {a.in_chans}", file=sys.stderr)
        return 2
    start = a.layer_start if a.layer_start >= 0 else (len(files) - a.in_chans) // 2
    imgs = [np.clip(cv2.imread(f, 0), 0, CLIP_MAX) for f in files[start:start + a.in_chans]]
    if any(i is None for i in imgs):
        return 2
    h0, w0 = imgs[0].shape
    pad0, pad1 = (a.tile - h0 % a.tile) % a.tile, (a.tile - w0 % a.tile) % a.tile
    vol = np.stack([np.pad(i, [(0, pad0), (0, pad1)]) for i in imgs], axis=2)
    H, W = vol.shape[:2]
    m = (np.pad(cv2.imread(a.mask, 0), [(0, pad0), (0, pad1)]) if a.mask
         else np.full((H, W), 255, np.uint8))

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Memoised on (repo, device) so a resident server pays the download/parse and
    # the weight load once rather than per segment. Keyed by the repo id, so two
    # iterations of the 1667 model never share an entry.
    _mk = (a.model, str(dev))
    model = _MODEL_CACHE.get(_mk)
    if model is None:
        model = AutoModel.from_pretrained(a.model, trust_remote_code=True).eval().to(dev)
        _MODEL_CACHE[_mk] = model

    # Stack, clip and scale ON THE DEVICE, and cut the windows with one unfold instead of
    # one python slice each: 177 s -> 1.5 s per face at 1024^2 (FINDINGS 64).
    from .tiling import tile_origins
    from . import _progress
    dvol = torch.from_numpy(np.ascontiguousarray(vol.transpose(2, 0, 1))).to(dev).float().div_(255.0)
    st = max(16, int(round(a.tile / f)))          # source pixels one net tile covers
    ss = max(1, int(round(a.stride / f)))
    origins = tile_origins(H, W, st, ss)
    keep = [i for i, (y1, x1) in enumerate(origins) if m[y1:y1 + st, x1:x1 + st].all()]
    pred = torch.zeros((H, W), device=dev); cnt = torch.zeros((H, W), device=dev)
    # A Gaussian window, not a flat count. With flat weights a pixel covered by four
    # tiles and its neighbour covered by two are each divided by their own count, so any
    # per-tile offset in the model's output survives as a STEP at every coverage boundary
    # -- a hard `stride` grid over the whole map, which is what the prospect tile showed.
    # Weighting by a window that decays to ~0 at the tile edge makes the contribution of a
    # tile vanish where its neighbour takes over, so the sum is smooth; dividing by the
    # sum of the SAME weights keeps it a partition of unity, hence a probability.
    _w1 = torch.exp(-0.5 * (torch.linspace(-2.0, 2.0, st, device=dev) ** 2))
    win = (_w1[:, None] * _w1[None, :])
    win = win / win.max()
    pw = _progress.writer(len(keep), (h0, w0), model="resnet3d_1667",
                          origins=[origins[k] for k in keep], tile=st, stride=ss)
    bs = a.bs if a.bs > 0 else _auto_batch(a.in_chans, st, dev)
    i, oom = 0, 0
    while i < len(keep):
        idx = keep[i:i + bs]
        try:
            # NOT one unfold of the whole stack: at 3806x3334 with tile 256 / stride 128
            # that is ~730 patches of 62x256x256 float32 = 11.85 GiB materialised before a
            # single forward, which is exactly the allocation that OOM'd job 125 on a card
            # with 3.76 GiB free. The stack is sliced per batch instead (a view per tile),
            # the batch is sized from the FREE VRAM at call time, and an OOM halves it and
            # retries rather than failing the job. Same fault, same fix as hires_ink.
            x = torch.stack([dvol[:, y1:y1 + st, x1:x1 + st] for y1, x1 in (origins[k] for k in idx)])
            if st != a.tile:
                x = F.interpolate(x, size=(a.tile, a.tile), mode="area" if f < 1 else "bilinear",
                                  **({} if f < 1 else {"align_corners": False}))
            x = x.unsqueeze(1)
            with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                out = model(x)
            y = torch.sigmoid(getattr(out, "logits", out).float())
            y = F.interpolate(y, scale_factor=UPSCALE, mode="bilinear").squeeze(1)
            if st != a.tile:                          # back to source pixels
                y = F.interpolate(y.unsqueeze(1), size=(st, st), mode="bilinear",
                                  align_corners=False).squeeze(1)
            for j, k in enumerate(idx):
                y1, x1 = origins[k]
                pred[y1:y1 + st, x1:x1 + st] += y[j] * win
                cnt[y1:y1 + st, x1:x1 + st] += win
            del x, out, y
            i += len(idx)
        except torch.cuda.OutOfMemoryError:
            if bs <= 1:
                raise
            bs = max(1, bs // 2); oom += 1
            torch.cuda.empty_cache()
            continue
        if pw is not None and pw.due():
            _progress.publish(pw, min(i, len(keep)), pred, cnt, h0, w0,
                              [origins[k] for k in idx], st, i // max(bs, 1),
                              running=(max(0, i - len(idx)), i))
    if pw is not None:
        _progress.publish(pw, len(keep), pred, cnt, h0, w0, origins, st, -1, done=True)
    pred = pred.cpu().numpy().astype(np.float64); cnt = cnt.cpu().numpy().astype(np.float64)
    if cnt.max() <= 0:
        print("resnet3d_1667: no tile fell entirely inside the mask", file=sys.stderr)
        return 3
    out = np.clip(np.nan_to_num(np.divide(pred, cnt, out=np.zeros_like(pred), where=cnt > 1e-6)[:h0, :w0]), 0, 1)
    cv2.imwrite(a.out, (out * 255).astype(np.uint8))
    json.dump({"model": a.model, "spec": SPEC, "layer_start": int(start), "in_chans": a.in_chans,
               "reverse": bool(a.reverse), "tiles": int((cnt > 1e-6).sum()),
               "resample": a.resample, "resample_f": round(f, 4),
               "batch": int(bs), "oom_retries": int(oom),
               "src_tile": int(st), "src_stride": int(ss),
               "shape": [int(h0), int(w0)], "mean": float(out.mean()), "max": float(out.max())},
              open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
