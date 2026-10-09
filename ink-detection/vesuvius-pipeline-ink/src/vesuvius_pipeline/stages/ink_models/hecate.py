"""hecate: the Vesuvius Challenge staff model `scrollprize/hecate` (MIT), run on our renders.

Upstream (huggingface.co/scrollprize/hecate @ 9cb86e50, 2026-09-15): a ResNet-152 3-D encoder + 3-D U-Net
decoder (from `ink_canonical_2um`) that predicts ink through the depth of a surface volume and collapses
it to a 2-D map by learned depth attention. Two checkpoints; this module runs them through the
UNMODIFIED upstream `hecate.py` that ships beside the weights (imported, not copied):

  arm "9um"  hecate_9.6um.pth  patch 16 x 64 x 64 (Z,Y,X) at 9.6 um in ALL THREE axes, uint8 / 255, no clip
  arm "2um"  hecate_2.4um.pth  patch 64 x 256 x 256 at 2.4 um, uint8 / 200 -- NOT registered as a finish
             family: no render tier of ours is at 2.4 um except the PHerc1667 public volume.

THE INPUT CONTRACT, and why this module resamples DEPTH. Every render tier of ours steps its planes ONE
CT VOXEL apart along the normal whatever its in-plane --scale (colmean.plane_pitch_um; 7.91 um on the
S1/S5 controls, 9.362 / 8.64 um on other scans). Hecate's card: "Surface-conditioned renders supplied to
this model must be sampled at 9.6 um in both the surface plane and the depth direction." So the finish
stage renders this family at 9.6 um in plane (registry `um_per_px`), and here the native planes are
resampled to 17 planes 9.6 um apart centred on the stack middle (each a 9.6 um slab mean of the
linearly-interpolated native planes -- the same construction as the 4-plane z-mean that makes the public
PHerc1667 2.399 um volume 9.596 um isotropic). Hecate then evaluates its central 16 planes.

FACE. The finish stage hands this module the stack already in the order it wants (`<name>_reversed` is
the dispatcher's depth-reversed view), so predict() never reverses: plane order as given = upstream
`hecate.py` without --reverse. Which face reads ink on which scroll is MEASURED, not assumed
(docs/experiments/hecate_transfer/STATE.md).

What it was trained on (card + the Scroll Prize ink bucket, read 2026-10-05): PHercParis4 (S1) and
PHerc1667 (S4, the HF letter boxes we score with, wNNN included) labels are IN its training data; PHerc0172
(S5) has no labels there but its CT/pseudo-label exposure is not stated -- UNKNOWN, never "unseen".

Window: the network sees one 64 x 64 px patch (614 um at 9.6 um/px); upstream blends patches at a 32 px
stride with a floored Hann window, so one output pixel depends on input within a 127 px (1.22 mm) span.
Cost: a ResNet-152-depth 3-D net at full resolution -- 0.03-0.04 Mpx/s per face on an RTX 4060 Ti in bf16
(~0.5 MVox/s); budget minutes per Mpx.
"""
from __future__ import annotations

import json
import os

import numpy as np

from ... import alerts as _alerts

MEMSTACK_READY = True     # layers read through memstack-aware readers (grandprize_dense._layer_array)

CKPTS = {"9um": "hecate_9.6um.pth", "2um": "hecate_2.4um.pth"}
DEFAULT_ARM = "9um"
SAMPLING_UM = {"9um": 9.6, "2um": 2.4}
OUT_PLANES = {"9um": 17, "2um": 65}      # resampled planes handed to hecate (it reads the central 16 / 64)
SUBSAMPLES = 8                           # sub-samples per output slab when resampling depth
# INTENSITY FRAME (measured 2026-10-05, docs/experiments/hecate_transfer, FINDINGS "Hecate ... frame"): upstream
# divides raw uint8 by 255 with no normalisation, so it assumes the intensity calibration of the scans it trained on.
# Its S1 training input is the HF-bucket ESRF 2026 2.4 um Paris4 scan (p50 66 grey at 9.6 um); our S1 renders come
# from the 2023 7.91 um scan (p50 128): raw, Hecate reads CHANCE (0.49) on S1 although it trained on S1, and on its
# OWN S1 frame remapping just the intensity to our scan's distribution drops it 0.988 -> 0.78 (depth +-2 planes and
# blur sigma <= 1 px cost nothing). The label-free fix: a piecewise-linear QUANTILE map of the evaluated planes onto
# Hecate's training-frame quantile function (REF_V: mean of the HF S1 crops and the public S4 9.6 um crops, where it
# reads 0.99 / 0.84). Source quantiles come from the SCROLL's registered frame (hecate_frames.json, measured on our
# renders, with provenance) when there is one; otherwise from this segment itself, ANNOUNCED. `norm="none"` =
# upstream raw /255; `norm="segment"` forces the per-segment map.
# MEASURED 2026-10-05 on the full harness (S1 n = 6 parents): per-segment AFFINE match to mean 75 / std 42 -> S1 0.593
# [0.549, 0.636] AUC; quantile map onto REF_V -> 0.521 [0.500, 0.544]; raw -> 0.492. The affine match is the default;
# the quantile map is kept as norm="qmap" (it flattens the papyrus/air contrast shape Hecate uses).
REF_MEAN, REF_STD = 75.0, 42.0
REF_Q = [1, 5, 10, 25, 50, 75, 90, 95, 99]
REF_V = [25.9, 32.2, 35.5, 41.0, 63.2, 108.9, 131.5, 142.8, 162.6]
DEFAULT_NORM = "affine"
FRAMES_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hecate_frames.json")
MODEL_SUBDIR = "hecate"
MODEL_ROOTS = ("var/models",
               "/dev/shm/vpipe/ScrollPrizeTutorial/current/var/models",
               "/dev/shm/vpipe/var/models")
ENV_DIR = "VPIPE_HECATE_DIR"
UPSTREAM = {"repo": "scrollprize/hecate", "commit": "9cb86e500e944b11a06a7020403cde5dffb5bcb2",
            "md5": {"hecate_9.6um.pth": "2bb10eec6afc2a7fc1969e3e3b39d57f",
                    "hecate_2.4um.pth": "a6cd4beff0aa6da4fa99bcb87328538c",
                    "hecate.py": "6f633dd8efc3af75c0b8e9c4c85b8eec"}}

SPEC = {
    "layers": 16,
    "layer_select": "native planes resampled to 9.6 um slabs centred on the stack middle; hecate reads the central 16",
    "um_per_px": 9.6,
    "depth_um": 9.6,
    "normalisation": "label-free quantile map onto Hecate's training-frame quantiles (per-scroll registered frame, else "
                     "the segment's own, announced), then uint8 / 255, no clipping (upstream hecate.py); norm='none' = raw",
    "arch": "hecate (ResNet-152 3-D encoder + 3-D U-Net decoder + learned depth attention)",
    "patch": [16, 64, 64],
    "stride": 32,
    "precision": "bf16 on CUDA (upstream production setting), fp32 on CPU",
    "upstream": UPSTREAM,
}


def _roots() -> list[str]:
    out = [os.environ.get(ENV_DIR, "")]
    try:
        from ... import config
        out.append(str(config.repo_root() / "var" / "models" / MODEL_SUBDIR))
    except Exception as e:                               # noqa: BLE001 - no config is not fatal here
        print(f"[hecate] repo root unknown ({e}); searching the fixed roots only", flush=True)
    out += [os.path.join(r, MODEL_SUBDIR) for r in MODEL_ROOTS]
    return [r for r in out if r]


def model_dir(arm: str | None = None) -> str | None:
    """The directory holding BOTH the checkpoint and upstream hecate.py, or None."""
    name = CKPTS.get(arm or DEFAULT_ARM)
    if name is None:
        return None
    for r in _roots():
        if os.path.exists(os.path.join(r, name)) and os.path.exists(os.path.join(r, "hecate.py")):
            return os.path.abspath(r)
    return None


def checkpoint_path(arm: str | None = None) -> str | None:
    d = model_dir(arm)
    return os.path.join(d, CKPTS[arm or DEFAULT_ARM]) if d else None


def training_frame():
    from ...frame import ModelSpec
    return ModelSpec("hecate_9um", 9.6, 16).frame(source_id="model:hecate_9um")


def enabled() -> bool:
    return os.environ.get("VPIPE_HECATE", "0") == "1"


def available(fleet=None, arm: str | None = None) -> tuple[bool, str]:
    """Are the weights AND upstream hecate.py here. NOT whether it should run."""
    arm = arm or DEFAULT_ARM
    if arm not in CKPTS:
        return False, f"hecate has no arm {arm!r} (arms: {', '.join(CKPTS)})"
    p = checkpoint_path(arm)
    if p is None:
        return False, (f"hecate checkpoint not on this host (var/models/{MODEL_SUBDIR}/{CKPTS[arm]} + hecate.py, "
                       f"or ${ENV_DIR}; `vpipe setup models` fetches [models.hecate_{arm}])")
    return True, p


def describe(arm: str | None = None) -> dict:
    p = checkpoint_path(arm)
    if not p:
        return {}
    try:
        return json.load(open(p + ".metrics.json"))
    except Exception:                                    # noqa: BLE001 - a missing sidecar is not fatal
        return {"checkpoint": p, "caption": "no metrics sidecar"}


def resample_depth(plane, n_native: int, src_um: float, out_um: float, n_out: int,
                   sub: int = SUBSAMPLES) -> np.ndarray:
    """(n_out, H, W) uint8: plane k = mean over the slab [o_k - out_um/2, o_k + out_um/2] of the native
    planes linearly interpolated, o_k = (k - n_out//2) * out_um from the stack middle (n_native // 2).
    `plane(i)` returns native plane i as float32. Refuses (ValueError) when the slabs leave the stack."""
    mid = n_native // 2
    half = (n_out // 2) * out_um + out_um / 2
    if mid - half / src_um < 0 or mid + half / src_um > n_native - 1:
        raise ValueError(f"{n_native} planes at {src_um:g} um cannot supply +-{half:.0f} um about the middle")
    cache: dict = {}

    def P(i):
        if i not in cache:
            cache[i] = plane(i)
        return cache[i]
    out = []
    for k in range(n_out):
        o = (k - n_out // 2) * out_um
        acc = None
        for s in range(sub):
            z = mid + (o + ((s + 0.5) / sub - 0.5) * out_um) / src_um
            z0 = int(np.floor(z))
            f = z - z0
            v = (1.0 - f) * P(z0) + f * P(min(z0 + 1, n_native - 1))
            acc = v if acc is None else acc + v
        out.append(np.clip(acc / sub + 0.5, 0, 255).astype(np.uint8))
    return np.stack(out)


def scroll_of(path: str) -> str | None:
    """The scroll id from a work/results path (the finish stage names work dirs by segment id): the LONGEST registered
    scroll id (hecate_frames.json) that a path component starts with, else the component up to its first '_'."""
    try:
        known = sorted(json.load(open(FRAMES_JSON)).get("scrolls", {}), key=len, reverse=True)
    except (OSError, ValueError):
        known = []
    for comp in reversed(str(path).replace("\\", "/").split("/")):
        if comp.startswith("PHerc"):
            for k in known:
                if comp == k or comp.startswith(k + "_"):
                    return k
            return comp.split("_")[0]
    return None


def scroll_frame(scroll: str | None) -> dict | None:
    """The registered source quantiles of a scroll's renders (hecate_frames.json), or None."""
    if not scroll:
        return None
    try:
        frames = json.load(open(FRAMES_JSON))
    except (OSError, ValueError):
        return None
    f = frames.get("scrolls", {}).get(scroll)
    return f if f and f.get("status") != "quarantined" else None


def affine_map(x: np.ndarray, valid: np.ndarray, src_mean: float | None = None, src_std: float | None = None,
               planes: int = 16) -> tuple[np.ndarray, dict]:
    """(x - mu) / sd * REF_STD + REF_MEAN; mu/sd = the scroll's registered stats when given, else this stack's own
    (valid voxels of its first `planes` planes); invalid columns stay 0."""
    v = x[:planes][:, valid].astype(np.float64) if valid.any() else np.asarray([REF_MEAN])
    own_mu, own_sd = float(v.mean()), float(v.std()) or 1.0
    mu = float(src_mean) if src_mean is not None else own_mu
    sd = float(src_std) if src_std else own_sd
    y = np.clip((x.astype(np.float32) - mu) / sd * REF_STD + REF_MEAN + 0.5, 0, 255).astype(np.uint8)
    y[:, ~valid] = 0
    return y, {"src_mean": round(mu, 2), "src_std": round(sd, 2), "own_mean": round(own_mu, 2), "own_std": round(own_sd, 2),
               "ref_mean": REF_MEAN, "ref_std": REF_STD}


def quantile_map(x: np.ndarray, valid: np.ndarray, src_q=None, planes: int = 16) -> tuple[np.ndarray, dict]:
    """Piecewise-linear map of a (Z,H,W) uint8 stack from `src_q` (default: its own percentiles REF_Q over the valid
    voxels of its first `planes` planes) onto REF_V; 0 and 255 are fixed points; invalid columns stay 0."""
    own = np.percentile(x[:planes][:, valid], REF_Q) if valid.any() else np.asarray(REF_V, float)
    src = np.asarray(src_q if src_q is not None else own, float)
    xs = np.maximum.accumulate(np.concatenate([[0.0], src, [255.0]]) + np.arange(len(REF_Q) + 2) * 1e-6)
    y = np.interp(x.astype(np.float32), xs, np.concatenate([[0.0], REF_V, [255.0]]))
    y = np.clip(y + 0.5, 0, 255).astype(np.uint8)
    y[:, ~valid] = 0
    return y, {"src_q": [round(float(v), 2) for v in src], "own_q": [round(float(v), 2) for v in own], "ref_q": REF_V}


def load_upstream(arm: str | None = None):
    """(hecate module, model) from the upstream file beside the weights, md5-checked."""
    import hashlib
    import importlib.util
    d = model_dir(arm)
    if d is None:
        raise FileNotFoundError(available(arm=arm)[1])
    src = os.path.join(d, "hecate.py")
    md5 = hashlib.md5(open(src, "rb").read()).hexdigest()
    if md5 != UPSTREAM["md5"]["hecate.py"]:
        _alerts.alert(f"hecate: {src} md5 {md5} is not the audited upstream {UPSTREAM['md5']['hecate.py']}")
    spec = importlib.util.spec_from_file_location("_hecate_upstream", src)
    H = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(H)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return H, H.load_model(os.path.join(d, CKPTS[arm or DEFAULT_ARM]), dev)


def predict_inproc(layers_dir: str, mask_png: str | None, out_png: str, gpu: int = 0,
                   arm: str | None = None, batch: int = 16, stride: int | None = None,
                   plane_um: float | None = None, norm: str | None = None, **spec) -> dict | None:
    import time
    arm = arm or DEFAULT_ARM
    ok, ckpt = available(arm=arm)
    if not ok:
        _alerts.alert(f"hecate on this host: {ckpt}")
        return None
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu))
    import tempfile
    import torch
    from PIL import Image
    from .colmean import plane_pitch_um, render_um_per_px
    from .grandprize_dense import _layer_array
    from .ink9um_student import layer_files
    Image.MAX_IMAGE_PIXELS = None
    t0 = time.time()
    files = layer_files(layers_dir)
    if not files:
        _alerts.alert(f"hecate: no rendered layers in {layers_dir}")
        return None
    um = render_um_per_px(layers_dir, default=float("nan"))
    target = SAMPLING_UM[arm]
    if not np.isfinite(um) or abs(um / target - 1) > 0.02:
        # hecate.py refuses > 2 % off; so do we, loudly, rather than read letters at the wrong size
        _alerts.alert(f"hecate:{arm}: render {os.path.basename(str(layers_dir).rstrip('/'))} is {um:g} um/px "
                      f"in plane, the checkpoint needs {target:g} (+-2 %); not run")
        return None
    dz = float(plane_um) if plane_um else plane_pitch_um(layers_dir)
    if dz is None:
        _alerts.alert(f"hecate: plane pitch of {layers_dir} unknown (no frame.json, not a registered view); not run "
                      f"-- assuming the in-plane pitch would read the sheet at the wrong depth scale")
        return None
    x = resample_depth(lambda i: np.asarray(_layer_array(files[i]), np.float32), len(files), dz, target,
                       OUT_PLANES[arm])
    H_, W_ = x.shape[1:]
    mid = np.asarray(_layer_array(files[len(files) // 2]))[:H_, :W_] > 0
    norm = (norm or os.environ.get("VPIPE_HECATE_NORM", DEFAULT_NORM)).lower()
    norm_info: dict = {"mode": norm}
    if norm in ("affine", "qmap", "segment-affine", "segment-qmap"):
        sc = scroll_of(out_png) or scroll_of(layers_dir)
        fr = scroll_frame(sc) if not norm.startswith("segment") else None
        if fr is None and not norm.startswith("segment"):
            _alerts.alert(f"hecate: no registered intensity frame for scroll {sc or '?'} (hecate_frames.json); FALLBACK: "
                          f"this segment's own statistics ({os.path.basename(os.path.dirname(os.path.abspath(out_png)))})")
        if norm.endswith("qmap"):
            x, st = quantile_map(x, mid, src_q=fr["src_q"] if fr else None)
        else:
            x, st = affine_map(x, mid, src_mean=fr.get("src_mean") if fr else None, src_std=fr.get("src_std") if fr else None)
        norm_info.update(st, scroll=sc, source=("scroll frame: " + fr["source"]) if fr else "segment's own statistics (fallback)")
    elif norm != "none":
        raise ValueError(f"hecate norm={norm!r}: 'affine' (default), 'qmap', 'segment-affine', 'segment-qmap' or 'none'")
    # the INPUT-FRAME SENTINEL reads the tensor actually fed to the model (stages/input_frame.py via ink.py)
    try:
        from .. import input_frame as IF
        json.dump(IF.stack_stats(x[:16], mid), open(os.path.splitext(out_png)[0] + ".input_frame.json", "w"))
    except Exception as e:                               # noqa: BLE001 - never fail a prediction on its sentinel
        print(f"[hecate] input-frame stats failed: {type(e).__name__}: {e}", flush=True)
    t1 = time.time()
    HM, model = load_upstream(arm)
    torch.backends.cudnn.benchmark = True
    dev = next(model.parameters()).device.type
    # upstream's production precision is bf16; a card without native bf16 (V100, sm_70) runs fp32
    prec = "bf16" if dev == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else "fp32"
    res = np.zeros((H_, W_), np.uint8)
    work = os.path.dirname(os.path.abspath(out_png)) or "."
    os.makedirs(work, exist_ok=True)
    old_tmp = tempfile.tempdir
    tempfile.tempdir = work                # hecate's disk accumulators: never the (small) root disk
    try:
        bs = batch
        while True:
            try:
                HM.predict(model, x, res, reverse=False, stride=stride, batch_size=bs, precision=prec)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                print(f"[hecate] CUDA OOM; retry at batch {bs}", flush=True)
    finally:
        tempfile.tempdir = old_tmp
    if dev == "cuda":
        torch.cuda.synchronize()
    t2 = time.time()
    res = (res * mid).astype(np.uint8)
    if mask_png and os.path.exists(mask_png):
        mk = np.asarray(Image.open(mask_png).convert("L"))
        h, w = min(mk.shape[0], H_), min(mk.shape[1], W_)
        res[:h, :w] *= (mk[:h, :w] > 127)
    Image.fromarray(res).save(out_png)
    mpx = H_ * W_ / 1e6
    stats = {"wall_s": round(time.time() - t0, 2), "gpu_s": round(t2 - t1, 2), "read_resample_s": round(t1 - t0, 2),
             "shape": [H_, W_], "mpx": round(mpx, 3), "mpix_per_s": round(mpx / max(t2 - t1, 1e-6), 4),
             "mvox_per_s": round(mpx * 16 / max(t2 - t1, 1e-6), 3), "batch": bs, "precision": prec,
             "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
             "in_plane_um": um, "native_plane_um": dz, "resampled_planes": int(x.shape[0]),
             "stride": stride or 32, "norm": norm_info}
    with open(os.path.splitext(out_png)[0] + ".hecate.json", "w") as fh:
        json.dump({"model": f"hecate:{arm}", "checkpoint": ckpt, "spec": SPEC, "metrics": describe(arm),
                   "stats": stats}, fh, indent=1)
    return stats


def predict(layers_dir: str, mask_png: str, out_png: str, gpu: int = 0, arm: str | None = None,
            fleet=None, log: str | None = None, timeout: int = 4 * 3600, **spec) -> bool:
    """One rendered stack -> one probability PNG (uint8 = round(255 p), upstream convention). In-process
    when torch imports here, else handed to a torch interpreter (worker venvs on peers have none)."""
    ok, why = available(arm=arm)
    if not ok:
        _alerts.alert(f"hecate on this host: {why}")
        return False
    try:
        import torch  # noqa: F401
        return predict_inproc(layers_dir, mask_png, out_png, gpu=gpu, arm=arm, **spec) is not None
    except ImportError:
        pass
    from ... import config
    from ..finish import _run
    from .ink9um_student import _torch_python
    if fleet is None:
        fleet = config.load()
    py = _torch_python(fleet)
    if not py:
        _alerts.alert("hecate on this host: no torch interpreter (villa/vesuvius/.venv)")
        return False
    src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    cmd = [py, "-m", "vesuvius_pipeline.stages.ink_models.hecate", "--layers", layers_dir, "--out", out_png,
           "--gpu", "0"] + (["--mask", mask_png] if mask_png and os.path.exists(mask_png) else []) \
        + (["--arm", arm] if arm else [])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
               PYTHONPATH=src + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""))
    rc = _run(cmd, log or os.path.splitext(out_png)[0] + ".log", timeout, env=env)
    return rc == 0 and os.path.exists(out_png) and os.path.getsize(out_png) > 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mask", default=None)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--arm", default=None)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    raise SystemExit(0 if predict_inproc(a.layers, a.mask, a.out, gpu=a.gpu, arm=a.arm, batch=a.batch) else 1)
