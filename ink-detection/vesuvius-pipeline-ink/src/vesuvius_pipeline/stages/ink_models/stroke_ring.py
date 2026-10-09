"""stroke_ring: the UNTRAINED "stroke width and ring" ink score (user's name, 2026-09-15).

No checkpoint, no fit, no labels. The score is a fixed closed form of the rendered depth
stack, so its held-out AUC on a scroll it has never seen is not a transfer loss but the
same measurement -- the argument that puts the `colmean` family in the default set applies
here unchanged, and this score beats every one of those priors on the two scrolls where
they were all measured.

WHAT IT COMPUTES (FINDINGS 95.2 / 95.3, the configuration written down before it was
scored, `SxD|slide|fix|C1`):

    band_j      the depth average over a 23.7 um slab of the stack, at ten positions
                spaced 7.91 um apart from -39.6 um to +31.6 um about the SURFACE layer
    member_ji   G_sigma_i(band_j) - G_ring(band_j)      stroke-width centre MINUS ring
                with sigma_i in {126.6, 180, 253.1, 360, 506.2} um and ring 1012.5 um
    score       mean over the 50 members of each member z-scored over the scored mask

Every blur is a MASK-NORMALISED Gaussian, blur(x*v)/blur(v), so the sheet's edge does not
bleed the background in -- the unnormalised form inflated large-sigma cells by up to 0.047
AUC near invalid pixels (95.2), which is how that error was found.

EVERY LENGTH IS IN MICRONS, including the depth geometry. The score was validated on a
7.91 um cache whose planes are 7.91 um apart, where the band happened to be three planes;
production renders are 9.5 or 11.81 um with ~65 planes, where three planes would be a
28.5-35.4 um band -- a different statistic wearing the same name. Bands are therefore
integrated between FRACTIONAL depths from a depth cumulative sum, so the slab is 23.7 um
at every pitch, and only the ~12 planes the bands touch are ever read.

The band positions are fixed offsets from the surface layer (the middle of the rendered
stack, the same layer `make_mask` thresholds), NOT from a detected sheet: 95.2 measured
sheet-relative bands helping Scroll 5 (+0.034) but hurting Scroll 1 (-0.023) and Scroll 4
(-0.044), with a label-free sheet estimate too noisy to arbitrate (5-95 % spread 4-154 um).

WHY THOSE NUMBERS, each measured rather than chosen:
  * centre 253 um (FWHM 0.60 mm = the stroke width) beats 506 um on all three scrolls, and
    beats every narrower centre: below ~63 um the score is chance (0.51). The optimum is
    the stroke, not something larger -- the pre-registered prediction that it kept rising
    was FALSIFIED (95.2).
  * the ring is flat from 0.76 to 2 mm; 1012.5 um is the middle of that plateau.
  * ten depth bands, not one: depth bands correlate at rho ~ 0.4, so averaging ten of them
    buys a real factor, while the five scales correlate at rho ~ 0.85 and buy almost
    nothing (95.3 mechanism check). Averaging is also the best combiner measured: max,
    min, nested and soft-agreement operators (22 of them), a fitted 50-feature logistic,
    and median / trimmed estimators were ALL at or below this plain mean.

HELD-OUT NUMBERS (near-ink every-pixel AUC, box-unit 95 % CI; 95.3):

    Scroll 4 wNNN      0.621 [0.551, 0.700]   scored mask 0.666   (plain band mean 0.556)
    Scroll 5 (LOO)     0.616                  scored mask 0.600   (plain all-20 mean 0.581)
    Scroll 1 (2 folds) 0.555                  scored mask 0.542   (plain 0.540)

and with NOTHING tuned on the target scroll (leave one scroll out) 0.594 / 0.616 / 0.555
against 0.525 / 0.611 / 0.523 for the best single filter of 95.2. It is still well below a
Scroll-4-TRAINED U-Net on Scroll 4 (0.72-0.79): this is the untrained ceiling, not the
ceiling.

THE ML WINDOW. The integration support is the ring, not the centre: 2*3*1012.5 um = 6.1 mm
at 7.91 um/px. `registry.py` notes the First Letters 0.5 mm ML-window limit; that limit was
CHANGED for the First Letters eligible scrolls (user, 2026-09-15), which is why this family
ships the validated 1-2 mm ring by default. `tile_px` still declares the honest support, so
the real window is published with every prediction. A narrower ring is available as the
secondary arm `ring506` and scores lower everywhere it was measured.

SPEED and BACKENDS (measured 2026-09-15, 8 threads, the hub):
  * `gauss` (scipy on a 2x pyramid) is the DEFAULT: it is the path the 95.3 numbers were
    measured with, and it reproduces them to 0.00004 AUC.
  * `box` (three box passes per blur) is REJECTED as a default: it is a different
    statistic, not an approximation -- worst |dAUC| 0.00648 against a 0.002 bar
    (S5 segment +0.00648, S4 wNNN +0.00494) -- and it is NOT faster on the hub
    (0.32-0.35 vs 0.37 MP/s). It looked 2.5x faster only on another host, whose venv has no
    scipy at all, so that comparison was between two different machines' numpy.
  * `torch` runs the same pyramid on a GPU.
  Threads buy almost nothing (0.37 vs 0.36 MP/s at 8 vs 32 on the hub): the work is
  memory-bandwidth-bound, so a host runs several SEGMENTS in parallel rather than one
  segment on many cores.
"""
from __future__ import annotations

import math
import os

from ... import alerts as _alerts

import numpy as np

from .colmean import SURFACE_LAYER_UM, render_um_per_px
from .colmean import to_uint8 as _colmean_to_uint8  # kept for reference; see to_uint8 below

# reads an in-memory native-stack view (stages/memstack.py) through its layer readers
MEMSTACK_READY = True

GP_TARGET_UM = 7.91


def to_uint8(score, mask, lo_p: float = 1.0, hi_p: float = 99.0):
    """colmean's monotone stretch, with the percentiles taken on a SUBSAMPLE of the masked
    pixels once there are more than 4 M of them. np.percentile sorts everything it is given,
    which on a 17 MP segment costs more than the blurs do. The stretch is monotone either
    way, so the published image ranks its pixels identically and the AUC is unchanged; only
    the 1st/99th cut points move, by the sampling error of a 1-in-7 sample."""
    import numpy as _np
    v = score[mask > 0]
    if v.size == 0:
        return _np.zeros(score.shape, dtype=_np.uint8)
    s = v[::7] if v.size > 4_000_000 else v
    lo, hi = _np.percentile(s, [lo_p, hi_p])
    if not _np.isfinite(lo) or not _np.isfinite(hi) or hi <= lo:
        lo, hi = float(_np.min(v)), float(_np.max(v))
    if hi <= lo:
        return _np.zeros(score.shape, dtype=_np.uint8)
    out = _np.clip((score - lo) / (hi - lo), 0, 1) * 255.0
    return (out * (mask > 0)).astype(_np.uint8)

# The 95.3 configuration, in MICRONS.
SCALES_UM: tuple[float, ...] = (126.56, 180.0, 253.12, 360.0, 506.24)
BAND_UM = 3 * GP_TARGET_UM                       # 23.73 um slab
BAND_OFFSETS_UM: tuple[float, ...] = tuple(round(k * GP_TARGET_UM, 4) for k in range(-5, 5))
PYRAMID_MIN_SIGMA_PX = 4.0
DEFAULT_BACKEND = "gauss"

ARMS: dict[str, dict] = {
    "ring1012": dict(ring_um=1012.48, scales_um=SCALES_UM,
                     auc="S4 0.621 [0.551,0.700] | S5 0.616 | S1 0.555 (near-ink, held out, 95.3); "
                         "leave-one-scroll-out 0.594 / 0.616 / 0.555",
                     source="FINDINGS 95.3 SxD|slide|fix|C1"),
    "ring2025": dict(ring_um=2025.0, scales_um=SCALES_UM,
                     auc="within 0.002 of ring1012 (the ring is flat 0.76-2 mm, 95.2)",
                     source="FINDINGS 95.2, the far end of the ring plateau"),
    "ring506": dict(ring_um=506.24, scales_um=SCALES_UM,
                    auc="below ring1012 on all three scrolls (95.2 ring sweep)",
                    source="the NARROW ring; kept runnable, not the default"),
}

SPEC = {
    "layers": 65, "layer_start": 0, "layer_stack": 65, "tile": 0, "stride": 0,
    "um_per_px": GP_TARGET_UM,
    "normalisation": "none (raw rendered intensities; per-member z over the scored mask)",
    "arch": "closed_form_stroke_width_minus_ring", "output_stride_px": 1,
    "trained_on": "nothing -- no weights exist",
}


def training_frame():
    from ...frame import MODELS, ModelSpec
    return MODELS.get("stroke_ring", ModelSpec("stroke_ring", GP_TARGET_UM, 65)).frame(
        source_id="model:stroke_ring")


def check_frame(frame, tol: float = 0.05) -> None:
    """Physical lengths throughout: this score is not frame-bound the way a trained net is."""
    return None


def enabled() -> bool:
    return True


def default_arm() -> str:
    return "ring1012"


def available(fleet=None, arm: str | None = None) -> tuple[bool, str]:
    a = arm or default_arm()
    if a not in ARMS:
        return False, f"stroke_ring arm {arm!r} unknown; known: {sorted(ARMS)}"
    return True, f"stroke_ring:{a} (no checkpoint -- closed form)"


def describe(arm: str | None = None) -> dict:
    a = ARMS.get(arm or default_arm(), {})
    return {"family": "stroke_ring", "arm": arm or default_arm(), "spec": dict(SPEC), **a}


# ---- reading the stack ----------------------------------------------------------------
def _layer_files(layers_dir: str) -> list[str]:
    from .. import memstack as _MS
    names = _MS.list_names(layers_dir)              # an in-memory view of the native stack
    if names is not None:
        return names
    return sorted(f for f in os.listdir(layers_dir) if f.lower().endswith((".tif", ".tiff")))


def _read(layers_dir: str, files: list[str], k: int) -> np.ndarray:
    from .. import memstack as _MS
    st = _MS.get(layers_dir)
    if st is not None:
        return st.full(st.names.index(files[k])).astype(np.float32)
    import tifffile
    return tifffile.imread(os.path.join(layers_dir, files[k])).astype(np.float32)


def band_planes(layers_dir: str, dz_um: float | None = None, offsets_um=BAND_OFFSETS_UM,
                band_um: float = BAND_UM):
    """-> (bands, valid, detail). Each band is the depth MEAN over a `band_um` slab centred
    `off` microns from the surface layer, integrated between fractional plane depths so the
    slab is the same physical thickness at any plane pitch.

    Only the planes the slabs touch are read, each exactly once.
    """
    files = _layer_files(layers_dir)
    n = len(files)
    if not n:
        raise ValueError(f"no rendered layers in {layers_dir}")
    dz = float(dz_um or render_um_per_px(layers_dir))   # a render's planes are one voxel apart
    zc = n // 2
    half = band_um / 2.0
    lo_p = int(math.floor(zc + (min(offsets_um) - half) / dz))
    hi_p = int(math.ceil(zc + (max(offsets_um) + half) / dz))
    lo_c, hi_c = max(0, lo_p), min(n - 1, hi_p)
    cache = {k: _read(layers_dir, files, k) for k in range(lo_c, hi_c + 1)}
    if zc not in cache:
        cache[zc] = _read(layers_dir, files, zc)
    shape = cache[next(iter(cache))].shape
    # depth cumulative sum over the planes we hold: cum[i] = integral of the stack up to the
    # TOP of plane lo_c+i-1, in plane units, so a slab is a difference of two interpolations
    keys = sorted(cache)
    cum = np.zeros((len(keys) + 1,) + shape, dtype=np.float32)
    for i, k in enumerate(keys):
        cum[i + 1] = cum[i] + cache[k]

    def integ(u):                       # u in plane units relative to keys[0], clipped
        u = float(np.clip(u, 0.0, len(keys)))
        j = min(int(math.floor(u)), len(keys) - 1)
        return cum[j] + (u - j) * cache[keys[j]]

    bands, dropped = [], []
    for off in offsets_um:
        a_u = (zc + (off - half) / dz) - keys[0]
        b_u = (zc + (off + half) / dz) - keys[0]
        if b_u <= 0 or a_u >= len(keys) or (b_u - a_u) < 1e-6:
            dropped.append(off)
            continue
        a_c, b_c = max(a_u, 0.0), min(b_u, float(len(keys)))
        if (b_c - a_c) < 0.25 * (b_u - a_u):        # mostly outside the stack: not the same band
            dropped.append(off)
            continue
        bands.append((integ(b_c) - integ(a_c)) / (b_c - a_c))
    if not bands:
        raise ValueError(f"{n}-plane stack at {dz:g} um/plane holds none of the {band_um:g} um bands")
    valid = cache[zc] > 0
    detail = (f"{len(bands)}/{len(offsets_um)} bands, {band_um:g} um slabs, "
              f"{len(keys)} of {n} planes read at {dz:g} um/plane")
    if dropped:
        detail += f"; dropped offsets {[round(d, 1) for d in dropped]} um (outside the stack)"
    return bands, valid, detail


# ---- blurring -------------------------------------------------------------------------
class _Pyramid:
    def __init__(self, x):
        self.lv = {1: x}

    @staticmethod
    def _down(a):
        H, W = a.shape[-2:]
        ph, pw = (-H) % 2, (-W) % 2
        if ph or pw:
            a = np.pad(a, ((0, ph), (0, pw)))
        return a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2).mean((1, 3))

    def get(self, level):
        while level not in self.lv:
            m = max(self.lv)
            self.lv[2 * m] = self._down(self.lv[m])
        return self.lv[level]


def _level_for(sigma_px: float) -> int:
    level = 1
    while sigma_px / (2 * level) >= PYRAMID_MIN_SIGMA_PX:
        level *= 2
    return level


def _box3(a: np.ndarray, sigma_px: float) -> np.ndarray:
    """Three box passes per axis: variance-matched to a Gaussian, two cumulative sums each,
    and independent of sigma. 2.5-3x faster than scipy's Gaussian at these sigmas."""
    w = max(1, int(round(sigma_px * math.sqrt(12.0 / 3.0) / 2)) * 2 + 1)
    if w == 1:
        return a
    r = w // 2
    out = a
    for axis in (0, 1):
        out = np.moveaxis(out, axis, -1)
        for _ in range(3):
            c = np.cumsum(np.pad(out, ((0, 0),) * (out.ndim - 1) + ((r + 1, r),)), axis=-1)
            out = (c[..., w:] - c[..., :-w]) / float(w)
        out = np.moveaxis(out, -1, axis)
    return out


def _gauss(a: np.ndarray, sigma_px: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(a, sigma_px, mode="constant", cval=0.0, truncate=4.0)


def _blur(a: np.ndarray, sigma_px: float, backend: str) -> np.ndarray:
    """`gauss` NEVER degrades to `box` on its own. The box cascade is a DIFFERENT statistic,
    not an approximation: it missed the reproduction bar by 3x (worst |dAUC| 0.00648 against
    0.002) and was not faster on the same host. A host without scipy must be fixed or
    skipped, not silently given another score under this family's name."""
    if sigma_px < 0.5:
        return a
    if backend == "box":
        return _box3(a, sigma_px)
    try:
        return _gauss(a, sigma_px)
    except ImportError as e:
        raise RuntimeError(
            "stroke_ring needs scipy for its validated `gauss` backend; this interpreter has "
            "none. Install scipy or pass backend='box' deliberately, knowing it fails the "
            "0.002 reproduction bar (FINDINGS 95.3 / stroke_ring docstring).") from e


def _upsample(a: np.ndarray, shape, level: int) -> np.ndarray:
    if level == 1:
        return a[:shape[0], :shape[1]]
    try:
        from scipy.ndimage import zoom
        up = zoom(a, level, order=1, mode="nearest", grid_mode=True, prefilter=False)
    except ImportError:
        up = np.repeat(np.repeat(a, level, 0), level, 1)
    H, W = shape
    if up.shape[0] < H or up.shape[1] < W:
        up = np.pad(up, ((0, max(0, H - up.shape[0])), (0, max(0, W - up.shape[1]))), mode="edge")
    return up[:H, :W]


def masked_blur(num_pyr, den_pyr, sigma_px, shape, backend):
    level = _level_for(sigma_px)
    s = sigma_px / level
    nb = _blur(num_pyr.get(level), s, backend)
    db = _blur(den_pyr.get(level), s, backend)
    return _upsample(nb / np.clip(db, 1e-6, None), shape, level)


# ---- the score ------------------------------------------------------------------------
def stroke_ring_score(bands, valid, um_per_px, ring_um, scales_um=SCALES_UM, zref=None,
                      backend: str = DEFAULT_BACKEND, gpu: int | None = None):
    """Mean over (band x scale) of each centre-minus-ring member, z-scored over `zref`."""
    if backend == "torch":
        return _score_torch(bands, valid, um_per_px, ring_um, scales_um, zref, gpu)
    v = (valid > 0).astype(np.float32)
    ref = (zref if zref is not None else valid) > 0
    shape = v.shape
    den = _Pyramid(v)
    sig = [float(s) / um_per_px for s in scales_um] + [float(ring_um) / um_per_px]
    acc = np.zeros(shape, dtype=np.float32)
    n = 0
    for x in bands:
        num = _Pyramid(np.asarray(x, dtype=np.float32) * v)
        at = {s: masked_blur(num, den, s, shape, backend) for s in sig}
        ring = at[sig[-1]]
        for s in sig[:-1]:
            m = at[s] - ring
            r = m[ref]
            mu = float(r.mean()) if r.size else 0.0
            sd = float(r.std()) if r.size else 1.0
            acc += (m - mu) / max(sd, 1e-9)
            n += 1
        del num, at
    return acc / float(max(1, n)), n


def _score_torch(bands, valid, um_per_px, ring_um, scales_um, zref, gpu):
    """The same arithmetic on a GPU: avg_pool2d for the pyramid, a separable Gaussian for
    the blur. Checked against the CPU path by `stroke_ring_verify.py`."""
    import torch
    import torch.nn.functional as F
    dev = torch.device(f"cuda:{gpu or 0}" if torch.cuda.is_available() else "cpu")
    v = torch.from_numpy((valid > 0).astype(np.float32)).to(dev)[None, None]
    ref = torch.from_numpy(((zref if zref is not None else valid) > 0)).to(dev).reshape(-1)
    H, W = v.shape[-2:]

    def pyr(t, level):
        while level > 1:
            ph, pw = (-t.shape[-2]) % 2, (-t.shape[-1]) % 2
            if ph or pw:
                t = F.pad(t, (0, pw, 0, ph))
            t = F.avg_pool2d(t, 2)
            level //= 2
        return t

    def gblur(t, s):
        if s < 0.5:
            return t
        r = max(1, int(math.ceil(4 * s)))
        x = torch.arange(-r, r + 1, dtype=torch.float32, device=dev)
        k = torch.exp(-0.5 * (x / s) ** 2)
        k = k / k.sum()
        t = F.conv2d(F.pad(t, (r, r, 0, 0)), k.view(1, 1, 1, -1))
        return F.conv2d(F.pad(t, (0, 0, r, r)), k.view(1, 1, -1, 1))

    sig = [float(s) / um_per_px for s in scales_um] + [float(ring_um) / um_per_px]
    acc = torch.zeros(H * W, device=dev)
    n = 0
    for x in bands:
        xb = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(dev)[None, None] * v
        at = {}
        for s in sig:
            L = _level_for(s)
            nb, db = gblur(pyr(xb, L), s / L), gblur(pyr(v, L), s / L)
            r = nb / db.clamp_min(1e-6)
            if L > 1:
                # upsample by the EXACT pyramid factor and crop, as the numpy path does.
                # `interpolate(size=(H, W))` stretches instead, which shifts the grid
                # whenever H or W is not a multiple of L -- measured at 0.0025 AUC on
                # s5 segment (650 x 3742: H % 4 = 2, W % 8 = 6), i.e. 2.5x the
                # agreement bar, from nothing but resampling alignment.
                r = F.interpolate(r, scale_factor=L, mode="bilinear",
                                  align_corners=False)[..., :H, :W]
            at[s] = r.reshape(-1)
        ring = at[sig[-1]]
        for s in sig[:-1]:
            m = at[s] - ring
            rv = m[ref]
            # population std (ddof 0), which is what numpy's .std() uses; torch defaults
            # to the unbiased estimator and the two paths must standardise identically
            acc += (m - rv.mean()) / rv.std(correction=0).clamp_min(1e-9)
            n += 1
        del xb, at
    return (acc / max(1, n)).reshape(H, W).cpu().numpy(), n


def predict(layers_dir: str, mask_png: str, out_png: str, gpu: int = 0, arm: str | None = None,
            fleet=None, log: str | None = None, backend: str | None = None,
            um_per_px: float | None = None, **spec) -> bool:
    arm = arm or default_arm()
    ok, why = available(fleet, arm)
    if not ok:
        _alerts.alert(f"stroke_ring on this host: {why}")
        return False
    a = ARMS[arm]
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    import time
    t0 = time.time()
    # `frame.json` is authoritative when finish wrote one; an explicit override is next;
    # the 7.91 um fallback is LAST and is announced, because an ink9um render is 9.5 um and
    # inheriting 7.91 would shrink every physical length in this score by ~20 %.
    um = float(um_per_px) if um_per_px else render_um_per_px(layers_dir)
    if not um_per_px and not os.path.exists(os.path.join(layers_dir, "frame.json")):
        print(f"[stroke_ring] no frame.json beside {os.path.basename(layers_dir)}; "
              f"using {um:g} um/px -- pass um_per_px= if that is not this render's pitch", flush=True)
    # the DEPTH pitch is the CT voxel (2026-09-29; it used to be `um`, the in-plane pitch)
    from .colmean import plane_pitch_um
    dz = plane_pitch_um(layers_dir)
    if dz is None:
        dz = um
        _alerts.alert(f"stroke_ring: plane pitch of {os.path.basename(str(layers_dir).rstrip('/'))} unknown; "
                      f"assuming {um:g} um/plane")
    bands, valid, detail = band_planes(layers_dir, dz_um=dz)
    read_s = time.time() - t0
    if os.path.exists(mask_png):
        m = np.asarray(Image.open(mask_png).convert("L"))
        if m.shape != valid.shape:
            print(f"[stroke_ring] mask {m.shape} != render {valid.shape}; scoring the whole plane", flush=True)
            m = np.ones(valid.shape, dtype=np.uint8) * 255
    else:
        m = np.ones(valid.shape, dtype=np.uint8) * 255
    keep = (m > 0) & valid
    if not keep.any():
        print("[stroke_ring] empty mask; nothing to score", flush=True)
        return False
    t1 = time.time()
    score, n_members = stroke_ring_score(bands, keep, um, a["ring_um"], a["scales_um"],
                                         backend=backend or DEFAULT_BACKEND, gpu=gpu)
    calc_s = time.time() - t1
    Image.fromarray(to_uint8(score, (keep * 255).astype(np.uint8))).save(out_png)
    mp = valid.size / 1e6
    print(f"[stroke_ring:{arm}] {detail}; centres {'/'.join(f'{s:g}' for s in a['scales_um'])} um "
          f"- ring {a['ring_um']:g} um; {n_members} members @ {um:.2f} um/px; {mp:.1f} MP "
          f"read {read_s:.1f}s + score {calc_s:.1f}s = {mp / max(calc_s, 1e-6):.2f} MP/s "
          f"[{backend or DEFAULT_BACKEND}]", flush=True)
    return True
