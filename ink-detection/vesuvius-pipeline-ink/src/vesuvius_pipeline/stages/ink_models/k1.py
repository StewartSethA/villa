"""k1: UNTRAINED consensus of C1 (the reference bar) and the edge-aligned band E.

FINDINGS Sec 122 addendum 3 / Sec 122.1 part 3 (2026-09-26): K1 = mean of the per-segment
robust z of C1 and of `phy_edgeE_hi_s1012` -- untrained, no weights, no fit. K1 is the first
untrained arm measured to beat C1 on UNSEEN objects: near-64 AUC S4 0.634 (+0.040 vs C1,
5/5 segments), S5 0.667 (+0.027, 12/13), S1 0.520 (-0.008, 2/7); on 19 held-out segments of
6 wholly unseen objects (never touched by any choice this score made), 18/19 segments and
6/6 objects improve, +0.052 [+0.029, +0.078] AUC object-level.

WHAT IT COMPUTES, reproduced here from the research scripts that measured the numbers above
(`scripts/ink_transfer/featbank.py` c1_C1, `featbank2.py` phy_edgeE_hi_s1012):

  C1   ten FIXED depth bands (23.7 um slabs, offsets -39.55..+31.64 um about the render's
       surface layer, one every 7.91 um) x five centre scales (126.56/180/253.12/360/506.24
       um) each minus a 1012.48 um surround, EVERY MEMBER ROBUST-Z-SCORED (median/MAD) over
       the scored mask, then averaged over all 50 members. This is the SAME band geometry,
       centre scales and ring as `stroke_ring`'s `ring1012` arm (reused here via
       `stroke_ring.band_planes` / `stroke_ring._Pyramid` / `stroke_ring.masked_blur`) with
       ONE difference: stroke_ring standardises each member by MEAN/STD, C1 by MEDIAN/MAD --
       which is why C1's own published numbers (S4 0.594/S5 0.641/S1 0.528) differ from
       stroke_ring's (S4 0.621/S5 0.616/S1 0.555) despite the identical band geometry.
  E    the grey value 7.2 um INSIDE the ink-side half-max sheet edge (sub-plane linear
       interpolation of the depth profile), pooled at sigma 100 um (stroke-interior pooling)
       minus a 1012.5 um surround. The sheet edge is found PER PIXEL from the stack's own
       depth profile (laterally smoothed at 31.6 um): peak plane, then the last plane at or
       above half-max on the high-index side ("hi" -- the ink-side edge that reads
       +0.033 AUC alone on S4; the low-index "lo" edge is not used here). No render-stage
       change is needed: this is exactly what the docstring instruction means by "the
       per-column sheet edge comes from the stack's depth profile".
  K1 = 0.5 * (robust_z(C1) + robust_z(E)), both z-scored over the SAME reference population
       (the scored mask -- here, the pixels `mask_png` marks, matching how the research
       scripts z-score over `sub["mask"]`).

WORKING PITCH. `featbank.py` computes all of this at 15.82 um lateral (a 2x2 block mean of
the native 7.91 um cache, `--f 2`) -- FINDINGS: "Reproduces published C1/K1 exactly ... at
15.8 um lateral". `lateral_pool` defaults to 2 here for the same reason: it is the pitch the
published numbers were measured at, not an efficiency choice.

DEPTH SPAN THIS NEEDS, VERIFIED: C1's ten bands span -39.55..+31.64 um (71.2 um total) about
the surface layer -- 9 plane-widths at 7.91 um/plane. E's edge search reads a WIDER window
(+-`edge_half_window_um`, default 80 um each side = ~20 planes total at 7.91 um/plane) so the
half-max can be found even when the sheet sits a few planes off-centre. A 26-layer ink_9um
render (span ~205 um at 7.91 um, or ~247 um at the 9.5 um it is actually rendered at) and a
65-layer Grand Prize render (span 514 um) both clear this with margin; anything narrower
triggers the loud clip warning below rather than silently truncating.

REUSE, DELIBERATE. `stroke_ring.py` already ships the exact band geometry, masked-Gaussian
pyramid and pitch handling C1 needs (SAME BAND_OFFSETS_UM, SAME five centre scales, SAME
1012-ish ring -- verified: `UM*128 = 1012.48` in the research script is `stroke_ring`'s own
`ring1012` value to the last decimal). Importing it instead of re-deriving the pyramid means
this module and `stroke_ring` are two independently-reviewed call sites of one blur
implementation, not two chances for the same rounding bug.

VALIDATION. `scripts/ink_transfer/k1_repro_validate.py` runs this module's `predict()` and
its internal `compute_k1()` against the frozen `cache7p91` arrays (raw.npy re-exported as a
TIFF stack, lm.npy as mask/label) for the labelled S4/S5/S1 segments and compares near-64 AUC
against the published table above. See that script's output / FINDINGS for the reproduction
numbers and tolerance actually measured -- this docstring states the target, not a claim that
it was hit; read the dated FINDINGS section for the result.

COST. No GPU, no torch: CPU only (scipy Gaussian on a 2x pyramid, same backend as
stroke_ring). Reads ~20-25 of the rendered layers regardless of stack length. Throughput is
printed by `predict()` and reported in Mvoxels/s in the validation script's output.
"""
from __future__ import annotations

import math
import os

from ... import alerts as _alerts

import numpy as np

from . import stroke_ring as SR
from .colmean import render_um_per_px

# reads an in-memory native-stack view (stages/memstack.py) through its layer readers
MEMSTACK_READY = True

GP_TARGET_UM = 7.91

# ---- C1 geometry: reused byte-for-byte from stroke_ring (see module docstring) ------------
C1_SCALES_UM = SR.SCALES_UM                          # (126.56, 180.0, 253.12, 360.0, 506.24)
C1_BAND_OFFSETS_UM = SR.BAND_OFFSETS_UM               # (-39.55 .. +31.64), 10 offsets, 7.91 um apart
C1_BAND_UM = SR.BAND_UM                               # 23.73 um slab
C1_RING_UM = SR.ARMS["ring1012"]["ring_um"]           # 1012.48 um -- identical to UM*128

# ---- edge-aligned band E: featbank2.py's phy_edgeE_hi_s1012, reproduced here --------------
EDGE_SMOOTH_UM = 31.6           # lateral smoothing sigma used to locate the sheet edge
EDGE_INSIDE_UM = 7.2            # depth offset from the half-max edge, toward the sheet interior
EDGE_CENTER_UM = 100.0          # stroke-interior pooling sigma
EDGE_SURR_UM = 1012.5           # surround sigma (phy_edgeE_hi_s1012, not phy_edgeE_hi_s400)
EDGE_HALF_WINDOW_UM_DEFAULT = 80.0   # +- window read for edge search (~20 planes at 7.91 um)

DEFAULT_LATERAL_POOL = 2        # 15.82 um working pitch at a 7.91 um/px render -- see docstring
DEFAULT_BACKEND = "gauss"

ARMS: dict[str, dict] = {
    "consensus": dict(
        auc="S4 0.634 [near64, n=5 seg] | S5 0.667 [n=13] | S1 0.520 [n=7] "
            "(FINDINGS Sec 122.1 part 3, 2026-09-26; delta vs C1 +0.040 / +0.027 / -0.008); "
            "unseen objects (6 objects, 19 segments): +0.052 [+0.029,+0.078] AUC object-level, 6/6",
        source="scripts/ink_transfer/featbank.py (c1_C1) + featbank2.py (phy_edgeE_hi_s1012); "
               "K1 = mean(robust_z(C1), robust_z(E)), FINDINGS Sec 122 addendum 3",
    ),
}

SPEC = {
    "layers": 65, "layer_start": 0, "layer_stack": 65, "tile": 0, "stride": 0,
    "um_per_px": GP_TARGET_UM,
    "normalisation": "none (raw rendered intensities; C1/E members robust-z (median/MAD) over the scored mask)",
    "arch": "closed_form_C1_plus_edge_band_consensus", "output_stride_px": 1,
    "trained_on": "nothing -- no weights exist",
}


def training_frame():
    from ...frame import MODELS, ModelSpec
    return MODELS.get("k1", ModelSpec("k1", GP_TARGET_UM, 65)).frame(source_id="model:k1")


def check_frame(frame, tol: float = 0.05) -> None:
    """Physical lengths throughout, like stroke_ring/colmean: not frame-bound the way a
    trained net is."""
    return None


def enabled() -> bool:
    return True


def default_arm() -> str:
    return "consensus"


def available(fleet=None, arm: str | None = None) -> tuple[bool, str]:
    a = arm or default_arm()
    if a not in ARMS:
        return False, f"k1 arm {arm!r} unknown; known: {sorted(ARMS)}"
    return True, f"k1:{a} (no checkpoint -- closed form)"


def describe(arm: str | None = None) -> dict:
    a = ARMS.get(arm or default_arm(), {})
    return {"family": "k1", "arm": arm or default_arm(), "spec": dict(SPEC), **a}


# ---- shared small helpers ---------------------------------------------------------------
def _pool2(a: np.ndarray, f: int) -> np.ndarray:
    """Block-mean lateral downsample by integer factor f (edge-padded)."""
    if f <= 1:
        return a
    H, W = a.shape[-2:]
    ph, pw = (-H) % f, (-W) % f
    if ph or pw:
        a = np.pad(a, ((0, ph), (0, pw)), mode="edge")
    Hh, Wp = a.shape[-2:]
    return a.reshape(Hh // f, f, Wp // f, f).mean((1, 3))


def _rz(v: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Robust z (median/MAD over `ref`), matching featbank.py's `rz`."""
    r = v[ref]
    r = r[np.isfinite(r)]
    if r.size == 0:
        return np.zeros_like(v)
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med))) * 1.4826
    if mad < 1e-9:
        sd = float(r.std())
        mad = sd if sd > 1e-9 else 1.0
    out = (v - med) / mad
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).clip(-50, 50)


def _band_member(band: np.ndarray, valid: np.ndarray, pitch: float, scales_um, ring_um,
                  backend: str):
    """centre-scale maps and the ring map for one band, via stroke_ring's masked pyramid."""
    v = valid.astype(np.float32)
    num = SR._Pyramid(np.asarray(band, dtype=np.float32) * v)
    den = SR._Pyramid(v)
    shape = band.shape
    at = {s_um: SR.masked_blur(num, den, float(s_um) / pitch, shape, backend)
          for s_um in (*scales_um, ring_um)}
    return at


def c1_map(layers_dir: str, um: float, f: int, ref_native: np.ndarray, backend: str = DEFAULT_BACKEND,
           gpu: int = 0, dz_um: float | None = None):
    """-> (c1_raw [pooled H,W], valid [pooled], ref [pooled], detail str). Ten fixed bands x
    five centre scales minus the 1012.48 um ring, robust-z per member, averaged."""
    if backend == "torch":
        return c1_map_torch(layers_dir, um, f, ref_native, gpu, dz_um=dz_um)
    bands, valid0, detail = SR.band_planes(layers_dir, dz_um=dz_um or um, offsets_um=C1_BAND_OFFSETS_UM,
                                            band_um=C1_BAND_UM)
    valid = _pool2(valid0.astype(np.float32), f) > 0.5
    ref = _pool2((ref_native & valid0).astype(np.float32), f) > 0.5
    if not ref.any():
        ref = valid
    pitch = um * f
    acc = np.zeros(valid.shape, dtype=np.float32)
    n = 0
    for band in bands:
        bp = _pool2(band, f)
        at = _band_member(bp, valid, pitch, C1_SCALES_UM, C1_RING_UM, backend)
        ring = at[C1_RING_UM]
        for s_um in C1_SCALES_UM:
            m = at[s_um] - ring
            acc += _rz(m, ref & valid)
            n += 1
    return acc / max(1, n), valid, ref, f"C1: {detail}, {n} members"


# ---- GPU (torch) backend -------------------------------------------------------------
# A direct port of the same closed form onto a CUDA device, for the "measure a GPU port"
# placement question (K1 is CPU-bound / memory-bandwidth-limited on a single segment, per
# stroke_ring's own note that threads buy almost nothing -- so the two real placement
# options are MORE PARALLEL SEGMENTS PER CPU HOST, or ONE SEGMENT FASTER ON A GPU). This
# mirrors `stroke_ring._score_torch`'s pyramid/blur pattern; NOT checked bit-for-bit against
# the CPU path the way `stroke_ring_verify.py` checks stroke_ring's -- see the dated
# FINDINGS entry for the agreement actually measured before this is proposed for production.
def _torch_or_none():
    try:
        import torch
        return torch
    except ImportError:
        return None


def _t_pyr(t, level, Fnn):
    while level > 1:
        ph, pw = (-t.shape[-2]) % 2, (-t.shape[-1]) % 2
        if ph or pw:
            t = Fnn.pad(t, (0, pw, 0, ph))
        t = Fnn.avg_pool2d(t, 2)
        level //= 2
    return t


def _t_gblur(t, s, torch, Fnn):
    if s < 0.5:
        return t
    r = max(1, int(math.ceil(4 * s)))
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=t.device)
    k = torch.exp(-0.5 * (x / s) ** 2)
    k = k / k.sum()
    t = Fnn.conv2d(Fnn.pad(t, (r, r, 0, 0)), k.view(1, 1, 1, -1))
    return Fnn.conv2d(Fnn.pad(t, (0, 0, r, r)), k.view(1, 1, -1, 1))


def _t_masked_at(x4, v4, sig_list, pitch, torch, Fnn):
    """x4, v4: [1,1,H,W] on device. -> {s_um: [1,1,H,W]} masked-blur maps at base res."""
    H, W = x4.shape[-2:]
    out = {}
    for s_um in sig_list:
        s_px = float(s_um) / pitch
        L = SR._level_for(s_px)
        nb = _t_gblur(_t_pyr(x4, L, Fnn), s_px / L, torch, Fnn)
        db = _t_gblur(_t_pyr(v4, L, Fnn), s_px / L, torch, Fnn)
        r = nb / db.clamp_min(1e-6)
        if L > 1:
            r = Fnn.interpolate(r, scale_factor=L, mode="bilinear", align_corners=False)[..., :H, :W]
        out[s_um] = r
    return out


def _rz_torch(v, ref, torch):
    r = v[ref]
    r = r[torch.isfinite(r)]
    if r.numel() == 0:
        return torch.zeros_like(v)
    med = torch.median(r)
    mad = torch.median((r - med).abs()) * 1.4826
    if float(mad) < 1e-9:
        sd = r.std()
        mad = sd if float(sd) > 1e-9 else torch.tensor(1.0, device=v.device)
    out = (v - med) / mad
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).clamp(-50, 50)


def c1_map_torch(layers_dir: str, um: float, f: int, ref_native: np.ndarray, gpu: int = 0,
                 dz_um: float | None = None):
    torch = _torch_or_none()
    if torch is None:
        raise RuntimeError("k1 torch backend needs torch installed on this interpreter")
    import torch.nn.functional as Fnn
    dev = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    bands, valid0, detail = SR.band_planes(layers_dir, dz_um=dz_um or um, offsets_um=C1_BAND_OFFSETS_UM,
                                            band_um=C1_BAND_UM)
    valid = _pool2(valid0.astype(np.float32), f) > 0.5
    ref = _pool2((ref_native & valid0).astype(np.float32), f) > 0.5
    if not ref.any():
        ref = valid
    pitch = um * f
    vt = torch.from_numpy(valid.astype(np.float32)).to(dev)[None, None]
    reft = torch.from_numpy(ref & valid).to(dev)
    acc = torch.zeros(valid.shape, device=dev)
    n = 0
    for band in bands:
        bp = _pool2(band, f)
        xt = torch.from_numpy(bp.astype(np.float32)).to(dev)[None, None] * vt
        at = _t_masked_at(xt, vt, (*C1_SCALES_UM, C1_RING_UM), pitch, torch, Fnn)
        ring = at[C1_RING_UM][0, 0]
        for s_um in C1_SCALES_UM:
            m = at[s_um][0, 0] - ring
            acc += _rz_torch(m, reft, torch)
            n += 1
    return acc.cpu().numpy() / max(1, n), valid, ref, f"C1(torch,{dev}): {detail}, {n} members"


def _find_edges(Ps: np.ndarray):
    """Ps [Z,H,W] laterally-smoothed depth profile -> (peak, lo, hi) plane-index (float,
    sub-plane, LOCAL to Ps's own z=0), sub-plane linear interpolation. Identical algorithm
    to `featbank_crevice.edges`."""
    Z, H, W = Ps.shape
    j = Ps.argmax(0).astype(np.float32)
    pk = Ps.max(0)
    base = Ps.min(0)
    half = base + 0.5 * (pk - base)
    ar = np.arange(Z, dtype=np.int64).reshape(Z, 1, 1)
    above = Ps >= half[None]
    big = np.full((Z, H, W), Z, dtype=np.int64)
    lo_i = np.where(above, ar, big).min(0)
    neg1 = np.full((Z, H, W), -1, dtype=np.int64)
    hi_i = np.where(above, ar, neg1).max(0)

    def interp(i_in, i_out):
        i_in = np.clip(i_in, 0, Z - 1)
        i_out = np.clip(i_out, 0, Z - 1)
        a = np.take_along_axis(Ps, i_in[None], axis=0)[0]
        b = np.take_along_axis(Ps, i_out[None], axis=0)[0]
        denom = np.clip(a - b, 1e-6, None)
        t = np.clip((a - half) / denom, 0.0, 1.0)
        return i_in.astype(np.float32) + t * (i_out.astype(np.float32) - i_in.astype(np.float32))

    lo = interp(lo_i, lo_i - 1)
    hi = interp(hi_i, hi_i + 1)
    return j, lo, hi


def _sample_depth(P: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """P [Z,H,W], idx [H,W] continuous plane index (LOCAL to P) -> linear interpolation."""
    Z = P.shape[0]
    idx = np.clip(idx, 0.0, float(Z - 1))
    j = np.clip(np.floor(idx).astype(np.int64), 0, Z - 2)
    t = idx - j
    a = np.take_along_axis(P, j[None], axis=0)[0]
    b = np.take_along_axis(P, (j + 1)[None], axis=0)[0]
    return a * (1.0 - t) + b * t


def edge_map(layers_dir: str, um: float, f: int, ref_native: np.ndarray,
             half_window_um: float = EDGE_HALF_WINDOW_UM_DEFAULT, backend: str = DEFAULT_BACKEND,
             gpu: int = 0, dz_um: float | None = None):
    """-> (E_raw [pooled H,W], valid [pooled], ref [pooled], detail str). The grey value
    EDGE_INSIDE_UM inside the high-index half-max sheet edge, pooled EDGE_CENTER_UM minus a
    EDGE_SURR_UM surround."""
    if backend == "torch":
        return edge_map_torch(layers_dir, um, f, ref_native, half_window_um, gpu, dz_um=dz_um)
    files = SR._layer_files(layers_dir)
    n = len(files)
    if not n:
        raise ValueError(f"no rendered layers in {layers_dir}")
    zc = n // 2
    dz = float(dz_um or um)                  # DEPTH pitch: a render's planes are one CT voxel apart
    half_planes = int(math.ceil(half_window_um / dz))
    z0, z1 = max(0, zc - half_planes), min(n - 1, zc + half_planes)
    clipped = (z0 != zc - half_planes) or (z1 != zc + half_planes)
    P = np.stack([_pool2(SR._read(layers_dir, files, k), f) for k in range(z0, z1 + 1)]).astype(np.float32)
    valid = P[zc - z0] > 0
    vf = valid.astype(np.float32)
    pitch = um * f
    den = SR._Pyramid(vf)
    sig_px = EDGE_SMOOTH_UM / pitch
    Ps = np.empty_like(P)
    for k in range(P.shape[0]):
        num = SR._Pyramid(P[k] * vf)
        Ps[k] = SR.masked_blur(num, den, sig_px, vf.shape, backend)
    _pk, _lo, hi = _find_edges(Ps)
    del Ps
    inside_planes = EDGE_INSIDE_UM / dz
    e_raw = _sample_depth(P, hi - inside_planes) * valid
    num = SR._Pyramid(e_raw.astype(np.float32) * vf)
    c = SR.masked_blur(num, den, EDGE_CENTER_UM / pitch, vf.shape, backend)
    s = SR.masked_blur(num, den, EDGE_SURR_UM / pitch, vf.shape, backend)
    e = c - s
    ref = _pool2((ref_native & valid).astype(np.float32), f) > 0.5 if ref_native.shape == valid.shape \
        else valid
    if not ref.any():
        ref = valid
    detail = (f"E: {z1 - z0 + 1} of {n} planes read (z{z0}..z{z1}, zc={zc}), "
              f"half-window {half_window_um:g} um" + (" CLIPPED to the stack" if clipped else ""))
    if clipped:
        print(f"[k1] {detail} -- edge search window narrower than requested; sheet-edge "
              f"localisation may be degraded on a short stack", flush=True)
    return e, valid, ref, detail


def edge_map_torch(layers_dir: str, um: float, f: int, ref_native: np.ndarray,
                    half_window_um: float = EDGE_HALF_WINDOW_UM_DEFAULT, gpu: int = 0,
                    dz_um: float | None = None):
    torch = _torch_or_none()
    if torch is None:
        raise RuntimeError("k1 torch backend needs torch installed on this interpreter")
    import torch.nn.functional as Fnn
    dev = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    files = SR._layer_files(layers_dir)
    n = len(files)
    if not n:
        raise ValueError(f"no rendered layers in {layers_dir}")
    zc = n // 2
    dz = float(dz_um or um)                  # DEPTH pitch: a render's planes are one CT voxel apart
    half_planes = int(math.ceil(half_window_um / dz))
    z0, z1 = max(0, zc - half_planes), min(n - 1, zc + half_planes)
    clipped = (z0 != zc - half_planes) or (z1 != zc + half_planes)
    P = np.stack([_pool2(SR._read(layers_dir, files, k), f) for k in range(z0, z1 + 1)]).astype(np.float32)
    valid = P[zc - z0] > 0
    vf = valid.astype(np.float32)
    pitch = um * f
    vt = torch.from_numpy(vf).to(dev)[None, None]
    sig_px = EDGE_SMOOTH_UM / pitch
    L = SR._level_for(sig_px)
    Ps = np.empty_like(P)
    for k in range(P.shape[0]):
        xt = torch.from_numpy(P[k]).to(dev)[None, None] * vt
        nb = _t_gblur(_t_pyr(xt, L, Fnn), sig_px / L, torch, Fnn)
        db = _t_gblur(_t_pyr(vt, L, Fnn), sig_px / L, torch, Fnn)
        r = nb / db.clamp_min(1e-6)
        if L > 1:
            r = Fnn.interpolate(r, scale_factor=L, mode="bilinear",
                                align_corners=False)[..., :xt.shape[-2], :xt.shape[-1]]
        Ps[k] = r[0, 0].cpu().numpy()
    _pk, _lo, hi = _find_edges(Ps)
    del Ps
    inside_planes = EDGE_INSIDE_UM / dz
    e_raw = _sample_depth(P, hi - inside_planes) * valid
    et = torch.from_numpy(e_raw.astype(np.float32)).to(dev)[None, None] * vt
    at = _t_masked_at(et, vt, (EDGE_CENTER_UM, EDGE_SURR_UM), pitch, torch, Fnn)
    e = (at[EDGE_CENTER_UM] - at[EDGE_SURR_UM])[0, 0].cpu().numpy()
    ref = _pool2((ref_native & valid).astype(np.float32), f) > 0.5 if ref_native.shape == valid.shape \
        else valid
    if not ref.any():
        ref = valid
    detail = (f"E(torch,{dev}): {z1 - z0 + 1} of {n} planes read (z{z0}..z{z1}, zc={zc}), "
              f"half-window {half_window_um:g} um" + (" CLIPPED to the stack" if clipped else ""))
    return e, valid, ref, detail


from .colmean import plane_pitch_um  # noqa: E402  (shared: the CT voxel between planes)


def compute_k1(layers_dir: str, um: float, ref_native: np.ndarray, f: int = DEFAULT_LATERAL_POOL,
               half_window_um: float = EDGE_HALF_WINDOW_UM_DEFAULT, backend: str = DEFAULT_BACKEND,
               gpu: int = 0, dz_um: float | None = None):
    """-> (k1 [pooled H,W], native_shape, f, detail). The float score at the WORKING (pooled)
    pitch; `predict()` upsamples it back to the render's native grid for publication."""
    c1, valid_c, ref_c, d1 = c1_map(layers_dir, um, f, ref_native, backend, gpu, dz_um=dz_um)
    e, valid_e, ref_e, d2 = edge_map(layers_dir, um, f, ref_native, half_window_um, backend, gpu, dz_um=dz_um)
    H = min(c1.shape[0], e.shape[0])
    W = min(c1.shape[1], e.shape[1])
    c1, e = c1[:H, :W], e[:H, :W]
    valid = valid_c[:H, :W] & valid_e[:H, :W]
    ref = (ref_c[:H, :W] & valid) if ref_c.any() else valid
    z_c1 = _rz(c1, ref)
    z_e = _rz(e, ref)
    k1 = 0.5 * (z_c1 + z_e)
    return k1, valid, f"{d1}; {d2}"


# ---- production entry point --------------------------------------------------------------
def predict(layers_dir: str, mask_png: str, out_png: str, gpu: int = 0, arm: str | None = None,
            fleet=None, log: str | None = None, um_per_px: float | None = None,
            lateral_pool: int = DEFAULT_LATERAL_POOL, backend: str | None = None,
            plane_um: float | None = None, **spec) -> bool:
    arm = arm or default_arm()
    ok, why = available(fleet, arm)
    if not ok:
        _alerts.alert(f"k1 on this host: {why}")
        return False
    import time
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    t0 = time.time()
    um = float(um_per_px) if um_per_px else render_um_per_px(layers_dir)
    from .. import memstack as _MS
    if not um_per_px and _MS.get(layers_dir) is None and not os.path.exists(os.path.join(layers_dir, "frame.json")):
        print(f"[k1] no frame.json beside {os.path.basename(layers_dir)}; using {um:g} um/px "
              f"-- pass um_per_px= if that is not this render's pitch", flush=True)
    # 🔴 THE DEPTH PITCH IS THE VOXEL, NOT THE IN-PLANE PITCH (fixed 2026-09-29). Every render
    # tier steps its layers ONE CT VOXEL apart along the normal (render_gpu: slice_step 1.0),
    # whatever --scale it renders at. K1 read `um` (7.91, the in-plane pitch of the Grand Prize
    # frame) as the plane pitch too, so on the 8.64 / 9.362 um scans every C1 band offset and
    # the E edge window/inside distance were 9-18 % shallower than FINDINGS 122 specifies.
    # (On the 7.91 um S1/S5 controls the two coincide, which is why the scores reproduced.)
    dz = float(plane_um) if plane_um else plane_pitch_um(layers_dir)
    if dz is None:
        dz = um
        _alerts.alert(f"k1: plane pitch of {os.path.basename(str(layers_dir).rstrip('/'))} unknown "
                      f"(no frame.json frame, not a registered view); assuming {um:g} um/plane")
    files = SR._layer_files(layers_dir)
    if not files:
        _alerts.alert(f"k1: no rendered layers in {layers_dir}")
        return False
    native_shape = SR._read(layers_dir, files, len(files) // 2).shape
    if os.path.exists(mask_png):
        m = np.asarray(Image.open(mask_png).convert("L"))
        if m.shape != native_shape:
            print(f"[k1] mask {m.shape} != render {native_shape}; scoring the whole plane", flush=True)
            m = np.ones(native_shape, dtype=np.uint8) * 255
    else:
        m = np.ones(native_shape, dtype=np.uint8) * 255
    ref_native = m > 0
    if not ref_native.any():
        print("[k1] empty mask; nothing to score", flush=True)
        return False
    f = int(round(float(lateral_pool))) or 1
    bk = backend or DEFAULT_BACKEND
    k1, valid_p, detail = compute_k1(layers_dir, um, ref_native, f=f, backend=bk, gpu=gpu, dz_um=dz)
    k1_native = SR._upsample(k1, native_shape, f)
    valid_native = SR._upsample(valid_p.astype(np.float32), native_shape, f) > 0.5
    keep_native = ref_native & valid_native
    calc_s = time.time() - t0
    Image.fromarray(SR.to_uint8(k1_native, (keep_native * 255).astype(np.uint8))).save(out_png)
    mp = float(keep_native.sum()) / 1e6
    print(f"[k1:{arm}] {detail}; working pitch {um * f:.2f} um/px (f={f}), planes {dz:.3f} um apart; {mp:.2f} Mpx scored, "
          f"{calc_s:.1f}s = {mp / max(calc_s, 1e-6):.3f} MVox/s [{bk}]", flush=True)
    return True
