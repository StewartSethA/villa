"""Label-free consistency checks for an umbilicus polyline (a scroll's axis centre, level-0 voxel coordinates).

WHY. An automatically detected umbilicus (see umbilicus_raycross.py) is validated against a human trace on
one scroll only. These checks say WHICH stretches of another scroll's curve deserve a human look, without
labels. A check that never fails on a deliberately corrupted control is not a check, so `corrupt()` and the
tests build shifted / jittered / drifted / stepped copies of a curve and require the checks to separate them
from the clean one (VALIDATION section of the README: what separates, and what does not).

TWO KINDS, and the difference matters:
  * CURVE checks (numpy on the polyline only, milliseconds): stored score, jitter about a local line,
    consecutive-point slope, agreement with another variant file, error against a human trace.
  * VOLUME checks (read the surface prediction and CT at one pyramid level, ~16 slices, plane reads only):
    the detector's OWN criterion is re-evaluated at the polyline. From the true umbilicus every ray leaves
    through every winding once, so the mean ray-crossing count is maximal there. We recompute the field in a
    window about the polyline and report (a) the offset from the polyline to the field's soft centroid,
    (b) the angular spread of the per-ray counts AT the polyline, (c) the offset to the centroid of the sheet
    mask (informational). These are independent of the STORED numbers, but NOT independent of the detector's
    principle: a scroll whose windings are not closed around the axis could fool both.

Tolerances are advisory flags, never gates. Units: distances in mm = voxels x voxel_um / 1000; voxel_um is a
required argument, never guessed.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

LEVEL = 2                     # the pyramid level the detector ran at
N_SLICES = 16
WINDOW_MM = 12.0              # half-width of the search window about the polyline
STRIDE_L2 = 8                 # candidate spacing inside the window, level-2 voxels
N_RAYS = 48

# Tolerances, DECLARED BEFORE the corruption experiment was run (2026-09-25): 3x an assumed detector median error
# of 0.80 mm for the offset (that figure was later found to rest on 6 slices; the 884-slice figure is 0.73 mm), and
# a jitter bound of 1.5 mm (the smoother's own span is ~5 mm). They are advisory flags, never gates.
# `offset_p90_mm` was added AFTER the corruption experiment (the p50 cannot see a local error: a 500 voxel step across
# a quarter of z left it at 1.3 mm while the p90 read 5.0-5.5 mm): 1.5x the largest clean-control p90 (3.27 mm) =
# 4.8 mm. It is fitted to the experiment, NOT pre-declared, and is not independent evidence of its own value.
TOL = {"offset_p50_mm": 2.4, "offset_p90_mm": 4.8, "jitter_p90_mm": 1.5, "agree_p90_mm": 3.0, "aniso_p50": 0.6}


# ----------------------------------------------------------------------------- loading
def md5_file(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def load_curve_file(path) -> dict:
    """{'z','x','y','score' (float arrays, level-0 voxels, sorted by z), 'fmt', 'prov'}. Understands three
    formats: `control_points` (the spiral fitter's umbilicus.json), `curve_dense` (z/y/x arrays) and the raw
    per-slice detections written by umbilicus_raycross.py detect."""
    d = json.loads(Path(path).read_text())
    if isinstance(d, dict) and "control_points" in d:
        cp = d["control_points"]
        z = np.array([p["z"] for p in cp], float)
        x = np.array([p["x"] for p in cp], float)
        y = np.array([p["y"] for p in cp], float)
        sc = np.array([p.get("score", np.nan) for p in cp], float)
        fmt = "control_points"
    elif isinstance(d, dict) and "points" in d and "scale" in d:
        pts = d["points"]
        s = float(d["scale"])
        z = np.array([p["z"] for p in pts], float)
        x = np.array([p["x_soft"] for p in pts], float) * s
        y = np.array([p["y_soft"] for p in pts], float) * s
        sc = np.array([p.get("prom", np.nan) for p in pts], float)
        fmt = "dense_detections"
    elif isinstance(d, dict) and {"z", "y", "x"} <= set(d):
        z, x, y = (np.array(d[k], float) for k in "zxy")
        sc = np.array(d.get("support", [np.nan] * len(z)), float)
        fmt = "curve_dense"
    else:
        raise ValueError(f"{path}: not an umbilicus file (keys {list(d)[:6] if isinstance(d, dict) else type(d)})")
    o = np.argsort(z, kind="stable")
    return {"z": z[o], "x": x[o], "y": y[o], "score": sc[o], "fmt": fmt, "prov": d.get("_provenance") if isinstance(d, dict) else None}


def xy_at(curve: dict, z) -> tuple[np.ndarray, np.ndarray]:
    zz = np.asarray(z, float)
    return np.interp(zz, curve["z"], curve["x"]), np.interp(zz, curve["z"], curve["y"])


# ----------------------------------------------------------------------------- curve checks
def _pcts(a, qs=(50, 90)) -> dict:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, **{f"p{q}": None for q in qs}, "max": None}
    return {"n": int(a.size), **{f"p{q}": float(np.percentile(a, q)) for q in qs}, "max": float(a.max())}


def jitter_mm(curve: dict, voxel_um: float, span_frac: float = 0.03) -> dict:
    """Leave-one-out residual of each point from a straight line fitted to its neighbours within +-3 % of the z span
    (about 5 mm on these scrolls). A smooth axis leaves ~0; per-point noise or a kink shows as the residual."""
    z, x, y = curve["z"], curve["x"], curve["y"]
    if len(z) < 6:
        return {"n": 0, "p50": None, "p90": None, "max": None, "note": "fewer than 6 points"}
    w = span_frac * (z[-1] - z[0])
    res = []
    for i in range(len(z)):
        m = (np.abs(z - z[i]) <= w) & (np.arange(len(z)) != i)
        if m.sum() < 4:
            continue
        a = np.polyfit(z[m], x[m], 1)
        b = np.polyfit(z[m], y[m], 1)
        res.append(np.hypot(np.polyval(a, z[i]) - x[i], np.polyval(b, z[i]) - y[i]) * voxel_um / 1000.0)
    out = _pcts(res)
    out["window_z_vox"] = float(w)
    return out


def slope_stats(curve: dict) -> dict:
    """Lateral displacement per unit z between consecutive points (dimensionless): the axis tilts slowly."""
    dz = np.diff(curve["z"])
    ok = dz > 0
    s = np.hypot(np.diff(curve["x"]), np.diff(curve["y"]))[ok] / dz[ok]
    return _pcts(s)


def score_stats(curve: dict, low: float = 10.0) -> dict:
    s = curve["score"][np.isfinite(curve["score"])]
    if s.size == 0:
        return {"n": 0, "median": None, "p10": None, "n_low": None, "low_thr": low}
    return {"n": int(s.size), "median": float(np.median(s)), "p10": float(np.percentile(s, 10)),
            "n_low": int((s < low).sum()), "low_thr": low}


def agreement_mm(a: dict, b: dict, voxel_um: float) -> dict:
    """Lateral distance from each point of `a` to `b` interpolated at the same z, where b covers it."""
    m = (a["z"] >= b["z"][0]) & (a["z"] <= b["z"][-1])
    if m.sum() == 0:
        return {"n": 0, "p50": None, "p90": None, "max": None}
    bx, by = xy_at(b, a["z"][m])
    d = np.hypot(a["x"][m] - bx, a["y"][m] - by) * voxel_um / 1000.0
    out = _pcts(d)
    out["overlap_frac"] = float(m.mean())
    return out


def reference_error_mm(curve: dict, ref: dict, voxel_um: float) -> dict:
    """Error of `curve` against a HUMAN trace: at each reference point's z, distance to the curve. n = reference points."""
    m = (ref["z"] >= curve["z"][0]) & (ref["z"] <= curve["z"][-1])
    cx, cy = xy_at(curve, ref["z"][m])
    d = np.hypot(cx - ref["x"][m], cy - ref["y"][m]) * voxel_um / 1000.0
    out = _pcts(d)
    out["n_ref_total"] = int(len(ref["z"]))
    return out


def curve_checks(curve: dict, voxel_um: float | None) -> dict:
    """Everything that needs only the polyline. voxel_um None -> distances stay in voxels (unit stated)."""
    vu = voxel_um if voxel_um else None
    k = (vu or 1000.0)
    unit = "mm" if vu else "voxel/1000 (voxel size unknown: NOT mm)"
    return {"n_points": int(len(curve["z"])), "z_min": float(curve["z"][0]), "z_max": float(curve["z"][-1]),
            "z_span_mm": None if not vu else float((curve["z"][-1] - curve["z"][0]) * vu / 1000.0),
            "median_dz_vox": float(np.median(np.diff(curve["z"]))) if len(curve["z"]) > 1 else None,
            "score": score_stats(curve), "jitter": jitter_mm(curve, k), "slope": slope_stats(curve),
            "unit": unit, "voxel_um": vu}


# ----------------------------------------------------------------------------- volume access (plane reads)
def read_plane(zarr_dir, level: int, zl: int) -> np.ndarray:
    import zarr
    return np.asarray(zarr.open(str(Path(zarr_dir) / str(level)), mode="r")[int(zl)])


def level_shape(zarr_dir, level: int):
    try:
        return tuple(json.loads((Path(zarr_dir) / str(level) / ".zarray").read_text())["shape"])
    except (OSError, ValueError, KeyError):
        return None


def masks_at(ct, pred, zl: int, level: int = LEVEL, cache_dir: Path | None = None):
    """(sheet, body) boolean planes at pyramid `level`, slice index zl: body = CT > 5, sheet = pred > 0 & body.
    `ct` and `pred` are OME-Zarr roots (one array per pyramid level). If `cache_dir` is given the planes are cached
    there as packed bits (about 0.5 MB per plane)."""
    cd = cache_dir
    f = None
    if cd is not None:
        cd.mkdir(parents=True, exist_ok=True)
        f = cd / f"L{level}_z{zl}.npz"
        if f.exists():
            try:
                d = np.load(f)
                sh = tuple(d["shape"])
                n = sh[0] * sh[1]
                return (np.unpackbits(d["sheet"])[:n].reshape(sh).astype(bool), np.unpackbits(d["body"])[:n].reshape(sh).astype(bool))
            except Exception:                           # noqa: BLE001,S110 - a corrupt cache file is rebuilt below
                pass
    body = read_plane(ct, level, zl) > 5
    sheet = (read_plane(pred, level, zl) > 0) & body
    if f is not None:
        np.savez_compressed(f, shape=np.array(body.shape), sheet=np.packbits(sheet), body=np.packbits(body))
    return sheet, body


# ----------------------------------------------------------------------------- the detector's criterion, re-evaluated
def _ray_table(max_r: int, n_rays: int):
    ang = np.arange(n_rays) * (2 * np.pi / n_rays)
    r = np.arange(1, max_r + 1, dtype=np.float32)
    return (np.rint(np.sin(ang)[:, None] * r[None]).astype(np.int32), np.rint(np.cos(ang)[:, None] * r[None]).astype(np.int32))


def ray_counts(sheet: np.ndarray, cy: np.ndarray, cx: np.ndarray, n_rays: int = N_RAYS, batch: int = 96) -> np.ndarray:
    """[n_candidates, n_rays] number of sheet entries (0 -> 1 rises) along each ray, nearest-voxel sampling (same
    definition as umbilicus_raycross.ray_counts, evaluated on CPU)."""
    H, W = sheet.shape
    max_r = int(np.hypot(H, W) / 2) + 2
    dy, dx = _ray_table(max_r, n_rays)
    flat = sheet.reshape(-1)
    out = np.zeros((len(cy), n_rays), np.int32)
    for i in range(0, len(cy), batch):
        y = cy[i:i + batch, None, None].astype(np.int32) + dy[None]
        x = cx[i:i + batch, None, None].astype(np.int32) + dx[None]
        ok = (y >= 0) & (y < H) & (x >= 0) & (x < W)
        s = flat[np.where(ok, y * W + x, 0)] & ok
        prev = np.concatenate([np.zeros_like(s[..., :1]), s[..., :-1]], -1)
        out[i:i + batch] = (s & ~prev).sum(-1)
    return out


def _soft(v, cy, cx, n):
    idx = np.argsort(v)[-n:]
    w = v[idx] - v[idx].min() + 1e-6
    return float((cy[idx] * w).sum() / w.sum()), float((cx[idx] * w).sum() / w.sum())


def criterion_field(sheet: np.ndarray, body: np.ndarray, y0: float, x0: float, R: int, stride: int = STRIDE_L2, n_rays: int = N_RAYS):
    """The detector's criterion (mean ray crossings) on a stride grid inside the disc of radius R (level voxels) about (y0, x0),
    restricted to the scroll body. Returns (gy, gx, mean) or None when the window holds no body."""
    H, W = sheet.shape
    gy, gx = np.mgrid[int(y0) - R:int(y0) + R + 1:stride, int(x0) - R:int(x0) + R + 1:stride]
    gy, gx = gy.ravel(), gx.ravel()
    keep = (gy >= 0) & (gy < H) & (gx >= 0) & (gx < W) & (np.hypot(gy - y0, gx - x0) <= R)
    gy, gx = gy[keep], gx[keep]
    keep = body[gy, gx]
    gy, gx = gy[keep], gx[keep]
    if len(gy) < 30:
        return None
    return gy, gx, ray_counts(sheet, gy, gx, n_rays).mean(1)


def slice_check(sheet: np.ndarray, body: np.ndarray, y0: float, x0: float, voxel_um: float, level: int = LEVEL,
                window_mm: float = WINDOW_MM, stride: int = STRIDE_L2, n_rays: int = N_RAYS, field=None) -> dict:
    """One slice. (y0, x0) is the polyline's position in THIS level's voxels. Returns offsets in mm.
    `field` = a precomputed criterion_field on a window that CONTAINS this one (the corruption test reuses one big field)."""
    mm_per = voxel_um * (2 ** level) / 1000.0
    R = int(window_mm / mm_per)
    H, W = sheet.shape
    if body.sum() < 3000 or sheet.sum() < 500:
        return {"skipped": "slice has no scroll body / sheet (outside the scanned scroll)"}
    if field is None:
        field = criterion_field(sheet, body, y0, x0, R, stride, n_rays)
    if field is None:
        return {"skipped": "polyline window is outside the scroll body"}
    gy, gx, mean = field
    inw = np.hypot(gy - y0, gx - x0) <= R
    gy, gx, mean = gy[inw], gx[inw], mean[inw]
    if len(gy) < 30:
        return {"skipped": "polyline window is outside the scroll body"}
    top = max(10, int(0.02 * len(mean)))
    sy, sx = _soft(mean, gy, gx, top)
    yi, xi = int(round(min(max(y0, 0), H - 1))), int(round(min(max(x0, 0), W - 1)))
    at = ray_counts(sheet, np.array([yi]), np.array([xi]), n_rays)[0].astype(float)
    med = float(np.median(at))
    yy, xx = np.nonzero(sheet)
    return {"offset_mm": float(np.hypot(sy - y0, sx - x0) * mm_per),
            "offset_dx_mm": float((sx - x0) * mm_per), "offset_dy_mm": float((sy - y0) * mm_per),
            "crit_at_point": float(at.mean()), "crit_max_window": float(mean.max()),
            "crit_deficit": float(1.0 - at.mean() / max(mean.max(), 1e-6)),
            "aniso": float(at.std() / max(med, 1e-6)), "rayfrac_low": float((at < 0.5 * max(med, 1e-6)).mean()),
            "sheet_centroid_offset_mm": float(np.hypot(yy.mean() - y0, xx.mean() - x0) * mm_per),
            "point_inside_body": bool(body[yi, xi]), "n_candidates": int(len(gy))}


def pick_slices(curve: dict, n: int = N_SLICES) -> list[int]:
    """Level-0 z values, evenly spaced through the polyline's z-range (cell centres, so the two ends are not sampled)."""
    z0, z1 = curve["z"][0], curve["z"][-1]
    return [int(z0 + (i + 0.5) / n * (z1 - z0)) for i in range(n)]


def summarize_slices(rows: list[dict]) -> dict:
    ok = [r for r in rows if "offset_mm" in r]
    out = {"n_slices_requested": len(rows), "n_slices_used": len(ok),
           "n_skipped": len(rows) - len(ok), "skipped_reasons": sorted({r["skipped"] for r in rows if "skipped" in r})}
    for k in ("offset_mm", "aniso", "crit_deficit", "sheet_centroid_offset_mm", "rayfrac_low"):
        out[k] = _pcts([r[k] for r in ok])
    out["n_point_outside_body"] = int(sum(1 for r in ok if not r["point_inside_body"]))
    return out


def volume_checks(ct, pred, curve: dict, voxel_um: float, cache_dir: Path | None = None, zs: list[int] | None = None,
                  n_slices: int = N_SLICES, level: int = LEVEL, window_mm: float = WINDOW_MM, stride: int = STRIDE_L2) -> dict:
    """Re-evaluate the detector's criterion at the polyline on `n_slices` slices of the OME-Zarr roots `ct` and `pred`.
    Raises FileNotFoundError when the two levels cannot be paired (the caller records 'not measured' with the reason)."""
    a, b = level_shape(ct, level), level_shape(pred, level)
    if a is None or b is None or a != b:
        raise FileNotFoundError(f"CT level {level} shape {a} != prediction shape {b}: cannot pair them")
    scale = 2 ** level
    zs = zs or pick_slices(curve, n_slices)
    px, py = xy_at(curve, zs)
    rows, t0 = [], time.time()
    for z, x, y in zip(zs, px, py, strict=True):
        zl = min(int(z) // scale, a[0] - 1)
        try:
            sheet, body = masks_at(ct, pred, zl, level, cache_dir)
        except (FileNotFoundError, OSError, ValueError, KeyError) as e:
            rows.append({"z": int(z), "skipped": f"read failed: {type(e).__name__}: {e}"[:160]})
            continue
        r = slice_check(sheet, body, y / scale, x / scale, voxel_um, level, window_mm=window_mm, stride=stride)
        r["z"] = int(z)
        rows.append(r)
    return {"slices": rows, "summary": summarize_slices(rows), "seconds": round(time.time() - t0, 1),
            "params": {"level": level, "n_slices": len(zs), "window_mm": window_mm, "stride_vox": stride, "n_rays": N_RAYS,
                       "criterion": "mean 0->1 sheet crossings along n_rays rays; offset = polyline to soft centroid of top 2% of the window's field",
                       "ct": str(ct), "pred": str(pred)}}


def volume_checks_multi(ct, pred, curves: dict, voxel_um: float, cache_dir: Path | None, zs: list[int], extra_mm: float,
                        level: int = LEVEL, window_mm: float = WINDOW_MM, stride: int = STRIDE_L2) -> dict:
    """Same check for several polylines that share the slices `zs` (the corruption test): ONE criterion field per slice on a
    window widened by `extra_mm`, then each polyline is scored inside its own WINDOW_MM disc. Returns {name: volume_checks-like dict}."""
    a = level_shape(ct, level)
    if a is None or a != level_shape(pred, level):
        raise FileNotFoundError(f"CT level {level} shape {a} != prediction shape {level_shape(pred, level)}")
    scale = 2 ** level
    mm_per = voxel_um * scale / 1000.0
    Rb = int((window_mm + extra_mm) / mm_per)
    ref = next(iter(curves.values()))
    rows = {k: [] for k in curves}
    t0 = time.time()
    for z in zs:
        zl = min(int(z) // scale, a[0] - 1)
        sheet, body = masks_at(ct, pred, zl, level, cache_dir)
        rx, ry = xy_at(ref, [z])
        fld = None if (body.sum() < 3000 or sheet.sum() < 500) else criterion_field(sheet, body, ry[0] / scale, rx[0] / scale, Rb, stride)
        for name, c in curves.items():
            x, y = xy_at(c, [z])
            r = slice_check(sheet, body, y[0] / scale, x[0] / scale, voxel_um, level, window_mm=window_mm, stride=stride, field=fld)
            r["z"] = int(z)
            rows[name].append(r)
    return {k: {"slices": v, "summary": summarize_slices(v), "seconds": round(time.time() - t0, 1)} for k, v in rows.items()}


# ----------------------------------------------------------------------------- verdicts and provenance
def flags(curve_c: dict, vol: dict | None, agree: dict | None) -> list[str]:
    """Advisory flags against the declared TOL. Empty list = nothing tripped (NOT 'verified')."""
    out = []
    j = (curve_c or {}).get("jitter") or {}
    if j.get("p90") is not None and j["p90"] > TOL["jitter_p90_mm"]:
        out.append(f"jitter p90 {j['p90']:.2f} mm > {TOL['jitter_p90_mm']} mm")
    if agree and agree.get("flaggable", True) and agree.get("p90") is not None and agree["p90"] > TOL["agree_p90_mm"]:
        out.append(f"variant disagreement p90 {agree['p90']:.2f} mm > {TOL['agree_p90_mm']} mm")
    s = (vol or {}).get("summary") or {}
    o = (s.get("offset_mm") or {}).get("p50")
    if o is not None and o > TOL["offset_p50_mm"]:
        out.append(f"criterion offset p50 {o:.2f} mm > {TOL['offset_p50_mm']} mm")
    o9 = (s.get("offset_mm") or {}).get("p90")
    if o9 is not None and o9 > TOL["offset_p90_mm"]:
        out.append(f"criterion offset p90 {o9:.2f} mm > {TOL['offset_p90_mm']} mm (some slices are far off)")
    if s.get("n_slices_used") is not None and s.get("n_slices_requested") and s["n_slices_used"] < 0.75 * s["n_slices_requested"]:
        out.append(f"only {s['n_slices_used']} of {s['n_slices_requested']} slices could be checked (the polyline leaves the scroll body on the rest)")
    return out


def provenance_block() -> dict:
    """Where and when a set of numbers was produced, and by which version of THIS file (md5)."""
    return {"host": platform.node(), "utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "module": Path(__file__).name, "module_md5": md5_file(__file__)}


# ----------------------------------------------------------------------------- corruption (how the checks are tested)
def corrupt(curve: dict, kind: str, a: float, b: float = 0.0, seed: int = 1) -> dict:
    """A deliberately wrong copy of `curve` (level-0 voxels): shift x by a and y by b; linear drift 0 -> a over z; a step of
    `a` across the middle quarter of z; per-point Gaussian jitter of sigma `a` on each axis (seeded)."""
    o = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in curve.items()}
    z = o["z"]
    t = (z - z[0]) / (z[-1] - z[0])
    if kind == "shift":
        o["x"] = o["x"] + a
        o["y"] = o["y"] + b
    elif kind == "drift":
        o["x"] = o["x"] + a * t
    elif kind == "step":
        o["x"] = np.where((t > 0.375) & (t < 0.625), o["x"] + a, o["x"])
    elif kind == "jitter":
        r = np.random.default_rng(seed)
        o["x"] = o["x"] + r.normal(0, a, len(z))
        o["y"] = o["y"] + r.normal(0, a, len(z))
    else:
        raise ValueError(kind)
    return o


# ----------------------------------------------------------------------------- command line
def main(argv=None) -> int:
    import argparse
    import sys
    ap = argparse.ArgumentParser(description="Label-free checks of an umbilicus.json (curve + optional volume).")
    ap.add_argument("curve", help="umbilicus.json (control_points), curve_dense, or raw detections")
    ap.add_argument("--voxel-um", type=float, required=True, help="voxel pitch of the level-0 volume, um (never guessed)")
    ap.add_argument("--ct", help="CT OME-Zarr root (enables the volume checks; needs --pred)")
    ap.add_argument("--pred", help="surface-prediction OME-Zarr root")
    ap.add_argument("--level", type=int, default=LEVEL)
    ap.add_argument("--variant", help="a second umbilicus file to compare with")
    ap.add_argument("--reference", help="a human trace to report the error against")
    ap.add_argument("--out", help="write the JSON result here")
    a = ap.parse_args(argv)
    curve = load_curve_file(a.curve)
    res = {"file": str(a.curve), "md5": md5_file(a.curve), "voxel_um": a.voxel_um, "curve": curve_checks(curve, a.voxel_um),
           "provenance": provenance_block()}
    agree = None
    if a.variant:
        agree = res["agreement"] = agreement_mm(curve, load_curve_file(a.variant), a.voxel_um)
    if a.reference:
        res["reference_error"] = reference_error_mm(curve, load_curve_file(a.reference), a.voxel_um)
    vol = None
    if a.ct and a.pred:
        try:
            vol = res["volume"] = volume_checks(a.ct, a.pred, curve, a.voxel_um, level=a.level)
        except FileNotFoundError as e:
            res["volume"] = {"not_measured": str(e)}
    res["flags"] = flags(res["curve"], vol, agree)
    txt = json.dumps(res, indent=1, default=float)
    if a.out:
        Path(a.out).write_text(txt)
    print(txt if not a.out else f"wrote {a.out}; flags: {res['flags'] or 'none tripped (NOT the same as verified)'}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
