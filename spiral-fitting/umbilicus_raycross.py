#!/usr/bin/env python3
"""Estimate a scroll's umbilicus from its surface prediction by counting ray crossings.

The criterion
-------------
Every winding of a rolled sheet is a closed curve about the umbilicus, so a ray
from the true umbilicus to the outside crosses every winding exactly once,
whatever the deformation. For each candidate centre in an axial (xy) slice we
cast ``n_rays`` rays and count how many times each one rises from "not sheet" to
"sheet". The umbilicus maximises the MEAN crossing count over the rays. This is
indifferent to the long, near-parallel stretches of lamination that make a
normal-convergence estimate ridge along the common normal direction.

Two subcommands, run in this order::

    umbilicus_raycross.py detect --pred SURF.zarr [--ct CT.zarr] --level 1 \\
            --z-step 40 --out detections.json
    umbilicus_raycross.py build  detections.json --out umbilicus.json

``detect`` needs a GPU for realistic sizes (``--device cpu`` works and is only
practical for tiny inputs, or at coarse pyramid levels). ``build`` is CPU only.

The output is ``umbilicus.json`` in the schema ``umbilicus.json_umbilicus_z_to_yx``
reads: ``control_points`` of integer LEVEL-0 voxel indices ``{x, y, z, score}``.
``score`` is the estimate's own confidence (1..99). It is deliberately capped
below 100, the value the hand-placed traces use, so a machine estimate cannot be
mistaken for a human one.

See README.md ("Estimating an umbilicus automatically") for the measured
accuracy, its limits, and the provenance of every number quoted there.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time

import numpy as np
from scipy import ndimage as ndi

TOOL_VERSION = "1"


# --------------------------------------------------------------------------- #
# Ray crossings
# --------------------------------------------------------------------------- #

def ray_counts(sheet, body, cand_yx, n_rays=96, max_r=None, step=1.0,
               device="cuda:0", batch=512):
    """Sheet crossings along ``n_rays`` rays from each candidate centre.

    sheet, body : (H, W) bool/uint8 masks. ``body`` is the scroll cross-section
                  (used only for a diagnostic; the count is over ``sheet``).
    cand_yx     : (N, 2) float candidate centres, (row, col).
    Returns (rises [N, n_rays], body_samples [N, n_rays]). A crossing is a
    sample where the sheet mask is set and the previous sample along the ray
    was not; sampling is nearest-neighbour, one sample per ``step`` pixels.
    """
    import torch

    H, W = sheet.shape
    if max_r is None:
        max_r = int(np.hypot(H, W) / 2) + 2
    S = torch.tensor(sheet.astype(np.float32), device=device)[None, None]
    B = torch.tensor(body.astype(np.float32), device=device)[None, None]
    ang = torch.arange(n_rays, device=device, dtype=torch.float32) * (2 * np.pi / n_rays)
    dy = torch.sin(ang)
    dx = torch.cos(ang)
    rs = torch.arange(1, int(max_r / step) + 1, device=device, dtype=torch.float32) * step
    cy = torch.tensor(cand_yx[:, 0], dtype=torch.float32, device=device)
    cx = torch.tensor(cand_yx[:, 1], dtype=torch.float32, device=device)
    outs = []
    for i in range(0, len(cy), batch):
        by, bx = cy[i:i + batch], cx[i:i + batch]
        Y = by[:, None, None] + dy[None, :, None] * rs[None, None, :]      # [b, rays, steps]
        X = bx[:, None, None] + dx[None, :, None] * rs[None, None, :]
        gy = (Y / (H - 1)) * 2 - 1
        gx = (X / (W - 1)) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1).reshape(1, -1, 1, 2)
        s = torch.nn.functional.grid_sample(S, grid, mode="nearest",
                                            align_corners=True, padding_mode="zeros")
        b = torch.nn.functional.grid_sample(B, grid, mode="nearest",
                                            align_corners=True, padding_mode="zeros")
        s = s.reshape(len(by), n_rays, -1)
        b = b.reshape(len(by), n_rays, -1)
        prev = torch.cat([torch.zeros_like(s[..., :1]), s[..., :-1]], -1)
        rises = ((s > 0.5) & (prev <= 0.5)).float().sum(-1)                  # [b, rays]
        outs.append(torch.stack([rises, b.sum(-1)], dim=0).cpu())
    r = torch.cat([o[0] for o in outs], 0).numpy()
    inb = torch.cat([o[1] for o in outs], 0).numpy()
    return r, inb


def body_from_pred(pred_mask, close_iter=6):
    """Scroll cross-section from the surface prediction alone: close, fill
    holes, keep the largest component. Used when no CT is supplied."""
    m = ndi.binary_closing(pred_mask, iterations=close_iter, border_value=0)
    m = ndi.binary_fill_holes(m)
    lab, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (int(np.argmax(sizes)) + 1)
    return m


# --------------------------------------------------------------------------- #
# Per-slice estimate, with the diagnostics the confidence is built from
# --------------------------------------------------------------------------- #

def _soft(v, cys, cxs, frac, n=None):
    n = int(n) if n else max(20, int(frac * len(v)))
    n = min(n, len(v))
    idx = np.argsort(v)[-n:]
    w = v[idx] - v[idx].min() + 1e-6
    if w.sum() <= 0:
        w = np.ones_like(w)
    cy = float((cys[idx] * w).sum() / w.sum())
    cx = float((cxs[idx] * w).sum() / w.sum())
    spread = float(np.sqrt(((cys[idx] - cy) ** 2 + (cxs[idx] - cx) ** 2).mean()))
    return cy, cx, spread


def slice_stats(sheet, body, stride=4, n_rays=96, erode=6, min_body=3000,
                device="cuda:0", top_frac=0.01, batch=512, coarse=0, half=0):
    """Estimate the umbilicus in one slice; returns a dict, or None if the slice
    holds too little scroll to judge (an end-cap, a fragment).

    ``y_soft``/``x_soft`` (the score-weighted centroid of the top ``top_frac``
    of candidates) is the estimate. ``coarse > 0`` enables a two-stage search: a
    stride-``coarse`` sweep of the whole interior, then a stride-``stride`` sweep
    of a +/-``half`` box about the coarse centroid. Each slice is independent of
    its neighbours, so the search cannot drift. ``coarse=0`` is the single-stage
    reference.
    """
    if body.sum() < min_body or sheet.sum() < 500:
        return None
    cand = ndi.binary_erosion(body, iterations=erode)
    ay, ax = np.nonzero(cand)
    # top-N for the soft centroid is fixed by the FULL-body candidate count at
    # the working stride, so the two-stage search averages over exactly the same
    # number of candidates the single-stage search would have.
    nfull = int(((ay % stride == 0) & (ax % stride == 0)).sum())
    top_n = max(20, int(top_frac * nfull))
    if coarse:
        k = (ay % coarse == 0) & (ax % coarse == 0)
        c0y, c0x = ay[k].astype(np.float64), ax[k].astype(np.float64)
        if len(c0y) < 50:
            return None
        r0, _ = ray_counts(sheet, body, np.stack([c0y, c0x], 1).astype(np.float32),
                           n_rays=n_rays, device=device, batch=batch)
        m0 = r0.mean(1)
        med_full = float(np.median(m0))
        gy, gx, _ = _soft(m0, c0y, c0x, top_frac, n=max(20, int(top_frac * len(m0))))
        box = (np.abs(ay - gy) <= half) & (np.abs(ax - gx) <= half)
        ay, ax = ay[box], ax[box]
    else:
        med_full = None
    k = (ay % stride == 0) & (ax % stride == 0)
    cys, cxs = ay[k].astype(np.float64), ax[k].astype(np.float64)
    if len(cys) < 50:
        return None
    r, _ = ray_counts(sheet, body, np.stack([cys, cxs], 1).astype(np.float32),
                      n_rays=n_rays, device=device, batch=batch)
    mean = r.mean(1)
    j = int(np.argmax(mean))
    peak = float(mean.max())
    # the prominence reference must be the WHOLE-slice median, not the median
    # inside the refinement box, or the two-stage search reports a collapsed
    # prominence that is not comparable with the single-stage one.
    med = float(np.median(mean)) if med_full is None else med_full
    ys, xs, spread = _soft(mean, cys, cxs, top_frac, n=top_n)

    # Ray-subset agreement. Subsets must keep 360 degrees of coverage (the
    # topological argument needs rays in every direction), so CONTIGUOUS sectors
    # are not a valid subset; interleaved ones are.
    yA, xA, _ = _soft(r[:, 0::2].mean(1), cys, cxs, top_frac, n=top_n)
    yB, xB, _ = _soft(r[:, 1::2].mean(1), cys, cxs, top_frac, n=top_n)
    split_inter = float(np.hypot(yA - yB, xA - xB))
    qs = np.array([_soft(r[:, i::4].mean(1), cys, cxs, top_frac, n=top_n)[:2]
                   for i in range(4)])
    jack = float(np.sqrt(((qs - qs.mean(0)) ** 2).sum(1).mean()))

    # Angular profile AT the argmax candidate. At the true umbilicus every ray
    # crosses every winding once, so this vector should be flat; a damaged
    # sector shows as a run of low counts.
    prof = r[j].astype(float)
    pmed = float(np.median(prof))
    sd_at_peak = float(prof.std())
    rayfrac_low = float((prof < 0.5 * max(pmed, 1e-6)).mean())
    aniso = float(sd_at_peak / max(pmed, 1e-6))

    # Plateau geometry: the set of candidates that every winding encloses.
    pl = mean >= peak - 1.0
    npl = int(pl.sum())
    py, px = cys[pl], cxs[pl]
    pcy, pcx = float(py.mean()), float(px.mean())
    plat_rg = float(np.sqrt(((py - pcy) ** 2 + (px - pcx) ** 2).mean()))

    diag = float(np.hypot(*sheet.shape))
    return dict(y=float(cys[j]), x=float(cxs[j]), y_soft=ys, x_soft=xs,
                y_plat=pcy, x_plat=pcx,
                mean_peak=peak, mean_med=med,
                mean_p10=float(np.percentile(mean, 10)),
                spread=spread, split_inter=split_inter, jack=jack,
                sd_at_peak=sd_at_peak, aniso=aniso, rayfrac_low=rayfrac_low,
                ray_med=pmed, prof=[round(float(v), 2) for v in prof],
                plateau_n=npl, plateau_rg=plat_rg,
                prom=float((peak - med) / max(peak, 1e-6)),
                nbody=int(body.sum()), nsheet=int(sheet.sum()),
                ncand=int(len(cys)), ncand_full=nfull, top_n=int(top_n), diag=diag)


# --------------------------------------------------------------------------- #
# Curve: confidence weights and a robust smoother
# --------------------------------------------------------------------------- #

def robust_smooth(zs, vals, w0, frac=0.10, iters=4, sigma_reject=2.5, zeval=None):
    """Tricube-kernel weighted local-LINEAR smoother with IRLS outlier
    rejection (span = ``frac`` of the z range). Returns (values at ``zeval``,
    final per-point weights)."""
    zs = np.asarray(zs, float)
    vals = np.asarray(vals, float)
    w0 = np.asarray(w0, float).clip(1e-3, None)
    w = w0.copy()
    span = max(frac * (zs.max() - zs.min()), 1e-6)
    ze = zs if zeval is None else np.asarray(zeval, float)

    def eval_at(zq, w):
        out = np.empty(len(zq))
        for i, z in enumerate(zq):
            d = np.abs(zs - z)
            kk = np.clip(1.0 - (d / span) ** 3, 0, None) ** 3
            ww = kk * w
            if ww.sum() < 1e-9:
                out[i] = vals[int(np.argmin(d))]
                continue
            A = np.stack([np.ones_like(zs), zs - z], 1)
            AT = A.T * ww
            try:
                out[i] = np.linalg.solve(AT @ A, AT @ vals)[0]
            except np.linalg.LinAlgError:
                out[i] = np.average(vals, weights=ww)
        return out

    for _ in range(iters):
        fitted = eval_at(zs, w)
        resid = vals - fitted
        s = np.median(np.abs(resid - np.median(resid))) * 1.4826
        if s < 1e-9:
            break
        w = w0 / (1.0 + (resid / (sigma_reject * s)) ** 2)
    return eval_at(ze, w), w


def weights(pts, ref_q=75, s0=8.0, j0=12.0):
    """Per-slice confidence in [0, 1]; every factor is in [0, 1] and they multiply.

    prom   peak prominence (peak - median) / peak of the crossing-count field
    spread spatial spread of the top-1 % candidates, relative to the slice diagonal
    split  distance between the soft centroids of two interleaved 48-ray halves
    jack   scatter of the four interleaved 72-ray jackknife centroids
    count  crossing count relative to the scroll's own p75 (suppresses end-caps)
    """
    g = lambda k: np.array([p[k] for p in pts], float)  # noqa: E731
    f = np.clip(g("prom"), 0, 1)
    f = f * np.exp(-(g("spread") / (0.05 * g("diag"))) ** 2)
    f = f * np.exp(-(g("split_inter") / s0) ** 2)
    f = f * np.exp(-(g("jack") / j0) ** 2)
    mp = g("mean_peak")
    f = f * np.clip(mp / max(np.percentile(mp, ref_q), 1e-6), 0, 1) ** 2
    return f


# --------------------------------------------------------------------------- #
# umbilicus.json
# --------------------------------------------------------------------------- #

def build_control_points(z0, y0, x0, conf, shape_level0, score_floor=1, score_cap=99):
    """Level-0 float curve + confidence in [0, 1] -> integer control points.
    Scores are capped below 100 (the hand-placed convention). Duplicate z are
    dropped: the loader interpolates z -> (y, x) and needs strictly increasing z."""
    nz, ny, nx = shape_level0
    zs = np.clip(np.round(z0), 0, nz - 1).astype(int)
    ys = np.clip(np.round(y0), 0, ny - 1).astype(int)
    xs = np.clip(np.round(x0), 0, nx - 1).astype(int)
    scores = np.clip(np.round(np.asarray(conf, float) * score_cap),
                     score_floor, score_cap).astype(int)
    pts, seen = [], set()
    for z, y, x, s in zip(zs, ys, xs, scores):
        if z in seen:
            continue
        seen.add(z)
        pts.append({"x": int(x), "y": int(y), "z": int(z), "score": int(s)})
    pts.sort(key=lambda p: p["z"])
    return pts


def write_umbilicus(path, control_points, provenance):
    doc = {"control_points": control_points, "_provenance": provenance}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=2)
    os.replace(tmp, path)


def verify_roundtrip(path):
    """Load ``path`` with the fitter's own reader (umbilicus.py, next to this
    file) and return the max |interpolated - written| over the control points."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    from umbilicus import json_umbilicus_z_to_yx
    fn = json_umbilicus_z_to_yx(path, coordinate_scale=1.0)
    with open(path) as f:
        pts = json.load(f)["control_points"]
    zs = np.array([p["z"] for p in pts], float)
    want = np.array([(p["y"], p["x"]) for p in pts], float)
    return float(np.abs(fn(zs) - want).max()), len(pts)


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #

def _resolve_device(requested):
    import torch
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"WARNING: --device {requested} requested but CUDA is not available; "
              f"falling back to CPU. This is orders of magnitude slower.",
              file=sys.stderr)
        return "cpu"
    return requested


def cmd_detect(a):
    import zarr

    dev = _resolve_device(a.device)
    scale = 2 ** a.level
    pred = zarr.open(f"{a.pred}/{a.level}", mode="r")
    ct = zarr.open(f"{a.ct}/{a.level}", mode="r") if a.ct else None
    zn = pred.shape[0] if ct is None else min(pred.shape[0], ct.shape[0])
    print(f"# pred {pred.shape} ct {None if ct is None else ct.shape} level={a.level} "
          f"stride={a.stride} coarse={a.coarse} half={a.half} device={dev}", flush=True)
    if ct is None:
        print("# no --ct: scroll body derived from the prediction alone "
              "(body_from_pred)", flush=True)
    zl = sorted({z // scale for z in range(a.z0, a.z1 if a.z1 else zn * scale, a.z_step)})
    zl = [z for z in zl if 0 <= z < zn]
    print(f"# {len(zl)} level-{a.level} slices", flush=True)

    res, i, t_io, t_dev, t0 = [], 0, 0.0, 0.0, time.time()
    while i < len(zl):
        zb = (zl[i] // a.block) * a.block
        grp = [z for z in zl[i:] if z < zb + a.block]
        i += len(grp)
        t = time.time()
        ze = min(zb + a.block, zn)
        P = np.asarray(pred[zb:ze])
        C = np.asarray(ct[zb:ze]) if ct is not None else None
        t_io += time.time() - t
        for z in grp:
            t = time.time()
            if C is not None:
                body = C[z - zb] > a.ct_threshold
                sheet = (P[z - zb] > 0) & body
            else:
                sr = P[z - zb] > 0
                body = body_from_pred(sr)
                sheet = sr & body
            r = slice_stats(sheet, body, stride=a.stride, n_rays=a.rays, device=dev,
                            coarse=a.coarse, half=a.half)
            t_dev += time.time() - t
            if r is None:
                print(f"z0={z * scale} skip (too little scroll)", flush=True)
                continue
            r["z"] = int(z * scale)
            r["zl"] = int(z)
            res.append(r)
            print(f"z0={r['z']} soft=({r['y_soft']:.1f},{r['x_soft']:.1f}) "
                  f"mean={r['mean_peak']:.1f} prom={r['prom']:.3f} "
                  f"spread={r['spread']:.1f} split={r['split_inter']:.1f} "
                  f"jack={r['jack']:.1f}", flush=True)
    doc = {
        "tool": "umbilicus_raycross.py", "tool_version": TOOL_VERSION,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "level": a.level, "scale": scale, "shape_level": list(pred.shape),
        "shape_level0": [int(v) * scale for v in pred.shape],
        "params": {"stride": a.stride, "coarse": a.coarse, "half": a.half, "rays": a.rays,
                   "z0": a.z0, "z1": a.z1, "z_step": a.z_step, "ct_threshold": a.ct_threshold,
                   "device": dev, "block": a.block},
        "pred": os.path.abspath(a.pred) if "://" not in a.pred else a.pred,
        "ct": (os.path.abspath(a.ct) if "://" not in a.ct else a.ct) if a.ct else None,
        "body": "ct" if ct is not None else "from_pred",
        "points": res,
    }
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=1)
    print(f"# wrote {a.out} n={len(res)} io={t_io:.0f}s compute={t_dev:.0f}s "
          f"wall={time.time() - t0:.0f}s", flush=True)
    return 0


def cmd_build(a):
    with open(a.detections, "rb") as f:
        raw = f.read()
    d = json.loads(raw)
    pts, scale = d["points"], d["scale"]
    if len(pts) < 5:
        sys.exit(f"only {len(pts)} detections; cannot smooth a curve")
    z = np.array([p["z"] for p in pts], float)
    ys = np.array([p["y_soft"] for p in pts], float) * scale
    xs = np.array([p["x_soft"] for p in pts], float) * scale
    w = weights(pts)
    zeval = np.arange(int(z.min()), int(z.max()) + 1, a.z_step, dtype=float)
    fy, wfin = robust_smooth(z, ys, w, frac=a.span, zeval=zeval)
    fx, _ = robust_smooth(z, xs, w, frac=a.span, zeval=zeval)
    # Local support: the smoother's summed post-IRLS weight inside its span,
    # relative to the scroll's own median. This is the score: it says "weak
    # here, strong there", which a single global confidence cannot.
    span = a.span * (z.max() - z.min())
    support = np.array([(wfin * np.clip(1 - (np.abs(z - q) / span) ** 3, 0, None) ** 3).sum()
                        for q in zeval])
    support = support / np.median(support)
    conf = np.clip(support / max(np.percentile(support, 90), 1e-9), 0, 1)
    cps = build_control_points(zeval, fy, fx, conf, d["shape_level0"])
    prov = {
        "generated_by": "spiral-fitting/umbilicus_raycross.py build (AUTOMATIC, not hand-traced)",
        "tool_version": TOOL_VERSION,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "detections_file": os.path.abspath(a.detections),
        "detections_md5": hashlib.md5(raw).hexdigest(),
        "detections_created_utc": d.get("created_utc"),
        "detection": {k: d.get(k) for k in ("level", "scale", "shape_level0", "params",
                                             "pred", "ct", "body")},
        "n_detections": len(pts),
        "smoother": {"kernel": "tricube local-linear", "span_fraction_of_z": a.span,
                     "irls_passes": 4, "reject_sigma": 2.5, "evaluated_every_z": a.z_step},
        "index_space": "level 0 of the surface prediction; coordinate_scale 1.0",
        "score": "1..99, local support of the curve at that z (NOT the hand-placed 100)",
    }
    write_umbilicus(a.out, cps, prov)
    err, n = verify_roundtrip(a.out)
    print(f"wrote {a.out}: {n} control points; z {cps[0]['z']}-{cps[-1]['z']}; "
          f"loader round-trip max err {err:.6f}")
    if err > 1e-3:  # the loader interpolates in float32
        sys.exit("FAILED: written file does not round-trip through umbilicus.json_umbilicus_z_to_yx")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="per-slice detections from a surface prediction")
    d.add_argument("--pred", required=True, help="surface prediction OME-Zarr root (levels 0/1/2...)")
    d.add_argument("--ct", help="CT OME-Zarr root (body mask = CT > --ct-threshold); "
                                "omit to derive the body from the prediction")
    d.add_argument("--level", type=int, default=1, help="pyramid level to detect at (default 1)")
    d.add_argument("--z0", type=int, default=0, help="first slice, LEVEL-0 index")
    d.add_argument("--z1", type=int, default=0, help="end slice, LEVEL-0 index (0 = whole scroll)")
    d.add_argument("--z-step", type=int, default=40, help="slice spacing, level-0 slices")
    d.add_argument("--stride", type=int, default=8, help="candidate stride in level pixels")
    d.add_argument("--coarse", type=int, default=32,
                   help="stage-1 candidate stride in level pixels (0 = single stage)")
    d.add_argument("--half", type=int, default=256, help="stage-2 half box, level pixels")
    d.add_argument("--rays", type=int, default=96)
    d.add_argument("--ct-threshold", type=int, default=5)
    d.add_argument("--block", type=int, default=128, help="z slab read per zarr access")
    d.add_argument("--device", default="cuda:0")
    d.add_argument("--out", required=True)
    d.set_defaults(fn=cmd_detect)

    b = sub.add_parser("build", help="smooth detections into umbilicus.json")
    b.add_argument("detections")
    b.add_argument("--out", required=True)
    b.add_argument("--z-step", type=int, default=20, help="control-point spacing, level-0 slices")
    b.add_argument("--span", type=float, default=0.10, help="smoother span, fraction of z range")
    b.set_defaults(fn=cmd_build)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
