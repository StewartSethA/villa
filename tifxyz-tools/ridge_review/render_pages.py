"""Render the review images and the page manifest from the per-sample .npz inputs.

  python render_pages.py SAMPLES_DIR BUILT_JSON [BUILT_JSON ...] --out ridge_review/site [--seed 20260929]

Per sample, into site/img/<id>/:
  ct_u.png, ct_v.png     sections through the cell containing the surface normal (vertical axis,
                         +normal up) and one surface tangent (horizontal): the papyrus sheet the
                         surface claims to follow should run horizontally through the centre
  ct_p-4/ct_p0/ct_p+4    planes parallel to the surface, 4 voxels below / at / above it
  ct_xy.png              the plain axis-aligned CT slice at the cell's z, the view VC3D shows
  pred_*.png             the surface prediction for the same views, transparent red, shown
                         only when the reviewer asks for it
The surface's own trace (the lattice row / column through the cell) is drawn in cyan on the
sections, the surface's crossings of the xy slice (interpolated along lattice edges) as cyan dots, and the cell
as a yellow ring. Nothing on the images says whether the guard removed or kept the cell.
Writes site/samples.js (window.SAMPLES = [...], shuffled with the seed) so the page works from a
plain file:// clone with no server.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

UP_SEC, UP_PLANE, UP_XY = 6, 4, 3


def stretch(a, lo, hi):
    return np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)


def to_rgb(g, up):
    return Image.fromarray(g).resize((g.shape[1] * up, g.shape[0] * up), Image.NEAREST).convert("RGB")


def pred_rgba(p, up):
    a = np.zeros(p.shape + (4,), np.uint8)
    on = p > 0
    a[on] = (230, 40, 40, 110)
    return Image.fromarray(a, "RGBA").resize((p.shape[1] * up, p.shape[0] * up), Image.NEAREST)


def dots(img, pts, up, flip_y=False, h=None):
    d = ImageDraw.Draw(img)
    for x, y in pts:
        X = (x + 0.5) * up
        Y = ((h - 1 - y) + 0.5) * up if flip_y else (y + 0.5) * up
        d.ellipse([X - 1.5, Y - 1.5, X + 1.5, Y + 1.5], fill=(60, 220, 230))
    return img


def polyline(img, pts, up, flip_y=False, h=None):
    """The surface's own trace through a section: consecutive lattice cells joined, cyan."""
    d = ImageDraw.Draw(img)
    xy = [((x + 0.5) * up, (((h - 1 - y) if flip_y else y) + 0.5) * up) for x, y in pts]
    for a, b in zip(xy[:-1], xy[1:], strict=True):
        if a is not None and b is not None:
            d.line([a, b], fill=(60, 220, 230), width=2)
    return img


def ring(img, x, y, up, flip_y=False, h=None, r=7):
    d = ImageDraw.Draw(img)
    X = (x + 0.5) * up
    Y = ((h - 1 - y) + 0.5) * up if flip_y else (y + 0.5) * up
    d.ellipse([X - r, Y - r, X + r, Y + r], outline=(250, 210, 30), width=2)
    return img


def render(npz, outdir):
    z = np.load(npz)
    cs, ps = z["ct_stack"], z["pred_stack"]
    D, H = (cs.shape[0] - 1) // 2, (cs.shape[1] - 1) // 2
    c, n, u, v = z["frame"]
    lo, hi = np.percentile(cs, 1), np.percentile(cs, 99)
    P, PV = z["lattice_xyz"], z["lattice_valid"]
    rel = None
    del rel
    L = (P.shape[0] - 1) // 2
    def trace(line_pts, line_ok, along):
        seg, out = [], []
        for p, ok in zip(line_pts, line_ok, strict=True):
            if not ok:
                if len(seg) > 1: out.append(seg)
                seg = []; continue
            r = p - c
            seg.append((float(r @ along) + H, float(r @ n) + D))
        if len(seg) > 1: out.append(seg)
        return out
    outdir.mkdir(parents=True, exist_ok=True)
    # section (normal x u): rows = normal offset (flip so +normal is up), cols = u
    sec_u = cs[:, H, :]
    sec_v = cs[:, :, H]
    img = to_rgb(np.flipud(stretch(sec_u, lo, hi)), UP_SEC)
    for seg in trace(P[L], PV[L], u):          # the lattice ROW through the cell runs along u
        polyline(img, seg, UP_SEC, True, 2 * D + 1)
    ring(img, H, D, UP_SEC, True, 2 * D + 1).save(outdir / "ct_u.png")
    pred_rgba(np.flipud(ps[:, H, :]), UP_SEC).save(outdir / "pred_u.png")
    img = to_rgb(np.flipud(stretch(sec_v, lo, hi)), UP_SEC)
    for seg in trace(P[:, L], PV[:, L], v):    # the lattice COLUMN through the cell, projected on v
        polyline(img, seg, UP_SEC, True, 2 * D + 1)
    ring(img, H, D, UP_SEC, True, 2 * D + 1).save(outdir / "ct_v.png")
    pred_rgba(np.flipud(ps[:, :, H]), UP_SEC).save(outdir / "pred_v.png")
    for k in (-4, 0, 4):
        img = to_rgb(stretch(cs[D + k], lo, hi), UP_PLANE)
        ring(img, H, H, UP_PLANE).save(outdir / f"ct_p{k:+d}.png")
        pred_rgba(ps[D + k], UP_PLANE).save(outdir / f"pred_p{k:+d}.png")
    xy, pxy = z["ct_xy"], z["pred_xy"]
    C = xy.shape[0] // 2
    l2, h2 = np.percentile(xy, 1), np.percentile(xy, 99)
    img = to_rgb(stretch(xy, l2, h2), UP_XY)
    # where the surface crosses this z: interpolate along every lattice edge whose ends straddle c_z
    cross = []
    for A, B, ok in ((P[:, :-1], P[:, 1:], PV[:, :-1] & PV[:, 1:]), (P[:-1], P[1:], PV[:-1] & PV[1:])):
        za, zb = A[..., 2] - c[2], B[..., 2] - c[2]
        m = ok & (za * zb <= 0) & (za != zb)
        t = (za[m] / (za[m] - zb[m]))[:, None]
        cross += list(A[m][:, :2] + t * (B[m][:, :2] - A[m][:, :2]))
    dots(img, [(x - round(c[0]) + C, y - round(c[1]) + C) for x, y in cross], UP_XY)
    ring(img, c[0] - round(c[0]) + C, c[1] - round(c[1]) + C, UP_XY, r=9).save(outdir / "ct_xy.png")
    pred_rgba(pxy, UP_XY).save(outdir / "pred_xy.png")
    return {"ct_percentiles_1_99": [float(lo), float(hi)]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("samples")
    ap.add_argument("built", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260929)
    a = ap.parse_args(argv)
    out = Path(a.out)
    items = []
    for b in a.built:
        items += json.load(open(b))
    for it in items:
        extra = render(Path(a.samples) / f"{it['sample_id']}.npz", out / "img" / it["sample_id"])
        it.update(extra)
    random.Random(a.seed).shuffle(items)
    (out / "samples.js").write_text("window.SAMPLES = " + json.dumps(items, indent=0) + ";\n")
    print(len(items), "samples rendered")


if __name__ == "__main__":
    main()
