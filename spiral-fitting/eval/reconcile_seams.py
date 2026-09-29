#!/usr/bin/env python3
"""Reconcile the winding numbering of adjacent spiral-fit z-windows into scroll-long CHAINS.

Each z-window is an independent fit; its winding index is window-local, and adjacent windows
were measured to disagree by 1-2 indices at their seams (2026-09-25, PHerc0211). A single global
offset per seam is not enough (the residual at the best offset was 0.75-1.1 pitch), so this
matches PER WINDING over the shared z band:

  profile(w)  per-angle-bin (72 bins about the given umbilicus at each point's own z) median
              radius of winding w's lattice points inside the overlap band
  pitch       this pair's own median radial spacing between adjacent windings (|r(w+1)-r(w)|),
              measured, not assumed
  match       for winding a of the lower window, the upper-window winding b minimising the
              median |r_a - r_b| over shared bins; ACCEPTED when that residual < --max-res
              pitches AND the runner-up is worse by > --margin pitches AND the match is mutual
              (b's best is a). Everything else is left unmatched, never forced.

Accepted matches are joined across all seams (union-find) into chains; a chain is one physical
sheet followed through z. Output: per seam the offset histogram, accepted fraction and residual
spread; per chain its member windings; and a global id per (window, winding).

usage: reconcile_seams.py UMBILICUS.json OUT.json --windows TAG:Z0:Z1:MESHDIR [TAG:Z0:Z1:MESHDIR ...]

MESHDIR is one window fit's mesh directory holding `w*_spliced_*` tifxyz windings.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import tifffile

NBIN = 72


def profiles(meshdir: Path, cps, z0, z1):
    uz = np.array([p["z"] for p in cps], float)
    ux = np.array([p["x"] for p in cps], float)
    uy = np.array([p["y"] for p in cps], float)
    out = {}
    for d in sorted(meshdir.glob("w*_spliced_*")):
        m = re.match(r"w(\d{3})", d.name)
        X, Y, Z = (tifffile.imread(d / f"{c}.tif").astype(np.float64) for c in "xyz")
        V = (X != -1) & (Z > 0) & (Z >= z0) & (Z < z1)
        if V.sum() < 50:
            continue
        x, y, z = X[V], Y[V], Z[V]
        cx, cy = np.interp(z, uz, ux), np.interp(z, uz, uy)
        th = np.arctan2(y - cy, x - cx)
        r = np.hypot(x - cx, y - cy)
        b = ((th + np.pi) / (2 * np.pi) * NBIN).astype(int) % NBIN
        prof = np.full(NBIN, np.nan)
        for i in range(NBIN):
            s = b == i
            if s.sum() >= 3:
                prof[i] = np.median(r[s])
        out[int(m.group(1))] = prof
    return out


def pitch_of(P):
    ks = sorted(P)
    d = [np.nanmedian(np.abs(P[k + 1] - P[k])) for k in ks if k + 1 in P]
    d = [x for x in d if np.isfinite(x)]
    return float(np.median(d)) if d else float("nan")


def dist(pa, pb):
    dd = np.abs(pa - pb)
    dd = dd[np.isfinite(dd)]
    return float(np.median(dd)) if dd.size >= 12 else np.inf


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("umbilicus", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--windows", nargs="+", required=True)
    ap.add_argument("--max-res", type=float, default=0.35)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--kmax", type=int, default=8)
    a = ap.parse_args(argv)
    umb = str(a.umbilicus)
    cps = sorted(json.loads(a.umbilicus.read_text())["control_points"], key=lambda p: p["z"])
    wins = []
    for w in a.windows:
        tag, z0, z1, mesh = w.split(":", 3)
        mesh = Path(mesh)
        wins.append((tag, int(z0), int(z1), mesh))
    wins.sort(key=lambda t: t[1])
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    seams = []
    for (ta, a0, a1, ma), (tb, b0, b1, mb) in zip(wins, wins[1:], strict=False):
        z0, z1 = max(a0, b0), min(a1, b1)
        if z1 <= z0:
            seams.append({"a": ta, "b": tb, "error": "no z overlap"})
            continue
        A, B = profiles(ma, cps, z0, z1), profiles(mb, cps, z0, z1)
        pitch = np.nanmedian([pitch_of(A), pitch_of(B)])
        bestA = {}
        for wa, pa in A.items():
            c = sorted((dist(pa, B[wb]), wb) for wb in range(wa - a.kmax, wa + a.kmax + 1) if wb in B)
            if c:
                bestA[wa] = c
        bestB = {}
        for wb, pb in B.items():
            c = sorted((dist(A[wa], pb), wa) for wa in range(wb - a.kmax, wb + a.kmax + 1) if wa in A)
            if c:
                bestB[wb] = c[0][1]
        acc, offs, res = [], [], []
        allbest = [c[0][0] / pitch for c in bestA.values() if np.isfinite(c[0][0])]
        for wa, c in bestA.items():
            d0, wb = c[0]
            d1 = c[1][0] if len(c) > 1 else np.inf
            if not np.isfinite(d0):
                continue
            ok = d0 < a.max_res * pitch and (d1 - d0) > a.margin * pitch and bestB.get(wb) == wa
            if ok:
                acc.append((wa, wb))
                offs.append(wb - wa)
                res.append(d0 / pitch)
                parent[find((ta, wa))] = find((tb, wb))
            else:
                find((ta, wa))
        for wb in B:
            find((tb, wb))
        u, c = np.unique(offs, return_counts=True) if offs else ([], [])
        seams.append({"a": ta, "b": tb, "overlap_z": [z0, z1], "pitch_vox": round(float(pitch), 2),
                      "windings_a": len(A), "windings_b": len(B), "accepted": len(acc),
                      "accepted_frac_of_a": round(len(acc) / max(1, len(A)), 3),
                      "offset_hist": {int(k): int(v) for k, v in zip(u, c, strict=True)},
                      "residual_pitch_p10_p50_p90": [round(float(np.percentile(res, q)), 3) for q in (10, 50, 90)] if res else None,
                      "best_residual_pitch_all_p10_p50_p90": [round(float(np.percentile(allbest, q)), 3) for q in (10, 50, 90)] if allbest else None,
                      "matches": acc})
        print(f"{ta}->{tb} z[{z0},{z1}) pitch {pitch:.1f} vox: accepted {len(acc)}/{len(A)}, best residual (pitch) p10/50/90 {np.round(np.percentile(allbest, [10, 50, 90]), 2) if allbest else None}, offsets {dict(zip(map(int, u), map(int, c), strict=True))}")
    chains = {}
    for node in list(parent):
        chains.setdefault(find(node), []).append(node)
    chains = sorted((sorted(v, key=lambda t: [w[0] for w in wins].index(t[0])) for v in chains.values()),
                    key=lambda v: (-len(v), v[0]))
    gid = {f"{t}:{w}": i for i, ch in enumerate(chains) for t, w in ch}
    span = [len({t for t, _ in ch}) for ch in chains]
    out = {"umbilicus": umb, "windows": [[w[0], w[1], w[2], str(w[3])] for w in wins], "params": vars(a) | {"out": str(a.out), "umbilicus": umb},
           "seams": seams, "n_chains": len(chains),
           "chains_spanning_windows_hist": {int(k): int(v) for k, v in zip(*np.unique(span, return_counts=True), strict=True)},
           "chains": [[f"{t}:{w}" for t, w in ch] for ch in chains], "global_id": gid}
    a.out.write_text(json.dumps(out, indent=1, default=str))
    print(f"{len(chains)} chains; windows spanned per chain: {out['chains_spanning_windows_hist']}")


if __name__ == "__main__":
    main()
