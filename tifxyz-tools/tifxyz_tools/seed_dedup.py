"""Sheet-aware seed dedup: refuse a tracer seed that lies ON a sheet we already hold.

A coverage mask (every grown lattice dilated by a fixed radius, e.g. 32 voxels) is sheet-BLIND: it
forbids the next wrap (25-35 voxels away) as firmly as the same sheet. This index asks the right
question of a seed point p from the surfaces themselves: is there an existing surface point q with

      |(p - q) . n_q| <= same_sheet_vox   (p is within a few voxels of q's sheet, along q's normal)
      lateral distance <= lateral_vox      (and q is within about a lattice step or two of p)

If so, growing from p would re-trace q's segment and the seed is refused; `check` names the coverer,
so the caller can extend that segment instead (a seed near a segment's rim is a merge candidate).
`before` restricts coverers to segments created earlier, which makes the check testable against
history; `own` excludes a segment's own surface.

Prototype: nothing here calls a tracer. Default policy `enabled=False` is for callers that keep a
switch; `check` itself ignores it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeedPolicy:
    enabled: bool = False
    same_sheet_vox: float = 3.0     # see VALIDATION.md for what these defaults were measured to do
    lateral_vox: float = 50.0
    stride: int = 3                 # lattice thinning when indexing (every 3rd cell of a 20-voxel lattice)


class SurfaceIndex:
    def __init__(self):
        from . import same_sheet as C
        self.C = C
        self._xyz, self._nrm, self._nok, self._seg, self._t = [], [], [], [], []
        self.names: list[str] = []
        self.tree = None

    def add(self, seg: str, created_ts: float, X, Y, Z, stride: int = 3):
        T = self.C.surface_points(X[::stride, ::stride], Y[::stride, ::stride], Z[::stride, ::stride])
        if not len(T["xyz"]):
            return
        k = len(self.names)
        self.names.append(seg)
        self._xyz.append(T["xyz"])
        self._nrm.append(T["nrm"])
        self._nok.append(T["nok"])
        self._seg.append(np.full(len(T["xyz"]), k, np.int32))
        self._t.append(np.full(len(T["xyz"]), created_ts))

    def finalize(self):
        from scipy.spatial import cKDTree
        self.xyz = np.concatenate(self._xyz).astype(np.float64)
        self.nrm = np.concatenate(self._nrm).astype(np.float64)
        self.nok = np.concatenate(self._nok)
        self.seg = np.concatenate(self._seg)
        self.t = np.concatenate(self._t)
        self.tree = cKDTree(self.xyz, leafsize=32, balanced_tree=False, compact_nodes=False)
        self._xyz = self._nrm = self._nok = self._seg = self._t = None
        return self

    def check(self, p, pol: SeedPolicy, before: float | None = None, own: str | None = None):
        """(covered?, detail). detail = {"by": seg, "along_normal": t, "lateral": lat} for the best coverer."""
        p = np.asarray(p, np.float64)
        r = float(np.hypot(pol.lateral_vox, pol.same_sheet_vox))
        best = None
        for j in self.tree.query_ball_point(p, r):
            if not self.nok[j]:
                continue
            if before is not None and self.t[j] >= before:
                continue
            nm = self.names[self.seg[j]]
            if own is not None and nm == own:
                continue
            d = p - self.xyz[j]
            t = abs(float(d @ self.nrm[j]))
            lat = float(np.sqrt(max(0.0, d @ d - t * t)))
            if t <= pol.same_sheet_vox and lat <= pol.lateral_vox and (best is None or t < best[1]):
                best = (nm, t, lat)
        if best is None:
            return False, {}
        return True, {"by": best[0], "along_normal": round(best[1], 2), "lateral": round(best[2], 1)}
