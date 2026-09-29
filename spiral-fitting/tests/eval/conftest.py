"""Synthetic Archimedean-spiral fixtures for the spiral-fit evaluation tools (spiral-fitting/eval).

Geometry (level-0 voxels): a spiral about (CX, CY), constant in z, r(theta) = R0 + PITCH * theta / 2pi.
Winding w covers theta in [2 pi w, 2 pi (w + 1)), so adjacent windings are PITCH apart along any ray.
The surface prediction marks every voxel within 1 voxel of the spiral; fitted windings are written
as fit_spiral.py-style tifxyz directories named `wNNN_spliced_<tag>`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))

CX, CY = 256.0, 256.0
R0, PITCH = 30.0, 20.0
NWIND = 9                     # windings 0..8, outermost radius R0 + 9 * PITCH = 210 < 256
NZ, NY, NX = 48, 512, 512


def radius(w, th):
    return R0 + PITCH * (w + th / (2 * np.pi))


def write_winding(root: Path, idx: int, geom_w: float, z0: float, z1: float, tag: str = "t",
                  ncol: int = 288, nrow: int = 12, dr: float = 0.0):
    """Winding directory `w{idx:03d}_spliced_{tag}` whose geometry is spiral winding `geom_w`
    (may differ from idx to plant a numbering offset), radially displaced by `dr` voxels."""
    th = np.linspace(0, 2 * np.pi, ncol, endpoint=False)
    zz = np.linspace(z0, z1, nrow)
    TH, ZZ = np.meshgrid(th, zz)
    r = radius(geom_w, TH) + dr
    d = root / f"w{idx:03d}_spliced_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(d / "x.tif", (CX + r * np.cos(TH)).astype(np.float32))
    tifffile.imwrite(d / "y.tif", (CY + r * np.sin(TH)).astype(np.float32))
    tifffile.imwrite(d / "z.tif", ZZ.astype(np.float32))
    return d


@pytest.fixture(scope="session")
def pred_zarr(tmp_path_factory) -> Path:
    """OME-Zarr-like group with arrays '0', '1', '2' (max-pooled) marking the spiral sheets."""
    root = tmp_path_factory.mktemp("pred") / "pred.zarr"
    yy, xx = np.mgrid[0:NY, 0:NX].astype(np.float64)
    dx, dy = xx - CX, yy - CY
    r = np.hypot(dx, dy)
    th = np.mod(np.arctan2(dy, dx), 2 * np.pi)
    # distance to the nearest spiral turn along the ray: (r - radius(0, th)) mod PITCH
    u = (r - radius(0, th)) / PITCH
    k = np.round(u)
    on = (np.abs(u - k) * PITCH <= 1.0) & (k >= 0) & (k < NWIND)
    sl = on.astype(np.uint8) * 255
    g = zarr.open_group(str(root), mode="w")
    vol = np.repeat(sl[None], NZ, axis=0)
    for lev in range(3):
        f = 2 ** lev
        v = vol[::f, :, :].reshape(NZ // f, NY // f, f, NX // f, f).max(axis=(2, 4)) if f > 1 else vol
        g.create_dataset(str(lev), data=v, chunks=(16, 128, 128), overwrite=True)
    return root


@pytest.fixture(scope="session")
def umbilicus(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("umb") / "umbilicus.json"
    p.write_text(json.dumps({"control_points": [{"z": 0, "x": CX, "y": CY}, {"z": NZ, "x": CX, "y": CY}]}))
    return p
