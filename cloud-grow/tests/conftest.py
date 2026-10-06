"""Fixtures: a FAKE VC3D kit (python scripts honouring the real CLI), a fake CT zarr, tifxyz writers.

The fake tracer grows a flat square lattice around its seed (pitch step_size, z constant) with `generations` rings;
the fake selfcross reads FAKE_SELFX (clean | fail | cross). They stand in for the real binaries ONLY so the loop, state
shim, guard wiring, packer and importer can be exercised offline; they say nothing about real geometry (see TESTING.md).
"""
import json
import os
import stat
import sys
import textwrap

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TRACER = '''#!{py}
import json, os, sys, time
import numpy as np, tifffile
a = sys.argv[1:]
vol = a[a.index("-v") + 1]; tgt = a[a.index("-t") + 1]; pj = a[a.index("--params") + 1]
P = json.load(open(pj))
gens, step, vox = int(P["generations"]), float(P["step_size"]), float(P["voxelsize"])
if os.environ.get("FAKE_TRACER") == "fail":
    print("boom"); sys.exit(3)
if "--resume" in a:
    src = a[a.index("--resume") + 1]
    seed = json.load(open(os.path.join(src, "meta.json")))["seed"]
else:
    i = a.index("--seed"); seed = [float(a[i + 1]), float(a[i + 2]), float(a[i + 3])]
g = gens
n = 2 * g + 1
jj, ii = np.meshgrid(np.arange(n), np.arange(n))
X = (seed[0] + (jj - g) * step).astype(np.float32)
Y = (seed[1] + (ii - g) * step).astype(np.float32)
Z = np.full((n, n), seed[2], np.float32)
gen = np.maximum(abs(jj - g), abs(ii - g)).astype(np.uint16)
d = os.path.join(tgt, "auto_grown_%d" % int(time.time() * 1000))
os.makedirs(d)
for k, A in zip("xyz", (X, Y, Z)): tifffile.imwrite(os.path.join(d, k + ".tif"), A)
tifffile.imwrite(os.path.join(d, "generations.tif"), gen)
area = (n - 1) ** 2 * (step * vox * 1e-4) ** 2
json.dump({{"area_cm2": area, "max_gen": gens, "seed": seed, "format": "tifxyz", "type": "seg"}}, open(os.path.join(d, "meta.json"), "w"))
'''

SELFX = '''#!{py}
import json, os, sys
a = sys.argv[1:]
out = a[a.index("-o") + 1]; src = a[a.index("--surface") + 1]
m = os.environ.get("FAKE_SELFX", "clean")
if m == "fail":
    sys.stderr.write("selfcross exploded"); sys.exit(2)
import tifffile
x = tifffile.imread(os.path.join(src, "x.tif")); tri = int((x > 0).sum()) * 2
rep = {{"census": [{{"triangles": tri, "transverse": 0, "transverse_contacts": []}}]}}
if m == "cross":
    rep["census"][0]["transverse"] = 4
    rep["census"][0]["transverse_contacts"] = [{{"quad1": [3, 3], "quad2": [30, 30]}}]
json.dump(rep, open(out, "w"))
'''


def _exe(path, text):
    with open(path, "w") as fh:
        fh.write(text)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)


@pytest.fixture
def kit(tmp_path):
    b = tmp_path / "kit" / "bin"
    b.mkdir(parents=True)
    (tmp_path / "kit" / "lib").mkdir()
    _exe(b / "vc_grow_seg_from_seed", TRACER.format(py=sys.executable))
    _exe(b / "vc_tifxyz_selfcross", SELFX.format(py=sys.executable))
    return str(b)


@pytest.fixture
def pins(kit):
    from cloud_grow import tools as T
    return {n: {"md5": T.md5_file(os.path.join(kit, n))} for n in T.REQUIRED_TOOLS}


@pytest.fixture
def ct_zarr(tmp_path):
    import zarr
    p = str(tmp_path / "ct.zarr")
    g = zarr.open(p, mode="w")
    for lv, s in ((1, (120, 600, 600)), (4, (16, 75, 75))):
        a = g.create_dataset(str(lv), shape=s, chunks=(32, 64, 64), dtype="u1", fill_value=0)
        a[:] = 200
    return p


@pytest.fixture
def grids(tmp_path):
    p = tmp_path / "grids"
    for a in ("xy", "xz", "yz"):
        (p / a).mkdir(parents=True)
    return str(p)


@pytest.fixture
def cfg(tmp_path, kit, ct_zarr, grids):
    from cloud_grow import runner as R
    return R.RunConfig(scroll="PHercTEST", workdir=str(tmp_path / "wd"), kit_bin=kit, ct_zarr=ct_zarr, normal_grids=grids,
                       voxel_um=9.362, tracer_volume="ct", first_gens=6, gen_step=4, wall_s=300, max_rounds=2,
                       policy_path=os.path.join(os.path.dirname(__file__), "..", "config", "guard_policy.production.json"))


def write_tifxyz(d, X, Y, Z, area, max_gen=1, extra=None):
    import tifffile
    os.makedirs(d, exist_ok=True)
    for k, A in zip("xyz", (X, Y, Z)):
        tifffile.imwrite(os.path.join(d, k + ".tif"), np.asarray(A, np.float32))
    m = {"area_cm2": area, "max_gen": max_gen, "format": "tifxyz", "type": "seg"}
    m.update(extra or {})
    with open(os.path.join(d, "meta.json"), "w") as fh:
        json.dump(m, fh)


def plane(n=12, pitch=20.0, x0=300.0, y0=300.0, z=100.0):
    jj, ii = np.meshgrid(np.arange(n), np.arange(n))
    return x0 + jj * pitch, y0 + ii * pitch, np.full((n, n), z)
