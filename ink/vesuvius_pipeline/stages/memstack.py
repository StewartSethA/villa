"""The ONE on-disk stack is the native one; every model frame is an in-memory view of it.

User directive, 2026-09-29: "delete all of the s200 voxel grids. If there are other
resamplings, delete those too. Keep only the native resolution. Only augment the inputs
dynamically at inference time, then discard after all models have made their passes."

So a segment is rendered ONCE per flatten, at the scan's own voxel pitch (one render pixel
per CT voxel in-plane; the renderer's layers are already one voxel apart in depth), into
`layers_native/`. A model trained at 7.91 um/px (the Grand Prize frame) or 9.5 um/px (ink_9um)
reads a `MemStack`: a lazy, resampled, optionally normalised VIEW of that stack that lives in
this process only and is dropped when the pass ends. Nothing resampled touches a disk.

How a model reads it without being rewritten. The ink wrappers take a layers DIRECTORY and read
TIFFs out of it. A MemStack registers itself under a pseudo-directory `mem:<id>`; the three
choke points the production families read through (`grandprize_dense._layer_files/_layer_array`,
`ink9um_student.layer_files`, `stroke_ring._layer_files/_read` + `colmean.render_um_per_px`)
ask this module first. A layer is a `LazyLayer`: it has `.shape/.ndim/.dtype`, answers a row
slice by resampling only the native rows that slice needs, and materialises (and caches, within
a RAM budget) on any other access -- so a whole-canvas consumer and a banded one both work, and
a segment too big for RAM still streams.

The geometry is the renderer's own, not an image-library convention. `vc_render_tifxyz` /
`scripts/gpu_render/render_gpu.py` place render pixel u of a W-wide canvas at lattice coordinate
`center + (u - (W-1)/2) / render_scale` -- a CENTRED map. Two renders of one flatten at scales
s_n and s_t are therefore related by

    u_n = (W_n - 1)/2 + (u_t - (W_t - 1)/2) * (s_n / s_t),     s_n / s_t = um_t / um_n

and that is the map used here (separable linear interpolation, per axis). `W_t` is the width
the renderer WOULD have produced at the target scale -- round(cols x s_t / grid_scale) from the
flatten's own x.tif and meta.json when the caller passes the flat, else round(W_n x um_n/um_t).

Outside-the-surface pixels are 0 in a render. A resampled pixel is 0 exactly when its NEAREST
native pixel is 0 (so the footprint does not bleed), and never below 1 otherwise.

Normalisation (`normalize=True`) is `render.normalize_layers` applied to the view instead of to
files: mean/std of the material (>0) of the MIDDLE resampled layer, affine to 150/25, clip to
[1, 255], truncating uint8 cast, zeros kept -- byte-for-byte the transform the on-disk Grand
Prize stack used to carry.
"""
from __future__ import annotations

import itertools
import json
import os
import threading
from pathlib import Path

import numpy as np

PREFIX = "mem:"
GP_NORM_MEAN = 150.0
GP_NORM_STD = 25.0
_REG: dict[str, "MemStack"] = {}
_LOCK = threading.Lock()
_IDS = itertools.count(1)


def cache_budget_bytes() -> int:
    """RAM the resampled layers may occupy per stack before rows are recomputed on demand."""
    return int(float(os.environ.get("VPIPE_MEMSTACK_CACHE_GB", "24")) * 2**30)


def is_mem(path) -> bool:
    return isinstance(path, str) and path.startswith(PREFIX)


def _split(path: str) -> tuple[str, str | None]:
    """'mem:7/12.tif' -> ('mem:7', '12.tif'); 'mem:7' -> ('mem:7', None)."""
    p = path.rstrip("/")
    head, sep, tail = p.partition("/")
    return head, (tail or None) if sep else None


def get(path: str) -> "MemStack | None":
    if not is_mem(path):
        return None
    head, _ = _split(path)
    with _LOCK:
        return _REG.get(head)


def list_names(layers_dir: str, exts=(".tif", ".tiff")) -> list[str] | None:
    """Layer names of a registered stack, or None when `layers_dir` is a real directory."""
    st = get(layers_dir)
    return None if st is None else list(st.names)


def read_layer(path: str):
    """The LazyLayer behind 'mem:<id>/<name>', or None for a real file path."""
    st = get(path)
    if st is None:
        return None
    _, name = _split(path)
    if name is None:
        raise FileNotFoundError(f"{path}: a stack, not a layer")
    return st.layer(st.names.index(os.path.basename(name)))


def um_per_px(layers_dir: str) -> float | None:
    st = get(layers_dir)
    return None if st is None else st.um_per_px


def plane_um(layers_dir: str) -> float | None:
    """um between adjacent layers of a registered view (the CT voxel), or None."""
    st = get(layers_dir)
    return None if st is None else st.plane_um


def _layer_files(d: str) -> list[str]:
    return sorted(f for f in os.listdir(d) if f.lower().endswith((".tif", ".tiff")))


def _open_native(fp: str):
    import tifffile
    try:
        a = tifffile.memmap(fp, mode="r")
    except (ValueError, OSError):
        a = tifffile.imread(fp)
    if a.ndim == 3:
        a = a[..., 0]
    if a.dtype != np.uint8:
        a = np.clip(a.astype(np.float32) * ((255.0 / 65535.0) if a.max() > 255 else 1.0), 0, 255).astype(np.uint8)
    return a


def render_width(flat: str | None, scale: float) -> tuple[int, int] | None:
    """(H, W) the renderer produces for `flat` at `scale` (render_gpu.py: round(n * scale / gscale)),
    or None when the flatten cannot be read."""
    if not flat:
        return None
    try:
        import tifffile
        with tifffile.TiffFile(os.path.join(flat, "x.tif")) as tf:
            srows, scols = tf.pages[0].shape
        gs = [float(v) for v in json.load(open(os.path.join(flat, "meta.json")))["scale"][:2]]
    except (OSError, ValueError, KeyError, IndexError):
        return None
    scale = round(float(scale), 4)        # every render tier is handed `--scale {scale:.4f}`
    return max(1, int(round(srows * scale / gs[1]))), max(1, int(round(scols * scale / gs[0])))


def _axis_map(n_src: int, n_dst: int, ratio: float):
    """Per destination index: (i0, i1, w, nearest) into the source axis, centred convention."""
    u = np.arange(n_dst, dtype=np.float64)
    j = (n_src - 1) / 2.0 + (u - (n_dst - 1) / 2.0) * ratio
    j = np.clip(j, 0.0, n_src - 1.0)
    i0 = np.floor(j).astype(np.int64)
    i1 = np.minimum(i0 + 1, n_src - 1)
    w = (j - i0).astype(np.float32)
    nn = np.clip(np.rint(j), 0, n_src - 1).astype(np.int64)
    return i0, i1, w, nn


class LazyLayer:
    """One resampled layer. Array-like enough for np.asarray, a[r0:r1, :W], a[::s, ::s], .shape."""
    ndim = 2
    dtype = np.dtype(np.uint8)

    def __init__(self, stack: "MemStack", k: int):
        self._st, self._k = stack, k
        self.shape = stack.shape

    def __array__(self, dtype=None, copy=None):
        a = self._st.full(self._k)
        return a if dtype is None else a.astype(dtype)

    def __len__(self):
        return self.shape[0]

    def rows_device(self, r0: int, r1: int):
        """Rows [r0, r1) as a uint8 tensor ON THE VIEW'S CARD (resampled and normalised there),
        or None when the view resamples on the CPU. Lets a GPU consumer skip the D2H + H2D."""
        return self._st.rows_device(self._k, r0, r1)

    def __getitem__(self, key):
        if isinstance(key, tuple) and key and isinstance(key[0], slice) and key[0].step in (None, 1) \
                and not self._st.cached(self._k):
            r0, r1, _ = key[0].indices(self.shape[0])
            rows = self._st.rows(self._k, r0, r1)
            return rows[(slice(None),) + key[1:]]
        return self._st.full(self._k)[key]


class MemStack:
    """A resampled (and optionally normalised / depth-reversed) in-memory view of a native stack."""

    def __init__(self, native_dir: str, src_um: float, dst_um: float, *, out_shape=None,
                 normalize: bool = False, reverse: bool = False, device: str | None = None,
                 label: str = "", layer_names: list[str] | None = None, _shared=None,
                 plane_um: float | None = None):
        self.native_dir = str(native_dir)
        self.src_um, self.dst_um = float(src_um), float(dst_um)
        self.normalize, self.reverse, self.label = bool(normalize), bool(reverse), label
        self.device = device
        # layer spacing: every render tier steps ONE CT VOXEL per plane, so a view of a native
        # stack (one px per voxel) has planes src_um apart whatever its in-plane pitch
        self.plane_um = float(plane_um) if plane_um else self.src_um
        if _shared is not None:                      # a reversed view shares arrays and cache
            self._sh = _shared
        else:
            files = layer_names or _layer_files(self.native_dir)
            if not files:
                raise FileNotFoundError(f"no rendered layers in {self.native_dir}")
            nat = [_open_native(os.path.join(self.native_dir, f)) for f in files]
            Hn, Wn = nat[len(nat) // 2].shape[:2]
            ratio = self.dst_um / self.src_um          # native px per target px
            if out_shape is None:
                out_shape = (max(1, int(round(Hn / ratio))), max(1, int(round(Wn / ratio))))
            Ht, Wt = int(out_shape[0]), int(out_shape[1])
            self._sh = {"files": list(files), "nat": nat, "native_shape": (Hn, Wn), "shape": (Ht, Wt),
                        "ratio": ratio, "rmap": _axis_map(Hn, Ht, ratio), "cmap": _axis_map(Wn, Wt, ratio),
                        "cache": {}, "cache_bytes": 0, "norm": None, "stats": {"rows_resampled": 0, "s": 0.0}}
        self.shape = self._sh["shape"]
        n = len(self._sh["files"])
        self.names = [f"{i:02d}.tif" for i in range(n)]
        self.key: str | None = None
        if self.normalize and self._sh["norm"] is None:
            self._sh["norm"] = self._norm_stats()

    @classmethod
    def from_array(cls, stack, um: float, *, normalize: bool = False, reverse: bool = False,
                   label: str = "", device: str | None = None,
                   plane_um: float | None = None) -> "MemStack":
        """A view of a (Z, H, W) uint8 stack ALREADY in memory at the model frame -- e.g. a
        render made straight into RAM at the model's scale (render_gpu.render(write_tif=False)).
        No resample (ratio 1); normalisation and depth reversal as for any view."""
        v = cls.__new__(cls)
        n, H, W = stack.shape
        v.native_dir, v.src_um, v.dst_um = "<in-memory render>", float(um), float(um)
        v.normalize, v.reverse, v.label, v.device, v.key = bool(normalize), bool(reverse), label, device, None
        if not plane_um:
            raise ValueError("from_array needs plane_um: the render's layer spacing (the CT voxel), "
                             "which an in-memory array cannot tell")
        v.plane_um = float(plane_um)
        v._sh = {"files": [f"{i:02d}.tif" for i in range(n)], "nat": [stack[i] for i in range(n)],
                 "native_shape": (H, W), "shape": (H, W), "ratio": 1.0,
                 "rmap": _axis_map(H, H, 1.0), "cmap": _axis_map(W, W, 1.0),
                 "cache": {}, "cache_bytes": 0, "norm": None, "stats": {"rows_resampled": 0, "s": 0.0}}
        v.shape = (H, W)
        v.names = list(v._sh["files"])
        if v.normalize:
            v._sh["norm"] = v._norm_stats()
        return v

    # -- identity --------------------------------------------------------------------
    @property
    def um_per_px(self) -> float:
        return self.dst_um

    @property
    def n_layers(self) -> int:
        return len(self.names)

    def provenance(self) -> dict:
        """What this view IS, for the prediction's meta (CLAUDE.md: provenance travels)."""
        mu_sd = self._sh["norm"]
        return {"native_dir": self.native_dir, "native_um_per_px": self.src_um,
                "model_um_per_px": self.dst_um, "resample_factor": self.dst_um / self.src_um,
                "native_shape": list(self._sh["native_shape"]), "model_shape": list(self.shape),
                "layers": self.n_layers, "method": "separable linear, renderer-centred grid, "
                "nearest-zero footprint", "normalize": ("gp 150/25 from middle-layer material "
                f"(mean {mu_sd[0]:.2f}, std {mu_sd[1]:.2f})" if mu_sd else None) if self.normalize else None,
                "plane_um": self.plane_um, "depth_reversed": self.reverse, "in_memory": True, "written_to_disk": False,
                "rows_resampled": self._sh["stats"]["rows_resampled"],
                "resample_s": round(self._sh["stats"]["s"], 3)}

    # -- registration ------------------------------------------------------------------
    def register(self) -> str:
        with _LOCK:
            if self.key is None:
                self.key = f"{PREFIX}{os.getpid()}.{next(_IDS)}"
                _REG[self.key] = self
        return self.key

    def release(self) -> None:
        """Drop the registration and, when no other view holds it, every cached array."""
        with _LOCK:
            if self.key is not None:
                _REG.pop(self.key, None)
                self.key = None
            others = any(v._sh is self._sh for v in _REG.values())
        if not others:
            self._sh["cache"].clear()
            self._sh["cache_bytes"] = 0

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, *exc):
        self.release()

    def reversed_view(self) -> "MemStack":
        v = MemStack.__new__(MemStack)
        v.__dict__.update(self.__dict__)
        v.reverse = not self.reverse
        v.key = None
        return v

    # -- the arithmetic ------------------------------------------------------------------
    def _src_index(self, k: int) -> int:
        return (len(self._sh["files"]) - 1 - k) if self.reverse else k

    def _resample_rows(self, src_k: int, r0: int, r1: int) -> np.ndarray:
        """Target rows [r0, r1) of native layer `src_k`, resampled, NOT normalised, uint8."""
        import time
        t0 = time.time()
        a = self._sh["nat"][src_k]
        if self._identity():                          # a view AT the stack's own frame: no arithmetic
            out = np.array(a[r0:r1], dtype=np.uint8)
            self._sh["stats"]["s"] += time.time() - t0
            return out
        i0, i1, w, nn = (x[r0:r1] for x in self._sh["rmap"])
        c0, c1, wc, cn = self._sh["cmap"]
        lo, hi = int(min(i0.min(), nn.min())), int(max(i1.max(), nn.max())) + 1
        blk = np.asarray(a[lo:hi])
        out = None
        if self.device and str(self.device).startswith("cuda"):
            try:
                out = self._resample_torch(blk, i0 - lo, i1 - lo, w, nn - lo, c0, c1, wc, cn)
            except Exception as e:                   # noqa: BLE001 - the CPU path is exact too
                print(f"[memstack] GPU resample failed ({type(e).__name__}: {e}); CPU", flush=True)
                self.device = None
        if out is None:
            b = blk.astype(np.float32)
            rows = b[i0 - lo] * (1.0 - w)[:, None] + b[i1 - lo] * w[:, None]
            v = rows[:, c0] * (1.0 - wc)[None, :] + rows[:, c1] * wc[None, :]
            near = blk[nn - lo][:, cn]
            out = np.clip(np.rint(v), 1, 255).astype(np.uint8)
            out[near == 0] = 0
        st = self._sh["stats"]
        st["rows_resampled"] += int(r1 - r0)
        st["s"] += time.time() - t0
        return out

    def _resample_torch(self, blk, i0, i1, w, nn, c0, c1, wc, cn, keep_on_device: bool = False):
        import torch
        dev = torch.device(self.device)
        b = torch.from_numpy(np.ascontiguousarray(blk)).to(dev)
        bf = b.float()
        ti0, ti1, tn = (torch.from_numpy(x).to(dev) for x in (i0, i1, nn))
        tc0, tc1, tcn = (torch.from_numpy(x).to(dev) for x in (c0, c1, cn))
        tw = torch.from_numpy(w).to(dev)[:, None]
        twc = torch.from_numpy(wc).to(dev)[None, :]
        rows = bf.index_select(0, ti0) * (1.0 - tw) + bf.index_select(0, ti1) * tw
        v = rows.index_select(1, tc0) * (1.0 - twc) + rows.index_select(1, tc1) * twc
        near = b.index_select(0, tn).index_select(1, tcn)
        out = torch.round(v).clamp_(1, 255).to(torch.uint8)
        out[near == 0] = 0
        return out if keep_on_device else out.cpu().numpy()

    def _identity(self) -> bool:
        return self._sh["ratio"] == 1.0 and tuple(self._sh["shape"]) == tuple(self._sh["native_shape"])

    def _on_card(self) -> bool:
        return bool(self.device and str(self.device).startswith("cuda"))

    def _apply_norm(self, a: np.ndarray) -> np.ndarray:
        if not self.normalize or self._sh["norm"] is None:
            return a
        mu, sd = self._sh["norm"]
        f = a.astype(np.float32)
        z = f > 0
        b = (f - mu) / sd * GP_NORM_STD + GP_NORM_MEAN
        b[~z] = 0
        return np.clip(b, 1, 255).astype(np.uint8) * z.astype(np.uint8)

    def _norm_stats(self):
        """render.normalize_layers' statistics, on the resampled middle layer."""
        mid_src = len(self._sh["files"]) // 2
        mid = self._resample_rows(mid_src, 0, self.shape[0]).astype(np.float32)
        m = mid > 0
        if m.sum() < 1000:
            return None
        mu, sd = float(mid[m].mean()), float(mid[m].std())
        return None if sd < 1e-3 else (mu, sd)

    # -- access ----------------------------------------------------------------------------
    def cached(self, k: int) -> bool:
        return (self._src_index(k), self.normalize) in self._sh["cache"]

    def rows(self, k: int, r0: int, r1: int) -> np.ndarray:
        ck = (self._src_index(k), self.normalize)
        c = self._sh["cache"].get(ck)
        if c is not None:
            return c[r0:r1]
        if self._on_card():                           # resample + normalise on the card, one D2H
            return self.rows_device(k, r0, r1).cpu().numpy()
        return self._apply_norm(self._resample_rows(self._src_index(k), r0, r1))

    def rows_device(self, k: int, r0: int, r1: int):
        """`rows` computed and kept on the view's CUDA device (None without one)."""
        if not (self.device and str(self.device).startswith("cuda")):
            return None
        import time
        import torch
        ck = (self._src_index(k), self.normalize)
        c = self._sh["cache"].get(ck)
        if c is not None:
            return torch.from_numpy(np.ascontiguousarray(c[r0:r1])).to(self.device, non_blocking=True)
        t0 = time.time()
        src_k = self._src_index(k)
        a = self._sh["nat"][src_k]
        if self._identity():
            out = torch.from_numpy(np.ascontiguousarray(a[r0:r1])).to(self.device, non_blocking=True)
        else:
            i0, i1, w, nn = (x[r0:r1] for x in self._sh["rmap"])
            c0, c1, wc, cn = self._sh["cmap"]
            lo, hi = int(min(i0.min(), nn.min())), int(max(i1.max(), nn.max())) + 1
            out = self._resample_torch(np.asarray(a[lo:hi]), i0 - lo, i1 - lo, w, nn - lo, c0, c1, wc, cn,
                                       keep_on_device=True)
        if self.normalize and self._sh["norm"] is not None:
            mu, sd = self._sh["norm"]
            f = out.float()
            z = f > 0
            b = (f - mu) / sd * GP_NORM_STD + GP_NORM_MEAN
            out = (b.clamp(1, 255).to(torch.uint8) * z).to(torch.uint8)   # trunc cast, as on the host
        st = self._sh["stats"]
        st["rows_resampled"] += int(r1 - r0)
        st["s"] += time.time() - t0
        return out

    def full(self, k: int) -> np.ndarray:
        ck = (self._src_index(k), self.normalize)
        c = self._sh["cache"].get(ck)
        if c is not None:
            return c
        if self._on_card():
            a = self.rows_device(k, 0, self.shape[0]).cpu().numpy()
        else:
            a = self._apply_norm(self._resample_rows(self._src_index(k), 0, self.shape[0]))
        if self._sh["cache_bytes"] + a.nbytes <= cache_budget_bytes():
            self._sh["cache"][ck] = a
            self._sh["cache_bytes"] += a.nbytes
        return a

    def layer(self, k: int) -> LazyLayer:
        return LazyLayer(self, k)

    def mask(self, out_png: str | None = None) -> np.ndarray:
        """render.make_mask on the middle layer of THIS view (closing x3 + hole fill)."""
        from scipy import ndimage
        m = self.full(self.n_layers // 2) > 0
        m = ndimage.binary_fill_holes(ndimage.binary_closing(m, iterations=3))
        if out_png:
            from PIL import Image
            Path(out_png).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((m * 255).astype(np.uint8)).save(out_png)
        return m


def open_view(native_dir: str, src_um: float, dst_um: float, *, flat: str | None = None,
              native_scale: float | None = None, normalize: bool = False, reverse: bool = False,
              device: str | None = None, label: str = "", plane_um: float | None = None) -> MemStack:
    """A MemStack whose target canvas is exactly the one the renderer would have produced at the
    model's scale (`flat` + `native_scale` known), else the proportional canvas."""
    shape = None
    if flat and native_scale:
        shape = render_width(flat, float(native_scale) * float(src_um) / float(dst_um))
    return MemStack(native_dir, src_um, dst_um, out_shape=shape, normalize=normalize, reverse=reverse,
                    device=device, label=label, plane_um=plane_um)
