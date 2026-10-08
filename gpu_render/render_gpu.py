#!/usr/bin/env python
"""GPU reimplementation of vc_render_tifxyz (tif-output / renderBands path).

Reproduces the CPU pipeline exactly:
  QuadSurface::gen -> prepareBaseAndDirs -> readMultiSlice(trilinear,u8) -> tif

Volume access: the scroll zarr is *uncompressed* (compressor: null), 128^3 raw
uint8 chunk files, so chunks are read with plain file I/O and uploaded to a
GPU-resident chunk pool addressed through a dense chunk->slot LUT.
"""
import argparse, ctypes, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import tifffile
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cuda_rt import Module

CHUNK = 128
CHUNK_VOX = CHUNK ** 3


class Timer:
    def __init__(self):
        self.t = {}

    def add(self, k, dt):
        self.t[k] = self.t.get(k, 0.0) + dt

    def report(self):
        return dict(sorted(self.t.items(), key=lambda kv: -kv[1]))


# ---------------------------------------------------------------- surface
def load_surface(seg_dir):
    pts = np.stack([tifffile.imread(os.path.join(seg_dir, f"{a}.tif"))
                    for a in "xyz"], axis=-1).astype(np.float32)
    meta = json.load(open(os.path.join(seg_dir, "meta.json")))
    scale = np.array(meta["scale"][:2], dtype=np.float32)
    if meta.get("components"):
        raise NotImplementedError("multi-component surfaces not handled")
    # load_quad_from_tifxyz_impl(): "Invalidate by z<=0" -> (-1,-1,-1)
    pts[pts[..., 2] <= 0.0] = -1.0
    # main(): replace sentinel -1 (tested on x only) with NaN in all 3
    pts[pts[..., 0] == -1.0] = np.nan
    return pts, scale


def valid_mask(pts):
    """QuadSurface::validMask"""
    nan = np.isnan(pts).any(axis=-1)
    sent = (pts[..., 0] == -1.0) & (pts[..., 1] == -1.0) & (pts[..., 2] == -1.0)
    return np.where(nan | sent, np.uint8(0), np.uint8(255))


def normal_cache(pts):
    """QuadSurface::gen's _normalCache build, via grid_normal_int."""
    rows, cols, _ = pts.shape
    nc = np.full((rows, cols, 3), np.nan, dtype=np.float32)
    if rows < 3 or cols < 3:
        return nc
    with np.errstate(invalid='ignore', divide='ignore'):
        xl = pts[1:-1, :-2]; xr = pts[1:-1, 2:]
        yu = pts[:-2, 1:-1]; yd = pts[2:, 1:-1]
        xv = (xr - xl).astype(np.float32)
        yv = (yd - yu).astype(np.float32)
        n = np.empty_like(xv)
        n[..., 0] = xv[..., 1] * yv[..., 2] - xv[..., 2] * yv[..., 1]
        n[..., 1] = xv[..., 2] * yv[..., 0] - xv[..., 0] * yv[..., 2]
        n[..., 2] = xv[..., 0] * yv[..., 1] - xv[..., 1] * yv[..., 0]
        len2 = (n[..., 0] * n[..., 0] + n[..., 1] * n[..., 1] + n[..., 2] * n[..., 2]).astype(np.float32)
        inv = (np.float32(1.0) / np.sqrt(len2)).astype(np.float32)
        out = n * inv[..., None]
        # bad: any of the four neighbours has x == -1 (sentinel); or degenerate
        bad = ((xl[..., 0] == -1.0) | (xr[..., 0] == -1.0) |
               (yu[..., 0] == -1.0) | (yd[..., 0] == -1.0) |
               (len2 == 0.0) | np.isnan(len2))
        out[bad] = np.nan
        nc[1:-1, 1:-1] = out
    return nc


# ---------------------------------------------------------------- volume
class ZarrU8:
    def __init__(self, path, level=0):
        za = json.load(open(os.path.join(path, str(level), ".zarray")))
        assert za["compressor"] is None, "only uncompressed zarr supported by this fast path"
        assert za["dtype"] in ("|u1", "u1"), za["dtype"]
        assert za["chunks"] == [CHUNK] * 3, za["chunks"]
        self.root = os.path.join(path, str(level))
        self.shape = tuple(za["shape"])          # (z, y, x)
        self.cn = tuple((s + CHUNK - 1) // CHUNK for s in self.shape)
        self.sep = za.get("dimension_separator", ".")

    def chunk_path(self, cz, cy, cx):
        # honour the store's dimension_separator: the bucket volumes are nested ("/"),
        # the pyramid levels and sparse volumes we write ourselves are flat (".")
        if self.sep == "/":
            return os.path.join(self.root, str(cz), str(cy), str(cx))
        return os.path.join(self.root, f"{cz}{self.sep}{cy}{self.sep}{cx}")


class PoolTooSmall(RuntimeError):
    """The chunks one band needs do not all fit in the pool at once.

    render() answers this by splitting the band into fewer rows, never by
    evicting a chunk the band is about to sample: an evicted chunk's LUT entry is
    -1, sample_band reads -1 as "missing chunk -> 0", and the render comes back
    ok with chunk-shaped black wedges in it (the "triangle holes", 2026-09-24)."""


class ChunkPool:
    """GPU-resident LRU pool of raw 128^3 uint8 chunks + dense chunk->slot LUT."""

    def __init__(self, vol, nslots, device, io_threads=16):
        self.vol = vol
        self.dev = device
        self.nslots = nslots
        self.pool = torch.empty(nslots * CHUNK_VOX, dtype=torch.uint8, device=device)
        cnz, cny, cnx = vol.cn
        self.lut = torch.full((cnz * cny * cnx,), -1, dtype=torch.int32, device=device)
        self.marks = torch.zeros(cnz * cny * cnx, dtype=torch.uint8, device=device)
        self.resident = {}          # key -> slot
        self.lastuse = {}           # key -> band index, MONOTONE across renders
        # A reused pool must not restart its clock: eviction picks victims with
        # `lastuse < band`, so a second render starting again at band 0 would find
        # no evictable chunk in a full pool and stall. band_base keeps the index
        # increasing for the life of the pool.
        self.band_base = 0
        self.free = list(range(nslots))
        self.absent = set()         # keys with no chunk file on disk
        self.batch = 64
        self._cuda = str(device).startswith("cuda")
        self.stage = torch.empty(self.batch * CHUNK_VOX, dtype=torch.uint8)
        if self._cuda:
            self.stage = self.stage.pin_memory()
        self.stage_np = self.stage.numpy()
        self.pool_exec = ThreadPoolExecutor(max_workers=io_threads)
        self.bytes_read = 0
        self.chunks_loaded = 0

    def _read(self, key):
        cnz, cny, cnx = self.vol.cn
        cx = key % cnx
        cy = (key // cnx) % cny
        cz = key // (cnx * cny)
        p = self.vol.chunk_path(cz, cy, cx)
        try:
            with open(p, "rb", buffering=0) as f:
                b = f.read()
        except FileNotFoundError:
            return key, None
        if len(b) != CHUNK_VOX:
            return key, None
        return key, b

    def _sync(self):
        if self._cuda:
            torch.cuda.synchronize()

    def ensure(self, keys, band, timer):
        """Make every chunk in `keys` resident (or known-absent) for this band.

        A chunk this band needs is NEVER a victim, whatever its lastuse: evicting
        one sets its LUT entry to -1 and the band then samples 0 there, silently.
        Until 2026-09-24 the victim set was `lastuse < band`, which includes this
        band's own already-resident chunks (they were last used by an earlier
        band), so a pool smaller than a band's working set -- the resident server
        OOM-halves its 4 GB pool to 128 slots on a busy card -- rendered ok=True
        with 41 % of a PHerc0125 segment black. If the band cannot fit, raise
        PoolTooSmall and let the caller split the band."""
        keyset = set(keys)
        want = [k for k in keyset if k not in self.absent]
        if len(want) > self.nslots:
            raise PoolTooSmall(f"band needs {len(want)} chunks, pool holds {self.nslots}")
        need = [k for k in keys if k not in self.resident and k not in self.absent]
        if not need:
            for k in keys:
                if k in self.resident:
                    self.lastuse[k] = band
            return
        # evict LRU, never a chunk this band needs
        deficit = len(need) - len(self.free)
        if deficit > 0:
            victims = sorted(((self.lastuse[k], k) for k in self.resident
                              if k not in keyset and self.lastuse[k] < band))
            if len(victims) < deficit:
                raise PoolTooSmall(f"chunk pool too small: need {len(need)}, "
                                   f"pool {self.nslots}, evictable {len(victims)}")
            ev_keys = [k for _, k in victims[:deficit]]
            idx = torch.tensor([k for k in ev_keys], dtype=torch.long, device=self.dev)
            self.lut[idx] = -1
            for k in ev_keys:
                self.free.append(self.resident.pop(k))
                self.lastuse.pop(k, None)

        got_keys, got_slots = [], []
        for i0 in range(0, len(need), self.batch):
            grp = need[i0:i0 + self.batch]

            def _load(pair):
                bi, key = pair
                cnz, cny, cnx = self.vol.cn
                cx = key % cnx
                cy = (key // cnx) % cny
                cz = key // (cnx * cny)
                p = self.vol.chunk_path(cz, cy, cx)
                try:
                    with open(p, "rb", buffering=0) as f:
                        n = f.readinto(memoryview(self.stage_np[bi * CHUNK_VOX:(bi + 1) * CHUNK_VOX]))
                except FileNotFoundError:
                    return key, False
                return key, (n == CHUNK_VOX)

            t0 = time.perf_counter()
            res = list(self.pool_exec.map(_load, list(enumerate(grp))))
            t1 = time.perf_counter()
            timer.add("disk_read", t1 - t0)

            for bi, (key, ok) in enumerate(res):
                if not ok:
                    self.absent.add(key)
                    continue
                slot = self.free.pop()
                self.pool[slot * CHUNK_VOX:(slot + 1) * CHUNK_VOX].copy_(
                    self.stage[bi * CHUNK_VOX:(bi + 1) * CHUNK_VOX], non_blocking=True)
                self.resident[key] = slot
                got_keys.append(key)
                got_slots.append(slot)
                self.bytes_read += CHUNK_VOX
                self.chunks_loaded += 1
            self._sync()                  # staging buffer is reused next batch
            timer.add("h2d_chunks", time.perf_counter() - t1)
        if got_keys:
            self.lut[torch.tensor(got_keys, dtype=torch.long, device=self.dev)] = \
                torch.tensor(got_slots, dtype=torch.int32, device=self.dev)
        for k in keys:
            if k in self.resident:
                self.lastuse[k] = band


# ---------------------------------------------------------------- render
class Resident:
    """The per-card state a render can KEEP: the compiled kernels, the open zarr
    and the GPU chunk pool.

    Measured on the V100, per process, with nothing pathological in it:
    `import torch` 7.4-8.8 s, CUDA context 1.9-3.7 s, NVRTC compile 0.4-0.8 s,
    pool allocation 0.6 s -- 11.3 s at OMP_NUM_THREADS=4 and 14.7 s unset. (The
    directory-scan and consolidated-metadata suspicions are refuted: the chunk
    directory listing and the .zarray open are 0.00 s.) None of it is avoidable
    per process; all of it is avoidable per RENDER, which is what this holds.

    The chunk pool is the part that compounds: it keeps chunks GPU-resident, so a
    second render of the same surface re-reads far less of the volume.
    """

    def __init__(self, vol_path, group_idx=0, device=0, pool_gb=None, io_threads=16,
                 step_fma=True, warp_fma=True, norm_fma=1):
        self.key = (os.path.realpath(vol_path), int(group_idx), int(device),
                    bool(step_fma), bool(warp_fma), int(norm_fma))
        torch.cuda.set_device(device)
        src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels.cu")).read()
        self.mod = Module(f"#define STEP_FMA {1 if step_fma else 0}\n"
                          f"#define WARP_FMA {1 if warp_fma else 0}\n"
                          f"#define NORM_FMA {int(norm_fma)}\n" + src, device=device)
        self.vol = ZarrU8(vol_path, level=group_idx)
        if pool_gb is None:
            freeb, _ = torch.cuda.mem_get_info(device)
            pool_gb = max(1.0, (freeb / 2**30) - 2.5)
        nslots = max(64, int(pool_gb * 2**30 // CHUNK_VOX))
        self.pool = None
        while self.pool is None:
            try:
                self.pool = ChunkPool(self.vol, nslots, f"cuda:{device}", io_threads=io_threads)
            except torch.OutOfMemoryError:
                if nslots <= 64:
                    raise
                nslots = max(64, nslots // 2)
                torch.cuda.empty_cache()
        # A halved pool renders correctly (bands split to fit) but slower; say so.
        want = max(64, int(pool_gb * 2**30 // CHUNK_VOX))
        if nslots < want:
            print(f"render_gpu: chunk pool OOM-halved to {nslots} of {want} slots "
                  f"({nslots * CHUNK_VOX / 2**30:.2f} GB) on cuda:{device}", file=sys.stderr, flush=True)
        self.nslots = nslots


def render(seg_dir, vol_path, out_dir, num_slices=65, slice_step=1.0, tgt_scale=1.0,
           group_idx=0, scale_seg=1.0, band_h=128, device=0, pool_gb=None,
           step_fma=True, warp_fma=True, norm_fma=1, write_tif=True, io_threads=16, max_bands=None, verbose=True,
           resident=None):
    timer = Timer()
    t_start = time.perf_counter()
    dev = f"cuda:{device}"
    torch.cuda.set_device(device)

    if resident is not None:
        mod = resident.mod
    else:
        src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels.cu")).read()
        mod = Module(f"#define STEP_FMA {1 if step_fma else 0}\n"
                     f"#define WARP_FMA {1 if warp_fma else 0}\n"
                     f"#define NORM_FMA {int(norm_fma)}\n" + src, device=device)

    pts, scale = load_surface(seg_dir)
    srows, scols, _ = pts.shape
    vmask = valid_mask(pts)
    ncache = normal_cache(pts)   # numpy reference (overwritten by GPU build below)

    ds_scale = np.float32(2.0 ** -group_idx)
    render_scale = float(tgt_scale) * (float(scale_seg) * 1.0 * float(ds_scale))

    sxr = render_scale / float(scale[0])
    syr = render_scale / float(scale[1])
    full_w = max(1, int(round(scols * sxr)))
    full_h = max(1, int(round(srows * syr)))

    # offsets (buildOffsetList, no accum offsets)
    center = 0.5 * (max(1, num_slices) - 1.0)
    offsets = np.array([np.float32((zi - center) * slice_step) for zi in range(num_slices)],
                       dtype=np.float32)

    vol = resident.vol if resident is not None else ZarrU8(vol_path, level=group_idx)
    SZ, SY, SX = vol.shape
    CNZ, CNY, CNX = vol.cn

    if pool_gb is None:
        freeb, _ = torch.cuda.mem_get_info(device)
        pool_gb = max(1.0, (freeb / 2**30) - 2.5)
    nslots = max(64, int(pool_gb * 2**30 // CHUNK_VOX))
    pool = resident.pool if resident is not None else None
    if pool is not None:
        nslots = resident.nslots
    while pool is None:
        try:
            pool = ChunkPool(vol, nslots, dev, io_threads=io_threads)
        except torch.OutOfMemoryError:
            if nslots <= 64:
                raise
            nslots = max(64, nslots // 2)
            torch.cuda.empty_cache()
            print(f"  pool alloc OOM, retrying with {nslots} slots", flush=True)
    if verbose:
        print(f"grid {srows}x{scols} scale {scale} -> render {full_w}x{full_h} "
              f"({num_slices} slices)  pool {nslots} slots ({nslots*2/1024:.1f} GB)", flush=True)

    d_pts = torch.from_numpy(np.ascontiguousarray(pts)).to(dev)
    d_val = torch.from_numpy(np.ascontiguousarray(vmask)).to(dev)
    d_nc = torch.empty(srows * scols * 3, dtype=torch.float32, device=dev)
    mod.launch("build_normals", ((srows * scols + 255) // 256,), (256,),
               [d_pts, ctypes.c_int(srows), ctypes.c_int(scols), d_nc])
    torch.cuda.synchronize()
    d_off = torch.from_numpy(offsets).to(dev)

    W = full_w
    out_host = np.zeros((num_slices, full_h, W), dtype=np.uint8)

    d_base = torch.empty(band_h * W * 3, dtype=torch.float32, device=dev)
    d_dirs = torch.empty(band_h * W * 3, dtype=torch.float32, device=dev)
    d_out = torch.empty(num_slices * band_h * W, dtype=torch.uint8, device=dev)
    h_stage = torch.empty(num_slices * band_h * W, dtype=torch.uint8).pin_memory()
    h_stage_np = h_stage.numpy()

    # gen() geometry constants (float32 ops matched to QuadSurface::gen)
    f32 = np.float32
    center_x = f32(float(scols) / 2.0 / float(scale[0]))
    center_y = f32(float(srows) / 2.0 / float(scale[1]))
    sxw = float(scale[0]) / float(render_scale)     # double, as in gen()
    syw = float(scale[1]) / float(render_scale)

    timer.add("setup", time.perf_counter() - t_start)
    nbands = (full_h + band_h - 1) // band_h
    if max_bands:
        nbands = min(nbands, max_bands)

    # Work list of (y0, rows). A band whose chunks do not fit the pool at once is
    # split in two and retried (PoolTooSmall), never rendered with chunks missing.
    # `tick` is the LRU clock: one step per attempted band, monotone across renders.
    work = [(bi * band_h, min(band_h, full_h - bi * band_h)) for bi in range(nbands)][::-1]
    if resident is not None:
        pool.absent.clear()               # a chunk missing at an earlier render may exist now
    tick = 0
    band_splits = 0
    done_bands = 0
    while work:
        y0, dh = work.pop()
        tick += 1
        # computeCanvasOrigin + crop offset (crop is the full canvas here)
        u0 = f32(f32(-0.5) * f32(full_w - f32(1.0)))
        v0 = f32(f32(-0.5) * f32(full_h - f32(1.0)))
        v0 = f32(v0 + f32(y0))
        # gen(): ul = internal_loc(offset/scale + _center, ptr=0, _scale)
        ul_x = f32(f32(f32(u0) / f32(render_scale)) + center_x) * scale[0]
        ul_y = f32(f32(f32(v0) / f32(render_scale)) + center_y) * scale[1]
        ox = float(ul_x) - 4.0 * sxw
        oy = float(ul_y) - 4.0 * syw

        n_px = dh * W
        blk, grd = 256, (n_px + 255) // 256

        torch.cuda.synchronize(); t0 = time.perf_counter()
        mod.launch("gen_band", (grd,), (blk,), [
            d_pts, d_val, d_nc,
            ctypes.c_int(srows), ctypes.c_int(scols),
            ctypes.c_float(ox), ctypes.c_float(oy),
            ctypes.c_float(sxw), ctypes.c_float(syw),
            ctypes.c_int(W), ctypes.c_int(dh),
            ctypes.c_float(scale_seg), ctypes.c_float(float(ds_scale)),
            d_base, d_dirs])
        torch.cuda.synchronize(); t1 = time.perf_counter()
        timer.add("k_gen", t1 - t0)

        pool.marks.zero_()
        mod.launch("mark_chunks", (grd,), (blk,), [
            d_base, d_dirs, d_off, ctypes.c_int(num_slices),
            ctypes.c_int(W), ctypes.c_int(dh),
            ctypes.c_int(SZ), ctypes.c_int(SY), ctypes.c_int(SX),
            ctypes.c_int(CNZ), ctypes.c_int(CNY), ctypes.c_int(CNX),
            pool.marks])
        torch.cuda.synchronize(); t2 = time.perf_counter()
        timer.add("k_mark", t2 - t1)

        keys = torch.nonzero(pool.marks, as_tuple=False).flatten().cpu().numpy()
        t3 = time.perf_counter(); timer.add("marks_d2h", t3 - t2)

        try:
            pool.ensure(keys.tolist(), pool.band_base + tick, timer)
        except PoolTooSmall as e:
            if dh <= 1:
                raise RuntimeError(f"chunk pool too small for a single row: {e}") from e
            h1 = dh // 2
            work.append((y0 + h1, dh - h1))
            work.append((y0, h1))
            band_splits += 1
            continue
        t4 = time.perf_counter()

        mod.launch("sample_band", (grd,), (blk,), [
            d_base, d_dirs, d_off, ctypes.c_int(num_slices),
            ctypes.c_int(W), ctypes.c_int(dh),
            ctypes.c_int(SZ), ctypes.c_int(SY), ctypes.c_int(SX),
            ctypes.c_int(CNY), ctypes.c_int(CNX),
            pool.lut, pool.pool, d_out])
        torch.cuda.synchronize(); t5 = time.perf_counter()
        timer.add("k_sample", t5 - t4)

        nb = num_slices * n_px
        h_stage[:nb].copy_(d_out[:nb], non_blocking=True)
        torch.cuda.synchronize()
        src_v = h_stage_np[:nb].reshape(num_slices, dh, W)
        for zi in range(num_slices):
            out_host[zi, y0:y0 + dh, :] = src_v[zi]
        t6 = time.perf_counter()
        timer.add("d2h_out", t6 - t5)

        done_bands += 1
        if verbose and (done_bands % 20 == 1 or not work):
            el = time.perf_counter() - t_start
            print(f"  band {done_bands} (rows {y0}+{dh}/{full_h})  {el:.1f}s  resident={len(pool.resident)}",
                  flush=True)

    pool.band_base += tick + 1            # keep the LRU clock monotone across renders
    if band_splits and verbose:
        print(f"  pool of {nslots} slots split {band_splits} band(s) to fit", flush=True)
    t_w0 = time.perf_counter()
    # Write the stack as ONE .npy beside the TIFFs. Measured on a real 65-layer
    # render: a consumer reads it in 9.4-10.3 s against 20.4-23.7 s for the 65
    # TIFFs (and 0.00-0.03 s memory-mapped), for the same bytes -- max|delta| 0.
    # The TIFFs stay, because prospect, the masks and every external reader expect
    # them; this is an additional, cheaper door into the same data.
    # 🔴 OFF BY DEFAULT since 2026-09-29: nothing in production reads stack.npy (only
    # scripts/bench/*), and it doubled every stack on disk -- a 3.4 kpx segment's 1.2 GB native
    # stack carried a second 1.2 GB copy, and after normalize_layers the .npy was not even the
    # same pixels as the TIFFs (raw vs normalised). VC_RENDER_NPY=1 restores it for a benchmark.
    if write_tif and os.environ.get("VC_RENDER_NPY", "0") == "1":
        os.makedirs(out_dir, exist_ok=True)
        tmp = os.path.join(out_dir, ".stack.npy.tmp.npy")
        np.save(tmp, out_host)
        os.replace(tmp, os.path.join(out_dir, "stack.npy"))   # atomic: no half file
    timer.add("npy_write", time.perf_counter() - t_w0)
    t_w0 = time.perf_counter()
    if write_tif:
        os.makedirs(out_dir, exist_ok=True)
        with ThreadPoolExecutor(max_workers=min(32, num_slices)) as ex:
            list(ex.map(lambda zi: tifffile.imwrite(os.path.join(out_dir, f"{zi:02d}.tif"),
                                                    np.ascontiguousarray(out_host[zi])), range(num_slices)))
    timer.add("tif_write", time.perf_counter() - t_w0)

    total = time.perf_counter() - t_start
    return dict(total=total, timers=timer.report(), out=out_host,
                shape=(full_h, full_w), chunks_loaded=pool.chunks_loaded,
                bytes_read=pool.bytes_read, nslots=nslots, band_splits=band_splits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", required=True)
    ap.add_argument("--vol", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("-n", "--num-slices", type=int, default=65)
    ap.add_argument("--slice-step", type=float, default=1.0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("-g", "--group", type=int, default=0)
    ap.add_argument("--band-h", type=int, default=128)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--pool-gb", type=float, default=None)
    ap.add_argument("--no-step-fma", action="store_true")
    ap.add_argument("--no-warp-fma", action="store_true")
    ap.add_argument("--norm-fma", type=int, default=1)
    ap.add_argument("--no-tif", action="store_true")
    ap.add_argument("--io-threads", type=int, default=16)
    ap.add_argument("--max-bands", type=int, default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    r = render(a.seg, a.vol, a.out, num_slices=a.num_slices, slice_step=a.slice_step,
               tgt_scale=a.scale, group_idx=a.group, band_h=a.band_h, device=a.device,
               pool_gb=a.pool_gb, step_fma=not a.no_step_fma, warp_fma=not a.no_warp_fma, norm_fma=a.norm_fma, write_tif=not a.no_tif,
               io_threads=a.io_threads, max_bands=a.max_bands)
    out = r.pop("out")
    print(f"\nTOTAL {r['total']:.2f}s  shape={r['shape']}  "
          f"chunks={r['chunks_loaded']} ({r['bytes_read']/2**30:.2f} GiB)")
    for k, v in r["timers"].items():
        print(f"  {k:14s} {v:8.2f}s  {100*v/r['total']:5.1f}%")
    print(f"  nonzero_frac   {float((out > 0).mean()):.6f}   mean={out.mean():.4f}")
    if a.json:
        json.dump({k: v for k, v in r.items()}, open(a.json, "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
