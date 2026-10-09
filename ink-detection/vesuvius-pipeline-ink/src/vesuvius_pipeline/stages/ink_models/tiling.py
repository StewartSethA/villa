"""Windowed inference over one rendered layer stack, on the device.

Every ink wrapper here does the same thing: cut a (C, H, W) stack into overlapping
tiles, run the model, and average the outputs back. Doing that with one python slice per
window costs thousands of tiny kernel launches, and doing the clip/scale/transpose on the
host costs hundreds of megabytes of numpy copies per call. Measured by the ensemble study
(FINDINGS 64) at 1024^2 per face: TimeSformer 30 s -> 1.4 s, i3d 46 s -> 1.5 s,
resnet3d_1667 177 s -> 1.5 s, GPU utilisation 0-1 % -> 59-83 %.

`unfold_tiles` is the replacement, and `naive_tiles` is the reference implementation it
must agree with exactly -- `scripts/tests/test_ink_tiling.py` pins that.
"""
from __future__ import annotations


def tile_origins(H: int, W: int, tile: int, stride: int) -> list[tuple[int, int]]:
    """Top-left corners, row-major -- the order both implementations must produce."""
    return [(y, x) for y in range(0, H - tile + 1, stride) for x in range(0, W - tile + 1, stride)]


def unfold_tiles(dvol, tile: int, stride: int):
    """(C, H, W) tensor -> ((N, C, tile, tile) view, origins). One op, no python loop."""
    C, H, W = dvol.shape
    ys = len(range(0, H - tile + 1, stride))
    xs = len(range(0, W - tile + 1, stride))
    u = dvol.unfold(1, tile, stride).unfold(2, tile, stride)      # (C, ny, nx, tile, tile)
    u = u.permute(1, 2, 0, 3, 4).reshape(ys * xs, C, tile, tile)
    return u, tile_origins(H, W, tile, stride)


def naive_tiles(dvol, tile: int, stride: int):
    """The reference: one slice per window. Kept so the fast path has something to be
    equal TO -- a faster tiler that quietly reorders or drops a window would show up as a
    slightly different prediction map and nothing else."""
    import torch
    origins = tile_origins(dvol.shape[1], dvol.shape[2], tile, stride)
    return torch.stack([dvol[:, y:y + tile, x:x + tile] for y, x in origins]), origins
