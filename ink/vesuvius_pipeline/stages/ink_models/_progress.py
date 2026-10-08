"""Live partial results for the zoo wrappers.

The wrappers share one blending loop (`pred`/`cnt` accumulators over unfolded tiles), so
they share one hook: `writer()` before the loop, `publish()` when it is due. Everything
here is defensive -- an ink family must never fail because nobody could watch it.
"""
from __future__ import annotations


def writer(n_tiles: int, shape, model: str = "", origins=None, tile: int = 0,
           stride: int | None = None):
    """`origins`/`tile` publish the TILE SCHEDULE once, so the viewer can draw what is
    pending and what is running instead of only how far the bar got."""
    try:
        from .. import ink_progress as IP
        pw = IP.from_env(n_tiles, shape=shape, model=model)
        if pw is not None and origins and tile:
            pw.set_schedule(origins, tile, stride=stride)
        return pw
    except Exception:                        # noqa: BLE001 - progress is never a gate
        return None


def publish(pw, done_n, pred, cnt, h0, w0, origins, tile, batch_i, done=False,
            running=None):
    """Current blended map (pred/cnt, cropped back to the unpadded size) plus the bbox
    of the tiles this batch filled."""
    try:
        import torch
        cur = (pred / cnt.clamp(min=1.0))[:h0, :w0]
        sc = 1024.0 / max(h0, w0)
        if sc < 1.0:
            cur = torch.nn.functional.interpolate(cur[None, None], scale_factor=sc,
                                                  mode="area")[0, 0]
        bbox = None
        if origins:
            ys = [o[0] for o in origins]; xs = [o[1] for o in origins]
            bbox = (min(ys), min(xs), max(ys) + tile, max(xs) + tile)
        # what has actually been written, from the blender's own count array: the tiles a
        # mask skipped were never run, and no schedule can know that
        cov = (cnt[:h0, :w0] > 0).float()
        if sc < 1.0:
            cov = torch.nn.functional.interpolate(cov[None, None], scale_factor=sc, mode="area")[0, 0]
        pw.update(done_n, torch.nan_to_num(cur).clamp(0, 1), bbox=bbox, batch=batch_i,
                  done=done, running=running, coverage=(cov > 0).float())
    except Exception:                        # noqa: BLE001 - a dropped frame, not a failed run
        pass
