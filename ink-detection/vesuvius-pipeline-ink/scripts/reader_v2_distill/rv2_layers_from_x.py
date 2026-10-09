#!/usr/bin/env python3
"""Turn distillation segments (<data>/<seg>/x.npy, the ink9um teacher's own 17-layer uint8 window) back into a
17-file layer stack, so `scripts/hires_ink/ink9um_teacher_targets.py --ckpt <reader-v2>` can run Reader v2 on
EXACTLY the pixels the ink_9um targets were made from (same window, same crop, same validity mask = layer 8 > 0).
Writes <out>/<seg>/layers/NN.tif and <out>/list.tsv (seg<TAB>layers_dir). Resumable; one bad segment is reported
and skipped. Usage: rv2_layers_from_x.py <data> <out> [--only a,b]
"""
import argparse
import os

import numpy as np
import tifffile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("out")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    keep = set(filter(None, a.only.split(",")))
    rows = []
    for seg in sorted(os.listdir(a.data)):
        if keep and seg not in keep:
            continue
        src = os.path.join(a.data, seg, "x.npy")
        if not os.path.exists(os.path.join(a.data, seg, "meta.json")) or not os.path.exists(src):
            continue
        ld = os.path.abspath(os.path.join(a.out, seg, "layers"))   # absolute: the teacher symlinks to it
        try:
            if not os.path.exists(os.path.join(ld, "16.tif")):
                x = np.load(src, mmap_mode="r")
                assert x.shape[0] == 17 and x.dtype == np.uint8, (x.shape, x.dtype)
                os.makedirs(ld, exist_ok=True)
                for k in range(17):
                    tifffile.imwrite(os.path.join(ld, f"{k:02d}.tif.part"), np.asarray(x[k]))
                    os.replace(os.path.join(ld, f"{k:02d}.tif.part"), os.path.join(ld, f"{k:02d}.tif"))
            rows.append((seg, ld))
        except Exception as e:                       # noqa: BLE001 - one bad segment must not stop the batch
            print(f"[layers] {seg}: FAILED {type(e).__name__}: {e}", flush=True)
    with open(os.path.join(a.out, "list.tsv"), "w") as fh:
        for seg, ld in rows:
            fh.write(f"{seg}\t{ld}\n")
    print(f"[layers] {len(rows)} segments listed in {a.out}/list.tsv", flush=True)


if __name__ == "__main__":
    main()
