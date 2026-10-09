#!/usr/bin/env python3
"""Throughput of the dense students vs the production ink9um_student vs the Reader v2 teacher, one GPU, one big segment.

Students: one face over the whole segment through the deployed tiling (2048 px tiles + halo, fp16), after a warm-up
pass, 3 repeats -> MVox/s = 17 x H x W / wall (host->device, normaliser, net, device->host included; the x array is
already in RAM). Teacher: Reader v2's own timing from the targets run on the SAME GPU (meta.json teacher_both_faces_s,
two faces) over every distillation segment -> MVox/s per face, p10/p50/p90 over segments (wrap/IO excluded).
Usage: rv2_bench.py --src <repo>/src --seg <data_rv2/SEG> --rv2-data <data_rv2> --ckpt a=path ... --prod <ink9um_student.ckpt>
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--seg", required=True)
    ap.add_argument("--rv2-data", required=True)
    ap.add_argument("--ckpt", action="append", default=[]); ap.add_argument("--prod", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.src)
    import torch
    from vesuvius_pipeline.stages.ink_models import ink9um_student as S
    from vesuvius_pipeline.stages.ink_models import reader_v2_dense as R
    torch.backends.cudnn.benchmark = True
    x = np.load(os.path.join(a.seg, "x.npy"))
    arrs = [x[j] for j in range(17)]
    H, W = arrs[0].shape
    gpu = torch.cuda.get_device_name(0)
    res = {"segment": os.path.basename(a.seg), "shape": [17, H, W], "MVox": round(17 * H * W / 1e6, 1), "gpu": gpu, "arms": {}}

    def timeit(fn):
        fn()                                               # warm-up (cuDNN autotune)
        ts = []
        for _ in range(3):
            torch.cuda.synchronize(); t0 = time.time(); fn(); torch.cuda.synchronize(); ts.append(time.time() - t0)
        return ts

    for spec in a.ckpt:
        arm, p = spec.split("=", 1)
        m, ck = R.load(p, "cuda")
        ts = timeit(lambda: R.predict_array(m, arrs, ck["config"]["norm"], "cuda", halo=int(ck["config"]["halo"])))
        res["arms"][arm] = {"wall_s": [round(t, 2) for t in ts], "MVox_per_s_median": round(17 * H * W / 1e6 / float(np.median(ts)), 1),
                            "params": ck.get("params"), "rf_total": ck.get("rf_total")}
        print(arm, res["arms"][arm], flush=True)
        del m; torch.cuda.empty_cache()
    if a.prod:
        m, ck = S.load(a.prod, "cuda")
        ts = timeit(lambda: S.predict_array(m, arrs, None, "cuda", norm=ck["config"]["norm"], halo=int(ck["config"]["halo"])))
        res["arms"]["ink9um_student"] = {"wall_s": [round(t, 2) for t in ts],
                                         "MVox_per_s_median": round(17 * H * W / 1e6 / float(np.median(ts)), 1),
                                         "params": ck.get("params")}
        print("ink9um_student", res["arms"]["ink9um_student"], flush=True)
    rates = []
    for mj in glob.glob(os.path.join(a.rv2_data, "*", "meta.json")):
        mt = json.load(open(mj))
        if mt.get("teacher_both_faces_s"):
            rates.append(2 * 17 * mt["shape"][1] * mt["shape"][2] / 1e6 / mt["teacher_both_faces_s"])
    if rates:
        res["arms"]["teacher_rv2"] = {"MVox_per_s_per_face_p10_p50_p90": [round(float(np.percentile(rates, q)), 1) for q in (10, 50, 90)],
                                      "n_segments": len(rates), "note": "targets run on this GPU, stride 64 Hann, AMP; "
                                      "concurrent with student training for part of the run (an upper bound on its cost)"}
        print("teacher_rv2", res["arms"]["teacher_rv2"], flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
