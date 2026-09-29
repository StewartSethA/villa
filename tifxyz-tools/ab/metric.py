#!/usr/bin/env python3
"""Clean cm2 per CPU-hour, per arm, with a PAIRED seed bootstrap of the ratio to arm A.

Reads `ab/ovn20260929/rows_reaudited.jsonl` (one row per seed x arm, re-audited 2026-09-29), applies the same correction and the same clean-area
rule as scripts/growth_ab/reaudit_metric_table.py (nothing_left rows -> area 0; clean = selfx
density exactly 0 AND verified/area >= 0.75), and reports per arm: self-intersection-free count,
clean / raw cm2, CPU-h, clean cm2 per CPU-h, and the ratio to arm A with a PAIRED bootstrap over
seeds (the same resampled seed set scores both arms). Arm G (separate, unpaired draw) is excluded.

usage: python ab/metric.py ab/ovn20260929/rows_reaudited.jsonl [--boot 4000] [--seed 7]
"""
import argparse
import collections
import json

import numpy as np

MATERIAL_FRAC_TH = 0.75
ARMS = "ABDEF"


def correct_row(r):
    r = dict(r)
    if r.get("stop_reason") and "nothing_left" in r["stop_reason"]:
        r["area_cm2"], r["verified_cm2"] = 0.0, 0.0
    else:
        real = r.get("_real_published_area_cm2")
        if isinstance(real, (int, float)) and r.get("_path_exists"):
            r["area_cm2"] = real
    return r


def clean_area(r):
    if r.get("selfx_density") != 0.0:
        return 0.0
    v, a = r.get("verified_cm2"), r.get("area_cm2") or 0.0
    if v is not None and a > 0 and v / a < MATERIAL_FRAC_TH:
        return 0.0
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    raw = [json.loads(line) for line in open(a.rows) if line.strip()]
    rows = [correct_row(r) for r in raw if r.get("arm") in ARMS]
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["seed_id"]][r["arm"]] = r
    common = sorted(s for s in by if all(x in by[s] for x in ARMS))
    print(f"rows {len(rows)}  paired seeds {len(common)}")

    def stat(ss, arm):
        rs = [by[s][arm] for s in ss]
        return sum(clean_area(r) for r in rs) / (sum(r.get("cpu_s") or 0.0 for r in rs) / 3600.0)

    for arm in ARMS:
        rs = [by[s][arm] for s in common]
        print(f"{arm}: selfx0 {sum(r.get('selfx_density') == 0.0 for r in rs)}/{len(rs)}  "
              f"clean {sum(clean_area(r) for r in rs):.2f} cm2  raw {sum(r.get('area_cm2') or 0 for r in rs):.2f} cm2  "
              f"cpu {sum(r.get('cpu_s') or 0 for r in rs) / 3600:.2f} h  clean/cpu-h {stat(common, arm):.4f}")
    rng = np.random.default_rng(a.seed)
    for arm in ARMS[1:]:
        v = []
        for _ in range(a.boot):
            ss = [common[i] for i in rng.integers(0, len(common), len(common))]
            base = stat(ss, "A")
            if base > 0:
                v.append(stat(ss, arm) / base)
        lo, hi = np.percentile(v, [2.5, 97.5])
        print(f"{arm}/A {stat(common, arm) / stat(common, 'A'):.2f}x  paired 95% CI [{lo:.2f}, {hi:.2f}]  "
              f"(valid resamples {len(v)}/{a.boot})")
    for sc in sorted({by[s]["A"]["scroll"] for s in common}):
        ss = [s for s in common if by[s]["A"]["scroll"] == sc]
        print(sc, len(ss), " ".join(f"{x}={stat(ss, x):.3f}" for x in ARMS))
    for arm in ARMS:
        c = np.array([clean_area(by[s][arm]) for s in common])
        print(f"{arm} per-seed clean cm2 p10/p50/p90/max "
              + "/".join(f"{q:.2f}" for q in np.percentile(c, [10, 50, 90, 100])) + f"  seeds>=1cm2 {(c >= 1).sum()}")


if __name__ == "__main__":
    main()
