"""Expected grow throughput / cost from the vast benchmark model (EXTRAPOLATED, not measured on the target box).

Source: docs/experiments/vast_benchmark_2026-10-06 (model.json, model.txt). Slot speed = k x PassMark single-thread
(k = 0.0254 cm2/h per ST point; 7 CPU models, R2 0.81, NOT validated out of sample, +-25 %); machine = physical cores x slot x 1.2
(SMT, one host); production-equivalent verified cm2/h = x prod_factor 0.2 (calibration range 0.13..0.33 across hosts).
The benchmark workload is round-1 0.5-1 cm2 seeds; late resume rounds, the 16 GiB cache and the guard are NOT in it.
"""
from __future__ import annotations

K_PER_ST = 0.025386313465783666
SMT = 1.2
PROD_FACTOR = 0.2
PROD_FACTOR_RANGE = (0.13, 0.33)
GB_PER_GROW = 3.0          # DEPLOY.md: RAM >= max(64, 3 x grows + 40) with the production 16 GiB cache
RAM_BASE_GB = 40.0


def estimate(passmark_st: float, phys_cores: int, target_cm2: float = 100.0, usd_per_hour: float | None = None,
             smt: bool = True) -> dict:
    slot = K_PER_ST * passmark_st
    machine = phys_cores * slot * (SMT if smt else 1.0)
    out = {"label": "EXTRAPOLATED (benchmark model, not measured on this box)",
           "slot_cm2_h_benchmark_workload": round(slot, 1), "machine_cm2_h_benchmark_workload": round(machine, 1)}
    lo, hi = (machine * f for f in PROD_FACTOR_RANGE)
    prod = machine * PROD_FACTOR
    out["prod_equiv_verified_cm2_h"] = {"low": round(lo, 1), "mid": round(prod, 1), "high": round(hi, 1)}
    out["hours_for_target_cm2"] = {"target_cm2": target_cm2, "mid": round(target_cm2 / prod, 2) if prod else None,
                                   "slow_end": round(target_cm2 / lo, 2) if lo else None}
    if usd_per_hour is not None and prod:
        out["usd_per_100_cm2"] = {"mid": round(usd_per_hour / prod * 100, 3), "range": [round(usd_per_hour / hi * 100, 3), round(usd_per_hour / lo * 100, 3)],
                                  "note": "compute only; data transfer/storage excluded (dominant for short rentals)"}
    out["ram_gb_needed"] = ram_needed_gb(phys_cores)
    return out


def ram_needed_gb(n_grows: int) -> float:
    return max(64.0, GB_PER_GROW * n_grows + RAM_BASE_GB)
