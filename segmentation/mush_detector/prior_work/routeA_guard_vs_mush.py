#!/usr/bin/env python3
"""Task 3(b): Route A guard-stop / selfcross / guard_nothing_left rates for seeds
inside vs outside the PHerc0125 mush mask."""
import json
import sqlite3

import numpy as np
import zarr

DB = "/home/seth/ScrollPrizeTutorial/var/pipeline.db"
MASK = "/mnt/raid7/experiments/mush_labels/PHerc0125/mush_mask.zarr"


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - half) / denom, (centre + half) / denom)


def main():
    db = sqlite3.connect(DB)
    cur = db.cursor()
    cur.execute("""
        select s.seg,
               max(case when m.name='seed_x' then m.value end) as x,
               max(case when m.name='seed_y' then m.value end) as y,
               max(case when m.name='seed_z' then m.value end) as z
        from segment s join metric m on m.seg = s.seg
        where s.scroll='PHerc0125' and m.name in ('seed_x','seed_y','seed_z')
        group by s.seg
        having x is not null and y is not null and z is not null
    """)
    rows = cur.fetchall()
    print(f"{len(rows)} PHerc0125 segments with a recorded seed position", flush=True)

    mush = zarr.open(MASK, mode="r")
    blob = mush["blob"]
    mlevel = int(mush.attrs["mush_level"])
    mf = 2 ** mlevel
    z_lo, z_hi = mush.attrs["z_valid_range_level0"]
    Zb, Yb, Xb = blob.shape

    segs, xs, ys, zs = zip(*rows)
    xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
    in_range = (zs >= z_lo) & (zs <= z_hi)
    mz = np.round(zs / mf).astype(np.int64)
    my = np.round(ys / mf).astype(np.int64)
    mx = np.round(xs / mf).astype(np.int64)
    ok = in_range & (mz >= 0) & (mz < Zb) & (my >= 0) & (my < Yb) & (mx >= 0) & (mx < Xb)
    blobval = np.zeros(len(segs), np.uint8)
    idx_ok = np.flatnonzero(ok)
    for z0 in np.unique(mz[idx_ok]):
        sel = idx_ok[mz[idx_ok] == z0]
        plane = np.asarray(blob[z0])
        blobval[sel] = plane[my[sel], mx[sel]]
    inside = ok & (blobval >= 128)
    outside = ok & (blobval < 128)
    print(f"seeds with z in mush-labelled range [{z_lo},{z_hi}]: {ok.sum()} "
          f"(inside mush: {inside.sum()}, outside: {outside.sum()}); "
          f"{(~in_range).sum()} segments have seed z outside the labelled range and are excluded", flush=True)

    seg_set_in = set(np.array(segs)[inside])
    seg_set_out = set(np.array(segs)[outside])
    all_segs = seg_set_in | seg_set_out
    qmarks = ",".join("?" * len(all_segs))
    cur.execute(f"""
        select seg, reason from attempt
        where stage='grow' and seg in ({qmarks}) and reason is not null
    """, list(all_segs))
    reasons = {}
    for seg, reason in cur.fetchall():
        reasons.setdefault(seg, []).append(reason)

    def rates(seg_set, label):
        n = len(seg_set)

        def k_of(pred):
            k = sum(1 for s in seg_set if any(pred(r) for r in reasons.get(s, [])))
            return {"k": k, "rate": k / n if n else None, "ci95": wilson_ci(k, n)}

        out = {
            "label": label, "n_segments": n,
            "any_guard_stop": k_of(lambda r: r.startswith("guard_")),
            "guard_selfcross_nonzero": k_of(lambda r: r == "guard_selfcross_nonzero"),
            "guard_nothing_left": k_of(lambda r: r == "guard_nothing_left"),
            "efficiency_floor": k_of(lambda r: r == "efficiency_floor"),
            "interrupted_any": k_of(lambda r: r.startswith("interrupted")),
        }
        print(label, out, flush=True)
        return out

    result = {"scroll": "PHerc0125", "mask_z_valid_range_level0": [z_lo, z_hi],
              "n_segments_total_with_seed": len(rows), "n_excluded_outside_z_range": int((~in_range).sum()),
              "note": ("ALL 50 guard_* reasons that occur anywhere in PHerc0125's grow attempts "
                       "(23 guard_roughness, 27 guard_selfcross_nonzero) are on 'PHerc0125_merged_*' "
                       "segments, which carry no seed_x/y/z metric (merged segments are not re-seeded) "
                       "and so cannot be spatially classified inside/outside mush at all -- the guard_*/"
                       "selfcross/guard_nothing_left comparison below is therefore a TRUE ZERO for every "
                       "normally-seeded segment, not a null result from low power on those specific reasons. "
                       "efficiency_floor and interrupted_any are reported as better-powered proxies for "
                       "'this seed struggled to grow'."),
              "inside_mush": rates(seg_set_in, "inside_mush"),
              "outside_mush": rates(seg_set_out, "outside_mush")}
    with open("/home/seth/ScrollPrizeTutorial/docs/experiments/mush_labels/PHerc0125_routeA_guard_vs_mush.json", "w") as fh:
        json.dump(result, fh, indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
