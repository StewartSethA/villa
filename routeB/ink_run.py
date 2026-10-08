"""Run ink families on one rendered stack.  usage: ink_run.py --layers DIR --mask PNG --out DIR --families a,b
Imports the vendored vesuvius_pipeline.stages.ink_models from <branch root>/ink (cwd = <root>/models so 'var/models/<ckpt>' resolves)."""
import argparse
import os
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ink"))
FAMILIES = {"sharp_dense": ("reader_v2_dense", "sharp"), "reader_v2_dense": ("reader_v2_dense", None),
            "ink9um_student": ("ink9um_student", None), "sharp_dense_human": ("reader_v2_dense", "human_cv14k")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--families", required=True)
    a = ap.parse_args()
    from vesuvius_pipeline.stages.ink_models import FAMILIES as MODS
    rc = 0
    for fam in a.families.split(","):
        mod, arm = FAMILIES[fam]
        out_png = os.path.join(a.out, f"{fam}.png")
        try:
            kw = {"arm": arm} if arm else {}
            ok, why = MODS[mod].available(None, **kw)
            if not ok:
                print(f"INK FAIL {fam}: weights missing: {why}", flush=True)
                rc = 1
                continue
            ok = MODS[mod].predict(a.layers, a.mask, out_png, 0, **kw)
            print(f"INK {'OK' if ok else 'FAIL'} {fam} -> {out_png}", flush=True)
            rc |= 0 if ok else 1
        except Exception:
            traceback.print_exc()
            print(f"INK FAIL {fam}: exception", flush=True)
            rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
