"""Family `ensemble`: a fitted linear combination of the other families' prediction maps.

This family runs no network. Every other ink family has already written its PNG for the
segment; this one reads them, applies a weight vector fitted held-out by mesh
(scripts/ensemble/eval.py), and writes one more map. That is why it is cheap, and why it
is also fragile in a specific way: a weight vector is only meaningful over the exact
feature list it was fitted on, so a missing family is refused, never silently dropped.

The weights ship as JSON, `var/models/ensemble_<name>.json`:

    {"features": ["gp_s5_fwd", ...], "weights": [...], "bias": -1.2,
     "fitted_on": "...", "auc": {...}}

Gated off by default (`VPIPE_ENSEMBLE=1`), like every family that has not earned a place
in the default set.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

SPEC = {
    "kind": "post-hoc combiner",
    "inputs": "one prediction PNG per member family, in the segment's own frame",
    "um_per_px": 7.91,
    "normalisation": "each member map / 255, then logit",
    "license": "n/a (weights fitted here)",
}


def enabled() -> bool:
    return os.environ.get("VPIPE_ENSEMBLE", "0") == "1"


def _root(fleet):
    from ... import config
    return fleet.root or config.repo_root()


def weight_files(fleet) -> dict[str, str]:
    """name -> ensemble_<name>.json. The rules ship WITH THE PACKAGE (they are a few
    numbers, not weights) so a fresh clone has them; var/models, which is gitignored,
    overrides by name for a locally refitted vector."""
    out: dict[str, str] = {}
    for d in (os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights"),
              os.path.join(str(_root(fleet)), "var", "models")):
        for p in sorted(glob.glob(os.path.join(d, "ensemble_*.json"))):
            out[os.path.basename(p)[len("ensemble_"):-len(".json")]] = p
    return out


def available(fleet) -> tuple[bool, str]:
    if not enabled():
        return False, "ensemble: VPIPE_ENSEMBLE is not 1"
    w = weight_files(fleet)
    if not w:
        return False, "ensemble: no var/models/ensemble_*.json"
    return True, f"ensemble: {sorted(w)}"


def load_weights(path: str) -> dict:
    w = json.load(open(path))
    if w.get("rule") == "median":
        w.setdefault("weights", [1.0] * len(w["features"]))
    if len(w["features"]) != len(w["weights"]):
        raise ValueError(f"{path}: {len(w['features'])} features but {len(w['weights'])} weights")
    return w


def face_of(feature: str) -> str:
    return feature.rsplit("_", 1)[1]


def family_of(feature: str) -> str:
    return feature.rsplit("_", 1)[0]


def combine(maps: dict[str, np.ndarray], w: dict, eps: float = 1e-4) -> np.ndarray:
    """Dispatch on the weight file's `rule`: "logistic" (default) or "median"."""
    if w.get("rule") == "median":
        return combine_median(maps, w)
    return combine_logistic(maps, w, eps)


def combine_median(maps: dict[str, np.ndarray], w: dict) -> np.ndarray:
    """Per-pixel median of the members. No weights are fitted, which is exactly why it is
    the rule that survived leave-one-scroll-out: the fitted vectors encode the training
    scrolls' model AND face preferences, and both flip between scrolls (FINDINGS 64)."""
    _check(maps, w)
    return np.median(np.stack([maps[f].astype(np.float64) for f in w["features"]], 0), axis=0)


def _check(maps: dict[str, np.ndarray], w: dict) -> None:
    """maps: feature name -> float array in [0, 1]. Every feature of the weight vector must
    be present; a missing member is an error, because dropping one silently rescales the
    remaining weights into a combination nobody fitted."""
    missing = [f for f in w["features"] if f not in maps]
    if missing:
        raise KeyError(f"ensemble: missing member maps {missing}")
    shapes = {maps[f].shape for f in w["features"]}
    if len(shapes) != 1:
        raise ValueError(f"ensemble: member maps disagree on shape: {shapes}")


def combine_logistic(maps: dict[str, np.ndarray], w: dict, eps: float = 1e-4) -> np.ndarray:
    _check(maps, w)
    z = np.full(maps[w["features"][0]].shape, float(w.get("bias", 0.0)), np.float64)
    for f, c in zip(w["features"], w["weights"]):
        p = np.clip(maps[f].astype(np.float64), eps, 1 - eps)
        z += c * np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def predict_from_pngs(seg_dir: str, weights_json: str, out_png: str) -> bool:
    """Read `<seg_dir>/<family>.png` for every member, combine, write `out_png`."""
    import cv2
    w = load_weights(weights_json)
    maps = {}
    for f in w["features"]:
        p = os.path.join(seg_dir, f"{family_of(f)}.png")
        if face_of(f) == "rev":
            p = os.path.join(seg_dir, f"{family_of(f)}_reversed.png")
        a = cv2.imread(p, 0)
        if a is None:
            raise FileNotFoundError(f"ensemble: no member map {p}")
        maps[f] = a.astype(np.float32) / 255.0
    out = combine(maps, w)
    cv2.imwrite(out_png, (out * 255).astype(np.uint8))
    json.dump({"weights_json": os.path.abspath(weights_json), "features": w["features"],
               "fitted_on": w.get("fitted_on", ""), "spec": SPEC,
               "shape": list(out.shape), "mean": float(out.mean())},
              open(os.path.splitext(out_png)[0] + ".json", "w"), indent=1)
    return True


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-dir", required=True, help="directory holding one PNG per member family")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return 0 if predict_from_pngs(a.seg_dir, a.weights, a.out) else 1


if __name__ == "__main__":
    raise SystemExit(_main())
