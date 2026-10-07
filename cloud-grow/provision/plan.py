"""Fleet plan: validation, shard assignment (disjoint z bands / seed streams), projected cost.

Stdlib + cloud_grow.cost only. NOTHING here talks to a provider. All money numbers come from the vast-benchmark model
(cloud_grow/cost.py: EXTRAPOLATED, +-25 %, not validated out of sample) and the plan's own $/h (the user's, unverified).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cloud_grow import cost as C  # noqa: E402

PROVIDERS = ("ssh", "aws", "gcp", "vast")
SECRET_LIKE = re.compile(r"(AKIA[0-9A-Z]{16}|BEGIN [A-Z ]*PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|hub[_.]token|\bsk-[A-Za-z0-9]{20,})", re.I)
PRIVATE_IP = re.compile(r"\b(10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+)\b")


class PlanError(ValueError):
    pass


def _utc(s: str) -> dt.datetime:
    d = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)


def load(path: str) -> dict:
    with open(path) as fh:
        txt = fh.read()
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml   # optional; JSON needs nothing
        except ImportError:
            raise PlanError("YAML plan needs `pip install pyyaml`; or use the JSON plan format (same keys)")
        m = SECRET_LIKE.search(txt)
        if m:
            raise PlanError("plan file contains something secret-like: secrets come from env vars, never the plan")
        return validate(yaml.safe_load(txt))
    m = SECRET_LIKE.search(txt)
    if m:
        raise PlanError(f"plan file contains something secret-like ({m.group(0)[:6]}...): secrets come from env vars, never the plan")
    return validate(json.loads(txt))


def validate(p: dict) -> dict:
    req = ("name", "provider", "scrolls", "box", "spend_cap_usd", "deadline_utc", "region", "upload", "code", "kit")
    miss = [k for k in req if k not in p]
    if miss:
        raise PlanError(f"plan missing {miss}")
    if p["provider"] not in PROVIDERS:
        raise PlanError(f"provider must be one of {PROVIDERS}")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,30}", p["name"]):
        raise PlanError("name must be [a-z0-9-], <= 31 chars (used in instance names/tags)")
    if not (isinstance(p["spend_cap_usd"], (int, float)) and p["spend_cap_usd"] > 0):
        raise PlanError("spend_cap_usd must be > 0 (no cap, no fleet)")
    for k in ("cores", "ram_gb", "disk_gb", "passmark_st", "usd_per_hour"):
        if not (isinstance(p["box"].get(k), (int, float)) and p["box"][k] > 0):
            raise PlanError(f"box.{k} must be a positive number")
    need = C.ram_needed_gb(int(p["box"].get("grows", p["box"]["cores"])))
    if p["box"]["ram_gb"] < need:
        raise PlanError(f"box.ram_gb {p['box']['ram_gb']} < {need} needed for {p['box'].get('grows', p['box']['cores'])} grows (cost.ram_needed_gb)")
    if not p["scrolls"]:
        raise PlanError("no scrolls")
    seen = set()
    for s in p["scrolls"]:
        if s["scroll"] in seen:
            raise PlanError(f"scroll {s['scroll']} listed twice (one shard set per scroll)")
        seen.add(s["scroll"])
        if int(s.get("boxes", 0)) < 1:
            raise PlanError(f"{s['scroll']}: boxes must be >= 1")
        if s.get("z_extent") is not None:
            z0, z1 = s["z_extent"]
            if not (0 <= z0 < z1):
                raise PlanError(f"{s['scroll']}: bad z_extent")
    up = p["upload"]
    if up.get("kind") not in ("s3", "rsync"):
        raise PlanError("upload.kind must be s3 or rsync")
    if not up.get("dest"):
        raise PlanError("upload.dest required")
    for c in ("code", "kit"):
        if not (p[c].get("url") and re.fullmatch(r"[0-9a-f]{64}", p[c].get("sha256", ""))):
            raise PlanError(f"{c}.url and {c}.sha256 (64 hex) required: the box verifies what it downloads")
    if p["provider"] == "ssh" and not p.get("hosts"):
        raise PlanError("provider ssh needs hosts: [user@host, ...] (you supply them)")
    n = n_boxes(p)
    if p["provider"] == "ssh" and len(p["hosts"]) < n:
        raise PlanError(f"plan wants {n} boxes, {len(p['hosts'])} hosts given")
    _utc(p["deadline_utc"])
    m = PRIVATE_IP.search(json.dumps(p))
    if m:
        raise PlanError(f"private IP {m.group(0)} in plan: do not commit private addresses (keep the real plan outside git)")
    return p


def n_boxes(p: dict) -> int:
    return sum(int(s["boxes"]) for s in p["scrolls"])


def shards(p: dict) -> list[dict]:
    """One dict per box. Boxes of one scroll get DISJOINT level-0 z bands when z_extent is given (equal split); otherwise only
    distinct rng seeds (seeds may then land near each other: the seeder only knows its own box's coverage; ANNOUNCED)."""
    out, i = [], 0
    for s in p["scrolls"]:
        k = int(s["boxes"])
        ze = s.get("z_extent")
        for j in range(k):
            b = {"box": f"{p['name']}-{i:02d}", "index": i, "scroll": s["scroll"], "shard": j, "of": k,
                 "rng_seed": 1000 * (1 + p.get("seed_epoch", 0)) + i,
                 "seeds_per_batch": int(s.get("seeds_per_batch", p.get("seeds_per_batch", 16)))}
            if ze:
                z0, z1 = ze
                w = (z1 - z0) / k
                b["zmin"], b["zmax"] = int(z0 + j * w), int(z0 + (j + 1) * w) if j < k - 1 else int(z1)
                b["shard_kind"] = "zband"
            else:
                b["zmin"] = b["zmax"] = None
                b["shard_kind"] = "rng-only (OVERLAP POSSIBLE; give z_extent to make shards disjoint)"
            out.append(b)
            i += 1
    return out


def hours_budget(p: dict, now: dt.datetime | None = None) -> dict:
    """Hours the fleet may run = min(until deadline, spend cap / fleet $/h). The box self-destructs at this horizon."""
    now = now or dt.datetime.now(dt.timezone.utc)
    until_dl = max(0.0, (_utc(p["deadline_utc"]) - now).total_seconds() / 3600.0)
    rate = n_boxes(p) * p["box"]["usd_per_hour"]
    by_cap = p["spend_cap_usd"] / rate
    h = min(until_dl, by_cap)
    return {"hours_to_deadline": round(until_dl, 2), "hours_by_cap": round(by_cap, 2), "hours": round(h, 2),
            "binding": "deadline" if until_dl <= by_cap else "spend cap", "fleet_usd_per_hour": round(rate, 3)}


def projection(p: dict, now: dt.datetime | None = None) -> dict:
    hb = hours_budget(p, now)
    b = p["box"]
    est = C.estimate(b["passmark_st"], int(b["cores"]), usd_per_hour=b["usd_per_hour"])
    n = n_boxes(p)
    pe = est["prod_equiv_verified_cm2_h"]
    h = hb["hours"]
    per_scroll = []
    for s in p["scrolls"]:
        k = int(s["boxes"])
        per_scroll.append({"scroll": s["scroll"], "boxes": k, "verified_cm2_box_claims": {q: round(pe[q] * k * h, 0) for q in ("low", "mid", "high")}})
    data_gb = {s["scroll"]: None for s in p["scrolls"]}
    return {"label": est["label"] + "; compute only; data/egress excluded; $/h is the plan's own number (unverified)",
            "boxes": n, **hb, "max_compute_usd": round(hb["fleet_usd_per_hour"] * h, 2), "spend_cap_usd": p["spend_cap_usd"],
            "fleet_cm2_per_hour_prod_equiv": {q: round(pe[q] * n, 1) for q in ("low", "mid", "high")},
            "verified_cm2_by_horizon": {q: round(pe[q] * n * h, 0) for q in ("low", "mid", "high")},
            "per_scroll": per_scroll, "ram_gb_needed_per_box": est["ram_gb_needed"],
            "usd_per_100cm2": (est.get("usd_per_100_cm2") or {}), "scroll_data_gb_note": data_gb and "see `python -m cloud_grow fetch` preview per scroll (prediction_bytes in config/scrolls.json)",
            "caveat": "verified_cm2 = the BOX'S OWN claim; nothing is validated against human annotation (D6); self-crossing rate unmeasured on rented output"}


def fmt_projection(pr: dict) -> str:
    l, m, hgh = (pr["verified_cm2_by_horizon"][k] for k in ("low", "mid", "high"))
    return (f"PROJECTED: {pr['boxes']} boxes x ${pr['fleet_usd_per_hour'] / pr['boxes']:.3f}/h for {pr['hours']} h "
            f"(binding: {pr['binding']}) = max ${pr['max_compute_usd']} compute vs cap ${pr['spend_cap_usd']}; "
            f"verified cm2 (box claims, EXTRAPOLATED +-25 %, model range) low/mid/high = {l:.0f}/{m:.0f}/{hgh:.0f}; "
            f"RAM >= {pr['ram_gb_needed_per_box']} GB/box")
