"""PULL-BASED import of grows done on an UNTRUSTED rented box (cloud_grow_2026-10-08). The box gets NO hub token, NO ssh keys, NO DB; it only writes a staging tree that a human /
another tool rsyncs to the hub. This module is the hub-side gate (D2/D7: library method + dry-run CLI; nothing is applied without --apply; existing artifacts and meta.json are never touched):

  staging/<seg>/export.json        schema vpipe-remote-grow-export/1 (docs/experiments/vast_benchmark_2026-10-06/manifest_spec.json) with identity.seg/scroll, tool.md5, run.host
  staging/<seg>/md5.txt            `md5  relpath` for EVERY other file (find . -type f ! -name md5.txt | sort | xargs md5sum)
  staging/<seg>/r*/<ckpt>/{x,y,z,generations}.tif + meta.json (+ guard sidecars)

verify(seg_dir)   integrity (manifest <-> tree, no symlinks, no path escapes, size caps, file set) + the FLEET detector on every newest checkpoint: transverse census (ZERO tolerated),
                  resume_gate.check (fold-over / normal-reversal / close pairs / hairpin fractions, fail closed), contact types (selfcontact.contact_pairs counts, reported).
                  The box's own claims (kit md5s, params) are PROVENANCE ONLY: the hub cannot verify the box's binary, so it re-verifies the OUTPUT.
run(...)          scan -> verify -> PASS: copy the verified checkpoint tree to <import_root>/<box>/<seg>/ (never in place), provenance sidecar, new segment row + `tifxyz` artifact
                  (meta cloud_import) -- a seg name that already exists in the DB is refused (never modify existing); FAIL: copy to <quarantine_root>/<seg>/ with a verdict.json and NO DB artifact,
                  NO segment row (a quarantined segment must never become a grow/finish candidate). `counters()` = quarantine/import counts for the Blockers surface.
CLI: python -m vesuvius_pipeline.cloud_import scan|verify|import --staging DIR [--seg S] [--apply]
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

SCHEMA = "vpipe-remote-grow-export/1"
# claims a box may make about its tracer lib (first 8 hex of libvc_tracer.so md5): fleet patched lib; min_separation kit. PROVENANCE ONLY -- see module docstring.
ALLOWED_TRACER_LIB_PREFIXES = ("94be1eae", "8b51204d")
MAX_FILE_BYTES = 512 * 1024 * 1024
MAX_FILES = 5000
TIFXYZ_FILES = ("x.tif", "y.tif", "z.tif", "meta.json")


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def roots(fleet=None, import_root=None, quarantine_root=None) -> tuple[Path, Path]:
    base = Path(getattr(fleet, "root", ".")) / "var"
    return Path(import_root or base / "cloud_import"), Path(quarantine_root or base / "cloud_import_quarantine")


def scan(staging: str) -> list[str]:
    p = Path(staging)
    return sorted(d.name for d in p.iterdir() if d.is_dir() and not d.is_symlink()) if p.is_dir() else []


def _integrity(sd: Path) -> tuple[dict, list[str]]:
    probs: list[str] = []
    ex = {}
    exp = sd / "export.json"
    if not exp.is_file() or exp.is_symlink():
        return ex, ["export.json missing"]
    try:
        ex = json.loads(exp.read_text())
    except ValueError:
        return {}, ["export.json is not JSON"]
    if ex.get("schema") != SCHEMA:
        probs.append(f"export.json schema {ex.get('schema')!r} != {SCHEMA}")
    ident = ex.get("identity") or {}
    if ident.get("seg") != sd.name:
        probs.append(f"identity.seg {ident.get('seg')!r} != directory {sd.name!r}")
    if not ident.get("scroll"):
        probs.append("identity.scroll missing")
    if not (ex.get("tool") or {}).get("md5") or not (ex.get("run") or {}).get("host"):
        probs.append("tool.md5 / run.host missing")
    mf = sd / "md5.txt"
    if not mf.is_file() or mf.is_symlink():
        return ex, probs + ["md5.txt missing"]
    listed = {}
    for ln in mf.read_text().splitlines():
        if not ln.strip():
            continue
        try:
            m, rel = ln.split(None, 1)
        except ValueError:
            probs.append(f"bad md5.txt line {ln[:40]!r}")
            continue
        rel = rel.strip().lstrip("*")
        rel = rel[2:] if rel.startswith("./") else rel
        if rel.startswith("/") or ".." in Path(rel).parts:
            probs.append(f"path escape in manifest: {rel[:60]}")
            continue
        listed[rel] = m
    files = []
    for dp, dn, fn in os.walk(sd, followlinks=False):
        for n in fn + dn:
            if (Path(dp) / n).is_symlink():
                probs.append(f"symlink: {(Path(dp) / n).relative_to(sd)}")
        for n in fn:
            fp = Path(dp) / n
            if fp.is_symlink():
                continue
            rel = str(fp.relative_to(sd))
            if rel != "md5.txt":
                files.append((rel, fp))
    if len(files) > MAX_FILES:
        probs.append(f"{len(files)} files > cap {MAX_FILES}")
    have = {r for r, _ in files}
    for r in sorted(have - set(listed)):
        probs.append(f"file not in manifest: {r}")
    for r in sorted(set(listed) - have):
        probs.append(f"manifest file missing: {r}")
    for rel, fp in files:
        if rel in listed:
            if fp.stat().st_size > MAX_FILE_BYTES:
                probs.append(f"too large: {rel}")
            elif _md5(fp) != listed[rel]:
                probs.append(f"md5 mismatch: {rel}")
    return ex, probs


def _checkpoints(sd: Path) -> list[Path]:
    out = [p.parent for p in sd.glob("r*/*/x.tif") if all((p.parent / f).is_file() for f in TIFXYZ_FILES)]
    return sorted(out, key=lambda d: (d.parent.name, (d / "meta.json").stat().st_mtime, d.name))


def verify(sd, pol, override: dict | None = None) -> dict:
    """Verdict for one staged segment dir. `pol` = GuardPolicy with the FLEET selfcross binary resolved. Never raises; any inability to check = FAIL (fail closed)."""
    sd = Path(sd)
    v = {"seg": sd.name, "pass": False, "problems": [], "checks": {}, "kit_claims": {}}
    try:
        ex, probs = _integrity(sd)
        v["problems"] += probs
        claim = ((ex.get("tool") or {}).get("kit_libs_md5") or {}) if isinstance((ex.get("tool") or {}).get("kit_libs_md5"), dict) else {}
        lib = str(claim.get("libvc_tracer.so") or (ex.get("tool") or {}).get("lib_md5") or "")
        v["kit_claims"] = {"tool_md5": (ex.get("tool") or {}).get("md5"), "libvc_tracer_md5": lib or None, "release_sha": (ex.get("tool") or {}).get("release_sha")}
        if not lib[:8] in ALLOWED_TRACER_LIB_PREFIXES:
            v["problems"].append(f"claimed libvc_tracer md5 {lib[:8] or 'absent'!r} not in the allowed patched kits {ALLOWED_TRACER_LIB_PREFIXES}")
        cks = _checkpoints(sd)
        if not cks:
            v["problems"].append("no tifxyz checkpoint (x/y/z.tif + meta.json) under r*/")
        if not v["problems"]:
            from . import resume_gate as RG, selfcontact as SC
            from . import growth_guard as GG
            newest = cks[-1]
            res = RG.check(str(newest), pol, override)
            v["checks"]["resume_gate"] = res
            if not res.get("ran", True):
                v["problems"].append(f"gate could not run: {res.get('reasons')}")
            elif not res["pass"]:
                v["problems"].append(f"degeneracy gate: {res['reasons']}")
            X, Y, Z = GG._read_xyz(str(newest))
            pr = SC.contact_pairs(X, Y, Z)
            v["checks"]["contact_pairs"] = {"coincident": int((pr["type"] == 0).sum()), "grazing": int((pr["type"] == 1).sum()), "coplanar": int((pr["type"] == 2).sum())}
            v["checks"]["newest_checkpoint"] = str(newest.relative_to(sd))
            v["checks"]["cells"] = int(((X > 0) & (Y > 0) & (Z > 0)).sum())
        v["pass"] = not v["problems"]
    except Exception as e:                                  # noqa: BLE001 - fail closed
        v["problems"].append(f"verify error {type(e).__name__}: {str(e)[:160]}")
        v["pass"] = False
    return v


def run(staging: str, fleet, db, pol, box: str, apply: bool = False, only: str | None = None, import_root=None, quarantine_root=None, log=print) -> dict:
    from .db import pipeline_db
    imp, qua = roots(fleet, import_root, quarantine_root)
    st = {"scanned": 0, "passed": 0, "quarantined": 0, "imported": 0, "refused_existing": 0, "apply": apply}
    rows = []
    for name in scan(staging):
        if only and name != only:
            continue
        st["scanned"] += 1
        sd = Path(staging) / name
        v = verify(sd, pol)
        exists = db.execute("SELECT 1 FROM segment WHERE seg=?", (name,)).fetchone() is not None
        if v["pass"] and exists:
            v["pass"] = False
            v["problems"].append("segment already exists in the DB: an import never modifies existing segments/artifacts")
            st["refused_existing"] += 1
        rows.append(v)
        if v["pass"]:
            st["passed"] += 1
            if apply:
                dst = imp / box / name
                if dst.exists():
                    v["problems"].append(f"import dir exists: {dst}")
                    v["pass"] = False
                    st["passed"] -= 1
                else:
                    shutil.copytree(sd, dst, symlinks=False)
                    newest = dst / v["checks"]["newest_checkpoint"]
                    prov = {"kind": "cloud_import", "box": box, "imported_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "export_json_md5": _md5(sd / "export.json"),
                            "kit_claims_UNVERIFIED": v["kit_claims"], "hub_verification": {k: v["checks"].get(k) for k in ("resume_gate", "contact_pairs", "cells")},
                            "detector": os.path.basename(str(pol.selfcross_bin or "")), "note": "output re-verified on the hub with the fleet detector; the box's binary/params are claims only"}
                    (newest / "cloud_import.provenance.json").write_text(json.dumps(prov, indent=1))
                    sc = json.loads((sd / "export.json").read_text())["identity"]["scroll"]
                    P = pipeline_db()
                    P.upsert_segment(db, name, scroll=sc, route="A")
                    P.record_artifact(db, name, "grow", "tifxyz", str(newest), None, meta={"cloud_import": True, "box": box})
                    P.record_metric(db, name, "cloud_import", 1.0, text=json.dumps({"box": box, "gate": "pass"}), stage="grow")
                    db.commit()
                    st["imported"] += 1
        if not v["pass"]:
            st["quarantined"] += 1
            if apply:
                q = qua / box / name
                q.parent.mkdir(parents=True, exist_ok=True)
                if not q.exists():
                    shutil.copytree(sd, q, symlinks=False)
                (q / "verdict.json").write_text(json.dumps(v, indent=1))
        log(f"cloud_import {name}: {'PASS' if v['pass'] else 'QUARANTINE'} {v['problems'][:3]}")
    st["verdicts"] = rows
    return st


def counters(fleet=None, import_root=None, quarantine_root=None) -> dict:
    imp, qua = roots(fleet, import_root, quarantine_root)
    n = lambda p: sum(1 for b in p.iterdir() for s in b.iterdir()) if p.is_dir() else 0
    return {"imported": n(imp), "quarantined": n(qua)}


if __name__ == "__main__":
    import argparse
    from dataclasses import replace
    from . import config, growth_guard as GG
    from .remotedb import connect_for, hub_token
    from .stages import grow as G
    ap = argparse.ArgumentParser(prog="python -m vesuvius_pipeline.cloud_import")
    ap.add_argument("cmd", choices=["scan", "verify", "import"]); ap.add_argument("--staging", required=True); ap.add_argument("--seg"); ap.add_argument("--box", default="cloud")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    if a.cmd == "scan":
        print(json.dumps(scan(a.staging)))
        raise SystemExit(0)
    fl = config.load()
    db = connect_for(fl, hub_token(fl))
    pol = G.resolve_selfcross_bin(fl, replace(GG.policy_from_db(db), selfcross=True, selfcross_threads=2))
    out = run(a.staging, fl, db, pol, a.box, apply=(a.apply and a.cmd == "import"), only=a.seg)
    out.pop("verdicts", None) if a.cmd == "import" else None
    print(json.dumps(out, indent=1, default=str))
