"""Hub-side importer + verifier for a remote grow export (the SOTA report's blocker G1).

TRUST MODEL: the rented box is untrusted. Nothing it claims is accepted; every claim is re-measured here:
  1. tarball sha256 (from `.sha256` sidecar or --expect-sha256), safe extraction (no absolute paths, `..`, links);
  2. manifest schema + md5 tree: every listed file present with the listed md5, NO unlisted file, tree sha256 matches;
  3. refuse markers: `selfx_unverified.json` in the final checkpoint, UNPINNED tools (unless --accept-unpinned);
  4. RE-MEASURE lattice area (guard lattice_area_cm2, voxel pitch from --voxel-um or the manifest) for every shipped
     checkpoint and compare with the claimed area: relative difference above --area-tol refuses;
  5. D3 chain: for each round with a shipped `resume_from`, cells(final of that round) >= cells(resume checkpoint);
  6. RE-RUN the self-crossing census with the HUB's own vc_tifxyz_selfcross on the FINAL checkpoint (fail closed:
     census could not run = refuse; any transverse contact above --max-selfx-density = refuse);
  7. optionally re-score material at CT level 1 with the hub's CT (--ct-zarr): box claim vs hub value is recorded.
REGISTRATION is append-only SIDECARS (D10): `<registry>/imports.jsonl` (one line per verdict) and
`<registry>/<seg>/<tree_sha256_12>.import.json`; optionally an INSERT-only sqlite table `remote_grow_import`.
Nothing in the extracted export, the hub DB's existing rows, or any meta/result/manifest is modified. --dry-run
does every check and writes nothing. Re-importing the same (seg, tree sha256) is a no-op (`already_imported`).
UNVALIDATED against human annotation (D6): a PASS means "the box's claims reproduce on the hub", not "the sheet is right".
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tarfile
import tempfile
import time
from dataclasses import dataclass, field

import numpy as np

from . import growth_guard as GG
from . import manifest as M

SCHEMA = "vpipe-remote-grow-export/1"


@dataclass
class ImportOptions:
    voxel_um: float | None = None
    area_tol: float = 0.01
    max_selfx_density: float = 0.0
    accept_unpinned: bool = False
    selfcross_bin: str = ""
    selfcross_env: dict | None = None
    ct_zarr: str = ""
    skip_selfx: bool = False            # only for tests / hubs without the tool; recorded, and the verdict can never be PASS
    dry_run: bool = False
    expect_sha256: str = ""


@dataclass
class Verdict:
    seg: str
    status: str = "REFUSED"             # PASS | REFUSED
    reasons: list = field(default_factory=list)
    checks: dict = field(default_factory=dict)
    tree_sha256: str = ""
    final_checkpoint: str = ""
    registered: bool = False

    def refuse(self, why: str):
        self.reasons.append(why)


def safe_extract(tar_path: str, dest: str) -> str:
    """Extract, refusing absolute paths, `..`, links and devices. Returns the single top-level dir."""
    with tarfile.open(tar_path, "r:*") as tf:
        top = set()
        for m in tf.getmembers():
            n = os.path.normpath(m.name)
            if os.path.isabs(m.name) or n.startswith("..") or ".." in n.split(os.sep):
                raise ValueError(f"unsafe path in tar: {m.name!r}")
            if not (m.isfile() or m.isdir()):
                raise ValueError(f"unsafe member type in tar (link/device): {m.name!r}")
            top.add(n.split(os.sep)[0])
        if len(top) != 1:
            raise ValueError(f"tar must contain exactly one top-level directory, found {sorted(top)}")
        tf.extractall(dest, filter="data") if "filter" in tarfile.TarFile.extractall.__code__.co_varnames else tf.extractall(dest)
    return os.path.join(dest, next(iter(top)))


def _cells(ckpt: str) -> int:
    X, Y, Z = GG._read_xyz(ckpt)
    return int(((X > 0) & (Y > 0) & (Z > 0)).sum())


def _lattice_area(ckpt: str, voxel_um: float) -> float:
    P, V = GG.lattice_frame(*GG._read_xyz(ckpt))
    return GG.lattice_area_cm2(P, V, voxel_um)


def verify_dir(seg_dir: str, opt: ImportOptions) -> Verdict:
    seg = os.path.basename(seg_dir.rstrip("/"))
    v = Verdict(seg=seg)
    mp = os.path.join(seg_dir, M.MANIFEST_NAME.format(seg=seg))
    if not os.path.isfile(mp):
        v.refuse("no manifest " + M.MANIFEST_NAME.format(seg=seg))
        return v
    try:
        man = json.load(open(mp))
    except ValueError as e:
        v.refuse(f"manifest unreadable: {e}")
        return v
    if man.get("schema") != SCHEMA:
        v.refuse(f"manifest schema {man.get('schema')!r} != {SCHEMA!r}")
        return v
    # ---- 2. md5 tree
    tp = os.path.join(seg_dir, M.TREE_NAME.format(seg=seg))
    if not os.path.isfile(tp):
        v.refuse("no md5 tree file")
        return v
    text = open(tp).read()
    import hashlib
    v.tree_sha256 = hashlib.sha256(text.encode()).hexdigest()
    if v.tree_sha256 != (man.get("ship") or {}).get("md5tree_sha256"):
        v.refuse("md5 tree sha256 does not match the manifest's claim")
    listed = {}
    for ln in text.splitlines():
        h, _, rel = ln.partition("  ")
        listed[rel] = h
    actual = set(M.shippable_files(seg_dir, seg))
    bad = [r for r, h in listed.items() if not os.path.isfile(os.path.join(seg_dir, r)) or M.md5_file(os.path.join(seg_dir, r)) != h]
    extra = sorted(actual - set(listed))
    v.checks["md5_tree"] = {"n_files": len(listed), "n_bad": len(bad), "n_unlisted": len(extra)}
    if bad:
        v.refuse(f"md5 mismatch / missing in {len(bad)} file(s), first: {bad[0]}")
    if extra:
        v.refuse(f"{len(extra)} file(s) not in the md5 tree, first: {extra[0]}")
    if v.reasons:
        return v
    # ---- 3. refuse markers
    tool = man.get("tool") or {}
    v.checks["tool_pin_status"] = tool.get("pin_status")
    if tool.get("pin_status") != "PINNED" and not opt.accept_unpinned:
        v.refuse(f"tool pin status {tool.get('pin_status')!r} (a different build gives different geometry); --accept-unpinned to override")
    rounds = man.get("rounds") or []
    final = (man.get("final") or {}).get("checkpoint")
    if not final:
        v.refuse("manifest names no final checkpoint")
        return v

    def local(p):
        """Map the box's absolute checkpoint path onto the shipped tree: the part from `r<N>/` on."""
        parts = str(p).replace("\\", "/").split("/")
        for i, s in enumerate(parts):
            if s.startswith("r") and s[1:].isdigit() and i + 1 < len(parts):
                return os.path.join(seg_dir, *parts[i:])
        return os.path.join(seg_dir, os.path.basename(str(p)))
    fdir = local(final)
    v.final_checkpoint = os.path.relpath(fdir, seg_dir)
    if not os.path.isdir(fdir):
        v.refuse(f"final checkpoint {v.final_checkpoint} not shipped")
        return v
    if os.path.exists(os.path.join(fdir, "selfx_unverified.json")):
        v.refuse("final checkpoint carries selfx_unverified.json: its newest cells were never self-crossing checked on the box")
    vox = opt.voxel_um or ((man.get("inputs") or {}).get("voxel_um"))
    if not vox:
        v.refuse("no voxel pitch (pass --voxel-um from the registry)")
        return v
    if opt.voxel_um and (man.get("inputs") or {}).get("voxel_um") and abs(opt.voxel_um - man["inputs"]["voxel_um"]) > 1e-6:
        v.refuse(f"voxel pitch conflict: hub {opt.voxel_um} vs box {man['inputs']['voxel_um']} (upstream wins; refusing)")
        return v
    # ---- 4. area re-measure, every shipped round checkpoint
    areas = []
    for r in rounds:
        for key, claim_key in (("checkpoint", "area_cm2_postguard"),):
            ck, claim = r.get(key), r.get(claim_key)
            if not ck or claim is None:
                continue
            d = local(ck)
            if not os.path.isdir(d):
                v.refuse(f"round {r.get('round')}: checkpoint {os.path.relpath(d, seg_dir)} not shipped")
                continue
            try:
                got = _lattice_area(d, vox)
            except Exception as e:                   # noqa: BLE001
                v.refuse(f"round {r.get('round')}: lattice unreadable ({type(e).__name__}: {e})")
                continue
            rel = abs(got - claim) / max(abs(claim), 1e-9)
            guard_written = os.path.basename(d).startswith(("guarded_", "selfx_held_"))
            # guard-written meta area == lattice_area_cm2 exactly (measured: 30 of 30 guarded checkpoints, rel diff 0);
            # the RAW tracer's area_cm2 uses another formula (measured n=80 raw checkpoints: rel diff p10/p50/p90 = -0.0004/+0.0066/+0.027,
            # min -0.56, max +0.065), so for raw checkpoints the claim is recorded, not enforced; the registered area is the hub measure.
            areas.append({"round": r.get("round"), "checkpoint": os.path.basename(d), "claimed_cm2": claim, "remeasured_cm2": round(got, 6),
                          "rel_diff": round(rel, 5), "enforced": guard_written})
            if guard_written and rel > opt.area_tol:
                v.refuse(f"round {r.get('round')}: area claimed {claim:.4f} cm2, hub lattice measure {got:.4f} cm2 (rel {rel:.3%} > {opt.area_tol:.1%})")
    v.checks["area"] = areas
    # ---- 5. D3 chain
    chain = []
    for r in rounds:
        rs, ck = r.get("resume_from"), r.get("checkpoint")
        if not rs or not ck:
            continue
        a, b = local(rs), local(ck)
        if os.path.isdir(a) and os.path.isdir(b):
            n0, n1 = _cells(a), _cells(b)
            chain.append({"round": r.get("round"), "cells_resume": n0, "cells_final": n1})
            if n1 < n0:
                v.refuse(f"D3 violated in round {r.get('round')}: {n1} cells < resume surface {n0} cells (resumes only expand)")
    v.checks["d3_chain"] = chain
    # ---- 6. selfx re-run on the HUB
    if opt.skip_selfx:
        v.checks["selfx"] = {"ran": False, "skipped": "operator (--skip-selfx): verdict can never be PASS"}
        v.refuse("selfx re-check skipped by operator: not verified")
    else:
        pol = GG.GuardPolicy(selfcross=True, selfcross_bin=opt.selfcross_bin, selfcross_env=opt.selfcross_env,
                             selfcross_max_triangles=0)
        X, Y, Z = GG._read_xyz(fdir)
        mask, info = GG.selfcross_check(fdir, X.shape, pol)
        v.checks["selfx"] = {k: info.get(k) for k in ("ran", "density", "triangles", "error", "skipped", "wall_s")}
        if not info.get("ran"):
            v.refuse(f"selfx census could not run on the hub ({info.get('error') or info.get('skipped')}): fail closed")
        elif (info.get("density") or 0.0) > opt.max_selfx_density:
            v.refuse(f"selfx density {info['density']} > {opt.max_selfx_density} on the final checkpoint")
    # ---- 7. optional material re-score
    if opt.ct_zarr and os.path.isdir(opt.ct_zarr):
        try:
            from . import scoring as SC
            sc = SC.score_checkpoint(fdir, GG.ZarrSampler(opt.ct_zarr, 1))
            claim = r_last = rounds[-1].get("material_frac") if rounds else None
            v.checks["material"] = {"hub_material_frac": sc.material_frac, "box_claim": claim, "hub_verified_cm2": sc.verified_cm2,
                                    "hub_verdict": sc.verdict}
        except Exception as e:                       # noqa: BLE001
            v.checks["material"] = {"error": f"{type(e).__name__}: {e}"}
    if not v.reasons:
        v.status = "PASS"
    return v


def _already(registry: str, seg: str, tree: str) -> bool:
    p = os.path.join(registry, "imports.jsonl")
    if not os.path.exists(p):
        return False
    with open(p) as fh:
        for ln in fh:
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if r.get("seg") == seg and r.get("tree_sha256") == tree and r.get("status") == "PASS":
                return True
    return False


def register(v: Verdict, registry: str, sqlite_path: str = "", tar_sha256: str = "") -> bool:
    """Append-only: one JSONL line + one new sidecar file (+ one INSERT). Never updates or deletes."""
    os.makedirs(registry, exist_ok=True)
    rec = {"schema": "vpipe-remote-grow-import/1", "t_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "seg": v.seg, "status": v.status, "reasons": v.reasons, "checks": v.checks, "tree_sha256": v.tree_sha256,
           "final_checkpoint": v.final_checkpoint, "tar_sha256": tar_sha256,
           "validation_status": "UNVALIDATED against human annotation (D6): PASS = box claims reproduce on the hub"}
    d = os.path.join(registry, v.seg)
    os.makedirs(d, exist_ok=True)
    side = os.path.join(d, f"{v.tree_sha256[:12] or 'none'}.{int(time.time())}.import.json")
    with open(side, "x") as fh:
        json.dump(rec, fh, indent=1, default=str)
    with open(os.path.join(registry, "imports.jsonl"), "a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    if sqlite_path:
        db = sqlite3.connect(sqlite_path)
        db.execute("CREATE TABLE IF NOT EXISTS remote_grow_import (id INTEGER PRIMARY KEY AUTOINCREMENT, seg TEXT NOT NULL, "
                   "status TEXT NOT NULL, tree_sha256 TEXT, final_checkpoint TEXT, tar_sha256 TEXT, record_json TEXT, recorded_utc TEXT)")
        db.execute("INSERT INTO remote_grow_import(seg,status,tree_sha256,final_checkpoint,tar_sha256,record_json,recorded_utc) VALUES(?,?,?,?,?,?,?)",
                   (v.seg, v.status, v.tree_sha256, v.final_checkpoint, tar_sha256, json.dumps(rec, default=str), rec["t_utc"]))
        db.commit()
        db.close()
    return True


def import_export(src: str, registry: str, opt: ImportOptions, sqlite_path: str = "") -> Verdict:
    """`src` = a .tar.gz export or an already-extracted segment directory."""
    tmp = None
    tar_sha = ""
    try:
        if os.path.isdir(src):
            seg_dir = src
        else:
            tar_sha = M.sha256_file(src)
            want = opt.expect_sha256
            side = src + ".sha256"
            if not want and os.path.exists(side):
                want = open(side).read().split()[0]
            if want and want != tar_sha:
                v = Verdict(seg=os.path.basename(src))
                v.refuse(f"tarball sha256 {tar_sha} != expected {want}")
                if not opt.dry_run:
                    register(v, registry, sqlite_path, tar_sha)
                return v
            tmp = tempfile.mkdtemp(prefix="import_remote_grow_")
            try:
                seg_dir = safe_extract(src, tmp)
            except (ValueError, tarfile.TarError) as e:
                v = Verdict(seg=os.path.basename(src))
                v.refuse(f"unsafe or unreadable tarball: {e}")
                if not opt.dry_run:
                    register(v, registry, sqlite_path, tar_sha)
                return v
        v = verify_dir(seg_dir, opt)
        if v.status == "PASS" and _already(registry, v.seg, v.tree_sha256):
            v.checks["already_imported"] = True
            return v
        if not opt.dry_run:
            v.registered = register(v, registry, sqlite_path, tar_sha)
        return v
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
