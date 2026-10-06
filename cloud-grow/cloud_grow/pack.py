"""Export packer: tarball of the grown tifxyz dirs + sidecars + checksums (~0.5 MB per segment).

Ships: every r*/auto_grown_* and guarded_g_* / selfx_held_* checkpoint dir (x/y/z/generations.tif, meta.json,
guard.json), guard_reasons/*.npz, params_r*.json, round*.log, rounds.jsonl, alerts, <seg>.export.json, <seg>.md5tree.
Never ships: cache_root/, any credential, the CT/prediction/grids themselves. Refuses (pre-flight) a seg_dir that
contains a file whose name or content looks like a token/key.
"""
from __future__ import annotations

import json
import os
import re
import tarfile

from . import manifest as M

SECRET_NAME = re.compile(r"(hub\.token|id_rsa|id_ed25519|\.pem$|\.key$|\.env$|credentials|\.netrc)", re.I)
SECRET_CONTENT = re.compile(rb"(BEGIN [A-Z ]*PRIVATE KEY|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}|hub_token\s*[=:])")
TEXT_SUFFIX = (".json", ".log", ".jsonl", ".txt", ".md5tree", ".toml", ".sh", ".py")


class PackError(RuntimeError):
    pass


def scan_for_secrets(seg_dir: str, seg: str) -> list[str]:
    bad = []
    for rel in M.shippable_files(seg_dir, seg):
        if SECRET_NAME.search(os.path.basename(rel)):
            bad.append(f"{rel}: secret-like file name")
            continue
        if rel.endswith(TEXT_SUFFIX) and os.path.getsize(os.path.join(seg_dir, rel)) < 50 * 1024 * 1024:
            with open(os.path.join(seg_dir, rel), "rb") as fh:
                if SECRET_CONTENT.search(fh.read()):
                    bad.append(f"{rel}: secret-like content")
    return bad


def pack_segment(seg_dir: str, seg: str, run_context: dict, out_tar: str, seed: dict | None = None) -> dict:
    bad = scan_for_secrets(seg_dir, seg)
    if bad:
        raise PackError("refusing to pack, secret-like material: " + "; ".join(bad))
    man = M.build_manifest(seg_dir, seg, run_context, seed=seed)
    files = M.shippable_files(seg_dir, seg) + [M.MANIFEST_NAME.format(seg=seg), M.TREE_NAME.format(seg=seg)]
    tmp = out_tar + ".part"
    with tarfile.open(tmp, "w:gz") as tf:
        for rel in sorted(set(files)):
            tf.add(os.path.join(seg_dir, rel), arcname=f"{seg}/{rel}", recursive=False)
    os.replace(tmp, out_tar)
    sha = M.sha256_file(out_tar)
    with open(out_tar + ".sha256", "w") as fh:
        fh.write(f"{sha}  {os.path.basename(out_tar)}\n")
    nbytes = os.path.getsize(out_tar)
    return {"tar": out_tar, "sha256": sha, "bytes": nbytes, "mb": round(nbytes / 1e6, 3), "n_files": len(set(files)),
            "tree_sha256": man["ship"]["md5tree_sha256"]}


def pack_all(out_root: str, scroll: str, run_context: dict, out_dir: str, db=None) -> list[dict]:
    """One tarball per segment dir under out_root/<seg>/ that has a rounds.jsonl."""
    os.makedirs(out_dir, exist_ok=True)
    res = []
    for seg in sorted(os.listdir(out_root)):
        sd = os.path.join(out_root, seg)
        if not os.path.isfile(os.path.join(sd, "rounds.jsonl")):
            continue
        seed = None
        if db is not None:
            r = db.execute("SELECT x,y,z,score,dist_l4,centre,window_mean,source,provenance_json FROM seed WHERE seg=?", (seg,)).fetchone()
            if r:
                seed = {"xyz": [r["x"], r["y"], r["z"]], "score": r["score"], "dist_l4": r["dist_l4"], "centre": r["centre"],
                        "window_mean": r["window_mean"], "source": r["source"], "provenance": json.loads(r["provenance_json"] or "{}")}
        res.append(pack_segment(sd, seg, run_context, os.path.join(out_dir, f"{seg}.tar.gz"), seed=seed))
    return res
