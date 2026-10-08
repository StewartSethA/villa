import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pull_box8 as P  # noqa: E402

SHIM = """#!/bin/bash
# fake ssh: skip options, drop the host, run the rest locally
while [ $# -gt 0 ]; do case "$1" in -o|-p|-i) shift 2;; -*) shift;; *) break;; esac; done
shift
exec bash -c "$*"
"""


def mk_payload(home, scroll="PHerc0211"):
    pd = home / "out" / scroll / "q4500"
    (pd / "tiled").mkdir(parents=True)
    files = []
    for i in range(3):
        p = pd / "tiled" / f"f{i}.bin"
        p.write_bytes(os.urandom(5000 + i))
        files.append({"path": str(p.relative_to(pd)), "size": p.stat().st_size, "md5": hashlib.md5(p.read_bytes()).hexdigest()})
    (pd / "PAYLOAD.json").write_text(json.dumps({"scroll": scroll, "status": "complete", "files": files, "total_bytes": sum(f["size"] for f in files)}))
    (pd / "DONE").write_text("complete\n")
    (home / "out" / "ALLDONE.json").write_text("{}")
    return pd


def run_pull(tmp_path, monkeypatch, home, dest):
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    (bindir / "ssh").write_text(SHIM)
    (bindir / "ssh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    return P.main(["--host", "fake", "--remote-home", str(home), "--dest", str(dest), "--once"])


def test_pull_verifies_and_marks(tmp_path, monkeypatch):
    home, dest = tmp_path / "box", tmp_path / "local"
    pd = mk_payload(home)
    assert run_pull(tmp_path, monkeypatch, home, dest) == 0
    assert (dest / "PHerc0211" / "q4500" / "PULLED.json").exists()
    assert json.loads((pd / "PULLED.json").read_text())["bytes"] == 15003    # box-side marker the governor reads
    row = json.loads((dest / "pull_ledger.jsonl").read_text().splitlines()[-1])
    assert row["md5_failures"] == 0 and row["bytes_local"] == 15003


def test_corrupt_payload_is_never_marked_pulled(tmp_path, monkeypatch):
    home, dest = tmp_path / "box", tmp_path / "local"
    pd = mk_payload(home)
    with open(pd / "tiled" / "f1.bin", "r+b") as f:       # corrupt in place AFTER the manifest was written (same size)
        f.write(b"XXXX")
    assert run_pull(tmp_path, monkeypatch, home, dest) == 1
    assert not (dest / "PHerc0211" / "q4500" / "PULLED.json").exists()
    assert not (pd / "PULLED.json").exists()
