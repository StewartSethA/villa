"""VC3D tool resolution, identification and pin gate.

Rule (D24): never trust a label, identify the artifact. Every tool the run will execute is `md5sum`-ed and
`file -L`-ed, compared with config/tool_pins.json, and the result is written into the manifest. A mismatch
refuses to start unless the operator passes allow_unpinned, which is ANNOUNCED (stderr + alerts) and recorded
(`pin_status: "UNPINNED-ALLOWED"`), because a different build gives different geometry.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

PINS_PATH = Path(__file__).resolve().parent.parent / "config" / "tool_pins.json"
REQUIRED_TOOLS = ("vc_grow_seg_from_seed", "vc_tifxyz_selfcross")


class ToolError(RuntimeError):
    pass


def md5_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def file_type(path: str | os.PathLike) -> str:
    """`file -L` output (identifies shims / shell scripts masquerading as binaries)."""
    try:
        r = subprocess.run(["file", "-L", "-b", str(path)], capture_output=True, text=True, timeout=30)
        return (r.stdout or r.stderr).strip()
    except (OSError, subprocess.SubprocessError) as e:
        return f"file(1) unavailable: {type(e).__name__}"


def load_pins(path: str | os.PathLike | None = None) -> dict:
    with open(path or PINS_PATH) as fh:
        return json.load(fh)["tools"]


def kit_bin_dir(explicit: str | None = None) -> str:
    """The directory holding the tools: --kit, else $VC_BIN, else the PATH location of the tracer."""
    cand = explicit or os.environ.get("VC_BIN")
    if cand:
        return str(cand)
    for d in os.environ.get("PATH", "").split(os.pathsep):
        if d and os.access(os.path.join(d, "vc_grow_seg_from_seed"), os.X_OK):
            return d
    raise ToolError("no VC3D tool directory: pass --kit <dir>/bin, set VC_BIN, or put vc_grow_seg_from_seed on PATH "
                    "(see cloud-grow/tools/BUILD_TOOLS.md)")


def tool_env(bin_dir: str, **extra) -> dict:
    """LD_LIBRARY_PATH=<kit>/lib: a portable kit's binaries fail with exit 127 without it."""
    env = dict(os.environ)
    lib = os.path.join(os.path.dirname(os.path.abspath(bin_dir)), "lib")
    if os.path.isdir(lib):
        env["LD_LIBRARY_PATH"] = lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env.update({k: str(v) for k, v in extra.items()})
    return env


def identify(bin_dir: str, pins: dict | None = None, names=REQUIRED_TOOLS) -> dict:
    """{tool: {path, md5, file, pin, pin_status}}; pin_status in PINNED | MISMATCH | NO-PIN | MISSING."""
    pins = pins if pins is not None else load_pins()
    out = {}
    for n in names:
        p = os.path.join(bin_dir, n)
        if not os.path.isfile(p):
            out[n] = {"path": p, "md5": None, "file": None, "pin": pins.get(n), "pin_status": "MISSING"}
            continue
        m = md5_file(p)
        pin = pins.get(n) or {}
        want, pre = pin.get("md5"), pin.get("md5_prefix")
        if want:
            st = "PINNED" if m == want else "MISMATCH"
        elif pre:
            st = "PINNED" if m.startswith(pre) else "MISMATCH"
        else:
            st = "NO-PIN"
        out[n] = {"path": p, "md5": m, "file": file_type(p), "pin": pin or None, "pin_status": st}
    return out


def gate(ident: dict, allow_unpinned: bool = False, announce=None) -> str:
    """Raise ToolError unless every tool is PINNED (or the operator accepted otherwise). Returns the run's
    overall pin status string for the manifest."""
    bad = {n: v for n, v in ident.items() if v["pin_status"] in ("MISSING", "MISMATCH", "NO-PIN")}
    missing = [n for n, v in bad.items() if v["pin_status"] == "MISSING"]
    if missing:
        raise ToolError(f"required tool(s) missing: {missing}")
    if not bad:
        return "PINNED"
    detail = "; ".join(f"{n}: md5 {v['md5']} is {v['pin_status']} (pin {v['pin']})" for n, v in bad.items())
    if not allow_unpinned:
        raise ToolError("tool pin gate FAILED (a different build gives different geometry): " + detail +
                        " -- pass --allow-unpinned to run anyway; it will be recorded in the manifest")
    msg = "UNPINNED tools accepted by operator: " + detail
    (announce or (lambda m: print("[cloud-grow] " + m, flush=True)))(msg)
    return "UNPINNED-ALLOWED"
