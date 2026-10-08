"""Fetch + verify everything the box needs; idempotent, resumable, FAIL LOUD (no fallbacks). Pins live in <package root>/pins/."""
from __future__ import annotations

import json
import os
import subprocess
import tarfile
import time
from pathlib import Path

from . import net

KIT_URL_ENV = "KIT_URL"


class BootstrapError(RuntimeError):
    pass


def package_root() -> Path:
    """The deploy checkout root (contains pins/, umbilicus/, src/). ROUTEA_ROOT overrides (tests)."""
    if os.environ.get("ROUTEA_ROOT"):
        return Path(os.environ["ROUTEA_ROOT"])
    return Path(__file__).resolve().parents[3]


def load_pin(name: str, root: Path | None = None) -> dict:
    p = (root or package_root()) / "pins" / name
    if not p.is_file():
        raise BootstrapError(f"missing pin file {p}")
    return json.loads(p.read_text())


# ---- kit -------------------------------------------------------------------------------------------------------------------------------
def _safe_extract(tar: tarfile.TarFile, dst: Path) -> None:
    dst = dst.resolve()
    for m in tar.getmembers():
        t = (dst / m.name).resolve()
        if not str(t).startswith(str(dst) + os.sep) and t != dst:
            raise BootstrapError(f"unsafe path in kit tarball: {m.name}")
        if m.issym() or m.islnk():
            lt = (t.parent / m.linkname).resolve()
            if not str(lt).startswith(str(dst) + os.sep):
                raise BootstrapError(f"unsafe link in kit tarball: {m.name} -> {m.linkname}")
    tar.extractall(dst)


def verify_kit(kit: Path, pin: dict) -> list[str]:
    """Per-file md5 of the pinned files + the three identity md5s; returns the list of problems (empty = verified)."""
    probs = []
    for rel, md5 in pin["files_md5"].items():
        p = kit / rel
        if not p.is_file():
            probs.append(f"missing {rel}")
        elif net.md5_file(p) != md5:
            probs.append(f"md5 mismatch {rel}")
    return probs


def selfcheck_kit(kit: Path) -> None:
    env = dict(os.environ, LD_LIBRARY_PATH=str(kit / "lib"))
    r = subprocess.run([str(kit / "bin" / "vc_tifxyz_selfcross"), "--help"], capture_output=True, text=True, env=env, timeout=60)
    if r.returncode != 0 or "transverse" not in (r.stdout + r.stderr):
        raise BootstrapError(f"kit self-check failed: vc_tifxyz_selfcross --help rc={r.returncode}: {(r.stderr or r.stdout)[-300:]} (missing system library? glibc >= 2.17 needed)")
    r = subprocess.run([str(kit / "bin" / "vc_grow_seg_from_seed"), "--help"], capture_output=True, text=True, env=env, timeout=60)
    if r.returncode != 0:
        raise BootstrapError(f"kit self-check failed: vc_grow_seg_from_seed --help rc={r.returncode}: {(r.stderr or r.stdout)[-300:]}")


def ensure_kit(work: Path, kit_url: str | None = None, root: Path | None = None, log=print) -> dict:
    """-> {'dir': Path, 'md5': {...}, 'downloaded': bool}. Existing, fully verified kit dir is reused (no network)."""
    pin = load_pin("kit.json", root)
    kit = work / "kit"
    mark = kit / ".verified"
    if mark.is_file() and mark.read_text().strip() == pin["tarball_sha256"] and not verify_kit(kit, pin):
        selfcheck_kit(kit)
        return {"dir": kit, "md5": pin["identity_md5"], "downloaded": False}
    url = kit_url or os.environ.get(KIT_URL_ENV) or pin.get("url") or ""
    if not url:
        raise BootstrapError(f"no kit available: set {KIT_URL_ENV} (or --kit-url) to the HTTPS URL of {pin['tarball_name']} (sha256 {pin['tarball_sha256']}); see README 'Kit hosting'")
    tgz = work / "downloads" / pin["tarball_name"]
    t0 = time.time()
    net.download(url, tgz, sha256=pin["tarball_sha256"])
    log(f"kit: downloaded + sha256-verified {pin['tarball_name']} in {time.time() - t0:.0f} s")
    import shutil
    shutil.rmtree(kit, ignore_errors=True)
    kit.mkdir(parents=True)
    with tarfile.open(tgz) as tf:
        _safe_extract(tf, kit)
    probs = verify_kit(kit, pin)
    if probs:
        raise BootstrapError(f"kit verification failed ({len(probs)} problems): {probs[:5]}")
    for exe in ("vc_grow_seg_from_seed", "vc_tifxyz_selfcross"):
        os.chmod(kit / "bin" / exe, 0o755)
    selfcheck_kit(kit)
    mark.write_text(pin["tarball_sha256"])
    return {"dir": kit, "md5": pin["identity_md5"], "downloaded": True}


# ---- per-scroll inputs --------------------------------------------------------------------------------------------------------------------
def write_volume_meta(zarr_dir: Path, scroll: str, voxel_um: float) -> None:
    """VC3D wants meta.json next to the volume (the prediction ships metadata.json): same content stages/data._write_meta writes."""
    shape = None
    za = zarr_dir / "0" / ".zarray"
    if za.is_file():
        shape = json.loads(za.read_text()).get("shape")
    meta = {"type": "vol", "format": "zarr", "uuid": f"{scroll}_routeA_cloud", "name": f"{scroll}-routeA-cloud", "voxelsize": float(voxel_um), "min": 0.0, "max": 255.0, "source": "open-data surface prediction"}
    if shape and len(shape) == 3:
        meta.update({"slices": shape[0], "height": shape[1], "width": shape[2]})
    (zarr_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def ensure_scroll(work: Path, scroll: str, root: Path | None = None, log=print, workers: int = 32) -> dict:
    """Prediction zarr + normal grids from the public bucket into work/data/<scroll>/. -> {'pred': Path, 'grids': Path, 'voxel_um': float, 'stats': {...}}."""
    sp = load_pin("scrolls.json", root)
    if scroll not in sp["scrolls"]:
        raise BootstrapError(f"{scroll}: not in pins/scrolls.json (known: {sorted(sp['scrolls'])})")
    s = sp["scrolls"][scroll]
    if not s.get("voxel_um"):
        raise BootstrapError(f"{scroll}: voxel_um unknown in pins (the tracer needs the volume's voxel size); refusing to guess")
    base = work / "data" / scroll
    stats = {}
    for kind in ("prediction", "grids"):
        name = s[kind]
        prefix = f"{scroll}/representations/predictions/surfaces/{name}"
        done = base / (name + ".complete")
        if done.is_file() and done.read_text().strip() == json.dumps({"objects": s.get(f"{kind}_objects"), "bytes": s.get(f"{kind}_bytes")}, sort_keys=True):
            stats[kind] = {"cached": True}
            continue
        log(f"{scroll}: syncing {kind} s3://{sp['bucket']}/{prefix}")
        st = net.s3_sync(sp["bucket"], prefix, base / name, workers=workers, log=log)
        for k in ("objects", "bytes"):
            exp = s.get(f"{kind}_{k}")
            if exp is not None and exp != st[k]:
                raise BootstrapError(f"{scroll} {kind}: S3 listing {k}={st[k]} differs from the pin {exp}: upstream changed; re-pin deliberately")
        done.write_text(json.dumps({"objects": st["objects"], "bytes": st["bytes"]}, sort_keys=True))
        stats[kind] = st
    pred, grids = base / s["prediction"], base / s["grids"]
    if not (pred / "0" / ".zarray").is_file():
        raise BootstrapError(f"{pred}: no level-0 array after sync")
    if not any((grids / "xy").glob("*.grid")):
        raise BootstrapError(f"{grids}: no xy/*.grid after sync")
    write_volume_meta(pred, scroll, s["voxel_um"])
    return {"pred": pred, "grids": grids, "voxel_um": float(s["voxel_um"]), "stats": stats}
