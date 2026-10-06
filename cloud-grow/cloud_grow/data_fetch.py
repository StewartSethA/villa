"""Resume-safe data pullers with a size preview: CT OME-Zarr levels, surface prediction, normal grids.

Upstream layout (anonymous, public, region us-east-1; same bucket `vesuvius-challenge-open-data` the fleet's
`vpipe data ensure` pulls with rclone, src/vesuvius_pipeline/data.py):
    <scroll>/volumes/<scan>-<um>um-1.2m-<keV>keV-masked.zarr/{.zattrs,.zgroup,0/..,1/..,..5/..}
    <scroll>/representations/predictions/surfaces/<scan>-surface-<ts>-surface-m7-L0-th0.2.zarr
    <scroll>/representations/predictions/surfaces/<...>.normal-grids/{xy,xz,yz}/
Scroll facts (volume ids, zarr names, voxel pitch, prediction names + bytes) are in config/scrolls.json, exported
from the fleet registry (scroll_registry/<scroll>.json, upstream-sourced fields). A scroll with `null` there
(PHerc0332, PHercParis4: multi-scan) must be given --volume-name / --prediction-name explicitly.

Guarantees: nothing is downloaded before the preview is printed (or --yes); each file downloads to `<name>.part`
and resumes with an HTTP Range request; the final file is size-checked against the listing and, when the S3 ETag is a
plain md5 (single-part object), md5-checked; a file that fails is re-fetched, never kept; the store gets
`.cache_complete` only when EVERY file verified, `.fetching` exists while in progress. Per-file results are appended to
`fetch_log.jsonl` in the store root. No credentials are used or stored.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

BUCKET = "vesuvius-challenge-open-data"
S3_REGION = "us-east-1"
DEFAULT_BASE = f"https://{BUCKET}.s3.{S3_REGION}.amazonaws.com/"
SCROLLS_JSON = Path(__file__).resolve().parent.parent / "config" / "scrolls.json"


@dataclass(frozen=True)
class Obj:
    key: str
    size: int
    etag: str = ""


class FetchError(RuntimeError):
    pass


# ------------------------------------------------------------------ sources
class HttpS3Source:
    """Anonymous S3 over HTTPS (ListObjectsV2 + ranged GET). `base` may point at any S3-compatible mirror."""

    def __init__(self, base: str = DEFAULT_BASE, timeout: float = 60.0):
        self.base = base if base.endswith("/") else base + "/"
        self.timeout = timeout

    def list(self, prefix: str, delimiter: str = "") -> tuple[list[Obj], list[str]]:
        objs, prefixes, token = [], [], None
        while True:
            q = {"list-type": "2", "prefix": prefix}
            if delimiter:
                q["delimiter"] = delimiter
            if token:
                q["continuation-token"] = token
            url = self.base + "?" + urllib.parse.urlencode(q)
            with urllib.request.urlopen(url, timeout=self.timeout) as r:
                root = ET.fromstring(r.read())
            ns = re.match(r"\{.*\}", root.tag)
            ns = ns.group(0) if ns else ""
            for c in root.findall(f"{ns}Contents"):
                objs.append(Obj(c.findtext(f"{ns}Key"), int(c.findtext(f"{ns}Size") or 0), (c.findtext(f"{ns}ETag") or "").strip('"')))
            for p in root.findall(f"{ns}CommonPrefixes"):
                prefixes.append(p.findtext(f"{ns}Prefix"))
            if (root.findtext(f"{ns}IsTruncated") or "").lower() == "true":
                token = root.findtext(f"{ns}NextContinuationToken")
            else:
                return objs, prefixes

    def open_range(self, key: str, start: int = 0):
        req = urllib.request.Request(self.base + urllib.parse.quote(key))
        if start:
            req.add_header("Range", f"bytes={start}-")
        return urllib.request.urlopen(req, timeout=self.timeout)


class DirSource:
    """A local directory laid out like the bucket (tests, or a mirror already on disk)."""

    def __init__(self, root: str):
        self.root = Path(root)

    def list(self, prefix: str, delimiter: str = "") -> tuple[list[Obj], list[str]]:
        objs, prefixes = [], set()
        base = self.root
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            key = p.relative_to(base).as_posix()
            if not key.startswith(prefix):
                continue
            rest = key[len(prefix):]
            if delimiter and delimiter in rest:
                prefixes.add(prefix + rest.split(delimiter)[0] + delimiter)
                continue
            with open(p, "rb") as fh:
                et = hashlib.md5(fh.read()).hexdigest()
            objs.append(Obj(key, p.stat().st_size, et))
        return objs, sorted(prefixes)

    def open_range(self, key: str, start: int = 0):
        fh = open(self.root / key, "rb")
        fh.seek(start)
        return fh


# ------------------------------------------------------------------ registry
def load_scrolls(path: str | os.PathLike | None = None) -> dict:
    with open(path or SCROLLS_JSON) as fh:
        return json.load(fh)["scrolls"]


def scroll_info(scroll: str, path=None) -> dict:
    sc = load_scrolls(path)
    if scroll not in sc:
        raise FetchError(f"unknown scroll {scroll!r}; known: {sorted(sc)}")
    return sc[scroll]


# ------------------------------------------------------------------ planning
def level_of(rel: str) -> int | None:
    """Pyramid level of a path relative to the zarr root (`3/0/1/2` -> 3), None for root metadata."""
    first = rel.split("/", 1)[0]
    return int(first) if first.isdigit() else None


def plan_zarr(src, zarr_prefix: str, levels: set[int] | None) -> list[Obj]:
    prefix = zarr_prefix.rstrip("/") + "/"
    objs, _ = src.list(prefix)
    out = []
    for o in objs:
        lv = level_of(o.key[len(prefix):])
        if lv is None or levels is None or lv in levels:
            out.append(o)
    return out


def find_grids_prefix(src, scroll: str, scan_ts: str) -> str | None:
    base = f"{scroll}/representations/predictions/surfaces/"
    _, prefixes = src.list(base, delimiter="/")
    grids = [p for p in prefixes if p.rstrip("/").endswith(".normal-grids")]
    same = [g for g in grids if os.path.basename(g.rstrip("/")).startswith(scan_ts)] if scan_ts else []
    pick = same or grids
    if not pick:
        return None
    return sorted(pick, key=lambda n: ("m7" not in n, "L0" not in n, n))[0]


def preview(objs: list[Obj], dest: str, label: str) -> dict:
    total = sum(o.size for o in objs)
    probe = os.path.abspath(dest)
    while not os.path.isdir(probe) and os.path.dirname(probe) != probe:
        probe = os.path.dirname(probe)                       # nearest existing ancestor: the preview creates nothing
    free = shutil.disk_usage(probe).free
    return {"label": label, "n_files": len(objs), "total_gb": round(total / 2**30, 3), "free_gb_at_dest": round(free / 2**30, 1),
            "fits": total * 1.05 + 20 * 2**30 < free, "dest": dest}


# ------------------------------------------------------------------ download engine
def _etag_is_md5(etag: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{32}", etag or ""))


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _file_ok(path: str, o: Obj, check_md5: bool) -> bool:
    try:
        if os.path.getsize(path) != o.size:
            return False
    except OSError:
        return False
    return (not check_md5) or (not _etag_is_md5(o.etag)) or _md5(path) == o.etag


def fetch_one(src, o: Obj, dest_file: str, retries: int = 4, check_md5: bool = True) -> str:
    """Returns 'skipped' | 'downloaded' | 'resumed'. Raises FetchError after `retries` failed verifications."""
    if os.path.exists(dest_file) and _file_ok(dest_file, o, check_md5):
        return "skipped"
    os.makedirs(os.path.dirname(dest_file), exist_ok=True)
    part = dest_file + ".part"
    how = "downloaded"
    last = ""
    for attempt in range(retries):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have > o.size:
            os.remove(part)
            have = 0
        if have and attempt == 0:
            how = "resumed"
        try:
            if have < o.size or o.size == 0:
                with src.open_range(o.key, have) as r, open(part, "ab" if have else "wb") as out:
                    shutil.copyfileobj(r, out, 1 << 20)
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(min(2 ** attempt, 20))
            continue
        if _file_ok(part, o, check_md5):
            os.replace(part, dest_file)
            return how
        last = f"verification failed (size {os.path.getsize(part) if os.path.exists(part) else None} vs {o.size}, etag {o.etag})"
        try:
            os.remove(part)                     # a partial that verifies wrong is never resumed again
        except OSError:
            pass
    raise FetchError(f"{o.key}: {last}")


def fetch_objects(src, objs: list[Obj], strip_prefix: str, dest: str, workers: int = 16, check_md5: bool = True,
                  progress=None) -> dict:
    """Download all `objs` under `dest` (key minus strip_prefix). Marker discipline: `.fetching` while running,
    `.cache_complete` only if every file verified. Returns counts; failures are listed, never swallowed."""
    os.makedirs(dest, exist_ok=True)
    Path(dest, ".fetching").touch()
    Path(dest, ".cache_complete").unlink(missing_ok=True)
    counts = {"skipped": 0, "downloaded": 0, "resumed": 0}
    failed: list[str] = []
    lock = threading.Lock()
    logp = os.path.join(dest, "fetch_log.jsonl")

    def work(o: Obj):
        rel = o.key[len(strip_prefix):]
        return o, fetch_one(src, o, os.path.join(dest, rel), check_md5=check_md5)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex, open(logp, "a") as log:
        futs = {ex.submit(work, o): o for o in objs}
        n = 0
        for f in as_completed(futs):
            o = futs[f]
            n += 1
            try:
                _o, how = f.result()
                with lock:
                    counts[how] += 1
                    log.write(json.dumps({"key": o.key, "size": o.size, "etag": o.etag, "result": how}) + "\n")
            except FetchError as e:
                with lock:
                    failed.append(str(e))
                    log.write(json.dumps({"key": o.key, "size": o.size, "result": "FAILED", "error": str(e)[:300]}) + "\n")
            if progress and n % 500 == 0:
                progress(n, len(objs))
    if failed:
        return {**counts, "failed": failed, "complete": False}
    Path(dest, ".fetching").unlink(missing_ok=True)
    Path(dest, ".cache_complete").touch()
    return {**counts, "failed": [], "complete": True}


# ------------------------------------------------------------------ VC3D meta.json
def write_meta(zarr_dir: str, scroll: str, voxel_um: float | None = None, source: str = "") -> bool:
    """VC3D needs meta.json; the zarr ships .zattrs. Derive it (port of data._write_meta). Never overwrites."""
    zd = Path(zarr_dir)
    if (zd / "meta.json").exists():
        return False
    shape = None
    for cand in ("0/.zarray", "0/zarr.json"):
        p = zd / cand
        if p.exists():
            try:
                shape = json.loads(p.read_text()).get("shape")
            except ValueError:
                pass
    vox = None
    m = re.search(r"-([0-9.]+)um-", zd.name)
    if m:
        vox = float(m.group(1))
    vox = vox or voxel_um or 9.362
    meta = {"type": "vol", "format": "zarr", "uuid": f"{scroll}_localcache", "name": f"{scroll}-localcache", "voxelsize": vox,
            "min": 0.0, "max": 255.0, "source": source}
    if shape and len(shape) == 3:
        meta.update({"slices": shape[0], "height": shape[1], "width": shape[2]})
    (zd / "meta.json").write_text(json.dumps(meta, indent=2))
    return True
