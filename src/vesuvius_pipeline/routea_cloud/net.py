"""Network helpers: every transfer is verified (sha256 for pinned files, size + md5/ETag for open-data objects); any failure raises (no silent fallback)."""
from __future__ import annotations

import hashlib
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

S3_ENDPOINT_ENV = "ROUTEA_S3_ENDPOINT"          # e.g. a LAN mirror of the bucket (tests, proof run); default = the public AWS endpoint
UA = {"User-Agent": "routeA-cloud-deploy/1"}


class Counter:
    def __init__(self):
        self.bytes = 0
        self.files = 0
        self._l = threading.Lock()

    def add(self, n, f=0):
        with self._l:
            self.bytes += n
            self.files += f


DOWNLOADED = Counter()          # bytes fetched over the network by this process (reported by the driver)


class FetchError(RuntimeError):
    pass


def sha256_file(path, bufsize: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(bufsize), b""):
            h.update(b)
    return h.hexdigest()


def md5_file(path, bufsize: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(bufsize), b""):
            h.update(b)
    return h.hexdigest()


def _open(url, headers=None, timeout=60):
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout)


def download(url: str, dst, sha256: str | None = None, retries: int = 6, chunk: int = 1 << 20) -> Path:
    """Resumable (Range) download to `dst`; verifies sha256 when given. A wrong hash deletes the file and raises."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and sha256 and sha256_file(dst) == sha256:
        return dst
    part = dst.with_name(dst.name + ".part")
    last = None
    for attempt in range(retries):
        try:
            have = part.stat().st_size if part.exists() else 0
            hdr = {"Range": f"bytes={have}-"} if have else {}
            with _open(url, hdr, timeout=60) as r:
                if have and r.status == 200:                   # server ignored Range: restart
                    have = 0
                mode = "ab" if have else "wb"
                with open(part, mode) as f:
                    while True:
                        b = r.read(chunk)
                        if not b:
                            break
                        f.write(b)
                        DOWNLOADED.add(len(b))
            break
        except urllib.error.HTTPError as e:
            if e.code == 416 and part.exists():                # range past EOF: the part is complete
                break
            last = e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last = e
        time.sleep(min(30, 2 ** attempt))
    else:
        raise FetchError(f"download failed after {retries} tries: {url}: {last}")
    if sha256:
        got = sha256_file(part)
        if got != sha256:
            part.unlink(missing_ok=True)
            raise FetchError(f"sha256 mismatch for {url}: expected {sha256}, got {got}")
    os.replace(part, dst)
    return dst


# ---- anonymous S3 (public bucket) ---------------------------------------------------------------------------------------------------
def s3_base(bucket: str) -> str:
    ep = os.environ.get(S3_ENDPOINT_ENV)
    return (ep.rstrip("/") + "/" + bucket) if ep else f"https://{bucket}.s3.amazonaws.com"


def s3_list(bucket: str, prefix: str, timeout: int = 60) -> list[dict]:
    """[{key,size,etag}] under `prefix` (ListObjectsV2, anonymous, paginated)."""
    out, token = [], None
    base = s3_base(bucket)
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        url = base + "/?" + urllib.parse.urlencode(q)
        last = None
        for attempt in range(6):
            try:
                with _open(url, timeout=timeout) as r:
                    root = ET.fromstring(r.read())
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, ET.ParseError) as e:
                last = e
                time.sleep(min(20, 2 ** attempt))
        else:
            raise FetchError(f"S3 list failed: {url}: {last}")
        ns = {"s": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}
        q_ = (lambda t: f"s:{t}") if ns else (lambda t: t)
        for c in root.findall(q_("Contents"), ns):
            out.append({"key": c.find(q_("Key"), ns).text, "size": int(c.find(q_("Size"), ns).text), "etag": (c.find(q_("ETag"), ns).text or "").strip('"')})
        trunc = root.find(q_("IsTruncated"), ns)
        if trunc is not None and trunc.text == "true":
            token = root.find(q_("NextContinuationToken"), ns).text
        else:
            return out


def _get_object(bucket: str, key: str, dst: Path, size: int, etag: str, retries: int = 6) -> int:
    url = s3_base(bucket) + "/" + urllib.parse.quote(key)
    part = dst.with_name(dst.name + ".part")
    last = None
    for attempt in range(retries):
        try:
            h = hashlib.md5()
            n = 0
            with _open(url, timeout=120) as r, open(part, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b); h.update(b); n += len(b)
            if n != size:
                raise FetchError(f"size {n} != listed {size}")
            if etag and "-" not in etag and h.hexdigest() != etag:
                raise FetchError(f"md5 {h.hexdigest()} != etag {etag}")
            os.replace(part, dst)
            DOWNLOADED.add(n, 1)
            return n
        except (FetchError, urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last = e
            time.sleep(min(20, 2 ** attempt))
    raise FetchError(f"object failed after {retries} tries: {key}: {last}")


def s3_sync(bucket: str, prefix: str, dst_dir, workers: int = 32, log=print) -> dict:
    """Mirror every object under `prefix` into dst_dir/<relative key>; idempotent (a file with the listed size is kept), verified, raises on any failure.
    Returns {objects, bytes, downloaded_objects, downloaded_bytes, seconds}."""
    t0 = time.time()
    dst_dir = Path(dst_dir)
    objs = [o for o in s3_list(bucket, prefix.rstrip("/") + "/") if not o["key"].endswith("/")]
    if not objs:
        raise FetchError(f"no objects under s3://{bucket}/{prefix}")
    todo = []
    for o in objs:
        rel = o["key"][len(prefix.rstrip("/")) + 1:]
        if ".." in Path(rel).parts or rel.startswith("/"):
            raise FetchError(f"unsafe key {o['key']}")
        dst = dst_dir / rel
        if dst.exists() and dst.stat().st_size == o["size"]:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        todo.append((o, dst))
    got = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_get_object, bucket, o["key"], dst, o["size"], o["etag"]) for o, dst in todo]
        done = 0
        for f in as_completed(futs):
            got += f.result()                       # raises on the first failure: fail loud
            done += 1
            if done % 2000 == 0:
                log(f"  s3 {prefix}: {done}/{len(todo)} objects, {got / 1e9:.2f} GB")
    bad = [o["key"] for o in objs if not (dst_dir / o["key"][len(prefix.rstrip('/')) + 1:]).exists()]
    if bad:
        raise FetchError(f"{len(bad)} objects missing after sync, e.g. {bad[:3]}")
    return {"objects": len(objs), "bytes": sum(o["size"] for o in objs), "downloaded_objects": len(todo), "downloaded_bytes": got, "seconds": round(time.time() - t0, 1)}
