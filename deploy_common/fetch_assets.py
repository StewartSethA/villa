#!/usr/bin/env python3
"""Verified, resumable, fail-loud asset downloader shared by the Route A / Route B deploy branches.

stdlib only (runs before any venv exists).  No default is a private host: every URL comes from a manifest.

Manifest (JSON):  {"assets": [ {asset}, ... ]}   asset =
  name      unique id (ledger key)
  kind      "http"      one file: url, dest, optional size / md5 / sha256
            "s3prefix"  mirror every object under bucket/prefix (anonymous ListObjectsV2) into dest/;
                        optional include / exclude regexes on the key RELATIVE to prefix; each object is
                        verified by size and by its S3 ETag (md5) unless the ETag is multipart
            "s3keys"    explicit object keys (bucket, prefix, keys[]) -> dest/<key>; a 404 is an absent chunk (zeros), counted
            "tar"       like http, then extract (safe) into dest/ and delete nothing
  dest      path relative to --dest
  optional  true -> a failure is reported loudly but does not make the exit code non-zero
  note      free text, printed

Verification policy (announced per asset, never silent):
  md5/sha256 pinned in the manifest  -> hash checked (strong)
  only size known (HEAD Content-Length or manifest size) -> SIZE-ONLY, printed as such; md5 is then RECORDED in the
  ledger so a later run can detect corruption of the local copy.
Resume: HTTP files are fetched in 32 MiB blocks (parallel), a <dest>.part + <dest>.part.map keep the finished
blocks; re-running continues.  A finished asset leaves  <dest root>/.fetched/<name>.json  (size, md5, source, time);
a re-run with a valid ledger entry and matching file size downloads nothing.
Exit status: 0 only if every non-optional asset is present and verified.  Totals (bytes over the network, wall
seconds) are appended to <dest root>/.fetched/_totals.json for the proof reports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

UA = {"User-Agent": "routeB-deploy-fetch/1"}
BLOCK = 32 << 20
S3_ENDPOINT_ENV = "DEPLOY_S3_ENDPOINT"      # optional mirror of the open-data bucket (tests); default = public AWS endpoint


def _p(m):
    print(m, flush=True)


class FetchError(RuntimeError):
    pass


class _Counter:
    def __init__(self):
        self.bytes = 0
        self._l = threading.Lock()

    def add(self, n):
        with self._l:
            self.bytes += n


NET = _Counter()


def md5_file(p, buf=1 << 22):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(buf), b""):
            h.update(b)
    return h.hexdigest()


def sha256_file(p, buf=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(buf), b""):
            h.update(b)
    return h.hexdigest()


def _open(url, headers=None, timeout=60, method="GET"):
    req = urllib.request.Request(url, headers={**UA, **(headers or {})}, method=method)
    return urllib.request.urlopen(req, timeout=timeout)


def _retry(fn, what, tries=6):
    last = None
    for a in range(tries):
        try:
            return fn()
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, ET.ParseError, FetchError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403, 404):
                raise FetchError(f"{what}: HTTP {e.code} {e.reason}") from e
            last = e
            time.sleep(min(20, 2 ** a))
    raise FetchError(f"{what}: failed after {tries} tries: {last}")


def head_size(url):
    def go():
        with _open(url, method="HEAD") as r:
            n = r.headers.get("Content-Length")
            return int(n) if n is not None else None
    return _retry(go, f"HEAD {url}")


# ---------------------------------------------------------------- http (parallel blocks, resumable)
def _get_block(url, part, off, n):
    def go():
        with _open(url, {"Range": f"bytes={off}-{off + n - 1}"}, timeout=120) as r:
            if r.status != 206:
                raise FetchError(f"server ignored Range (HTTP {r.status})")
            got = 0
            with open(part, "r+b") as f:
                f.seek(off)
                while got < n:
                    b = r.read(min(1 << 20, n - got))
                    if not b:
                        raise FetchError(f"short block at {off}: {got}/{n}")
                    f.write(b)
                    got += len(b)
                    NET.add(len(b))
    _retry(go, f"block {off} of {url}")


def download_http(url, dst: Path, size=None, connections=8, log=_p):
    dst.parent.mkdir(parents=True, exist_ok=True)
    total = head_size(url)
    if size is not None and total is not None and size != total:
        raise FetchError(f"{url}: manifest size {size} != server Content-Length {total}")
    total = total if total is not None else size
    part = dst.with_name(dst.name + ".part")
    mapf = dst.with_name(dst.name + ".part.map")
    if total is None:                               # unknown length: single stream, no resume
        def go():
            with _open(url, timeout=120) as r, open(part, "wb") as f:
                for b in iter(lambda: r.read(1 << 20), b""):
                    f.write(b); NET.add(len(b))
        _retry(go, f"GET {url}")
        os.replace(part, dst)
        return dst.stat().st_size
    done = set()
    if part.exists() and mapf.exists():
        try:
            m = json.loads(mapf.read_text())
            if m.get("total") == total and m.get("block") == BLOCK:
                done = set(m["done"])
        except Exception:
            done = set()
    if not part.exists() or part.stat().st_size != total:
        done = set()
        with open(part, "wb") as f:
            f.truncate(total)
    blocks = [(i, i * BLOCK, min(BLOCK, total - i * BLOCK)) for i in range((total + BLOCK - 1) // BLOCK)]
    todo = [b for b in blocks if b[0] not in done]
    t0, base = time.time(), NET.bytes
    lock, lastsave, lastlog = threading.Lock(), [time.time()], [time.time()]

    def work(b):
        _get_block(url, part, b[1], b[2])
        with lock:
            done.add(b[0])
            if time.time() - lastsave[0] > 5:
                mapf.write_text(json.dumps({"total": total, "block": BLOCK, "done": sorted(done)}))
                lastsave[0] = time.time()
            if time.time() - lastlog[0] > 30:
                lastlog[0] = time.time()
                g = (NET.bytes - base) / 1e9
                log(f"    {dst.name}: {len(done)}/{len(blocks)} blocks, {g:.2f} GB this run, {g * 1e3 / max(time.time() - t0, 1e-9):.0f} MB/s")
    try:
        with ThreadPoolExecutor(max_workers=connections) as ex:
            for f in as_completed([ex.submit(work, b) for b in todo]):
                f.result()
    finally:
        mapf.write_text(json.dumps({"total": total, "block": BLOCK, "done": sorted(done)}))
    if len(done) != len(blocks) or part.stat().st_size != total:
        raise FetchError(f"{url}: incomplete ({len(done)}/{len(blocks)} blocks)")
    os.replace(part, dst)
    mapf.unlink(missing_ok=True)
    return total


# ---------------------------------------------------------------- anonymous S3
def s3_base(bucket):
    ep = os.environ.get(S3_ENDPOINT_ENV)
    return (ep.rstrip("/") + "/" + bucket) if ep else f"https://{bucket}.s3.amazonaws.com"


def s3_list(bucket, prefix):
    out, token, base = [], None, s3_base(bucket)
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        url = base + "/?" + urllib.parse.urlencode(q)

        def go():
            with _open(url) as r:
                return ET.fromstring(r.read())
        root = _retry(go, f"S3 list {url}")
        ns = {"s": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}
        t = (lambda x: f"s:{x}") if ns else (lambda x: x)
        for c in root.findall(t("Contents"), ns):
            out.append({"key": c.find(t("Key"), ns).text, "size": int(c.find(t("Size"), ns).text),
                        "etag": (c.find(t("ETag"), ns).text or "").strip('"')})
        tr = root.find(t("IsTruncated"), ns)
        if tr is not None and tr.text == "true":
            token = root.find(t("NextContinuationToken"), ns).text
        else:
            return out


def _s3_get(bucket, key, dst: Path, size, etag):
    url = s3_base(bucket) + "/" + urllib.parse.quote(key)
    part = dst.with_name(dst.name + ".part")

    def go():
        h, n = hashlib.md5(), 0
        with _open(url, timeout=120) as r, open(part, "wb") as f:
            for b in iter(lambda: r.read(1 << 20), b""):
                f.write(b); h.update(b); n += len(b)
        if n != size:
            raise FetchError(f"size {n} != listed {size} for {key}")
        if etag and "-" not in etag and h.hexdigest() != etag:
            raise FetchError(f"md5 {h.hexdigest()} != ETag {etag} for {key}")
        os.replace(part, dst)
        NET.add(n)
        return n
    return _retry(go, f"S3 object {key}")


def sync_s3prefix(bucket, prefix, dst: Path, include=None, exclude=None, workers=32, log=_p, subprefixes=None):
    """subprefixes: list only these key prefixes (relative to prefix; 'a/b/' = a directory, '.zattrs' = one key) instead of the whole
    tree -- a z-slab of a 130k-object zarr is listed in seconds, not minutes."""
    prefix = prefix.rstrip("/")
    inc = re.compile(include) if include else None
    exc = re.compile(exclude) if exclude else None
    objs = []
    listing = []
    for sp in (subprefixes or [""]):
        listing += s3_list(bucket, prefix + "/" + sp)
    for o in listing:
        rel = o["key"][len(prefix) + 1:]
        if not rel or rel.endswith("/") or (inc and not inc.search(rel)) or (exc and exc.search(rel)):
            continue
        if ".." in Path(rel).parts or rel.startswith("/"):
            raise FetchError(f"unsafe key {o['key']}")
        objs.append((o, rel))
    if not objs:
        raise FetchError(f"no objects under s3://{bucket}/{prefix}/ (include={include!r})")
    todo = []
    for o, rel in objs:
        d = dst / rel
        if d.exists() and d.stat().st_size == o["size"]:
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        todo.append((o, d))
    got = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_s3_get, bucket, o["key"], d, o["size"], o["etag"]) for o, d in todo]
        for i, f in enumerate(as_completed(futs), 1):
            got += f.result()
            if i % 2000 == 0:
                log(f"    s3 {prefix}: {i}/{len(todo)} objects, {got / 1e9:.2f} GB")
    bad = [rel for o, rel in objs if not (dst / rel).exists()]
    if bad:
        raise FetchError(f"{len(bad)} objects missing after sync, e.g. {bad[:3]}")
    return {"objects": len(objs), "bytes": sum(o["size"] for o, _ in objs), "fetched_objects": len(todo), "fetched_bytes": got}


def sync_s3keys(bucket, prefix, keys, dst: Path, workers=32, log=_p):
    """Fetch an explicit list of object keys (relative to prefix) -- e.g. the zarr chunks one render needs.  A 404 is an ABSENT chunk
    (sparse/masked volume: reads as zeros) and is counted and reported, not an error; every other failure raises."""
    prefix = prefix.rstrip("/")
    todo, have = [], 0
    for k in keys:
        if ".." in Path(k).parts or k.startswith("/"):
            raise FetchError(f"unsafe key {k}")
        d = dst / k
        if d.exists() or (dst / (k + ".absent")).exists():
            have += 1
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        todo.append((k, d))
    absent, got = [0], [0]

    def one(k, d):
        url = s3_base(bucket) + "/" + urllib.parse.quote(prefix + "/" + k)
        part = d.with_name(d.name + ".part")

        def go():
            try:
                with _open(url, timeout=120) as r:
                    etag = (r.headers.get("ETag") or "").strip('"')
                    h, n = hashlib.md5(), 0
                    with open(part, "wb") as f:
                        for b in iter(lambda: r.read(1 << 20), b""):
                            f.write(b); h.update(b); n += len(b)
            except urllib.error.HTTPError as e:
                if e.code in (403, 404):             # a missing key answers 403 on a bucket without anonymous ListBucket for GET
                    return -1
                raise
            if etag and "-" not in etag and h.hexdigest() != etag:
                raise FetchError(f"md5 {h.hexdigest()} != ETag {etag} for {k}")
            os.replace(part, d)
            NET.add(n)
            return n
        n = _retry(go, f"S3 object {k}")
        if n < 0:
            (dst / (k + ".absent")).write_text("")
            absent[0] += 1
        else:
            got[0] += n
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(one, k, d) for k, d in todo]), 1):
            f.result()
            if i % 500 == 0:
                log(f"    s3keys: {i}/{len(todo)} chunks, {got[0] / 1e9:.2f} GB")
    return {"objects": len(keys), "cached": have, "fetched_objects": len(todo) - absent[0], "absent": absent[0], "fetched_bytes": got[0], "bytes": got[0]}


# ---------------------------------------------------------------- tar
def safe_extract(tf: tarfile.TarFile, dst: Path):
    dst = dst.resolve()
    for m in tf.getmembers():
        t = (dst / m.name).resolve()
        if t != dst and not str(t).startswith(str(dst) + os.sep):
            raise FetchError(f"unsafe path in archive: {m.name}")
        if m.issym() or m.islnk():
            lt = (t.parent / m.linkname).resolve()
            if not str(lt).startswith(str(dst) + os.sep):
                raise FetchError(f"unsafe link in archive: {m.name} -> {m.linkname}")
    tf.extractall(dst)


# ---------------------------------------------------------------- driver
def _ledger(root: Path, name):
    return root / ".fetched" / f"{name}.json"


def fetch_one(a: dict, root: Path, connections=8, log=_p) -> dict:
    name, kind = a["name"], a["kind"]
    dst = root / a["dest"]
    L = _ledger(root, name)
    L.parent.mkdir(parents=True, exist_ok=True)
    t0, n0 = time.time(), NET.bytes
    if kind in ("http", "tar"):
        url = a["url"]
        want_md5, want_sha, want_size = a.get("md5"), a.get("sha256"), a.get("size")
        if L.exists() and dst.exists():
            try:
                led = json.loads(L.read_text())
                if led.get("size") == dst.stat().st_size and (not want_md5 or led.get("md5") == want_md5) \
                        and (not want_sha or led.get("sha256") == want_sha) and led.get("url") == url:
                    log(f"  [ok ] {name}: already verified ({led['size'] / 1e9:.3f} GB, {led['verify']})")
                    return {"name": name, "status": "cached", "bytes_net": 0}
            except Exception:
                pass
        if dst.exists() and (want_size is None or dst.stat().st_size == want_size):
            pass                                        # present but no ledger: verify below instead of re-downloading
        else:
            log(f"  [get] {name}: {url}")
            download_http(url, dst, size=want_size, connections=int(a.get("connections", connections)), log=log)
        size = dst.stat().st_size
        rec = {"name": name, "url": url, "size": size, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if want_size is not None and size != want_size:
            raise FetchError(f"{name}: size {size} != manifest {want_size}")
        if want_sha:
            rec["sha256"] = sha256_file(dst)
            if rec["sha256"] != want_sha:
                dst.unlink()
                raise FetchError(f"{name}: sha256 {rec['sha256']} != pinned {want_sha} (file deleted)")
            rec["verify"] = "sha256 pinned"
        if want_md5 or not want_sha:
            rec["md5"] = md5_file(dst)
            if want_md5:
                if rec["md5"] != want_md5:
                    dst.unlink()
                    raise FetchError(f"{name}: md5 {rec['md5']} != pinned {want_md5} (file deleted)")
                rec["verify"] = "md5 pinned"
            elif "verify" not in rec:
                rec["verify"] = "SIZE-ONLY vs server Content-Length (md5 recorded, not pinned)"
        if kind == "tar":
            ex = root / a.get("extract_to", a["dest"] + ".d")
            ex.mkdir(parents=True, exist_ok=True)
            with tarfile.open(dst) as tf:
                safe_extract(tf, ex)
        rec["bytes_net"] = NET.bytes - n0
        L.write_text(json.dumps(rec, indent=1))
        log(f"  [ok ] {name}: {size / 1e9:.3f} GB, {rec['verify']}, {rec['bytes_net'] / 1e9:.3f} GB over the network, {time.time() - t0:.0f} s")
        return {"name": name, "status": "fetched", "bytes_net": rec["bytes_net"]}
    if kind == "s3prefix":
        log(f"  [get] {name}: s3://{a['bucket']}/{a['prefix']} -> {a['dest']}")
        r = sync_s3prefix(a["bucket"], a["prefix"], dst, a.get("include"), a.get("exclude"), workers=int(a.get("workers", 32)), log=log, subprefixes=a.get("subprefixes"))
        r.update({"name": name, "verify": "size + S3 ETag(md5) per object", "bytes_net": NET.bytes - n0})
        L.write_text(json.dumps(r, indent=1))
        log(f"  [ok ] {name}: {r['objects']} objects {r['bytes'] / 1e9:.3f} GB ({r['fetched_objects']} fetched, {r['fetched_bytes'] / 1e9:.3f} GB), {time.time() - t0:.0f} s")
        return {"name": name, "status": "fetched" if r["fetched_objects"] else "cached", "bytes_net": r["bytes_net"]}
    if kind == "s3keys":
        log(f"  [get] {name}: {len(a['keys'])} object keys under s3://{a['bucket']}/{a['prefix']}")
        r = sync_s3keys(a["bucket"], a["prefix"], a["keys"], dst, workers=int(a.get("workers", 32)), log=log)
        r.update({"name": name, "verify": "S3 ETag(md5) per object; 404 = absent chunk (zeros)", "bytes_net": NET.bytes - n0})
        L.write_text(json.dumps(r, indent=1))
        log(f"  [ok ] {name}: {r['objects']} keys, {r['fetched_objects']} fetched ({r['fetched_bytes'] / 1e9:.3f} GB), {r['absent']} absent, {r['cached']} cached, {time.time() - t0:.0f} s")
        return {"name": name, "status": "fetched", "bytes_net": r["bytes_net"]}
    raise FetchError(f"{name}: unknown kind {kind!r}")


def fetch(manifest: dict | str | Path, dest, only=None, connections=8, log=_p) -> dict:
    """Library entry point.  Returns {"ok": bool, "failed": [...], "bytes_net": int, "assets": [...]}."""
    if not isinstance(manifest, dict):
        manifest = json.loads(Path(manifest).read_text())
    root = Path(dest)
    root.mkdir(parents=True, exist_ok=True)
    names = set(only) if only else None
    res, failed = [], []
    t0 = time.time()
    for a in manifest["assets"]:
        if names is not None and a["name"] not in names:
            continue
        if a.get("note"):
            log(f"  note {a['name']}: {a['note']}")
        try:
            res.append(fetch_one(a, root, connections, log))
        except Exception as e:                                  # per-item: report all failures, not just the first
            msg = f"{a['name']}: {type(e).__name__}: {e}"
            (log if a.get("optional") else lambda m: print(m, file=sys.stderr))(f"  [FAIL{' (optional)' if a.get('optional') else ''}] {msg}")
            if not a.get("optional"):
                failed.append(msg)
    tot = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "wall_s": round(time.time() - t0, 1),
           "bytes_net": NET.bytes, "failed": failed}
    p = root / ".fetched" / "_totals.json"
    prev = json.loads(p.read_text()) if p.exists() else []
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(prev + [tot], indent=1))
    return {"ok": not failed, "failed": failed, "bytes_net": NET.bytes, "assets": res}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--only", help="comma-separated asset names")
    ap.add_argument("--connections", type=int, default=8)
    ap.add_argument("--list", action="store_true", help="print the manifest (name, kind, size) and exit")
    a = ap.parse_args(argv)
    m = json.loads(Path(a.manifest).read_text())
    if a.list:
        for x in m["assets"]:
            print(f"{x['name']:40s} {x['kind']:9s} {x.get('size', '?')} {x.get('url') or 's3://%s/%s' % (x.get('bucket'), x.get('prefix'))}")
        return 0
    r = fetch(m, a.dest, only=a.only.split(",") if a.only else None, connections=a.connections)
    print(f"FETCH {'OK' if r['ok'] else 'FAILED'}: {r['bytes_net'] / 1e9:.3f} GB over the network")
    for f in r["failed"]:
        print("FETCH FAIL:", f, file=sys.stderr)
    return 0 if r["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
