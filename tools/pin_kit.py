#!/usr/bin/env python3
"""Pin a kit: usage pin_kit.py KITDIR TARBALL [URL] > pins/kit.json (KITDIR = bin/ + lib/ the tarball was made from: tar -C KITDIR -cf - bin lib | xz -T8 -6 > TARBALL)."""
import hashlib, json, os, sys
kit, tgz = sys.argv[1], sys.argv[2]
def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
files = {}
for root, _, fs in os.walk(kit):
    for f in fs:
        p = os.path.join(root, f)
        if os.path.isfile(p) and not os.path.islink(p):
            files[os.path.relpath(p, kit)] = md5(p)
ident = {"vc_grow_seg_from_seed": files["bin/vc_grow_seg_from_seed"], "vc_tifxyz_selfcross": files["bin/vc_tifxyz_selfcross"], "libvc_tracer.so": files["lib/libvc_tracer.so"], "libvc_core.so": files["lib/libvc_core.so"]}
print(json.dumps({"tarball_name": os.path.basename(tgz), "tarball_sha256": sha(tgz), "tarball_bytes": os.path.getsize(tgz), "url": sys.argv[3] if len(sys.argv) > 3 else "", "identity_md5": ident, "files_md5": dict(sorted(files.items())),
                  "note": "url is intentionally empty: set KIT_URL (HTTPS) to wherever the tarball is hosted (GitHub release asset / S3); the sha256 pin makes the host untrusted-safe. Lib 94be1eae = patched tracer with in-solve self_collision (hard transverse guard); exe b747f765 = fleet exe; selfcross 73800e99 = fleet detector."}, indent=1))
