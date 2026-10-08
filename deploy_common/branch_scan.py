#!/usr/bin/env python3
"""Branch hygiene scan: secrets/tokens/keys, private hosts and paths, oversized files.  Exit 1 on any hit.

usage: branch_scan.py DIR [--max-mb 50] [--list] [--allow FILE_REGEX ...]
Scans every file under DIR (skipping .git).  Text files are grepped line by line; binary files are only
size-checked.  A hit prints  path:line: rule: <the matching text, truncated>.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

RULES = [
    ("private-ip-192.168", r"\b192\.168\.\d{1,3}\.\d{1,3}\b"),
    ("private-ip-10.", r"(?<![\d.])10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    ("private-ip-172.16-31", r"\b172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b"),
    ("home-path", r"/home/(seth|jacob|ubuntu|user)\b"),
    ("fleet-path", r"/mnt/(raid7|bpxp|raid10T|4TB|8TB)\b|/media/seth|24TB-12SSD|pny-shelf|sas-r0"),
    ("fleet-host", r"\b(seth|jacob)@[\w.-]+|\bv100\.local\b|\blifestar\b|\bpny\b\s*[:=/]|hub\.token"),
    ("private-key", r"-----BEGIN (RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY"),
    ("aws-key", r"\bAKIA[0-9A-Z]{16}\b|aws_secret_access_key\s*[=:]"),
    ("github-token", r"\bgh[pousr]_[A-Za-z0-9]{30,}\b|github_pat_[A-Za-z0-9_]{30,}"),
    ("hf-token", r"\bhf_[A-Za-z0-9]{30,}\b"),
    ("generic-secret", r"(?i)\b(api[_-]?key|secret|passwd|password|token)\b\s*[=:]\s*['\"][A-Za-z0-9/+_\-]{16,}['\"]"),
    ("wandb-key", r"WANDB_API_KEY\s*=\s*\S{20,}"),
    ("email", r"[\w.+-]+@(yahoo|gmail|hotmail|outlook)\.com"),
]
RX = [(n, re.compile(p)) for n, p in RULES]


def is_text(b: bytes) -> bool:
    return b"\0" not in b[:4096]


def scan(root: Path, max_mb=50.0, allow=(), skip_rules=()):
    hits, files = [], []
    allow = [re.compile(a) for a in allow]
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d != ".git"]
        for f in fn:
            p = Path(dp) / f
            rel = str(p.relative_to(root))
            try:
                sz = p.stat().st_size if not p.is_symlink() else 0
            except OSError:
                continue
            files.append((rel, sz))
            if sz > max_mb * 1e6:
                hits.append((rel, 0, "oversized", f"{sz / 1e6:.1f} MB > {max_mb} MB"))
            if p.is_symlink():
                tgt = os.readlink(p)
                if os.path.isabs(tgt):
                    hits.append((rel, 0, "absolute-symlink", tgt))
                continue
            if any(a.search(rel) for a in allow) or sz > 20e6 or rel.endswith("branch_scan.py"):   # the rule table itself names the patterns
                continue
            try:
                raw = p.read_bytes()
            except OSError:
                continue
            if not is_text(raw):
                continue
            for i, line in enumerate(raw.decode("utf-8", "replace").splitlines(), 1):
                for n, rx in RX:
                    if n in skip_rules:
                        continue
                    m = rx.search(line)
                    if m and n == "private-ip-10." and re.search(r"==|version\s*=|cu1\d|pythonhosted|\d\.\d+\.\d+\.\d+-", line):
                        continue                         # a package version (10.3.7.77), not an address
                    if m:
                        hits.append((rel, i, n, line.strip()[:140]))
    return hits, files


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--max-mb", type=float, default=50.0)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--allow", nargs="*", default=[], help="file-path regexes exempt from the text rules (size still checked)")
    ap.add_argument("--skip-rule", nargs="*", default=[])
    a = ap.parse_args(argv)
    hits, files = scan(Path(a.dir), a.max_mb, a.allow, a.skip_rule)
    tot = sum(s for _, s in files)
    if a.list:
        for r, s in sorted(files, key=lambda x: -x[1]):
            print(f"{s:>12d}  {r}")
    print(f"scanned {len(files)} files, {tot / 1e6:.2f} MB total; largest {max((s for _, s in files), default=0) / 1e6:.2f} MB")
    for r, i, n, t in hits:
        print(f"HIT {r}:{i}: {n}: {t}")
    print(f"BRANCH SCAN {'FAILED' if hits else 'CLEAN'}: {len(hits)} hit(s)")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
