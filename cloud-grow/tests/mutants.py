#!/usr/bin/env python3
"""Mutation check (D16: a test never seen to fail is a hope). Copies the tree to a temp dir, applies ONE mutant at a time,
runs pytest, and requires it to go RED. Usage: python tests/mutants.py   (exit 1 if any mutant survives)."""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MUTANTS = [
    ("importer D3 check off", "cloud_grow/importer.py", "if n1 < n0:", "if False:"),
    ("importer selfx density check off", "cloud_grow/importer.py", 'elif (info.get("density") or 0.0) > opt.max_selfx_density:', "elif False:"),
    ("importer selfx-unrunnable not fail-closed", "cloud_grow/importer.py", 'if not info.get("ran"):\n            v.refuse(f"selfx census could not run', 'if False:\n            v.refuse(f"selfx census could not run'),
    ("importer md5 mismatch ignored", "cloud_grow/importer.py", "    if bad:\n        v.refuse(", "    if False:\n        v.refuse("),
    ("importer accepts unlisted files", "cloud_grow/importer.py", "    if extra:\n        v.refuse(", "    if False:\n        v.refuse("),
    ("importer tar path traversal allowed (our check AND tarfile's filter off)", "cloud_grow/importer.py",
     'if os.path.isabs(m.name) or n.startswith("..") or ".." in n.split(os.sep):', "if False:",
     'tf.extractall(dest, filter="data") if "filter" in tarfile.TarFile.extractall.__code__.co_varnames else tf.extractall(dest)', "tf.extractall(dest)"),
    ("importer ignores guarded area overclaim", "cloud_grow/importer.py", "if guard_written and rel > opt.area_tol:", "if False:"),
    ("importer ignores unpinned tool", "cloud_grow/importer.py", 'if tool.get("pin_status") != "PINNED" and not opt.accept_unpinned:', "if False:"),
    ("runner not fail-closed on guard exception", "cloud_grow/runner.py", "if pol.selfcross and pol.selfcross_fail_closed and not gg_judged:", "if False:"),
    ("runner newest checkpoint by name", "cloud_grow/runner.py", "k = (os.path.getmtime(mp), d)", "k = (0, d)"),
    ("runner re-grows instead of resuming (D3)", "cloud_grow/runner.py", "    if resume is None and prior:", "    if False:"),
    ("policy typo silently ignored", "cloud_grow/runner.py", "if k.startswith(GG.PREFIX) and k[len(GG.PREFIX):] not in fields_:", "if False:"),
    ("tool gate lets mismatch through", "cloud_grow/tools.py", 'if v["pin_status"] in ("MISSING", "MISMATCH", "NO-PIN")', 'if v["pin_status"] in ("MISSING",)'),
    ("fetch skips md5 verification", "cloud_grow/data_fetch.py", "return (not check_md5) or (not _etag_is_md5(o.etag)) or _md5(path) == o.etag", "return True"),
    ("fetch marks complete despite failures", "cloud_grow/data_fetch.py", "    if failed:\n        return", "    if False:\n        return"),
    ("pack ships secrets", "cloud_grow/pack.py", "    if bad:\n        raise PackError", "    if False:\n        raise PackError"),
    ("manifest ships cache_root", "cloud_grow/manifest.py", 'EXCLUDE_DIRS = {"cache_root"}', "EXCLUDE_DIRS = set()"),
    ("seeding ignores edge margin", "cloud_grow/seeding.py", "if not (EDGE_MARGIN_VOX <= z", "if False and not (EDGE_MARGIN_VOX <= z"),
    ("seeding coverage not dilated by RADIUS_L4", "cloud_grow/seeding.py", "return ndimage.binary_dilation(cov, iterations=RADIUS_L4) if cov.any() else cov", "return cov"),
    ("preflight ram check off", "cloud_grow/preflight.py", "ok = ram_gb is not None and ram_gb >= need_ram", "ok = True"),
    # ---- provision/ (fleet orchestrator, watchdog, deploy)
    ("provision: spend cap not required", "provision/plan.py", 'p["spend_cap_usd"] > 0)', "True)"),
    ("provision: --yes interlock off", "provision/fleet.py", "    if not args.yes:\n        raise FleetError", "    if False:\n        raise FleetError"),
    ("provision: FLEET_EXECUTE interlock off", "provision/fleet.py", 'if os.environ.get("FLEET_EXECUTE") != "1":\n        raise FleetError', "if False:\n        raise FleetError"),
    ("provision: secrets not redacted", "provision/fleet.py", 's = s.replace(v, "<REDACTED>")', "s = s"),
    ("provision: z bands not disjoint", "provision/plan.py", "int(z0 + j * w)", "int(z0)"),
    ("provision: leftover instances not detected", "provision/fleet.py", "if not ad.verify_empty(o):", "if False:"),
    ("provision: bootstrap dead-man switch missing", "provision/bootstrap.sh.tmpl", 'shutdown -P +"$HOURS_MIN"', 'true +"$HOURS_MIN"'),
    ("watchdog: cost tripwire off", "provision/watchdog.py", "if usd >= cap:", "if usd >= cap * 100:"),
    ("watchdog: restarts on tool drift", "provision/watchdog.py", ' and not pb.get("tool_md5_drift"):\n            if row["restarts"]', ':\n            if row["restarts"]'),
    ("watchdog: unlimited restarts", "provision/watchdog.py", 'if row["restarts"] < MAX_RESTARTS:', "if True:"),
    ("watchdog: failure-rate trouble off", "provision/watchdog.py", "failed / tot > FAIL_RATE_TROUBLE:", "False:"),
    ("deploy: bad checksum imported", "provision/deploy.py", "            if got != want:", "            if False:"),
    ("deploy: registers without dry-run PASS", "provision/deploy.py", 'if v.get("status") == "PASS":\n            v = import_fn(tar, scroll, False)', "if True:\n            v = import_fn(tar, scroll, False)"),
    ("deploy: read key == write key allowed", "provision/deploy.py", 'in wk:', "in []:"),
    ("deploy: quota not checked", "provision/deploy.py", "if have < need:", "if False:"),
]


def main() -> int:
    survived = []
    only = sys.argv[1:]
    for name, rel, *edits in MUTANTS:
        if only and not any(o in name for o in only):
            continue
        tmp = tempfile.mkdtemp(prefix="mut_")
        try:
            dst = os.path.join(tmp, "cg")
            shutil.copytree(ROOT, dst, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".git"))
            p = os.path.join(dst, rel)
            s = open(p).read()
            if any(old not in s for old in edits[0::2]):
                print(f"BROKEN MUTANT (pattern not found): {name}")
                survived.append(name)
                continue
            for old, new in zip(edits[0::2], edits[1::2]):
                s = s.replace(old, new, 1)
            open(p, "w").write(s)
            r = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider"], cwd=dst, capture_output=True, text=True)
            red = r.returncode != 0
            print(("RED  (killed)  " if red else "GREEN (SURVIVED) ") + name)
            if not red:
                survived.append(name)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    n = len([m for m in MUTANTS if not only or any(o in m[0] for o in only)])
    print(f"{n - len(survived)}/{n} mutants killed")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
