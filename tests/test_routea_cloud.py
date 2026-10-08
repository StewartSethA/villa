"""routeA-cloud-deploy tests: offline, synthetic; a fake S3/HTTP server, a synthetic zarr, a synthetic kit tarball. Tests that need the real vc_tifxyz_selfcross skip without it."""
from __future__ import annotations

import hashlib
import http.server
import io
import json
import os
import re
import tarfile
import threading
import urllib.parse
from pathlib import Path

import numpy as np
import pytest

from vesuvius_pipeline.routea_cloud import bootstrap, net, seedprop, settings

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]


# ---- fake S3 / HTTP server ----------------------------------------------------------------------------------------------------------------
class _Server:
    def __init__(self, objects: dict[str, bytes], bucket="b", corrupt: set | None = None):
        outer = self
        self.objects, self.bucket, self.corrupt = objects, bucket, corrupt or set()

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                u = urllib.parse.urlparse(self.path)
                q = urllib.parse.parse_qs(u.query)
                if "list-type" in q:                                        # ListObjectsV2 (no pagination needed in tests)
                    pre = q.get("prefix", [""])[0]
                    xml = "".join(f"<Contents><Key>{k}</Key><Size>{len(v)}</Size><ETag>\"{hashlib.md5(v).hexdigest()}\"</ETag></Contents>" for k, v in sorted(outer.objects.items()) if k.startswith(pre))
                    body = f"<?xml version='1.0'?><ListBucketResult xmlns='http://s3.amazonaws.com/doc/2006-03-01/'><IsTruncated>false</IsTruncated>{xml}</ListBucketResult>".encode()
                    self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
                key = urllib.parse.unquote(u.path.split("/", 2)[2]) if u.path.count("/") >= 2 else urllib.parse.unquote(u.path.lstrip("/"))
                if key not in outer.objects:
                    self.send_response(404); self.end_headers(); return
                data = outer.objects[key]
                if key in outer.corrupt:
                    data = b"X" * len(data)
                rng = self.headers.get("Range")
                if rng and rng.startswith("bytes="):
                    start = int(rng[6:].split("-")[0])
                    if start >= len(data):
                        self.send_response(416); self.end_headers(); return
                    self.send_response(206); self.send_header("Content-Length", str(len(data) - start)); self.end_headers(); self.wfile.write(data[start:]); return
                self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def s3(monkeypatch):
    srv = _Server({"sc/p/0/.zarray": b'{"shape":[10,10,10]}', "sc/p/0/0/0/0": b"abc" * 100, "sc/p/.zattrs": b"{}"})
    monkeypatch.setenv(net.S3_ENDPOINT_ENV, srv.url)
    yield srv
    srv.close()


def test_s3_sync_is_verified_idempotent_and_fails_loud_on_corruption(tmp_path, s3):
    st = net.s3_sync("b", "sc/p", tmp_path / "d")
    assert st["objects"] == 3 and (tmp_path / "d" / "0" / "0" / "0" / "0").read_bytes() == b"abc" * 100
    assert net.s3_sync("b", "sc/p", tmp_path / "d")["downloaded_objects"] == 0                    # idempotent: nothing re-fetched
    s3.corrupt.add("sc/p/0/0/0/0")
    with pytest.raises(net.FetchError):                                                          # md5 != ETag: never silently accepted
        net.s3_sync("b", "sc/p", tmp_path / "bad")
    with pytest.raises(net.FetchError):
        net.s3_sync("b", "sc/none", tmp_path / "none")                                           # nothing under the prefix = error


def test_download_verifies_sha256_and_resumes(tmp_path, s3):
    data = s3.objects["sc/p/0/0/0/0"]
    ok = net.download(s3.url + "/b/sc/p/0/0/0/0", tmp_path / "f", sha256=hashlib.sha256(data).hexdigest())
    assert ok.read_bytes() == data
    part = tmp_path / "g.part"; part.write_bytes(data[:100])                                      # a partial file is continued with a Range request
    assert net.download(s3.url + "/b/sc/p/0/0/0/0", tmp_path / "g", sha256=hashlib.sha256(data).hexdigest()).read_bytes() == data
    with pytest.raises(net.FetchError):
        net.download(s3.url + "/b/sc/p/0/0/0/0", tmp_path / "h", sha256="0" * 64)
    assert not (tmp_path / "h").exists()


# ---- kit ------------------------------------------------------------------------------------------------------------------------------------
def _fake_kit(tmp_path, selfcross_ok=True):
    kd = tmp_path / "kitsrc"; (kd / "bin").mkdir(parents=True); (kd / "lib").mkdir()
    sc = "#!/bin/sh\necho 'Census a tifxyz surface for non-adjacent transverse self-intersections.'\n" if selfcross_ok else "#!/bin/sh\nexit 3\n"
    (kd / "bin" / "vc_tifxyz_selfcross").write_text(sc); (kd / "bin" / "vc_grow_seg_from_seed").write_text("#!/bin/sh\necho help\n")
    for e in ("vc_tifxyz_selfcross", "vc_grow_seg_from_seed"):
        os.chmod(kd / "bin" / e, 0o755)
    (kd / "lib" / "libvc_tracer.so").write_bytes(b"tracer"); (kd / "lib" / "libvc_core.so").write_bytes(b"core")
    tgz = tmp_path / "kit.tar.xz"
    with tarfile.open(tgz, "w:xz") as t:
        t.add(kd / "bin", "bin"); t.add(kd / "lib", "lib")
    def md5(p): return hashlib.md5(Path(p).read_bytes()).hexdigest()
    files = {str(p.relative_to(kd)): md5(p) for p in kd.rglob("*") if p.is_file()}
    pin = {"tarball_name": "kit.tar.xz", "tarball_sha256": hashlib.sha256(tgz.read_bytes()).hexdigest(), "url": "", "identity_md5": {}, "files_md5": files}
    root = tmp_path / "root"; (root / "pins").mkdir(parents=True); (root / "pins" / "kit.json").write_text(json.dumps(pin))
    return tgz, pin, root


def test_kit_is_sha256_pinned_md5_verified_and_fails_loud_without_a_url(tmp_path, monkeypatch):
    tgz, pin, root = _fake_kit(tmp_path)
    monkeypatch.delenv(bootstrap.KIT_URL_ENV, raising=False)
    with pytest.raises(bootstrap.BootstrapError, match="KIT_URL"):
        bootstrap.ensure_kit(tmp_path / "w1", root=root)                                         # no URL = a loud error, not a fallback
    srv = _Server({"kit.tar.xz": tgz.read_bytes()}, bucket="")
    try:
        r = bootstrap.ensure_kit(tmp_path / "w2", srv.url + "/kit.tar.xz", root=root, log=lambda *_: None)
        assert r["downloaded"] and (tmp_path / "w2" / "kit" / "bin" / "vc_tifxyz_selfcross").exists()
        assert bootstrap.ensure_kit(tmp_path / "w2", srv.url + "/kit.tar.xz", root=root)["downloaded"] is False     # verified kit reused offline
        bad = json.loads((root / "pins" / "kit.json").read_text()); bad["tarball_sha256"] = "0" * 64
        (root / "pins" / "kit.json").write_text(json.dumps(bad))
        with pytest.raises(net.FetchError, match="sha256"):
            bootstrap.ensure_kit(tmp_path / "w3", srv.url + "/kit.tar.xz", root=root)
    finally:
        srv.close()


def test_kit_selfcheck_failure_is_loud(tmp_path):
    _tgz, _pin, _root = _fake_kit(tmp_path, selfcross_ok=False)
    with pytest.raises(bootstrap.BootstrapError, match="self-check"):
        bootstrap.selfcheck_kit(tmp_path / "kitsrc")


def test_unknown_scroll_and_missing_voxel_refuse_instead_of_guessing(tmp_path):
    root = tmp_path / "root"; (root / "pins").mkdir(parents=True)
    (root / "pins" / "scrolls.json").write_text(json.dumps({"bucket": "b", "scrolls": {"X": {"prediction": "p.zarr", "grids": "g.normal-grids", "voxel_um": None}}}))
    with pytest.raises(bootstrap.BootstrapError, match="not in pins"):
        bootstrap.ensure_scroll(tmp_path / "w", "Y", root=root)
    with pytest.raises(bootstrap.BootstrapError, match="voxel_um"):
        bootstrap.ensure_scroll(tmp_path / "w", "X", root=root)


# ---- seeds ----------------------------------------------------------------------------------------------------------------------------------
def test_seed_proposer_is_deterministic_separated_and_on_the_surface(tmp_path, monkeypatch):
    monkeypatch.setattr(seedprop, "EDGE_MARGIN_VOX", 5)
    import zarr
    shape = (120, 120, 120)
    a = np.zeros(shape, np.uint8)
    a[:, 20:100, 58:62] = 255; a[:, 58:62, 20:100] = 255                                          # two thick sheets
    store = zarr.open(str(tmp_path / "pred.zarr"), mode="w")
    store.create_dataset("0", data=a, chunks=(40, 40, 40))
    seeds = seedprop.propose(tmp_path / "pred.zarr", 3, rng_seed=5, min_sep=15.0, log=lambda *_: None)
    again = seedprop.propose(tmp_path / "pred.zarr", 3, rng_seed=5, min_sep=15.0, log=lambda *_: None)
    assert seeds == again and len(seeds) == 3
    for s in seeds:
        assert a[s["z"], s["y"], s["x"]] == 255 or a[max(0, s["z"] - 3):s["z"] + 4, s["y"] - 3:s["y"] + 4, s["x"] - 3:s["x"] + 4].mean() > seedprop.SEED_WINDOW_MEAN
    for i in range(3):
        for j in range(i):
            d = np.linalg.norm([seeds[i][k] - seeds[j][k] for k in "xyz"])
            assert d >= 15.0


# ---- pins / settings / package hygiene ------------------------------------------------------------------------------------------------------
def test_pins_are_consistent():
    kit = json.loads((ROOT / "pins" / "kit.json").read_text())
    assert len(kit["tarball_sha256"]) == 64 and kit["identity_md5"]["libvc_tracer.so"].startswith("94be1eae") and kit["identity_md5"]["vc_tifxyz_selfcross"].startswith("73800e99")
    assert kit["files_md5"]["bin/vc_grow_seg_from_seed"] == kit["identity_md5"]["vc_grow_seg_from_seed"] and kit["url"].startswith("https://")      # public release URL (override with KIT_URL)
    sc = json.loads((ROOT / "pins" / "scrolls.json").read_text())
    assert sc["bucket"] == "vesuvius-challenge-open-data" and "PHerc0332" in sc["scrolls"] and "PHerc0211" in sc["scrolls"] and sc["scrolls"]["PHerc0332"]["voxel_um"] == 9.596
    lock = (ROOT / "pins" / "requirements.lock").read_text()
    assert "--hash=sha256:" in lock and re.search(r"^numpy==1\.26\.4", lock, re.M)
    assert (ROOT / "umbilicus" / "PHerc0332" / "umbilicus.json").is_file()


def test_settings_snapshot_builds_the_production_policy():
    pol = settings.policy("/kit", threads=2, root=ROOT)
    assert pol.selfcross and pol.selfcross_scrub and pol.segment_abort_as_flag and pol.stop_policy == "legacy" and pol.selfcross_bin == "/kit/bin/vc_tifxyz_selfcross"
    assert settings.self_collision_on(ROOT) is True and settings.gate_override(ROOT)["GATE_TRANSVERSE_CELLS"] == 0


def test_the_package_contains_no_hub_secrets_hosts_or_big_files():
    bad = re.compile(r"(192\.168\.|ssh-rsa|ssh-ed25519|BEGIN [A-Z ]*PRIVATE|hub\.token|/home/(seth|jacob|phi1)|/mnt/(raid|4T|bpxp)|password)", re.I)
    for p in (q for d in ("routeA_run.sh", "pins", "src", "umbilicus", "tests", "patches", "tools", "pytest.ini") for q in ((ROOT / d).rglob("*") if (ROOT / d).is_dir() else [ROOT / d])):
        if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts and p.name != "test_routea_cloud.py":
            assert p.stat().st_size < 5_000_000, p
            if p.suffix in (".py", ".sh", ".md", ".json", ".txt", ".lock", ".env", ".diff", ".in", ".ini"):
                m = bad.search(p.read_text(errors="ignore"))
                assert not m, f"{p}: {m.group(0)}"


def test_driver_never_imports_hub_modules():
    src = "\n".join(p.read_text() for p in (ROOT / "src" / "vesuvius_pipeline" / "routea_cloud").glob("*.py"))
    for forbidden in ("remotedb", "hub_token", "pipeline_db", "pipeline_setting", "sqlite3", "ssh ", "connect_for"):
        assert forbidden not in src.replace("# ", ""), forbidden
