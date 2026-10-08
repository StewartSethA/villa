"""Stand-in for `routeB.box8 job`: behaviour scripted by env STUB_PLAN = {"<tag>": ["multinomial"|"oom"|"stall"|"ok", ... per attempt]}.  Writes the same
artifacts a real job leaves (tiled/manifest.json + a tile, fit/out/.../meshes, markers)."""
import argparse, json, os, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--job-json"); ap.add_argument("--gpu"); ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
sp = json.loads(Path(a.job_json).read_text()); j = sp["job"]
H = Path(os.environ["ROUTEB_HOME"]); tag = j["tag"]
cnt = H / "stubcount"; cnt.mkdir(exist_ok=True)
cf = cnt / (j["scroll"] + "__" + tag); n = int(cf.read_text()) + 1 if cf.exists() else 1; cf.write_text(str(n))
plan = json.loads(os.environ.get("STUB_PLAN", "{}")).get(tag, [])
beh = plan[n - 1] if n - 1 < len(plan) else "ok"
with open(H / "stub_events.log", "a") as f:
    f.write(f"{time.time()} start {j['id']} gpu={a.gpu} attempt={n} beh={beh} ov={json.dumps(sp['overrides'])}\n")
time.sleep(float(os.environ.get("STUB_SLEEP", "1.0")))
rd = H / "runs" / j["scroll"] / tag
if beh == "loaded20m":
    fl = rd / "fit"; fl.mkdir(parents=True, exist_ok=True)
    (fl / "fit.log").write_text("loaded 20,000,000 tracks within z-roi [4500, 17500)\n"); time.sleep(600)
if beh == "multinomial":
    print("RuntimeError: number of categories cannot exceed 2^24"); sys.exit(1)
if beh == "oom":
    print("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 GiB"); sys.exit(1)
if beh == "stall":
    print("PROGRESS Optimizing — 5/100 iterations (5.0%) — 1.0 it/s", flush=True); time.sleep(600)
if beh == "env":
    print("ModuleNotFoundError: No module named 'torch'"); sys.exit(1)
(rd / "tiled" / "tiles" / f"{j['scroll']}_{tag}_w001_z00x0").mkdir(parents=True, exist_ok=True)
(rd / "tiled" / "tiles" / f"{j['scroll']}_{tag}_w001_z00x0" / "x.tif").write_bytes(os.urandom(2048))
(rd / "tiled" / "manifest.json").write_text(json.dumps({"n_tiles": 1, "n_windings": 1, "tiles": [{"seg": f"{j['scroll']}_{tag}_w001_z00x0"}]}))
(rd / "fit" / "out" / "m" / "meshes" / "w001").mkdir(parents=True, exist_ok=True)
(rd / "fit" / "out" / "m" / "meshes" / "w001" / "x.tif").write_bytes(os.urandom(4096))
(rd / "fit" / "out" / "m" / "big.ckpt").write_bytes(b"CKPT" * 100)
(rd / "fit" / "out" / "m" / "satisfaction_metrics_fitted.json").write_text("{}")
(rd / ".done.tiles.json").write_text("{}")
print("JOBOK", j["id"])
