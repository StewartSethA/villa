"""reader_v2: DomRusso2's Reader v2 ink checkpoint (MIT), run through the ink9um family's OWN code path.

Upstream: github.com/DomRusso2/reader-v2 @ 02c7b469 (2026-09-28); weights huggingface.co/domenicor046/reader-v2 @
72633d8a (reader-v2-step040000.pth = the released model; reader-v2-init-ft12k.pth = its training init). The SAME
architecture and checkpoint format as the team's ink_9um (`vesuvius_unet_3d_stem_2d`, 68,175,426 parameters,
identical state_dict keys and shapes, config embedded), so it is a checkpoint swap and nothing else: predict()
calls `stages.ink.run_ink9um` -> ink_dashboard/run_ink9um.py -> villa's `vesuvius.ink_detection.inference.infer`
with INK9UM_CKPT pointed at this file. Proven 2026-10-05 (scripts/reader_v2/ref_repro.py, PHerc0841 tile,
707,333 valid px): our pinned villa CLI and the author's reference `koine_machines.inference.infer` (villa
merge-ink-pipelines 3ea17f54) with the author's flags give BYTE-IDENTICAL uint8 maps; the init-ft12k
checkpoint through the same check differs (r 0.938), so the check can fail.

INPUT CONTRACT (from the checkpoint's own config, applied by the CLI, not by us): 17 centred planes, 128 px patches,
50 % overlap with a Hann blend, per-PATCH robust-MAD normalisation (1-99 % clip, median / MAD) -- so an affine
intensity difference between scans is normalised away per patch (unlike hecate's raw /255). Trained at native
9.362 um (113 keV), 8.64 um (116 keV), 7.91 um (PHerc0172, 53 keV) and on the 9.6 um pooled ink_9um corpus; the
render here is the ink9um tier, ~9.5 um.

WHAT IT TRAINED ON (train_config.json, read 2026-10-05) -- PHerc0139, PHerc0814, PHerc0500P2, PHerc0009B, PHerc0343P,
PHerc0172 (S5: 4 segments) AND the ink_9um corpus INCLUDING PHerc1667 (S4: wNNN labels, wNNN
pseudo-labels) and PHercParis4 (S1: selected segments). No S1, S4 or S5 number of it is transfer.
Held out entirely: PHerc0841 (but the author used it for go/no-go and checkpoint choice).
"""
from __future__ import annotations

import hashlib
import json
import os

from ... import alerts as _alerts

# NATIVE-ONLY MODE (finish.native_render_enabled, 2026-10-06 user order "into production by default"): the dispatcher hands
# an in-memory view key. predict() then MATERIALISES the CLI's own centred 17 planes of the view (already in the face
# order the dispatcher wants) as TIFFs in a temp dir beside out_png -- never /dev/shm -- runs the same ink9um wrapper on
# that real directory, and deletes it. 17 x H x W uint8 on the work disk; the CLI would have read only these 17 anyway.
MEMSTACK_READY = True
DEPTH = 17

CKPTS = {"": "reader-v2-step040000.pth", "ft12k": "reader-v2-init-ft12k.pth",
         # dense_native (Erwin Nieuwlaar, MIT; huggingface.co/Nieuwlaar/ink9um-dense-native, step 16 000): the SAME 508 tensors (names and shapes identical to
         # reader-v2-step040000.pth, checked 2026-10-09). The public release is a bare safetensors state_dict; the .pth is rebuilt with the author's own recipe
         # (a rebuild script from the author's repository: model + train_dense_native.json + step 16000), md5 below.
         "dense_native": "dense_native-016000.pth"}
MODEL_SUBDIR = "reader_v2"
MODEL_ROOTS = ("var/models", "/dev/shm/vpipe/ScrollPrizeTutorial/current/var/models", "/dev/shm/vpipe/var/models")
ENV_DIR = "VPIPE_READER_V2_DIR"
UPSTREAM = {"repo": "domenicor046/reader-v2", "commit": "72633d8aa6d40e9e737ccea4c9928fbc624dab45",
            "code": "github.com/DomRusso2/reader-v2@02c7b46987813c78f56dd99f38c5a84b65a9cd2a",
            "md5": {"dense_native-016000.pth": "96e6054f69551914342af5c966259b88",
                    "reader-v2-step040000.pth": "7f261ac1b55e9aa19c4ad2fefe22a996",
                    "reader-v2-init-ft12k.pth": "071c11f90f8135f868c7e6b1f0ea89d4"},
            "sha256": {"dense_native-016000.pth": "04ce4dd969f6a4d1daa87fd8675245749ce54b842866df7ba7f339c9c7616cb6",
                       "reader-v2-step040000.pth": "654ec5acec2b6c4788d9cb326e2f0b8c2c730584949a934ef775db7f6ab5d3a6",
                       "reader-v2-init-ft12k.pth": "6bad92971b029857f23d1ab46a623b0f2cd6c9605c05d1939c8f859b0d98f79b"}}
TARGET_UM = 9.5

SPEC = {
    "layers": 17,
    "layer_select": "17 centred planes of the ink9um render (upstream select_layer_indices)",
    "um_per_px": TARGET_UM,
    "normalisation": "per-patch robust MAD (1-99 % clip, median/MAD), from the checkpoint config",
    "arch": "vesuvius_unet_3d_stem_2d (= ink_9um hybrid_3d2d)",
    "patch": [17, 128, 128],
    "stride": 64,
    "blend": "hann",
    "upstream": UPSTREAM,
}


def _roots() -> list[str]:
    out = [os.environ.get(ENV_DIR, "")]
    try:
        from ... import config
        out.append(str(config.repo_root() / "var" / "models" / MODEL_SUBDIR))
    except Exception as e:                               # noqa: BLE001 - no config is not fatal here
        print(f"[reader_v2] repo root unknown ({e}); searching the fixed roots only", flush=True)
    out += [os.path.join(r, MODEL_SUBDIR) for r in MODEL_ROOTS]
    return [r for r in out if r]


def checkpoint_path(arm: str | None = None) -> str | None:
    name = CKPTS.get(arm or "")
    if name is None:
        return None
    for r in _roots():
        p = os.path.join(r, name)
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


def training_frame():
    from ...frame import ModelSpec
    return ModelSpec("reader_v2", TARGET_UM, 17, tile_px=128).frame(source_id="model:reader_v2")


def enabled() -> bool:
    return os.environ.get("VPIPE_READER_V2", "0") == "1"


def available(fleet=None, arm: str | None = None) -> tuple[bool, str]:
    """Are the weights here. NOT whether the family should run."""
    if (arm or "") not in CKPTS:
        return False, f"reader_v2 has no arm {arm!r} (arms: default, {', '.join(a for a in CKPTS if a)})"
    p = checkpoint_path(arm)
    if p is None:
        return False, (f"reader_v2 checkpoint not on this host (var/models/{MODEL_SUBDIR}/{CKPTS[arm or '']} or "
                       f"${ENV_DIR}; `vpipe setup models` fetches [models.reader_v2])")
    return True, p


def centred(n: int, depth: int = DEPTH) -> list[int]:
    """Upstream center_crop_layer_indices: the upper centre for even excess."""
    if n <= depth:
        return list(range(n))
    s = n // 2 - depth // 2
    return list(range(s, s + depth))


def _materialise(view_key: str, work: str) -> tuple[str, float]:
    """The centred 17 planes of an in-memory view as 00.tif..16.tif in a temp dir under `work` (caller deletes)."""
    import tempfile
    import numpy as np
    import tifffile
    from .. import memstack as MS
    names = MS.list_names(view_key)
    tmp = tempfile.mkdtemp(prefix="reader_v2_layers_", dir=work)
    for j, k in enumerate(centred(len(names))):
        tifffile.imwrite(os.path.join(tmp, f"{j:02d}.tif"), np.ascontiguousarray(np.asarray(MS.read_layer(view_key.rstrip("/") + "/" + names[k]), np.uint8)))
    um = MS.um_per_px(view_key)
    with open(os.path.join(tmp, "frame.json"), "w") as fh:
        json.dump({"um_per_px": um, "planes": f"centred {DEPTH} of {len(names)} (view {view_key})"}, fh)
    return tmp, float(um)


_AUDITED: dict = {}


def _audit(path: str, arm: str | None) -> tuple[bool, str]:
    """md5 of the file against the audited upstream release, cached per (path, size, mtime)."""
    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime)
    if key not in _AUDITED:
        h = hashlib.md5()
        with open(path, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 22), b""):
                h.update(b)
        _AUDITED[key] = h.hexdigest()
    want = UPSTREAM["md5"][CKPTS[arm or ""]]
    got = _AUDITED[key]
    return got == want, f"md5 {got} (audited upstream {want})"


def predict(layers_dir: str, mask_png: str | None, out_png: str, gpu: int = 0, arm: str | None = None,
            fleet=None, log: str | None = None, **spec) -> bool:
    """One rendered stack (already in the face the dispatcher wants) -> one probability PNG, through
    stages.ink.run_ink9um with only INK9UM_CKPT swapped."""
    ok, ck = available(fleet, arm)
    if not ok:
        _alerts.alert(f"reader_v2 on this host: {ck}")
        return False
    good, why = _audit(ck, arm)
    if not good:
        _alerts.alert(f"reader_v2: {ck} is not the audited release ({why}); not run")
        return False
    from ... import config
    from .. import ink as INK
    from .. import memstack as MS
    if fleet is None:
        fleet = config.load()
    seg = "reader_v2_" + os.path.basename(os.path.dirname(os.path.abspath(out_png)))
    log = log or os.path.splitext(out_png)[0] + ".log"
    if os.environ.get("VPIPE_READER_V2_PATH", "fast").strip().lower() != "cli":
        return _predict_fast(fleet, layers_dir, out_png, gpu, ck, arm, why, log)
    tmp, src, um = None, layers_dir, None
    try:
        if MS.list_names(layers_dir) is not None:
            tmp, um = _materialise(layers_dir, os.path.dirname(os.path.abspath(out_png)))
            src = tmp
        ok, detail = INK.run_ink9um(fleet, seg, src, out_png, str(gpu), TARGET_UM, "forward", log, ckpt=ck, um_per_px=um)
    finally:
        if tmp:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
    if ok:
        with open(os.path.splitext(out_png)[0] + ".reader_v2.json", "w") as fh:
            json.dump({"model": "reader_v2" + (f":{arm}" if arm else ""), "checkpoint": ck, "audit": why,
                       "spec": SPEC, "detail": detail}, fh, indent=1)
    return bool(ok)


FAST_RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_reader_v2_villa.py")


def _planes(layers_dir: str):
    """(17,H,W) uint8 of the CLI's centred planes, from an in-memory view or a directory of TIFFs, face order as given."""
    import numpy as np
    from .. import memstack as MS
    from .grandprize_dense import _layer_array
    names = MS.list_names(layers_dir)
    if names is not None:
        fs = [layers_dir.rstrip("/") + "/" + n for n in names]
        um = MS.um_per_px(layers_dir)
    else:
        fs = [os.path.join(layers_dir, f) for f in sorted(os.listdir(layers_dir)) if f.lower().endswith((".tif", ".tiff"))]
        from .colmean import render_um_per_px
        um = render_um_per_px(layers_dir, default=float("nan"))
    return np.stack([np.asarray(_layer_array(fs[k]), np.uint8) for k in centred(len(fs))]), um


def _predict_fast(fleet, layers_dir, out_png, gpu, ck, arm, audit, log) -> bool:
    """DEFAULT path (2026-10-06): one villa-interpreter subprocess running _reader_v2_villa.py (the checkpoint's own
    128 px / stride 64 / Hann / per-patch robust-MAD contract, batched on the GPU) on the centred 17 planes. Measured
    against the villa CLI: r 0.999996-0.999997, max 1 LSB (scripts/reader_v2/fast_check.py); 2-4x less wall per face,
    the CLI's ~20 s DataLoader start and ~100 patch/s host loop gone. VPIPE_READER_V2_PATH=cli restores the CLI path."""
    import shutil
    import tempfile
    import time
    import numpy as np
    from .. import ink as INK
    work = os.path.dirname(os.path.abspath(out_png))
    tmp = tempfile.mkdtemp(prefix="reader_v2_fast_", dir=work)       # the work disk, never /dev/shm (D18)
    try:
        t0 = time.time()
        x, um = _planes(layers_dir)
        if um and np.isfinite(um) and abs(um / TARGET_UM - 1) > 0.05:
            _alerts.alert(f"reader_v2: {layers_dir} is {um:g} um/px, the checkpoint reads {TARGET_UM} (+-5 %); not run")
            return False
        xp = os.path.join(tmp, "x17.npy")
        np.save(xp, x)
        env = INK.wrapper_env(fleet, str(gpu))
        cmd = [env["VILLA_PY"], FAST_RUNNER, xp, ck, out_png]
        rc = INK._run(cmd, log or os.path.splitext(out_png)[0] + ".log", 3600, env=env)
        ok = rc == 0 and os.path.exists(out_png) and os.path.getsize(out_png) > 0
        if ok:
            st = {}
            try:
                st = json.load(open(os.path.splitext(out_png)[0] + ".fast.json"))
            except (OSError, ValueError):
                pass
            st["planes_s"] = round(time.time() - t0, 2)
            with open(os.path.splitext(out_png)[0] + ".reader_v2.json", "w") as fh:
                json.dump({"model": "reader_v2" + (f":{arm}" if arm else ""), "checkpoint": ck, "audit": audit, "path": "fast",
                           "um_per_px": um, "spec": SPEC, "stats": st}, fh, indent=1)
        else:
            _alerts.alert(f"reader_v2 fast path rc={rc} on {layers_dir}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
