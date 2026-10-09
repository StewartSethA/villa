"""torch.compile for dense-model inference, implemented ONCE (user order 2026-10-08: compile all models' inference by default).

`maybe_compile(model, key, enabled, dev, in_ch)` -> the model to call.  Compiled once per process per `key` (checkpoint path + mtime + device) and reused; the first use runs a
NUMERICS SELF-CHECK (max |logit diff| compiled vs eager on a fixed random batch, tolerance TOL = 0.05 logits for fp16 models) and falls back to EAGER with an ANNOUNCED alert when
torch is too old (< 2.1), compile raises, or the numerics disagree.  Dynamic shapes: `dynamic=False` plus fixed-shape tiling (reader_v2_dense.predict_array pads every tile to the full tile+2*halo size when the model is compiled; automatic-dynamic compilation was measured 2x SLOWER than eager on real mixed-shape tiles).  The Inductor FX cache persists across processes, so the 12-24 s compile is one-off per host.
Measured (V100, 1024^2 tile, fp16): 539 k student 144 -> 67 ms (2.1x), +3-D stem 426 -> 201 ms, max logit diff 0.0000 (docs/experiments/sharp_fn_tolerant_2026-10-07).
"""
from __future__ import annotations

TOL = 0.05
MIN_SPEEDUP = 1.15
BUCKETS = (512, 1024, 1536, 2048, 2304)          # tile-input side buckets: a compiled model sees at most len(BUCKETS)^2 shapes (recompiles bounded; cache_size_limit 32)
_CACHE: dict = {}


def _note(msg: str) -> None:
    """informational (not an operator alert): auto mode choosing eager is the normal outcome until a benchmark exists"""
    print("NOTE", msg, flush=True)


def bucket(n: int, mult: int = 4) -> int:
    for b in BUCKETS:
        if n <= b:
            return max(b, ((n + mult - 1) // mult) * mult)
    return ((n + mult - 1) // mult) * mult


def sidecar(ck: str) -> str:
    return ck + ".compile.json"


def decision(ck: str, n_shift: int, side: int, mode="auto") -> bool:
    """Should this call run compiled?  mode: '0'/False -> eager; '1'/True -> compiled; 'auto' -> the MEASURED benchmark beside the checkpoint (bench_compile.py): compiled iff an entry for
    (shift_tta N, input-side bucket >= side) shows speedup >= MIN_SPEEDUP end to end with border-aware numerics within TOL; no benchmark on file -> eager (announced once)."""
    m = str(mode).lower()
    if m in ("0", "false", "off"):
        return False
    if m in ("1", "true", "on"):
        return True
    import json, os
    f = sidecar(ck)
    try:
        d = json.load(open(f))
    except Exception:
        if ("nobench", ck) not in _CACHE:
            _CACHE[("nobench", ck)] = 1
            _note(f"torch.compile auto: no benchmark {os.path.basename(f)} for {os.path.basename(ck)}: eager (run scripts/bench_compile.py)")
        return False
    if d.get("provisional"):
        if ("prov", ck) not in _CACHE:
            _CACHE[("prov", ck)] = 1
            _note(f"torch.compile auto: {os.path.basename(f)} was measured on a busy GPU (provisional): eager until re-benchmarked on an idle card")
        return False
    b = bucket(side)
    cands = [e for e in d.get("entries", []) if int(e["N"]) == int(n_shift) and int(e["bucket"]) >= b]
    if not cands:
        return False
    e = min(cands, key=lambda e: int(e["bucket"]))
    return bool(e.get("speedup", 0) >= MIN_SPEEDUP and e.get("numerics_ok", False))


def _alert(msg: str) -> None:
    try:
        from ... import alerts as _alerts
        _alerts.alert(msg)
    except Exception:
        print("ALERT", msg, flush=True)


def maybe_compile(model, key, enabled: bool, dev: str, in_ch: int):
    if not enabled or not str(dev).startswith("cuda"):
        return model
    if key in _CACHE:
        return _CACHE[key]
    try:
        import torch
        ver = tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2])
        if ver < (2, 1) or not hasattr(torch, "compile"):
            _alert(f"torch.compile unavailable on this host (torch {torch.__version__} < 2.1): eager inference for {key}")
            _CACHE[key] = model
            return model
        try:
            import torch._dynamo as _d
            _d.config.cache_size_limit = max(32, int(getattr(_d.config, "cache_size_limit", 8)))
        except Exception:
            pass
        cm = torch.compile(model, dynamic=False)      # static shapes: predict_array pads every tile to ONE shape when the model is compiled (dynamic-shape kernels measured 2x SLOWER than eager)
        p = next(model.parameters())
        g = torch.Generator(device="cpu").manual_seed(0)
        x = torch.randn(1, in_ch, 192, 192, generator=g).to(p.device, p.dtype).contiguous(memory_format=torch.channels_last)
        with torch.no_grad():
            ye = model(x).float()
            yc = cm(x).float()
        d = float((ye - yc).abs().max())
        if not (d == d) or d > TOL:
            _alert(f"torch.compile numerics mismatch for {key}: max |logit diff| {d:.4f} > {TOL}: eager inference")
            _CACHE[key] = model
            return model
        _CACHE[key] = cm
        return cm
    except Exception as e:
        _alert(f"torch.compile failed for {key} ({type(e).__name__}: {str(e)[:200]}): eager inference")
        _CACHE[key] = model
        return model
