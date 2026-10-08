"""GPU kernel smoke test: step 2 of a box8 run (after the link check, after the env is built, BEFORE any fetch or fit).

For every allowed GPU, in its own subprocess with CUDA_VISIBLE_DEVICES=<i>:  real torch matmuls (fp32 / fp16 / bf16, checked against the CPU), an index/multinomial
gather, a 3-D grid_sample, a Triton kernel compile+launch (the fit's flow/gap integrators are Triton), a torch.compile (inductor) round trip (the 6-9 min first-step
compile of every fit), and the vc_spiral native extension import.  Prints one row per GPU: name, compute capability, torch / CUDA build, arch list, OK or FAIL (the failing
test and a plain-language reason), and exits 6 if any GPU fails: a fit on a card whose kernels are missing would die hours later.
  python -m routeB.gpusmoke [--gpus 0,1,2] [--timeout 420] [--skip-compile]
A 'no kernel image' error means this torch build has no kernels for the card's architecture (RTX 5090 = Blackwell sm_120 needs the cu128/cu129 builds:
--torch-cuda cu129, selected automatically when the compute capability is >= 12.0).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor


def child(skip_compile: bool = False) -> dict:
    import torch
    info = {"torch": torch.__version__, "cuda": torch.version.cuda, "available": torch.cuda.is_available(), "tests": []}
    if not info["available"]:
        info["tests"].append({"name": "cuda_available", "ok": False, "err": "torch.cuda.is_available() is False (driver / CUDA mismatch?)"})
        return info
    p = torch.cuda.get_device_properties(0)
    info.update(device=p.name, cc=f"{p.major}.{p.minor}", mem_gib=round(p.total_memory / 2 ** 30, 2), arch_list=torch.cuda.get_arch_list())

    def run(name, fn):
        t = time.time()
        try:
            fn()
            info["tests"].append({"name": name, "ok": True, "s": round(time.time() - t, 1)})
        except Exception as e:                      # noqa: BLE001 - the error text is the result
            info["tests"].append({"name": name, "ok": False, "err": f"{type(e).__name__}: {e}"[:400], "s": round(time.time() - t, 1)})

    def matmul(dt, tol):
        a = torch.randn(512, 512, device="cuda", dtype=dt)
        b = torch.randn(512, 512, device="cuda", dtype=dt)
        c = (a @ b).float().cpu()
        torch.cuda.synchronize()
        ref = a.float().cpu() @ b.float().cpu()
        assert torch.allclose(c, ref, rtol=tol, atol=tol * 20), f"matmul {dt} differs from the CPU by {(c - ref).abs().max().item():.3g}"

    def gather():
        x = torch.rand(1 << 20, device="cuda")
        idx = torch.multinomial(x, 1000)
        y = x[idx].sum().item()
        assert y == y

    def grid():
        import torch.nn.functional as F
        v = torch.randn(1, 1, 16, 16, 16, device="cuda")
        g = torch.rand(1, 4, 4, 4, 3, device="cuda") * 2 - 1
        assert F.grid_sample(v, g, align_corners=True).isfinite().all().item()

    def triton_k():
        import triton
        import triton.language as tl

        @triton.jit
        def add(x, y, o, n, B: tl.constexpr):
            i = tl.program_id(0) * B + tl.arange(0, B)
            m = i < n
            tl.store(o + i, tl.load(x + i, mask=m) + tl.load(y + i, mask=m), mask=m)
        x = torch.arange(1000, device="cuda", dtype=torch.float32)
        o = torch.empty_like(x)
        add[(4,)](x, x, o, 1000, B=256)
        torch.cuda.synchronize()
        assert torch.equal(o, x * 2)

    def compiled():
        f = torch.compile(lambda t: (t * 2 + 1).sin().sum())
        assert f(torch.randn(256, device="cuda")).isfinite().item()

    def vc():
        import importlib
        m = importlib.import_module("vc_spiral.spiral_sampling")
        assert hasattr(m, "PatchSatisfactionAtlas"), "PatchSatisfactionAtlas missing"

    def speed():
        a = torch.randn(4096, 4096, device="cuda")
        b = torch.randn(4096, 4096, device="cuda")

        def tfl(tf32, n=10):
            torch.backends.cuda.matmul.allow_tf32 = tf32
            torch.cuda.synchronize()
            a @ b
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(n):
                a @ b
            torch.cuda.synchronize()
            return 2 * 4096 ** 3 * n / (time.time() - t) / 1e12
        info["fp32_tflops"] = round(tfl(False), 1)
        info["tf32_tflops"] = round(tfl(True), 1)
        torch.backends.cuda.matmul.allow_tf32 = False
        x = torch.empty(1 << 27, device="cuda")
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(10):
            x.clone()
        torch.cuda.synchronize()
        info["copy_GBps"] = round(10 * 2 * x.numel() * 4 / (time.time() - t) / 1e9)
        free, total = torch.cuda.mem_get_info()
        info["vram_free_gib"], info["vram_total_gib"] = round(free / 2 ** 30, 1), round(total / 2 ** 30, 1)

    run("matmul_fp32", lambda: matmul(torch.float32, 2e-2))
    run("speed", speed)
    run("matmul_fp16", lambda: matmul(torch.float16, 5e-2))
    run("matmul_bf16", lambda: matmul(torch.bfloat16, 1e-1))
    run("gather_multinomial", gather)
    run("grid_sample_3d", grid)
    run("triton_kernel", triton_k)
    if not skip_compile:
        run("torch_compile", compiled)
    run("vc_spiral_import", vc)
    return info


def spawn(gpu: str, timeout: float, skip_compile: bool) -> dict:
    cmd = [sys.executable, "-m", "routeB.gpusmoke", "--child"] + (["--skip-compile"] if skip_compile else [])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=os.pathsep.join([str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), os.environ.get("PYTHONPATH", "")]))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"gpu": gpu, "tests": [{"name": "timeout", "ok": False, "err": f"no answer within {timeout:.0f} s (hung compile or driver?)"}]}
    for line in r.stdout.splitlines():
        if line.startswith("GPUSMOKE_JSON "):
            d = json.loads(line[len("GPUSMOKE_JSON "):])
            d["gpu"] = gpu
            return d
    return {"gpu": gpu, "tests": [{"name": "child", "ok": False, "err": (r.stderr or r.stdout)[-400:] or f"exit {r.returncode}"}]}


def reason(d: dict) -> str:
    bad = [t for t in d["tests"] if not t["ok"]]
    if not bad:
        return ""
    t = bad[0]
    err = t.get("err", "")
    cc = d.get("cc", "?")
    sm = "sm_" + cc.replace(".", "") if cc != "?" else "sm_?"
    if "no kernel image" in err or "no kernel" in err.lower():
        blackwell = " (Blackwell)" if cc != "?" and float(cc) >= 12.0 else ""
        return (f"no kernel image for {sm}: this torch build ({d.get('torch')}, arch list {d.get('arch_list')}) does not support {d.get('device', 'this GPU')}{blackwell}. "
                f"Use --torch-cuda cu129 (or cu128).")
    return f"{t['name']} failed: {err}"


def table(results: list[dict]) -> str:
    L = ["GPU SMOKE TEST (real kernels per allowed GPU, before any fetch or fit)"]
    for d in results:
        ok = all(t["ok"] for t in d["tests"]) and d["tests"]
        head = f"  GPU {d['gpu']}: {d.get('device', '?')}  cc {d.get('cc', '?')}  {d.get('mem_gib', '?')} GiB  torch {d.get('torch', '?')}  cuda {d.get('cuda', '?')}"
        L.append(f"{head}   {'OK' if ok else 'FAIL'}   ({sum(t.get('s', 0) for t in d['tests']):.0f} s)")
        L.append(f"      arch list: {' '.join(d.get('arch_list', [])) or '?'}")
        if d.get("fp32_tflops") is not None:
            L.append(f"      speed: fp32 {d['fp32_tflops']} TFLOPs, tf32 {d.get('tf32_tflops')} TFLOPs, copy {d.get('copy_GBps')} GB/s, VRAM free {d.get('vram_free_gib')}/{d.get('vram_total_gib')} GiB "
                     f"(V100 fp32 ref ~15.7: x{d['fp32_tflops'] / 15.7:.1f}; a matmul proxy, NOT a fit-speed measurement)")
        if ok:
            L.append("      passed: " + ", ".join(t["name"] for t in d["tests"]))
        else:
            L.append("      !!! " + reason(d))
    return "\n".join(L)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="routeB.gpusmoke")
    ap.add_argument("--gpus", default=None)
    ap.add_argument("--timeout", type=float, default=420.0)
    ap.add_argument("--skip-compile", action="store_true")
    ap.add_argument("--child", action="store_true")
    a = ap.parse_args(argv)
    if a.child:
        print("GPUSMOKE_JSON " + json.dumps(child(a.skip_compile)), flush=True)
        return 0
    if a.gpus:
        gpus = [g for g in a.gpus.split(",") if g]
    else:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True).stdout
        gpus = [x.strip() for x in out.splitlines() if x.strip()]
    if not gpus:
        print("GPU SMOKE TEST FAILED: no GPU found", file=sys.stderr)
        return 6
    with ThreadPoolExecutor(len(gpus)) as ex:
        res = list(ex.map(lambda g: spawn(g, a.timeout, a.skip_compile), gpus))
    print(table(res), flush=True)
    home = os.environ.get("ROUTEB_HOME")
    if home:
        from pathlib import Path
        try:
            (Path(home) / "box8").mkdir(parents=True, exist_ok=True)
            (Path(home) / "box8" / "gpusmoke.json").write_text(json.dumps(res, indent=1))
        except OSError:
            pass
    bad = [d for d in res if not (d["tests"] and all(t["ok"] for t in d["tests"]))]
    if bad:
        print(f"GPU SMOKE TEST FAILED on GPU(s) {[d['gpu'] for d in bad]}: nothing was fetched, nothing was started.", file=sys.stderr)
        return 6
    return 0


if __name__ == "__main__":
    sys.exit(main())
