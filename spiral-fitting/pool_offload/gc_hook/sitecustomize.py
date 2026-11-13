# Route B fit_spiral MEMORY-LEAK FIX (sheetgeom study 2026-10-29, docs/ROUTEB_MEMORY.md section 3). Env-gated; does nothing unless set.
# fit_spiral builds new transform objects every iteration; they sit in reference cycles (each holds a full gap-logit lattice copy and a full-res
# flow copy) that only Python's cyclic GC frees, so device memory grows with the iteration count until a collection happens to run.
# Put this directory first on PYTHONPATH of the fit process and set ONE of:
#   FIT_GC_EVERY_STEP=N   gc.collect() after every N-th optimiser step (wraps torch.optim.AdamW.step; N=1 recommended: rate-independent)
#   FIT_GC_EVERY_S=T      gc.collect() every T seconds from a side thread, started once the log shows training ("step 0:") -- needs FIT_GC_LOG
# Measured: PHerc0343 z 9000-10500 shell 250: torch peak 8.52 GiB -> 3.14 GiB; one fit over z 1200-17500 (32-voxel flow grid) 20.1 GiB on a V100-32
# (without it: OOM at iteration 18).
import os, sys, gc, time, threading
if "fit_spiral" in " ".join(sys.argv) and os.environ.get("FIT_LOWP") == "adam_bf16":
    # Optional memory reduction (ported from the rbmem probe hook, 2026-10-30): AdamW whose two moment buffers are bfloat16 (update math fp32;
    # params and grads unchanged).  Installed BEFORE the gc / snapshot wrappers below so they wrap this class's step.
    import torch

    class AdamWBF16State(torch.optim.Optimizer):
        def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, **_ignored):
            super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

        @torch.no_grad()
        def step(self, closure=None):
            for g in self.param_groups:
                b1, b2 = g["betas"]
                for p in g["params"]:
                    if p.grad is None: continue
                    st = self.state[p]
                    if not st:
                        st["step"] = torch.zeros((), dtype=torch.float32)
                        st["exp_avg"] = torch.zeros_like(p, dtype=torch.bfloat16); st["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.bfloat16)
                    st["step"] += 1; t = float(st["step"])
                    lr = float(g["lr"]) if not torch.is_tensor(g["lr"]) else float(g["lr"].item())
                    if g["weight_decay"]: p.mul_(1 - lr * g["weight_decay"])
                    m = st["exp_avg"].float().mul_(b1).add_(p.grad, alpha=1 - b1)
                    v = st["exp_avg_sq"].float().mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                    st["exp_avg"].copy_(m); st["exp_avg_sq"].copy_(v)
                    den = (v / (1 - b2 ** t)).sqrt_().add_(g["eps"])
                    p.addcdiv_(m, den, value=-lr / (1 - b1 ** t))
                    del m, v, den

    torch.optim.AdamW = AdamWBF16State
    print("fit_gc: FIT_LOWP=adam_bf16 -> torch.optim.AdamW replaced (bf16 moments)", flush=True)
if "fit_spiral" in " ".join(sys.argv):
    if os.environ.get("FIT_GC_EVERY_STEP"):
        import torch
        _n = max(1, int(os.environ["FIT_GC_EVERY_STEP"])); _orig = torch.optim.AdamW.step; _c = [0, 0.0]
        def _step(self, *a, **k):
            r = _orig(self, *a, **k); _c[0] += 1
            if _c[0] % _n == 0:
                t = time.perf_counter(); gc.collect(); _c[1] += time.perf_counter() - t
                if _c[0] % 1000 == 0:
                    print(f"fit_gc: gc.collect after step {_c[0]}: mean {1000 * _c[1] / (_c[0] // _n):.1f} ms", flush=True)
            return r
        torch.optim.AdamW.step = _step
        print(f"fit_gc: gc.collect() after every {_n} optimiser step(s)", flush=True)
    elif os.environ.get("FIT_GC_EVERY_S"):
        _p = float(os.environ["FIT_GC_EVERY_S"]); _lf = os.environ.get("FIT_GC_LOG", "")
        def _loop():
            while _lf:
                time.sleep(5)
                try:
                    if b"step 0:" in open(_lf, "rb").read(): break
                except Exception:
                    pass
            while True:
                time.sleep(_p); gc.collect()
        threading.Thread(target=_loop, daemon=True).start()
        print(f"fit_gc: gc.collect() every {_p} s", flush=True)
