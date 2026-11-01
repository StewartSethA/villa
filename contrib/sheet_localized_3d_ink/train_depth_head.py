#!/usr/bin/env python3
"""Fine-tune a reader-v3-style 2-D ink net with a sheet-localised depth head (see README.md and depth_head.py).

Data per segment (any scroll with ink labels): ROOT/<seg>/{vol17.zarr (17,H,W) u8, core.zarr (H,W) u8 (ink = >=128), sup.zarr (H,W) u8 (judged = >0),
maybe.zarr (H,W) u8 ('maybe' = >=128: never trained on), hec3d.zarr (16,H,W) u8 (Hecate teacher volume, hecate_teacher_volumes.py)}.
Windows are drawn at random from the left 90 % of every segment (positive-biased); the right 10 % x-band of every segment is the held-out set (same segments, NOT a transfer test:
run the upstream-test harness on the final checkpoint for that).  Each batch can be mixed with windows of your general training store (--aux-factory): without it the net specialises to
these segments and loses transfer (we measured 0.75-0.79 instead of 0.889-0.897 on the upstream test).

Recipe: SGD momentum 0.99 nesterov, lr 2e-4, 200-step warm-up (the reader-v3 fine-tune recipe; AdamW collapsed the held-out AUC), per-window robust-MAD normalisation, flips + rot90 on the GPU,
masked BCE for the 2-D logit + weight x masked BCE for the 16 depth logits.  --depth-w 0 is the single-task control.  Stops when the held-out 2-D AUC has not gained > 0.002 for --patience evals.
usage: train_depth_head.py ROOT OUTDIR --net-factory pkg.mod:fn --ckpt reader-v3.pth [--aux-factory pkg.mod:fn] [--depth-w 1.0] [--film 1] [--seed 0]
  net-factory:  fn(ckpt_path, device) -> nn.Module mapping (B,1,17,H,W) float -> (B,1,H,W) logits, exposing net.model.network.shared_decoder (the reader-v3 layout)
  aux-factory:  fn(tile, batch) -> iterator of dicts {x uint8 (B,17,T,T), y uint8 (B,T,T), w uint8 (B,T,T)}"""
import argparse, importlib, json, os, queue, threading, time
import numpy as np, torch, torch.nn.functional as F, zarr
from scipy.stats import rankdata
from depth_head import attach_depth_head, depth_loss, snap_targets

ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("out"); ap.add_argument("--net-factory", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--aux-factory", default=None)
ap.add_argument("--tile", type=int, default=320); ap.add_argument("--batch", type=int, default=6); ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--warmup", type=int, default=200)
ap.add_argument("--depth-w", type=float, default=1.0); ap.add_argument("--film", type=int, default=1); ap.add_argument("--hmin", type=float, default=0.3)
ap.add_argument("--eval", type=int, default=500); ap.add_argument("--patience", type=int, default=10); ap.add_argument("--max-steps", type=int, default=40000); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=False)
dev = "cuda"; torch.manual_seed(a.seed); rng_np = np.random.default_rng(a.seed)


def robust_mad(x: torch.Tensor, lo=1.0, hi=99.0) -> torch.Tensor:
    """per-sample clip to [p_lo, p_hi], subtract the median, divide by 1.4826 x MAD (fallback std, floor span/20)."""
    out = torch.empty_like(x)
    for i in range(x.shape[0]):
        v = x[i]; s = torch.sort(v.flatten()[:: max(1, v.numel() // 4_000_000)]).values; m = s.numel()
        ql, qh = s[int(round(lo / 100 * (m - 1)))], s[int(round(hi / 100 * (m - 1)))]
        v = v.clamp(ql, qh); med = v.median(); mad = (v - med).abs().median() * 1.4826
        if not torch.isfinite(mad) or mad < 1e-6:
            mad = v.std() if float(v.std()) >= 1e-6 else torch.ones((), device=x.device)
        mad = torch.maximum(mad, (qh - ql) / 20.0) if float(qh - ql) > 0 else mad
        out[i] = torch.nan_to_num((v - med) / mad)
    return out


class Segments:
    def __init__(self, root, tile, held_out=False, seed=0):
        self.T, self.rng, self.held, self.segs = tile, np.random.default_rng(seed), held_out, []
        for d in sorted(os.listdir(root)):
            p = f"{root}/{d}/"
            if not all(os.path.exists(p + f) for f in ("vol17.zarr", "core.zarr", "sup.zarr", "maybe.zarr", "hec3d.zarr")): continue
            z = {k: zarr.open(p + f"{k}.zarr", mode="r") for k in ("vol17", "core", "sup", "maybe", "hec3d")}
            H, W = z["core"].shape
            if H < tile or W < tile: continue
            sup = np.asarray(z["sup"][::32, ::32]) > 0; core = np.asarray(z["core"][::32, ::32]) >= 128; xb = int(0.9 * sup.shape[1])
            if held_out: sup[:, :xb] = False
            else: sup[:, xb:] = False; core[:, xb:] = False
            if sup.sum() == 0: continue
            self.segs.append(dict(tag=d, z=z, H=H, W=W, sup=np.argwhere(sup), pos=np.argwhere(core & sup), xmax=xb * 32))
        assert self.segs, f"no segment with all five arrays under {root}"

    def one(self):
        s = self.segs[self.rng.integers(len(self.segs))]; T = self.T
        cand = s["pos"] if (self.rng.random() < 0.7 and len(s["pos"])) else s["sup"]
        cy, cx = cand[self.rng.integers(len(cand))] * 32 + self.rng.integers(0, 32, 2)
        y0 = int(np.clip(cy - self.rng.integers(T // 4, 3 * T // 4), 0, s["H"] - T)); x0 = int(np.clip(cx - self.rng.integers(T // 4, 3 * T // 4), 0, s["W"] - T))
        if self.held: x0 = int(np.clip(x0, min(s["xmax"], s["W"] - T), s["W"] - T))
        else: x0 = int(np.clip(x0, 0, max(0, min(s["xmax"], s["W"]) - T)))
        sl = (slice(y0, y0 + T), slice(x0, x0 + T)); z = s["z"]
        c = np.asarray(z["core"][sl]); m = np.asarray(z["maybe"][sl]); sp = np.asarray(z["sup"][sl])
        return np.asarray(z["vol17"][:, sl[0], sl[1]]), (c >= 128).astype(np.uint8), ((sp > 0) & (m < 128)).astype(np.uint8), np.asarray(z["hec3d"][:, sl[0], sl[1]])

    def batch(self, n):
        it = [self.one() for _ in range(n)]
        return [torch.from_numpy(np.stack([i[k] for i in it])) for k in range(4)]


def auc(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y.astype(bool)].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


fn_mod, fn_name = a.net_factory.split(":"); net = getattr(importlib.import_module(fn_mod), fn_name)(a.ckpt, dev); head = attach_depth_head(net).to(dev)
params = list(net.parameters()) + list(head.parameters())
opt = torch.optim.SGD(params, a.lr, momentum=0.99, nesterov=True, weight_decay=3e-5)
fwd = lambda x: net(x[:, None]).float()[:, 0]
train, val = Segments(a.root, a.tile, False, a.seed), Segments(a.root, a.tile, True, 7)
aux = iter(getattr(importlib.import_module(a.aux_factory.split(":")[0]), a.aux_factory.split(":")[1])(a.tile, max(2, a.batch))) if a.aux_factory else None
q = queue.Queue(8)
for k in range(a.threads):
    s_k = Segments(a.root, a.tile, False, a.seed + 1 + k); threading.Thread(target=lambda s=s_k: [q.put(s.batch(a.batch)) for _ in iter(int, 1)], daemon=True).start()
VW = [val.one() for _ in range(96)]
datasets = dict(train_segments=[s["tag"] for s in train.segs], held_out_band="right 10 % x-band of the same segments", aux=a.aux_factory, args=vars(a))
json.dump(datasets, open(a.out + "/datasets.json", "w"), indent=1)                                  # written before the first step


def evaluate():
    net.eval(); ys, ss, cors = [], [], []
    with torch.no_grad():
        for v, c, m, h in VW:
            x = robust_mad(torch.from_numpy(v)[None].to(dev).float()); p = torch.sigmoid(fwd(x))[0].cpu().numpy(); msk = m > 0; ys.append(c[msk]); ss.append(p[msk])
            core = (c > 0) & msk
            if core.sum() > 50:
                pd = torch.sigmoid(head.logits())[0].cpu().numpy()[:, core].ravel(); hh = h.astype(np.float32)[:, core].ravel() / 255
                cors.append(float(np.corrcoef(pd, hh)[0, 1]) if pd.std() > 0 and hh.std() > 0 else float("nan"))
    net.train(); return auc(np.concatenate(ys), np.concatenate(ss)), float(np.nanmean(cors)) if cors else float("nan")


best, plateau, hist, t0 = evaluate()[0], 0, [], time.time(); net.train()
print(f"step 0: held-out 2-D AUC {best:.4f}", flush=True)
for step in range(1, a.max_steps + 1):
    x, y, w, h = q.get()
    if aux is not None:
        ab = next(aux); x, y, w = torch.cat([x, ab["x"]]), torch.cat([y, ab["y"]]), torch.cat([w, ab["w"]]); h = torch.cat([h, torch.zeros(ab["x"].shape[0], 16, *h.shape[2:], dtype=h.dtype)])
    hm = torch.zeros(x.shape[0], device=dev); hm[:a.batch] = 1.0                                       # the depth loss only on windows that have a teacher volume
    x = robust_mad(x.to(dev).float()); y = y.to(dev).float(); w = w.to(dev).float(); h = h.to(dev)
    if rng_np.random() < 0.5: x, y, w, h = x.flip(-1), y.flip(-1), w.flip(-1), h.flip(-1)
    k = int(rng_np.integers(4)); x, y, w, h = [torch.rot90(t, k, (-2, -1)) for t in (x, y, w, h)]
    for g in opt.param_groups: g["lr"] = a.lr * min(1.0, step / max(1, a.warmup))
    with torch.autocast("cuda", dtype=torch.float16): lg = fwd(x)
    loss = (F.binary_cross_entropy_with_logits(lg, y, reduction="none") * w).sum() / w.sum().clamp(min=1)
    if a.depth_w > 0:
        t3, mk = snap_targets(y, w, h, film=a.film, hmin=a.hmin); loss = loss + a.depth_w * depth_loss(head.logits(), t3, mk * hm[:, None, None])
    else: head.feat = None
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
    if step % 50 == 0: print(f"step {step} loss {float(loss):.4f} {step * x.shape[0] / (time.time() - t0):.1f} samples/s", flush=True)
    if step % a.eval == 0:
        a2d, r = evaluate(); hist.append(dict(step=step, auc2d=round(a2d, 4), r_teacher=round(r, 3))); print(f"EVAL step {step}: held-out 2-D AUC {a2d:.4f}, depth-profile r with teacher {r:.3f}", flush=True)
        torch.save(dict(state_dict=net.state_dict(), head=head.state_dict(), step=step), a.out + "/last.pt")
        if a2d > best + 0.002: best, plateau = a2d, 0
        else: plateau += 1
        json.dump(dict(hist=hist, converged=plateau >= a.patience), open(a.out + "/result.json", "w"), indent=1)
        if plateau >= a.patience: print("CONVERGED (plateau rule)"); break
