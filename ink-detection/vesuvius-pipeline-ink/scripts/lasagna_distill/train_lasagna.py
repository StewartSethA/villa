#!/usr/bin/env python3
"""Distil the published Lasagna fields (nx, ny, grad_mag, cos) from CT: a 3-D U-Net, DDP over the box's GPUs.

Data: build_patches.py output, <root>/<scroll>/{x.npy (N,128,128,128) u8 CT, y.npy (N,4,128,128,128) u8 fields, meta.json}.
Targets are the PUBLISHED uint8 encodings, regressed in [0,1] (u8/255); a field value of 0 is "unset" in the published
encoding and is masked out of every loss and metric (an unlabelled voxel is not a target, D23 in spirit).
Loss = masked Huber on all four channels + DIR_W x (1 - cos) of the in-plane normal direction (nx, ny) where the decoded
magnitude exceeds DIR_MIN_MAG (nx = (u8-128)/127).  Input is standardised per patch over its non-zero voxels
(scan-agnostic; INPUT_NORM, disclosed).  Held-out SCROLLS are scored separately every eval (never a pooled number); the run
stops on the plateau rule (D34) or the hard wall-clock cap, which is reported as NON-CONVERGED.

  torchrun --nproc_per_node=4 train_lasagna.py --train DIR --val DIR --out DIR [--bs 4 --crop 96 --max-hours 20]
Profiling (D39): --profile N runs N steps, prints steps/s and MVox/s after warm-up, writes nothing else.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CH = ("nx", "ny", "grad_mag", "cos")
HUBER_DELTA = 0.05        # in u8/255 units (~13 grey levels)
DIR_W = 0.5               # weight of the in-plane direction loss
DIR_MIN_MAG = 0.3         # decoded |(nx,ny)| below this has no stable direction
INPUT_NORM = "per-patch mean/std over voxels with CT > 0"


def say(m):
    if int(os.environ.get("RANK", 0)) == 0:
        print(f"[{time.strftime('%H:%M:%S')}] lasagna: {m}", flush=True)


class Patches(torch.utils.data.Dataset):
    def __init__(self, root, scrolls=None, crop=96, n_per_epoch=1_000_000, fixed=None, seed=0):
        self.items, self.meta = [], {}
        for d in sorted(Path(root).iterdir()):
            mp = d / "meta.json"
            if not (mp.exists() and (not scrolls or d.name in scrolls)):
                continue
            m = json.loads(mp.read_text())
            if not m.get("done"):
                continue
            self.meta[d.name] = m
            self.items += [(d.name, i) for i in range(m["n"])]
        self.root, self.crop, self.n, self.seed = Path(root), crop, n_per_epoch, seed
        self.fixed = fixed                      # None = random crops; int = n fixed full patches per scroll
        self._x, self._y = {}, {}
        if fixed:
            by = {}
            for s, i in self.items:
                by.setdefault(s, []).append(i)
            rng = np.random.default_rng(seed)
            self.items = [(s, int(i)) for s in sorted(by) for i in rng.choice(by[s], min(fixed, len(by[s])), replace=False)]
        say(f"{root}: {len(self.items)} patches from {len(self.meta)} scrolls ({', '.join(sorted(self.meta))})")

    def __len__(self):
        return len(self.items) if self.fixed else self.n

    def _arr(self, s):
        if s not in self._x:
            self._x[s] = np.load(self.root / s / "x.npy", mmap_mode="r")
            self._y[s] = np.load(self.root / s / "y.npy", mmap_mode="r")
        return self._x[s], self._y[s]

    def __getitem__(self, k):
        if self.fixed:
            s, i = self.items[k]
            o = (0, 0, 0)
            c = 128
        else:
            rng = np.random.default_rng((self.seed, k, torch.initial_seed() % (1 << 31)))
            s, i = self.items[int(rng.integers(len(self.items)))]
            c = self.crop
            o = tuple(int(rng.integers(0, 128 - c + 1)) for _ in range(3))
        x, y = self._arr(s)
        sl = (slice(o[0], o[0] + c), slice(o[1], o[1] + c), slice(o[2], o[2] + c))
        return torch.from_numpy(np.ascontiguousarray(x[i][sl])), torch.from_numpy(np.ascontiguousarray(y[i][(slice(None),) + sl])), s


def prep(x, y):
    """uint8 -> standardised float input, [0,1] targets, validity mask (field != 0)."""
    x = x.float()
    m = (x > 0).float()
    n = m.sum((1, 2, 3), keepdim=True).clamp(min=1)
    mu = (x * m).sum((1, 2, 3), keepdim=True) / n
    sd = (((x - mu) * m) ** 2).sum((1, 2, 3), keepdim=True).div(n).sqrt().clamp(min=1.0)
    xin = ((x - mu) / sd * m).unsqueeze(1)
    return xin, y.float() / 255.0, (y > 0).float()


class Block(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.a = nn.Sequential(nn.Conv3d(i, o, 3, padding=1, bias=False), nn.GroupNorm(8, o), nn.SiLU(),
                               nn.Conv3d(o, o, 3, padding=1, bias=False), nn.GroupNorm(8, o), nn.SiLU())

    def forward(self, x):
        return self.a(x)


class UNet3D(nn.Module):
    def __init__(self, base=24, depth=4, out=4):
        super().__init__()
        ch = [base * 2 ** k for k in range(depth)]
        self.enc = nn.ModuleList([Block(1 if k == 0 else ch[k - 1], c) for k, c in enumerate(ch)])
        self.up = nn.ModuleList([nn.ConvTranspose3d(ch[k], ch[k - 1], 2, stride=2) for k in range(depth - 1, 0, -1)])
        self.dec = nn.ModuleList([Block(2 * ch[k - 1], ch[k - 1]) for k in range(depth - 1, 0, -1)])
        self.head = nn.Conv3d(ch[0], out, 1)

    def forward(self, x):
        skips = []
        for k, e in enumerate(self.enc):
            x = e(x if k == 0 else F.max_pool3d(x, 2))
            skips.append(x)
        for u, d, s in zip(self.up, self.dec, reversed(skips[:-1])):
            x = d(torch.cat([u(x), s], 1))
        return self.head(x)


def loss_fn(p, t, m):
    h = F.huber_loss(p, t, reduction="none", delta=HUBER_DELTA)
    base = (h * m).sum() / m.sum().clamp(min=1)
    nx, ny = (p[:, 0] * 255 - 128) / 127, (p[:, 1] * 255 - 128) / 127
    tx, ty = (t[:, 0] * 255 - 128) / 127, (t[:, 1] * 255 - 128) / 127
    mm = m[:, 0] * m[:, 1] * (torch.sqrt(tx ** 2 + ty ** 2) > DIR_MIN_MAG).float()
    cosang = (nx * tx + ny * ty) / (torch.sqrt(nx ** 2 + ny ** 2).clamp(min=1e-3) * torch.sqrt(tx ** 2 + ty ** 2).clamp(min=1e-3))
    d = ((1 - cosang) * mm).sum() / mm.sum().clamp(min=1)
    return base + DIR_W * d, base.detach(), d.detach()


@torch.no_grad()
def metrics(p, t, m):
    """Per-scroll sums for held-out scoring: masked MAE per channel (u8 units) and in-plane angular error (deg)."""
    out = {}
    for c, name in enumerate(CH):
        e = (p[:, c] - t[:, c]).abs() * 255 * m[:, c]
        out[f"mae_u8_{name}"] = (e.sum().item(), m[:, c].sum().item())
    nx, ny = (p[:, 0] * 255 - 128) / 127, (p[:, 1] * 255 - 128) / 127
    tx, ty = (t[:, 0] * 255 - 128) / 127, (t[:, 1] * 255 - 128) / 127
    mm = (m[:, 0] * m[:, 1] * (torch.sqrt(tx ** 2 + ty ** 2) > DIR_MIN_MAG).float()) > 0
    ang = torch.rad2deg(torch.acos(((nx * tx + ny * ty) / (torch.sqrt(nx ** 2 + ny ** 2).clamp(min=1e-3) * torch.sqrt(tx ** 2 + ty ** 2).clamp(min=1e-3))).clamp(-1, 1)))
    out["ang_deg_vals"] = ang[mm].detach().float().cpu().numpy()
    return out


def evaluate(model, loader, dev, step, tb, rank, world):
    model.eval()
    acc = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x, y, s in loader:
            xin, t, m = prep(x.to(dev), y.to(dev))
            p = model(xin).float()
            for i in range(len(s)):
                r = metrics(p[i:i + 1], t[i:i + 1], m[i:i + 1])
                a = acc.setdefault(s[i], {"ang": [], **{k: [0.0, 0.0] for k in r if k != "ang_deg_vals"}})
                a["ang"].append(r["ang_deg_vals"])
                for k, v in r.items():
                    if k != "ang_deg_vals":
                        a[k][0] += v[0]
                        a[k][1] += v[1]
    res = {}
    for s, a in acc.items():
        ang = np.concatenate(a["ang"]) if a["ang"] else np.zeros(1)
        res[s] = {k: (v[0] / max(v[1], 1)) for k, v in a.items() if k != "ang"}
        res[s].update({"ang_deg_p50": float(np.percentile(ang, 50)), "ang_deg_p90": float(np.percentile(ang, 90)), "n_voxels_scored": float(len(ang))})
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=4, help="per GPU")
    ap.add_argument("--crop", type=int, default=96)
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--val-per-scroll", type=int, default=48)
    ap.add_argument("--max-hours", type=float, default=20.0, help="hard cap: reaching it is reported NON-CONVERGED")
    ap.add_argument("--patience", type=int, default=6, help="evals without a >0.5%% relative gain => converged")
    ap.add_argument("--profile", type=int, default=0)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--no-benchmark", action="store_true", help="disable cudnn.benchmark autotuning (on by default; D39 profile)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.backends.cudnn.benchmark = not a.no_benchmark
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    if world > 1:
        torch.distributed.init_process_group("nccl")
    torch.manual_seed(a.seed + rank)
    out = Path(a.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    tb = None
    if rank == 0 and not a.profile:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(str(out / f"tb_unet3d_b{a.base}_crop{a.crop}_in1_out4_bs{a.bs}x{world}"))

    tr = Patches(a.train, crop=a.crop, seed=a.seed)
    sampler = None
    tl = torch.utils.data.DataLoader(tr, batch_size=a.bs, num_workers=a.workers, persistent_workers=True, prefetch_factor=4,
                                     pin_memory=True)
    va = Patches(a.val, fixed=a.val_per_scroll, seed=1)
    vl = torch.utils.data.DataLoader(torch.utils.data.Subset(va, list(range(rank, len(va), world))), batch_size=2, num_workers=2)
    model = UNet3D(a.base).to(dev).to(memory_format=torch.channels_last_3d)
    npar = sum(p.numel() for p in model.parameters())
    if a.compile:
        model = torch.compile(model)
    if world > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local])
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2, min_lr=1e-5)
    say(f"U-Net3D base {a.base}: {npar / 1e6:.2f} M params; window {a.crop}^3 voxels at pyramid level 2 (4x the scan voxel pitch, ~35 um/voxel at 8.64 um scans); "
        f"{world} GPU x bs {a.bs}; input norm: {INPUT_NORM}; loss Huber({HUBER_DELTA}) + {DIR_W} x direction")

    t0, step, best, since, last_t, hist = time.time(), 0, float("inf"), 0, time.time(), []
    it = iter(tl)
    warm = 20
    status = "RUNNING"
    while True:
        try:
            x, y, _ = next(it)
        except StopIteration:
            it = iter(tl)
            continue
        xin, t, m = prep(x.to(dev, non_blocking=True), y.to(dev, non_blocking=True))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(xin.contiguous(memory_format=torch.channels_last_3d)).float()
        loss, base_l, dir_l = loss_fn(p, t, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step == warm:
            torch.cuda.synchronize()
            t_warm = time.time()
        if a.profile and step == a.profile:
            torch.cuda.synchronize()
            dt = time.time() - t_warm
            n = (a.profile - warm)
            mv = n * a.bs * world * a.crop ** 3 / 1e6 / dt
            say(f"PROFILE bs={a.bs} crop={a.crop} workers={a.workers}: {n / dt:.2f} steps/s, {mv:.0f} MVox/s over {n} steps after {warm} warm-up, "
                f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
            return
        if rank == 0 and step % 50 == 0:
            sps = 50 / (time.time() - last_t)
            last_t = time.time()
            if tb:
                tb.add_scalar("train/loss", loss.item(), step)
                tb.add_scalar("train/huber", base_l.item(), step)
                tb.add_scalar("train/dir_loss", dir_l.item(), step)
                tb.add_scalar("speed/steps_per_s", sps, step)
                tb.add_scalar("speed/MVox_per_s_train", sps * a.bs * world * a.crop ** 3 / 1e6, step)
            if step % 200 == 0:
                el = (time.time() - t0) / 3600
                say(f"step {step} loss {loss.item():.4f} huber {base_l.item():.4f} dir {dir_l.item():.4f} lr {opt.param_groups[0]['lr']:.1e} "
                    f"{sps:.2f} it/s | elapsed {el:.2f} h, cap {a.max_hours} h (ETA to cap {max(0, a.max_hours - el):.1f} h; convergence ETA unknown until plateau)")
        if step % a.eval_every == 0:
            res = evaluate(model.module if world > 1 else model, vl, dev, step, tb, rank, world)
            if world > 1:
                gathered = [None] * world
                torch.distributed.all_gather_object(gathered, res)
                merged = {}
                for g in gathered:
                    for s, d in g.items():
                        merged.setdefault(s, []).append(d)
                res = {s: {k: float(np.mean([d[k] for d in ds])) for k in ds[0]} for s, ds in merged.items()}
            sel = float(np.mean([r["ang_deg_p50"] for r in res.values()]))
            if world > 1:
                sel_t = torch.tensor([sel], device=dev)
                torch.distributed.broadcast(sel_t, 0)
                sel = sel_t.item()
            sched.step(sel)
            improved = sel < best * 0.995
            if improved:
                best, since = sel, 0
            else:
                since += 1
            if rank == 0:
                for s, r in res.items():
                    for k, v in r.items():
                        tb.add_scalar(f"val_{s}/{k}", v, step)
                tb.add_scalar("converge/heldout_ang_deg_p50_mean", sel, step)
                hist.append({"step": step, "sel": sel, "per_scroll": res})
                (out / "val_history.json").write_text(json.dumps(hist, indent=1))
                core = model.module if world > 1 else model
                torch.save({"model": core.state_dict(), "step": step, "args": vars(a), "sel": sel}, out / "last.pt")
                if improved:
                    torch.save({"model": core.state_dict(), "step": step, "args": vars(a), "sel": sel}, out / "best.pt")
                say(f"[val] step {step} held-out in-plane angular error p50 (mean of scrolls) {sel:.2f} deg (best {best:.2f}, {since} evals since gain); "
                    + "; ".join(f"{s}: nx MAE {r['mae_u8_nx']:.1f} ny {r['mae_u8_ny']:.1f} gm {r['mae_u8_grad_mag']:.1f} u8, ang p50/p90 {r['ang_deg_p50']:.1f}/{r['ang_deg_p90']:.1f} deg" for s, r in res.items()))
            stop = torch.tensor([int(since >= a.patience or (time.time() - t0) / 3600 >= a.max_hours)], device=dev)
            if world > 1:
                torch.distributed.broadcast(stop, 0)
            if stop.item():
                status = "CONVERGED" if since >= a.patience else "NON-CONVERGED (wall-clock cap hit)"
                break
    if rank == 0:
        (out / "STATUS.json").write_text(json.dumps({"status": status, "steps": step, "hours": round((time.time() - t0) / 3600, 2),
                                                     "best_heldout_ang_deg_p50": best, "rule": f"no >0.5% relative gain in {a.patience} consecutive evals, cap {a.max_hours} h"}, indent=1))
        say(f"{status} at step {step}, best held-out angular error p50 {best:.2f} deg")


if __name__ == "__main__":
    main()
