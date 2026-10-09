#!/usr/bin/env python3
"""Distil Reader v2 (or, for the paired baseline, ink_9um) into the SMALL-RF DENSE student
`reader_v2_dense` (src/vesuvius_pipeline/stages/ink_models/reader_v2_dense.py). 2026-10-06.

A copy of scripts/hires_ink/ink9um_distill.py (the recipe of the production ink9um_student and the
sharp_student ablation arms -- loss, hot-crop sampler, scroll-balanced weights, whole-segment hold-out,
best-checkpoint selection on held-out teacher agreement are UNCHANGED) with:
  * the normaliser = reader_v2_dense.apply_norm ('local<k>' / 'ms<a>-<b>' small windows), so the
    receptive field through normaliser + net is measured (rf_total) and asserted <= --rf-cap;
  * --label-subj: every --val-every steps ("epoch"), per scroll GROUP of the labelled subjects:
    near64 AUC and near64 BCE vs the binary label (oracle face per tile) -> TB label_auc/<group>,
    label_bce/<group>, plus a CT | label | prediction image of one fixed tile per group. LOGGED ONLY:
    the checkpoint is chosen on teacher agreement, never on labels.
  * --loss softbce+logit+edge (recipe + the sharp ablation's Sobel edge term).

Data: `ink9um_teacher_targets.py` output -- per segment the teacher's own 17-layer input window
(x.npy, uint8) and its probability maps for BOTH faces (t_fwd.png / t_rev.png). Distillation, so
no labels: the target is the teacher's probability, a SOFT target, never thresholded; the loss is
BCE of the student logit against that probability (the cross-entropy to a sigmoid teacher; its
gradient p - t does not saturate at confident pixels), masked to the rendered surface (the
teacher's own validity mask: middle layer > 0).

Held out: WHOLE segments (`--val` names, or every segment whose name hashes into the val bucket),
scored every `--val-every` steps with the DEPLOYED inference path (`predict_array`: 2048 px tiles,
96 px halo, no overlap) against the teacher over the whole segment -- pixel Pearson r, MAE and
the same numbers per face.

TensorBoard (CLAUDE.md): train loss per step; val r/MAE per held-out segment and pooled; fixed
held-out crops as CT | teacher | student images; throughput (train MVox/s, val inference MVox/s).
"""
import argparse
import glob
import hashlib
import json
import math
import os
import sys
import threading
import queue
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _import_student(src_root):
    sys.path.insert(0, src_root)
    from vesuvius_pipeline.stages.ink_models import ink9um_student as S   # noqa: E402
    from vesuvius_pipeline.stages.ink_models import reader_v2_dense as R   # noqa: E402
    return S, R


PRELOAD = {"on": True}


def load_seg(d):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    # PRELOAD (2026-10-06): read x.npy into anonymous memory. A memmap faulted in at random crop positions on a
    # box whose page cache is full stalls on reclaim; systemd-oomd counts that as memory PRESSURE and killed the
    # whole training scope at 39.7 GB peak (host, rv2-queue.scope, 09:34Z), far below its MemoryMax.
    x = np.load(os.path.join(d, "x.npy"), mmap_mode=None if PRELOAD["on"] else "r")
    tf = np.asarray(Image.open(os.path.join(d, "t_fwd.png")))
    tr = np.asarray(Image.open(os.path.join(d, "t_rev.png")))
    meta = json.load(open(os.path.join(d, "meta.json")))
    lm = os.path.join(d, "lossmask.npy")      # optional: pixels the loss may use (1) -- e.g. S4 with
    #                                          its HF VALIDATION boxes cut out, so the student never
    #                                          fits the teacher there
    return {"name": os.path.basename(d), "x": x, "t": {"fwd": tf, "rev": tr}, "meta": meta,
            "lossmask": np.load(lm, mmap_mode="r") if os.path.exists(lm) else None}


def measure_rf(model, dev, size=512, in_ch=17):
    """Receptive field of the centre output pixel, by backprop: full extent of nonzero input
    gradient, and the radius holding 99 % of the |gradient| mass."""
    import torch
    model = model.float().eval()
    x = torch.randn(1, in_ch, size, size, device=dev, requires_grad=True)
    y = model(x)
    y[0, 0, size // 2, size // 2].backward()
    g = x.grad.abs().sum(1)[0].cpu().numpy()
    nz = np.argwhere(g > g.max() * 1e-6)
    ext = int(max(nz[:, 0].max() - nz[:, 0].min(), nz[:, 1].max() - nz[:, 1].min()) + 1)
    yy, xx = np.mgrid[:size, :size]
    r = np.maximum(np.abs(yy - size // 2), np.abs(xx - size // 2)).ravel()
    order = np.argsort(r)
    cm = np.cumsum(g.ravel()[order]) / g.sum()
    r99 = int(r[order][np.searchsorted(cm, 0.99)])
    return {"rf_extent_px": ext, "rf_r99_px": r99, "rf_r99_window_px": 2 * r99 + 1}


_SOBEL = None


def _edge_l1(p, t, w):
    """Masked L1 between Sobel gradient magnitude of student prob `p` and teacher prob `t`,
    (B, H, W) each, weight `w` same shape. Ablation B (softbce+edge)."""
    import torch
    import torch.nn.functional as F
    global _SOBEL
    if _SOBEL is None or _SOBEL.device != p.device:
        kx = torch.tensor([[1., 0, -1], [2, 0, -2], [1, 0, -1]], device=p.device)
        ky = kx.t()
        _SOBEL = torch.stack([kx, ky])[:, None]      # (2,1,3,3)
    g_p = F.conv2d(p[:, None], _SOBEL, padding=1)
    g_t = F.conv2d(t[:, None], _SOBEL, padding=1)
    mag_p = (g_p ** 2).sum(1).clamp_min(1e-12).sqrt()
    mag_t = (g_t ** 2).sum(1).clamp_min(1e-12).sqrt()
    return ((mag_p - mag_t).abs() * w).sum() / w.sum().clamp(min=1)


def pearson(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    a -= a.mean(); b -= b.mean()
    return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--src", required=True, help="repo src/ holding vesuvius_pipeline")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val", default="", help="comma list of held-out segment names")
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--exclude", default="", help="comma list never used (label controls)")
    ap.add_argument("--exclude-prefix", default="PHerc0172,PHercParis4",
                    help="held-out SCROLLS: never trained on (their label controls are the label test)")
    ap.add_argument("--rescan-every", type=int, default=2000, help="pick up newly written segments")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--margin", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--widths", default="32,48,96,128,192")
    ap.add_argument("--blocks", default="", help="residual blocks per encoder level, e.g. 0,1,2,3,3")
    ap.add_argument("--skip-off", default="",
                    help="sharp_student ablation C: comma list of decoder-step indices (0=finest) "
                         "whose skip connection is zeroed, e.g. '0,1' drops the two finest skips")
    ap.add_argument("--upscale", type=int, default=1,
                    help="sharp_student ablation C: PixelShuffle head factor (1=off, 2 or 4=finer "
                         "output grid trained by average-pooling back to native for the loss)")
    ap.add_argument("--target-sharpen", default="",
                    help="sharp_student ablation (2026-10-02): 'sigma_px:amount' -- unsharp-mask the teacher "
                         "TARGET on the fly (t + amount*(t - gauss(t, sigma)), clipped to [0,1]) so the "
                         "student is asked for a crisper edge than the teacher's own")
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-max", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--norm", default="ms9-17", help="reader_v2_dense norm: local<k> or ms<a>-<b>...")
    ap.add_argument("--rf-cap", type=int, default=64, help="refuse to train if the architectural support "
                    "(net extent + largest norm window - 1) exceeds this, px (user rule 2026-09-08)")
    ap.add_argument("--label-subj", default="", help="labelled subjects root (ink9um_tta_prep layout) for "
                    "per-group TB label metrics every --val-every; logged only, never used for selection")
    ap.add_argument("--resume", default="", help="checkpoint to continue from (weights + step; fresh optimiser, same schedule)")
    ap.add_argument("--loss", default="softbce",
                    choices=["softbce", "softbce+logit", "softbce+edge", "softbce+dice", "softbce+logit+edge"],
                    help="softbce: BCE to the teacher probability; +logit adds 0.1 x masked MSE between "
                         "student and teacher LOGITS, which weights the teacher's confident pixels (the "
                         "letters) far more than BCE does near p = 0.3; +edge (sharp_student ablation B, "
                         "2026-10-01) adds 1.0 x masked L1 between Sobel-gradient-magnitude of student "
                         "prob and teacher prob, penalising a student edge that is SHALLOWER than the "
                         "teacher's own (the teacher's edge is itself the blur floor under hypothesis 1, "
                         "so this cannot exceed it, only test whether matching it is being left on the "
                         "table by plain BCE); +dice adds 1.0 x soft Dice loss (1 - 2*sum(p*t)/sum(p+t) "
                         "over the masked region), which is known to sharpen boundaries vs pointwise BCE "
                         "in segmentation literature by weighting the OVERLAP rather than every pixel "
                         "equally")
    ap.add_argument("--hot-frac", type=float, default=0.0,
                    help="fraction of crops centred on the TEACHER's most confident 32-px blocks "
                         "(its own output, top 3 %% per segment and > 0.45 -- no labels read)")
    ap.add_argument("--run-prefix", default="ink9um_student",
                    help="run-name stem (progressive distillation: e.g. pd_g2_chain)")
    ap.add_argument("--targets-desc", default="ink9um",
                    help="what produced t_fwd/t_rev (run name + checkpoint field): the teacher, or a parent student")
    ap.add_argument("--teacher-id", default="checkpoints/ink_9um/hybrid_3d2d-seed42/step-075000.pth",
                    help="identity of the model whose maps are the targets, recorded in the checkpoint")
    ap.add_argument("--label-val", default="",
                    help="labelled held-out sets scored every --val-every: 'controls_data:controls_labels:s4_data:s4_labels' "
                         "(scripts/ink_next/progdistil/pd_labelval.py) -> TB label_auc/<scroll>, label_bce/<scroll>")
    ap.add_argument("--no-preload", action="store_true", help="memmap x.npy instead of reading it into RAM")
    # ---- convergence regime (2026-10-06, user: "undertrained, did not converge") -------------------------------
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of the weights (0 = off); the EMA weights are "
                    "what is validated, selected and saved as state_dict (raw weights kept as raw_state_dict)")
    ap.add_argument("--sched", default="cosine", choices=["cosine", "plateau"],
                    help="cosine over --steps (old), or plateau: warm-up then constant lr, x --plateau-factor when the "
                         "pooled held-out teacher r has not improved by --plateau-eps for --plateau-patience "
                         "validations; STOP (converged) on the plateau after --plateau-max-red reductions")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--plateau-patience", type=int, default=4)
    ap.add_argument("--plateau-eps", type=float, default=0.002)
    ap.add_argument("--plateau-factor", type=float, default=0.3)
    ap.add_argument("--plateau-max-red", type=int, default=2)
    ap.add_argument("--min-steps", type=int, default=0)
    ap.add_argument("--auto-resume", action="store_true", help="continue from <out>/resume.pt (full optimiser/EMA/"
                    "schedule state) if it exists -- the queue relaunches a killed run with this")
    # ---- segment-edge false positives (user 2026-10-06: arm0 'picked up segment edges') --------------------------
    ap.add_argument("--outside-w", type=float, default=0.0, help="loss weight on crop pixels OUTSIDE the rendered "
                    "surface (middle layer == 0), target 0 there (the teacher's own map is 0 there); 0 = unsupervised (old)")
    ap.add_argument("--edge-frac", type=float, default=0.0, help="fraction of crops centred on the surface boundary")
    # ---- training-time augmentation (user 2026-10-06: 3-D warp + scroll domain shift) ----------------------------
    ap.add_argument("--warp3d", type=float, default=0.0, help="probability of a random smooth 3-D warp per sample "
                    "(coarse + fine displacement, stretch/squash, pinch, bulge; scripts/ensemble_rv2/warp3d.py)")
    ap.add_argument("--warp-mode", default="full", choices=["full", "inplane", "depth"],
                    help="full: coarse+fine in-plane AND depth fields + stretch/pinch/bulge; inplane: coarse in-plane "
                         "field only, the same shift in every layer, 0-2 px rms (the only TTA warp that helped Reader v2, "
                         "FINDINGS 2026-10-06 ensembles); depth: depth fields + stretch/pinch/bulge only, no in-plane motion")
    ap.add_argument("--warp-mod", default=os.path.join(HERE, "..", "ensemble_rv2"), help="dir holding warp3d.py")
    ap.add_argument("--domain-aug", type=float, default=0.0, help="probability per sample of scroll-domain jitter: "
                    "in-plane resolution (um/px) x0.8-1.25, gamma, contrast/offset, in-plane + depth blur, noise")
    ap.add_argument("--val-tile", type=int, default=1024, help="tile px for in-run validation inference (GPU memory: "
                    "3 runs share one 16 GB card; the halo still covers the receptive field, so tiles join seamlessly)")
    ap.add_argument("--aug-pad", type=int, default=24, help="extra px read around each crop when augmenting, so a "
                    "zoom/warp samples real data rather than replicated borders")
    a = ap.parse_args()
    PRELOAD["on"] = not a.no_preload
    S, R = _import_student(a.src)
    import torch
    import torch.nn.functional as F
    from torch.utils.tensorboard import SummaryWriter
    torch.backends.cudnn.benchmark = True
    dev = "cuda"
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)

    segs = sorted(d for d in glob.glob(os.path.join(a.data, "*")) if os.path.exists(os.path.join(d, "meta.json")))
    excl = set(filter(None, a.exclude.split(",")))
    xpre = tuple(filter(None, a.exclude_prefix.split(",")))

    def is_val(name):
        # a WHOLE-SEGMENT hold-out, decided by the name alone so segments written after the start
        # are split by the same rule; held-out scrolls are validation-only
        return (name in set(filter(None, a.val.split(","))) or name.startswith(xpre) or
                int(hashlib.md5(name.encode()).hexdigest(), 16) % 1000 < a.val_frac * 1000)

    def prep(d):
        s = load_seg(d)
        s["st"] = S.segment_stats([s["x"][j] for j in range(s["x"].shape[0])])
        s["valid"] = np.asarray(s["x"][8]) > 0
        if s["lossmask"] is not None:
            s["valid"] &= np.asarray(s["lossmask"]).astype(bool)
        # the teacher's confident blocks: 32-px block means of max(face probabilities)
        tm = np.maximum(s["t"]["fwd"], s["t"]["rev"]).astype(np.float32) / 255
        Hb, Wb = tm.shape[0] // 32, tm.shape[1] // 32
        bm = tm[:Hb * 32, :Wb * 32].reshape(Hb, 32, Wb, 32).mean((1, 3))
        vb = s["valid"][:Hb * 32, :Wb * 32].reshape(Hb, 32, Wb, 32).mean((1, 3)) > 0.8
        thr = max(0.45, float(np.percentile(bm[vb], 97))) if vb.any() else 1.0
        s["hot"] = np.argwhere((bm >= thr) & vb) * 32 + 16
        # surface-boundary blocks: 32-px blocks that are partly rendered (edge-centred crops, --edge-frac)
        vf = s["valid"][:Hb * 32, :Wb * 32].reshape(Hb, 32, Wb, 32).mean((1, 3))
        s["edge"] = np.argwhere((vf > 0.15) & (vf < 0.85)) * 32 + 16
        return s

    segs = [d for d in segs if os.path.basename(d) not in excl]
    train = [prep(d) for d in segs if not is_val(os.path.basename(d))]
    # periodic validation: up to val_max held-out segments, trained scrolls first, then held-out scrolls
    vd = [d for d in segs if is_val(os.path.basename(d))]
    vd = sorted(vd, key=lambda d: (os.path.basename(d).startswith(xpre), os.path.basename(d)))
    val = [prep(d) for d in vd[:a.val_max]]
    known = {os.path.basename(d) for d in segs}
    widths = tuple(int(w) for w in a.widths.split(","))
    blocks = tuple(int(b) for b in a.blocks.split(",")) if a.blocks else None
    tpx = sum(int(s["valid"].sum()) for s in train)
    if blocks or a.skip_off or a.upscale != 1:
        raise SystemExit("rv2_distill: --blocks/--skip-off/--upscale are not supported by reader_v2_dense")
    in_ch = 17 * R.norm_channels(a.norm)
    HL = max(R.norm_reach(a.norm), 1)
    halo = R.norm_reach(a.norm) + 64
    rf0 = R.measure_total_rf(R.build(widths, in_ch), a.norm)
    print(f"[distill] rf (untrained, through normaliser): {rf0}", flush=True)
    if a.rf_cap > 0 and rf0["arch_support_px"] > a.rf_cap:
        raise SystemExit(f"rv2_distill: architectural support {rf0['arch_support_px']} px > cap {a.rf_cap} px")
    run = (f"{a.run_prefix}__unet2d-{len(widths)}lvl-w{'-'.join(map(str, widths))}-bn-relu"
           f"__rf{rf0['arch_support_px']}px-r99win{rf0['total_r99_window_px']}px"
           f"__in-17L-9p5um-{a.norm}__train-crop{a.crop}__deploy-tile{R.TILE}-halo{halo}"
           f"__distill-{a.loss.replace('+', '-')}-{a.targets_desc}-2faces{'-hot' + str(a.hot_frac) if a.hot_frac else ''}"
           f"{'-tsharp' + a.target_sharpen.replace(':', 'x') if a.target_sharpen else ''}"
           f"{f'-outw{a.outside_w:g}-edgefrac{a.edge_frac:g}' if (a.outside_w or a.edge_frac) else ''}"
           f"{f'__aug-warp3d{a.warp3d:g}' + ('' if a.warp_mode == 'full' else '-' + a.warp_mode) if a.warp3d else ''}{f'-domain{a.domain_aug:g}' if a.domain_aug else ''}"
           f"__b{a.batch}-lr{a.lr:g}-{a.sched}{'-ema' + format(a.ema, 'g') if a.ema else ''}__steps{a.steps}")
    os.makedirs(a.out, exist_ok=True)
    # TB dir: the run name compacted under the 255-byte path-component limit (full name in split.json / TB text)
    tbname = (run.replace("unet2d-", "u").replace("-bn-relu", "").replace("__train-crop", "__crop")
              .replace("__deploy-tile2048", "").replace("__distill-", "__").replace("-2faces", "").replace("softbce", "sbce"))
    if len(tbname) > 240:
        tbname = tbname[:230] + "~" + hashlib.md5(run.encode()).hexdigest()[:8]
    tb = SummaryWriter(os.path.join(a.out, "tb", tbname))
    tb.add_text("config/run_name", run, 0)
    print(f"[distill] run {run}", flush=True)
    print(f"[distill] train {len(train)} segs ({tpx / 1e6:.0f} Mpx valid), val {len(val)}: "
          f"{[s['name'] for s in val]}", flush=True)
    json.dump({"run": run, "train": [s["name"] for s in train], "val": [s["name"] for s in val],
               "args": vars(a)}, open(os.path.join(a.out, "split.json"), "w"), indent=1)

    skip_off = None
    upscale = 1
    model = R.build(widths, in_ch).to(dev).to(memory_format=torch.channels_last)
    nparam = sum(p.numel() for p in model.parameters())
    rf = rf0
    print(f"[distill] params {nparam}, rf {rf}", flush=True)
    tb.add_text("config/rf_untrained", json.dumps(rf0), 0)
    step0 = 0
    if a.resume:
        rk = torch.load(a.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(rk["state_dict"]); step0 = int(rk["step"])
        print(f"[distill] resumed {a.resume} at step {step0}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    warm = a.warmup
    PL = {"scale": 1.0, "best": -1e9, "bad": 0, "red": 0, "converged": False, "history": []}
    if a.sched == "plateau":
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * PL["scale"])
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(
            # a resumed run gets a FRESH warm-up + cosine over its remaining steps (a fine-tune), not
            # the tail of the original schedule
            opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, a.steps - step0)))))
    scaler = torch.amp.GradScaler("cuda")
    import copy as _copy
    ema = None
    if a.ema:
        ema = _copy.deepcopy(model).eval()
        for p_ in ema.parameters():
            p_.requires_grad_(False)

    EMA_N = {"n": 0}

    def ema_update():
        # decay ramps up (1+n)/(10+n) -> a.ema so the average is not anchored on the random init; floating buffers
        # (BN running stats) are averaged with the weights they belong to, integer ones copied
        EMA_N["n"] += 1
        d = min(a.ema, (1 + EMA_N["n"]) / (10 + EMA_N["n"]))
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)
            for be, bm in zip(ema.buffers(), model.buffers()):
                if be.dtype.is_floating_point:
                    be.mul_(d).add_(bm.detach(), alpha=1 - d)
                else:
                    be.copy_(bm)

    RESUME = os.path.join(a.out, "resume.pt")
    resumed_best = None
    if a.auto_resume and os.path.exists(RESUME):
        rs = torch.load(RESUME, map_location="cpu", weights_only=False)
        model.load_state_dict(rs["model"]); opt.load_state_dict(rs["opt"]); sched.load_state_dict(rs["sched"])
        scaler.load_state_dict(rs["scaler"]); PL.update(rs["PL"]); step0 = int(rs["step"]); resumed_best = rs["best"]
        if ema is not None and rs.get("ema") is not None:
            ema.load_state_dict(rs["ema"])
        EMA_N["n"] = step0
        print(f"[distill] AUTO-RESUMED {RESUME} at step {step0} (plateau state {PL})", flush=True)

    # ---- training-time augmentation (GPU, per sample) ---------------------------------------------------------
    # Teacher targets were computed on the UN-warped stack. A warp moves the stack by d(z,y,x); the target and the
    # validity mask are pulled by the SAME in-plane displacement taken at the centre plane z0 (first-order: ink is a
    # property of the papyrus column, so depth-only components -- squash/stretch/pinch/bulge/z fields -- leave the
    # target unchanged). The resolution jitter is an in-plane zoom about the crop centre applied to x, target and mask.
    W3 = None
    if a.warp3d:
        sys.path.insert(0, os.path.abspath(a.warp_mod))
        import warp3d as W3                                               # noqa: E402
        _wsrc = os.path.join(os.path.abspath(a.warp_mod), "warp3d.py")
        print(f"[distill] warp3d module {_wsrc} md5 {hashlib.md5(open(_wsrc, 'rb').read()).hexdigest()}", flush=True)
    E = a.aug_pad if (a.warp3d or a.domain_aug) else 0
    arng = np.random.default_rng(a.seed + 7777)

    def _gblur(t, sig, axis):
        """t (N,C,H,W) blur along axis 2 (H) / 3 (W) / 1 (C = depth) with a Gaussian of sigma px."""
        k = int(2 * math.ceil(2.5 * sig) + 1)
        ax = torch.arange(k, device=t.device, dtype=torch.float32) - k // 2
        g = torch.exp(-ax ** 2 / (2 * sig * sig)); g = g / g.sum()
        if axis == 1:
            n, c, h, w = t.shape
            u = t.permute(0, 2, 3, 1).reshape(-1, 1, c)
            u = F.conv1d(F.pad(u, (k // 2, k // 2), mode="replicate"), g.view(1, 1, k))
            return u.reshape(n, h, w, c).permute(0, 3, 1, 2)
        C = t.shape[1]
        if axis == 2:
            return F.conv2d(F.pad(t, (0, 0, k // 2, k // 2), mode="replicate"), g.view(1, 1, k, 1).expand(C, 1, k, 1), groups=C)
        return F.conv2d(F.pad(t, (k // 2, k // 2, 0, 0), mode="replicate"), g.view(1, 1, 1, k).expand(C, 1, 1, k), groups=C)

    AUGLOG = {"n": 0, "warp": 0, "dom": 0}

    def _sfield(n, ch, Z, H, W, L, Lz, dev, rng):
        """n x ch independent unit-rms smooth fields (n,ch,Z,H,W): iid N(0,1) on a control grid of spacing L px /
        Lz planes, trilinear (align_corners) upsampled, rms-normalised per field -- warp3d.smooth_field's
        construction, batched (Lz None: constant along z)."""
        gh, gw = max(2, int(math.ceil(H / L)) + 1), max(2, int(math.ceil(W / L)) + 1)
        gz = 1 if Lz is None else max(2, int(math.ceil(Z / Lz)) + 1)
        g = torch.from_numpy(rng.standard_normal((n, ch, gz, gh, gw)).astype(np.float32)).to(dev)
        if Lz is None:
            f = F.interpolate(g[:, :, 0], size=(H, W), mode="bilinear", align_corners=True)[:, :, None]
        else:
            f = F.interpolate(g, size=(Z, H, W), mode="trilinear", align_corners=True)
        return f / (f.pow(2).mean((2, 3, 4), keepdim=True).sqrt() + 1e-9)

    def augment(xb, t, w, rng, chunk=4):
        """Batched (chunks of `chunk`) 3-D warp + domain jitter. xb (B,17,Hx,Wx) raw grey 0..255 with halo HL+E;
        t, w (B,P+2E,P+2E). Each sample draws its own fields and amplitudes; correlation lengths are drawn once per
        chunk. Returns x cropped to halo HL and t, w to P."""
        B, Z, Hx, Wx = xb.shape
        Ht = t.shape[1]
        dev = xb.device
        xo, to, wo = [], [], []
        zz = (torch.arange(Z, device=dev, dtype=torch.float32) - (Z - 1) / 2).view(1, Z, 1, 1)
        yy = torch.arange(Hx, device=dev, dtype=torch.float32).view(1, 1, Hx, 1)
        xx = torch.arange(Wx, device=dev, dtype=torch.float32).view(1, 1, 1, Wx)
        for c0 in range(0, B, chunk):
            xc, tc, wc = xb[c0:c0 + chunk].float(), t[c0:c0 + chunk], w[c0:c0 + chunk]
            n = xc.shape[0]
            dw = torch.from_numpy((rng.random(n) < a.warp3d).astype(np.float32)).to(dev).view(n, 1, 1, 1)
            dd = (rng.random(n) < a.domain_aug) if a.domain_aug else np.zeros(n, bool)
            AUGLOG["n"] += n; AUGLOG["warp"] += int(dw.sum()); AUGLOG["dom"] += int(dd.sum())
            # displacement fields are built on a 4x coarser in-plane grid (the finest correlation length is 16 px)
            # and bilinearly upsampled ONCE -- the full-resolution build cost ~4x the training step on a shared card
            Hl, Wl = (Hx - 1) // 4 + 1 + (1 if (Hx - 1) % 4 else 0), (Wx - 1) // 4 + 1 + (1 if (Wx - 1) % 4 else 0)
            d3 = torch.zeros((n, 3, Z, Hl, Wl), device=dev)                                   # (dy, dx, dz) coarse grid
            if a.warp3d and float(dw.sum()) > 0:
                U = lambda lo, hi: torch.from_numpy(rng.uniform(lo, hi, n).astype(np.float32)).to(dev).view(n, 1, 1, 1)
                fc = _sfield(n, 3, Z, Hl, Wl, rng.uniform(256, 512) / 4, rng.uniform(4, 8), dev, rng)  # coarse y, x, z
                ff = _sfield(n, 3, Z, Hl, Wl, rng.uniform(16, 64) / 4, rng.uniform(1, 3), dev, rng)   # fine y, x, z
                st = _sfield(n, 3, 1, Hl, Wl, rng.uniform(256, 512) / 4, None, dev, rng)[:, :, 0]      # s(y,x)
                if a.warp_mode == "inplane":
                    ac = U(0, 2)
                    d3[:, 0] = dw * ac * fc[:, 0, Z // 2:Z // 2 + 1]      # one in-plane field for every layer
                    d3[:, 1] = dw * ac * fc[:, 1, Z // 2:Z // 2 + 1]
                else:
                    ac, af = U(0, 3), U(0, 1.5)
                    if a.warp_mode == "full":
                        d3[:, 0] = dw * (ac * fc[:, 0] + af * ff[:, 0])
                        d3[:, 1] = dw * (ac * fc[:, 1] + af * ff[:, 1])
                    d3[:, 2] = dw * (U(0, 2) * fc[:, 2] + U(0, 1) * ff[:, 2]
                                     + U(0.1, 0.3) * st[:, 0:1] * zz                               # stretch / squash
                                     + U(0, 0.3) * st[:, 1:2].abs() * zz * torch.exp(-zz ** 2 / (2 * W3.PINCH_W ** 2))  # pinch
                                     + U(0, 1) * st[:, 2:3] * torch.tanh(zz / W3.BULGE_W))         # bulge
                del fc, ff, st
            d3 = F.interpolate(d3.view(n, 3 * Z, Hl, Wl), size=(4 * (Hl - 1) + 1, 4 * (Wl - 1) + 1), mode="bilinear",
                               align_corners=True)[:, :, :Hx, :Wx].reshape(n, 3, Z, Hx, Wx)
            dy, dx, dz = d3[:, 0], d3[:, 1], d3[:, 2]
            sc = np.where(dd, np.exp(rng.uniform(np.log(0.8), np.log(1.25), n)), 1.0).astype(np.float32)
            sct = torch.from_numpy(sc - 1.0).to(dev).view(n, 1, 1, 1)
            dy = dy + sct * (yy - (Hx - 1) / 2); dx = dx + sct * (xx - (Wx - 1) / 2)
            # x: out(p) = x(p + d(p)), trilinear, border padding (warp3d.warp_stack's convention), batched
            gx = (xx + dx) * (2.0 / (Wx - 1)) - 1; gy = (yy + dy) * (2.0 / (Hx - 1)) - 1
            gz = (zz + (Z - 1) / 2 + dz) * (2.0 / (Z - 1)) - 1
            xw = F.grid_sample(xc[:, None], torch.stack([gx, gy, gz], -1), mode="bilinear", padding_mode="border",
                               align_corners=True)[:, 0]
            del gx, gy, gz
            # target + mask: the same in-plane displacement at the centre plane, in the target frame (x frame - HL)
            k = Z // 2
            tdy = dy[:, k, HL:HL + Ht, HL:HL + Ht]; tdx = dx[:, k, HL:HL + Ht, HL:HL + Ht]
            ty = torch.arange(Ht, device=dev, dtype=torch.float32).view(1, Ht, 1) + tdy
            tx = torch.arange(Ht, device=dev, dtype=torch.float32).view(1, 1, Ht) + tdx
            g2 = torch.stack([tx * (2.0 / (Ht - 1)) - 1, ty * (2.0 / (Ht - 1)) - 1], -1)
            tw = F.grid_sample(tc[:, None], g2, mode="bilinear", padding_mode="zeros", align_corners=True)[:, 0]
            ww = (F.grid_sample(wc[:, None], g2, mode="bilinear", padding_mode="zeros", align_corners=True)[:, 0] > 0.5).float()
            del dz, dy, dx
            if dd.any():
                ii = torch.from_numpy(np.nonzero(dd)[0]).to(dev); k_ = len(ii)
                V = lambda lo, hi: torch.from_numpy(rng.uniform(lo, hi, k_).astype(np.float32)).to(dev).view(k_, 1, 1, 1)
                xi = xw[ii]
                vm = (xi > 0.5).float()
                xi = 255.0 * (xi.clamp(0, 255) / 255.0) ** torch.exp(V(np.log(0.7), np.log(1.4)))     # gamma
                m = xi.mean((1, 2, 3), keepdim=True)
                xi = (xi - m) * V(0.8, 1.25) + m + V(-10, 10)                                         # contrast, offset
                for j in range(k_):
                    if rng.random() < 0.5:                                                           # in-plane blur
                        s_ = rng.uniform(0.3, 1.5)
                        xi[j] = _gblur(_gblur(xi[j:j + 1], s_, 2), s_, 3)[0]
                    if rng.random() < 0.3:                                                           # depth blur
                        xi[j] = _gblur(xi[j:j + 1], rng.uniform(0.3, 1.0), 1)[0]
                xi = xi + torch.randn_like(xi) * V(0, 6)                                              # noise
                xw[ii] = xi.clamp(1, 255) * vm                       # the unrendered region stays exactly 0
            xo.append(xw); to.append(tw); wo.append(ww)
        xo = torch.cat(xo); to = torch.cat(to); wo = torch.cat(wo)
        if E:
            xo = xo[:, :, E:-E, E:-E]; to = to[:, E:-E, E:-E]; wo = wo[:, E:-E, E:-E]
        return xo.contiguous(), to.contiguous(), wo.contiguous()

    def _pull2d(img, dy, dx):
        H_, W_ = img.shape
        ys = torch.arange(H_, device=img.device, dtype=torch.float32).view(H_, 1) + dy
        xs = torch.arange(W_, device=img.device, dtype=torch.float32).view(1, W_) + dx
        g = torch.stack([xs * (2.0 / max(W_ - 1, 1)) - 1, ys * (2.0 / max(H_ - 1, 1)) - 1], -1)[None]
        return F.grid_sample(img[None, None], g, mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0]

    def _warp2d_stack(v, dy, dx):
        return torch.stack([_pull2d(v[j], dy, dx) for j in range(v.shape[0])])

    # ---- sampler: crops weighted by valid area, centre on the surface --------------------------
    P = a.crop
    WT = {"w": None}

    def reweight():
        # SCROLL-BALANCED: each scroll gets weight sqrt(its surface area), split over its
        # segments by area -- the render backlog is dominated by two scrolls, and an
        # area-proportional sampler would spend most steps on them.
        area = np.array([s["valid"].sum() for s in train], np.float64)
        sc = [s["name"].split("_")[0] for s in train]
        tot = {k: sum(a_ for a_, k2 in zip(area, sc) if k2 == k) for k in set(sc)}
        w = np.array([a_ / tot[k] * np.sqrt(tot[k]) for a_, k in zip(area, sc)])
        WT["w"] = w / w.sum()
        print("[distill] sampling share by scroll: " + ", ".join(
            f"{k} {sum(w_ for w_, k2 in zip(WT['w'], sc) if k2 == k):.3f}" for k in sorted(tot)), flush=True)
    reweight()

    def one(r):
        for _ in range(20):
            tr, w = train, WT["w"]
            if len(w) != len(tr):
                continue
            s = tr[r.choice(len(tr), p=w)]
            H, W = s["valid"].shape
            if H < P or W < P:
                continue
            edge_crop = False
            if a.hot_frac and len(s["hot"]) and r.random() < a.hot_frac:
                cy, cx = s["hot"][r.integers(len(s["hot"]))] + r.integers(-P // 3, P // 3 + 1, 2)
                y0 = int(min(max(0, cy - P // 2), H - P)); x0 = int(min(max(0, cx - P // 2), W - P))
            elif a.edge_frac and len(s["edge"]) and r.random() < a.edge_frac:
                cy, cx = s["edge"][r.integers(len(s["edge"]))] + r.integers(-P // 4, P // 4 + 1, 2)
                y0 = int(min(max(0, cy - P // 2), H - P)); x0 = int(min(max(0, cx - P // 2), W - P))
                edge_crop = True
            else:
                y0 = int(r.integers(0, H - P + 1)); x0 = int(r.integers(0, W - P + 1))
            v = s["valid"][y0:y0 + P, x0:x0 + P]
            if edge_crop:
                if v.mean() < 0.2:
                    continue
            elif v[P // 2 - 32:P // 2 + 32, P // 2 - 32:P // 2 + 32].mean() < 0.5:
                continue
            face = "rev" if r.random() < 0.5 else "fwd"
            # read a HALO of LOCAL_WIN//2 (+ E augmentation pad) so the local statistics are computed as at inference
            H_, W_ = s["valid"].shape
            G = HL + E
            ya, xa = max(0, y0 - G), max(0, x0 - G)
            yb, xb = min(H_, y0 + P + G), min(W_, x0 + P + G)
            x = np.zeros((17, P + 2 * G, P + 2 * G), np.uint8)
            x[:, ya - (y0 - G):yb - (y0 - G), xa - (x0 - G):xb - (x0 - G)] = s["x"][:, ya:yb, xa:xb]
            if ya > y0 - G: x[:, :ya - (y0 - G)] = x[:, ya - (y0 - G):ya - (y0 - G) + 1]
            if yb < y0 + P + G: x[:, yb - (y0 - G):] = x[:, yb - (y0 - G) - 1:yb - (y0 - G)]
            if xa > x0 - G: x[:, :, :xa - (x0 - G)] = x[:, :, xa - (x0 - G):xa - (x0 - G) + 1]
            if xb < x0 + P + G: x[:, :, xb - (x0 - G):] = x[:, :, xb - (x0 - G) - 1:xb - (x0 - G)]
            if face == "rev":
                x = x[::-1]
            if E:
                t = np.zeros((P + 2 * E, P + 2 * E), np.uint8); v = np.zeros((P + 2 * E, P + 2 * E), bool)
                ya, xa = max(0, y0 - E), max(0, x0 - E); yb, xb = min(H_, y0 + P + E), min(W_, x0 + P + E)
                t[ya - (y0 - E):yb - (y0 - E), xa - (x0 - E):xb - (x0 - E)] = s["t"][face][ya:yb, xa:xb]
                v[ya - (y0 - E):yb - (y0 - E), xa - (x0 - E):xb - (x0 - E)] = s["valid"][ya:yb, xa:xb]
            else:
                t = s["t"][face][y0:y0 + P, x0:x0 + P]
            st = s["st"]
            return x, t, v, np.array([st["lo"], st["hi"], st["med"], st["mad"]], np.float32)
        raise RuntimeError("could not sample a crop")

    q = queue.Queue(maxsize=8)

    def worker(k):
        r = np.random.default_rng(a.seed * 1000 + k)
        while True:
            items = [one(r) for _ in range(a.batch)]
            xb = np.stack([i[0] for i in items]); tb_ = np.stack([i[1] for i in items])
            vb = np.stack([i[2] for i in items]); sb = np.stack([i[3] for i in items])
            q.put((torch.from_numpy(np.ascontiguousarray(xb)).pin_memory(),
                   torch.from_numpy(tb_).pin_memory(), torch.from_numpy(vb).pin_memory(),
                   torch.from_numpy(sb)))

    for k in range(a.workers):
        threading.Thread(target=worker, args=(k,), daemon=True).start()

    # fixed held-out crops for images
    vis = []
    for s in val[:6]:
        H, W = s["valid"].shape
        ys, xs = np.nonzero(s["valid"][::64, ::64])
        k = len(ys) // 2
        cy, cx = min(max(ys[k] * 64, 384), H - 384), min(max(xs[k] * 64, 384), W - 384)
        vis.append((s, cy - 384, cx - 384))

    LV = None
    if a.label_subj:
        sys.path.insert(0, HERE)
        import rv2_lib as L
        LV = L.load_subjects(a.label_subj, S.select_layers)
        LVG = sorted({v["group"] for v in LV})
        LVIS = {g: next(v for v in LV if v["group"] == g) for g in LVG}       # one fixed tile per group
        print(f"[distill] label-val: {len(LV)} tiles, groups {LVG}", flush=True)

    def label_val(m, step):
        """Per group: near64 AUC + BCE vs the binary label, oracle face per tile; one image per group."""
        import torch as _t
        acc = {}
        for v in LV:
            best = None
            for face in ("fwd", "rev"):
                xx = v["x"] if face == "fwd" else v["x"][::-1]
                lg_ = R.predict_array(m, [xx[j] for j in range(17)], a.norm, dev, tile=a.val_tile, halo=halo)
                p_ = 1 / (1 + np.exp(-lg_))
                au = L.auc(p_[v["near"]], v["ink"][v["near"]])
                if best is None or au > best[0]:
                    best = (au, p_)
            au, p_ = best
            y = v["ink"][v["near"]].astype(np.float32); q = np.clip(p_[v["near"]], 1e-6, 1 - 1e-6)
            bce = float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())
            acc.setdefault(v["group"], []).append((au, bce))
            if LVIS.get(v["group"]) is v:
                ct = v["x"][8].astype(np.float32) / 255.0
                im = np.concatenate([ct, v["ink"].astype(np.float32) * (0.5 + 0.5 * v["near"]), p_], 1)
                tb.add_image(f"label_tiles/{v['group']}/{v['name']}/CT|label(near64 bright)|pred", im[None], step)
        out = {}
        for g, vals in acc.items():
            out[f"label_auc_near64/{g}"] = float(np.mean([u[0] for u in vals]))
            out[f"label_bce_near64/{g}"] = float(np.mean([u[1] for u in vals]))
        for k, val in out.items():
            tb.add_scalar(k, val, step)
        print(f"[distill] step {step} label-val (logged only) " +
              " ".join(f"{k.split('/')[1][:14]}={val:.3f}" for k, val in sorted(out.items()) if "auc" in k), flush=True)
        return out

    def validate(step):
        model.eval()
        from vesuvius_pipeline.stages.ink_models.ink9um_student import fold_bn
        import copy
        m = fold_bn(copy.deepcopy(ema if ema is not None else model)).eval().half().to(memory_format=torch.channels_last)
        res = {}
        pooled = {"fwd": [], "rev": []}
        vox = 0; gs = 0.0
        for s in val:
            arrs = [s["x"][j] for j in range(17)]
            for face in ("fwd", "rev"):
                aa = arrs if face == "fwd" else arrs[::-1]
                torch.cuda.synchronize(); t0 = time.time()
                lg = R.predict_array(m, aa, a.norm, dev, tile=a.val_tile, halo=halo)
                torch.cuda.synchronize(); gs += time.time() - t0
                vox += lg.size * 17
                p = 1 / (1 + np.exp(-lg))
                t = (s["t"][face].astype(np.float32) + 0.5) / 255.0
                v = s["valid"]
                r = pearson(p[v], t[v]); mae = float(np.abs(p[v] - t[v]).mean())
                res[f"{s['name']}/{face}"] = {"r": r, "mae": mae}
                tb.add_scalar(f"val_r/{face}/{s['name']}", r, step)
                tb.add_scalar(f"val_mae/{face}/{s['name']}", mae, step)
                pooled[face].append(r)
        for face, rs in pooled.items():
            tb.add_scalar(f"val_r_mean/{face}", float(np.mean(rs)), step)
        tb.add_scalar("throughput/val_infer_MVox_per_s", vox / 1e6 / max(gs, 1e-9), step)
        for s, y0, x0 in vis:
            arrs = [np.asarray(s["x"][j][y0:y0 + 768, x0:x0 + 768]) for j in range(17)]
            lg = R.predict_array(m, arrs, a.norm, dev, tile=a.val_tile, halo=halo)
            p = 1 / (1 + np.exp(-lg))
            ct = arrs[8].astype(np.float32) / 255.0
            t = s["t"]["fwd"][y0:y0 + 768, x0:x0 + 768].astype(np.float32) / 255.0
            im = np.concatenate([ct, t, p], 1)
            tb.add_image(f"heldout/{s['name']}/CT|teacher|student", im[None], step)
        if LV is not None:
            res["_label_val"] = label_val(m, step)
        model.train()
        mr = {f: float(np.mean(v)) for f, v in pooled.items()}
        print(f"[distill] step {step} val r fwd {mr['fwd']:.4f} rev {mr['rev']:.4f} "
              f"infer {vox / 1e6 / max(gs, 1e-9):.0f} MVox/s", flush=True)
        return res, mr

    def save(step, res, mr, tag=""):
        dep = ema if ema is not None else model
        ck = {"state_dict": dep.state_dict(), "step": step,
              "raw_state_dict": model.state_dict() if ema is not None else None, "ema_decay": a.ema or None,
              "sched": a.sched, "plateau": {k: v for k, v in PL.items()}, "aug": {"warp3d": a.warp3d, "warp_mode": a.warp_mode,
              "domain_aug": a.domain_aug, "outside_w": a.outside_w, "edge_frac": a.edge_frac, "aug_pad": E},
              "config": {"widths": list(widths), "blocks": None, "in_ch": in_ch, "crop": P, "tile": R.TILE, "halo": halo,
                         "norm": a.norm, "skip_off": list(skip_off) if skip_off else None, "upscale": upscale},
              "rf": rf, "rf_total": R.measure_total_rf(dep, a.norm), "params": nparam, "run": run,
              "family": "reader_v2_dense", "loss": a.loss, "hot_frac": a.hot_frac, "target_sharpen": a.target_sharpen,
              "teacher": a.teacher_id, "targets_desc": a.targets_desc,
              "train": [s["name"] for s in train], "val": [s["name"] for s in val],
              "val_result": res, "val_r_mean": mr}
        torch.save(ck, os.path.join(a.out, f"student{tag}.ckpt"))

    model.train()
    best = resumed_best or {"metric": -1.0, "step": -1}
    t_last = time.time(); vox_acc = 0
    step = step0
    for step in range(step0 + 1, a.steps + 1):
        xb, tb_, vb, sb = q.get()
        xb = xb.to(dev, non_blocking=True).float()
        sb = sb.to(dev)
        t = (tb_.to(dev, non_blocking=True).float() + 0.5) / 255.0
        w = vb.to(dev, non_blocking=True).float()
        if E:
            # on the GPU (2026-10-06 measured on the shared V100: 194 ms per batch of 32 for warp + domain with the
            # coarse-grid fields; the CPU-thread variant ran < 0.33 step/s at load 61/64 and hit the 70 GB scope cap)
            xb, t, w = augment(xb, t, w, arng, chunk=16)
        if a.outside_w:
            # unrendered pixels (middle layer exactly 0 -- NOT the lossmask cut-outs, which stay unsupervised):
            # target 0 (the teacher's own map is 0 there), weight outside_w
            outside = (xb[:, 8, HL:-HL, HL:-HL] == 0).float()
            t = t * (1 - outside) + (0.5 / 255.0) * outside
            w = torch.maximum(w, a.outside_w * outside)
        xb = R.apply_norm(xb, a.norm)[:, :, HL:-HL, HL:-HL].contiguous(memory_format=torch.channels_last)
        if a.target_sharpen:
            _sg, _am = (float(v) for v in a.target_sharpen.split(":"))
            _k = int(2 * round(3 * _sg) + 1)
            _ax = torch.arange(_k, device=dev, dtype=torch.float32) - _k // 2
            _g = torch.exp(-_ax ** 2 / (2 * _sg ** 2)); _g = _g / _g.sum()
            _tb = F.conv2d(F.pad(t[:, None], (_k // 2, _k // 2, 0, 0), mode="replicate"), _g.view(1, 1, 1, _k))
            _tb = F.conv2d(F.pad(_tb, (0, 0, _k // 2, _k // 2), mode="replicate"), _g.view(1, 1, _k, 1))[:, 0]
            t = (t + _am * (t - _tb)).clamp(0.0, 1.0)
        M = a.margin
        with torch.autocast("cuda", dtype=torch.float16):
            lg = model(xb)[:, 0]
        lg = lg.float()
        if upscale > 1:
            # ablation C (PixelShuffle head): average the finer-grid logits back to the native
            # grid before the loss, so supervision is IDENTICAL to upscale=1 -- the net is free
            # to place sub-native structure, but nothing at the native resolution changes
            lg = F.avg_pool2d(lg[:, None], upscale)[:, 0]
        lg = lg[:, M:-M, M:-M]; t = t[:, M:-M, M:-M]; w = w[:, M:-M, M:-M]
        loss = (F.binary_cross_entropy_with_logits(lg, t, reduction="none") * w).sum() / w.sum().clamp(min=1)
        if a.loss in ("softbce+logit", "softbce+logit+edge"):
            tl = torch.logit(t.clamp(0.5 / 255, 1 - 0.5 / 255))
            loss = loss + 0.1 * (((lg - tl) ** 2) * w).sum() / w.sum().clamp(min=1)
            if a.loss == "softbce+logit+edge":
                loss = loss + 1.0 * _edge_l1(torch.sigmoid(lg), t, w)
        elif a.loss == "softbce+edge":
            p = torch.sigmoid(lg)
            loss = loss + 1.0 * _edge_l1(p, t, w)
        elif a.loss == "softbce+dice":
            p = torch.sigmoid(lg)
            inter = (p * t * w).sum(); denom = ((p + t) * w).sum().clamp(min=1e-6)
            loss = loss + 1.0 * (1.0 - 2.0 * inter / denom)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        if ema is not None:
            ema_update()
        vox_acc += xb.numel()
        if step % 50 == 0:
            with torch.no_grad():
                p = torch.sigmoid(lg)
                mse = float(((p - t) ** 2 * w).sum() / w.sum().clamp(min=1))
            dt = time.time() - t_last
            tb.add_scalar("train/loss_softbce", float(loss.detach()), step)
            tb.add_scalar("train/mse_prob", mse, step)
            tb.add_scalar("train/lr", sched.get_last_lr()[0], step)
            tb.add_scalar("throughput/train_MVox_per_s", vox_acc / 1e6 / dt, step)
            if step % 200 == 0:
                print(f"[distill] step {step} loss {float(loss):.4f} mse {mse:.5f} "
                      f"{vox_acc / 1e6 / dt:.0f} MVox/s q={q.qsize()}", flush=True)
            t_last = time.time(); vox_acc = 0
        if a.rescan_every and step % a.rescan_every == 0:
            new = [d for d in sorted(glob.glob(os.path.join(a.data, "*")))
                   if os.path.exists(os.path.join(d, "meta.json")) and os.path.basename(d) not in known
                   and os.path.basename(d) not in excl]
            for d in new:
                known.add(os.path.basename(d))
                if not is_val(os.path.basename(d)):
                    try:
                        s_ = prep(d)
                    except Exception as e:                  # noqa: BLE001 - a half-written segment
                        known.discard(os.path.basename(d)); print(f"[distill] skip {d}: {e}", flush=True)
                        continue
                    train = train + [s_]
            if new:
                reweight()
                tb.add_scalar("data/train_segments", len(train), step)
                tb.add_scalar("data/train_valid_Mpx", sum(int(s["valid"].sum()) for s in train) / 1e6, step)
                print(f"[distill] step {step}: train now {len(train)} segments", flush=True)
        if E and step % 2000 == 0:
            with torch.no_grad():
                xm = xb[0, 8, M:-M, M:-M].float(); xm = (xm - xm.min()) / (xm.max() - xm.min() + 1e-6)
                tb.add_image("train_aug/normalised-mid|target|weight",
                             torch.cat([xm, t[0], w[0].clamp(0, 1)], 1)[None].cpu(), step)
                tb.add_scalar("train_aug/frac_warped", AUGLOG["warp"] / max(1, AUGLOG["n"]), step)
                tb.add_scalar("train_aug/frac_domain", AUGLOG["dom"] / max(1, AUGLOG["n"]), step)
        if step % a.val_every == 0 or step == a.steps:
            res, mr = validate(step)
            save(step, res, mr, tag=f"_step{step}")
            save(step, res, mr)       # "student.ckpt" -- always the LATEST, unchanged behaviour
            metric = float(np.mean([mr["fwd"], mr["rev"]]))  # pooled val Pearson r vs the teacher
            tb.add_scalar("val_r_mean/pooled", metric, step)
            if metric > best["metric"]:
                best = {"metric": metric, "step": step}
                save(step, res, mr, tag="_best")   # "student_best.ckpt" -- BEST by validation, not last
                print(f"[distill] step {step}: NEW BEST val_r_mean {metric:.4f} -> student_best.ckpt", flush=True)
            if a.sched == "plateau":
                lv = res.get("_label_val") or {}
                PL["history"].append({"step": step, "val_r": metric, "lr": sched.get_last_lr()[0],
                                      "label_auc_mean": float(np.mean([v for k, v in lv.items() if "auc" in k]))
                                      if lv else None})
                if metric > PL["best"] + a.plateau_eps:
                    PL["best"] = metric; PL["bad"] = 0
                else:
                    PL["bad"] += 1
                if PL["bad"] >= a.plateau_patience and step >= a.min_steps:
                    if PL["red"] < a.plateau_max_red:
                        PL["red"] += 1; PL["scale"] *= a.plateau_factor; PL["bad"] = 0
                        print(f"[distill] step {step}: PLATEAU (val r {metric:.4f}, best {PL['best']:.4f}) -> lr x"
                              f"{a.plateau_factor} (reduction {PL['red']}/{a.plateau_max_red})", flush=True)
                    else:
                        PL["converged"] = True
                tb.add_scalar("plateau/bad_validations", PL["bad"], step)
                tb.add_scalar("plateau/lr_scale", PL["scale"], step)
            torch.save({"model": model.state_dict(), "ema": ema.state_dict() if ema is not None else None,
                        "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                        "PL": PL, "step": step, "best": best}, RESUME + ".tmp")
            os.replace(RESUME + ".tmp", RESUME)
            if PL["converged"]:
                print(f"[distill] step {step}: CONVERGED -- no val-r gain > {a.plateau_eps} for {a.plateau_patience} "
                      f"validations after {PL['red']} lr reductions", flush=True)
                break
    json.dump({"best": best, "last_step": step, "plateau": PL, "converged": PL["converged"]},
              open(os.path.join(a.out, "convergence.json"), "w"), indent=1)
    print(f"[distill] final: best val_r_mean {best['metric']:.4f} at step {best['step']} "
          f"(student_best.ckpt); last at step {step} (student.ckpt); converged={PL['converged']}", flush=True)
    tb.close()


if __name__ == "__main__":
    main()
