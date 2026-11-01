"""Sheet-localised depth head for a 2-D ink U-Net with a 3-D depth-fusion stem (reader-v3 style).

The 2-D ink model answers "is there ink at this (row, col)?".  A depth head additionally answers "at which of the K planes of
the input window, and on which sheet?".  It is a 1x1 convolution on the finest decoder stage that outputs K logits per pixel
(one per input plane, K = 16 in our runs: the central 16 of the 17 planes, 9.6 um apart).

Supervision needs no 3-D human labels: the target is the 2-D label snapped onto the ink-bearing sheet,
    t3[k, y, x] = label[y, x] * profile[k, y, x]
where `profile` is a teacher's depth profile (e.g. Hecate's 3-D probability volume) normalised to its peak and restricted to
+- `film` planes around that peak (so ink on the neighbouring sheets is NOT a target), and a centred film of 3 planes where the teacher is silent.
Pixels outside the judged region are masked; confident negatives get 0 in every plane.

Only torch is required.  Nothing here knows about a particular network: `attach_depth_head` hooks the last decoder stage.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def snap_targets(label: torch.Tensor, judged: torch.Tensor, teacher: torch.Tensor, *, film: int = 1, hmin: float = 0.3,
                 centre: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    """label (B,H,W) in {0,1}; judged (B,H,W) in {0,1}; teacher (B,K,H,W) uint8 or float in [0,1] (Hecate-style 3-D probability).
    Returns (t3 (B,K,H,W), mask (B,H,W)).  `film` = half width in planes of the allowed film around the teacher's peak plane."""
    hp = teacher.float()
    if teacher.dtype == torch.uint8:
        hp = hp / 255.0
    K = hp.shape[1]
    pk = hp.amax(1, keepdim=True)
    am = hp.argmax(1, keepdim=True)
    idx = torch.arange(K, device=hp.device).view(1, K, 1, 1)
    near = ((idx - am).abs() <= film).float()
    prof = torch.where(pk >= hmin, hp / pk.clamp(min=1e-6) * near, torch.zeros_like(hp))
    fallback = torch.zeros_like(hp)
    fallback[:, max(0, centre - 1):centre + 2] = 1.0                     # a 3-plane film at the centre where the teacher is silent
    prof = torch.where(pk >= hmin, prof, fallback)
    return label.float()[:, None] * prof, judged.float()


def depth_loss(logits: torch.Tensor, t3: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked BCE over (B,K,H,W) depth logits; mask (B,H,W)."""
    bce = F.binary_cross_entropy_with_logits(logits, t3, reduction="none")
    return (bce * mask[:, None]).sum() / (mask.sum() * logits.shape[1]).clamp(min=1.0)


class DepthHead(nn.Module):
    """1x1 conv on the finest decoder stage; captures that stage's output with a forward hook."""

    def __init__(self, decoder_stage: nn.Module, channels: int, planes: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(channels, planes, 1)
        self.feat: torch.Tensor | None = None
        decoder_stage.register_forward_hook(lambda _m, _i, out: setattr(self, "feat", out))

    def logits(self) -> torch.Tensor:
        assert self.feat is not None, "run the network forward first"
        out = self.conv(self.feat.float())
        self.feat = None
        return out


def attach_depth_head(net: nn.Module, planes: int = 16) -> DepthHead:
    """reader-v3 layout: net.model.network.shared_decoder.stages[-1] is the finest decoder stage."""
    dec = net.model.network.shared_decoder
    ch = int(list(dec.decoder_stage_output_channels)[-1])
    return DepthHead(dec.stages[len(dec.stages) - 1], ch, planes)
