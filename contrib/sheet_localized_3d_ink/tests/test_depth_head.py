"""Unit tests (synthetic tensors) for the target construction: correctness of the masking and the film restriction.
Real-data evidence is in the README, not here."""
import torch
from contrib.sheet_localized_3d_ink.depth_head import snap_targets, depth_loss


def _teacher(peak_plane: int, K: int = 16, H: int = 4, W: int = 4):
    t = torch.zeros(1, K, H, W)
    t[:, peak_plane] = 1.0
    t[:, peak_plane + 5 if peak_plane + 5 < K else peak_plane - 5] = 0.9            # a second sheet far away: must NOT become a target
    return t


def test_film_excludes_other_sheet():
    t3, m = snap_targets(torch.ones(1, 4, 4), torch.ones(1, 4, 4), _teacher(4), film=1)
    assert t3[0, 3:6, 0, 0].tolist() == [0.0, 1.0, 0.0]                           # only the teacher's own plane inside the film is ink
    assert t3[0, 9, 0, 0].item() == 0.0                                            # the far sheet (plane 9) is excluded
    assert t3[0, 4, 0, 0].item() == 1.0                                            # the peak plane is the target


def test_label_zero_gives_zero_everywhere_and_mask_is_judged():
    j = torch.zeros(1, 4, 4); j[:, :2] = 1
    t3, m = snap_targets(torch.zeros(1, 4, 4), j, _teacher(4))
    assert t3.sum().item() == 0.0 and m.sum().item() == 8


def test_silent_teacher_falls_back_to_centre_film():
    t3, _ = snap_targets(torch.ones(1, 4, 4), torch.ones(1, 4, 4), torch.zeros(1, 16, 4, 4))
    assert t3[0, :, 0, 0].nonzero().flatten().tolist() == [6, 7, 8]


def test_loss_ignores_unjudged_pixels():
    logits = torch.zeros(1, 16, 4, 4); t3 = torch.zeros(1, 16, 4, 4); m = torch.zeros(1, 4, 4)
    assert depth_loss(logits, t3, m).item() == 0.0
    m[:, 0, 0] = 1
    assert abs(depth_loss(logits, t3, m).item() - 0.6931) < 1e-3
