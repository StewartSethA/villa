"""The finish stage's ink-family registry -- ONE row per family, and nothing heavy.

This module is imported by `vesuvius_pipeline.settings`, which every scheduler pass and
every finish task loads, and by peers whose worker interpreter has no torch and no numpy.
So it holds the declaration and NOTHING that imports a model: `stages/ink_models/__init__`
re-exports it beside the modules that do the work.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InkFamily:
    """One ink prediction family the FINISH stage may run, declared once, here.

    Adding a model to the production pipeline used to mean editing five places: a tuple in
    `settings.FINISH`, a tuple in `finish._zoo`, a membership test in `finish.variants`, a
    third in the availability loop and a fourth in the dispatch -- and the four tuples were
    spelled `("i3d", "resnet3d_1667")` independently, so a family added to three of them
    ran with no operator toggle or appeared in the UI and never executed. It is now a
    CHECKPOINT PLUS ONE ROW: this row names the module that owns the input contract, which
    exported arm of it to load, and what the operator's checkbox says.

    `settings.FINISH` grows a `finish.<name>_enabled` key from each row,
    `stages.ink.family_plan` reads that key, `variants()` expands the row into
    published family names, and the Workflow tab renders the checkbox from the server's
    key list -- so nothing below this file has to learn the new name.

    Fields:
      name    the published family name AND the settings-key stem. It becomes a PNG
              filename (`results/ink_detection/<seg>/<name>.png`), so keep it to
              [a-z0-9_]: no ':' -- the arm goes in `arm`.
      module  key into FAMILIES; the module holding SPEC, available() and predict().
      arm     the `family:arm` selector passed to available()/predict(); "" = the
              module's own default checkpoint.
      sides   run the verso face as `<name>_reversed` too. Which end of a rendered stack
              is the recto is a property of the render, and reading it backwards is worth
              ~0.13 AUC on these models (docs/wiki/ink_models.md).
      env     the LEGACY environment variable this key replaced, honoured loudly by
              settings.value() when it disagrees with the stored row. "" for a family
              that never had one -- a new family gets a DB key and no env var, because
              an env var on whichever box launched the worker is exactly the invisible,
              per-host, restart-losing knob the settings table exists to replace.
      default "0" -- an unvalidated family must not cost a GPU pass on every segment
              until an operator asks for it. Only a family with a published score on the
              labelled tiles may ship "1".
      multiscale  this family belongs to the BEST-GENERALISING SET and is run at every
              scale of `stages.ink.MULTISCALE` -- the render scale times 0.5, 0.67, 1,
              1.5, 2, so a 7.91 um/px row is read at 15.82 / 11.81 / 7.91 / 5.27 / 3.96
              um/px (`_s050` COARSEST, `_s200` finest) -- published as `<name>`,
              `<name>_s050`, ... so the gallery shows
              one card per physical scale. Gated by `finish.multiscale_enabled`. Set it
              only for a family whose CROSS-SCROLL held-out AUC is published: the bracket
              costs four extra RENDERS per segment, which is the expensive half, and
              those renders are SHARED by every multiscale family at that scale.
      um_per_px / layers / tile_px  the frame the checkpoint was TRAINED in, which
              `frame.MODELS` is built from. `tile_px` is the ML WINDOW -- the square of
              pixels the net sees at once -- because First Letters
              (scrollprize.org/prizes) rejects a window wider than 0.5 mm and the window
              is physical: tile_px x the um/px of the render the model actually read.
              Declare the MEASURED receptive field where one was measured, not the tile
              the inference loop happens to crop: a family with no entry here reads
              "window unknown" on every prediction and passes the prize check by default.
      publish_band  WHEN this family runs and publishes inside one finish. The finish
              stage publishes incrementally -- one card appears the moment its variant
              is done -- so this is what an operator sees FIRST on a segment nobody has
              looked at yet (`stages.ink.order_variants`, user 2026-09-15):
                0  the high-res / native-pitch read of the sheet
                1  the best CROSS-SCROLL generalisers: the untrained column scores,
                   whose numbers are transfer numbers by construction, and the smallest
                   bounded-window trained head
                2  everything else with a native-pitch pass (the default)
                4  UNSCORED and HEAVY: a full forward pass with no published
                   cross-scroll number to justify going first
              The bracket SCALES of any family always follow every band-0..2 pass: a
              second reading of a sheet nobody has seen is worth less than a first one,
              and each scale costs a render. Band 3 is reserved for them and is never
              set on a row.
              This lives HERE and not in a tuple in finish.py because a list of family
              names spelled twice is exactly what this registry exists to prevent
              (tests/test_ink_family_registry.py enforces it).
      label / help  what the Workflow tab's checkbox and its caption say. `help` must
              state what the family COSTS and what it is worth, so the operator can
              decide without reading this file.
    """
    name: str
    module: str
    arm: str = ""
    sides: bool = True
    env: str = ""
    default: str = "0"
    um_per_px: float = 7.91
    layers: int | None = None
    tile_px: int | None = None
    multiscale: bool = False
    hires2x: bool = False       # also read at 2x the render scale when finish.hires2x_enabled
    publish_band: int = 2
    label: str = ""
    help: str = ""
    # WITHDRAWN (user, 2026-10-01: "there are still degenerate/washed out pixelclf maps showing"): a non-empty
    # reason hides every map this family ever published from the ink galleries (nothing is deleted;
    # `?show_withdrawn=1` brings them back). Disabling a family stops NEW maps; this retires the old ones.
    withdrawn: str = ""
    # AUTO-ENABLE (user, 2026-10-02: "make sure they become enabled in the ink detection pipeline by default
    # once trained"): True = `maintenance.auto_enable_ink_winners` switches `finish.<name>_enabled` on, once per
    # checkpoint md5, the moment the family's checkpoint sits beside a `<ckpt>.gate.json` that says
    # {"pass": true, "ckpt_md5": <this checkpoint>} -- the sharpness-and-AUC gate scripts/hires_ink/sharp_gate.py
    # writes. No gate file, or a gate that fails, leaves the row at `default` ("0").
    auto_enable: bool = False
    # extra keyword arguments passed to the module's predict() for this row (pairs, hashable): e.g. (("blend_stride", 63), ("blend_patch", 128))
    predict_kw: tuple = ()

    @property
    def selector(self) -> str:
        """What `available()` and the dispatch take: 'pixelclf:32x13', or just 'i3d'."""
        return f"{self.module}:{self.arm}" if self.arm else self.module


# Every finish-stage ink family, in the order the Workflow tab lists them. A checkpoint on
# the host plus one row here is the WHOLE change (CLAUDE.md, "Adding an ink model").
# THE SHARP-STUDENT ABLATION ARMS (user, 2026-10-05: "all of the sharp student models"): the 12 short from-scratch runs of
# docs/experiments/sharp_student (2026-10-02) plus arm2, each a checkpoint of the SAME module and input contract as
# ink9um_student, selected by `arm` -> var/models/ink9um_student_<arm>.ckpt. OFF by default and not auto-enabled: measured
# (docs/experiments/sharp_student, AUC near-64, S5 unseen n=11 parents) all but arm2 (0.675) sit at 0.52-0.62 against the production
# student's 0.690, and no arm beat its own baseline beyond the CI. They exist so they can be RUN and LOOKED AT, not shipped.
_SHARP_ABL_ARMS = ("arch_3lvl", "arm1_edge", "arm1_edge_up2", "arm1_stride32", "base_repro", "loss_dice", "loss_edge",
                   "sharpen_2px", "sharpen_4px", "skipoff_01", "upscale2", "upscale4", "arm2")


def _sharp_abl_rows() -> tuple["InkFamily", ...]:
    return tuple(
        InkFamily(f"ink9um_student_abl_{a}", "ink9um_student", arm=f"abl_{a}", default="0",
                  um_per_px=9.5, layers=17, tile_px=431, publish_band=4,
                  label=f"ink9um_student ablation arm {a} (sharp_student study, 2026-10-02)",
                  help="One of the sharp_student ablation checkpoints (short from-scratch distillation runs, 1,500 steps; see "
                       "docs/experiments/sharp_student/STATE.md). Same 17-layer 9.5 um input as ink9um_student. NOT a better model: "
                       "on S5 unseen it scores 0.52-0.68 AUC against the production student's 0.69. Run it to look at its maps. "
                       "The declared 431 px window is the production student's; the narrow-RF arms read less.")
        for a in _SHARP_ABL_ARMS)

def _sharp_p2_rows() -> tuple["InkFamily", ...]:
    """Phase-2 sharp/dense attempts (2026-10-08): every loadable sharp_train run (last.pt and best.pt) wrapped by scripts/sharp_strategy/p2_convert.py as
    var/models/reader_v2_dense_p2_<run>_<last|best>.ckpt; the manifest (ordered by 54-tile skeleton F) is committed beside this file. Default OFF: they exist to be RUN and LOOKED AT."""
    import json
    import os
    f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sharp_p2_manifest.json")
    try:
        fams = json.load(open(f))["families"]
    except (OSError, ValueError):
        return ()
    return tuple(
        InkFamily("sharp_" + m["arm"].replace("-", "_"), "reader_v2_dense", arm=m["arm"], default="0", predict_kw=(("blend_stride", 0), ("shift_tta", 4), ("tile", 1024)),
                  um_per_px=9.5, layers=17, tile_px=int(m["extent_px"]) + 8,
                  label=f"sharp_dense attempt {m['run']} ({m['tag']}.pt; 54-tile skeleton F {m['skF4_am_54tiles']}; spec {m['spec']})",
                  help=f"sharp_train run {m['run']} {m['tag']}.pt wrapped as a reader_v2_dense-family checkpoint (md5 {m['md5'][:8]}, logit_bias {m['logit_bias']}, measured extent {m['extent_px']} px). "
                       "Skeleton F r4 at the area-matched threshold on the 54 tiles is that of the run's last.pt result (best.pt shares it as a label only). docs/experiments/sharp_fn_tolerant_2026-10-07/. Not shipped; run it to look at its maps.")
        for m in fams)



INK_FAMILIES: tuple[InkFamily, ...] = (
    InkFamily("i3d", "i3d", env="VPIPE_I3D", um_per_px=7.91, layers=30, tile_px=64, publish_band=4,
                 label="i3d",
                 help="InceptionI3d + conv decoder, our 2024 PHerc1667 fine-tune "
                      "(var/models/i3d_*.ckpt). 30 layers from index 17 of the Grand Prize "
                      "render, tile 64. Two extra passes per segment (both faces). Unscored."),
    InkFamily("resnet3d_1667", "resnet3d_1667", env="VPIPE_RESNET3D_1667",
                 um_per_px=7.91, layers=62, tile_px=256, publish_band=4, label="resnet3d_1667",
                 help="ResNet3D-50 + 2-D U-Net decoder, the six released PHerc.1667 "
                      "iterations (var/models/PHerc.1667-iteration-*). 62 CENTRED layers of "
                      "the Grand Prize render, tile 256. Two extra passes per segment. "
                      "Unscored here; iteration-0 reads 0.7613 vs iteration-5's 0.6684 on a "
                      "segment none of them saw (FINDINGS 61.3)."),
    # The SMALL families: a bounded receptive field measured by backprop, not asserted.
    # 0.6-1.2 MB of weights against the Grand Prize's 456 MB, and they share the Grand
    # Prize render, so the marginal cost of one is a forward pass over an existing stack.
    InkFamily("pixelclf_32x13", "pixelclf", arm="32x13",
                 um_per_px=7.91, layers=26, tile_px=31,
                 label="pixelclf:32x13 (small, 31 px receptive field)",
                 help="Per-pixel ink classifier with a BOUNDED receptive field -- 31 px "
                      "measured by backprop, 13 depth layers, 1.2 MB of weights. The patch-size "
                      "study measured the cross-scroll optimum near 32x32x13: small windows "
                      "generalise, wide deep ones memorise stroke texture (FINDINGS 70). "
                      "Shares the Grand Prize render, so a pass is cheap. PRELIMINARY: the "
                      "step count is in the checkpoint name and in its .metrics.json.",
                 withdrawn="dead step-5000 checkpoint: chance AUC 0.4775-0.4994, byte-identical to its own reversed "
                           "pass on 418 of 421 segments (2026-09-21/30); superseded by patchclf_32x13"),
    InkFamily("pixelclf_16x13", "pixelclf", arm="16x13",
                 um_per_px=7.91, layers=26, tile_px=17, multiscale=True, publish_band=1,
                 label="pixelclf:16x13 (smallest, 17 px receptive field)",
                 help="The same per-pixel classifier at half the window: 17 px measured "
                      "receptive field, 0.6 MB. The narrowest thing we can run that still "
                      "reads a depth column -- the control for 'how much context does ink "
                      "detection actually need'. PRELIMINARY.",
                 withdrawn="same dead pixelclf checkpoint lineage as pixelclf_32x13; superseded by patchclf_16x13"),
    # The patch-size study's REAL classifier (pixelclf recovery, 2026-09-30, coordinator-
    # authorised): a sliding-window centre classifier, not a dense per-pixel head -- every
    # pixel inside one `stride` cell gets the same score (publish_band set to that stride).
    # Trained with scripts/patchsize/run_grid.py --save-models (never done before this
    # recovery), reproducing FINDINGS section 66 on the labelled tiles the study itself
    # used. NOT scored yet on the production labelled-tile set; default off until it is.
    InkFamily("patchclf_32x13", "patchclf", arm="32x13",
                 um_per_px=7.91, layers=13, tile_px=32, publish_band=8,
                 label="patchclf:32x13 (window classifier, FINDINGS 66 recipe, now with saved weights)",
                 help="The patch-size study's actual 32x32x13 classifier, convolutionally "
                      "applied at an 8 px stride. Published numbers it reproduces: within-"
                      "scroll (Scroll 5) 0.739, cross-scroll LOSO (train Scroll 1+5, held "
                      "out Scroll 4) 0.629. See docs/experiments/pixelclf_recovery/."),
    InkFamily("patchclf_16x13", "patchclf", arm="16x13",
                 um_per_px=7.91, layers=13, tile_px=16, publish_band=4,
                 label="patchclf:16x13 (window classifier, smaller window)",
                 help="The same classifier at 16x16x13, 4 px stride. Published within-scroll "
                      "number: 0.699 (Scroll 5). Cross-scroll LOSO number measured for the "
                      "first time in this recovery -- see docs/experiments/pixelclf_recovery/."),
    # The UNTRAINED channel-mean scores: no checkpoint, no fit, closed form over the depth
    # column (stages/ink_models/colmean.py). They are in the DEFAULT set because they are
    # the only families whose cross-scroll number is not a transfer loss -- there is nothing
    # in them that could have learned a scroll -- and because Simple-ink Phase 0 measured
    # that no trained model here beats them off-scroll (FINDINGS: "No transferred score
    # beats the prior column mean on its target"; "No GP-family model transfers to Scroll 4
    # at 7.91 um", where the best trained head reads 0.54-0.56 against 0.51-0.61 untrained).
    # `sides` is False on every one of them: each score is symmetric about the surface
    # layer, so the verso pass would publish the same image under another name.
    # `tile_px` is the GAUSSIAN INTEGRATION SUPPORT (2*3*sigma+1), not the 1x1 per-pixel
    # score: that support is the honest in-plane window, it is over the First Letters
    # 0.5 mm limit, and `_record_window` says so on every prediction rather than passing by
    # default the way a family with no declared window does.
    InkFamily("normsurf", "colmean", arm="normsurf", sides=False, default="1",
                 um_per_px=7.91, layers=1, tile_px=385, multiscale=True, publish_band=1,
                 label="normsurf (untrained: surface / its own background, stroke-width integration)",
                 help="THE MOST CONSISTENT untrained score measured: the surface layer divided by "
                      "its own 506 um local background, then integrated at the stroke width "
                      "(300 um). Near-ink held-out AUC S1 0.571 / S5 0.617 / S4 0.592 -- the only "
                      "prior never significantly worse than a baseline on any scroll (FINDINGS "
                      "Phase 1b). NO WEIGHTS: nothing was fitted on any scroll, so those are "
                      "transfer numbers by construction. Seconds of CPU per cm2, no GPU, shares "
                      "the render."),
    InkFamily("colmean_pm8", "colmean", arm="pm8", sides=False, default="1",
                 um_per_px=7.91, layers=17, tile_px=97, multiscale=True, publish_band=1,
                 label="colmean:pm8 (untrained column brightness, +-8 layers)",
                 help="Mean of the 17 rendered layers centred on the surface (+-63 um), integrated "
                      "by a 127 um mask-normalised Gaussian, brighter = ink. Sign, window and "
                      "sigma were ALL fixed a priori, so its held-out AUC is the same measurement "
                      "on a scroll it has never seen: S1 0.524 / S5 0.650 / S4 0.596. This is the "
                      "score every trained model in FINDINGS is asked to beat across scrolls, and "
                      "none of them does. CPU only, shares the render."),
    InkFamily("surface_layer", "colmean", arm="surface", sides=False, default="1",
                 um_per_px=7.91, layers=1, tile_px=97, multiscale=True, publish_band=1,
                 label="surface_layer (untrained, the surface layer alone)",
                 help="The single rendered layer at the sheet, integrated by a 127 um Gaussian. "
                      "A-priori sign and sigma, nothing fitted. S1 0.539 / S5 0.576 / S4 0.609 -- "
                      "the best of the priors on Scroll 4, and the paired baseline every Phase 1 "
                      "comparison is made against. CPU only, shares the render."),
    InkFamily("colmean_pm16", "colmean", arm="pm16", sides=False, default="0",
                 um_per_px=7.91, layers=33, tile_px=193, multiscale=True,
                 label="colmean:pm16 (column brightness, +-16 layers, Scroll 5 pick)",
                 help="The same column-brightness score over 33 layers with a 253 um Gaussian: the "
                      "widest column the Phase 0 sweep picked on the Scroll 5 VAL segment. "
                      "S5 0.645 (SEEN) but S1 0.541 / S4 0.592 -- a val-selected sigma does not "
                      "survive the move, which is why the a-priori pm8 ships on and this does not. "
                      "CPU only."),
    InkFamily("colstd_pm16", "colmean", arm="std_pm16", sides=False, default="0",
                 um_per_px=7.91, layers=33, tile_px=193, multiscale=True,
                 label="colstd:pm16 (untrained depth-wise contrast, Scroll 5 pick)",
                 help="Standard deviation ALONG the depth column, 1x1 in plane: the depth "
                      "profile's shape rather than its level. S5 0.644 (SEEN), S1 0.502 / "
                      "S4 0.491 -- a Scroll 5 phenomenon. CPU only."),
    InkFamily("coldiffabs_pm4", "colmean", arm="diffabs_pm4", sides=False, default="0",
                 um_per_px=7.91, layers=9, tile_px=193, multiscale=True,
                 label="coldiffabs:pm4 (untrained depth-wise roughness, Scroll 4 pick)",
                 help="Mean |L(j+1)-L(j)| over 9 layers about the surface -- depth roughness, 1x1 "
                      "in plane. S4 0.554 (SEEN), S5 0.573, S1 0.483. CPU only."),
    # THE STROKE-WIDTH-AND-RING SCORE (user's name, 2026-09-15): the untrained statistic
    # FINDINGS 95.2 / 95.3 measured against 1,480 alternatives. It is the same KIND of thing
    # as the colmean rows above -- a closed form over the render, no checkpoint, no fit, so
    # its cross-scroll number is a measurement rather than a transfer loss -- and it beats
    # every one of them on the two scrolls where all were measured. `sides` is False: one
    # pass reads ten bands spanning the sheet, and the verso view would re-read the same
    # twelve layers in the other order for the same score.
    # THE ML WINDOW: the ring is the support, 2*3*1012.5 um = 6.08 mm at 7.91 um/px, which
    # `tile_px` declares honestly so the window is published with every prediction. The
    # First Letters 0.5 mm ML-window limit noted above WAS CHANGED for the First Letters
    # eligible scrolls (user, 2026-09-15), which is why this family ships the validated
    # 1-2 mm ring by default; `stroke_ring:ring506` is the narrow-ring arm and scores lower.
    InkFamily("stroke_ring", "stroke_ring", arm="ring1012", sides=False, default="1",
              um_per_px=7.91, layers=12, tile_px=769, multiscale=False, publish_band=1,
              label="stroke_ring (untrained: stroke-width centre minus a 1 mm ring, 10 depth bands)",
              help="THE BEST UNTRAINED SCORE MEASURED: ten 23.7 um depth bands through the "
                   "sheet, each read as a 253 um (stroke-width) Gaussian centre minus a 1012 um "
                   "ring, averaged over 10 bands x 5 centre scales. Held-out near-ink AUC "
                   "S4 0.621 [0.551,0.700] / S5 0.616 / S1 0.555, and 0.594 / 0.616 / 0.555 with "
                   "nothing tuned on the target scroll -- above every colmean prior on S4 and S1 "
                   "(FINDINGS 95.2, 95.3). NO WEIGHTS: nothing was fitted on any scroll. Centre "
                   "253 um beats wider AND narrower; the mean beats max/min/median, a fitted "
                   "logistic and robust estimators. CPU only, reads 12 of the rendered layers, "
                   "shares the render."),
    InkFamily("grandprize_dense", "grandprize_dense",
                 um_per_px=7.91, layers=26, tile_px=64, publish_band=4,
                 label="grandprize_dense (per-pixel Grand Prize head, DEFAULT checkpoint = the distill export)",
                 help="The Grand Prize TimeSformer backbone with a dense pixel-shuffle head: "
                      "output stride 1 px instead of the shipped 16, at roughly the shipped "
                      "cost. Shares the Grand Prize render. This row reads the UNNAMED "
                      "checkpoint var/models/grandprize_dense.ckpt, which is byte-identical to "
                      "grandprize_dense_distill.ckpt (md5 26172f30, checked 2026-09-15) -- so it "
                      "duplicates the `grandprize_dense_distill` row below and stays OFF; the two "
                      "named rows say which head they are."),
    # THE BASIC SET (user, 2026-09-15 pivot): the two dense heads by NAME, so the operator's
    # checkbox says which arm runs and the published PNG carries the arm in its filename.
    # `grandprize_dense:<arm>` resolves to var/models/grandprize_dense_<arm>.ckpt.
    InkFamily("grandprize_dense_dense64", "grandprize_dense", arm="dense64", hires2x=True,
                 um_per_px=7.91, layers=26, tile_px=64, publish_band=0,
                 label="grandprize_dense:dense64 (dense head, whole 64 px tile, S1+S5)",
                 help="The dense pixel-shuffle head trained from labels on Scroll 1 + Scroll 5 "
                      "(no distillation). Letter-F1 front on the labelled tiles: 0.5053 at "
                      "1.305 MP/s compiled -- the knee and the fastest point on that axis "
                      "(FINDINGS 73.6/73.7); held-out AUC 0.8117 / cross-scroll 0.7144. "
                      "Shares the Grand Prize render; one forward pass per face."),
    InkFamily("grandprize_dense_distill", "grandprize_dense", arm="distill", hires2x=True,
                 um_per_px=7.91, layers=26, tile_px=64, publish_band=0,
                 label="grandprize_dense:distill (dense head warm-started from the GP teacher, step 4000)",
                 help="The dense head distilled from the frozen shipped Grand Prize head, then "
                      "label-trained on Scroll 1 + Scroll 5. Held-out AUC 0.8462 / cross-scroll "
                      "0.7401 -- the most general dense head -- and 5.2x faster than the shipped "
                      "model compiled (FINDINGS 68.11, 68.14, 73.7). Shares the Grand Prize "
                      "render; one forward pass per face."),
    # A DENSE, STRIDE-FREE STUDENT OF ink_9um (FINDINGS, 2026-09-25 "ink9um_student"): the
    # teacher's own input (17 centred layers of the ~9.5 um render, so it SHARES the ink9um
    # render), a 3.7 M-parameter 2-D U-Net run on 2048 px tiles with a halo -- no overlapped
    # 128 px patch grid. `tile_px` declares the whole input dependency: the 129 px local
    # normalisation window plus the measured receptive field (~50 px), 179 px = 1.7 mm.
    InkFamily("ink9um_student", "ink9um_student",
              um_per_px=9.5, layers=17, tile_px=431, publish_band=4,
              label="ink9um_student (dense distilled ink_9um, 2048 px tiles, no overlap)",
              help="A 3.76 M-parameter U-Net distilled from ink_9um (68 M) on 209 unlabelled segments "
                   "of 21 scrolls; reads the ink9um render (17 centred layers, ~9.5 um/px), 33/129/257 px "
                   "local standardisation, 2048 px tiles + 224 px halo, no overlap. S4 HF v2 validation "
                   "boxes 0.852 vs teacher 0.845 AUC (n = 4 segments; wNNN, never used, 0.8717 vs 0.8716); "
                   "S5 near-ink +0.04 AUC over the teacher; S1 -0.12 (teacher trained on S1). Wall time per "
                   "face 7-16x below the teacher at stride 64 and 2-7x below it at stride 128 on the same "
                   "V100. Window 431 px = 4.1 mm (257 px normaliser + 175 px measured receptive field). "
                   "One seed; FINDINGS 2026-09-25 ink9um_student. Optional D4 TTA "
                   "(VPIPE_INK9UM_STUDENT_TTA=d4, default off, 8x cost): +0.008-0.023 AUC, "
                   "CI excluding 0 in 3 of 6 scoreable groups, never significantly negative. "
                   "A shift ensemble was also measured and does NOT help this checkpoint "
                   "(flat to significantly negative) -- not shipped. FINDINGS 2026-09-30."),
    # THE SHARP STUDENT (user, 2026-10-02: "Are the sharp ink detection models training? ... enabled ... by default
    # once trained"). Same module and input contract as ink9um_student, a different checkpoint
    # (var/models/ink9um_student_sharp.ckpt) distilled/fine-tuned for a crisper edge. OFF until the gate sidecar
    # var/models/ink9um_student_sharp.gate.json passes (sharpness p50 clearly better than production AND AUC not
    # worse beyond its CI -- scripts/hires_ink/sharp_gate.py); then maintenance.auto_enable_ink_winners turns it ON.
    InkFamily("ink9um_student_sharp", "ink9um_student", arm="sharp", auto_enable=True,
              um_per_px=9.5, layers=17, tile_px=431, publish_band=4,
              label="ink9um_student:sharp (narrow-RF, sharpened ink9um student)",
              help="Same 17-layer ~9.5 um/px input as ink9um_student, a checkpoint chosen by the sharp_student "
                   "ablation for a crisper edge (production ink9um_student reads 200-230 um 10-90 % edge "
                   "spread against a ~15 um label edge). Auto-enabled by the gate sidecar; see "
                   "docs/experiments/sharp_student/STATE.md for its measured AUC and sharpness. Cost per "
                   "face: that of ink9um_student or less."),
    # HECATE (user, 2026-10-05: "download Hecate AND start using it in our ink-detection pipeline"): the
    # Vesuvius Challenge staff model scrollprize/hecate @ 9cb86e50 (MIT), run through the UNMODIFIED upstream
    # hecate.py beside the weights (var/models/hecate/). Its 9.6 um checkpoint needs 9.6 um in plane AND in
    # depth: rendered at 9.6 um here, the native one-voxel planes resampled to 9.6 um slabs in the module.
    # tile_px = its 64 px patch (614 um); upstream's 32 px Hann blend makes one output pixel depend on a
    # 127 px span. The 2.4 um checkpoint is NOT a row: no render tier of ours is at 2.4 um.
    InkFamily("hecate_9um", "hecate", arm="9um",
              um_per_px=9.6, layers=16, tile_px=64, publish_band=4,
              label="hecate:9um (upstream staff model, ResNet-152 3-D + depth attention, 9.6 um isotropic)",
              help="scrollprize/hecate 9.6 um checkpoint (MIT): 3-D ink through depth, collapsed to 2-D by learned "
                   "attention; reads 16 planes at 9.6 um (our renders resampled in depth), 64 px patches, 32 px "
                   "Hann blend, bf16. TRAINED ON Scroll 1 and Scroll 4 labels (the HF letter boxes, wNNN "
                   "included) -- no S1/S4 number of it is transfer; S5 exposure unknown. HEAVY: ~0.03-0.04 Mpx/s "
                   "per face on an RTX 4060 Ti (minutes per Mpx). Scores: docs/experiments/hecate_transfer/."),
    # READER V2 (user, 2026-10-05: "research TTA with reader_v2"): DomRusso2's ink_9um-architecture checkpoint
    # (domenicor046/reader-v2 @ 72633d8a, MIT), run through the ink9um family's own wrapper with ONLY the
    # checkpoint swapped -- byte-identical to the author's koine_machines CLI on a test tile (scripts/reader_v2/
    # ref_repro.py). tile_px = the 128 px patch the net sees (1.22 mm at 9.5 um); the 50 % Hann overlap makes an
    # output pixel depend on up to a 255 px span.
    InkFamily("reader_v2", "reader_v2",
              um_per_px=9.5, layers=17, tile_px=128, publish_band=4,
              label="reader_v2 (DomRusso2 Reader v2: ink_9um architecture, native-scan dense labels, step 40k)",
              help="A drop-in ink_9um checkpoint (68 M params) trained 40k steps on dense labels from finer-scan "
                   "model maps of PHerc0139/0814/0500P2/0009B/0343P/0172 plus the ink_9um corpus. TRAINED ON S1 "
                   "(Paris4 wNNN), S4 (PHerc1667 wNNN, the HF boxes) and S5 (PHerc0172, 4 segments): no "
                   "S1/S4/S5 number of it is transfer. Same cost as ink9um (one 128 px / stride-64 pass per face). "
                   "Scores and TTA study: docs/experiments/reader_v2/STATE.md."),
    # DENSE_NATIVE: Erwin Nieuwlaar's fine-tune of KLAVIS's
    # ink9um-dense (dense9um-w016excluded-step075000) with dense native-scan pseudo-labels on 10 PHerc0139 segments (~30 % of each batch), step 16 000 of 20 000.
    # Same 68.2 M-parameter ink_9um architecture and tensor names as reader_v2: an ARM of the reader_v2 module (checkpoint swap only). Intended for z-score-average ensembles (scripts/ink_ensemble/zavg.py).
    InkFamily("dense_native", "reader_v2", arm="dense_native", default="0",
              um_per_px=9.5, layers=17, tile_px=128, publish_band=4,
              label="dense_native (Nieuwlaar: ink9um-dense + native PHerc0139 pseudo-labels, step 16k; MIT)",
              help="ink_9um architecture (68 M params), 17 centred planes, 128 px patches, 50 % Hann overlap, per-patch robust-MAD normalisation. Trained on dense "
                   "pseudo-labels incl. 10 native PHerc0139 segments (ids omitted); PHerc0139 title AUC 0.9548 vs 0.9147 for its "
                   "init. Source: huggingface.co/Nieuwlaar/ink9um-dense-native (weights sha256 d3d95dc4...), rebuilt .pth md5 96e6054f. Same cost as reader_v2."),
    # READER V2 DENSE (user, 2026-10-05: "start the dense distillation"; 2026-10-06: "add it to production as an
    # alternative"): Reader v2 distilled into a 539 k-parameter 3-level U-Net with a 9/17 px normaliser --
    # architectural support 59 px (measured, <= the 64 px cap). Per-pixel output, 2048 px tiles, no overlap.
    # Default weights = the CONVERGENCE retrain (rv2c_sharpen2: plateau schedule to convergence, EMA, unrendered
    # region supervised to 0 + edge-centred crops; best ckpt step 32k of 80k, md5 fe425e4f). On 54 labelled tiles /
    # 32 parents: near64 AUC 0.663 [0.631,0.690] (12k-step predecessor 0.633; teacher 0.809; ink9um_student 0.697),
    # UNSEEN 0.678 [0.607,0.698]; rise40 p50 388 um (ink9um_student 488). Below its teacher: an alternative, not a
    # replacement. The 12k weights are kept as arm sharpen2_12k.
    InkFamily("reader_v2_dense", "reader_v2_dense",
              um_per_px=9.5, layers=17, tile_px=59, publish_band=4,
              label="reader_v2_dense (Reader v2 distilled, small-RF dense student, 59 px support)",
              help="A 539 k-parameter U-Net (3 levels, 9/17 px local standardisation) distilled from Reader v2 on "
                   "131 unlabelled segments, trained to convergence (2026-10-06); per-pixel output at ~9.5 um/px, "
                   "window 59 px = 0.56 mm (measured support). Reader v2 trained on S1 (PHercParis4 selected segments), "
                   "S4 (PHerc1667 wNNN list), 4 PHerc0172 segments and PHerc0139/0814/0500P2/0009B/0343P: "
                   "S1/S4 numbers are not transfer; S5 2024 ctl segments are unseen SEGMENTS of a seen scroll; "
                   "PHerc0841 and the Kaggle fragments are unseen scrolls. near64 AUC 0.663 [0.631,0.690] over 54 "
                   "tiles / 32 parents (teacher 0.809, ink9um_student 0.697); no segment-edge response (unrendered "
                   "rim 0.002 vs 0.40 for the 12k weights). Arms: sharpen2_12k (previous default), base, rf33. "
                   "docs/experiments/reader_v2_distill/STATE.md."),
    # SHARP DENSE (user, 2026-10-07, top priority: "we absolutely need sharp ink detectors in production, ASAP ... SHARP,
    # DENSE, FULL-RES fine-tune using all of the v2 data and all data we have"). Same module, input contract and
    # architecture class as reader_v2_dense (539 k-parameter 3-level U-Net, 9/17 px local standardisation, per-pixel logit
    # at the native ~9.5 um pitch, no downsampling of the output, 59 px architectural support), a different checkpoint
    # (var/models/reader_v2_dense_sharp.ckpt) from the sharp_strategy study recipe: real Reader-v2 dense-native labels +
    # our human labels, buffer-zone supervision, long fine-tune. docs/experiments/sharp_prod_2026-10-07/STATE.md.
    InkFamily("sharp_dense", "reader_v2_dense", arm="sharp",
              um_per_px=9.5, layers=17, tile_px=59, publish_band=4,
              label="sharp_dense (sharp full-res dense U-Net, real dense + human labels, 59 px support)",
              help="A 539 k-parameter dense U-Net (stride-1 output, ~9.5 um/px, 9/17 px local standardisation, 59 px "
                   "= 0.56 mm measured support) trained on Reader-v2 dense-native labels + human labels with buffer-zone "
                   "supervision, no teacher soft targets. Optimised for edge sharpness over coverage; its near64 AUC is "
                   "BELOW the ink9um_student's (see docs/experiments/sharp_prod_2026-10-07/STATE.md for the measured "
                   "numbers and limits)."),
    # SHARP DENSE HUMAN (user, 2026-10-08: "I need that sharp dense reader v2 trained on human labels right now"). Same module,
    # input contract and architecture as reader_v2_dense / sharp_dense; checkpoint var/models/reader_v2_dense_human_cv14k.ckpt =
    # sharp_train run CV_Hunfiltered_s0: HUMAN labels only (hl pool: S1, S4 post-reading HF, PHerc0139 open data and one S5 mesh;
    # the label-trust filter is NOT applied), loss bce, confirmed negatives w_cn 5, positives eroded 2 px, 2 px unsupervised gap,
    # 8 px RING negatives (weight 1.0) kept, converged by the D34 rule (plateau at 12k + 2k anneal; the held-out curve was still creeping, so
    # marginal), seed 0, held-out val near64 AUC 0.605, logit_bias 0 (uncalibrated). docs/experiments/sharp_fn_tolerant_2026-10-07.
    InkFamily("sharp_dense_human", "reader_v2_dense", arm="human_cv14k",
              um_per_px=9.5, layers=17, tile_px=59, publish_band=4,
              label="sharp_dense_human (sharp dense U-Net, human labels only, 59 px support)",
              help="Same 539 k-parameter dense U-Net as sharp_dense, trained ONLY on human labels (no teacher-derived rv2L labels, "
                   "no teacher soft targets); confirmed negatives w5 AND 8 px ring negatives (2 px unsupervised gap). Converged by the D34 rule at "
                   "14k steps (seed 0; marginal plateau). Label-trust filter not yet applied. Uncalibrated logit (bias 0)."),
    # STRIDE-BLENDED dense maps (user, 2026-10-08: "finer stride ... include the stride in the family name"). Same checkpoints as reader_v2_dense / sharp_dense, run as
    # OVERLAPPED 128 px patches with a floored 2-D Hann blend at stride 63 (NOT a multiple of the 4 px decoder period) instead of 2048 px tiles: removes the 4 px grid (P=4 depth
    # 9.3x -> 0.71x, 54 tiles) at 4.1 forward passes, localisation 30.8 um (shift-TTA x4: 34.3), skeleton F unchanged (-0.001). stride 64 / 32 do NOT remove it (9.4x / 9.3x).
    InkFamily("reader_v2_dense_stride63", "reader_v2_dense", predict_kw=(("blend_stride", 63), ("blend_patch", 128), ("shift_tta", 1)),
              um_per_px=9.5, layers=17, tile_px=59, publish_band=4,
              label="reader_v2_dense, 128 px patches blended at stride 63 (grid removed)",
              help="reader_v2_dense run as overlapped 128 px Hann-blended patches at stride 63 px (4.1 passes). The stride is not a multiple of the decoder period, so the 4 px grid averages away. "
                   "Output pitch is still 9.5 um: a finer stride averages aliasing, it adds no resolution."),
    InkFamily("sharp_dense_stride63", "reader_v2_dense", arm="sharp", predict_kw=(("blend_stride", 63), ("blend_patch", 128), ("shift_tta", 1)),
              um_per_px=9.5, layers=17, tile_px=59, publish_band=4,
              label="sharp_dense, 128 px patches blended at stride 63 (grid removed)",
              help="sharp_dense run as overlapped 128 px Hann-blended patches at stride 63 px (4.1 passes). See reader_v2_dense_stride63."),
    # DISTILLATION STUDENTS of reader_v2 (sharp program, 2026-10-08; docs/experiments/sharp_fn_tolerant_2026-10-07). Checkpoints are sharp_train students wrapped by
    # scripts/sharp_strategy/sharp_student_to_prod.py (calibrated: logit_bias = logit(pooled 1 % confirmed-negative FPR threshold)). Tile inference with halo + shift x4 (their support is not 59 px).
    InkFamily("student_cnx", "reader_v2_dense", arm="student_cnx", predict_kw=(("blend_stride", 0), ("shift_tta", 4), ("tile", 1024)),
              um_per_px=9.5, layers=17, tile_px=149,
              label="student_cnx (reader_v2 distillation, ConvNeXt blocks, 1.4 M params, converged 28k, 54-tile skeleton F 0.139)",
              help="4-level ConvNeXt-block U-Net distilled from reader_v2 (112 segments, soft BCE + 0.1 logit-MSE, BlurPool), declared architectural support ~135-149 px. 163 kMAC/px. Statistically equal to reader_v2_dense on the 54 tiles at 21 % fewer MACs; "
                   "still 0.078 skeleton F below reader_v2 (docs/experiments/sharp_fn_tolerant_2026-10-07/RESULTS.md)."),
    InkFamily("student_blur", "reader_v2_dense", arm="student_blur", predict_kw=(("blend_stride", 0), ("shift_tta", 4)),
              um_per_px=9.5, layers=17, tile_px=59,
              label="student_blur (reader_v2 distillation, 0.54 M params, BlurPool, converged 20k, 54-tile skeleton F 0.133)",
              help="The shipped reader_v2_dense topology (3 levels 48-64-128) re-distilled with BlurPool strided downsampling and our short recipe; control for student_cnx."),
    # reader_v2_dense with the SAME weights but a calibrated output (user, 2026-10-08: "I need a sharper version of this reader_v2_dense map"). The shipped checkpoint has logit_bias 0, so its background sits near
    # p ~ 0.4 and the map looks flat/grey next to sharp_dense (logit_bias 2.18). Calibrating at the pooled 1 % confirmed-negative FPR threshold is a monotone shift: AUC, skeleton F, edge width unchanged -- only display contrast.
    InkFamily("reader_v2_dense_cal", "reader_v2_dense", arm="cal",
              um_per_px=9.5, layers=17, tile_px=59, publish_band=4,
              label="reader_v2_dense, calibrated output (same weights; contrast only)",
              help="Identical weights to reader_v2_dense; logit_bias set so that 0.5 is the pooled 1 % confirmed-negative-FPR operating point. A monotone map: it changes display contrast, not sharpness or accuracy."),
    # K1 (FINDINGS Sec 122.1 part 3, 2026-09-26): untrained consensus of C1 (ten fixed depth
    # bands x five centre scales minus a 1012 um ring, stroke_ring's own geometry, robust-z
    # combined) and the edge-aligned band E (grey value 7.2 um inside the ink-side half-max
    # sheet edge, pooled 100 um minus a 1012.5 um surround). No checkpoint, no fit. Default
    # OFF: ships here so it can be scored end-to-end and reproduced against the published
    # research numbers before the coordinator turns it on.
    InkFamily("k1", "k1", sides=False, default="0",
              um_per_px=7.91, layers=65, tile_px=769, publish_band=1,
              label="k1 (untrained: C1 + edge-aligned band consensus)",
              help="Mean robust z of C1 (stroke_ring's band/scale/ring geometry, median/MAD "
                   "standardised) and the sheet-edge band 7.2 um inside the ink-side half-max "
                   "edge (100 um pooling, 1012.5 um surround). First untrained arm measured to "
                   "beat C1 on UNSEEN objects: near-64 AUC S4 0.634 (+0.040 vs C1, 5/5 "
                   "segments) / S5 0.667 (+0.027, 12/13) / S1 0.520 (-0.008, 2/7); 6 wholly "
                   "unseen objects (19 segments, none touched by any choice this score made) "
                   "+0.052 [+0.029,+0.078] AUC object-level, 6/6 (FINDINGS Sec 122.1 part 3). "
                   "NO WEIGHTS. CPU only, ~20-25 of the rendered layers read regardless of "
                   "stack length. PRELIMINARY: awaiting production-code reproduction of the "
                   "published numbers before the coordinator enables it."),
    *_sharp_abl_rows(),
    *_sharp_p2_rows(),
)
INK_BY_NAME: dict[str, InkFamily] = {f.name: f for f in INK_FAMILIES}


def ink_family(name: str) -> InkFamily | None:
    """The registry row for a planned model name, or None when it is not a zoo family
    (`grandprize`, `ink9um`, `unet3d` and `warp*` are handled in stages/ink.py)."""
    return INK_BY_NAME.get(name)


def withdrawn_families() -> dict[str, str]:
    """{family name: reason} for every withdrawn family, its `_reversed` face included."""
    out = {}
    for f in INK_FAMILIES:
        if f.withdrawn:
            out[f.name] = out[f.name + "_reversed"] = f.withdrawn
    return out


def withdrawn_sql(col: str = "meta_json") -> tuple[str, list[str]]:
    """An SQL fragment excluding artifacts of withdrawn families by their meta_json family field (both faces:
    the pattern has no closing quote, so `pixelclf_32x13` also matches `pixelclf_32x13_reversed`)."""
    names = [f.name for f in INK_FAMILIES if f.withdrawn]
    return ("".join(" AND coalesce(%s, '') NOT LIKE ?" % col for _ in names),   # NULL meta must not drop a row
            ['%%"family": "%s%%' % n for n in names])
