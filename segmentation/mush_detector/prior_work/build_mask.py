#!/usr/bin/env python3
"""Build the PHerc0125 3-D mush mask from the 9 registered label slices.

Per labelled slice (level0 z): rasterize the GIMP label alpha into the
scroll's level-3 grid using the fitted affine (scale + translation from
register.py), separate thick "blob" (crushed zone) from thin "crack"
(tear) by morphological opening at screenshot resolution (before the ~8x
level-3 downsample, where a crack a few voxels wide would vanish), then
fill the z-gaps BETWEEN labelled planes by linear interpolation of each
channel's signed-distance field (shape-based interpolation, Raya & Udupa
1990) -- never extrapolated beyond the labelled z range.

Output: an OME-Zarr-ish store with "blob" and "crack" uint8 arrays (0..255
confidence, 128 ~= the interpolated boundary) at level 3, plus a JSON
manifest with per-slice registration stats and areas.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import zarr
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

SRC_DIR = Path("/home/seth/ScrollPrizeTutorial/data/Scrolls/crushed scroll label")
REG_DIR = Path(
    "/tmp/claude-1000/-home-seth-ScrollPrizeTutorial/2009ddb2-9fa9-4865-b0ef-fcf612867acf/scratchpad/mush/reg_out"
)
ZARR_PATH = "/mnt/raid7/scroll_volume_cache/PHerc0125.zarr"
OUT_PATH = Path("/mnt/raid7/experiments/mush_labels/PHerc0125/mush_mask.zarr")
MANIFEST_PATH = Path("/home/seth/ScrollPrizeTutorial/docs/experiments/mush_labels/PHerc0125_mask_manifest.json")

SLICES = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
OUT_LEVEL = 3
VOX_UM = 9.362
OPEN_RADIUS_SCREENSHOT_PX = 6     # blob/crack split; see STATE.md for the calibration note
SDF_CLIP = 16.0                    # level-3 px, ~= 1.2 mm; interpolation band half-width


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def label_layer_name(z):
    for suf in ("label", "Label"):
        p = SRC_DIR / f"0125-z{z}spiral.xcf"
        exp = Path(
            "/home/seth/ScrollPrizeTutorial/data/labels/mush/PHerc0125"
        ) / f"0125-z{z}spiral__{suf}.png"
        if exp.exists():
            return exp
    raise FileNotFoundError(z)


def load_label_alpha(z):
    p = label_layer_name(z)
    im = np.array(Image.open(p))
    return im[..., 3]  # alpha = paint strength, 0..255 (soft, not thresholded)


def main():
    g_ct = zarr.open(ZARR_PATH, mode="r")
    lvl = g_ct[str(OUT_LEVEL)]
    Z3, Y3, X3 = lvl.shape
    factor_l0 = 2 ** OUT_LEVEL  # level0 vox per level3 px

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.open(str(OUT_PATH), mode="w")
    chunks = (128, 128, 128)
    # signed "confidence" field, int16 internally then packed to uint8 on write;
    # keep a float buffer only for the planes actually touched (labelled z range).
    z3_lo = SLICES[0] // factor_l0
    z3_hi = SLICES[-1] // factor_l0 + 1
    nplanes = z3_hi - z3_lo

    blob_sdf_planes = {}   # z3 index (labelled only) -> float32 (Y3,X3) SDF, clipped
    crack_sdf_planes = {}
    manifest = {
        "scroll": "PHerc0125",
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_md5": hashlib.md5(Path(__file__).read_bytes()).hexdigest(),
        "out_level": OUT_LEVEL,
        "level0_vox_um": VOX_UM,
        "level3_px_um": VOX_UM * factor_l0,
        "open_radius_screenshot_px": OPEN_RADIUS_SCREENSHOT_PX,
        "sdf_clip_level3_px": SDF_CLIP,
        "z_valid_range_level0": [SLICES[0], SLICES[-1]],
        "slices": [],
    }

    for z in SLICES:
        regf = REG_DIR / f"reg_z{z}.json"
        reg = json.loads(regf.read_text())
        fs = reg["fit_scale_vox_per_screenshot_px"]          # level0 vox per canvas px
        x0_l0, y0_l0 = reg["match_topleft_level0_xy"]
        alpha = load_label_alpha(z)                           # full screenshot res
        r0, r1, c0, c1 = reg["canvas_bbox_screenshot_px"]
        alpha = alpha[r0:r1, c0:c1]                            # canvas-local crop, matches the fit
        mask = alpha > 0
        npaint = int(mask.sum())

        if npaint == 0:
            manifest["slices"].append({"z_level0": z, "note": "no painted pixels", "fit_ncc": reg["fit_ncc"]})
            continue

        core = binary_erosion(mask, structure=disk(OPEN_RADIUS_SCREENSHOT_PX))
        blob_screenshot = binary_dilation(core, structure=disk(OPEN_RADIUS_SCREENSHOT_PX)) & mask
        crack_screenshot = mask & ~blob_screenshot

        v, u = np.nonzero(mask)  # row=v, col=u, canvas-local
        x_l0 = x0_l0 + u.astype(np.float64) * fs
        y_l0 = y0_l0 + v.astype(np.float64) * fs
        x3 = np.round(x_l0 / factor_l0).astype(np.int64)
        y3 = np.round(y_l0 / factor_l0).astype(np.int64)
        ok = (x3 >= 0) & (x3 < X3) & (y3 >= 0) & (y3 < Y3)

        z3 = z // factor_l0

        def rasterize(local_mask_2d):
            sel = local_mask_2d[v, u] & ok
            plane = np.zeros((Y3, X3), dtype=bool)
            plane[y3[sel], x3[sel]] = True
            return plane

        blob_plane = rasterize(blob_screenshot)
        crack_plane = rasterize(crack_screenshot)

        def sdf(plane_bool):
            if not plane_bool.any():
                return -np.full((Y3, X3), SDF_CLIP, dtype=np.float32)
            inside = distance_transform_edt(plane_bool).astype(np.float32)
            outside = distance_transform_edt(~plane_bool).astype(np.float32)
            s = np.where(plane_bool, inside, -outside)
            return np.clip(s, -SDF_CLIP, SDF_CLIP)

        blob_sdf_planes[z3] = sdf(blob_plane)
        crack_sdf_planes[z3] = sdf(crack_plane)

        manifest["slices"].append({
            "z_level0": z, "z3": z3,
            "fit_ncc": reg["fit_ncc"], "fit_residual_1_minus_ncc": reg["fit_residual_1_minus_ncc"],
            "fit_scale_vox_per_screenshot_px": fs,
            "painted_px_screenshot": npaint,
            "blob_px_screenshot": int(blob_screenshot.sum()),
            "crack_px_screenshot": int(crack_screenshot.sum()),
            "blob_cm2": float(blob_plane.sum()) * (VOX_UM * factor_l0 * 1e-4) ** 2,
            "crack_cm2": float(crack_plane.sum()) * (VOX_UM * factor_l0 * 1e-4) ** 2,
        })
        print(f"z={z}: ncc={reg['fit_ncc']:.3f} blob_px={blob_screenshot.sum()} crack_px={crack_screenshot.sum()}", flush=True)

    # write arrays, filling the labelled z-range by SDF interpolation between neighbours
    labelled_z3 = sorted(blob_sdf_planes.keys())
    blob_arr = store.create_dataset("blob", shape=(Z3, Y3, X3), chunks=chunks, dtype="u1", fill_value=0)
    crack_arr = store.create_dataset("crack", shape=(Z3, Y3, X3), chunks=chunks, dtype="u1", fill_value=0)

    def to_u8(sdf_plane):
        return np.clip((sdf_plane / SDF_CLIP) * 127 + 128, 0, 255).astype(np.uint8)

    for i in range(len(labelled_z3) - 1):
        za, zb = labelled_z3[i], labelled_z3[i + 1]
        sa_b, sb_b = blob_sdf_planes[za], blob_sdf_planes[zb]
        sa_c, sb_c = crack_sdf_planes[za], crack_sdf_planes[zb]
        for z3 in range(za, zb + 1):
            t = 0.0 if zb == za else (z3 - za) / (zb - za)
            blob_arr[z3] = to_u8((1 - t) * sa_b + t * sb_b)
            crack_arr[z3] = to_u8((1 - t) * sa_c + t * sb_c)
    if len(labelled_z3) == 1:
        z3 = labelled_z3[0]
        blob_arr[z3] = to_u8(blob_sdf_planes[z3])
        crack_arr[z3] = to_u8(crack_sdf_planes[z3])

    store.attrs["mush_level"] = OUT_LEVEL
    store.attrs["scroll"] = "PHerc0125"
    store.attrs["z_valid_range_level0"] = [SLICES[0], SLICES[-1]]
    store.attrs["channels"] = {"blob": "crushed/mushy zone, 0..255 soft confidence, 128=boundary",
                                "crack": "crack/tear, 0..255 soft confidence, 128=boundary"}
    store.attrs["interpolation"] = "linear blend of per-plane signed distance fields between labelled z slices (shape-based interpolation); never extrapolated outside z_valid_range_level0"
    store.attrs["code_md5"] = manifest["code_md5"]
    store.attrs["built_utc"] = manifest["built_utc"]

    total_blob_cm2 = sum(s["blob_cm2"] for s in manifest["slices"] if "blob_cm2" in s)
    total_crack_cm2 = sum(s["crack_cm2"] for s in manifest["slices"] if "crack_cm2" in s)
    manifest["total_blob_cm2_at_labelled_slices"] = total_blob_cm2
    manifest["total_crack_cm2_at_labelled_slices"] = total_crack_cm2
    manifest["out_path"] = str(OUT_PATH)
    manifest["n_labelled_planes"] = len(labelled_z3)
    manifest["n_interpolated_planes_total"] = int(nplanes)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=1))
    print("DONE", OUT_PATH, "manifest", MANIFEST_PATH)


if __name__ == "__main__":
    main()
