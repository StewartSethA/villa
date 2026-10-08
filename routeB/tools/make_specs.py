#!/usr/bin/env python3
"""Build routeB/scrolls/<scroll>.json (+ copy the umbilicus files) from OUR registry and the live upstream listings.
Run in the main repo (needs src/vesuvius_pipeline/scroll_registry, tools/umbilicus_auto/output); the OUTPUT is what the deploy branch carries,
so the branch never reads the registry.  Every field records where it came from (registry vs live listing vs assumed)."""
import json, re, shutil, sys, urllib.request, xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
REG = REPO / "src/vesuvius_pipeline/scroll_registry"
UM = REPO / "tools/umbilicus_auto/output"
BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
DL = "https://dl.ash2txt.org/datasets/spiral_datasets"
UP = json.loads((REPO / "docs/experiments/readerB_2026-10-08/data/upstream_inputs.json").read_text())
PLAN = json.loads((REPO / "docs/experiments/readerB_2026-10-08/data/plan_numbers.json").read_text())
SHELL = PLAN["inputs"]["shell"]          # fitted / pre-fit shell_outer_winding_idx; others ASSUMED 200
# spiral_outward_sense actually USED by the converged production fits where the registry value is null / disagrees (STATE docs)
SENSE_FALLBACK = {"PHerc0125": "CW"}


def ls(prefix, delim="/"):
    url = f"{BUCKET}/?list-type=2&prefix={prefix}&delimiter={delim}"
    root = ET.fromstring(urllib.request.urlopen(url, timeout=60).read())
    ns = {"s": root.tag.split("}")[0].strip("{")}
    return ([c.find("s:Prefix", ns).text for c in root.findall("s:CommonPrefixes", ns)],
            [(c.find("s:Key", ns).text, int(c.find("s:Size", ns).text)) for c in root.findall("s:Contents", ns)])


def http_ls(url):
    try:
        h = urllib.request.urlopen(url, timeout=60).read().decode()
    except Exception:
        return []
    return re.findall(r'<a href="([^"?][^"]*)"', h)


def head(url):
    try:
        r = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60)
        return int(r.headers["Content-Length"])
    except Exception:
        return None


def val(d, *ks):
    for k in ks:
        d = (d or {}).get(k)
    return d.get("value") if isinstance(d, dict) and "value" in d else d


def main():
    for f in sorted(REG.glob("*.json")):
        r = json.loads(f.read_text())
        S = r["scroll"]
        vol = val(r, "volume", "upstream_zarr")
        vid = val(r, "volume", "volume_id")
        vox = val(r, "volume", "voxel_um")
        if not vol:
            vols, _ = ls(f"{S}/volumes/")
            print(S, "volume from live listing", vols)
        spec = {"scroll": S, "voxel_um": vox, "volume_id": vid, "volume_zarr": vol,
                "volume_s3_prefix": f"{S}/volumes/{vol}" if vol else None,
                "roles": val(r, "roles")}
        # tracks: live listing of dl.ash2txt.org
        ts_dirs = [d for d in http_ls(f"{DL}/{S}/") if re.fullmatch(r"\d{14}/", d)]
        tr = None
        for d in ts_dirs:
            base = f"{DL}/{S}/{d}tracks/"
            names = [n for n in http_ls(base) if n.endswith(".dbm") or n.endswith(".extract.json") or n.endswith(".crossings.npz")]
            if any(n.endswith(".dbm") for n in names):
                tr = {"ts": d.strip("/"), "base_url": base, "files": {n: head(base + n) for n in names if "vctracks" not in n}}
        spec["tracks"] = tr
        # lasagna: live listing
        pre, _ = ls(f"{S}/representations/predictions/lasagna/")
        las = None
        for p in sorted(pre)[::-1]:
            if p.rstrip("/").endswith("lasagna/"):
                continue
            sub, keys = ls(p)
            zs = {re.search(r"_(nx|ny|cos|grad_mag)\.ome\.zarr", s).group(1): s.rstrip("/") for s in sub if re.search(r"_(nx|ny|cos|grad_mag)\.ome\.zarr/$", s)}
            if "nx" in zs and "ny" in zs:
                las = {"prefix": p.rstrip("/"), "fields": zs, "json": [k for k, _ in keys if k.endswith(".lasagna.json")]}
                break
        spec["lasagna"] = las
        spec["spiral_outward_sense"] = val(r, "spiral_outward_sense") or SENSE_FALLBACK.get(S)
        spec["spiral_outward_sense_source"] = ("registry (upstream catalog/volume properties)" if val(r, "spiral_outward_sense")
                                                else ("SENSE_FALLBACK (the sense the production fit used)" if S in SENSE_FALLBACK else "UNKNOWN: pass --sense CW|ACW; run an A/B"))
        spec["shell_outer_winding_idx"] = SHELL.get(S, 200)
        spec["shell_source"] = "fitted/pre-fit estimate (readerB plan)" if S in SHELL else "ASSUMED 200 (no fit yet; pass --shell N)"
        up = val(r, "umbilicus", "path")
        spec["umbilicus_registry_path"] = up
        spec["umbilicus_status"] = val(r, "umbilicus", "status")
        udir = OUT / "routeB" / "umbilicus" / S
        udir.mkdir(parents=True, exist_ok=True)
        if up and (REPO / up).exists():
            shutil.copy(REPO / up, udir / "umbilicus.json")
            spec["umbilicus_file"] = f"routeB/umbilicus/{S}/umbilicus.json"
        # also carry our own auto estimates
        for extra in ("umbilicus.json", "umbilicus.consensus-v2.json", "umbilicus.consensus-v3ct.json"):
            if (UM / S / extra).exists() and not (udir / extra).exists() and (UM / S / extra).stat().st_size < 2_000_000:
                shutil.copy(UM / S / extra, udir / ("estimate_" + extra))
        (OUT / "routeB/scrolls" / f"{S}.json").write_text(json.dumps(spec, indent=1))
        print(S, "tracks", (tr or {}).get("ts"), "lasagna", bool(las), "sense", spec["spiral_outward_sense"], "umb", up and Path(up).name)


if __name__ == "__main__":
    main()
