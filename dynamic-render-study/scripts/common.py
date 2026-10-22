"""Study configuration. One JSON file, all paths relative to the file's own directory (or absolute if you choose).
Lookup order: $STUDY_CONFIG, ./config.json, <study root>/config.json, <study root>/config.example.json (with a warning)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
KEYS = ("data_dir", "volume", "cpu_renderer", "tmp_dir", "out_dir", "num_slices")
def load_config():
    cands = [os.environ.get("STUDY_CONFIG"), "config.json", os.path.join(ROOT, "config.json"), os.path.join(ROOT, "config.example.json")]
    for p in cands:
        if p and os.path.exists(p):
            if p.endswith("config.example.json"): print("WARNING: using config.example.json; copy it to config.json and edit", file=sys.stderr)
            c = json.load(open(p)); base = os.path.dirname(os.path.abspath(p))
            miss = [k for k in KEYS if k not in c]
            if miss: raise SystemExit("config %s misses keys %s" % (p, miss))
            for k in KEYS:
                if k != "num_slices" and not os.path.isabs(c[k]): c[k] = os.path.normpath(os.path.join(base, c[k]))
            os.makedirs(c["tmp_dir"], exist_ok=True); os.makedirs(c["out_dir"], exist_ok=True)
            return c
    raise SystemExit("no config.json found; copy config.example.json to config.json")
