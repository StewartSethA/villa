"""Study configuration: one JSON file, paths relative to the file's own directory. Lookup: $STUDY_CONFIG, ./config.json, <study root>/config.json, <study root>/config.example.json."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
KEYS = ("data_dir", "flat_dir", "cpu_flattener", "lasagna_dir", "lasagna_python")
def load_config():
    for p in [os.environ.get("STUDY_CONFIG"), "config.json", os.path.join(ROOT, "config.json"), os.path.join(ROOT, "config.example.json")]:
        if p and os.path.exists(p):
            if p.endswith("config.example.json"): print("WARNING: using config.example.json; copy it to config.json and edit", file=sys.stderr)
            c = json.load(open(p)); base = os.path.dirname(os.path.abspath(p)); miss = [k for k in KEYS if k not in c]
            if miss: raise SystemExit("config %s misses keys %s" % (p, miss))
            for k in KEYS:
                if k != "lasagna_python" and not os.path.isabs(c[k]): c[k] = os.path.normpath(os.path.join(base, c[k]))
            os.makedirs(c["flat_dir"], exist_ok=True); return c
    raise SystemExit("no config.json found; copy config.example.json to config.json")
