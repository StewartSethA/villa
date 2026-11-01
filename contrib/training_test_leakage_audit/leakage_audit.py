#!/usr/bin/env python3
"""Does a model's training set overlap the test set?  Compare training manifests with a test list by physical segment.

A manifest is a JSON list of training entries; an entry needs `name` and/or `physical_segment` (a segment directory name such as
"20250108000005-w030_2025010818", or "46527_2um_try2").  A test list is a JSON list of such ids or a text file with one id per line.
Matching is by the segment's id token (the leading numeric timestamp, or the whole id for names without one), so a representation called
"pherc0139-w017" that points at the physical segment 20250108000005-w030 is found even though its name does not contain the id.

usage: leakage_audit.py TRAIN_MANIFEST.json [TRAIN_MANIFEST2.json ...] --test TEST_LIST.(json|txt) [--out report.json]
exit code 1 when anything overlaps (so it can gate a training or evaluation job)."""
import argparse, json, re, sys


def seg_tokens(s: str) -> set:
    """identifying tokens of a segment id: its 12-14 digit timestamp(s), and any dash-separated part of >= 10 characters that is not purely numeric
    (so "20260226000000-46527_2um_try2" and the bare test id "46527_2um_try2" share a token)."""
    s = str(s).strip().lower()
    toks = set(re.findall(r"(\d{12,14})", s))
    for part in re.split(r"[-/]", s.split("__")[-1]):
        if len(part) >= 10 and not part.isdigit():
            toks.add(part)
    return toks or {s}


def seg_token(s: str) -> str:
    """first token (kept for callers that want a single key)"""
    return sorted(seg_tokens(s))[0]


def load_list(path):
    if path.endswith(".json"):
        d = json.load(open(path))
        return [e if isinstance(e, str) else (e.get("physical_segment") or e.get("name")) for e in d]
    return [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]


def audit(train_entries, test_ids):
    test = {}
    for t in test_ids:
        for tok in seg_tokens(t):
            test[tok] = t
    hits = []
    for e in train_entries:
        for key in ("physical_segment", "name"):
            if e.get(key):
                m = seg_tokens(e[key]) & set(test)
                if m:
                    hits.append(dict(train_name=e.get("name"), train_segment=e.get("physical_segment"), test_segment=test[sorted(m)[0]], label_kind=e.get("label_kind"), matched_on=key)); break
    return hits


def write_datasets_json(path, train, val, test, label_source, **extra):
    """the record every training run should publish BEFORE its first step: segment names per role, label source/version, anything else needed to audit leakage."""
    json.dump(dict(train=train, validation=val, test=test, label_source=label_source, **extra), open(path, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("manifests", nargs="+"); ap.add_argument("--test", required=True); ap.add_argument("--out", default=None); a = ap.parse_args()
    test = load_list(a.test); rep = {}
    for m in a.manifests:
        hits = audit(json.load(open(m)), test); rep[m] = dict(n_hits=len(hits), test_segments_hit=sorted({h["test_segment"] for h in hits}), hits=hits)
        print(f"{m}: {len(hits)} training entries overlap {len(rep[m]['test_segments_hit'])} of {len(test)} test segments")
        for h in hits: print("  ", h)
    if a.out: json.dump(rep, open(a.out, "w"), indent=1)
    sys.exit(1 if any(r["n_hits"] for r in rep.values()) else 0)
