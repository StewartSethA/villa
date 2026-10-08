#!/usr/bin/env python3
"""Neutralise private path strings in the copied umbilicus provenance fields (keeps basenames). usage: scrub_umbilicus.py DIR"""
import glob, json, re, sys
bad = re.compile(r"(/home/|/mnt/|192\.168|seth|jacob|phi1)")
def scrub(o):
    if isinstance(o, dict): return {k: scrub(v) for k, v in o.items()}
    if isinstance(o, list): return [scrub(v) for v in o]
    if isinstance(o, str) and bad.search(o): return re.sub(r"(/[\w.\-@]+)+/([\w.\-]+)", r"<path>/\2", o)
    return o
for p in glob.glob(sys.argv[1] + "/*/umbilicus.json"):
    json.dump(scrub(json.load(open(p))), open(p, "w"), separators=(",", ":"))
    t = open(p).read()
    assert not bad.search(t), p
