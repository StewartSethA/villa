"""Anonymous, resumable pull of one prefix of the open-data bucket (no aws cli needed).  usage: s3_pull.py PREFIX DEST [--threads 32]   -> DEST/<relative key> ; a file is skipped when it already exists with the listed size; DEST/.done marks a complete prefix."""
import sys, os, re, urllib.request, urllib.parse, time
from concurrent.futures import ThreadPoolExecutor
B = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
def listing(prefix):
    tok = None; out = []
    while True:
        u = f"{B}/?list-type=2&prefix={urllib.parse.quote(prefix)}" + (f"&continuation-token={urllib.parse.quote(tok)}" if tok else ""); x = urllib.request.urlopen(u, timeout=120).read().decode()
        out += [(k, int(s)) for k, s in re.findall(r"<Key>([^<]*)</Key>.*?<Size>(\d+)</Size>", x, flags=re.S)]
        m = re.search(r"<NextContinuationToken>([^<]*)</NextContinuationToken>", x)
        if not m: return out
        tok = m.group(1)
def get(job):
    k, sz, dest, prefix = job; p = os.path.join(dest, k[len(prefix):].lstrip("/"))
    if os.path.exists(p) and os.path.getsize(p) == sz: return 0
    os.makedirs(os.path.dirname(p), exist_ok=True)
    for i in range(6):
        try:
            urllib.request.urlretrieve(B + "/" + urllib.parse.quote(k), p + ".part"); os.replace(p + ".part", p)
            return 0 if os.path.getsize(p) == sz else 1
        except Exception: time.sleep(2 + i)
    return 1
if __name__ == "__main__":
    prefix, dest = sys.argv[1], sys.argv[2]; th = int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 32
    prefix = prefix.rstrip("/") + "/"; ks = [(k, s) for k, s in listing(prefix) if s > 0 or True]; os.makedirs(dest, exist_ok=True)
    with ThreadPoolExecutor(th) as ex: bad = sum(ex.map(get, [(k, s, dest, prefix) for k, s in ks]))
    print(f"{prefix}: {len(ks)} files, {sum(s for _, s in ks)/1e9:.2f} GB, failed {bad}", flush=True)
    if bad == 0: open(os.path.join(dest, ".done"), "w").write(time.strftime("%F %T"))
    sys.exit(1 if bad else 0)
