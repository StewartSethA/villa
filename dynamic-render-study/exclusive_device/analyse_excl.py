import json,numpy as np,collections
import sys
R=json.load(open(sys.argv[1] if len(sys.argv)>1 else "results_excl.json")); rng=np.random.default_rng(0)
name={"P":"stored read (pre-rendered surface-volume zarr)","C":"CPU vc_render_tifxyz (16 threads)","G":"GPU render_gpu"}
print("rows",len(R),"segments",len({r["segment"] for r in R}),"windows",len({(r["segment"],r["window"]) for r in R}))
out={}
for st in ("cold","warm"):
    for p in "PCG":
        x=[r for r in R if r["path"]==p and r["state"]==st]; by=collections.defaultdict(list)
        for r in x: by[(r["segment"],r["window"])].append(r["wall_s"])
        med=np.array([np.median(v) for v in by.values()]); w=np.array([r["wall_s"] for r in x])
        bs=[np.median(rng.choice(med,len(med))) for _ in range(2000)]
        out[f"{st}_{p}"]=dict(n_calls=len(x),n_windows=len(med),p10=float(np.percentile(w,10)),p50=float(np.percentile(w,50)),p90=float(np.percentile(w,90)),median_of_window_medians=float(np.median(med)),ci95=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))])
        print(st,name[p],"calls",len(x),"windows",len(med),"wall s p10/p50/p90 %.3f/%.3f/%.3f"%(np.percentile(w,10),np.percentile(w,50),np.percentile(w,90)),"CI95 of median over windows [%.3f, %.3f]"%tuple(out[f"{st}_{p}"]["ci95"]))
# determinism: md5 per (window,path) identical across repeats?
g=collections.defaultdict(set)
for r in R: g[(r["segment"],r["window"],r["path"])].add(r["md5"])
for p in "PCG": print(p,"windows with >1 distinct md5 across the 5 repeats:",sum(len(v)>1 for k,v in g.items() if k[2]==p),"of",sum(1 for k in g if k[2]==p))
fe=lambda p:[r["frac_equal_vs_stored"] for r in R if r["path"]==p and "frac_equal_vs_stored" in r]
for p in "CG": v=np.array(fe(p)); print(p,"fraction of voxels equal to stored: p10/p50/p90 %.5f/%.5f/%.5f"%tuple(np.percentile(v,[10,50,90])),"byte-identical calls",int((v==1).sum()),"of",len(v),"max |diff| p50/max",int(np.median([r["max_abs_vs_stored"] for r in R if r["path"]==p and "max_abs_vs_stored" in r])),max(r["max_abs_vs_stored"] for r in R if r["path"]==p and "max_abs_vs_stored" in r))
ex=collections.defaultdict(lambda:collections.defaultdict(list))
for r in R:
    if r["path"]=="G" and r["state"]=="cold" and r["repeat"]==0: pass
first=[r for r in R if r["path"]=="G" and r["repeat"]==0][:1]
c=[r["wall_s"] for r in R if r["path"]=="C"]; g_=[r["wall_s"] for r in R if r["path"]=="G"]
print("speedup GPU vs CPU (median of per-window medians):",round(np.median(c)/np.median(g_),2))
json.dump(out,open(sys.argv[2] if len(sys.argv)>2 else "summary_excl.json","w"),indent=1)
