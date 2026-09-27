"""Error analysis for blocking misses."""
import os, sys, time, multiprocessing as mp, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocking import Index, hash_terms_par, topk_query
from terms import record_terms
WORK = os.environ.get("BER_WORK", "../../../work")
DATA = os.environ.get("BER_DATA", "../../../dataset")
NQ = int(os.environ.get("BER_NQ", "6000")); SRC = int(os.environ.get("BER_SRC","3"))
CTRY = os.environ.get("BER_CTRY","India")

def main():
    gt = pd.read_csv(f"{DATA}/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    rng = np.random.default_rng(7)
    gt = gt.iloc[np.sort(rng.choice(len(gt), NQ, replace=False))].reset_index(drop=True)
    s1q = gt.source1_entity_id.str.slice(3).astype(np.int64).values
    truth = [set(s.split(",")) if s else set() for s in gt.matched_entity_ids.values]
    d1 = pd.read_parquet(f"{WORK}/train_s1.parquet")
    d1 = d1[d1.eid.isin(set(s1q.tolist()))].reset_index(drop=True)
    pos = {e: i for i, e in enumerate(s1q)}
    d1["qi"] = [pos[e] for e in d1.eid.values]
    dd = pd.read_parquet(f"{WORK}/train_s{SRC}.parquet")
    sub = dd[(dd.country.astype(str)==CTRY).values].reset_index(drop=True); del dd
    q = d1[(d1.country.astype(str)==CTRY).values].reset_index(drop=True)
    with mp.Pool(THREADS) as pool:
        th,ti,tc = hash_terms_par(sub.nname.values, sub.naddr.values, sub.dom.values, pool)
        idx = Index(th,ti,tc,len(sub)); del th,ti,tc
        qh,qi,qc = hash_terms_par(q.nname.values,q.naddr.values,q.dom.values,pool)
    Q,D,S = topk_query(idx,qh,qi,len(q),200,20000,tw=(1.0,1.0,1.0,0.25),b=0.4)
    o=np.lexsort((-S,Q)); Q,D,S=Q[o],D[o],S[o]
    qs=np.searchsorted(Q,np.arange(len(q)+1))
    eid2row={e:i for i,e in enumerate(sub.eid.values)}
    pref=f"S{SRC}-"
    miss=[]; nt=0; got=0
    for k in range(len(q)):
        a,e=qs[k],qs[k+1]
        retr=set(D[a:e].tolist())
        for x in truth[q.qi.values[k]]:
            if not x.startswith(pref): continue
            nt+=1
            r=eid2row.get(int(x[3:]))
            if r is None: continue
            if r in retr: got+=1
            else: miss.append((k,r,S[a] if e>a else 0, e-a))
    print(f"recall@200 = {got/nt:.4f}  misses={len(miss)}  ntrue={nt}")
    # why missed: shared terms?
    nosh=0; details=[]
    for (k,r,tops,ncand) in miss[:400]:
        t1=set(record_terms(q.nname.values[k],q.naddr.values[k],q.dom.values[k]))
        t2=set(record_terms(sub.nname.values[r],sub.naddr.values[r],sub.dom.values[r]))
        sh=t1&t2
        if not sh: nosh+=1
        details.append((k,r,sh,tops,ncand))
    print(f"of first 400 misses: {nosh} share NO term at all ({nosh/len(details):.2%})")
    shown=0
    for (k,r,sh,tops,ncand) in details:
        if shown>=14: continue
        # idf of shared
        hh=np.fromiter((hash(x)&0xFFFFFFFFFFFFFFFF for x in sh),dtype=np.uint64,count=len(sh))
        tid=idx.lookup(hh) if len(sh) else np.empty(0,np.int64)
        info=sorted([(sh_i, int(idx.df[t]) if t>=0 else -1) for sh_i,t in zip(sh,tid)],key=lambda z:z[1])
        print("-"*100)
        print("S1 :",q.nname.values[k],"||",q.naddr.values[k])
        print("S%d:"%SRC,sub.nname.values[r],"||",sub.naddr.values[r])
        print("   shared(%d):"%len(sh),[(a.replace('\x00',':'),b) for a,b in info[:12]],"topscore=%.1f ncand=%d"%(tops,ncand))
        shown+=1

if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8"); main()
