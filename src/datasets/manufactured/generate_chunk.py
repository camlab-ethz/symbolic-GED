#!/usr/bin/env python
"""Generate (u, f=L(u)) pairs for a chunk of operators."""
import os, json, argparse, sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from datasets.manufactured.u_sampler import sample_u
from datasets.manufactured.operator_apply import apply_operator
from datasets.manufactured.utils import UniquenessTracker

SCRATCH = os.environ.get('SCRATCH', '/tmp')

def to_py(obj):
    if isinstance(obj, dict): return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, list): return [to_py(v) for v in obj]
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    return obj

def process(idx, row, tracks, k, comp, seed):
    results = []
    fam, dim, t_o, s_o, L = row['family'], int(row['dim']), int(row['temporal_order']), int(row['spatial_order']), row['pde']
    for tr in tracks:
        rng = np.random.default_rng(seed + idx*100 + (0 if tr=='A' else 50))
        tracker, u_l, f_l = UniquenessTracker(), [], []
        for _ in range(100):
            if len(u_l) >= k: break
            try:
                u = sample_u(tr, fam, dim, t_o, s_o, comp, rng)
                u_s = str(u)
                if not tracker.is_unique_u(u_s) or len(u_s)>3000: continue
                f_s = str(apply_operator(L, u, dim, t_o))
                if len(f_s)>10000: continue
                u_l.append(u_s); f_l.append(f_s)
            except: continue
        if len(u_l)>=k:
            results.append(to_py({'track':tr,'family':fam,'dim':dim,'temporal_order':t_o,
                'spatial_order':s_o,'operator_str':L,'k':k,'u_str_list':u_l,'f_str_list':f_l,'meta':{'idx':idx}}))
    return results

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--start', type=int, required=True)
    p.add_argument('--end', type=int, required=True)
    p.add_argument('--k', type=int, default=8)
    p.add_argument('--track', default='both', choices=['A','B','both'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--complexity', type=int, default=2)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    
    df = pd.read_csv(a.csv).iloc[a.start:a.end]
    tracks = ['A','B'] if a.track=='both' else [a.track]
    
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(a.out, 'w') as f:
        for idx, row in df.iterrows():
            for rec in process(idx, row.to_dict(), tracks, a.k, a.complexity, a.seed):
                f.write(json.dumps(rec)+'\n')
                total += 1
            if idx % 100 == 0: print(f"Processed {idx}, total records: {total}")
    print(f"Chunk complete: {total} records -> {a.out}")

if __name__ == '__main__': main()
