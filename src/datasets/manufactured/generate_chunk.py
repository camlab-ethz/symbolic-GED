#!/usr/bin/env python
"""Generate (u, f=L(u)) pairs for a chunk of operators with proper filtering."""
import os
import json
import argparse
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from datasets.manufactured.u_sampler import sample_u
from datasets.manufactured.operator_apply import apply_operator
from datasets.manufactured.utils import UniquenessTracker, canonical_print, canonical_hash
from datasets.manufactured.filters import apply_all_filters

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCRATCH = os.environ.get('SCRATCH', '/tmp')

# Retry budgets
MAX_ATTEMPTS_PER_U = 400
MAX_ATTEMPTS_PER_OPERATOR = 3000


def to_py(obj):
    """Convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_py(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def process(idx, row, tracks, k, comp, seed):
    """
    Process a single operator and generate (u, f) pairs for requested tracks.
    
    Uses canonical printing for uniqueness, applies filters, and tracks rejections.
    """
    results = []
    fam = row['family']
    dim = int(row['dim'])
    t_o = int(row['temporal_order'])
    s_o = int(row['spatial_order'])
    L = row['pde']
    
    for tr in tracks:
        # Seed deterministically per operator + track
        rng = np.random.default_rng(seed + idx * 100 + (0 if tr == 'A' else 50))
        
        # Reset tracker for this (operator, track) pair
        tracker = UniquenessTracker()
        
        u_list = []
        f_list = []
        u_canon_list = []  # For set-level uniqueness
        
        # Track rejection reasons
        rejections = {
            'duplicate': 0,
            'complexity': 0,
            'stability': 0,
            'informative': 0,
            'operator_error': 0,
            'filter_error': 0,
        }
        
        attempts = 0
        while len(u_list) < k and attempts < MAX_ATTEMPTS_PER_OPERATOR:
            attempts += 1
            
            try:
                # Sample u
                u_expr = sample_u(tr, fam, dim, t_o, s_o, comp, rng)
                
                # Canonical form for uniqueness
                u_canon = canonical_print(u_expr, do_simplify=False)
                
                # Check per-set uniqueness
                if not tracker.is_unique_u(u_canon):
                    rejections['duplicate'] += 1
                    continue
                
                # Apply operator to get f
                try:
                    f_expr = apply_operator(L, u_expr, dim, t_o, simplify_level="light")
                except Exception:
                    rejections['operator_error'] += 1
                    continue
                
                # Apply filters
                try:
                    filter_result = apply_all_filters(
                        u_expr, f_expr, L, dim, t_o, track=tr
                    )
                except Exception:
                    rejections['filter_error'] += 1
                    continue
                
                if not filter_result['passed']:
                    # Track specific rejection reason
                    if not filter_result.get('complexity', True):
                        rejections['complexity'] += 1
                    elif not filter_result.get('stability', True):
                        rejections['stability'] += 1
                    elif not filter_result.get('informative', True):
                        rejections['informative'] += 1
                    continue
                
                # Passed all filters - add to lists
                f_canon = canonical_print(f_expr, do_simplify=False)
                
                u_list.append(u_canon)
                f_list.append(f_canon)
                u_canon_list.append(u_canon)
                
            except Exception:
                continue
        
        # Only emit record if we got enough pairs
        if len(u_list) >= k:
            # Check set-level uniqueness
            if tracker.is_unique_set(u_canon_list):
                record = to_py({
                    'track': tr,
                    'family': fam,
                    'dim': dim,
                    'temporal_order': t_o,
                    'spatial_order': s_o,
                    'operator_str': L,
                    'k': k,
                    'u_str_list': u_list,
                    'f_str_list': f_list,
                    'meta': {
                        'idx': idx,
                        'attempts': attempts,
                        'rejection_stats': rejections,
                    },
                })
                results.append(record)
        else:
            # Log failure
            logger.warning(f"Operator {idx} track {tr}: only got {len(u_list)}/{k} pairs after {attempts} attempts")
    
    return results


def main():
    p = argparse.ArgumentParser(description='Generate (u, f=L(u)) pairs for a chunk of operators')
    p.add_argument('--csv', required=True, help='Path to operators CSV')
    p.add_argument('--start', type=int, required=True, help='Start index')
    p.add_argument('--end', type=int, required=True, help='End index')
    p.add_argument('--k', type=int, default=8, help='Number of (u,f) pairs per operator')
    p.add_argument('--track', default='both', choices=['A', 'B', 'both'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--complexity', type=int, default=2, choices=[1, 2, 3])
    p.add_argument('--out', required=True, help='Output JSONL path')
    a = p.parse_args()
    
    df = pd.read_csv(a.csv).iloc[a.start:a.end]
    tracks = ['A', 'B'] if a.track == 'both' else [a.track]
    
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    
    total_records = 0
    total_operators = len(df)
    
    logger.info(f"Processing operators {a.start}-{a.end} ({total_operators} total)")
    logger.info(f"Tracks: {tracks}, k={a.k}, complexity={a.complexity}")
    
    with open(a.out, 'w') as f:
        for i, (idx, row) in enumerate(df.iterrows()):
            records = process(idx, row.to_dict(), tracks, a.k, a.complexity, a.seed)
            for rec in records:
                f.write(json.dumps(rec) + '\n')
                total_records += 1
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{total_operators} operators, {total_records} records")
    
    logger.info(f"Chunk complete: {total_records} records -> {a.out}")


if __name__ == '__main__':
    main()
