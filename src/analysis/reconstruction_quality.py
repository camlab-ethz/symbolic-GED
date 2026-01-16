#!/usr/bin/env python3
"""Analyze reconstruction quality for all 4 VAE models.

This script decodes the validation and test sets and classifies each output as:
1. Exact match (✓): Decoded PDE = Input PDE
2. Structure preserved (≈): Same family, only coefficients differ  
3. Family changed (↔): Valid PDE, but different family
4. Invalid (✗): Syntax error or incomplete expression

Usage:
    python -m analysis.reconstruction_quality \
        --config configs/paths_48000_fixed.yaml \
        --splits val,test \
        --outdir analysis_results/reconstruction_quality_48000
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vae.module import VAEModule
from pde.grammar import decode_production_sequence, PROD_COUNT
from pde.chr_tokenizer import PDETokenizer
from analysis.pde_classifier import PDEClassifier
from analysis.physics import is_valid_pde


def load_model(checkpoint_path: str, device: str = 'cuda') -> VAEModule:
    """Load VAE model from checkpoint."""
    model = VAEModule.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    model = model.to(device)
    return model


def get_tokenization_type(checkpoint_path: str) -> str:
    """Infer tokenization type from checkpoint path."""
    if 'grammar' in checkpoint_path.lower():
        return 'grammar'
    elif 'token' in checkpoint_path.lower():
        return 'token'
    else:
        raise ValueError(f"Cannot infer tokenization from path: {checkpoint_path}")


def decode_grammar_ids(prod_ids: np.ndarray) -> Optional[str]:
    """Decode grammar production IDs to PDE string."""
    try:
        # Filter out padding (-1 or values >= PROD_COUNT)
        valid_ids = [int(p) for p in prod_ids if 0 <= p < PROD_COUNT]
        if not valid_ids:
            return None
        pde_str = decode_production_sequence(valid_ids)
        return pde_str
    except Exception:
        return None


def decode_token_ids(token_ids: np.ndarray, tokenizer: PDETokenizer) -> Optional[str]:
    """Decode token IDs to PDE string (infix notation)."""
    try:
        # Convert to list of ints and truncate at EOS token (id=2)
        token_list = []
        for t in token_ids:
            tid = int(t)
            if tid == 2:  # EOS token - stop here
                break
            token_list.append(tid)
        
        if not token_list:
            return None
        
        # Use decode_to_infix to convert prefix notation back to infix
        pde_str = tokenizer.decode_to_infix(token_list)
        if pde_str and pde_str.strip():
            return pde_str
        return None
    except Exception:
        return None


def extract_structure(pde_str: str) -> Optional[str]:
    """Extract coefficient-agnostic structure from PDE string.
    
    Examples:
        'dt(u) - 1.935*dxx(u)' -> 'dt(u) - C*dxx(u)'
        'dtt(u) + 0.5*dt(u) - 2.0*dxx(u)' -> 'dtt(u) + C*dt(u) - C*dxx(u)'
    """
    if pde_str is None:
        return None
    
    # Replace numeric coefficients followed by * with 'C*'
    # Pattern: digits with optional decimal, followed by *
    # But preserve the operator before it
    pattern = r'(\d+\.?\d*)\s*\*'
    structure = re.sub(pattern, 'C*', pde_str)
    
    # Also replace standalone numbers at the end or after operators
    # (like in Poisson: dxx(u) + dyy(u) - 3.565)
    pattern2 = r'([+\-])\s*(\d+\.?\d*)(\s*)$'
    structure = re.sub(pattern2, r'\1 C\3', structure)
    
    # Replace standalone numbers in the middle too
    pattern3 = r'([+\-])\s*(\d+\.?\d*)(\s*[+\-])'
    structure = re.sub(pattern3, r'\1 C\3', structure)
    
    # Remove all whitespace for consistent comparison
    structure = structure.replace(' ', '')
    
    return structure


def classify_reconstruction(
    input_pde: str,
    decoded_pde: Optional[str],
    input_family: str,
    classifier: PDEClassifier
) -> Tuple[str, Dict]:
    """Classify the reconstruction quality.
    
    Returns:
        category: One of 'exact', 'structure_preserved', 'structure_changed', 'family_changed', 'invalid'
        details: Dictionary with additional info
    """
    details = {
        'input_pde': input_pde,
        'decoded_pde': decoded_pde,
        'input_family': input_family,
        'decoded_family': None,
        'input_structure': None,
        'decoded_structure': None,
    }
    
    # Check if decoding failed
    if decoded_pde is None or decoded_pde.strip() == '':
        return 'invalid', details
    
    # Normalize decoded PDE whitespace for consistent classification
    # Remove spaces around operators to match input format
    decoded_pde_normalized = decoded_pde.replace(' ', '')
    
    # Check syntax validity
    if not is_valid_pde(decoded_pde_normalized):
        return 'invalid', details
    
    # Classify the decoded PDE (use normalized version for consistent classification)
    try:
        decoded_labels = classifier.classify(decoded_pde_normalized)
        decoded_family = decoded_labels.family if decoded_labels.family else 'unknown'
        details['decoded_family'] = decoded_family
    except Exception:
        return 'invalid', details
    
    # Normalize both PDEs for comparison (remove all whitespace)
    input_normalized = input_pde.replace(' ', '')
    decoded_normalized = decoded_pde_normalized
    
    # Check exact match
    if input_normalized == decoded_normalized:
        details['input_structure'] = extract_structure(input_pde)
        details['decoded_structure'] = extract_structure(decoded_pde)
        return 'exact', details
    
    # Extract structures
    input_structure = extract_structure(input_pde)
    decoded_structure = extract_structure(decoded_pde)
    details['input_structure'] = input_structure
    details['decoded_structure'] = decoded_structure
    
    # Check if same family
    if decoded_family == input_family:
        # Same family - check if structure is preserved
        if input_structure == decoded_structure:
            return 'structure_preserved', details
        else:
            # Same family but structure differs (e.g., u^3 -> u^355)
            return 'structure_changed', details
    else:
        # Different family
        return 'family_changed', details


def analyze_model(
    model: VAEModule,
    tokenization: str,
    input_ids: np.ndarray,
    metadata_df: pd.DataFrame,
    indices: np.ndarray,
    classifier: PDEClassifier,
    token_tokenizer: Optional[PDETokenizer],
    device: str = 'cuda',
    batch_size: int = 256
) -> pd.DataFrame:
    """Analyze reconstruction quality for one model."""
    
    results = []
    n_samples = len(indices)
    
    for batch_start in tqdm(range(0, n_samples, batch_size), desc="Analyzing"):
        batch_end = min(batch_start + batch_size, n_samples)
        batch_indices = indices[batch_start:batch_end]
        
        # Get input sequences
        batch_ids = input_ids[batch_indices]
        batch_ids_tensor = torch.tensor(batch_ids, dtype=torch.long, device=device)
        
        # Encode to get latent representation
        with torch.no_grad():
            # Create one-hot encoding (B, T, P)
            # The encoder expects (B, T, P) and internally permutes to (B, P, T)
            if tokenization == 'grammar':
                # For grammar, input is production IDs
                x_onehot = torch.zeros(len(batch_ids), model.max_length, model.P, device=device)
                for i, seq in enumerate(batch_ids):
                    for t, pid in enumerate(seq):
                        if 0 <= pid < model.P:
                            x_onehot[i, t, pid] = 1.0
                mu, logvar = model.encode(x_onehot)  # Use model.encode which handles shape
            else:
                # For token, similar approach
                x_onehot = torch.zeros(len(batch_ids), model.max_length, model.P, device=device)
                for i, seq in enumerate(batch_ids):
                    for t, tid in enumerate(seq):
                        if 0 <= tid < model.P:
                            x_onehot[i, t, tid] = 1.0
                mu, logvar = model.encode(x_onehot)  # Use model.encode which handles shape
            
            # Use mean for deterministic decoding
            z = mu
            
            # Decode
            if tokenization == 'grammar':
                # Use constrained decoding
                decoded_ids = model.generate_constrained(z, greedy=True)
                decoded_ids = decoded_ids.cpu().numpy()
            else:
                # Greedy decoding for token
                logits = model.decoder(z)
                decoded_ids = logits.argmax(dim=-1).cpu().numpy()
        
        # Process each sample in batch
        for i, idx in enumerate(batch_indices):
            row = metadata_df.iloc[idx]
            input_pde = row['pde']
            input_family = row['family']
            
            # Decode to string
            if tokenization == 'grammar':
                decoded_pde = decode_grammar_ids(decoded_ids[i])
            else:
                decoded_pde = decode_token_ids(decoded_ids[i], token_tokenizer)
            
            # Classify
            category, details = classify_reconstruction(
                input_pde, decoded_pde, input_family, classifier
            )
            
            results.append({
                'idx': int(idx),
                'input_pde': input_pde,
                'decoded_pde': decoded_pde,
                'input_family': input_family,
                'decoded_family': details.get('decoded_family'),
                'category': category,
                'input_structure': details.get('input_structure'),
                'decoded_structure': details.get('decoded_structure'),
            })
    
    return pd.DataFrame(results)


def summarize_results(df: pd.DataFrame) -> Dict:
    """Compute summary statistics."""
    total = len(df)
    
    categories = ['exact', 'structure_preserved', 'structure_changed', 'family_changed', 'invalid']
    summary = {
        'total': total,
        'counts': {},
        'percentages': {},
    }
    
    for cat in categories:
        count = (df['category'] == cat).sum()
        summary['counts'][cat] = int(count)
        summary['percentages'][cat] = round(100 * count / total, 2) if total > 0 else 0.0
    
    # Also compute "valid" = exact + structure_preserved + family_changed
    valid_count = total - summary['counts']['invalid']
    summary['counts']['valid'] = valid_count
    summary['percentages']['valid'] = round(100 * valid_count / total, 2) if total > 0 else 0.0
    
    # Family preservation rate (among valid)
    if valid_count > 0:
        family_preserved = summary['counts']['exact'] + summary['counts']['structure_preserved']
        summary['percentages']['family_preserved'] = round(100 * family_preserved / valid_count, 2)
    else:
        summary['percentages']['family_preserved'] = 0.0
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze reconstruction quality")
    parser.add_argument('--config', type=str, default='configs/paths_48000_fixed.yaml',
                        help='Path config file')
    parser.add_argument('--splits', type=str, default='val,test',
                        help='Comma-separated splits to analyze')
    parser.add_argument('--outdir', type=str, 
                        default='analysis_results/reconstruction_quality_48000',
                        help='Output directory')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for encoding/decoding')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    args = parser.parse_args()
    
    # Load paths config
    import yaml
    with open(args.config) as f:
        paths = yaml.safe_load(f)
    
    # Setup paths - handle nested config structure
    dataset_cfg = paths.get('dataset', paths)
    training_cfg = paths.get('training', paths)
    
    dataset_csv = dataset_cfg.get('csv_metadata', dataset_cfg.get('dataset_csv', 'data/raw/pde_dataset_48000_fixed.csv'))
    splits_dir = Path(dataset_cfg.get('split_dir', dataset_cfg.get('splits_dir', 'data/splits_48000_fixed')))
    tokenized_dir = Path(dataset_cfg.get('tokenized_dir', 'data/tokenized_48000_fixed'))
    
    # Grammar and token IDs paths
    grammar_ids = tokenized_dir / 'grammar_full.npy'
    token_ids = tokenized_dir / 'token_full.npy'
    
    # Fallback to individual files if full doesn't exist
    if not grammar_ids.exists():
        grammar_ids = tokenized_dir / 'grammar' / 'train.npy'  # Will need to combine
    if not token_ids.exists():
        token_ids = tokenized_dir / 'token' / 'train.npy'
    
    checkpoints_dir = Path(training_cfg.get('checkpoint_root', training_cfg.get('checkpoints_dir', 'checkpoints_48000_fixed')))
    
    # Define model checkpoints
    models_config = {
        'grammar_beta2e4': {
            'tokenization': 'grammar',
            'checkpoint': checkpoints_dir / 'grammar_vae' / 'beta_2e-4_seed_42',
        },
        'grammar_beta1e2': {
            'tokenization': 'grammar', 
            'checkpoint': checkpoints_dir / 'grammar_vae' / 'beta_0.01_seed_42',
        },
        'token_beta2e4': {
            'tokenization': 'token',
            'checkpoint': checkpoints_dir / 'token_vae' / 'beta_2e-4_seed_42',
        },
        'token_beta1e2': {
            'tokenization': 'token',
            'checkpoint': checkpoints_dir / 'token_vae' / 'beta_0.01_seed_42',
        },
    }
    
    # Find best checkpoint in each directory
    for name, cfg in models_config.items():
        ckpt_dir = cfg['checkpoint']
        if ckpt_dir.exists():
            # First try direct .ckpt files
            ckpts = list(ckpt_dir.glob('*.ckpt'))
            if ckpts:
                cfg['checkpoint'] = str(ckpts[0])
            else:
                # Try nested directories (e.g., best-epoch=600-val/seq_acc=0.9969.ckpt)
                ckpts = list(ckpt_dir.glob('*/*.ckpt'))
                if ckpts:
                    cfg['checkpoint'] = str(ckpts[0])
                else:
                    # Try even deeper
                    ckpts = list(ckpt_dir.glob('**/*.ckpt'))
                    if ckpts:
                        cfg['checkpoint'] = str(ckpts[0])
                    else:
                        print(f"Warning: No checkpoint found in {ckpt_dir}")
                        cfg['checkpoint'] = None
        else:
            print(f"Warning: Checkpoint dir not found: {ckpt_dir}")
            cfg['checkpoint'] = None
    
    # Load metadata
    print(f"Loading metadata from {dataset_csv}...")
    metadata_df = pd.read_csv(dataset_csv)
    
    # Load tokenized data
    print(f"Loading grammar IDs from {grammar_ids}...")
    grammar_ids_data = np.load(grammar_ids)
    
    print(f"Loading token IDs from {token_ids}...")
    token_ids_data = np.load(token_ids)
    
    # Initialize classifier and tokenizer
    classifier = PDEClassifier()
    token_tokenizer = PDETokenizer()
    
    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    splits = [s.strip() for s in args.splits.split(',')]
    all_summaries = []
    
    for split in splits:
        print(f"\n{'='*60}")
        print(f"Processing {split} split...")
        print(f"{'='*60}")
        
        # Load split indices
        indices_file = splits_dir / f'{split}_indices.npy'
        if not indices_file.exists():
            print(f"Warning: {indices_file} not found, skipping")
            continue
        indices = np.load(indices_file)
        print(f"Loaded {len(indices)} indices for {split}")
        
        split_outdir = outdir / split
        split_outdir.mkdir(parents=True, exist_ok=True)
        
        # Process each model
        for model_name, cfg in models_config.items():
            if cfg['checkpoint'] is None:
                print(f"Skipping {model_name} (no checkpoint)")
                continue
                
            print(f"\n--- {model_name} ---")
            tokenization = cfg['tokenization']
            
            # Load model
            print(f"Loading model from {cfg['checkpoint']}...")
            model = load_model(cfg['checkpoint'], args.device)
            
            # Get appropriate input data
            if tokenization == 'grammar':
                input_ids = grammar_ids_data
            else:
                input_ids = token_ids_data
            
            # Analyze
            results_df = analyze_model(
                model=model,
                tokenization=tokenization,
                input_ids=input_ids,
                metadata_df=metadata_df,
                indices=indices,
                classifier=classifier,
                token_tokenizer=token_tokenizer if tokenization == 'token' else None,
                device=args.device,
                batch_size=args.batch_size
            )
            
            # Save detailed results
            results_file = split_outdir / f'{model_name}_detailed.csv'
            results_df.to_csv(results_file, index=False)
            print(f"Saved detailed results to {results_file}")
            
            # Compute and save summary
            summary = summarize_results(results_df)
            summary['model'] = model_name
            summary['tokenization'] = tokenization
            summary['split'] = split
            
            print(f"\nSummary for {model_name} on {split}:")
            print(f"  Total: {summary['total']}")
            print(f"  Exact match:        {summary['counts']['exact']:5d} ({summary['percentages']['exact']:.2f}%)")
            print(f"  Structure preserved:{summary['counts']['structure_preserved']:5d} ({summary['percentages']['structure_preserved']:.2f}%)")
            print(f"  Structure changed:  {summary['counts']['structure_changed']:5d} ({summary['percentages']['structure_changed']:.2f}%)")
            print(f"  Family changed:     {summary['counts']['family_changed']:5d} ({summary['percentages']['family_changed']:.2f}%)")
            print(f"  Invalid:            {summary['counts']['invalid']:5d} ({summary['percentages']['invalid']:.2f}%)")
            print(f"  ---")
            print(f"  Valid total:        {summary['counts']['valid']:5d} ({summary['percentages']['valid']:.2f}%)")
            print(f"  Family preserved:   {summary['percentages']['family_preserved']:.2f}% (of valid)")
            
            all_summaries.append(summary)
            
            # Free model memory
            del model
            torch.cuda.empty_cache()
    
    # Save combined summary
    if all_summaries:
        summary_rows = []
        for s in all_summaries:
            row = {
                'split': s['split'],
                'model': s['model'],
                'tokenization': s['tokenization'],
                'total': s['total'],
                'exact': s['counts']['exact'],
                'exact_pct': s['percentages']['exact'],
                'structure_preserved': s['counts']['structure_preserved'],
                'structure_preserved_pct': s['percentages']['structure_preserved'],
                'structure_changed': s['counts']['structure_changed'],
                'structure_changed_pct': s['percentages']['structure_changed'],
                'family_changed': s['counts']['family_changed'],
                'family_changed_pct': s['percentages']['family_changed'],
                'invalid': s['counts']['invalid'],
                'invalid_pct': s['percentages']['invalid'],
                'valid_pct': s['percentages']['valid'],
                'family_preserved_pct': s['percentages']['family_preserved'],
            }
            summary_rows.append(row)
        
        summary_df = pd.DataFrame(summary_rows)
        summary_file = outdir / 'summary.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"\n{'='*60}")
        print(f"Saved combined summary to {summary_file}")
        
        # Also save as JSON
        with open(outdir / 'summary.json', 'w') as f:
            json.dump(all_summaries, f, indent=2)


if __name__ == '__main__':
    main()
