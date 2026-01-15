#!/usr/bin/env python3
"""Complete Grammar vs Token VAE Comparison Pipeline.

This script runs the full comparison between grammar-based and token-based
tokenization for PDE VAEs, generating tables and figures for a paper.

Usage:
    python scripts/run_tokenization_comparison.py

Output:
    experiments/comparison/
    ├── latents/
    │   ├── grammar_latents.npz
    │   └── token_latents.npz
    ├── results/
    │   ├── clustering_comparison.json
    │   ├── classification_comparison.json
    │   └── summary_table.csv
    └── figures/
        ├── latent_tsne_grammar.png
        ├── latent_tsne_token.png
        └── comparison_barplot.png
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Setup paths
SCRIPT_DIR = Path(__file__).parent
LIBGEN_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(LIBGEN_DIR))

from vae.module import VAEModule, GrammarVAEModule
from analysis import (
    compute_clustering_metrics,
    compute_all_clustering,
    train_classifier,
    assign_physics_labels,
    PDE_PHYSICS,
    classify_pde_type,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    'grammar_checkpoint': 'checkpoints/grammar_vae/best-epoch=189-seqacc=val/seq_acc=0.9907.ckpt',
    'token_checkpoint': 'checkpoints/token_vae/best-epoch=314-seqacc=val/seq_acc=0.9841.ckpt',
    'dataset_csv': 'pde_dataset_48444_clean.csv',
    'output_dir': 'experiments/comparison',
    'batch_size': 256,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Data paths
    'grammar_ids': 'examples_out/prod_48444_ids_int16_clean.npy',
    'grammar_masks': 'examples_out/prod_48444_masks_clean.npy',
    'token_ids': 'examples_out/token_48444_ids_int16_clean.npy',
    'token_masks': 'examples_out/token_48444_masks_clean.npy',
    
    # Model params (will be loaded from checkpoint)
    'grammar_vocab_size': 53,
    'grammar_max_length': 114,
    'token_vocab_size': 82,
    'token_max_length': 62,
}


# ============================================================================
# Utility Functions
# ============================================================================

def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load VAE model from checkpoint."""
    logger.info(f"Loading model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    hparams = checkpoint['hyper_parameters']
    
    # Handle both old and new module naming
    model = GrammarVAEModule(
        P=hparams['P'],
        max_length=hparams['max_length'],
        z_dim=hparams.get('z_dim', 26),
        lr=hparams.get('lr', 0.001),
        beta=hparams.get('beta', 1e-5),
        encoder_hidden=hparams.get('encoder_hidden', 128),
        encoder_conv_layers=hparams.get('encoder_conv_layers', 3),
        encoder_kernel=hparams.get('encoder_kernel', [7, 7, 7]),
        decoder_hidden=hparams.get('decoder_hidden', 80),
        decoder_layers=hparams.get('decoder_layers', 3),
        decoder_dropout=hparams.get('decoder_dropout', 0.1),
    )
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"  Loaded: P={hparams['P']}, z_dim={hparams.get('z_dim', 26)}")
    return model, hparams


def extract_latents(model, ids_path: str, batch_size: int = 256, device: str = 'cuda'):
    """Extract latent vectors from model.
    
    Uses model.P for vocab_size and model.max_length for sequence length.
    """
    logger.info(f"Extracting latents from {ids_path}")
    
    # Get vocab_size and max_length from model
    vocab_size = model.P
    max_length = model.max_length
    
    # Load data
    ids = np.load(ids_path)
    N, T = ids.shape
    
    # Use model's max_length
    T = min(T, max_length)
    
    all_mu = []
    all_logvar = []
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, N, batch_size), desc="Encoding"):
            batch_ids = ids[i:i+batch_size]
            
            # Convert to one-hot using model's vocab_size
            batch_onehot = np.zeros((len(batch_ids), max_length, vocab_size), dtype=np.float32)
            for b, seq in enumerate(batch_ids):
                for t, idx in enumerate(seq[:max_length]):
                    if 0 <= idx < vocab_size:
                        batch_onehot[b, t, idx] = 1.0
            
            batch_tensor = torch.from_numpy(batch_onehot).to(device)
            mu, logvar = model.encoder(batch_tensor)
            
            all_mu.append(mu.cpu().numpy())
            all_logvar.append(logvar.cpu().numpy())
    
    mu = np.concatenate(all_mu, axis=0)
    logvar = np.concatenate(all_logvar, axis=0)
    
    logger.info(f"  Extracted {N} latents, shape: {mu.shape}")
    return mu, logvar


def load_physics_labels(csv_path: str) -> Dict[str, np.ndarray]:
    """Load physics labels from dataset CSV."""
    logger.info(f"Loading physics labels from {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    labels = {
        'family': df['family'].values,
        'dim': df['dim'].values.astype(int),
        'temporal_order': df['temporal_order'].values.astype(int),
        'spatial_order': df['spatial_order'].values.astype(int),
        'nonlinear': df['nonlinear'].values.astype(bool),
    }
    
    # Add derived labels
    labels['pde_type'] = np.array([
        PDE_PHYSICS.get(f, {}).get('type', 'unknown') 
        for f in labels['family']
    ])
    
    labels['linearity'] = np.array([
        'nonlinear' if nl else 'linear' 
        for nl in labels['nonlinear']
    ])
    
    logger.info(f"  Loaded {len(df)} samples with {len(labels)} label types")
    return labels


def compute_comparison_metrics(grammar_latents: np.ndarray, 
                               token_latents: np.ndarray,
                               labels: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Compute clustering and classification metrics for comparison."""
    results = {
        'grammar': {},
        'token': {},
        'comparison': {}
    }
    
    label_names = ['family', 'pde_type', 'linearity', 'dim', 'spatial_order']
    
    for label_name in label_names:
        if label_name not in labels:
            continue
            
        label_values = labels[label_name]
        
        # Skip if too few unique values
        unique = np.unique(label_values)
        if len(unique) < 2:
            continue
        
        logger.info(f"Computing metrics for {label_name} ({len(unique)} classes)")
        
        # Grammar metrics
        g_clust = compute_clustering_metrics(grammar_latents, label_values, label_name)
        g_class = train_classifier(grammar_latents, label_values, label_name)
        
        # Token metrics
        t_clust = compute_clustering_metrics(token_latents, label_values, label_name)
        t_class = train_classifier(token_latents, label_values, label_name)
        
        results['grammar'][label_name] = {
            'clustering': g_clust,
            'classification': g_class
        }
        results['token'][label_name] = {
            'clustering': t_clust,
            'classification': t_class
        }
        
        # Comparison
        g_nmi = g_clust.get('nmi', 0)
        t_nmi = t_clust.get('nmi', 0)
        g_acc = g_class.get('accuracy_mean', 0)
        t_acc = t_class.get('accuracy_mean', 0)
        
        results['comparison'][label_name] = {
            'grammar_nmi': g_nmi,
            'token_nmi': t_nmi,
            'nmi_winner': 'Grammar' if g_nmi > t_nmi else 'Token',
            'nmi_diff': abs(g_nmi - t_nmi),
            'grammar_acc': g_acc,
            'token_acc': t_acc,
            'acc_winner': 'Grammar' if g_acc > t_acc else 'Token',
            'acc_diff': abs(g_acc - t_acc),
        }
    
    return results


def generate_summary_table(results: Dict[str, Any]) -> pd.DataFrame:
    """Generate summary comparison table."""
    rows = []
    
    for label_name, comp in results['comparison'].items():
        rows.append({
            'Physics Property': label_name.replace('_', ' ').title(),
            'Grammar NMI': f"{comp['grammar_nmi']:.3f}",
            'Token NMI': f"{comp['token_nmi']:.3f}",
            'NMI Winner': comp['nmi_winner'],
            'Grammar Acc': f"{comp['grammar_acc']:.3f}",
            'Token Acc': f"{comp['token_acc']:.3f}",
            'Acc Winner': comp['acc_winner'],
        })
    
    return pd.DataFrame(rows)


def generate_latex_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table for paper."""
    latex = """
\\begin{table}[h]
\\centering
\\caption{Grammar vs Token Tokenization: Clustering and Classification Metrics}
\\label{tab:tokenization_comparison}
\\begin{tabular}{lcccccc}
\\toprule
\\textbf{Property} & \\multicolumn{2}{c}{\\textbf{NMI}} & \\textbf{Winner} & \\multicolumn{2}{c}{\\textbf{Accuracy}} & \\textbf{Winner} \\\\
\\cmidrule(lr){2-3} \\cmidrule(lr){5-6}
& Grammar & Token & & Grammar & Token & \\\\
\\midrule
"""
    for _, row in df.iterrows():
        latex += f"{row['Physics Property']} & {row['Grammar NMI']} & {row['Token NMI']} & {row['NMI Winner']} & {row['Grammar Acc']} & {row['Token Acc']} & {row['Acc Winner']} \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    return latex


def print_results(results: Dict[str, Any]):
    """Print formatted results to console."""
    print("\n" + "=" * 80)
    print("GRAMMAR VS TOKEN TOKENIZATION COMPARISON")
    print("=" * 80)
    
    print("\n### Clustering Metrics (NMI - higher is better)")
    print("-" * 60)
    print(f"{'Property':<20} {'Grammar':>12} {'Token':>12} {'Winner':>12}")
    print("-" * 60)
    
    for label, comp in results['comparison'].items():
        g = comp['grammar_nmi']
        t = comp['token_nmi']
        winner = comp['nmi_winner']
        print(f"{label:<20} {g:>12.4f} {t:>12.4f} {winner:>12}")
    
    print("\n### Classification Accuracy (5-fold CV)")
    print("-" * 60)
    print(f"{'Property':<20} {'Grammar':>12} {'Token':>12} {'Winner':>12}")
    print("-" * 60)
    
    for label, comp in results['comparison'].items():
        g = comp['grammar_acc']
        t = comp['token_acc']
        winner = comp['acc_winner']
        print(f"{label:<20} {g:>12.4f} {t:>12.4f} {winner:>12}")
    
    # Summary
    grammar_wins_nmi = sum(1 for c in results['comparison'].values() if c['nmi_winner'] == 'Grammar')
    token_wins_nmi = sum(1 for c in results['comparison'].values() if c['nmi_winner'] == 'Token')
    grammar_wins_acc = sum(1 for c in results['comparison'].values() if c['acc_winner'] == 'Grammar')
    token_wins_acc = sum(1 for c in results['comparison'].values() if c['acc_winner'] == 'Token')
    
    print("\n### Summary")
    print("-" * 60)
    print(f"NMI: Grammar wins {grammar_wins_nmi}/{len(results['comparison'])}, Token wins {token_wins_nmi}/{len(results['comparison'])}")
    print(f"Acc: Grammar wins {grammar_wins_acc}/{len(results['comparison'])}, Token wins {token_wins_acc}/{len(results['comparison'])}")
    print("=" * 80)


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Run the full comparison pipeline."""
    start_time = datetime.now()
    logger.info("Starting Grammar vs Token Comparison Pipeline")
    
    # Create output directories
    output_dir = LIBGEN_DIR / CONFIG['output_dir']
    (output_dir / 'latents').mkdir(parents=True, exist_ok=True)
    (output_dir / 'results').mkdir(parents=True, exist_ok=True)
    (output_dir / 'figures').mkdir(parents=True, exist_ok=True)
    
    # Check if checkpoints exist
    grammar_ckpt = LIBGEN_DIR / CONFIG['grammar_checkpoint']
    token_ckpt = LIBGEN_DIR / CONFIG['token_checkpoint']
    
    if not grammar_ckpt.exists():
        logger.error(f"Grammar checkpoint not found: {grammar_ckpt}")
        return
    if not token_ckpt.exists():
        logger.error(f"Token checkpoint not found: {token_ckpt}")
        return
    
    device = CONFIG['device']
    logger.info(f"Using device: {device}")
    
    # ========== Step 1: Load Models ==========
    logger.info("\n[Step 1/4] Loading models...")
    grammar_model, grammar_hparams = load_model(str(grammar_ckpt), device)
    token_model, token_hparams = load_model(str(token_ckpt), device)
    
    # ========== Step 2: Extract Latents ==========
    logger.info("\n[Step 2/4] Extracting latents...")
    
    grammar_mu, grammar_logvar = extract_latents(
        grammar_model,
        str(LIBGEN_DIR / CONFIG['grammar_ids']),
        CONFIG['batch_size'],
        device
    )
    
    token_mu, token_logvar = extract_latents(
        token_model,
        str(LIBGEN_DIR / CONFIG['token_ids']),
        CONFIG['batch_size'],
        device
    )
    
    # Save latents
    np.savez(output_dir / 'latents' / 'grammar_latents.npz', 
             mu=grammar_mu, logvar=grammar_logvar)
    np.savez(output_dir / 'latents' / 'token_latents.npz',
             mu=token_mu, logvar=token_logvar)
    logger.info("  Saved latents to disk")
    
    # ========== Step 3: Load Labels & Compute Metrics ==========
    logger.info("\n[Step 3/4] Computing comparison metrics...")
    
    labels = load_physics_labels(str(LIBGEN_DIR / CONFIG['dataset_csv']))
    results = compute_comparison_metrics(grammar_mu, token_mu, labels)
    
    # Save results
    with open(output_dir / 'results' / 'comparison_results.json', 'w') as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump(results, f, indent=2, default=convert)
    
    # ========== Step 4: Generate Report ==========
    logger.info("\n[Step 4/4] Generating report...")
    
    # Print results
    print_results(results)
    
    # Generate summary table
    summary_df = generate_summary_table(results)
    summary_df.to_csv(output_dir / 'results' / 'summary_table.csv', index=False)
    
    # Generate LaTeX table
    latex_table = generate_latex_table(summary_df)
    with open(output_dir / 'results' / 'latex_table.tex', 'w') as f:
        f.write(latex_table)
    
    # Save markdown report
    report = f"""# Grammar vs Token Tokenization Comparison

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Models
- Grammar VAE: {CONFIG['grammar_checkpoint']}
- Token VAE: {CONFIG['token_checkpoint']}

## Summary Results

| Physics Property | Grammar NMI | Token NMI | NMI Winner | Grammar Acc | Token Acc | Acc Winner |
|-----------------|------------|-----------|------------|-------------|-----------|------------|
"""
    for _, row in summary_df.iterrows():
        report += f"| {row['Physics Property']} | {row['Grammar NMI']} | {row['Token NMI']} | {row['NMI Winner']} | {row['Grammar Acc']} | {row['Token Acc']} | {row['Acc Winner']} |\n"
    
    report += """

## Key Findings

"""
    # Add findings
    grammar_wins = sum(1 for c in results['comparison'].values() if c['nmi_winner'] == 'Grammar')
    total = len(results['comparison'])
    
    if grammar_wins > total / 2:
        report += f"- **Grammar tokenization wins** on {grammar_wins}/{total} physics properties for clustering (NMI)\n"
    else:
        report += f"- **Token tokenization wins** on {total - grammar_wins}/{total} physics properties for clustering (NMI)\n"
    
    # Find biggest differences
    biggest_diff = max(results['comparison'].items(), key=lambda x: x[1]['nmi_diff'])
    report += f"- Largest difference: **{biggest_diff[0]}** ({biggest_diff[1]['nmi_winner']} wins by {biggest_diff[1]['nmi_diff']:.3f} NMI)\n"
    
    with open(output_dir / 'results' / 'COMPARISON_REPORT.md', 'w') as f:
        f.write(report)
    
    # Summary
    elapsed = datetime.now() - start_time
    logger.info(f"\nPipeline complete in {elapsed}")
    logger.info(f"Results saved to: {output_dir}")
    

if __name__ == '__main__':
    main()
