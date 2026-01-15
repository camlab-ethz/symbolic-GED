#!/usr/bin/env python3
"""
Comprehensive evaluation script for Grammar VAE vs Token VAE.

This script produces a reproducible evaluation report comparing tokenizations on:
1. Latent-space physics organization (encoder quality)
2. Decoded semantics / validity (generator quality)
3. Interpolation smoothness, perturbation stability, prior sampling validity

Output: JSON + CSV tables + PNG plots with concrete numbers.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Matplotlib for non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Dimensionality reduction
from sklearn.manifold import TSNE
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("WARNING: UMAP not available, will skip UMAP plots")

# Analysis modules
from analysis.clustering import compute_all_clustering, train_classifier
from analysis.physics import PDE_PHYSICS, assign_physics_labels, is_valid_pde
from analysis.pde_classifier import PDEClassifier
from analysis.utils_pde import skeletonize_pde
from analysis.interpolation_analysis import (
    load_vae_model, decode_latent, run_interpolation_suite
)
from analysis.perturbation_analysis import (
    run_perturbation_analysis, run_prior_sampling
)

# VAE modules
from vae.module import VAEModule
from vae.utils.datamodule import GrammarVAEDataModule
from vae.utils import TokenVAEDataModule

# PDE processing
from pde import grammar as pde_grammar
from pde.chr_tokenizer import PDETokenizer


# Color palette for families
FAMILY_COLORS = {
    'heat': '#e41a1c',
    'wave': '#377eb8',
    'poisson': '#4daf4a',
    'advection': '#984ea3',
    'burgers': '#ff7f00',
    'kdv': '#ffff33',
    'reaction_diffusion_cubic': '#a65628',
    'allen_cahn': '#f781bf',
    'cahn_hilliard': '#999999',
    'fisher_kpp': '#66c2a5',
    'kuramoto_sivashinsky': '#fc8d62',
    'airy': '#8da0cb',
    'beam_plate': '#e78ac3',
    'telegraph': '#a6d854',
    'biharmonic': '#ffd92f',
    'sine_gordon': '#e5c494',
    'invalid': '#cccccc',
    'unknown': '#888888',
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def convert_for_json(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64, np.float16)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(convert_for_json(v) for v in obj)
    return obj


def load_latents_from_npz(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load latents and families from NPZ file."""
    data = np.load(npz_path, allow_pickle=True)
    grammar_latents = data['grammar_latents']
    token_latents = data['token_latents']
    families = data['families']
    return grammar_latents, token_latents, families


def encode_dataset_to_latents(
    model,
    tokenization: str,
    csv_path: Path,
    split: str,
    device: str,
    batch_size: int = 256
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Encode PDE dataset to latent vectors. Returns (latents, valid_families, valid_indices)."""
    print(f"  Encoding {tokenization} dataset...")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Filter by split if split column exists
    if 'split' in df.columns:
        df = df[df['split'] == split]
        original_indices = df.index.values
    elif split != 'all':
        # Try to infer from indices file
        # Fix: csv_path is data/raw/file.csv, so parent.parent is 'data', then we need 'data/splits'
        split_dir = csv_path.parent.parent / 'splits'
        if split_dir.exists():
            split_indices = np.load(split_dir / f'{split}_indices.npy')
            df = df.iloc[split_indices]
            original_indices = split_indices
        else:
            original_indices = np.arange(len(df))
    else:
        original_indices = np.arange(len(df))
    
    pde_strings = df['pde'].values.tolist()
    families = df['family'].values.tolist() if 'family' in df.columns else None
    
    # Get model parameters
    vocab_size = model.P
    max_length = model.max_length
    
    all_mu = []
    valid_families = []
    valid_original_indices = []
    
    model.eval()
    tokenizer = PDETokenizer() if tokenization == 'token' else None
    
    with torch.no_grad():
        for batch_idx in tqdm(range(0, len(pde_strings), batch_size), leave=False):
            batch_pdes = pde_strings[batch_idx:batch_idx+batch_size]
            batch_fams = families[batch_idx:batch_idx+batch_size] if families else [None] * len(batch_pdes)
            batch_orig_indices = original_indices[batch_idx:batch_idx+batch_size]
            
            # Convert to one-hot
            batch_onehot = torch.zeros(len(batch_pdes), max_length, vocab_size, dtype=torch.float32)
            valid_mask = []
            
            for b, pde in enumerate(batch_pdes):
                try:
                    if tokenization == 'grammar':
                        # Strip spaces and "=0" to match training data format
                        pde_cleaned = pde.replace(' ', '').replace('=0', '')
                        seq = pde_grammar.parse_to_productions(pde_cleaned)
                        for t, pid in enumerate(seq[:max_length]):
                            if 0 <= pid < vocab_size:
                                batch_onehot[b, t, pid] = 1.0
                    else:
                        ids = tokenizer.encode(pde)
                        for t, tid in enumerate(ids[:max_length]):
                            if 0 <= tid < vocab_size:
                                batch_onehot[b, t, tid] = 1.0
                    
                    valid_mask.append(True)
                    if families:
                        valid_families.append(batch_fams[b])
                    valid_original_indices.append(batch_orig_indices[b])
                except:
                    valid_mask.append(False)
            
            # Encode valid batch
            if any(valid_mask):
                valid_batch_indices = [i for i, v in enumerate(valid_mask) if v]
                if valid_batch_indices:
                    batch_onehot_valid = batch_onehot[valid_batch_indices].to(device)
                    mu, _ = model.encoder(batch_onehot_valid)
                    all_mu.append(mu.cpu().numpy())
    
    if all_mu:
        latents = np.concatenate(all_mu, axis=0)
        valid_original_indices = np.array(valid_original_indices)
    else:
        latents = np.array([]).reshape(0, model.z_dim)
        valid_original_indices = np.array([], dtype=int)
    
    # Filter valid families
    valid_families = [f for f in valid_families if f is not None]
    
    return latents, valid_families if families else None, valid_original_indices


def prepare_ground_truth_labels(
    families: List[str]
) -> Dict[str, np.ndarray]:
    """Prepare ground truth labels from families using PDE_PHYSICS."""
    labels_dict = assign_physics_labels(families)
    
    # Convert to numpy arrays and create proper label names
    result = {
        'family_gt': np.array(families),
    }
    
    # Map temporal strings to integers
    temporal_order_map = {'none': 0, 'first': 1, 'second': 2}
    temporal_orders = [temporal_order_map.get(t, 0) for t in labels_dict['temporal']]
    
    result['type_gt'] = np.array(labels_dict['type'])
    result['order_gt'] = np.array(labels_dict['order'], dtype=int)
    result['temporal_order_gt'] = np.array(temporal_orders, dtype=int)
    result['linearity_gt'] = np.array(['nonlinear' if nl else 'linear' for nl in labels_dict['linearity']])
    
    # Dimension (if available from PDE_PHYSICS, default to 1)
    dims = []
    for fam in families:
        if fam in PDE_PHYSICS:
            # Infer dimension from example or default to 1
            dims.append(1)  # Default, could be enhanced
        else:
            dims.append(1)
    result['dim_gt'] = np.array(dims, dtype=int)
    
    # Mechanisms as string list (simplified)
    mechanisms_list = []
    for fam in families:
        if fam in PDE_PHYSICS:
            mech = PDE_PHYSICS[fam].get('mechanisms', [])
            mechanisms_list.append('_'.join(sorted(mech)) if mech else 'none')
        else:
            mechanisms_list.append('none')
    result['mechanisms_gt'] = np.array(mechanisms_list)
    
    return result


# ============================================================================
# REPRESENTATION METRICS (Encoder Quality)
# ============================================================================

def compute_representation_metrics(
    latents: np.ndarray,
    labels_dict: Dict[str, np.ndarray],
    tokenization: str
) -> Dict:
    """Compute clustering and classification metrics using GT labels."""
    print(f"\n  Computing representation metrics for {tokenization}...")
    
    # Clustering metrics
    clustering_results = compute_all_clustering(latents, labels_dict)
    
    # Classification metrics
    classification_results = {}
    for label_name, labels in labels_dict.items():
        unique_vals = len(set(labels))
        if unique_vals < 2:
            continue
        
        try:
            clf_result = train_classifier(latents, labels, label_name=label_name)
            classification_results[label_name] = clf_result
        except Exception as e:
            print(f"    Warning: Classification failed for {label_name}: {e}")
            continue
    
    return {
        'clustering': clustering_results,
        'classification': classification_results
    }


# ============================================================================
# DECODED SEMANTICS METRICS (Generator Quality)
# ============================================================================

def compute_decoded_semantics_metrics(
    model,
    latents: np.ndarray,
    gt_labels_dict: Dict[str, np.ndarray],
    tokenization: str,
    device: str,
    use_constrained: bool = False
) -> Dict:
    """Compute decoded semantics metrics (validity, agreement with GT, skeleton agreement)."""
    print(f"\n  Computing decoded semantics for {tokenization}...")
    
    classifier = PDEClassifier()
    
    decoded_pdes = []
    decoded_labels = {
        'family': [],
        'pde_type': [],
        'linearity': [],
        'spatial_order': [],
        'temporal_order': [],
        'dimension': [],
        'valid': [],
    }
    
    # Decode all latents
    print(f"    Decoding {len(latents)} latents...")
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(latents)), leave=False):
            z = torch.from_numpy(latents[i:i+1]).float().to(device)
            pde = decode_latent(model, z, tokenization, use_constrained)
            
            decoded_pdes.append(pde if pde else '[INVALID]')
            
            if pde and pde != '[INVALID]':
                try:
                    labels = classifier.classify(pde)
                    decoded_labels['family'].append(labels.family)
                    decoded_labels['pde_type'].append(labels.pde_type)
                    decoded_labels['linearity'].append(labels.linearity)
                    decoded_labels['spatial_order'].append(labels.spatial_order)
                    decoded_labels['temporal_order'].append(labels.temporal_order)
                    decoded_labels['dimension'].append(labels.dimension)
                    decoded_labels['valid'].append(True)
                except:
                    decoded_labels['family'].append('unknown')
                    decoded_labels['pde_type'].append('invalid')
                    decoded_labels['linearity'].append('invalid')
                    decoded_labels['spatial_order'].append(-1)
                    decoded_labels['temporal_order'].append(-1)
                    decoded_labels['dimension'].append(-1)
                    decoded_labels['valid'].append(False)
            else:
                decoded_labels['family'].append('invalid')
                decoded_labels['pde_type'].append('invalid')
                decoded_labels['linearity'].append('invalid')
                decoded_labels['spatial_order'].append(-1)
                decoded_labels['temporal_order'].append(-1)
                decoded_labels['dimension'].append(-1)
                decoded_labels['valid'].append(False)
    
    # Convert to arrays
    decoded_labels_arr = {k: np.array(v) for k, v in decoded_labels.items()}
    
    # Validity rate
    validity_rate = decoded_labels_arr['valid'].mean()
    
    # Agreement with GT (only on valid decodes)
    valid_mask = decoded_labels_arr['valid']
    n_valid = valid_mask.sum()
    
    agreement_metrics = {}
    if n_valid > 0:
        for label_key in ['family', 'pde_type', 'linearity', 'dimension']:
            gt_key = label_key.replace('pde_type', 'type') + '_gt'
            if gt_key in gt_labels_dict:
                gt_vals = gt_labels_dict[gt_key][valid_mask]
                decoded_vals = decoded_labels_arr[label_key][valid_mask]
                agreement = (gt_vals == decoded_vals).mean()
                agreement_metrics[f'{label_key}_acc'] = float(agreement)
        
        # Spatial order (numeric comparison)
        if 'order_gt' in gt_labels_dict:
            gt_orders = gt_labels_dict['order_gt'][valid_mask]
            decoded_orders = decoded_labels_arr['spatial_order'][valid_mask]
            agreement_metrics['spatial_order_acc'] = float((gt_orders == decoded_orders).mean())
        
        # Temporal order
        if 'temporal_order_gt' in gt_labels_dict:
            gt_temporal = gt_labels_dict['temporal_order_gt'][valid_mask]
            decoded_temporal = decoded_labels_arr['temporal_order'][valid_mask]
            agreement_metrics['temporal_order_acc'] = float((gt_temporal == decoded_temporal).mean())
    
    # Skeletonized agreement
    skeleton_agreement = {}
    if n_valid > 0:
        skeleton_decoded_families = []
        for i in range(len(decoded_pdes)):
            if decoded_labels_arr['valid'][i]:
                pde = decoded_pdes[i]
                try:
                    skeleton = skeletonize_pde(pde)
                    labels_skel = classifier.classify(skeleton)
                    skeleton_decoded_families.append(labels_skel.family)
                except:
                    skeleton_decoded_families.append('unknown')
            else:
                skeleton_decoded_families.append('invalid')
        
        skeleton_decoded_families = np.array(skeleton_decoded_families)
        valid_skeleton_mask = (skeleton_decoded_families != 'invalid') & valid_mask
        
        if valid_skeleton_mask.sum() > 0:
            # Compare skeleton-decoded families with GT
            if 'family_gt' in gt_labels_dict:
                gt_fams = gt_labels_dict['family_gt'][valid_skeleton_mask]
                skel_fams = skeleton_decoded_families[valid_skeleton_mask]
                skeleton_agreement['skeleton_family_acc'] = float((gt_fams == skel_fams).mean())
    
    # Decoded family distribution
    family_dist = dict(Counter(decoded_labels_arr['family']))
    
    return {
        'validity_rate': float(validity_rate),
        'n_total': len(latents),
        'n_valid': int(n_valid),
        'agreement_with_gt': agreement_metrics,
        'skeleton_agreement': skeleton_agreement,
        'decoded_family_distribution': family_dist,
    }


# ============================================================================
# CSV GENERATION
# ============================================================================

def create_representation_csv(grammar_rep: Dict, token_rep: Dict, output_dir: Path, split: str):
    """Create CSV summary table for representation metrics."""
    rows = []
    
    # Clustering metrics
    for label in grammar_rep['clustering']:
        g_clust = grammar_rep['clustering'][label]
        t_clust = token_rep['clustering'][label]
        
        rows.append({
            'Category': 'Clustering',
            'Label': label,
            'Metric': 'ARI',
            'Grammar VAE': g_clust.get('ari', 0),
            'Token VAE': t_clust.get('ari', 0),
        })
        rows.append({
            'Category': 'Clustering',
            'Label': label,
            'Metric': 'NMI',
            'Grammar VAE': g_clust.get('nmi', 0),
            'Token VAE': t_clust.get('nmi', 0),
        })
        rows.append({
            'Category': 'Clustering',
            'Label': label,
            'Metric': 'Purity',
            'Grammar VAE': g_clust.get('purity', 0),
            'Token VAE': t_clust.get('purity', 0),
        })
        rows.append({
            'Category': 'Clustering',
            'Label': label,
            'Metric': 'Silhouette',
            'Grammar VAE': g_clust.get('silhouette', 0),
            'Token VAE': t_clust.get('silhouette', 0),
        })
    
    # Classification metrics
    for label in grammar_rep['classification']:
        g_cls = grammar_rep['classification'][label]
        t_cls = token_rep['classification'][label]
        
        rows.append({
            'Category': 'Classification',
            'Label': label,
            'Metric': 'Accuracy',
            'Grammar VAE': g_cls.get('accuracy', 0),
            'Token VAE': t_cls.get('accuracy', 0),
        })
    
    df = pd.DataFrame(rows)
    csv_path = output_dir / f'representation_metrics_{split}.csv'
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path.name}")


def create_decoded_csv(grammar_dec: Dict, token_dec: Dict, output_dir: Path, split: str):
    """Create CSV summary table for decoded semantics metrics."""
    rows = []
    
    # Validity
    rows.append({
        'Category': 'Validity',
        'Metric': 'Validity Rate',
        'Grammar VAE': grammar_dec.get('validity_rate', 0),
        'Token VAE': token_dec.get('validity_rate', 0),
    })
    rows.append({
        'Category': 'Validity',
        'Metric': 'N Valid',
        'Grammar VAE': grammar_dec.get('n_valid', 0),
        'Token VAE': token_dec.get('n_valid', 0),
    })
    rows.append({
        'Category': 'Validity',
        'Metric': 'N Total',
        'Grammar VAE': grammar_dec.get('n_total', 0),
        'Token VAE': token_dec.get('n_total', 0),
    })
    
    # Agreement with GT
    g_agree = grammar_dec.get('agreement_with_gt', {})
    t_agree = token_dec.get('agreement_with_gt', {})
    
    for key in set(list(g_agree.keys()) + list(t_agree.keys())):
        rows.append({
            'Category': 'Agreement',
            'Metric': key,
            'Grammar VAE': g_agree.get(key, 0),
            'Token VAE': t_agree.get(key, 0),
        })
    
    # Skeleton agreement
    g_skel = grammar_dec.get('skeleton_agreement', {})
    t_skel = token_dec.get('skeleton_agreement', {})
    
    for key in set(list(g_skel.keys()) + list(t_skel.keys())):
        rows.append({
            'Category': 'Skeleton Agreement',
            'Metric': key,
            'Grammar VAE': g_skel.get(key, 0),
            'Token VAE': t_skel.get(key, 0),
        })
    
    df = pd.DataFrame(rows)
    csv_path = output_dir / f'decoded_metrics_{split}.csv'
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path.name}")


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualizations(
    grammar_latents: np.ndarray,
    token_latents: np.ndarray,
    gt_families: np.ndarray,
    decoded_grammar_families: np.ndarray,
    decoded_token_families: np.ndarray,
    output_dir: Path,
    seed: int = 42,
    split: str = 'all'
):
    """Create UMAP and t-SNE plots for GT and decoded labels."""
    print("\n  Creating visualizations...")
    
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Create plots for each tokenization
    for tokenization, latents, decoded_families in [
        ('grammar', grammar_latents, decoded_grammar_families),
        ('token', token_latents, decoded_token_families),
    ]:
        # t-SNE with GT labels
        print(f"    t-SNE for {tokenization} (GT labels)...")
        tsne = TSNE(n_components=2, random_state=seed, perplexity=30, n_jobs=-1)
        embedding_tsne = tsne.fit_transform(latents)
        
        plot_embedding(
            embedding_tsne, gt_families,
            f"{tokenization.capitalize()} VAE - Ground Truth Labels ({split} split)",
            plots_dir / f"{tokenization}_tsne_gt_{split}.png"
        )
        
        # t-SNE with decoded labels
        print(f"    t-SNE for {tokenization} (decoded labels)...")
        plot_embedding(
            embedding_tsne, decoded_families,
            f"{tokenization.capitalize()} VAE - Decoded Labels ({split} split)",
            plots_dir / f"{tokenization}_tsne_decoded_{split}.png"
        )
        
        # UMAP if available
        if HAS_UMAP:
            print(f"    UMAP for {tokenization} (GT labels)...")
            reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1)
            embedding_umap = reducer.fit_transform(latents)
            
            plot_embedding(
                embedding_umap, gt_families,
                f"{tokenization.capitalize()} VAE - Ground Truth Labels (UMAP) ({split} split)",
                plots_dir / f"{tokenization}_umap_gt_{split}.png"
            )
            
            print(f"    UMAP for {tokenization} (decoded labels)...")
            plot_embedding(
                embedding_umap, decoded_families,
                f"{tokenization.capitalize()} VAE - Decoded Labels (UMAP) ({split} split)",
                plots_dir / f"{tokenization}_umap_decoded_{split}.png"
            )


def plot_embedding(embedding: np.ndarray, labels: np.ndarray, title: str, output_path: Path):
    """Plot 2D embedding colored by labels."""
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    
    unique_labels = sorted(set(labels))
    
    for label in unique_labels:
        mask = np.array(labels) == label
        if mask.sum() == 0:
            continue
        
        color = FAMILY_COLORS.get(label, '#cccccc')
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=color, label=label, s=10, alpha=0.5, edgecolors='black', linewidths=0.2
        )
    
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=8, markerscale=1.5, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      Saved: {output_path.name}")


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(
    all_results: Dict,
    output_dir: Path,
    split: str = 'all'
):
    """Generate comprehensive markdown report."""
    report_path = output_dir / f'REPORT_{split}.md'
    
    lines = [
        "# Tokenization Evaluation Report",
        "",
        "Comprehensive comparison of Grammar VAE vs Token VAE",
        "",
        "## 1. Representation Metrics (Encoder Quality)",
        "",
        "### Clustering Metrics",
        "",
        "| Label | Metric | Grammar VAE | Token VAE | Winner |",
        "|-------|--------|-------------|-----------|--------|",
    ]
    
    # Clustering metrics table
    grammar_clust = all_results['grammar']['representation']['clustering']
    token_clust = all_results['token']['representation']['clustering']
    
    for label_name in sorted(set(list(grammar_clust.keys()) + list(token_clust.keys()))):
        if label_name in grammar_clust and label_name in token_clust:
            g_metrics = grammar_clust[label_name]
            t_metrics = token_clust[label_name]
            
            for metric in ['ari', 'nmi', 'purity', 'silhouette', 'silhouette_wrt_labels']:
                if metric in g_metrics and metric in t_metrics:
                    g_val = g_metrics[metric]
                    t_val = t_metrics[metric]
                    if not np.isnan(g_val) and not np.isnan(t_val):
                        winner = 'Grammar' if g_val > t_val else 'Token'
                        lines.append(f"| {label_name} | {metric} | {g_val:.4f} | {t_val:.4f} | {winner} |")
    
    lines.extend([
        "",
        "### Classification Accuracy (5-fold CV)",
        "",
        "| Label | Grammar VAE | Token VAE | Winner |",
        "|-------|-------------|-----------|--------|",
    ])
    
    # Classification metrics
    grammar_clf = all_results['grammar']['representation']['classification']
    token_clf = all_results['token']['representation']['classification']
    
    for label_name in sorted(set(list(grammar_clf.keys()) + list(token_clf.keys()))):
        if label_name in grammar_clf and label_name in token_clf:
            g_acc = grammar_clf[label_name].get('accuracy_mean', np.nan)
            t_acc = token_clf[label_name].get('accuracy_mean', np.nan)
            if not np.isnan(g_acc) and not np.isnan(t_acc):
                winner = 'Grammar' if g_acc > t_acc else 'Token'
                lines.append(f"| {label_name} | {g_acc:.4f} | {t_acc:.4f} | {winner} |")
    
    # Decoded semantics
    lines.extend([
        "",
        "## 2. Decoded Semantics Metrics (Generator Quality)",
        "",
        "| Metric | Grammar VAE | Token VAE | Winner |",
        "|--------|-------------|-----------|--------|",
    ])
    
    grammar_dec = all_results['grammar']['decoded_semantics']
    token_dec = all_results['token']['decoded_semantics']
    
    lines.append(f"| Validity Rate | {grammar_dec['validity_rate']:.4f} | {token_dec['validity_rate']:.4f} | {'Grammar' if grammar_dec['validity_rate'] > token_dec['validity_rate'] else 'Token'} |")
    
    # Agreement metrics
    for key in sorted(set(list(grammar_dec['agreement_with_gt'].keys()) + list(token_dec['agreement_with_gt'].keys()))):
        if key in grammar_dec['agreement_with_gt'] and key in token_dec['agreement_with_gt']:
            g_val = grammar_dec['agreement_with_gt'][key]
            t_val = token_dec['agreement_with_gt'][key]
            winner = 'Grammar' if g_val > t_val else 'Token'
            lines.append(f"| {key} | {g_val:.4f} | {t_val:.4f} | {winner} |")
    
    # Interpolation summary
    if 'interpolation' in all_results:
        lines.extend([
            "",
            "## 3. Interpolation Summary",
            "",
            "| Metric | Grammar VAE | Token VAE | Winner |",
            "|--------|-------------|-----------|--------|",
        ])
        
        grammar_interp = all_results['grammar']['interpolation']['summary']
        token_interp = all_results['token']['interpolation']['summary']
        
        if 'validity_rate' in grammar_interp and 'validity_rate' in token_interp:
            g_val = grammar_interp['validity_rate']
            t_val = token_interp['validity_rate']
            winner = 'Grammar' if g_val > t_val else 'Token'
            lines.append(f"| Validity Rate | {g_val:.4f} | {t_val:.4f} | {winner} |")
        
        if 'avg_type_changes' in grammar_interp and 'avg_type_changes' in token_interp:
            g_val = grammar_interp['avg_type_changes']
            t_val = token_interp['avg_type_changes']
            winner = 'Token' if g_val > t_val else 'Grammar'  # Lower is better
            lines.append(f"| Avg Type Changes | {g_val:.2f} | {t_val:.2f} | {winner} |")
    
    # Perturbation summary
    if 'perturbation' in all_results:
        lines.extend([
            "",
            "## 4. Perturbation Summary",
            "",
            "| Metric | Grammar VAE | Token VAE | Winner |",
            "|--------|-------------|-----------|--------|",
        ])
        
        grammar_pert = all_results['grammar']['perturbation']
        token_pert = all_results['token']['perturbation']
        
        if 'preservation' in grammar_pert and 'preservation' in token_pert:
            for key in ['dim_preserved', 'type_preserved']:
                if key in grammar_pert['preservation'] and key in token_pert['preservation']:
                    g_val = grammar_pert['preservation'][key]
                    t_val = token_pert['preservation'][key]
                    winner = 'Grammar' if g_val > t_val else 'Token'
                    lines.append(f"| {key} | {g_val:.4f} | {t_val:.4f} | {winner} |")
    
    # Prior sampling summary
    if 'sampling' in all_results:
        lines.extend([
            "",
            "## 5. Prior Sampling Summary",
            "",
            "| Metric | Grammar VAE | Token VAE | Winner |",
            "|--------|-------------|-----------|--------|",
        ])
        
        grammar_samp = all_results['grammar']['sampling']
        token_samp = all_results['token']['sampling']
        
        if 'validity_rate' in grammar_samp and 'validity_rate' in token_samp:
            g_val = grammar_samp['validity_rate']
            t_val = token_samp['validity_rate']
            winner = 'Grammar' if g_val > t_val else 'Token'
            lines.append(f"| Validity Rate | {g_val:.4f} | {t_val:.4f} | {winner} |")
        
        if 'novelty_rate' in grammar_samp and 'novelty_rate' in token_samp:
            g_val = grammar_samp['novelty_rate']
            t_val = token_samp['novelty_rate']
            winner = 'Grammar' if g_val > t_val else 'Token'
            lines.append(f"| Novelty Rate | {g_val:.4f} | {t_val:.4f} | {winner} |")
    
    # Command to run
    lines.extend([
        "",
        "## Command to Run",
        "",
        "```bash",
        "python scripts/run_tokenization_eval.py \\",
        "    --grammar-ckpt <PATH_TO_GRAMMAR_CHECKPOINT> \\",
        "    --token-ckpt <PATH_TO_TOKEN_CHECKPOINT> \\",
        "    --csv-metadata <PATH_TO_CSV> \\",
        "    --split test \\",
        "    --outdir experiments/reports/tokenization_eval/",
        "```",
        "",
    ])
    
    # Write report
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"\n  Report saved to: {report_path}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive evaluation of Grammar VAE vs Token VAE'
    )
    parser.add_argument('--grammar-ckpt', type=str, required=True,
                       help='Path to Grammar VAE checkpoint')
    parser.add_argument('--token-ckpt', type=str, required=True,
                       help='Path to Token VAE checkpoint')
    parser.add_argument('--latent-npz', type=str, default=None,
                       help='Path to NPZ file with pre-computed latents (optional)')
    parser.add_argument('--csv-metadata', type=str, required=True,
                       help='Path to CSV with PDE strings and GT labels')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to use')
    parser.add_argument('--n_pairs', type=int, default=50,
                       help='Number of interpolation pairs')
    parser.add_argument('--n_steps', type=int, default=11,
                       help='Number of interpolation steps')
    parser.add_argument('--sigma', type=float, default=0.1,
                       help='Perturbation sigma')
    parser.add_argument('--n_samples', type=int, default=500,
                       help='Number of samples for perturbation/sampling')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--outdir', type=str, default='experiments/reports/tokenization_eval',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Setup output directory
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("COMPREHENSIVE TOKENIZATION EVALUATION")
    print("="*80)
    print(f"Grammar checkpoint: {args.grammar_ckpt}")
    print(f"Token checkpoint: {args.token_ckpt}")
    print(f"Split: {args.split}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # Load models
    print("\n[1/7] Loading models...")
    grammar_model, grammar_hparams = load_vae_model(args.grammar_ckpt, args.device)
    token_model, token_hparams = load_vae_model(args.token_ckpt, args.device)
    
    # Get latents
    print("\n[2/7] Getting latent vectors...")
    csv_path = Path(args.csv_metadata)
    
    if args.latent_npz and Path(args.latent_npz).exists():
        print(f"  Loading latents from {args.latent_npz}")
        data = np.load(args.latent_npz, allow_pickle=True)
        grammar_latents = data['grammar_latents']
        token_latents = data['token_latents']
        grammar_families = data['grammar_families'].tolist() if 'grammar_families' in data else None
        token_families = data['token_families'].tolist() if 'token_families' in data else None
    else:
        print("  Encoding dataset...")
        grammar_latents, grammar_families, _ = encode_dataset_to_latents(
            grammar_model, 'grammar', csv_path, args.split, args.device
        )
        token_latents, token_families, _ = encode_dataset_to_latents(
            token_model, 'token', csv_path, args.split, args.device
        )
        
        # Save latents for future use
        if args.latent_npz:
            np.savez(
                args.latent_npz,
                grammar_latents=grammar_latents,
                token_latents=token_latents,
                grammar_families=np.array(grammar_families) if grammar_families else np.array([]),
                token_families=np.array(token_families) if token_families else np.array([])
            )
            print(f"  Saved latents to {args.latent_npz}")
    
    print(f"  Grammar latents: {grammar_latents.shape}")
    print(f"  Token latents: {token_latents.shape}")
    
    # Prepare GT labels - each tokenization has its own labels based on valid samples
    grammar_labels_dict = prepare_ground_truth_labels(grammar_families) if grammar_families else {}
    token_labels_dict = prepare_ground_truth_labels(token_families) if token_families else {}
    
    # Representation metrics
    print("\n[3/7] Computing representation metrics...")
    grammar_representation = compute_representation_metrics(
        grammar_latents, grammar_labels_dict, 'grammar'
    )
    token_representation = compute_representation_metrics(
        token_latents, token_labels_dict, 'token'
    )
    
    # Save representation metrics
    rep_metrics = {
        'grammar': grammar_representation,
        'token': token_representation,
    }
    rep_json_path = output_dir / f'metrics_representation_{args.split}.json'
    with open(rep_json_path, 'w') as f:
        json.dump(convert_for_json(rep_metrics), f, indent=2)
    print(f"  Saved: {rep_json_path.name}")
    
    # Create CSV for representation metrics
    create_representation_csv(grammar_representation, token_representation, output_dir, args.split)
    
    # Decoded semantics metrics
    print("\n[4/7] Computing decoded semantics metrics...")
    grammar_decoded = compute_decoded_semantics_metrics(
        grammar_model, grammar_latents, grammar_labels_dict, 'grammar', args.device, use_constrained=True
    )
    token_decoded = compute_decoded_semantics_metrics(
        token_model, token_latents, token_labels_dict, 'token', args.device, use_constrained=False
    )
    
    # Get decoded families for visualization
    # (Would need to decode again or store from compute_decoded_semantics_metrics)
    # For now, we'll decode a subset for visualization
    print("    Decoding for visualization...")
    grammar_decoded_families = decode_families_for_viz(
        grammar_model, grammar_latents, 'grammar', args.device, use_constrained=True
    )
    token_decoded_families = decode_families_for_viz(
        token_model, token_latents, 'token', args.device, use_constrained=False
    )
    
    # Save decoded semantics metrics
    decoded_metrics = {
        'grammar': grammar_decoded,
        'token': token_decoded,
    }
    decoded_json_path = output_dir / f'metrics_decoded_{args.split}.json'
    with open(decoded_json_path, 'w') as f:
        json.dump(convert_for_json(decoded_metrics), f, indent=2)
    print(f"  Saved: {decoded_json_path.name}")
    
    create_decoded_csv(grammar_decoded, token_decoded, output_dir, args.split)
    
    # Visualizations
    print("\n[5/7] Creating visualizations...")
    families_arr = np.array(grammar_families) if grammar_families else np.array([])
    create_visualizations(
        grammar_latents, token_latents,
        families_arr,
        grammar_decoded_families,
        token_decoded_families,
        output_dir,
        seed=args.seed,
        split=args.split
    )
    
    # Interpolation analysis
    print("\n[6/7] Running interpolation analysis...")
    grammar_interp = run_interpolation_suite(
        grammar_model, grammar_latents, families_arr,
        'grammar', n_pairs=args.n_pairs, n_steps=args.n_steps,
        seed=args.seed, use_constrained=True, device=args.device
    )
    token_interp = run_interpolation_suite(
        token_model, token_latents, families_arr,
        'token', n_pairs=args.n_pairs, n_steps=args.n_steps,
        seed=args.seed, use_constrained=False, device=args.device
    )
    
    interp_json_path = output_dir / f'interpolation_{args.split}.json'
    with open(interp_json_path, 'w') as f:
        json.dump(convert_for_json({
            'grammar': grammar_interp,
            'token': token_interp
        }), f, indent=2)
    print(f"  Saved: {interp_json_path.name}")
    
    # Perturbation analysis
    print("\n  Running perturbation analysis...")
    grammar_pert = run_perturbation_analysis(
        grammar_model, grammar_latents, families_arr,
        'grammar', sigma=args.sigma, n_samples=args.n_samples,
        seed=args.seed, use_constrained=True, device=args.device
    )
    token_pert = run_perturbation_analysis(
        token_model, token_latents, families_arr,
        'token', sigma=args.sigma, n_samples=args.n_samples,
        seed=args.seed, use_constrained=False, device=args.device
    )
    
    pert_json_path = output_dir / f'perturbation_{args.split}.json'
    with open(pert_json_path, 'w') as f:
        json.dump(convert_for_json({
            'grammar': grammar_pert,
            'token': token_pert
        }), f, indent=2)
    print(f"  Saved: {pert_json_path.name}")
    
    # Prior sampling
    print("\n  Running prior sampling...")
    # Load training PDEs for novelty check (optional)
    training_pdes = None
    try:
        df_train = pd.read_csv(csv_path)
        if 'split' in df_train.columns:
            training_pdes = df_train[df_train['split'] == 'train']['pde'].values.tolist()
    except:
        pass
    
    grammar_sampling = run_prior_sampling(
        grammar_model, 'grammar', z_dim=grammar_hparams.get('z_dim', 26),
        n_samples=args.n_samples, seed=args.seed,
        use_constrained=True, device=args.device, training_pdes=training_pdes
    )
    token_sampling = run_prior_sampling(
        token_model, 'token', z_dim=token_hparams.get('z_dim', 26),
        n_samples=args.n_samples, seed=args.seed,
        use_constrained=False, device=args.device, training_pdes=training_pdes
    )
    
    sampling_json_path = output_dir / f'sampling_{args.split}.json'
    with open(sampling_json_path, 'w') as f:
        json.dump(convert_for_json({
            'grammar': grammar_sampling,
            'token': token_sampling
        }), f, indent=2)
    print(f"  Saved: {sampling_json_path.name}")
    
    # Compile all results
    all_results = {
        'grammar': {
            'representation': grammar_representation,
            'decoded_semantics': grammar_decoded,
            'interpolation': grammar_interp,
            'perturbation': grammar_pert,
            'sampling': grammar_sampling,
        },
        'token': {
            'representation': token_representation,
            'decoded_semantics': token_decoded,
            'interpolation': token_interp,
            'perturbation': token_pert,
            'sampling': token_sampling,
        },
    }
    
    # Generate report
    print("\n[7/7] Generating report...")
    generate_report(all_results, output_dir, args.split)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - metrics_representation_{args.split}.json + .csv")
    print(f"  - metrics_decoded_{args.split}.json + .csv")
    print(f"  - interpolation_{args.split}.json")
    print(f"  - perturbation_{args.split}.json")
    print(f"  - sampling_{args.split}.json")
    print(f"  - plots/*.png (with {args.split} suffix)")
    print(f"  - REPORT_{args.split}.md")
    print("\n" + "="*80)


def decode_families_for_viz(model, latents, tokenization, device, use_constrained=False):
    """Decode latents to get families for visualization."""
    classifier = PDEClassifier()
    families = []
    
    model.eval()
    with torch.no_grad():
        for i in range(len(latents)):
            z = torch.from_numpy(latents[i:i+1]).float().to(device)
            pde = decode_latent(model, z, tokenization, use_constrained)
            
            if pde and pde != '[INVALID]':
                try:
                    labels = classifier.classify(pde)
                    families.append(labels.family)
                except:
                    families.append('invalid')
            else:
                families.append('invalid')
    
    return np.array(families)


def create_representation_csv(grammar_results, token_results, output_dir):
    """Create CSV table for representation metrics."""
    rows = []
    
    # Clustering metrics
    for label_name in sorted(set(list(grammar_results['clustering'].keys()) + list(token_results['clustering'].keys()))):
        if label_name in grammar_results['clustering'] and label_name in token_results['clustering']:
            g_metrics = grammar_results['clustering'][label_name]
            t_metrics = token_results['clustering'][label_name]
            
            for metric in ['ari', 'nmi', 'purity', 'silhouette', 'silhouette_wrt_labels']:
                if metric in g_metrics and metric in t_metrics:
                    g_val = g_metrics[metric]
                    t_val = t_metrics[metric]
                    if not np.isnan(g_val) and not np.isnan(t_val):
                        winner = 'Grammar' if g_val > t_val else 'Token'
                        rows.append({
                            'label_name': label_name,
                            'metric': metric,
                            'grammar_value': g_val,
                            'token_value': t_val,
                            'winner': winner
                        })
    
    # Classification metrics
    for label_name in sorted(set(list(grammar_results['classification'].keys()) + list(token_results['classification'].keys()))):
        if label_name in grammar_results['classification'] and label_name in token_results['classification']:
            g_acc = grammar_results['classification'][label_name].get('accuracy_mean', np.nan)
            t_acc = token_results['classification'][label_name].get('accuracy_mean', np.nan)
            if not np.isnan(g_acc) and not np.isnan(t_acc):
                winner = 'Grammar' if g_acc > t_acc else 'Token'
                rows.append({
                    'label_name': label_name,
                    'metric': 'linear_probe_acc_mean',
                    'grammar_value': g_acc,
                    'token_value': t_acc,
                    'winner': winner
                })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'metrics_representation.csv', index=False)


def create_decoded_csv(grammar_results, token_results, output_dir):
    """Create CSV table for decoded semantics metrics."""
    rows = []
    
    # Validity
    rows.append({
        'metric': 'validity_rate',
        'grammar_value': grammar_results['validity_rate'],
        'token_value': token_results['validity_rate'],
        'winner': 'Grammar' if grammar_results['validity_rate'] > token_results['validity_rate'] else 'Token'
    })
    
    # Agreement metrics
    for key in sorted(set(list(grammar_results['agreement_with_gt'].keys()) + list(token_results['agreement_with_gt'].keys()))):
        if key in grammar_results['agreement_with_gt'] and key in token_results['agreement_with_gt']:
            g_val = grammar_results['agreement_with_gt'][key]
            t_val = token_results['agreement_with_gt'][key]
            winner = 'Grammar' if g_val > t_val else 'Token'
            rows.append({
                'metric': key,
                'grammar_value': g_val,
                'token_value': t_val,
                'winner': winner
            })
    
    # Skeleton agreement
    for key in sorted(set(list(grammar_results['skeleton_agreement'].keys()) + list(token_results['skeleton_agreement'].keys()))):
        if key in grammar_results['skeleton_agreement'] and key in token_results['skeleton_agreement']:
            g_val = grammar_results['skeleton_agreement'][key]
            t_val = token_results['skeleton_agreement'][key]
            winner = 'Grammar' if g_val > t_val else 'Token'
            rows.append({
                'metric': key,
                'grammar_value': g_val,
                'token_value': t_val,
                'winner': winner
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'metrics_decoded.csv', index=False)


if __name__ == '__main__':
    main()
