#!/usr/bin/env python3
"""Run true OOD (Out-of-Distribution) evaluation.

This script evaluates VAE models trained on IID families against
held-out OOD families (families the VAE has NEVER seen).

This is a TRUE OOD test because:
1. VAE encoder has never seen OOD PDEs during training
2. VAE decoder has never reconstructed OOD PDEs
3. We're testing if learned representations generalize to new physics

Usage:
    python scripts/run_ood_evaluation.py
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from scipy import stats

# Setup paths
SCRIPT_DIR = Path(__file__).parent
LIBGEN_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(LIBGEN_DIR))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# PDE type mapping
PDE_TYPES = {
    'heat': 'parabolic', 'wave': 'hyperbolic', 'poisson': 'elliptic',
    'advection': 'hyperbolic', 'burgers': 'hyperbolic', 'kdv': 'dispersive',
    'fisher_kpp': 'parabolic', 'allen_cahn': 'parabolic', 'cahn_hilliard': 'parabolic',
    'reaction_diffusion_cubic': 'parabolic',
    'sine_gordon': 'hyperbolic', 'telegraph': 'hyperbolic', 'biharmonic': 'elliptic',
    'kuramoto_sivashinsky': 'parabolic',
    'airy': 'dispersive', 'beam_plate': 'hyperbolic',
}


def load_model_and_encode(
    checkpoint_path: str,
    pde_strings: List[str],
    tokenization: str,
    device: str = 'cpu'
) -> np.ndarray:
    """Load model and encode PDEs to latent vectors."""
    from vae.module import VAEModule
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    hparams = checkpoint['hyper_parameters']
    
    model = VAEModule(
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
    
    P = hparams['P']
    max_length = hparams['max_length']
    
    latents = []
    
    for pde in pde_strings:
        try:
            if tokenization == 'grammar':
                from pde import grammar as pde_grammar
                seq = pde_grammar.parse_to_productions(pde.replace(' ', ''))
                x = torch.zeros(max_length, P)
                for t, pid in enumerate(seq[:max_length]):
                    if 0 <= pid < P:
                        x[t, pid] = 1.0
            else:
                from pde.chr_tokenizer import PDETokenizer
                tokenizer = PDETokenizer()
                ids = tokenizer.encode(pde)
                x = torch.zeros(max_length, P)
                for t, tid in enumerate(ids[:max_length]):
                    if 0 <= tid < P:
                        x[t, tid] = 1.0
            
            x = x.unsqueeze(0).to(device)
            with torch.no_grad():
                mu, _ = model.encoder(x)
            latents.append(mu.cpu().numpy()[0])
        except Exception as e:
            logger.warning(f"Failed to encode PDE: {pde[:50]}... Error: {e}")
            latents.append(np.zeros(model.z_dim))
    
    return np.array(latents)


def run_ood_classification(
    iid_latents: np.ndarray,
    iid_labels: np.ndarray,
    ood_latents: np.ndarray,
    ood_labels: np.ndarray,
    n_cv_folds: int = 5,
    random_state: int = 42
) -> Dict:
    """Run classification on IID and OOD data.
    
    Returns:
        Dictionary with IID and OOD accuracies
    """
    # Encode labels
    le = LabelEncoder()
    le.fit(iid_labels)
    
    y_iid = le.transform(iid_labels)
    
    # Check if OOD labels are in training set
    ood_in_iid = all(label in le.classes_ for label in np.unique(ood_labels))
    
    # IID cross-validation
    clf = LogisticRegression(max_iter=1000, random_state=random_state, n_jobs=-1)
    iid_scores = cross_val_score(clf, iid_latents, y_iid, cv=n_cv_folds, scoring='accuracy')
    
    result = {
        'iid_accuracy_mean': float(np.mean(iid_scores)),
        'iid_accuracy_std': float(np.std(iid_scores)),
        'iid_scores': iid_scores.tolist(),
        'n_iid': len(iid_latents),
        'n_ood': len(ood_latents),
        'ood_labels_in_train': ood_in_iid,
    }
    
    if ood_in_iid:
        # Train on all IID, test on OOD
        clf.fit(iid_latents, y_iid)
        y_ood = le.transform(ood_labels)
        ood_acc = clf.score(ood_latents, y_ood)
        result['ood_accuracy'] = float(ood_acc)
        result['generalization_gap'] = float(np.mean(iid_scores) - ood_acc)
        
        # Per-class accuracy on OOD
        y_pred = clf.predict(ood_latents)
        per_class = {}
        for label in np.unique(ood_labels):
            mask = ood_labels == label
            label_idx = le.transform([label])[0]
            correct = (y_pred[mask] == label_idx).sum()
            per_class[label] = float(correct / mask.sum())
        result['ood_per_class'] = per_class
    else:
        result['ood_accuracy'] = None
        result['generalization_gap'] = None
        result['ood_per_class'] = None
    
    return result


def main():
    """Run OOD evaluation."""
    
    print("=" * 70)
    print("TRUE OUT-OF-DISTRIBUTION EVALUATION")
    print("=" * 70)
    
    # Configuration
    split_dir = LIBGEN_DIR / "splits/ood_kdv_schrodinger"
    grammar_ckpt = LIBGEN_DIR / "checkpoints/grammar_vae_ood"
    token_ckpt = LIBGEN_DIR / "checkpoints/token_vae_ood"
    dataset_path = LIBGEN_DIR / "pde_dataset_48444_clean.csv"
    output_dir = LIBGEN_DIR / "experiments/ood_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if OOD-trained models exist
    grammar_ckpt_file = None
    token_ckpt_file = None
    
    if grammar_ckpt.exists():
        ckpts = list(grammar_ckpt.glob("*.ckpt"))
        if ckpts:
            grammar_ckpt_file = str(sorted(ckpts)[-1])
    
    if token_ckpt.exists():
        ckpts = list(token_ckpt.glob("*.ckpt"))
        if ckpts:
            token_ckpt_file = str(sorted(ckpts)[-1])
    
    # If OOD models don't exist, use the pre-trained latents with a note
    use_pretrained = False
    if grammar_ckpt_file is None or token_ckpt_file is None:
        print("\n⚠️  OOD-trained models not found!")
        print("   Using pre-trained latents (NOT true OOD - VAE saw all data)")
        print("   To run TRUE OOD experiments, first run:")
        print("   ./scripts/train_ood_vaes.sh")
        use_pretrained = True
    
    # Load dataset
    print(f"\n1. Loading dataset...")
    df = pd.read_csv(dataset_path)
    df['pde_type'] = df['family'].map(PDE_TYPES)
    df['linearity'] = df['nonlinear'].map({True: 'nonlinear', False: 'linear'})
    
    # Load split metadata
    if split_dir.exists():
        with open(split_dir / 'split_metadata.json') as f:
            split_meta = json.load(f)
        ood_families = split_meta['exclude_families']
        
        # Load indices
        train_idx = np.load(split_dir / 'train_indices.npy')
        val_idx = np.load(split_dir / 'val_indices.npy')
        test_iid_idx = np.load(split_dir / 'test_iid_indices.npy')
        test_ood_idx = np.load(split_dir / 'test_ood_indices.npy')
        
        print(f"   IID families: {split_meta['iid_families']}")
        print(f"   OOD families: {ood_families}")
        print(f"   Train: {len(train_idx)}, Val: {len(val_idx)}")
        print(f"   Test IID: {len(test_iid_idx)}, Test OOD: {len(test_ood_idx)}")
    else:
        print("   Creating OOD splits...")
        from scripts.create_ood_splits import create_ood_splits
        split_meta = create_ood_splits(
            str(dataset_path),
            exclude_families=['kdv', 'schrodinger'],
            output_dir=str(split_dir)
        )
        ood_families = ['kdv', 'schrodinger']
        
        train_idx = np.load(split_dir / 'train_indices.npy')
        val_idx = np.load(split_dir / 'val_indices.npy')
        test_iid_idx = np.load(split_dir / 'test_iid_indices.npy')
        test_ood_idx = np.load(split_dir / 'test_ood_indices.npy')
    
    results = {'timestamp': datetime.now().isoformat(), 'ood_families': ood_families}
    
    if use_pretrained:
        # Load pre-computed latents
        print(f"\n2. Loading pre-computed latents...")
        grammar_latents = np.load(LIBGEN_DIR / "experiments/comparison/latents/grammar_latents.npz")['mu']
        token_latents = np.load(LIBGEN_DIR / "experiments/comparison/latents/token_latents.npz")['mu']
        
        results['note'] = "Using pre-trained VAE latents (NOT true OOD - VAE was trained on ALL families)"
    else:
        # Encode with OOD-trained models
        print(f"\n2. Encoding PDEs with OOD-trained models...")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Get all PDE strings
        pde_strings = df['pde'].values
        
        print(f"   Encoding with Grammar VAE...")
        grammar_latents = load_model_and_encode(
            grammar_ckpt_file, pde_strings, 'grammar', device
        )
        
        print(f"   Encoding with Token VAE...")
        token_latents = load_model_and_encode(
            token_ckpt_file, pde_strings, 'token', device
        )
        
        results['note'] = "TRUE OOD evaluation - VAE was trained WITHOUT KdV and Schrödinger"
    
    # Run classification experiments
    print(f"\n3. Running classification experiments...")
    
    # Combine train+val for training the classifier
    iid_train_idx = np.concatenate([train_idx, val_idx])
    
    label_columns = ['pde_type', 'linearity', 'dim', 'spatial_order']
    
    for label_col in label_columns:
        print(f"\n   {label_col}:")
        
        labels = df[label_col].values
        
        for name, latents in [('grammar', grammar_latents), ('token', token_latents)]:
            result = run_ood_classification(
                iid_latents=latents[iid_train_idx],
                iid_labels=labels[iid_train_idx],
                ood_latents=latents[test_ood_idx],
                ood_labels=labels[test_ood_idx],
            )
            
            if label_col not in results:
                results[label_col] = {}
            results[label_col][name] = result
            
            iid_acc = result['iid_accuracy_mean']
            ood_acc = result.get('ood_accuracy', 'N/A')
            gap = result.get('generalization_gap', 'N/A')
            
            if isinstance(ood_acc, float):
                print(f"      {name}: IID={iid_acc:.1%}, OOD={ood_acc:.1%}, Gap={gap:+.1%}")
            else:
                print(f"      {name}: IID={iid_acc:.1%}, OOD={ood_acc}")
    
    # Statistical comparison
    print(f"\n4. Statistical comparison (Grammar vs Token)...")
    
    comparisons = {}
    for label_col in label_columns:
        g = results[label_col]['grammar']
        t = results[label_col]['token']
        
        # Paired t-test on IID scores
        if g.get('iid_scores') and t.get('iid_scores'):
            t_stat, p_value = stats.ttest_rel(g['iid_scores'], t['iid_scores'])
            
            comparisons[label_col] = {
                'grammar_iid': g['iid_accuracy_mean'],
                'token_iid': t['iid_accuracy_mean'],
                'grammar_ood': g.get('ood_accuracy'),
                'token_ood': t.get('ood_accuracy'),
                'iid_p_value': float(p_value),
                'iid_significant': p_value < 0.05,
            }
            
            sig = '*' if p_value < 0.05 else ''
            print(f"   {label_col}: Grammar IID={g['iid_accuracy_mean']:.1%} vs Token IID={t['iid_accuracy_mean']:.1%} (p={p_value:.4f}{sig})")
    
    results['comparisons'] = comparisons
    
    # Save results
    print(f"\n5. Saving results...")
    
    results_file = output_dir / 'ood_results.json'
    
    def convert(obj):
        """Recursively convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return convert(obj.tolist())
        if isinstance(obj, (np.float32, np.float64, np.floating)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64, np.integer)):
            return int(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj
    
    # Pre-process entire results dict
    results_serializable = convert(results)
    
    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"   Saved to: {results_file}")
    
    # Generate report
    report = f"""# OOD Evaluation Results

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Setup

- **OOD Families:** {', '.join(ood_families)}
- **Note:** {results.get('note', 'N/A')}

## Results

### Classification Accuracy

| Property | Grammar IID | Grammar OOD | Token IID | Token OOD | Winner |
|----------|-------------|-------------|-----------|-----------|--------|
"""
    
    for label_col in label_columns:
        g = results[label_col]['grammar']
        t = results[label_col]['token']
        
        g_iid = f"{g['iid_accuracy_mean']:.1%}"
        t_iid = f"{t['iid_accuracy_mean']:.1%}"
        g_ood = f"{g['ood_accuracy']:.1%}" if g.get('ood_accuracy') else "N/A"
        t_ood = f"{t['ood_accuracy']:.1%}" if t.get('ood_accuracy') else "N/A"
        
        # Winner based on OOD accuracy
        if g.get('ood_accuracy') and t.get('ood_accuracy'):
            winner = "Grammar" if g['ood_accuracy'] > t['ood_accuracy'] else "Token"
        else:
            winner = "N/A"
        
        report += f"| {label_col} | {g_iid} | {g_ood} | {t_iid} | {t_ood} | {winner} |\n"
    
    report += """
### Generalization Gap (IID - OOD)

| Property | Grammar Gap | Token Gap | Better Generalization |
|----------|-------------|-----------|----------------------|
"""
    
    for label_col in label_columns:
        g = results[label_col]['grammar']
        t = results[label_col]['token']
        
        g_gap = f"{g['generalization_gap']:+.1%}" if g.get('generalization_gap') else "N/A"
        t_gap = f"{t['generalization_gap']:+.1%}" if t.get('generalization_gap') else "N/A"
        
        if g.get('generalization_gap') is not None and t.get('generalization_gap') is not None:
            # Smaller gap = better generalization
            winner = "Grammar" if abs(g['generalization_gap']) < abs(t['generalization_gap']) else "Token"
        else:
            winner = "N/A"
        
        report += f"| {label_col} | {g_gap} | {t_gap} | {winner} |\n"
    
    report += """
## Key Findings

"""
    
    if not use_pretrained:
        report += """
### TRUE OOD Results

These results represent genuine out-of-distribution generalization because:
1. The VAE encoder has **never seen** KdV or Schrödinger PDEs during training
2. The latent representations for OOD families were computed at test time
3. This tests whether the learned physics concepts transfer to new equation families

"""
    else:
        report += """
### ⚠️ NOT TRUE OOD

**Warning:** These results use pre-trained VAE latents where the VAE was trained on ALL families.
The "OOD" here only means the LINEAR CLASSIFIER didn't see these families, but the VAE encoder did.

To run TRUE OOD experiments:
```bash
./scripts/train_ood_vaes.sh
python scripts/run_ood_evaluation.py
```

"""
    
    report_file = output_dir / 'OOD_REPORT.md'
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"   Report: {report_file}")
    
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
