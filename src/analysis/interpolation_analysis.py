"""Interpolation analysis for VAE latent spaces.

This module provides functions for:
- Interpolating between PDEs in latent space
- Measuring smoothness and physics preservation along interpolation paths
- Computing dimensionality and type preservation metrics

Updated to use rigorous PDEClassifier for accurate decoded PDE classification.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from collections import defaultdict
import logging

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

# Use rigorous classifier instead of simple heuristics
from analysis.pde_classifier import PDEClassifier, PDELabels

# Keep legacy imports for backward compatibility
from analysis.physics import PDE_PHYSICS

# Initialize rigorous classifier
_classifier = PDEClassifier()

logger = logging.getLogger(__name__)


def classify_pde_type(pde: str) -> str:
    """Classify PDE type using rigorous classifier."""
    try:
        labels = _classifier.classify(pde)
        return labels.pde_type
    except:
        return 'invalid'


def classify_spatial_dim(pde: str) -> int:
    """Classify spatial dimension using rigorous classifier."""
    try:
        labels = _classifier.classify(pde)
        return labels.dimension
    except:
        return -1


def classify_linearity(pde: str) -> str:
    """Classify linearity using rigorous classifier."""
    try:
        labels = _classifier.classify(pde)
        return labels.linearity
    except:
        return 'invalid'


def classify_order(pde: str) -> int:
    """Classify spatial order using rigorous classifier."""
    try:
        labels = _classifier.classify(pde)
        return labels.spatial_order
    except:
        return -1


def classify_family(pde: str) -> str:
    """Classify PDE family using rigorous classifier."""
    try:
        labels = _classifier.classify(pde)
        return labels.family
    except:
        return 'unknown'


def is_valid_pde(pde: str) -> bool:
    """Check if PDE string is valid."""
    if not pde or pde == '[INVALID]':
        return False
    
    # Basic structural checks
    if pde.count('(') != pde.count(')'):
        return False
    
    # Check for required components
    pde_lower = pde.lower()
    has_derivative = any(x in pde_lower for x in ['dx', 'dy', 'dz', 'dt', '_x', '_y', '_z', '_t'])
    has_variable = 'u' in pde_lower
    
    if not has_derivative or not has_variable:
        return False
    
    return True


def get_full_classification(pde: str) -> Optional[PDELabels]:
    """Get full classification using rigorous classifier."""
    try:
        return _classifier.classify(pde)
    except:
        return None


def load_vae_model(checkpoint_path: str, device: str = 'cuda'):
    """Load VAE model from checkpoint."""
    from vae.module import GrammarVAEModule

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    hparams = checkpoint['hyper_parameters']

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

    return model, hparams


def encode_pde(model, pde: str, tokenization: str, max_length: int, vocab_size: int, device: str):
    """Encode a PDE string to latent vector."""
    if tokenization == 'grammar':
        from pde import grammar as pde_grammar
        try:
            seq = pde_grammar.parse_to_productions(pde.replace(' ', ''))
            x = torch.zeros(max_length, vocab_size)
            for t, pid in enumerate(seq[:max_length]):
                if 0 <= pid < vocab_size:
                    x[t, pid] = 1.0
        except:
            return None
    else:
        from pde.chr_tokenizer import PDETokenizer
        tokenizer = PDETokenizer()
        try:
            ids = tokenizer.encode(pde)
            x = torch.zeros(max_length, vocab_size)
            for t, tid in enumerate(ids[:max_length]):
                if 0 <= tid < vocab_size:
                    x[t, tid] = 1.0
        except:
            return None

    x = x.unsqueeze(0).to(device)
    with torch.no_grad():
        mu, _ = model.encoder(x)
    return mu


def decode_latent(model, z: torch.Tensor, tokenization: str, use_constrained: bool = False) -> str:
    """Decode a latent vector to PDE string."""
    model.eval()
    with torch.no_grad():
        if use_constrained and hasattr(model, 'generate_constrained'):
            pred = model.generate_constrained(z, greedy=True).cpu().numpy()
        else:
            logits = model.decoder(z)
            pred = logits.argmax(dim=-1).cpu().numpy()

    ids = pred[0] if len(pred.shape) > 1 else pred

    if tokenization == 'grammar':
        from pde import grammar as pde_grammar
        try:
            valid_ids = [int(i) for i in ids if 0 <= i < pde_grammar.PROD_COUNT]
            return pde_grammar.decode_production_sequence(valid_ids)
        except:
            return None
    else:
        from pde.chr_tokenizer import PDETokenizer
        tokenizer = PDETokenizer()
        try:
            ids_list = [int(i) for i in ids if i >= 0]
            return tokenizer.decode_to_infix(ids_list, skip_special_tokens=True)
        except:
            return None


def interpolate_latents(z1: torch.Tensor, z2: torch.Tensor, n_steps: int = 11) -> List[torch.Tensor]:
    """Generate linear interpolation between two latent vectors."""
    alphas = np.linspace(0, 1, n_steps)
    return [(1 - alpha) * z1 + alpha * z2 for alpha in alphas]


def analyze_interpolation_path(
    model,
    z1: torch.Tensor,
    z2: torch.Tensor,
    tokenization: str,
    n_steps: int = 11,
    use_constrained: bool = False
) -> Dict:
    """Analyze a single interpolation path using rigorous PDEClassifier.

    Returns:
        Dictionary with decoded PDEs and physics analysis
    """
    interpolated_z = interpolate_latents(z1, z2, n_steps)

    results = {
        'alphas': np.linspace(0, 1, n_steps).tolist(),
        'pdes': [],
        'valid': [],
        'dims': [],
        'types': [],
        'linearities': [],
        'orders': [],
        'families': [],  # NEW: Track family classification
        'temporal_orders': [],  # NEW: Track temporal order
        'confidences': [],  # NEW: Track classifier confidence
    }

    for z in interpolated_z:
        pde = decode_latent(model, z, tokenization, use_constrained)

        if pde and is_valid_pde(pde):
            # Use rigorous classifier for all labels
            labels = get_full_classification(pde)
            
            if labels:
                results['pdes'].append(pde)
                results['valid'].append(True)
                results['dims'].append(labels.dimension)
                results['types'].append(labels.pde_type)
                results['linearities'].append(labels.linearity)
                results['orders'].append(labels.spatial_order)
                results['families'].append(labels.family)
                results['temporal_orders'].append(labels.temporal_order)
                results['confidences'].append(labels.confidence)
            else:
                # Classification failed
                results['pdes'].append(pde)
                results['valid'].append(False)
                results['dims'].append(-1)
                results['types'].append('invalid')
                results['linearities'].append('invalid')
                results['orders'].append(-1)
                results['families'].append('unknown')
                results['temporal_orders'].append(-1)
                results['confidences'].append(0.0)
        else:
            # Decoding failed or invalid PDE
            results['pdes'].append(pde if pde else '[DECODE_FAILED]')
            results['valid'].append(False)
            results['dims'].append(-1)
            results['types'].append('invalid')
            results['linearities'].append('invalid')
            results['orders'].append(-1)
            results['families'].append('unknown')
            results['temporal_orders'].append(-1)
            results['confidences'].append(0.0)

    return results


def count_changes(values: List) -> int:
    """Count number of changes in a sequence."""
    if len(values) < 2:
        return 0
    return sum(1 for i in range(1, len(values)) if values[i] != values[i-1])


def compute_path_metrics(path_results: Dict) -> Dict:
    """Compute metrics for a single interpolation path."""
    valid = path_results['valid']
    dims = path_results['dims']
    types = path_results['types']
    linearities = path_results['linearities']
    families = path_results.get('families', ['unknown'] * len(valid))
    temporal_orders = path_results.get('temporal_orders', [-1] * len(valid))
    confidences = path_results.get('confidences', [0.0] * len(valid))

    # Filter to valid steps only for some metrics
    valid_indices = [i for i, v in enumerate(valid) if v]
    
    # Compute valid-only sequences for transition analysis
    valid_types = [types[i] for i in valid_indices] if valid_indices else []
    valid_families = [families[i] for i in valid_indices] if valid_indices else []
    valid_dims = [dims[i] for i in valid_indices] if valid_indices else []

    return {
        'validity_rate': sum(valid) / len(valid) if valid else 0,
        'dim_changes': count_changes(dims),
        'type_changes': count_changes(types),
        'linearity_changes': count_changes(linearities),
        'family_changes': count_changes(families),  # NEW
        'temporal_order_changes': count_changes(temporal_orders),  # NEW
        'start_dim': dims[0] if dims else -1,
        'end_dim': dims[-1] if dims else -1,
        'dim_preserved': dims[0] == dims[-1] and all(d == dims[0] for d in dims) if dims else False,
        'start_type': types[0] if types else 'unknown',
        'end_type': types[-1] if types else 'unknown',
        'start_family': families[0] if families else 'unknown',  # NEW
        'end_family': families[-1] if families else 'unknown',  # NEW
        'family_preserved': families[0] == families[-1] and all(f == families[0] for f in families) if families else False,  # NEW
        # Smooth transition metrics (only count changes in valid PDEs)
        'valid_type_changes': count_changes(valid_types),  # NEW
        'valid_family_changes': count_changes(valid_families),  # NEW
        'valid_dim_changes': count_changes(valid_dims),  # NEW
        'avg_confidence': np.mean(confidences) if confidences else 0.0,  # NEW
        'n_valid': len(valid_indices),  # NEW
        'n_total': len(valid),  # NEW
    }


def run_interpolation_suite(
    model,
    latents: np.ndarray,
    families: np.ndarray,
    tokenization: str,
    n_pairs: int = 50,
    n_steps: int = 11,
    seed: int = 42,
    use_constrained: bool = False,
    device: str = 'cuda'
) -> Dict:
    """Run full interpolation analysis suite with rigorous PDE classification.

    Args:
        model: VAE model
        latents: (N, z_dim) latent vectors
        families: (N,) family labels
        tokenization: 'grammar' or 'token'
        n_pairs: Number of random pairs to sample
        n_steps: Interpolation steps
        seed: Random seed
        use_constrained: Use constrained decoding
        device: Device

    Returns:
        Comprehensive interpolation results with rigorous physics labels
    """
    np.random.seed(seed)

    logger.info(f"Running interpolation suite with rigorous PDEClassifier...")
    logger.info(f"  Tokenization: {tokenization}")
    logger.info(f"  Pairs: {n_pairs}, Steps: {n_steps}")

    # Define interesting family pairs (physics-motivated transitions)
    interesting_pairs = [
        ('heat', 'wave'),           # parabolic → hyperbolic (temporal order change)
        ('heat', 'advection'),      # diffusion → transport (spatial order change)
        ('burgers', 'kdv'),         # nonlinear transport → soliton (mechanism change)
        ('poisson', 'biharmonic'),  # 2nd → 4th order elliptic
        ('wave', 'telegraph'),      # undamped → damped hyperbolic
        ('heat', 'allen_cahn'),     # linear → nonlinear parabolic
        ('fisher_kpp', 'allen_cahn'),  # both reaction-diffusion
    ]

    # Group indices by family
    family_indices = defaultdict(list)
    for i, fam in enumerate(families):
        family_indices[str(fam)].append(i)

    results = {
        'canonical_pairs': [],
        'random_pairs': [],
        'metrics': {
            'dim_preservation': {'1D→1D': [], '2D→2D': [], '3D→3D': [], 'cross_dim': []},
            'type_changes': [],
            'validity_rates': [],
            'family_changes': [],  # NEW
            'linearity_changes': [],  # NEW
            'valid_type_changes': [],  # NEW: changes only among valid PDEs
            'avg_confidences': [],  # NEW
        }
    }

    # Canonical pairs (physics-motivated transitions)
    print(f"  Analyzing canonical family pairs...")
    for fam1, fam2 in interesting_pairs:
        if fam1 in family_indices and fam2 in family_indices:
            idx1 = np.random.choice(family_indices[fam1])
            idx2 = np.random.choice(family_indices[fam2])

            z1 = torch.from_numpy(latents[idx1:idx1+1]).float().to(device)
            z2 = torch.from_numpy(latents[idx2:idx2+1]).float().to(device)

            path_results = analyze_interpolation_path(
                model, z1, z2, tokenization, n_steps, use_constrained
            )
            path_metrics = compute_path_metrics(path_results)

            results['canonical_pairs'].append({
                'from_family': fam1,
                'to_family': fam2,
                'path': path_results,
                'metrics': path_metrics,
            })
            
            logger.info(f"    {fam1} → {fam2}: validity={path_metrics['validity_rate']:.1%}, "
                       f"type_changes={path_metrics['type_changes']}")

    # Random pairs
    print(f"  Analyzing {n_pairs} random pairs...")
    all_indices = list(range(len(latents)))
    sampled_pairs = []

    for _ in range(n_pairs):
        idx1, idx2 = np.random.choice(all_indices, 2, replace=False)
        sampled_pairs.append((idx1, idx2))

    for idx1, idx2 in sampled_pairs:
        z1 = torch.from_numpy(latents[idx1:idx1+1]).float().to(device)
        z2 = torch.from_numpy(latents[idx2:idx2+1]).float().to(device)

        fam1, fam2 = str(families[idx1]), str(families[idx2])

        path_results = analyze_interpolation_path(
            model, z1, z2, tokenization, n_steps, use_constrained
        )
        path_metrics = compute_path_metrics(path_results)

        results['random_pairs'].append({
            'from_family': fam1,
            'to_family': fam2,
            'metrics': path_metrics,
        })

        # Aggregate metrics
        results['metrics']['type_changes'].append(path_metrics['type_changes'])
        results['metrics']['validity_rates'].append(path_metrics['validity_rate'])
        results['metrics']['family_changes'].append(path_metrics['family_changes'])
        results['metrics']['linearity_changes'].append(path_metrics['linearity_changes'])
        results['metrics']['valid_type_changes'].append(path_metrics['valid_type_changes'])
        results['metrics']['avg_confidences'].append(path_metrics['avg_confidence'])

        # Track dimension preservation
        start_dim, end_dim = path_metrics['start_dim'], path_metrics['end_dim']
        if start_dim == end_dim and start_dim > 0:
            key = f'{start_dim}D→{end_dim}D'
            if key in results['metrics']['dim_preservation']:
                results['metrics']['dim_preservation'][key].append(
                    1.0 if path_metrics['dim_preserved'] else 0.0
                )
        elif start_dim > 0 and end_dim > 0:
            results['metrics']['dim_preservation']['cross_dim'].append(
                1.0 if path_metrics['dim_preserved'] else 0.0
            )

    # Compute summary statistics
    summary = {
        'avg_type_changes': np.mean(results['metrics']['type_changes']) if results['metrics']['type_changes'] else 0,
        'avg_valid_type_changes': np.mean(results['metrics']['valid_type_changes']) if results['metrics']['valid_type_changes'] else 0,
        'avg_family_changes': np.mean(results['metrics']['family_changes']) if results['metrics']['family_changes'] else 0,
        'avg_linearity_changes': np.mean(results['metrics']['linearity_changes']) if results['metrics']['linearity_changes'] else 0,
        'validity_rate': np.mean(results['metrics']['validity_rates']) if results['metrics']['validity_rates'] else 0,
        'avg_confidence': np.mean(results['metrics']['avg_confidences']) if results['metrics']['avg_confidences'] else 0,
        'dim_preservation': {},
    }

    for key, values in results['metrics']['dim_preservation'].items():
        if values:
            summary['dim_preservation'][key] = np.mean(values)
        else:
            summary['dim_preservation'][key] = 0.0

    # Overall dimension preservation
    all_dim_preserved = []
    for key in ['1D→1D', '2D→2D', '3D→3D']:
        all_dim_preserved.extend(results['metrics']['dim_preservation'].get(key, []))
    summary['dim_preservation']['Overall'] = np.mean(all_dim_preserved) if all_dim_preserved else 0.0

    results['summary'] = summary
    
    # Print summary
    print(f"\n  Summary for {tokenization}:")
    print(f"    Validity rate: {summary['validity_rate']:.1%}")
    print(f"    Avg type changes: {summary['avg_type_changes']:.2f}")
    print(f"    Avg family changes: {summary['avg_family_changes']:.2f}")
    print(f"    Avg classifier confidence: {summary['avg_confidence']:.2f}")

    return results


def compare_tokenizations(
    grammar_results: Dict,
    token_results: Dict
) -> Dict:
    """Compare interpolation results between tokenizations with full metrics."""
    comparison = {
        'grammar': grammar_results['summary'],
        'token': token_results['summary'],
        'comparison': {}
    }

    # Compare key metrics
    g_sum = grammar_results['summary']
    t_sum = token_results['summary']

    # Type changes (lower is better - smoother transitions)
    comparison['comparison']['type_changes'] = {
        'grammar': g_sum['avg_type_changes'],
        'token': t_sum['avg_type_changes'],
        'winner': 'Grammar' if g_sum['avg_type_changes'] < t_sum['avg_type_changes'] else 'Token',
        'description': 'Average number of PDE type changes along path (lower = smoother)'
    }
    
    # Valid type changes (changes among valid PDEs only)
    comparison['comparison']['valid_type_changes'] = {
        'grammar': g_sum.get('avg_valid_type_changes', 0),
        'token': t_sum.get('avg_valid_type_changes', 0),
        'winner': 'Grammar' if g_sum.get('avg_valid_type_changes', 0) < t_sum.get('avg_valid_type_changes', 0) else 'Token',
        'description': 'Type changes among valid PDEs only'
    }
    
    # Family changes (lower is better)
    comparison['comparison']['family_changes'] = {
        'grammar': g_sum.get('avg_family_changes', 0),
        'token': t_sum.get('avg_family_changes', 0),
        'winner': 'Grammar' if g_sum.get('avg_family_changes', 0) < t_sum.get('avg_family_changes', 0) else 'Token',
        'description': 'Average family changes along path'
    }
    
    # Linearity changes
    comparison['comparison']['linearity_changes'] = {
        'grammar': g_sum.get('avg_linearity_changes', 0),
        'token': t_sum.get('avg_linearity_changes', 0),
        'winner': 'Grammar' if g_sum.get('avg_linearity_changes', 0) < t_sum.get('avg_linearity_changes', 0) else 'Token',
        'description': 'Average linearity changes along path'
    }

    # Validity rate (higher is better)
    comparison['comparison']['validity_rate'] = {
        'grammar': g_sum['validity_rate'],
        'token': t_sum['validity_rate'],
        'winner': 'Grammar' if g_sum['validity_rate'] > t_sum['validity_rate'] else 'Token',
        'description': 'Fraction of interpolated points that decode to valid PDEs'
    }
    
    # Classifier confidence (higher is better)
    comparison['comparison']['avg_confidence'] = {
        'grammar': g_sum.get('avg_confidence', 0),
        'token': t_sum.get('avg_confidence', 0),
        'winner': 'Grammar' if g_sum.get('avg_confidence', 0) > t_sum.get('avg_confidence', 0) else 'Token',
        'description': 'Average classifier confidence in decoded PDEs'
    }

    # Dimension preservation
    for dim_key in ['1D→1D', '2D→2D', '3D→3D', 'Overall']:
        g_val = g_sum['dim_preservation'].get(dim_key, 0)
        t_val = t_sum['dim_preservation'].get(dim_key, 0)
        comparison['comparison'][f'dim_{dim_key}'] = {
            'grammar': g_val,
            'token': t_val,
            'winner': 'Grammar' if g_val > t_val else 'Token',
            'description': f'Dimension preservation for {dim_key} paths'
        }
    
    # Summary: count wins
    grammar_wins = sum(1 for k, v in comparison['comparison'].items() if v.get('winner') == 'Grammar')
    token_wins = sum(1 for k, v in comparison['comparison'].items() if v.get('winner') == 'Token')
    
    comparison['summary'] = {
        'grammar_wins': grammar_wins,
        'token_wins': token_wins,
        'overall_winner': 'Grammar' if grammar_wins > token_wins else 'Token' if token_wins > grammar_wins else 'Tie'
        }

    return comparison


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run interpolation analysis')
    parser.add_argument('--grammar-ckpt', type=str, required=True)
    parser.add_argument('--token-ckpt', type=str, required=True)
    parser.add_argument('--latent-file', type=str, default='visualization_maps/latent_data.npz')
    parser.add_argument('--n-pairs', type=int, default=50)
    parser.add_argument('--output', type=str, default='experiments/reports/interpolation_results.json')

    args = parser.parse_args()

    # Load latents
    data = np.load(args.latent_file, allow_pickle=True)
    grammar_latents = data['grammar_latents']
    token_latents = data['token_latents']
    families = data['families']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load models and run analysis
    print("Loading Grammar VAE...")
    grammar_model, grammar_hparams = load_vae_model(args.grammar_ckpt, device)

    print("Loading Token VAE...")
    token_model, token_hparams = load_vae_model(args.token_ckpt, device)

    print("\nRunning Grammar interpolation analysis...")
    grammar_results = run_interpolation_suite(
        grammar_model, grammar_latents, families, 'grammar',
        n_pairs=args.n_pairs, use_constrained=True, device=device
    )

    print("\nRunning Token interpolation analysis...")
    token_results = run_interpolation_suite(
        token_model, token_latents, families, 'token',
        n_pairs=args.n_pairs, use_constrained=False, device=device
    )

    print("\nComparing results...")
    comparison = compare_tokenizations(grammar_results, token_results)

    # Save results
    import json
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(args.output, 'w') as f:
        json.dump(convert({
            'grammar': grammar_results,
            'token': token_results,
            'comparison': comparison
        }), f, indent=2)

    print(f"\nResults saved to {args.output}")
    print(f"\nSummary:")
    print(f"  Grammar avg type changes: {grammar_results['summary']['avg_type_changes']:.2f}")
    print(f"  Token avg type changes: {token_results['summary']['avg_type_changes']:.2f}")
    print(f"  Grammar validity rate: {grammar_results['summary']['validity_rate']:.1%}")
    print(f"  Token validity rate: {token_results['summary']['validity_rate']:.1%}")
