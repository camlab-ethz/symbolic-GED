#!/usr/bin/env python3
"""Compare all 4 models using the SAME 1000 latent vectors.

Analyzes:
1. Correctness (valid strings, no cutoff PDEs, always PDEs)
2. Uniqueness (exact string matches in canonicalized form)
3. Operator pattern uniqueness and novel skeleton patterns
"""

import argparse
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Set, Tuple, Dict

import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vae.module import VAEModule
from pde.grammar import decode_production_sequence
from pde import PDETokenizer
from analysis.physics import is_valid_pde


def canonicalize_pde(pde_str: str) -> str:
    """Canonicalize PDE by normalizing term ordering.
    
    Example: "dt(u)+dx(u)" and "dx(u)+dt(u)" -> same canonical form
    
    Strategy:
    1. Extract core structures (ignoring coefficients)
    2. Sort by structure
    3. Normalize to form: sorted_terms = 0
    """
    if not pde_str or pde_str.startswith('[') or '=' not in pde_str:
        return pde_str
    
    try:
        # Split into LHS and RHS
        parts = pde_str.split('=', 1)
        if len(parts) != 2:
            return pde_str
        
        lhs, rhs = parts[0].strip(), parts[1].strip()
        
        # Parse into terms with signs
        lhs_terms = _parse_simple_terms(lhs)
        rhs_terms = _parse_simple_terms(rhs)
        
        # Move RHS terms to LHS (flip signs)
        all_terms = []
        for sign, term in lhs_terms:
            all_terms.append((sign, term))
        for sign, term in rhs_terms:
            # Flip sign when moving to LHS
            new_sign = '-' if sign == '+' else '+'
            if term != '0':  # Skip zero
                all_terms.append((new_sign, term))
        
        # Extract core structures (ignore coefficients)
        core_structures = []
        for sign, term in all_terms:
            core = _extract_core_structure(term)
            if core:
                core_structures.append((core, sign, term))
        
        # Sort by core structure, then by sign (positive first)
        core_structures.sort(key=lambda x: (x[0], 0 if x[1] == '+' else 1, x[2]))
        
        # Reconstruct: put all terms on LHS, RHS = 0
        if not core_structures:
            return "0 = 0"
        
        lhs_parts = []
        for core, sign, term in core_structures:
            if sign == '+':
                lhs_parts.append(f"+{term}")
            else:
                lhs_parts.append(f"-{term}")
        
        canonical_lhs = "".join(lhs_parts).lstrip('+')
        return f"{canonical_lhs} = 0"
    
    except Exception as e:
        # If canonicalization fails, return original
        return pde_str


def _parse_simple_terms(expr: str) -> List[Tuple[str, str]]:
    """Parse expression into (sign, term) pairs.
    
    Returns: List of (sign, term) where sign is '+' or '-'
    """
    if not expr or expr.strip() == '0':
        return [('+', '0')]
    
    expr = expr.strip()
    terms = []
    
    # Handle first term (might not have sign)
    i = 0
    while i < len(expr):
        if expr[i] in ['+', '-']:
            sign = expr[i]
            i += 1
            # Skip spaces
            while i < len(expr) and expr[i].isspace():
                i += 1
            # Find next term
            start = i
            while i < len(expr) and expr[i] not in ['+', '-']:
                i += 1
            term = expr[start:i].strip()
            if term:
                terms.append((sign, term))
        else:
            # First term (positive by default)
            start = i
            while i < len(expr) and expr[i] not in ['+', '-']:
                i += 1
            term = expr[start:i].strip()
            if term:
                terms.append(('+', term))
    
    return terms if terms else [('+', '0')]


def _extract_core_structure(term: str) -> str:
    """Extract core structure ignoring coefficients.
    
    Examples:
      "2.5*dt(u)" -> "dt"
      "u^3" -> "u^3"
      "dxx(u)" -> "dxx"
      "u*dx(u)" -> "u*dx"
      "3.14" -> "[CONST]"
    """
    term = term.strip()
    if not term:
        return ""
    
    # Extract coefficient if present
    coeff_match = re.match(r'^([+-]?\d+\.?\d*)\s*\*', term)
    if coeff_match:
        remainder = term[coeff_match.end():].strip()
    else:
        # Check if it's just a number
        if re.match(r'^([+-]?\d+\.?\d*)\s*$', term):
            return "[CONST]"
        remainder = term
    
    # Check for derivatives
    deriv_match = re.search(r'(d[xyz]{1,4}\(|dt\(|dtt\()', remainder)
    if deriv_match:
        deriv_type = deriv_match.group(1).rstrip('(')
        
        # Check what's inside
        inner_start = deriv_match.end()
        depth = 1
        i = inner_start
        while i < len(remainder) and depth > 0:
            if remainder[i] == '(':
                depth += 1
            elif remainder[i] == ')':
                depth -= 1
            i += 1
        
        inner = remainder[inner_start:i-1] if i > inner_start else ""
        
        # Check for special patterns in inner
        if 'u^3' in inner or re.search(r'u\s*\^\s*3|u\s*\*\*\s*3', inner):
            return f"{deriv_type}(u^3)"
        elif 'u^2' in inner or re.search(r'u\s*\^\s*2|u\s*\*\*\s*2', inner):
            return f"{deriv_type}(u^2)"
        else:
            return deriv_type
    
    # Check for nonlinear patterns
    if 'u^3' in remainder or re.search(r'u\s*\^\s*3|u\s*\*\*\s*3', remainder):
        return "u^3"
    if 'u^2' in remainder or re.search(r'u\s*\^\s*2|u\s*\*\*\s*2', remainder):
        return "u^2"
    if 'u*dx(' in remainder or 'u*dy(' in remainder or 'u*dz(' in remainder:
        return "u*dx"
    if '(dx(u))^2' in remainder or '(dy(u))^2' in remainder or '(dz(u))^2' in remainder:
        return "(dx)^2"
    if remainder.strip() == 'u':
        return "u"
    
    return remainder


# Removed old parsing functions - using simpler _parse_simple_terms and _extract_core_structure above


def extract_skeleton_pattern(pde_str: str) -> Tuple[Set[str], Set[str]]:
    """Extract skeleton pattern: derivative types and nonlinear patterns.
    
    Returns: (derivative_types, nonlinear_patterns)
    """
    deriv_types = set()
    nonlinear_patterns = set()
    
    # Extract all derivative types
    deriv_matches = re.findall(r'(d[xyz]{1,4}\(|dt\(|dtt\()', pde_str)
    deriv_types.update([d.rstrip('(') for d in deriv_matches])
    
    # Extract nonlinear patterns
    if 'u^3' in pde_str or 'u**3' in pde_str or re.search(r'u\s*\^\s*3|u\s*\*\*\s*3', pde_str):
        nonlinear_patterns.add('u^3')
    if 'u^2' in pde_str or 'u**2' in pde_str or re.search(r'u\s*\^\s*2|u\s*\*\*\s*2', pde_str):
        nonlinear_patterns.add('u^2')
    if 'u*dx(' in pde_str or 'u*dy(' in pde_str or 'u*dz(' in pde_str:
        nonlinear_patterns.add('u*dx')
    if '(dx(u))^2' in pde_str or '(dy(u))^2' in pde_str or '(dz(u))^2' in pde_str:
        nonlinear_patterns.add('(dx)^2')
    if 'dxx(u^3)' in pde_str or 'dyy(u^3)' in pde_str:
        nonlinear_patterns.add('dxx(u^3)')
    
    return (deriv_types, nonlinear_patterns)


def load_model(checkpoint_path, tokenization_type, device='cuda'):
    """Load VAE model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    model = VAEModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    model = model.to(device)
    model._tokenization_type = tokenization_type
    return model


def generate_from_grammar_vae(model, z, device='cuda'):
    """Generate sequences from grammar VAE."""
    z = z.to(device)
    with torch.no_grad():
        prod_ids = model.generate_constrained(z, greedy=True)
    return prod_ids.cpu()


def generate_from_token_vae(model, z, device='cuda'):
    """Generate sequences from token VAE."""
    z = z.to(device)
    with torch.no_grad():
        logits = model.decoder(z)  # (B, T, P)
        token_ids = logits.argmax(dim=-1)  # (B, T)
    return token_ids.cpu()


def prod_ids_to_string(prod_ids):
    """Convert production ID sequence to PDE string."""
    try:
        prod_ids_list = prod_ids.tolist() if isinstance(prod_ids, torch.Tensor) else prod_ids
        from pde.grammar import PROD_COUNT
        valid_prod_ids = [pid for pid in prod_ids_list if 0 <= pid < PROD_COUNT]
        if not valid_prod_ids:
            return "[INVALID: No valid productions]"
        pde_str = decode_production_sequence(valid_prod_ids)
        return pde_str if pde_str else "[INVALID: Empty sequence]"
    except Exception as e:
        return f"[ERROR: {e}]"


def token_ids_to_string(token_ids, tokenizer):
    """Convert token ID sequence to PDE string."""
    try:
        token_ids_list = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else token_ids
        vocab = tokenizer.vocab
        pad_id = vocab.pad_id
        
        valid_token_ids = []
        for tid in token_ids_list:
            if tid < 0 or tid == pad_id:
                continue
            if tid < len(vocab.id2word):
                valid_token_ids.append(tid)
        
        if not valid_token_ids:
            return "[INVALID: No valid tokens]"
        
        pde_str = tokenizer.decode_to_infix(valid_token_ids)
        return pde_str if pde_str else "[INVALID: Empty sequence]"
    except Exception as e:
        return f"[ERROR: {e}]"


def analyze_training_skeletons():
    """Analyze training dataset to extract skeleton patterns."""
    training_skeletons = {
        'derivative_combos': Counter(),
        'nonlinear_patterns': Counter(),
    }
    
    dataset_path = Path(__file__).parent.parent / "data/raw/pde_dataset_48000_fixed.csv"
    if not dataset_path.exists():
        return training_skeletons
    
    print("Analyzing training dataset for skeleton patterns...")
    with open(dataset_path, 'r') as f:
        header = f.readline()
        total = 0
        for line in f:
            parts = line.strip().split(',')
            if len(parts) > 0:
                pde = parts[0]
                deriv_types, nonlinear = extract_skeleton_pattern(pde)
                
                # Create sorted tuple for derivative combination
                deriv_combo = tuple(sorted(deriv_types))
                training_skeletons['derivative_combos'][deriv_combo] += 1
                training_skeletons['nonlinear_patterns'].update(nonlinear)
                total += 1
    
    print(f"  Analyzed {total} PDEs\n")
    return training_skeletons


def main():
    parser = argparse.ArgumentParser(description='Compare models using same latent vectors')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--grammar_beta2e4', type=str, default=None)
    parser.add_argument('--grammar_beta1e2', type=str, default=None)
    parser.add_argument('--token_beta2e4', type=str, default=None)
    parser.add_argument('--token_beta1e2', type=str, default=None)
    
    args = parser.parse_args()
    
    # Default checkpoint paths
    base_dir = Path(__file__).parent.parent
    if args.grammar_beta2e4 is None:
        args.grammar_beta2e4 = str(base_dir / "checkpoints/grammar_vae/beta_2e-4_seed_42/best-epoch=380-val/seq_acc=0.9978.ckpt")
    if args.grammar_beta1e2 is None:
        args.grammar_beta1e2 = str(base_dir / "checkpoints/grammar_vae/beta_0.01_seed_42/best-epoch=517-val/seq_acc=0.0755.ckpt")
    if args.token_beta2e4 is None:
        args.token_beta2e4 = str(base_dir / "checkpoints/token_vae/beta_2e-4_seed_42/best-epoch=433-val/seq_acc=0.9962.ckpt")
    if args.token_beta1e2 is None:
        args.token_beta1e2 = str(base_dir / "checkpoints/token_vae/beta_0.01_seed_42/best-epoch=136-val/seq_acc=0.0016.ckpt")
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 80)
    print("COMPARING MODELS WITH SAME LATENT VECTORS")
    print("=" * 80)
    print(f"Number of samples: {args.n_samples}")
    print(f"Random seed: {args.seed}")
    print()
    
    # Load models
    models = {}
    models['grammar_beta2e4'] = load_model(args.grammar_beta2e4, 'grammar', args.device)
    models['grammar_beta1e2'] = load_model(args.grammar_beta1e2, 'grammar', args.device)
    models['token_beta2e4'] = load_model(args.token_beta2e4, 'token', args.device)
    models['token_beta1e2'] = load_model(args.token_beta1e2, 'token', args.device)
    
    z_dim = models['grammar_beta2e4'].z_dim
    print(f"Latent dimension: {z_dim}\n")
    
    # Sample SAME latent vectors for all models
    print(f"Sampling {args.n_samples} random latent vectors (SAME for all models)...")
    z = torch.randn(args.n_samples, z_dim, device=args.device)
    print()
    
    # Initialize tokenizer
    tokenizer = PDETokenizer()
    
    # Analyze training skeletons
    training_skeletons = analyze_training_skeletons()
    training_deriv_combos = set(training_skeletons['derivative_combos'].keys())
    training_nonlinear = set(training_skeletons['nonlinear_patterns'].keys())
    
    # Generate from each model
    results = {}
    print("Generating sequences from all models...\n")
    
    for model_name, model in models.items():
        print(f"Generating from {model_name}...")
        tokenization_type = model._tokenization_type
        
        if tokenization_type == 'grammar':
            seq_ids = generate_from_grammar_vae(model, z, args.device)
        else:
            seq_ids = generate_from_token_vae(model, z, args.device)
        
        pde_strings = []
        canonicalized = []
        valid_pdes = []
        skeletons = []
        
        for i in range(args.n_samples):
            if tokenization_type == 'grammar':
                pde_str = prod_ids_to_string(seq_ids[i])
            else:
                pde_str = token_ids_to_string(seq_ids[i], tokenizer)
            
            pde_strings.append(pde_str)
            
            # Check validity
            is_valid = is_valid_pde(pde_str)
            if is_valid:
                valid_pdes.append(pde_str)
                canon = canonicalize_pde(pde_str)
                canonicalized.append(canon)
                
                deriv_types, nonlinear = extract_skeleton_pattern(pde_str)
                deriv_combo = tuple(sorted(deriv_types))
                skeletons.append({
                    'deriv_combo': deriv_combo,
                    'nonlinear': nonlinear,
                    'original': pde_str,
                    'canonical': canon,
                })
        
        results[model_name] = {
            'pde_strings': pde_strings,
            'valid_pdes': valid_pdes,
            'canonicalized': canonicalized,
            'skeletons': skeletons,
            'valid_count': len(valid_pdes),
            'valid_ratio': len(valid_pdes) / args.n_samples,
        }
        
        print(f"  Valid PDEs: {len(valid_pdes)}/{args.n_samples} ({100*len(valid_pdes)/args.n_samples:.2f}%)\n")
    
    # Analysis
    print("=" * 80)
    print("ANALYSIS RESULTS")
    print("=" * 80)
    print()
    
    # 1. Correctness
    print("1. CORRECTNESS:")
    print("-" * 80)
    print(f"{'Model':<25} {'Valid':<10} {'Valid %':<10}")
    print("-" * 80)
    for model_name in ['grammar_beta2e4', 'grammar_beta1e2', 'token_beta2e4', 'token_beta1e2']:
        valid = results[model_name]['valid_count']
        ratio = results[model_name]['valid_ratio']
        print(f"{model_name:<25} {valid:<10} {100*ratio:>6.2f}%")
    print()
    
    # 2. Uniqueness (canonicalized)
    print("2. UNIQUENESS (Canonicalized):")
    print("-" * 80)
    print(f"{'Model':<25} {'Unique':<10} {'Unique %':<10}")
    print("-" * 80)
    for model_name in ['grammar_beta2e4', 'grammar_beta1e2', 'token_beta2e4', 'token_beta1e2']:
        canonicalized = results[model_name]['canonicalized']
        unique_count = len(set(canonicalized))
        unique_ratio = unique_count / len(canonicalized) if canonicalized else 0
        print(f"{model_name:<25} {unique_count:<10} {100*unique_ratio:>6.2f}%")
    print()
    
    # 3. Skeleton patterns
    print("3. SKELETON PATTERNS:")
    print("-" * 80)
    
    for model_name in ['grammar_beta2e4', 'grammar_beta1e2', 'token_beta2e4', 'token_beta1e2']:
        skeletons = results[model_name]['skeletons']
        
        # Extract unique derivative combinations
        deriv_combos = Counter([s['deriv_combo'] for s in skeletons])
        nonlinear_patterns = Counter()
        for s in skeletons:
            nonlinear_patterns.update(s['nonlinear'])
        
        # Find novel patterns
        novel_deriv_combos = set(deriv_combos.keys()) - training_deriv_combos
        novel_nonlinear = set(nonlinear_patterns.keys()) - training_nonlinear
        
        print(f"\n{model_name}:")
        print(f"  Unique derivative combos: {len(deriv_combos)}")
        print(f"  Unique nonlinear patterns: {len(nonlinear_patterns)}")
        print(f"  Novel derivative combos: {len(novel_deriv_combos)}")
        if novel_deriv_combos:
            print(f"    Examples: {list(novel_deriv_combos)[:5]}")
        print(f"  Novel nonlinear patterns: {len(novel_nonlinear)}")
        if novel_nonlinear:
            print(f"    Examples: {sorted(novel_nonlinear)}")
    
    # Save results
    output_dir = base_dir / "generation_results"
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / f"model_comparison_same_latents_n{args.n_samples}_seed{args.seed}.txt"
    with open(results_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL COMPARISON (SAME LATENT VECTORS)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Number of samples: {args.n_samples}\n")
        f.write(f"Random seed: {args.seed}\n\n")
        
        f.write("1. CORRECTNESS:\n")
        f.write("-" * 80 + "\n")
        for model_name in ['grammar_beta2e4', 'grammar_beta1e2', 'token_beta2e4', 'token_beta1e2']:
            valid = results[model_name]['valid_count']
            ratio = results[model_name]['valid_ratio']
            f.write(f"{model_name}: {valid}/{args.n_samples} ({100*ratio:.2f}%)\n")
        
        f.write("\n2. UNIQUENESS (Canonicalized):\n")
        f.write("-" * 80 + "\n")
        for model_name in ['grammar_beta2e4', 'grammar_beta1e2', 'token_beta2e4', 'token_beta1e2']:
            canonicalized = results[model_name]['canonicalized']
            unique_count = len(set(canonicalized))
            unique_ratio = unique_count / len(canonicalized) if canonicalized else 0
            f.write(f"{model_name}: {unique_count} unique / {len(canonicalized)} valid ({100*unique_ratio:.2f}%)\n")
        
        f.write("\n3. SKELETON PATTERNS:\n")
        f.write("-" * 80 + "\n")
        for model_name in ['grammar_beta2e4', 'grammar_beta1e2', 'token_beta2e4', 'token_beta1e2']:
            f.write(f"\n{model_name}:\n")
            skeletons = results[model_name]['skeletons']
            deriv_combos = Counter([s['deriv_combo'] for s in skeletons])
            nonlinear_patterns = Counter()
            for s in skeletons:
                nonlinear_patterns.update(s['nonlinear'])
            
            novel_deriv_combos = set(deriv_combos.keys()) - training_deriv_combos
            novel_nonlinear = set(nonlinear_patterns.keys()) - training_nonlinear
            
            f.write(f"  Derivative combos: {len(deriv_combos)} unique\n")
            f.write(f"  Nonlinear patterns: {len(nonlinear_patterns)} unique\n")
            f.write(f"  Novel derivative combos: {len(novel_deriv_combos)}\n")
            if novel_deriv_combos:
                for combo in sorted(novel_deriv_combos)[:10]:
                    f.write(f"    {combo}\n")
            f.write(f"  Novel nonlinear: {len(novel_nonlinear)}\n")
            if novel_nonlinear:
                for nl in sorted(novel_nonlinear):
                    f.write(f"    {nl}\n")
    
    print(f"\nResults saved to: {results_file}")
    
    # Save actual PDE expressions for verification
    print("\nSaving generated PDE expressions...")
    for model_name in ['grammar_beta2e4', 'grammar_beta1e2', 'token_beta2e4', 'token_beta1e2']:
        pde_strings = results[model_name]['pde_strings']
        valid_pdes = results[model_name]['valid_pdes']
        
        # Save all PDEs
        all_pdes_file = output_dir / f"{model_name}_all_pdes_n{args.n_samples}_seed{args.seed}.txt"
        with open(all_pdes_file, 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Total generated: {args.n_samples}\n")
            f.write(f"Valid PDEs: {len(valid_pdes)}\n")
            f.write(f"Valid ratio: {100*len(valid_pdes)/args.n_samples:.2f}%\n")
            f.write("=" * 80 + "\n")
            f.write("\n")
            for i, pde_str in enumerate(pde_strings):
                is_valid = is_valid_pde(pde_str)
                f.write(f"{i+1:6d} [{'VALID' if is_valid else 'INVALID'}]: {pde_str}\n")
        
        # Save only valid PDEs
        valid_pdes_file = output_dir / f"{model_name}_valid_pdes_n{args.n_samples}_seed{args.seed}.txt"
        with open(valid_pdes_file, 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Valid PDEs: {len(valid_pdes)}/{args.n_samples} ({100*len(valid_pdes)/args.n_samples:.2f}%)\n")
            f.write("=" * 80 + "\n")
            f.write("\n")
            for i, pde_str in enumerate(valid_pdes):
                f.write(f"{i+1:6d}: {pde_str}\n")
        
        print(f"  {model_name}:")
        print(f"    All PDEs: {all_pdes_file.name}")
        print(f"    Valid PDEs: {valid_pdes_file.name}")
    
    return results


if __name__ == '__main__':
    main()
