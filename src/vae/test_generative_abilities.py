#!/usr/bin/env python3
"""Test generative abilities of all 4 VAE models using the same random latent vectors.

This script:
1. Loads all 4 model checkpoints
2. Samples N random latent vectors (same for all models)
3. Generates sequences from each model
4. Converts sequences to PDE strings
5. Validates syntax
6. Reports statistics
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vae.module import VAEModule
from pde.grammar import decode_production_sequence
from pde import PDETokenizer
from analysis.physics import is_valid_pde


def load_model(checkpoint_path, tokenization_type, device='cuda'):
    """Load VAE model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    model = VAEModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    model = model.to(device)
    # Store tokenization type for later use
    model._tokenization_type = tokenization_type
    print(f"  Model loaded: tokenization={tokenization_type}, z_dim={model.z_dim}, P={model.P}")
    return model


def generate_from_grammar_vae(model, z, device='cuda'):
    """Generate sequences from grammar VAE using grammar-constrained decoding."""
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
        # Remove padding (typically -1 or 0, but check what pad_id is used)
        # For grammar VAE, pad_id might be 0 or the last production ID
        # Filter out invalid production IDs
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
        # Remove padding (typically -1 or 0, pad_id is 0)
        # Filter out padding tokens (keep only valid token IDs)
        vocab = tokenizer.vocab
        pad_id = vocab.pad_id
        eos_id = vocab.eos_id if hasattr(vocab, 'eos_id') else None
        
        valid_token_ids = []
        for tid in token_ids_list:
            if tid < 0:
                continue
            if tid == pad_id:
                continue
            if eos_id is not None and tid == eos_id:
                break  # Stop at EOS token
            if tid < len(vocab.id2word):
                valid_token_ids.append(tid)
        
        if not valid_token_ids:
            return "[INVALID: No valid tokens]"
        
        # Use decode_to_infix to get readable infix notation
        pde_str = tokenizer.decode_to_infix(valid_token_ids)
        return pde_str if pde_str else "[INVALID: Empty sequence]"
    except Exception as e:
        return f"[ERROR: {e}]"


def test_generative_abilities(
    n_samples=1000,
    seed=42,
    device='cuda',
    grammar_beta2e4_ckpt=None,
    grammar_beta1e2_ckpt=None,
    token_beta2e4_ckpt=None,
    token_beta1e2_ckpt=None,
):
    """Test generative abilities of all 4 models."""
    
    # Default checkpoint paths
    base_dir = Path(__file__).parent.parent
    if grammar_beta2e4_ckpt is None:
        grammar_beta2e4_ckpt = base_dir / "checkpoints/grammar_vae/beta_2e-4_seed_42/best-epoch=380-val/seq_acc=0.9978.ckpt"
    if grammar_beta1e2_ckpt is None:
        grammar_beta1e2_ckpt = base_dir / "checkpoints/grammar_vae/beta_0.01_seed_42/best-epoch=517-val/seq_acc=0.0755.ckpt"
    if token_beta2e4_ckpt is None:
        token_beta2e4_ckpt = base_dir / "checkpoints/token_vae/beta_2e-4_seed_42/best-epoch=433-val/seq_acc=0.9962.ckpt"
    if token_beta1e2_ckpt is None:
        token_beta1e2_ckpt = base_dir / "checkpoints/token_vae/beta_0.01_seed_42/best-epoch=136-val/seq_acc=0.0016.ckpt"
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print("=" * 80)
    print("Testing Generative Abilities of 4 VAE Models")
    print("=" * 80)
    print(f"Number of samples: {n_samples}")
    print(f"Random seed: {seed}")
    print(f"Device: {device}")
    print()
    
    # Load all models
    print("Loading models...")
    models = {}
    models['grammar_beta2e4'] = load_model(str(grammar_beta2e4_ckpt), 'grammar', device)
    models['grammar_beta1e2'] = load_model(str(grammar_beta1e2_ckpt), 'grammar', device)
    models['token_beta2e4'] = load_model(str(token_beta2e4_ckpt), 'token', device)
    models['token_beta1e2'] = load_model(str(token_beta1e2_ckpt), 'token', device)
    print()
    
    # Get z_dim (should be same for all, but check)
    z_dims = {name: model.z_dim for name, model in models.items()}
    z_dim = z_dims[list(z_dims.keys())[0]]
    print(f"Using latent dimension: {z_dim}")
    print()
    
    # Sample random latent vectors (same for all models)
    print(f"Sampling {n_samples} random latent vectors from N(0, I)...")
    z = torch.randn(n_samples, z_dim, device=device)
    print()
    
    # Initialize tokenizer for token VAE
    tokenizer = PDETokenizer()
    
    # Generate sequences from each model
    results = {}
    
    print("Generating sequences...")
    print()
    
    for model_name, model in models.items():
        print(f"Generating from {model_name}...")
        tokenization_type = model._tokenization_type
        if tokenization_type == 'grammar':
            seq_ids = generate_from_grammar_vae(model, z, device)
        else:
            seq_ids = generate_from_token_vae(model, z, device)
        
        # Convert to PDE strings
        pde_strings = []
        valid_count = 0
        
        for i in range(n_samples):
            if tokenization_type == 'grammar':
                pde_str = prod_ids_to_string(seq_ids[i])
            else:
                pde_str = token_ids_to_string(seq_ids[i], tokenizer)
            
            pde_strings.append(pde_str)
            
            # Check if valid PDE
            if is_valid_pde(pde_str):
                valid_count += 1
        
        results[model_name] = {
            'sequences': seq_ids,
            'pde_strings': pde_strings,
            'valid_count': valid_count,
            'valid_ratio': valid_count / n_samples,
        }
        
        print(f"  Valid PDEs: {valid_count}/{n_samples} ({100*valid_count/n_samples:.2f}%)")
        print()
    
    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Model':<25} {'Valid PDEs':<15} {'Valid %':<10}")
    print("-" * 50)
    summary_lines = []
    for model_name in ['grammar_beta2e4', 'grammar_beta1e2', 'token_beta2e4', 'token_beta1e2']:
        valid_count = results[model_name]['valid_count']
        valid_ratio = results[model_name]['valid_ratio']
        line = f"{model_name:<25} {valid_count:<15} {100*valid_ratio:>6.2f}%"
        print(line)
        summary_lines.append(line)
    print()
    
    # Save results to files
    output_dir = base_dir / "generation_results"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Saving results to {output_dir}...")
    
    # Save summary statistics
    summary_file = output_dir / f"generation_summary_n{n_samples}_seed{seed}.txt"
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GENERATIVE ABILITIES TEST RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Number of samples: {n_samples}\n")
        f.write(f"Random seed: {seed}\n")
        f.write("\n")
        f.write("SUMMARY\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Model':<25} {'Valid PDEs':<15} {'Valid %':<10}\n")
        f.write("-" * 50 + "\n")
        for line in summary_lines:
            f.write(line + "\n")
        f.write("\n")
    
    print(f"  Summary saved to: {summary_file}")
    
    # Save all generated PDE strings for each model
    for model_name in ['grammar_beta2e4', 'grammar_beta1e2', 'token_beta2e4', 'token_beta1e2']:
        pde_strings = results[model_name]['pde_strings']
        valid_count = results[model_name]['valid_count']
        
        # Save all PDEs
        all_pdes_file = output_dir / f"{model_name}_all_pdes_n{n_samples}_seed{seed}.txt"
        with open(all_pdes_file, 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Total generated: {n_samples}\n")
            f.write(f"Valid PDEs: {valid_count}\n")
            f.write(f"Valid ratio: {100*valid_count/n_samples:.2f}%\n")
            f.write("=" * 80 + "\n")
            f.write("\n")
            for i, pde_str in enumerate(pde_strings):
                is_valid = is_valid_pde(pde_str)
                f.write(f"{i+1:6d} [{'VALID' if is_valid else 'INVALID'}]: {pde_str}\n")
        
        # Save only valid PDEs
        valid_pdes_file = output_dir / f"{model_name}_valid_pdes_n{n_samples}_seed{seed}.txt"
        valid_pdes = [pde for pde in pde_strings if is_valid_pde(pde)]
        with open(valid_pdes_file, 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Valid PDEs: {len(valid_pdes)}/{n_samples} ({100*len(valid_pdes)/n_samples:.2f}%)\n")
            f.write("=" * 80 + "\n")
            f.write("\n")
            for i, pde_str in enumerate(valid_pdes):
                f.write(f"{i+1:6d}: {pde_str}\n")
        
        print(f"  {model_name}: all PDEs -> {all_pdes_file.name}")
        print(f"  {model_name}: valid PDEs -> {valid_pdes_file.name}")
    
    print()
    
    # Print some example PDEs
    print("=" * 80)
    print("EXAMPLE GENERATED PDEs (first 5 valid from each model)")
    print("=" * 80)
    print()
    
    for model_name in ['grammar_beta2e4', 'grammar_beta1e2', 'token_beta2e4', 'token_beta1e2']:
        print(f"{model_name}:")
        pde_strings = results[model_name]['pde_strings']
        valid_pdes = [pde for pde in pde_strings if is_valid_pde(pde)]
        
        for i, pde in enumerate(valid_pdes[:5]):
            print(f"  {i+1}. {pde}")
        if len(valid_pdes) == 0:
            print("  (No valid PDEs generated)")
        print()
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test generative abilities of VAE models')
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='Number of random latent vectors to sample')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--grammar_beta2e4', type=str, default=None,
                       help='Path to grammar VAE (beta=2e-4) checkpoint')
    parser.add_argument('--grammar_beta1e2', type=str, default=None,
                       help='Path to grammar VAE (beta=1e-2) checkpoint')
    parser.add_argument('--token_beta2e4', type=str, default=None,
                       help='Path to token VAE (beta=2e-4) checkpoint')
    parser.add_argument('--token_beta1e2', type=str, default=None,
                       help='Path to token VAE (beta=1e-2) checkpoint')
    
    args = parser.parse_args()
    
    test_generative_abilities(
        n_samples=args.n_samples,
        seed=args.seed,
        device=args.device,
        grammar_beta2e4_ckpt=args.grammar_beta2e4,
        grammar_beta1e2_ckpt=args.grammar_beta1e2,
        token_beta2e4_ckpt=args.token_beta2e4,
        token_beta1e2_ckpt=args.token_beta1e2,
    )
