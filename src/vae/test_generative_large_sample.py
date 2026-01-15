#!/usr/bin/env python3
"""Test generative abilities with large sample size (20k) to verify validity rates.

This script generates 20,000 samples from each model and checks validity rates
to ensure the results are not due to cherry-picking.
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


def main():
    parser = argparse.ArgumentParser(description='Test generative abilities with large sample size')
    parser.add_argument('--n_samples', type=int, default=20000, help='Number of samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--grammar_beta2e4', type=str, default=None)
    parser.add_argument('--grammar_beta1e2', type=str, default=None)
    parser.add_argument('--token_beta2e4', type=str, default=None)
    parser.add_argument('--token_beta1e2', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=500, help='Batch size for generation')
    
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
    print("LARGE SAMPLE GENERATIVE TEST (20k samples)")
    print("=" * 80)
    print(f"Number of samples: {args.n_samples}")
    print(f"Random seed: {args.seed}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    # Load models
    models = {}
    models['grammar_beta2e4'] = load_model(args.grammar_beta2e4, 'grammar', args.device)
    models['grammar_beta1e2'] = load_model(args.grammar_beta1e2, 'grammar', args.device)
    models['token_beta2e4'] = load_model(args.token_beta2e4, 'token', args.device)
    models['token_beta1e2'] = load_model(args.token_beta1e2, 'token', args.device)
    
    z_dim = models['grammar_beta2e4'].z_dim
    print(f"Latent dimension: {z_dim}\n")
    
    # Initialize tokenizer
    tokenizer = PDETokenizer()
    
    # Generate from each model
    results = {}
    print("Generating sequences from all models...\n")
    
    for model_name, model in models.items():
        print(f"Generating from {model_name}...")
        tokenization_type = model._tokenization_type
        
        valid_count = 0
        total_generated = 0
        
        # Generate in batches
        for batch_start in range(0, args.n_samples, args.batch_size):
            batch_end = min(batch_start + args.batch_size, args.n_samples)
            batch_size = batch_end - batch_start
            
            # Sample random latent vectors
            z = torch.randn(batch_size, z_dim, device=args.device)
            
            if tokenization_type == 'grammar':
                seq_ids = generate_from_grammar_vae(model, z, args.device)
            else:
                seq_ids = generate_from_token_vae(model, z, args.device)
            
            # Check validity
            for i in range(batch_size):
                if tokenization_type == 'grammar':
                    pde_str = prod_ids_to_string(seq_ids[i])
                else:
                    pde_str = token_ids_to_string(seq_ids[i], tokenizer)
                
                total_generated += 1
                if is_valid_pde(pde_str):
                    valid_count += 1
            
            if (batch_start + args.batch_size) % 5000 == 0 or batch_end == args.n_samples:
                print(f"  Progress: {batch_end}/{args.n_samples} samples, "
                      f"{valid_count}/{total_generated} valid ({100*valid_count/total_generated:.2f}%)")
        
        valid_ratio = valid_count / total_generated if total_generated > 0 else 0
        results[model_name] = {
            'valid_count': valid_count,
            'total_generated': total_generated,
            'valid_ratio': valid_ratio,
        }
        
        print(f"  Final: {valid_count}/{total_generated} valid ({100*valid_ratio:.2f}%)\n")
    
    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Model':<25} {'Valid':<12} {'Total':<12} {'Valid %':<12}")
    print("-" * 80)
    for model_name in ['grammar_beta2e4', 'grammar_beta1e2', 'token_beta2e4', 'token_beta1e2']:
        valid = results[model_name]['valid_count']
        total = results[model_name]['total_generated']
        ratio = results[model_name]['valid_ratio']
        print(f"{model_name:<25} {valid:<12} {total:<12} {100*ratio:>10.2f}%")
    print()
    
    # Save results
    output_dir = base_dir / "generation_results"
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / f"large_sample_test_n{args.n_samples}_seed{args.seed}.txt"
    with open(results_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("LARGE SAMPLE GENERATIVE TEST RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Number of samples: {args.n_samples}\n")
        f.write(f"Random seed: {args.seed}\n\n")
        f.write("RESULTS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Model':<25} {'Valid':<12} {'Total':<12} {'Valid %':<12}\n")
        f.write("-" * 80 + "\n")
        for model_name in ['grammar_beta2e4', 'grammar_beta1e2', 'token_beta2e4', 'token_beta1e2']:
            valid = results[model_name]['valid_count']
            total = results[model_name]['total_generated']
            ratio = results[model_name]['valid_ratio']
            f.write(f"{model_name:<25} {valid:<12} {total:<12} {100*ratio:>10.2f}%\n")
    
    print(f"Results saved to: {results_file}")
    
    return results


if __name__ == '__main__':
    main()
