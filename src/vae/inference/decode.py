"""Decode test set predictions from both Grammar and Token VAEs and compare to ground truth.

This script:
1. Loads both trained models
2. Runs inference on test set
3. Decodes grammar productions → infix PDEs
4. Decodes character tokens → infix PDEs
5. Compares predictions with ground truth
6. Shows examples of correct and incorrect reconstructions
"""
import argparse
import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import pytorch_lightning as pl
from tqdm import tqdm

# Setup path to import from parent vae module
import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from vae.module import GrammarVAEModule
from vae.utils import GrammarVAEDataModule, TokenVAEDataModule
from pde.grammar import decode_production_sequence
from pde.chr_tokenizer import PDETokenizer


def load_ground_truth_pdEs(csv_path, test_indices):
    """Load ground truth PDEs from CSV for test indices."""
    import csv
    
    print(f"Loading ground truth from {csv_path}")
    pdEs = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        pdEs = [row['pde'] for row in reader]
    
    test_pdes = [pdEs[i] for i in test_indices]
    print(f"Loaded {len(test_pdes)} ground truth test PDEs")
    return test_pdes


def decode_grammar_sequence(prod_ids, grammar_decoder=None):
    """Decode grammar production sequence to infix PDE string.
    
    For now, returns the production sequence as a string.
    TODO: Implement actual tree grammar decoding.
    """
    # Use the canonical grammar decoder in pde_grammar
    try:
        decoded = decode_production_sequence(prod_ids.tolist() if hasattr(prod_ids, 'tolist') else list(prod_ids))
    except Exception:
        # Fall back to safe stringification
        decoded = f"<invalid_grammar_seq: {prod_ids[:10]}...>"

    # Pretty-format: add spaces around + and - for readability and append ' = 0'
    pretty = decoded.replace('+', ' + ').replace('-', ' - ').replace('*', '*')
    # Avoid double spaces
    pretty = ' '.join(pretty.split())
    if pretty and not pretty.endswith('= 0'):
        pretty = pretty + ' = 0'
    return pretty


def decode_token_sequence(token_ids, char_vocab=None):
    """Decode token IDs to infix PDE string.
    
    Lample & Charton tokenization: direct character mapping.
    """
    # Use the project's PDE tokenizer to decode token ID sequences back to infix
    try:
        tokenizer = PDETokenizer()
        # token_ids may be numpy array; convert to list
        ids = token_ids.tolist() if hasattr(token_ids, 'tolist') else list(token_ids)
        # Use decode_to_infix which expects token IDs mapped to vocabulary used in this repo
        pretty = tokenizer.decode_to_infix(ids, skip_special_tokens=True)
        # Ensure end marker ' = 0' for readability
        if pretty and not pretty.endswith('= 0'):
            pretty = pretty + ' = 0'
        return pretty
    except Exception:
        # Fallback to simple placeholder rendering
        return ''.join([f'<{int(t)}>' for t in token_ids if int(t) >= 0])


def predict_and_decode_grammar(model, datamodule, test_indices, device='cuda'):
    """Run Grammar VAE on test set and decode predictions."""
    model.eval()
    model.to(device)
    
    print("\n" + "="*80)
    print("GRAMMAR VAE: Decoding test set predictions")
    print("="*80)
    
    test_loader = datamodule.test_dataloader()
    
    all_predictions = []
    all_targets = []
    all_valid = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Grammar VAE inference")):
            x, targets, masks = batch
            x = x.to(device)
            
            # Forward pass
            logits, mu, logvar = model(x)
            
            # Apply grammar masks (same as in training/test)
            valid_mask = (masks.sum(dim=-1) > 0).float().to(device)
            logits_masked = logits.clone()
            non_padding = valid_mask.unsqueeze(-1) > 0
            invalid_mask = (masks.to(device) == 0) & non_padding
            logits_masked[invalid_mask] = float('-inf')
            
            # Get predictions
            preds = logits_masked.argmax(dim=-1)  # (B, T)

            # valid positions: timestep is valid if masks.sum(dim=-1) > 0
            valid_pos = (masks.sum(dim=-1) > 0).cpu().numpy()

            all_predictions.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())
            all_valid.append(valid_pos)
    
    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    # valid masks (per timestep) also concatenate
    all_valid = np.concatenate(all_valid, axis=0)

    print(f"Grammar predictions shape: {all_predictions.shape}")
    print(f"Grammar targets shape: {all_targets.shape}")
    print(f"Grammar valid-mask shape: {all_valid.shape}")
    
    # Decode to strings (placeholder for now)
    decoded_preds = []
    decoded_targets = []
    
    for i in range(len(all_predictions)):
        # Get non-padding tokens
        pred_seq = all_predictions[i]
        target_seq = all_targets[i]
        valid_seq = all_valid[i]

        # Decode (use canonical decoder)
        decoded_preds.append(decode_production_sequence(pred_seq.tolist() if hasattr(pred_seq, 'tolist') else list(pred_seq)))
        decoded_targets.append(decode_production_sequence(target_seq.tolist() if hasattr(target_seq, 'tolist') else list(target_seq)))
        decoded_preds[-1] = decoded_preds[-1].strip()
        decoded_targets[-1] = decoded_targets[-1].strip()
    
    return decoded_preds, decoded_targets, all_predictions, all_targets, all_valid


def predict_and_decode_token(model, datamodule, test_indices, device='cuda'):
    """Run Token VAE on test set and decode predictions."""
    model.eval()
    model.to(device)
    
    print("\n" + "="*80)
    print("TOKEN VAE: Decoding test set predictions")
    print("="*80)
    
    test_loader = datamodule.test_dataloader()
    
    all_predictions = []
    all_targets = []
    all_valid = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Token VAE inference")):
            x, targets, masks = batch
            x = x.to(device)
            
            # Forward pass
            logits, mu, logvar = model(x)
            
            # Apply masks
            valid_mask = (masks.sum(dim=-1) > 0).float().to(device)
            logits_masked = logits.clone()
            non_padding = valid_mask.unsqueeze(-1) > 0
            invalid_mask = (masks.to(device) == 0) & non_padding
            logits_masked[invalid_mask] = float('-inf')
            
            # Get predictions
            preds = logits_masked.argmax(dim=-1)  # (B, T)

            # Determine valid positions for this batch
            valid_pos = (masks.sum(dim=-1) > 0).cpu().numpy()

            all_predictions.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())
            all_valid.append(valid_pos)
    
    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    # valid masks (per timestep) also concatenate
    all_valid = np.concatenate(all_valid, axis=0)

    print(f"Token predictions shape: {all_predictions.shape}")
    print(f"Token targets shape: {all_targets.shape}")
    print(f"Token valid-mask shape: {all_valid.shape}")
    
    # Decode to strings
    decoded_preds = []
    decoded_targets = []
    
    for i in range(len(all_predictions)):
        pred_seq = all_predictions[i]
        target_seq = all_targets[i]
        valid_seq = all_valid[i]

        decoded_pred = decode_token_sequence(pred_seq)
        decoded_tgt = decode_token_sequence(target_seq)

        decoded_preds.append(decoded_pred.strip())
        decoded_targets.append(decoded_tgt.strip())
    
    return decoded_preds, decoded_targets, all_predictions, all_targets, all_valid


def main():
    parser = argparse.ArgumentParser(description='Decode test set predictions from both VAEs')
    parser.add_argument('--grammar_checkpoint', type=str, 
                       default='checkpoints/grammar_vae/best-epoch=189-seqacc=val/seq_acc=0.9907.ckpt')
    parser.add_argument('--token_checkpoint', type=str,
                       default='checkpoints/token_vae/best-epoch=314-seqacc=val/seq_acc=0.9841.ckpt')
    parser.add_argument('--csv_path', type=str,
                       default='../library_new_approach/datasets/pde_dataset.csv')
    parser.add_argument('--num_examples', type=int, default=20,
                       help='Number of examples to print')
    parser.add_argument('--output_file', type=str, default='decoded_test_comparisons.txt',
                       help='Output file for all comparisons')
    
    args = parser.parse_args()
    
    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config_vae.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load test indices
    split_dir = 'data_splits'
    test_indices = np.load(os.path.join(split_dir, 'test_indices.npy'))
    print(f"\nTest set: {len(test_indices)} samples")
    
    # Load ground truth PDEs
    ground_truth = load_ground_truth_pdEs(args.csv_path, test_indices)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # ========== GRAMMAR VAE ==========
    print("\n" + "="*80)
    print("Loading Grammar VAE")
    print("="*80)
    
    grammar_config = config['grammar']
    grammar_dm = GrammarVAEDataModule(
        prod_path=grammar_config['data']['prod_path'],
        masks_path=grammar_config['data']['masks_path'],
        split_dir=grammar_config['data']['split_dir'],
        batch_size=256,
        num_workers=4
    )
    grammar_dm.setup('test')
    
    grammar_model = GrammarVAEModule.load_from_checkpoint(args.grammar_checkpoint)
    
    grammar_preds, grammar_targets_decoded, grammar_pred_ids, grammar_target_ids, grammar_valid = \
        predict_and_decode_grammar(grammar_model, grammar_dm, test_indices, device)
    
    # ========== TOKEN VAE ==========
    print("\n" + "="*80)
    print("Loading Token VAE")
    print("="*80)
    
    token_config = config['token']
    token_dm = TokenVAEDataModule(
        token_path=token_config['data']['token_path'],
        masks_path=token_config['data']['masks_path'],
        split_dir=token_config['data']['split_dir'],
        vocab_size=token_config['model']['vocab_size'],
        batch_size=256,
        num_workers=4
    )
    token_dm.setup('test')
    
    token_model = GrammarVAEModule.load_from_checkpoint(args.token_checkpoint)
    
    token_preds, token_targets_decoded, token_pred_ids, token_target_ids, token_valid = \
        predict_and_decode_token(token_model, token_dm, test_indices, device)
    
    # ========== COMPARE AND SAVE ==========
    print("\n" + "="*80)
    print("Comparing predictions with ground truth")
    print("="*80)
    
    # Calculate exact matches using valid-position masks
    # grammar_valid and token_valid are lists/arrays of booleans (N, T)
    grammar_pred_ids = np.asarray(grammar_pred_ids)
    grammar_target_ids = np.asarray(grammar_target_ids)
    grammar_mask = np.asarray(grammar_valid)

    token_pred_ids = np.asarray(token_pred_ids)
    token_target_ids = np.asarray(token_target_ids)
    token_mask = np.asarray(token_valid)

    # ID-level masked equality: positions outside mask are ignored
    grammar_eq = (grammar_pred_ids == grammar_target_ids)
    grammar_correct = np.all(grammar_eq | (~grammar_mask), axis=1)

    token_eq = (token_pred_ids == token_target_ids)
    token_correct = np.all(token_eq | (~token_mask), axis=1)
    
    grammar_acc = grammar_correct.mean()
    token_acc = token_correct.mean()
    
    print(f"\nGrammar VAE sequence accuracy: {grammar_acc:.4f} ({grammar_correct.sum()}/{len(grammar_correct)})")
    print(f"Token VAE sequence accuracy: {token_acc:.4f} ({token_correct.sum()}/{len(token_correct)})")
    
    # Save detailed comparison
    with open(args.output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TEST SET DECODING COMPARISON\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total test samples: {len(test_indices)}\n")
        f.write(f"Grammar VAE accuracy: {grammar_acc:.4f}\n")
        f.write(f"Token VAE accuracy: {token_acc:.4f}\n")
        f.write("="*80 + "\n\n")
        
        def normalize_infix(s: str) -> str:
            # Remove trailing ' = 0' and all whitespace for canonical textual comparison
            if s is None:
                return ''
            s2 = s.strip()
            if s2.endswith('= 0'):
                s2 = s2[:-3]
            # Remove spaces and normalize multiple spaces
            s2 = ''.join(s2.split())
            return s2

        for i in range(len(test_indices)):
            f.write(f"\n{'='*80}\n")
            f.write(f"Example {i+1} (Index {test_indices[i]})\n")
            f.write(f"{'='*80}\n")
            f.write(f"GROUND TRUTH:      {ground_truth[i]}\n")
            f.write(f"GRAMMAR PREDICTED: {grammar_preds[i]}\n")
            # string-level check
            gt_norm = normalize_infix(ground_truth[i])
            gpred_norm = normalize_infix(grammar_preds[i])
            grammar_string_correct = (gt_norm == gpred_norm)

            f.write(f"GRAMMAR CORRECT (ids):   {'✓' if grammar_correct[i] else '✗'}\n")
            f.write(f"GRAMMAR CORRECT (str):   {'✓' if grammar_string_correct else '✗'}\n")
            f.write(f"TOKEN PREDICTED:   {token_preds[i]}\n")
            tpred_norm = normalize_infix(token_preds[i])
            token_string_correct = (gt_norm == tpred_norm)

            f.write(f"TOKEN CORRECT (ids):     {'✓' if token_correct[i] else '✗'}\n")
            f.write(f"TOKEN CORRECT (str):     {'✓' if token_string_correct else '✗'}\n")
    
    print(f"\nFull comparison saved to: {args.output_file}")
    
    # Print some examples
    print("\n" + "="*80)
    print(f"FIRST {args.num_examples} EXAMPLES")
    print("="*80)
    
    for i in range(min(args.num_examples, len(test_indices))):
        print(f"\n--- Example {i+1} (Test Index {test_indices[i]}) ---")
        print(f"Ground Truth:  {ground_truth[i]}")
        print(f"Grammar Pred:  {grammar_preds[i]} {'✓' if grammar_correct[i] else '✗'}")
        print(f"Token Pred:    {token_preds[i]} {'✓' if token_correct[i] else '✗'}")
    
    # Show some failures
    grammar_failures = np.where(~grammar_correct)[0]
    token_failures = np.where(~token_correct)[0]
    
    if len(grammar_failures) > 0:
        print("\n" + "="*80)
        print(f"GRAMMAR VAE FAILURES (first {min(5, len(grammar_failures))})")
        print("="*80)
        for idx in grammar_failures[:5]:
            print(f"\n--- Example {idx+1} (Test Index {test_indices[idx]}) ---")
            print(f"Ground Truth:  {ground_truth[idx]}")
            print(f"Grammar Pred:  {grammar_preds[idx]}")
    
    if len(token_failures) > 0:
        print("\n" + "="*80)
        print(f"TOKEN VAE FAILURES (first {min(5, len(token_failures))})")
        print("="*80)
        for idx in token_failures[:5]:
            print(f"\n--- Example {idx+1} (Test Index {test_indices[idx]}) ---")
            print(f"Ground Truth:  {ground_truth[idx]}")
            print(f"Token Pred:    {token_preds[idx]}")
    
    print(f"\n{'='*80}")
    print("Decoding complete!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
