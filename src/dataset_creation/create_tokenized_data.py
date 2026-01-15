"""
Create Tokenized Data Files

Tokenizes the full dataset using both Grammar and Token tokenization methods.
Saves tokenized data organized by split (train/val/test).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pde.grammar import (
    parse_to_productions,
    PROD_COUNT,
    build_masks_from_production_sequence,
)
from pde.chr_tokenizer import PDETokenizer


def tokenize_dataset(
    dataset_csv: str = "data/raw/pde_dataset_48000_fixed.csv",
    splits_dir: str = "data/splits_48000_fixed",
    output_dir: str = "data/tokenized_48000_fixed",
    grammar_max_len: int = 114,
    token_max_len: int = 62,
):
    """
    Tokenize full dataset for both Grammar and Token VAEs.

    Args:
        dataset_csv: Path to dataset CSV
        splits_dir: Directory containing split indices
        output_dir: Output directory for tokenized data
        grammar_max_len: Max sequence length for Grammar VAE
        token_max_len: Max sequence length for Token VAE
    """
    # Load dataset
    print(f"Loading dataset: {dataset_csv}")
    df = pd.read_csv(dataset_csv)
    n_total = len(df)
    print(f"Total samples: {n_total}")

    # Load splits
    splits_path = Path(splits_dir)
    train_idx = np.load(splits_path / "train_indices.npy")
    val_idx = np.load(splits_path / "val_indices.npy")
    test_idx = np.load(splits_path / "test_indices.npy")

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_idx)}")
    print(f"  Val:   {len(val_idx)}")
    print(f"  Test:  {len(test_idx)}")

    # Initialize tokenizer
    print("\nInitializing Token tokenizer...")
    tokenizer = PDETokenizer()
    print(f"  Vocabulary size: {tokenizer.vocab.vocab_size}")

    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Tokenize Grammar VAE data (all data, then split)
    print("\n" + "=" * 80)
    print("Tokenizing for Grammar VAE...")
    print("=" * 80)
    grammar_data = np.full((n_total, grammar_max_len), -1, dtype=np.int16)

    failed_grammar = []
    for i, pde in enumerate(df["pde"]):
        try:
            prod_ids = parse_to_productions(pde)
            if len(prod_ids) > grammar_max_len:
                print(
                    f"Warning: PDE {i} exceeds max length ({len(prod_ids)} > {grammar_max_len})"
                )
                prod_ids = prod_ids[:grammar_max_len]
            grammar_data[i, : len(prod_ids)] = prod_ids[:grammar_max_len]
        except Exception as e:
            failed_grammar.append((i, pde, str(e)))
            if len(failed_grammar) <= 5:
                print(f"Failed to tokenize Grammar {i}: {e}")

    if failed_grammar:
        print(f"\n⚠️  Failed to tokenize {len(failed_grammar)} PDEs for Grammar VAE")

    # Tokenize Token VAE data (all data, then split)
    print("\n" + "=" * 80)
    print("Tokenizing for Token VAE...")
    print("=" * 80)
    token_data = np.zeros((n_total, token_max_len), dtype=np.int16)

    failed_token = []
    for i, pde in enumerate(df["pde"]):
        try:
            ids = tokenizer.encode(pde, add_special_tokens=True)
            if len(ids) > token_max_len:
                print(
                    f"Warning: PDE {i} exceeds max length ({len(ids)} > {token_max_len})"
                )
                ids = ids[:token_max_len]
            token_data[i, : len(ids)] = ids[:token_max_len]
        except Exception as e:
            failed_token.append((i, pde, str(e)))
            if len(failed_token) <= 5:
                print(f"Failed to tokenize Token {i}: {e}")

    if failed_token:
        print(f"\n⚠️  Failed to tokenize {len(failed_token)} PDEs for Token VAE")

    # Save by split
    print("\n" + "=" * 80)
    print("Saving tokenized data by split...")
    print("=" * 80)

    # Grammar VAE
    grammar_dir = output_path / "grammar"
    grammar_dir.mkdir(exist_ok=True)
    np.save(grammar_dir / "train.npy", grammar_data[train_idx])
    np.save(grammar_dir / "val.npy", grammar_data[val_idx])
    np.save(grammar_dir / "test.npy", grammar_data[test_idx])
    print(f"\n✓ Grammar VAE data saved to {grammar_dir}/")
    print(f"  - train.npy: shape {grammar_data[train_idx].shape}")
    print(f"  - val.npy:   shape {grammar_data[val_idx].shape}")
    print(f"  - test.npy:  shape {grammar_data[test_idx].shape}")

    # Token VAE
    token_dir = output_path / "token"
    token_dir.mkdir(exist_ok=True)
    np.save(token_dir / "train.npy", token_data[train_idx])
    np.save(token_dir / "val.npy", token_data[val_idx])
    np.save(token_dir / "test.npy", token_data[test_idx])
    print(f"\n✓ Token VAE data saved to {token_dir}/")
    print(f"  - train.npy: shape {token_data[train_idx].shape}")
    print(f"  - val.npy:   shape {token_data[val_idx].shape}")
    print(f"  - test.npy:  shape {token_data[test_idx].shape}")

    # Also save full arrays (for datamodule compatibility - expects single file + split indices)
    np.save(output_path / "grammar_full.npy", grammar_data)
    np.save(output_path / "token_full.npy", token_data)

    # Create masks for Grammar VAE (proper masks from left-most derivation for ALL sequences)
    from pde.grammar import build_masks_from_production_sequence

    print("\nCreating Grammar masks for ALL sequences...")
    grammar_masks = np.zeros(
        (n_total, grammar_max_len, PROD_COUNT), dtype=bool
    )  # Use PROD_COUNT dynamically, start with all invalid
    failed_count = 0
    success_count = 0

    # Use the already-computed grammar_data instead of re-parsing from PDE strings
    # This is faster and more reliable (grammar_data was already computed and saved)
    print(f"  Using existing production sequences from grammar_data...")
    print(f"  Processing {n_total} sequences...")

    for i in range(n_total):
        if (i + 1) % 5000 == 0:
            print(f"    Progress: {i+1}/{n_total} ({100*(i+1)/n_total:.1f}%)")

        success = False
        # First try: Use existing production sequences from grammar_data
        try:
            # Get production IDs, filter out padding (-1)
            prod_ids = grammar_data[i].tolist()
            prod_ids_clean = [p for p in prod_ids if p >= 0]

            if len(prod_ids_clean) > 0:
                masks = build_masks_from_production_sequence(prod_ids_clean)
                for t, mask in enumerate(masks[:grammar_max_len]):
                    if t < grammar_max_len and len(mask) > 0:
                        grammar_masks[i, t, : min(len(mask), PROD_COUNT)] = mask[:PROD_COUNT]
                success = True
                success_count += 1
        except Exception as e:
            # Will try fallback
            pass

        # Fallback: parse from PDE strings if first method failed
        if not success:
            try:
                pde = df.iloc[i]["pde"]
                prod_ids = parse_to_productions(pde)
                prod_ids_clean = [p for p in prod_ids if p >= 0 and p < PROD_COUNT]  # Filter valid
                if len(prod_ids_clean) > 0 and len(prod_ids_clean) <= grammar_max_len:
                    masks = build_masks_from_production_sequence(prod_ids_clean)
                    for t, mask in enumerate(masks[:grammar_max_len]):
                        if t < grammar_max_len and len(mask) > 0:
                            grammar_masks[i, t, : min(len(mask), PROD_COUNT)] = mask[:PROD_COUNT]
                    success = True
                    success_count += 1
            except Exception as e:
                # Both methods failed
                pass

        if not success:
            failed_count += 1
            if failed_count <= 5:  # Show first few errors
                print(f"    Warning: Failed to generate mask for sequence {i}")

    if failed_count > 0:
        print(f"  ⚠️  Warning: {failed_count} sequences failed to generate masks")
    print(f"  ✅ Successfully generated masks for {success_count} sequences")

    np.save(output_path / "grammar_full_masks.npy", grammar_masks)
    print(f"  ✓ Grammar masks saved: shape {grammar_masks.shape}")

    # Token masks (all-valid for token sequences)
    token_masks = np.ones((n_total, token_max_len), dtype=bool)
    for i in range(n_total):
        valid_len = np.sum(token_data[i] != 0)
        if valid_len < token_max_len:
            token_masks[i, valid_len:] = False
    np.save(output_path / "token_full_masks.npy", token_masks)
    print(f"  ✓ Token masks saved: shape {token_masks.shape}")

    print(f"\n✅ Tokenization complete!")
    print(f"\nOutput directory structure:")
    print(f"  {output_dir}/")
    print(f"  ├── grammar/")
    print(f"  │   ├── train.npy")
    print(f"  │   ├── val.npy")
    print(f"  │   └── test.npy")
    print(f"  ├── token/")
    print(f"  │   ├── train.npy")
    print(f"  │   ├── val.npy")
    print(f"  │   └── test.npy")
    print(f"  ├── grammar_full.npy  (for compatibility)")
    print(f"  └── token_full.npy    (for compatibility)")


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Tokenize dataset for VAE training")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/raw/pde_dataset_48000_fixed.csv",
        help="Path to dataset CSV",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="data/splits_48000_fixed",
        help="Directory containing split indices",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/tokenized_48000_fixed",
        help="Output directory for tokenized data",
    )
    parser.add_argument(
        "--grammar-max-len",
        type=int,
        default=114,
        help="Max sequence length for Grammar VAE",
    )
    parser.add_argument(
        "--token-max-len",
        type=int,
        default=62,
        help="Max sequence length for Token VAE",
    )

    args = parser.parse_args()

    tokenize_dataset(
        dataset_csv=args.dataset,
        splits_dir=args.splits,
        output_dir=args.output,
        grammar_max_len=args.grammar_max_len,
        token_max_len=args.token_max_len,
    )


if __name__ == "__main__":
    main()
