"""
Create Train/Validation/Test Splits

Creates stratified train/val/test splits (typically 70/15/15 or 64/16/20).
Ensures balanced distribution across PDE families.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def create_splits(
    dataset_csv: str = "data/raw/pde_dataset_48000_fixed.csv",
    output_dir: str = "data/splits_48000_fixed",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    stratify_by: str = "family",
):
    """
    Create stratified train/validation/test splits.

    Args:
        dataset_csv: Path to dataset CSV file
        output_dir: Directory to save split indices
        train_ratio: Fraction for training set (default: 0.70)
        val_ratio: Fraction for validation set (default: 0.15)
        test_ratio: Fraction for test set (default: 0.15)
        random_state: Random seed for reproducibility
        stratify_by: Column name to stratify by (default: 'family')

    Returns:
        Dictionary with 'train', 'val', 'test' index arrays
    """
    # Verify ratios sum to 1.0
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")

    # Load dataset
    print(f"Loading dataset: {dataset_csv}")
    df = pd.read_csv(dataset_csv)
    n_total = len(df)
    print(f"Total samples: {n_total}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get stratification labels
    stratify_labels = df[stratify_by].values if stratify_by in df.columns else None

    # First split: train vs (val + test)
    test_size_1 = val_ratio + test_ratio  # Combined val+test size
    train_idx, val_test_idx = train_test_split(
        np.arange(n_total),
        test_size=test_size_1,
        random_state=random_state,
        stratify=stratify_labels,
    )

    # Second split: val vs test (from val_test subset)
    # Adjust test_size to get desired val_ratio and test_ratio
    test_size_2 = test_ratio / test_size_1  # Fraction of val_test that should be test
    val_idx, test_idx = train_test_split(
        val_test_idx,
        test_size=test_size_2,
        random_state=random_state,
        stratify=stratify_labels[val_test_idx] if stratify_labels is not None else None,
    )

    # Save indices
    np.save(output_path / "train_indices.npy", train_idx)
    np.save(output_path / "val_indices.npy", val_idx)
    np.save(output_path / "test_indices.npy", test_idx)

    # Print statistics
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_idx):6d} ({100*len(train_idx)/n_total:5.1f}%)")
    print(f"  Val:   {len(val_idx):6d} ({100*len(val_idx)/n_total:5.1f}%)")
    print(f"  Test:  {len(test_idx):6d} ({100*len(test_idx)/n_total:5.1f}%)")

    # Verify stratification
    if stratify_by in df.columns:
        print(f"\nFamily distribution:")
        print(f"{'Family':<20s} {'Train':>7s} {'Val':>7s} {'Test':>7s} {'Total':>7s}")
        print("-" * 50)
        for fam in sorted(df[stratify_by].unique()):
            train_count = np.sum(df.iloc[train_idx][stratify_by] == fam)
            val_count = np.sum(df.iloc[val_idx][stratify_by] == fam)
            test_count = np.sum(df.iloc[test_idx][stratify_by] == fam)
            total_count = train_count + val_count + test_count
            print(
                f"{fam:<20s} {train_count:7d} {val_count:7d} {test_count:7d} {total_count:7d}"
            )

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/raw/pde_dataset_48000_fixed.csv",
        help="Path to dataset CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/splits_48000_fixed",
        help="Output directory for split indices",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Training set ratio (default: 0.70)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.15, help="Test set ratio (default: 0.15)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    create_splits(
        dataset_csv=args.dataset,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.seed,
    )

    print(f"\n✅ Splits saved to: {args.output}/")
    print(f"  - train_indices.npy")
    print(f"  - val_indices.npy")
    print(f"  - test_indices.npy")


if __name__ == "__main__":
    main()
