#!/usr/bin/env python3
"""Create OOD (Out-of-Distribution) train/val/test splits.

This script creates split indices that exclude specified families from
training/validation, reserving them for true OOD testing.

Usage:
    python scripts/create_ood_splits.py --exclude-families kdv schrodinger
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Setup paths
SCRIPT_DIR = Path(__file__).parent
LIBGEN_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(LIBGEN_DIR))


def create_ood_splits(
    dataset_path: str,
    exclude_families: list,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """Create OOD train/val/test splits.
    
    Args:
        dataset_path: Path to the CSV dataset
        exclude_families: List of families to exclude (OOD families)
        output_dir: Directory to save split indices
        train_ratio: Ratio for training set (from IID data)
        val_ratio: Ratio for validation set (from IID data)
        test_ratio: Ratio for IID test set
        seed: Random seed
    """
    print("=" * 60)
    print("CREATING OOD SPLITS")
    print("=" * 60)
    
    # Load dataset
    print(f"\n1. Loading dataset from {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"   Total samples: {len(df)}")
    print(f"   Families: {df['family'].unique().tolist()}")
    
    # Create masks
    is_ood = df['family'].isin(exclude_families)
    is_iid = ~is_ood
    
    iid_indices = np.where(is_iid)[0]
    ood_indices = np.where(is_ood)[0]
    
    print(f"\n2. Splitting data")
    print(f"   IID samples: {len(iid_indices)}")
    print(f"   OOD samples ({exclude_families}): {len(ood_indices)}")
    
    # Split IID data into train/val/test
    # First split: train vs (val+test)
    train_idx, val_test_idx = train_test_split(
        iid_indices,
        train_size=train_ratio,
        random_state=seed,
        stratify=df.loc[iid_indices, 'family']
    )
    
    # Second split: val vs test
    val_size_adjusted = val_ratio / (val_ratio + test_ratio)
    val_idx, test_iid_idx = train_test_split(
        val_test_idx,
        train_size=val_size_adjusted,
        random_state=seed,
        stratify=df.loc[val_test_idx, 'family']
    )
    
    print(f"\n3. Split sizes:")
    print(f"   Train (IID only): {len(train_idx)}")
    print(f"   Val (IID only):   {len(val_idx)}")
    print(f"   Test (IID):       {len(test_iid_idx)}")
    print(f"   Test (OOD):       {len(ood_indices)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save splits
    np.save(os.path.join(output_dir, 'train_indices.npy'), train_idx)
    np.save(os.path.join(output_dir, 'val_indices.npy'), val_idx)
    np.save(os.path.join(output_dir, 'test_iid_indices.npy'), test_iid_idx)
    np.save(os.path.join(output_dir, 'test_ood_indices.npy'), ood_indices)
    
    # Also save a combined test set (for compatibility)
    np.save(os.path.join(output_dir, 'test_indices.npy'), test_iid_idx)
    
    # Save metadata
    metadata = {
        'exclude_families': exclude_families,
        'seed': seed,
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'n_train': len(train_idx),
        'n_val': len(val_idx),
        'n_test_iid': len(test_iid_idx),
        'n_test_ood': len(ood_indices),
        'iid_families': df.loc[iid_indices, 'family'].unique().tolist(),
        'ood_families': df.loc[ood_indices, 'family'].unique().tolist(),
    }
    
    import json
    with open(os.path.join(output_dir, 'split_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n4. Saved to {output_dir}:")
    print(f"   - train_indices.npy")
    print(f"   - val_indices.npy")
    print(f"   - test_iid_indices.npy")
    print(f"   - test_ood_indices.npy")
    print(f"   - split_metadata.json")
    
    # Print family distribution
    print(f"\n5. Family distribution:")
    print(f"\n   Training set:")
    train_families = df.loc[train_idx, 'family'].value_counts()
    for fam, count in train_families.items():
        print(f"      {fam}: {count}")
    
    print(f"\n   OOD test set:")
    ood_families = df.loc[ood_indices, 'family'].value_counts()
    for fam, count in ood_families.items():
        print(f"      {fam}: {count}")
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description='Create OOD splits')
    parser.add_argument('--dataset', type=str, 
                        default=str(LIBGEN_DIR / 'pde_dataset_48444_clean.csv'),
                        help='Path to dataset CSV')
    parser.add_argument('--exclude-families', nargs='+', 
                        default=['kdv', 'reaction_diffusion_cubic'],
                        help='Families to exclude for OOD testing')
    parser.add_argument('--output-dir', type=str,
                        default=str(LIBGEN_DIR / 'splits/ood_kdv_reaction_diffusion_cubic'),
                        help='Output directory for split indices')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    create_ood_splits(
        dataset_path=args.dataset,
        exclude_families=args.exclude_families,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
