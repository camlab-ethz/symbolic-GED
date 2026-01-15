"""
Validate PDE Dataset

Validates that:
1. All PDEs are syntactically correct
2. All labels match the PDE content
3. No duplicates exist
4. Family distribution is balanced

Usage:
    python validate_dataset.py [dataset_path]
"""

import pandas as pd
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.pde_classifier import PDEClassifier


def validate_dataset(dataset_path: str) -> dict:
    """Validate a PDE dataset.
    
    Args:
        dataset_path: Path to CSV dataset
        
    Returns:
        Dictionary with validation results
    """
    df = pd.read_csv(dataset_path)
    results = {
        'file': dataset_path,
        'total_rows': len(df),
        'issues': [],
        'accuracy': {},
    }
    
    # Check for required columns
    required_cols = ['pde', 'family', 'dim', 'temporal_order', 'spatial_order', 'nonlinear']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        results['issues'].append(f"Missing columns: {missing_cols}")
        return results
    
    # Check for missing values
    for col in required_cols:
        missing = df[col].isna().sum()
        if missing > 0:
            results['issues'].append(f"Column '{col}' has {missing} missing values")
    
    # Check for duplicates
    n_unique = df['pde'].nunique()
    if n_unique < len(df):
        n_dups = len(df) - n_unique
        results['issues'].append(f"{n_dups} duplicate PDEs found")
    
    # Validate labels with classifier
    classifier = PDEClassifier()
    
    correct_counts = {
        'family': 0, 
        'temporal_order': 0, 
        'spatial_order': 0,
        'dimension': 0, 
        'linearity': 0
    }
    
    for idx, row in df.iterrows():
        labels = classifier.classify(row['pde'])
        
        if labels.family == row['family']:
            correct_counts['family'] += 1
        if labels.temporal_order == row['temporal_order']:
            correct_counts['temporal_order'] += 1
        if labels.spatial_order == row['spatial_order']:
            correct_counts['spatial_order'] += 1
        if labels.dimension == row['dim']:
            correct_counts['dimension'] += 1
        is_nonlinear = labels.linearity == 'nonlinear'
        if is_nonlinear == row['nonlinear']:
            correct_counts['linearity'] += 1
    
    for label, count in correct_counts.items():
        acc = 100 * count / len(df)
        results['accuracy'][label] = acc
        if acc < 100.0:
            results['issues'].append(f"{label} accuracy: {acc:.2f}% (expected 100%)")
    
    # Check family balance
    family_counts = df['family'].value_counts()
    min_count = family_counts.min()
    max_count = family_counts.max()
    if max_count > 2 * min_count:
        results['issues'].append(f"Imbalanced families: {min_count} to {max_count}")
    
    results['family_counts'] = family_counts.to_dict()
    results['is_valid'] = len(results['issues']) == 0
    
    return results


def print_report(results: dict):
    """Print validation report."""
    print("=" * 70)
    print("DATASET VALIDATION REPORT")
    print("=" * 70)
    print(f"\nFile: {results['file']}")
    print(f"Total rows: {results['total_rows']}")
    
    print("\n" + "-" * 50)
    print("ACCURACY:")
    for label, acc in results.get('accuracy', {}).items():
        status = "✓" if acc == 100.0 else "✗"
        print(f"  {status} {label}: {acc:.2f}%")
    
    if results.get('issues'):
        print("\n" + "-" * 50)
        print("ISSUES FOUND:")
        for issue in results['issues']:
            print(f"  ✗ {issue}")
    
    print("\n" + "-" * 50)
    print("FAMILY DISTRIBUTION:")
    for family, count in sorted(results.get('family_counts', {}).items()):
        print(f"  {family:25s}: {count}")
    
    print("\n" + "=" * 70)
    if results.get('is_valid'):
        print("✓ DATASET IS VALID")
    else:
        print("✗ DATASET HAS ISSUES - see above")
    print("=" * 70)


def main():
    """Run validation on specified dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate PDE dataset')
    parser.add_argument('--dataset', type=str,
                       default='data/raw/pde_dataset_48000_fixed.csv',
                       help='Path to dataset CSV file')
    
    args = parser.parse_args()
    
    results = validate_dataset(args.dataset)
    print_report(results)
    
    return 0 if results['is_valid'] else 1


if __name__ == '__main__':
    sys.exit(main())
