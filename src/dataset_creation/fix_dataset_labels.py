"""
Fix known issues in pde_dataset_48444.csv

Issues fixed:
1. Cahn-Hilliard marked as nonlinear=True but actual PDEs are linear
   (The nonlinear terms dxx(u^3) were not included in generation)

This script creates a corrected version of the dataset.
"""

import pandas as pd
from pathlib import Path

from pde.normalize import normalize_pde_string


def fix_dataset(input_path: str, output_path: str) -> dict:
    """Fix known label issues in the dataset.
    
    Args:
        input_path: Path to original dataset
        output_path: Path for corrected dataset
        
    Returns:
        Dictionary with fix statistics
    """
    df = pd.read_csv(input_path)
    
    stats = {
        'total_rows': len(df),
        'fixes': {}
    }
    
    # Issue 1: Normalize PDE strings (operator-only, consistent power operator)
    # - strip '= 0' suffix if present
    # - normalize '**' → '^' (tokenizer/grammar compatibility)
    before = df["pde"].copy()
    df["pde"] = df["pde"].astype(str).apply(normalize_pde_string)
    n_changed = int((before != df["pde"]).sum())
    if n_changed > 0:
        stats["fixes"]["normalized_pde_strings"] = n_changed
        print(f"Normalized PDE strings for {n_changed} rows (strip '=0', '**'→'^')")
    
    # Issue 2: Cahn-Hilliard nonlinearity
    # The actual PDEs don't contain u^3 terms, so they're linear
    ch_mask = df['family'] == 'cahn_hilliard'
    ch_nonlinear_count = df.loc[ch_mask, 'nonlinear'].sum()
    
    if ch_nonlinear_count > 0:
        df.loc[ch_mask, 'nonlinear'] = False
        stats['fixes']['cahn_hilliard_nonlinear_to_linear'] = int(ch_nonlinear_count)
        print(f"Fixed {ch_nonlinear_count} Cahn-Hilliard PDEs: nonlinear=True → False")
    
    # Verify the fix
    # Check that all PDEs with u^3 or u^2 are marked nonlinear
    def has_nonlinear_term(pde: str) -> bool:
        indicators = [
            "u^2",
            "u^3",
            "u**2",
            "u**3",
            "u * dx",
            "u*dx",
            "u * dy",
            "u*dy",
            "u * dz",
            "u*dz",
            "(dx(u))^2",
        ]
        return any(ind in pde for ind in indicators)
    
    df['has_nl_term'] = df['pde'].apply(has_nonlinear_term)
    
    # Check for mismatches
    mismatch_nl = df[(df['has_nl_term'] == True) & (df['nonlinear'] == False)]
    mismatch_lin = df[(df['has_nl_term'] == False) & (df['nonlinear'] == True)]
    
    if len(mismatch_nl) > 0:
        print(f"\nWARNING: {len(mismatch_nl)} PDEs have nonlinear terms but labeled linear")
        print("Samples:")
        for _, row in mismatch_nl.head(3).iterrows():
            print(f"  {row['family']}: {row['pde'][:60]}...")
            
    if len(mismatch_lin) > 0:
        print(f"\nWARNING: {len(mismatch_lin)} PDEs labeled nonlinear but have no obvious nonlinear terms")
        print("Samples:")
        for _, row in mismatch_lin.head(3).iterrows():
            print(f"  {row['family']}: {row['pde'][:60]}...")
    
    # Drop helper column
    df = df.drop(columns=['has_nl_term'])
    
    # Save corrected dataset
    df.to_csv(output_path, index=False)
    print(f"\nSaved corrected dataset to: {output_path}")
    
    stats['corrected_rows'] = sum(v for v in stats['fixes'].values())
    
    return stats


def main():
    """Run the fixes."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix dataset labels')
    parser.add_argument('--input', type=str, 
                       default='data/raw/pde_dataset_48000.csv',
                       help='Input dataset CSV file')
    parser.add_argument('--output', type=str,
                       default='data/raw/pde_dataset_48000_fixed.csv',
                       help='Output dataset CSV file')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("FIXING DATASET LABELS")
    print("="*70)
    print(f"\nInput:  {args.input}")
    print(f"Output: {args.output}")
    print()
    
    stats = fix_dataset(args.input, args.output)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total rows: {stats['total_rows']}")
    print(f"Fixes applied: {stats['corrected_rows']}")
    for fix_name, count in stats['fixes'].items():
        print(f"  - {fix_name}: {count}")


if __name__ == '__main__':
    main()
