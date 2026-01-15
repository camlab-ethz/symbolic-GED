#!/usr/bin/env python3
"""
Helper script: Split telegrapher bridge dataset into train/test for VAE training

This script reads a telegrapher bridge CSV and creates separate files for:
- Training: endpoints only (diffusion-like + wave-like)
- Testing: middle tau values (unseen continuation region)
"""

import csv
import sys


def split_telegrapher_dataset(input_csv: str, output_train: str, output_test: str):
    """
    Split telegrapher bridge dataset by the 'split' column
    
    Args:
        input_csv: Path to telegrapher bridge CSV
        output_train: Path to output training CSV
        output_test: Path to output testing CSV
    """
    train_rows = []
    test_rows = []
    
    with open(input_csv, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            if row['split'] == 'train_endpoints':
                train_rows.append(row)
            elif row['split'] == 'test_middle':
                test_rows.append(row)
    
    # Write training set
    with open(output_train, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(train_rows)
    
    # Write test set
    with open(output_test, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(test_rows)
    
    print(f"Split complete:")
    print(f"  Training (endpoints):  {len(train_rows)} PDEs → {output_train}")
    print(f"  Testing (middle):      {len(test_rows)} PDEs → {output_test}")
    
    # Print tau ranges
    train_taus = sorted([float(row['tau']) for row in train_rows])
    test_taus = sorted([float(row['tau']) for row in test_rows])
    
    print(f"\nTau ranges:")
    print(f"  Train: {train_taus}")
    print(f"  Test:  {test_taus}")
    
    return train_rows, test_rows


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 split_telegrapher_data.py <input_csv> [output_train] [output_test]")
        print("\nExample:")
        print("  python3 split_telegrapher_data.py telegrapher_bridge.csv train.csv test.csv")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_train = sys.argv[2] if len(sys.argv) > 2 else 'telegrapher_train_endpoints.csv'
    output_test = sys.argv[3] if len(sys.argv) > 3 else 'telegrapher_test_middle.csv'
    
    split_telegrapher_dataset(input_csv, output_train, output_test)
    
    print("\n" + "=" * 80)
    print("NEXT: Train VAE on endpoints only")
    print("=" * 80)
    print(f"""
1. Encode training data:
   python3 ../pde_grammar.py {output_train}  # Grammar method
   python3 ../pde_tokenizer.py {output_train}  # Token method

2. Train VAE using only endpoint data:
   python3 ../../model/train.py --config ../config_train_grammar.yaml \\
       --dataset {output_train}

3. Evaluate on test (middle tau) to measure continuation performance:
   python3 ../../vae/decode_test_set.py \\
       --dataset {output_test} \\
       --checkpoint checkpoints/best.ckpt
    """)
