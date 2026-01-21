#!/usr/bin/env python3
"""Complete pipeline script for canonical dataset generation and training preparation."""

import sys
import subprocess
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

def run_command(cmd, description):
    """Run a command and check result."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Running: {cmd}")
    print()
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error: {description}")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        return False
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    print(f"✅ {description} completed")
    return True

def main():
    print("="*80)
    print("CANONICAL GRAMMAR FULL PIPELINE")
    print("="*80)
    
    # Change to src directory
    os.chdir(BASE_DIR)
    
    # Step 1: Verify grammar
    print("\nStep 1: Verifying grammar unambiguity...")
    verify_cmd = """
python3 -c "
import sys
sys.path.insert(0, '.')
from pde.grammar import parse_to_productions, decode_production_sequence, PROD_COUNT
from pde.normalize import canonicalize_operator_str

# Test u^2
s = 'u^2'
cs = canonicalize_operator_str(s)
seq = parse_to_productions(cs)
dec = decode_production_sequence(seq)
dec_cs = canonicalize_operator_str(dec)

if dec_cs == cs:
    print(f'✓ Grammar unambiguous (PROD_COUNT={PROD_COUNT})')
    sys.exit(0)
else:
    print('✗ Grammar issue')
    sys.exit(1)
"
"""
    if not run_command(verify_cmd, "Grammar verification"):
        print("\n❌ Grammar verification failed! Aborting.")
        sys.exit(1)
    
    # Step 2: Generate dataset
    print("\nStep 2: Generating canonical dataset...")
    dataset_path = "data/raw/pde_dataset_48000_canonical.csv"
    Path(dataset_path).parent.mkdir(parents=True, exist_ok=True)
    
    gen_cmd = f"python3 generate_canonical_dataset.py --output {dataset_path} --num_per_family 3000 --seed 42"
    if not run_command(gen_cmd, "Dataset generation"):
        print("\n❌ Dataset generation failed!")
        sys.exit(1)
    
    # Check dataset exists
    if not Path(dataset_path).exists():
        print(f"\n❌ Dataset file not found: {dataset_path}")
        sys.exit(1)
    
    # Step 3: Create splits
    print("\nStep 3: Creating data splits...")
    splits_dir = "data/splits_48000_canonical"
    splits_cmd = f"python3 datasets/operators/create_data_splits.py --dataset {dataset_path} --output {splits_dir}"
    if not run_command(splits_cmd, "Data splits creation"):
        print("\n⚠ Splits creation failed, but continuing...")
    
    # Step 4: Tokenize
    print("\nStep 4: Tokenizing dataset...")
    tokenized_dir = "data/tokenized_48000_canonical"
    tokenize_cmd = f"python3 datasets/operators/create_tokenized_data.py --dataset {dataset_path} --splits {splits_dir} --output {tokenized_dir}"
    if not run_command(tokenize_cmd, "Dataset tokenization"):
        print("\n⚠ Tokenization failed, but continuing...")
    
    # Step 5: Create config
    print("\nStep 5: Creating training config...")
    config_path = "configs/config_vae_48000_canonical.yaml"
    
    # Read original config
    original_config = Path("configs/config_vae_48000_operator.yaml")
    if original_config.exists():
        import yaml
        with open(original_config) as f:
            config = yaml.safe_load(f)
        
        # Update paths
        def update_paths(d):
            if isinstance(d, dict):
                for k, v in d.items():
                    if isinstance(v, str) and '48000_fixed' in v:
                        d[k] = v.replace('48000_fixed', '48000_canonical')
                    elif isinstance(v, dict):
                        update_paths(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        update_paths(item)
        
        update_paths(config)
        
        # Write new config
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"✅ Config created: {config_path}")
    else:
        print(f"⚠ Original config not found: {original_config}")
        print(f"   Please create {config_path} manually")
    
    # Summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print()
    print("Next steps:")
    print("1. Verify tokenized data exists:")
    print(f"   ls -lh {tokenized_dir}/")
    print()
    print("2. Train models for all betas:")
    print("   Grammar VAE: beta=2e-4, beta=1e-2")
    print("   Token VAE: beta=2e-4, beta=1e-2")
    print()
    print("   Example command:")
    print(f"   python -m vae.train.train \\")
    print(f"     --config {config_path} \\")
    print(f"     --tokenization grammar \\")
    print(f"     --beta 2e-4 \\")
    print(f"     --seed 42")
    print()
    print("   Or use SLURM script (update paths first)")
    print()

if __name__ == "__main__":
    main()
