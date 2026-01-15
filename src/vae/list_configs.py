#!/usr/bin/env python3
"""List available Grammar-VAE configurations"""

import yaml
import os

config_path = os.path.join(os.path.dirname(__file__), "configs.yaml")

with open(config_path, "r") as f:
    configs = yaml.safe_load(f)

print("\n" + "=" * 70)
print("Available Grammar-VAE Configurations")
print("=" * 70 + "\n")

for name, config in configs.items():
    desc = config.get("description", "No description")
    print(f"📦 {name}")
    print(f"   {desc}")
    print(
        f"   Architecture: z={config['z_dim']}, enc={config['encoder_conv_layers']}×{config['encoder_hidden']}, dec={config['decoder_layers']}×{config['decoder_hidden']}"
    )
    print(
        f"   Training: batch={config['batch_size']}, lr={config['lr']}, epochs={config['epochs']}"
    )
    print(
        f"   VAE: beta={config['beta']}, kl_anneal={config['kl_anneal_epochs']}, free_bits={config['free_bits']}"
    )
    print()

print("=" * 70)
print("Usage:")
print("  sbatch run_kusner_original.sh      # Train with kusner_original")
print(
    "  sbatch run_kusner_hyper.sh         # Train with kusner_hyperparams (RECOMMENDED)"
)
print("  sbatch run_enhanced_v1.sh          # Train with enhanced_v1")
print()
print("Or use custom config:")
print("  python -m src.grammar_vae.train --config <name> --prod ... --masks ...")
print("=" * 70 + "\n")
