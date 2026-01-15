# PDE VAE Pipeline

A clean, modular pipeline for training Variational Autoencoders on PDE sequences.
Supports both grammar-based and token-based tokenization methods.

## Quick Start

```bash
# Train with grammar tokenization
python -m vae.train --tokenization grammar

# Train with token tokenization
python -m vae.train --tokenization token

# Use custom config
python -m vae.train --config config.yaml --tokenization grammar

# Override hyperparameters
python -m vae.train --tokenization grammar --lr 0.0005 --batch_size 512
```

## Project Structure

```
src/
├── config.yaml              # Main configuration file
├── README.md                # This file
│
├── vae/                     # VAE implementation
│   ├── config.py            # Dataclass-based configuration
│   ├── module.py            # VAE Lightning module
│   ├── architecture.py      # Encoder/Decoder networks
│   ├── datamodule.py        # Grammar data loading
│   ├── token_datamodule.py  # Token data loading
│   ├── train.py             # Training script
│   └── utils.py             # Loss functions
│
├── analysis/                # Analysis tools
│   ├── clustering.py        # Clustering metrics
│   └── physics.py           # Physics classification
│
├── ood/                     # Out-of-distribution experiments
│   └── scenarios.py         # OOD scenario definitions
│
├── pde_grammar.py           # Grammar definition
├── pde_tokenizer.py         # Token vocabulary
├── pde_families.py          # PDE family definitions
└── generator.py             # PDE dataset generation
```

## Configuration

The configuration system uses dataclasses for type safety and validation:

```python
from vae import VAEConfig

# Load from YAML
config = VAEConfig.from_yaml('config.yaml')

# Create programmatically
config = VAEConfig(
    tokenization='grammar',
    model=ModelConfig(z_dim=32),
    training=TrainingConfig(lr=0.001)
)

# Save to YAML
config.to_yaml('my_config.yaml')
```

### Config Structure

```yaml
model:
  z_dim: 26
  encoder:
    hidden_size: 128
    kernel_sizes: [7, 7, 7]
  decoder:
    hidden_size: 80
    num_layers: 3

training:
  batch_size: 256
  lr: 0.001
  kl_annealing:
    beta: 0.0001
    anneal_epochs: 0
  lr_scheduler:
    type: plateau
    patience: 10
```

## API Usage

### Training

```python
from vae import VAEModule, VAEConfig
from vae.datamodule import GrammarVAEDataModule
import pytorch_lightning as pl

# Load config
config = VAEConfig.from_yaml('config.yaml')

# Create model and data
model = VAEModule.from_config(config)
dm = GrammarVAEDataModule(
    prod_path='examples_out/prod_48444_ids_int16_clean.npy',
    masks_path='examples_out/prod_48444_masks_clean.npy',
    split_dir='data_splits'
)

# Train
trainer = pl.Trainer(max_epochs=100)
trainer.fit(model, dm)
```

### Encoding & Decoding

```python
import torch
from vae import VAEModule

# Load trained model
model = VAEModule.load_from_checkpoint('checkpoints/best.ckpt')
model.eval()

# Encode
x = torch.randn(10, 114, 53)  # (batch, seq_len, vocab)
mu, logvar = model.encode(x)

# Decode
z = torch.randn(10, 26)  # (batch, z_dim)
logits = model.decode(z)

# Sample from prior
samples = model.sample_from_prior(n_samples=100, constrained=True)
```

### Analysis

```python
from analysis import compute_clustering_metrics, classify_pde_type

# Compute clustering metrics
metrics = compute_clustering_metrics(latents, labels, 'family')
print(f"NMI: {metrics['nmi']:.4f}, ARI: {metrics['ari']:.4f}")

# Classify PDE type
pde_type = classify_pde_type('dt(u) = dxx(u)')  # 'parabolic'
```

## Tokenization Methods

### Grammar (Production Sequences)
- Vocabulary: 53 grammar productions
- Max length: 114 tokens
- Enforces syntactic validity via grammar masks

### Token (Character Sequences)
- Vocabulary: 82 characters
- Max length: 62 tokens
- Lample & Charton style prefix notation

## Training Features

### KL Annealing
```bash
# Linear annealing over 50 epochs
python -m vae.train --tokenization grammar --kl_anneal_epochs 50

# Cyclical annealing
python -m vae.train --tokenization grammar --cyclical_beta --cycle_epochs 10
```

### Learning Rate Scheduling
```bash
# ReduceLROnPlateau (default)
python -m vae.train --tokenization grammar --lr_scheduler plateau

# Cosine annealing
python -m vae.train --tokenization grammar --lr_scheduler cosine

# Disable scheduling
python -m vae.train --tokenization grammar --lr_scheduler none
```

## OOD Experiments

Run out-of-distribution experiments:

```python
from ood import get_scenario, list_scenarios

# List available scenarios
print(list_scenarios())

# Get scenario
scenario = get_scenario('hard_families')
masks = scenario.get_masks(df)
```

Available scenarios:
- `hard_families`: Hold out complex PDE families
- `dispersive_ood`: Hold out dispersive PDEs
- `dim_ood_1d2d_to_3d`: Train 1D+2D, test 3D
- `nonlinear_ood`: Train linear only

## Requirements

```
torch>=2.0
pytorch-lightning>=2.0
numpy
pandas
scikit-learn
pyyaml
```

## Citation

```bibtex
@misc{pde_vae_2025,
  title={Grammar-Based vs Token-Based Tokenization for PDE VAEs},
  year={2025}
}
```
