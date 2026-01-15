# PDE VAE Pipeline

Pipeline for generating symbolic PDE datasets, training Variational Autoencoders with two tokenization methods (Grammar-based and Lample-Charton style), and analyzing latent space structure.

## Pipeline Overview

```
1. Dataset Creation    → 48,000 PDEs across 16 families
2. Tokenization        → Grammar (production sequences) + Token (character sequences)
3. VAE Training        → 4 models: grammar/token × β=2e-4/1e-2
4. Latent Analysis     → t-SNE, clustering metrics, interpolation, prior sampling
```

## Quick Start

### Full Pipeline (SLURM)

```bash
cd /cluster/work/math/ooikonomou/symbolic-GED/src

# 1. Create dataset + tokenize
bash dataset_creation/create_dataset.sh

# 2. Train 4 VAE models
sbatch scripts/slurm/vae_train_grid_48000_fixed.sbatch

# 3. Run all analyses (after training completes)
sbatch scripts/slurm/create_tsne_all_models.sbatch
sbatch scripts/slurm/run_clustering_metrics_all_models.sbatch
sbatch scripts/slurm/run_interpolation_examples_all_keys.sbatch
sbatch scripts/slurm/run_prior_sampling_shared_z_all_models.sbatch
```

## Project Structure

```
src/
├── configs/
│   ├── paths_48000_fixed.yaml          # Single source of truth for paths
│   ├── config_vae_48000_operator.yaml  # VAE training config
│   └── interpolation_examples_all_keys.yaml
│
├── pde/                        # PDE definitions & tokenization
│   ├── families.py             # 16 PDE family templates
│   ├── grammar.py              # Grammar-based tokenizer
│   ├── chr_tokenizer.py        # Lample-Charton character tokenizer
│   └── normalize.py            # PDE string normalization
│
├── dataset_creation/           # Dataset generation
│   ├── generator.py            # PDEGenerator class
│   ├── create_data_splits.py   # Stratified train/val/test splits
│   ├── create_tokenized_data.py
│   ├── validate_dataset.py     # Physics validation
│   └── create_dataset.sh       # Full pipeline script
│
├── vae/                        # VAE implementation
│   ├── module.py               # VAE Lightning module
│   ├── architecture.py         # Encoder/Decoder networks
│   ├── config.py               # Dataclass configuration
│   ├── train/train.py          # Training script
│   ├── interpolate_family_pairs.py
│   ├── prior_sampling_shared_z.py
│   └── utils/                  # Data modules
│
├── analysis/                   # Latent space analysis
│   ├── run_clustering_metrics.py  # NMI, ARI, Purity, Silhouette
│   ├── clustering.py
│   ├── pde_classifier.py       # Rule-based PDE classifier
│   └── physics.py              # Physics properties
│
├── scripts/
│   ├── plot_tsne_comparison.py
│   └── slurm/                  # SLURM batch scripts
│
└── tests/                      # Pytest tests
```

## PDE Families (16)

### Linear PDEs (8 families)

| Family | Equation | Physics | Dims |
|--------|----------|---------|------|
| **Heat** | ∂u/∂t = k∇²u | Diffusion, thermal conduction | 1-3D |
| **Wave** | ∂²u/∂t² = c²∇²u | Acoustic/electromagnetic waves | 1-3D |
| **Poisson** | ∇²u = f | Electrostatics, steady-state diffusion | 1-3D |
| **Advection** | ∂u/∂t + v·∇u = 0 | Transport, convection | 1-3D |
| **Airy** | ∂u/∂t + α∂³u/∂x³ = 0 | Linear dispersion | 1D |
| **Telegraph** | ∂²u/∂t² + a∂u/∂t = b²∇²u | Damped wave propagation | 1-3D |
| **Biharmonic** | ∇⁴u = f | Elasticity, plate bending (steady) | 1-2D |
| **Beam/Plate** | ∂²u/∂t² + κ∇⁴u = 0 | Vibrating beams/plates | 1-2D |

### Nonlinear PDEs (8 families)

| Family | Equation | Physics | Dims |
|--------|----------|---------|------|
| **Burgers** | ∂u/∂t + u∂u/∂x = ν∂²u/∂x² | Shock waves, fluid dynamics | 1D |
| **KdV** | ∂u/∂t + u∂u/∂x + δ∂³u/∂x³ = 0 | Shallow water solitons | 1D |
| **Allen-Cahn** | ∂u/∂t = ε²∇²u + u - u³ | Phase-field, interface dynamics | 1-3D |
| **Cahn-Hilliard** | ∂u/∂t = -γ∇⁴u + ∇²(u³-u) | Spinodal decomposition | 1-2D |
| **Fisher-KPP** | ∂u/∂t = D∇²u + ru(1-u) | Population dynamics, combustion | 1-3D |
| **Reaction-Diffusion** | ∂u/∂t = ∇²u ± gu³ | Pattern formation | 1-3D |
| **Kuramoto-Sivashinsky** | ∂u/∂t + ν∂²u + γ∂⁴u + αu∂u/∂x = 0 | Flame fronts, turbulence | 1D |
| **Sine-Gordon** | ∂²u/∂t² - c²∇²u + β sin(u) = 0 | Josephson junctions, dislocations | 1-3D |

### Dataset Statistics

- **Total PDEs**: 48,000 (3,000 per family)
- **Dimensions**: 24,000 1D / 15,000 2D / 9,000 3D
- **Temporal order**: 6,000 steady / 30,000 first-order / 12,000 second-order
- **Nonlinear**: 24,000 / 48,000 (50%)
- **Train/Val/Test**: 70% / 15% / 15% (stratified by family)

## Tokenization Methods

### Grammar VAE (Production Sequences)
- Based on context-free grammar
- Vocabulary: 56 productions
- Max length: 114 tokens
- Enforces syntactic validity via grammar masks

### Token VAE (Lample-Charton Style)
- Character-level tokenization with prefix notation
- Vocabulary: 82 tokens
- Max length: 62 tokens
- No structural constraints

## Training

### Configuration

Training uses `configs/config_vae_48000_operator.yaml`:

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
  epochs: 1000
```

### Manual Training

```bash
python -m vae.train.train \
    --config configs/config_vae_48000_operator.yaml \
    --tokenization grammar \
    --beta 2e-4 \
    --seed 42
```

## Evaluation

### Clustering Metrics
Computes NMI, ARI, Purity, Silhouette scores by family:
```bash
python -m analysis.run_clustering_metrics \
    --ckpt grammar_beta2e4:checkpoints_48000_fixed/grammar_vae/beta_2e-4_seed_42/best.ckpt \
    --splits train,val,test
```

### t-SNE Visualization
```bash
python scripts/plot_tsne_comparison.py \
    --grammar-ckpt checkpoints_48000_fixed/grammar_vae/beta_2e-4_seed_42/best.ckpt \
    --token-ckpt checkpoints_48000_fixed/token_vae/beta_2e-4_seed_42/best.ckpt
```

### Interpolation
```bash
python -m vae.interpolate_family_pairs \
    --config configs/interpolation_examples_all_keys.yaml
```

### Prior Sampling (Shared Z)
```bash
python vae/prior_sampling_shared_z.py \
    --n_samples 20000 \
    --seed 42
```

## Output Locations

| Output | Path |
|--------|------|
| Dataset CSV | `data/raw/pde_dataset_48000_fixed.csv` |
| Splits | `data/splits_48000_fixed/` |
| Tokenized | `data/tokenized_48000_fixed/` |
| Checkpoints | `checkpoints_48000_fixed/` |
| Clustering | `analysis_results/clustering_48000/` |
| Interpolation | `analysis_results/interpolation_examples_all_keys_48000/` |
| Prior Sampling | `analysis_results/prior_sampling_shared_z_48000/` |
| t-SNE Plots | `experiments/reports/tsne_48000_comparison_*/` |

## Documentation

- `DATASET_PIPELINE.md` - Dataset creation details
- `REPRODUCIBILITY.md` - Reproducibility guide
- `README_EVALUATION.md` - Evaluation details
- `vae/TRAINING_GUIDE.md` - Training guide

## Requirements

```
torch>=2.0
pytorch-lightning>=2.0
numpy
pandas
scikit-learn
pyyaml
matplotlib
```
