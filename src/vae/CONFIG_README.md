# Grammar-VAE Configuration System

## Quick Start

### 1. Using Preset Configurations

We provide several preset configurations in `configs.yaml`:

```bash
# Original Kusner et al. (2017) architecture
sbatch run_kusner_original.sh

# Kusner hyperparameters + Enhanced architecture (RECOMMENDED)
sbatch run_kusner_hyper.sh

# Enhanced architecture with aggressive KL annealing
sbatch run_enhanced_v1.sh
```

### 2. Custom Training

```bash
# Use a config and override specific parameters
python -m src.grammar_vae.train \
    --prod examples_out/prod_48444_ids_int16.npy \
    --masks examples_out/prod_48444_masks_packed.npy \
    --config kusner_original \
    --batch_size 256 \
    --lr 0.001 \
    --gpus 1

# Train without config (pure CLI)
python -m src.grammar_vae.train \
    --prod examples_out/prod_48444_ids_int16.npy \
    --masks examples_out/prod_48444_masks_packed.npy \
    --z_dim 56 \
    --encoder_hidden 256 \
    --decoder_hidden 384 \
    --batch_size 128 \
    --epochs 100 \
    --gpus 1
```

## Available Configurations

### `kusner_original`
**Original Grammar-VAE from Kusner et al. 2017**
- Small model (~1.5M params)
- 2 conv layers (9 channels, kernel=9)
- 3-layer GRU decoder (501 units)
- Fast KL annealing (10 epochs)
- Large batch size (500)
- **Best for**: Reproducing paper results

### `kusner_hyperparams_enhanced_arch` ⭐ RECOMMENDED
**Kusner's training recipe + modern architecture**
- Enhanced architecture (4.0M params)
- Kusner's hyperparameters (beta=1.0, fast annealing)
- Better stability with free_bits=0.5
- **Best for**: Getting best results with proven training strategy

### `enhanced_v1`
**Current experimental architecture**
- Multi-scale kernels, residual connections
- Aggressive KL annealing (beta=0.5, 90 epochs)
- Smaller batches (128)
- **Best for**: Experiments with slower, gentler training

### `minimal`
**Small model for debugging**
- Fast training (~5min/epoch)
- Useful for testing code changes
- **Best for**: Development and debugging

### `large`
**Maximum capacity model**
- 512 channels, 4 layers
- Very large (~20M params)
- **Best for**: If you have lots of compute and data

## Configuration File Format

Edit `src/grammar_vae/configs.yaml`:

```yaml
my_custom_config:
  description: "My custom architecture"
  
  # Architecture
  z_dim: 56
  encoder_hidden: 256
  encoder_conv_layers: 3
  encoder_kernel: [3, 5, 7]  # Can be int or list
  decoder_hidden: 384
  decoder_layers: 3
  decoder_dropout: 0.2
  
  # Training
  batch_size: 256
  lr: 0.001
  epochs: 100
  
  # VAE loss
  beta: 1.0
  kl_anneal_epochs: 20
  cyclical_beta: false
  cycle_epochs: 10
  free_bits: 0.5
  
  # Monitoring
  early_stop_patience: 15
  save_top_k: 3
```

Then run:
```bash
python -m src.grammar_vae.train \
    --config my_custom_config \
    --prod ... --masks ... --gpus 1
```

## Key Hyperparameters Explained

### `beta` (KL weight)
- Range: 0.0 to 1.0
- **0.5**: Weaker regularization, focuses on reconstruction
- **1.0**: Standard VAE (Kusner et al. used this)
- Higher = more regularized latent space

### `kl_anneal_epochs`
- Number of epochs to ramp beta from 0 → beta_max
- **10-20**: Fast annealing (Kusner et al.)
- **50-90**: Slow annealing (focuses on reconstruction first)

### `free_bits`
- Minimum total KL divergence across all latent dimensions
- **0.0**: No constraint (can fully collapse)
- **0.5-1.0**: Prevents posterior collapse
- **Fixed bug**: Now applies to TOTAL KL, not per-dimension!

### `encoder_kernel`
- Can be single int: `3` → all layers use kernel=3
- Can be list: `[3, 5, 7]` → different kernel per layer (multi-scale)

## Checkpoint Management

The training automatically saves:
1. **Best sequence accuracy**: `best-seq_acc-{epoch}-{val/seq_acc}.ckpt`
2. **Best total loss**: `best-loss-{epoch}-{val/loss}.ckpt`
3. **Last checkpoint**: `last.ckpt`

Configure with `--save_top_k N` to keep top-N checkpoints.

## Monitoring Training

Watch logs in real-time:
```bash
tail -f logs/kusner_original_*.out
```

Key metrics to track:
- `train/seq_acc`: Exact sequence match on training set
- `val/seq_acc`: **MAIN METRIC** - what you care about!
- `train/token_acc`: Per-token accuracy (easier to achieve)
- `train/kl_per_dim`: Average KL per latent dimension

## Troubleshooting

### Low sequence accuracy (<1%)
- Increase model capacity: use `enhanced_v1` or `large`
- Try Kusner's hyperparameters: `kusner_hyperparams_enhanced_arch`
- Increase `free_bits` to prevent collapse

### Model not learning (loss stuck)
- Decrease `beta` or increase `kl_anneal_epochs`
- Check `train/kl_per_dim` - should be >0.01
- Increase learning rate

### Training unstable (NaN loss)
- Decrease learning rate
- Increase `free_bits`
- Add gradient clipping in trainer

### OOM (out of memory)
- Decrease `batch_size`
- Decrease `encoder_hidden` or `decoder_hidden`
- Use `minimal` config for testing

## Critical Bug Fixes Applied

### ✅ Free Bits Fix
**Problem**: Was applying `free_bits=0.5` PER DIMENSION → minimum KL = 28!

**Fixed**: Now applies to TOTAL KL across all dimensions

**Impact**: Model can now learn efficient low-KL representations

### ✅ Checkpoint Monitoring
**Problem**: Was only saving best `val/loss` 

**Fixed**: Now saves both best `val/seq_acc` AND `val/loss`

**Impact**: Can recover models with best accuracy even if loss is higher

## Recommended Experiments

1. **Baseline**: Start with `kusner_original` to match paper
2. **Best performance**: Try `kusner_hyperparams_enhanced_arch`
3. **Slow training**: If rushed, try `enhanced_v1` with gentle annealing
4. **Maximum capacity**: If unlimited compute, try `large`

## Architecture Comparison

| Config | Params | Encoder | Decoder | KL Anneal | Batch |
|--------|--------|---------|---------|-----------|-------|
| kusner_original | 1.5M | 2×9ch | 3×501 | 10 epochs | 500 |
| kusner_hyper | 4.0M | 3×256ch | 3×384 | 20 epochs | 256 |
| enhanced_v1 | 4.0M | 3×256ch | 3×384 | 90 epochs | 128 |
| large | ~20M | 4×512ch | 4×512 | 30 epochs | 256 |
