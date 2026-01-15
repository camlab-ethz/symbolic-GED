# VAE Training Guide

Complete guide for training Variational Autoencoders on PDE sequences using Grammar or Token tokenization.

---

## 📁 Project Structure

```
vae/
├── train/                    # Training scripts
│   ├── train.py              # Main training script
│   ├── train_vae.sh          # Bash wrapper for training
│   └── __init__.py
├── inference/                # Inference scripts
│   ├── decode.py             # Decode test set predictions
│   └── __init__.py
├── utils/                    # Utilities
│   ├── datamodule.py         # Grammar VAE data loader
│   ├── token_datamodule.py   # Token VAE data loader
│   ├── utils.py              # Loss functions
│   └── __init__.py
├── module.py                 # VAE Lightning module (core model)
├── architecture.py           # Encoder/Decoder networks
├── config.py                 # Configuration dataclasses
└── TRAINING_GUIDE.md         # This file
```

---

## 🚀 Quick Start

### 1. Create Dataset

First, ensure you have a tokenized dataset ready:

```bash
cd /cluster/work/math/ooikonomou/symbolic-GED/src

# Create dataset (if not already created)
bash dataset_creation/create_dataset.sh
```

This creates:
- `data/raw/pde_dataset_48000_fixed.csv` - Dataset CSV (operator-only strings)
- `data/splits_48000_fixed/` - Train/val/test split indices
- `data/tokenized_48000_fixed/grammar_full.npy` - Grammar tokenized data
- `data/tokenized_48000_fixed/token_full.npy` - Token tokenized data
- `data/tokenized_48000_fixed/*_masks.npy` - Masks for constrained decoding

### 2. Train Grammar VAE

```bash
# Recommended (reporting-ready 48000_fixed config)
python3 -m vae.train.train --config configs/config_vae_48000_operator.yaml --tokenization grammar --beta 2e-4 --seed 42
```

### 3. Train Token VAE

```bash
python3 -m vae.train.train --config configs/config_vae_48000_operator.yaml --tokenization token --beta 2e-4 --seed 42
```

---

## ⚙️ Configuration

### Default Configuration

For the reporting-ready pipeline, use:
- `configs/config_vae_48000_operator.yaml`

If you omit `--config`, the training script falls back to `config_vae.yaml` (legacy/default).

```yaml
model:
  shared:
    z_dim: 26  # Latent dimension

training:
  batch_size: 256
  epochs: 1000
  learning_rate: 0.001
  beta: 0.0001  # KL weight
  seed: 42  # For reproducibility

grammar:
  model:
    max_length: 114
    vocab_size: 56  # Production rules
  saving:
    checkpoint_dir: "checkpoints_48000_fixed/grammar_vae"

token:
  model:
    max_length: 62
    vocab_size: 82  # Character tokens
  saving:
    checkpoint_dir: "checkpoints_48000_fixed/token_vae"
```

### Override Settings via CLI

```bash
python3 -m vae.train.train \
    --tokenization grammar \
    --seed 42 \
    --batch_size 512 \
    --lr 0.0005 \
    --epochs 500 \
    --beta 0.0001 \
    --gpus 1
```

### Configuration File Paths

The training script expects:
- **Grammar data**: `data/tokenized_48000_fixed/grammar_full.npy` + `grammar_full_masks.npy`
- **Token data**: `data/tokenized_48000_fixed/token_full.npy` + `token_full_masks.npy`
- **Splits**: `data/splits_48000_fixed/train_indices.npy`, `val_indices.npy`, `test_indices.npy`

These paths are set in `vae/config.py` and can be overridden via `--config` (recommended: `configs/config_vae_48000_operator.yaml`).

---

## 🔧 Training Options

### Basic Training

```bash
# Grammar VAE
python3 -m vae.train.train --tokenization grammar

# Token VAE
python3 -m vae.train.train --tokenization token
```

### Custom Hyperparameters

```bash
python3 -m vae.train.train \
    --tokenization grammar \
    --batch_size 512 \
    --lr 0.0005 \
    --epochs 500 \
    --beta 0.0001 \
    --seed 42
```

### GPU Training

```bash
python3 -m vae.train.train \
    --tokenization grammar \
    --gpus 1 \
    --num_workers 4
```

### Early Stopping

Early stopping is configured in your YAML config (e.g. `configs/config_vae_48000_operator.yaml`):

```yaml
training:
  early_stopping_patience: 20  # Stop if no improvement for 20 epochs
```

### Learning Rate Scheduling

```yaml
training:
  lr_scheduler: "plateau"  # Options: null, "plateau", "cosine", "step"
  lr_scheduler_patience: 10
  lr_scheduler_factor: 0.5
  lr_scheduler_min: 1e-6
```

Override via CLI:

```bash
python3 -m vae.train.train \
    --tokenization grammar \
    --lr_scheduler cosine \
    --lr_scheduler_min 1e-5
```

---

## 📊 Output and Logging

### Checkpoints

Models are saved to separate directories by tokenization type:

- **Grammar VAE**: `checkpoints_48000_fixed/grammar_vae/`
- **Token VAE**: `checkpoints_48000_fixed/token_vae/`

Best model is saved based on validation sequence accuracy:
```
checkpoints_48000_fixed/grammar_vae/
  └── best-epoch=189-seqacc=val/seq_acc=0.9907.ckpt
```

### Logs

Training logs are saved to:

- **Output log**: `logs/training/train_{tokenization}_{timestamp}.out`
- **Error log**: `logs/training/train_{tokenization}_{timestamp}.err`

Using the bash script:
```bash
bash vae/train/train_vae.sh grammar
# Logs: logs/training/train_grammar_20250109_153000.{out,err}
```

### TensorBoard Logs

PyTorch Lightning automatically logs to:
- `lightning_logs/version_{version}/`

View with:
```bash
tensorboard --logdir lightning_logs/
```

---

## 🔄 Reproducibility

### Seed Setting

The training script sets seeds for full reproducibility:

```python
# In train.py
if config.seed is not None:
    pl.seed_everything(config.seed, workers=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
```

**Default seed**: `42` (set in your YAML config)

### Ensure Reproducibility

1. **Set seed explicitly**:
   ```bash
   python3 -m vae.train.train --tokenization grammar --seed 42
   ```

2. **Use deterministic algorithms** (if needed):
   ```python
   torch.use_deterministic_algorithms(True)
   ```

3. **Same dataset splits**: The dataset creation uses a fixed seed (42) for train/val/test splits.

4. **Same data order**: DataLoader uses `workers=True` with seed, ensuring reproducible shuffling.

---

## 🧪 Running Experiments

### Example: Compare Different Beta Values

```bash
# Beta = 1e-5
python3 -m vae.train.train --tokenization grammar --beta 0.00001 --seed 42

# Beta = 1e-4 (default)
python3 -m vae.train.train --tokenization grammar --beta 0.0001 --seed 42

# Beta = 1e-3
python3 -m vae.train.train --tokenization grammar --beta 0.001 --seed 42
```

### Example: Different Latent Dimensions

Modify your YAML config (e.g. `configs/config_vae_48000_operator.yaml`):
```yaml
model:
  shared:
    z_dim: 32  # Increase from 26
```

Or override:
```bash
# Need to modify config.py or use config file
```

### Example: Different Batch Sizes

```bash
# Small batch (for debugging)
python3 -m vae.train.train --tokenization grammar --batch_size 32

# Large batch (for speed)
python3 -m vae.train.train --tokenization grammar --batch_size 512
```

---

## 🐛 Troubleshooting

### Out of Memory (OOM)

Reduce batch size:
```bash
python3 -m vae.train.train --tokenization grammar --batch_size 128
```

### Slow Training

1. **Increase batch size** (if memory allows):
   ```bash
   --batch_size 512
   ```

2. **Use more workers**:
   ```bash
   --num_workers 8
   ```

3. **Use GPU**:
   ```bash
   --gpus 1
   ```

### Checkpoint Not Found

Ensure dataset is created first:
```bash
bash dataset_creation/create_dataset.sh
```

Check data paths in `vae/config.py` or your `--config` YAML.

### Import Errors

Ensure you're in the correct directory:
```bash
cd /cluster/work/math/ooikonomou/symbolic-GED/src
```

Or add to PYTHONPATH:
```bash
export PYTHONPATH=/cluster/work/math/ooikonomou/symbolic-GED/src:$PYTHONPATH
```

---

## 📝 Training Checklist

Before training:

- [ ] Dataset created: `bash dataset_creation/create_dataset.sh`
- [ ] Data files exist: `data/tokenized_48000_fixed/grammar_full.npy`, `token_full.npy`
- [ ] Split indices exist: `data/splits_48000_fixed/train_indices.npy`, etc.
- [ ] Configuration file exists: `configs/config_vae_48000_operator.yaml`
- [ ] Output directories created: `checkpoints_48000_fixed/`, `logs_48000_fixed/`

After training:

- [ ] Checkpoint saved: `checkpoints_48000_fixed/{tokenization}_vae/beta_..._seed_.../best-*.ckpt`
- [ ] Logs reviewed: `slurm_logs/*.out` or `logs_48000_fixed/`
- [ ] TensorBoard logs: `lightning_logs/`
- [ ] Model can be loaded for inference

---

## 🔗 Next Steps

After training:

1. **Evaluate on test set**: Use `vae/inference/decode.py`
2. **Extract latents**: Use `training/extract_latents.py`
3. **Visualize**: Use analysis scripts in `scripts/` or `analysis/`

---

## 📚 References

- **PyTorch Lightning**: https://pytorch-lightning.readthedocs.io/
- **VAE Paper**: Kingma & Welling, "Auto-Encoding Variational Bayes" (2014)
- **Grammar VAE**: Gomez et al., "Grammar Variational Autoencoder" (ICML 2018)

---

**Last Updated**: January 2025  
**Maintained by**: Symbolic-GED Team
