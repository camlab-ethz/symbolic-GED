# VAE Training & Evaluation Guide

This folder contains a **unified VAE training pipeline** that handles both Grammar and Token tokenization methods.

---

## 📁 Essential Files

### Core Training
- **`train.py`** - Main training script (handles both Grammar & Token)
- **`module.py`** - VAE architecture (encoder-decoder with KL loss)
- **`datamodule.py`** - Grammar data loader (production sequences)
- **`token_datamodule.py`** - Token data loader (character sequences)
- **`architecture.py`** - Model components (encoder, decoder)

### Evaluation
- **`decode_test_set.py`** - Decode test set and compare both models

### Configuration
- **`configs.yaml`** - Training configurations (optional, can use parent configs)
- **`CONFIG_README.md`** - Config documentation

### Utilities
- **`utils.py`** - Helper functions
- **`list_configs.py`** - List available configurations

---

## 🚀 Quick Start

### Train Grammar VAE
```bash
cd /cluster/work/math/ooikonomou/symbolic-GED/src

python3 -m vae.train --tokenization grammar
```

### Train Token VAE
```bash
python3 -m vae.train --tokenization token
```

### With Custom Parameters
```bash
python3 -m vae.train --tokenization grammar \
  --batch_size 512 \
  --lr 0.0005 \
  --epochs 500 \
  --beta 0.0001
```

---

## 📊 Evaluate Both Models

```bash
cd /cluster/work/math/ooikonomou/symbolic-GED/src

python3 vae/decode_test_set.py \
  --grammar_checkpoint checkpoints/grammar_vae/best-epoch=189-seqacc=val/seq_acc=0.9907.ckpt \
  --token_checkpoint checkpoints/token_vae/best-epoch=314-seqacc=val/seq_acc=0.9841.ckpt \
  --csv_path pde_dataset_48444.csv \
  --output_file decoded_test_comparisons_readable.txt
```

**Output:**
- Readable human-friendly PDE comparisons
- Exact-match accuracy for both models
- Per-example correctness flags

---

## ⚙️ Configuration

### Using Parent Config Files (Recommended)

The training script automatically loads from `../config_vae.yaml`:

```yaml
# Grammar tokenization settings
grammar:
  model:
    max_length: 114
    vocab_size: 53
  data:
    prod_path: 'examples_out/prod_48444_ids_int16.npy'
    masks_path: 'examples_out/prod_48444_masks_packed.npy'
    split_dir: 'data_splits'

# Token tokenization settings  
token:
  model:
    max_length: 62
    vocab_size: 82
  data:
    token_path: 'examples_out/token_48444_ids_int16.npy'
    masks_path: 'examples_out/token_48444_masks_bool.npy'
    split_dir: 'data_splits'

# Shared model architecture
model:
  shared:
    z_dim: 74
  encoder:
    hidden_size: 256
    conv_sizes: [64, 128, 256]
    kernel_sizes: [2, 3, 4]
  decoder:
    hidden_size: 501
    num_layers: 2
    dropout: 0.1

# Shared training settings
training:
  batch_size: 256
  lr: 0.0001
  epochs: 1000
  beta: 0.0001
  clip: 5.0
  early_stopping_patience: 50
```

### Command-Line Overrides

```bash
# Override batch size
python3 -m vae.train --tokenization grammar --batch_size 128

# Override learning rate and beta
python3 -m vae.train --tokenization token --lr 0.0005 --beta 0.00001

# Override epochs and early stopping
python3 -m vae.train --tokenization grammar --epochs 500 --early_stop_patience 30
```

---

## 📦 Data Requirements

### Grammar Method
Required files in `examples_out/`:
- `prod_48444_ids_int16.npy` - Production IDs (N × 114)
- `prod_48444_masks_packed.npy` - Validity masks

### Token Method
Required files in `examples_out/`:
- `token_48444_ids_int16.npy` - Token IDs (N × 62)
- `token_48444_masks_bool.npy` - Validity masks

### Train/Val/Test Splits
Required files in `data_splits/`:
- `train_indices.npy`
- `val_indices.npy`
- `test_indices.npy`

---

## 🏗️ Architecture

Both models share the same architecture:

```
Input: PDE encoded as integer IDs
  ↓
Encoder: Conv1D layers → Latent space (z_dim=74)
  ↓
Sampling: μ, σ → z ~ N(μ, σ²)
  ↓
Decoder: GRU layers → Reconstructed sequence
  ↓
Output: Probability distribution over vocabulary
```

**Loss Function:**
```python
loss = reconstruction_loss + β * KL_divergence
```

- **Reconstruction Loss:** Cross-entropy (valid positions only, using masks)
- **KL Divergence:** Regularization to latent space
- **β:** Balances reconstruction vs regularization (default: 0.0001)

---

## 📈 Monitoring Training

### TensorBoard (if enabled)
```bash
tensorboard --logdir=checkpoints/grammar_vae
tensorboard --logdir=checkpoints/token_vae
```

### Check Logs
```bash
# View latest training log
tail -f lightning_logs/version_*/events.out.tfevents.*

# Or check metrics
grep "val/seq_acc" lightning_logs/version_*/hparams.yaml
```

### Checkpoints
Best models saved to:
- `checkpoints/grammar_vae/best-epoch=XXX-seqacc=val/seq_acc=0.XXXX.ckpt`
- `checkpoints/token_vae/best-epoch=XXX-seqacc=val/seq_acc=0.XXXX.ckpt`

---

## 🎯 Key Metrics

### Training Metrics
- `train/loss` - Total training loss
- `train/recon_loss` - Reconstruction loss
- `train/kl_loss` - KL divergence
- `train/seq_acc` - Sequence-level exact match

### Validation Metrics
- `val/loss` - Total validation loss
- `val/seq_acc` - Sequence-level exact match (used for checkpoint selection)

**Early Stopping:** Monitors `val/seq_acc` (higher is better)

---

## 🔧 Common Workflows

### Workflow 1: Train Both Models from Scratch

```bash
cd /cluster/work/math/ooikonomou/symbolic-GED/src

# Train Grammar VAE
python3 -m vae.train --tokenization grammar --epochs 500

# Train Token VAE
python3 -m vae.train --tokenization token --epochs 500

# Evaluate both
python3 vae/decode_test_set.py \
  --grammar_checkpoint checkpoints/grammar_vae/best-*.ckpt \
  --token_checkpoint checkpoints/token_vae/best-*.ckpt \
  --csv_path pde_dataset_48444.csv
```

### Workflow 2: Resume Training

```bash
# Resume from checkpoint
python3 -m vae.train --tokenization grammar \
  --resume_from checkpoints/grammar_vae/best-epoch=100.ckpt
```

### Workflow 3: Hyperparameter Search

```bash
# Try different β values
for beta in 0.00001 0.0001 0.001; do
  python3 -m vae.train --tokenization grammar --beta $beta --epochs 200
done
```

### Workflow 4: Telegrapher Bridge Experiment

```bash
# 1. Generate bridge dataset
cd examples
python3 generate_telegrapher_bridge.py
python3 split_telegrapher_data.py telegrapher_bridge_default.csv

# 2. Train on endpoints only
cd ..
python3 -m vae.train --tokenization grammar \
  --dataset examples/telegrapher_train_endpoints.csv

# 3. Test on middle τ (continuation)
python3 vae/decode_test_set.py \
  --grammar_checkpoint checkpoints/grammar_vae/best-*.ckpt \
  --csv_path examples/telegrapher_test_middle.csv
```

---

## 🐛 Troubleshooting

### Issue: "CUDA out of memory"
**Solution:**
```bash
python3 -m vae.train --tokenization grammar --batch_size 128  # Reduce batch size
```

### Issue: "Training diverges (NaN loss)"
**Solution:**
```bash
python3 -m vae.train --tokenization grammar --lr 0.00005 --beta 0.00001  # Lower LR & beta
```

### Issue: "Low reconstruction accuracy"
**Solution:**
- Lower β (e.g., `--beta 0.00001`) to prioritize reconstruction
- Increase model capacity: edit `config_vae.yaml` to increase `z_dim`, `hidden_size`
- Train longer: `--epochs 1000`

### Issue: "Posterior collapse"
**Solution:**
- Use free bits: Edit `module.py` to add free bits mechanism
- Cyclical β annealing: Start with low β, gradually increase
- Check latent utilization: Run `comprehensive_tokenization_analysis.py`

---

## 📚 Key Differences: Grammar vs Token

| Aspect | Grammar | Token |
|--------|---------|-------|
| **Tokenization** | Production rules (CFG) | Character-level (Lample & Charton) |
| **Vocabulary** | 53 production IDs | 82 character tokens |
| **Max Length** | 114 timesteps | 62 timesteps |
| **Data Loader** | `datamodule.py` | `token_datamodule.py` |
| **Masks** | Valid production masks | Valid token positions |
| **Notation** | Left-most derivation | Prefix notation (Polish) |
| **Strengths** | Explicit structure, deterministic | Compact, smooth latent |
| **Weaknesses** | Longer sequences | Less structured |

---

## 📖 Related Documentation

- `../README.md` - Main project documentation
- `../ENCODING_DECODING_GUIDE.md` - Tokenization details
- `../TELEGRAPHER_BRIDGE_README.md` - Continuation experiment
- `CONFIG_README.md` - Configuration reference

---

## 🗑️ Archived Files

Redundant/experimental files moved to `archive/`:
- `multi_train.py` - Multi-experiment runner (superseded by single `train.py`)
- `evaluate.py`, `evaluate_grammar_vae.py` - Old evaluation scripts (use `decode_test_set.py`)
- `decode_and_compare.py` - Duplicate functionality (use `decode_test_set.py`)
- `COMPARISON.md`, `SETUP_COMPLETE.md`, `STABILITY_GUIDE.md` - Old docs

---

## ✅ Quick Checklist

Before training:
- [ ] Data files exist (`examples_out/*.npy`)
- [ ] Split files exist (`data_splits/*.npy`)
- [ ] Config file has correct paths (`config_vae.yaml`)
- [ ] CUDA available (or set `--gpus 0` for CPU)

After training:
- [ ] Check `val/seq_acc` in logs (should be >0.95)
- [ ] Checkpoint saved to `checkpoints/`
- [ ] Run `decode_test_set.py` for evaluation

---

## 🎓 Citation

If you use this code, cite the tokenization comparison:

```bibtex
@misc{pde_vae_tokenization_2025,
  title={Comparing Grammar-Based and Character-Based Tokenization for PDE VAEs},
  author={Your Name},
  year={2025}
}
```

---

**Last updated:** November 26, 2025

**Questions?** Check the main `README.md` or the encoding guide.
