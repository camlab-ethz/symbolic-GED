# Evaluation Framework Quick Start

This document provides a quick guide to using the comprehensive evaluation framework.

## 🚀 Quick Start

### Prerequisites

1. **Checkpoints**: Trained VAE models
   - Grammar VAE: `checkpoints_48000_fixed/grammar_vae/.../best-epoch=...-val/seq_acc=....ckpt`
   - Token VAE: `checkpoints_48000_fixed/token_vae/.../best-epoch=...-val/seq_acc=....ckpt`

2. **Dataset**: CSV file with PDE strings and labels
   - `data/raw/pde_dataset_48000_fixed.csv`

3. **Dependencies**: All Python packages installed (see requirements)

### Run Full Evaluation

```bash
cd /cluster/work/math/ooikonomou/symbolic-GED/src

python scripts/run_tokenization_eval.py \
    --grammar-ckpt checkpoints_48000_fixed/grammar_vae/.../best-epoch=...-val/seq_acc=....ckpt \
    --token-ckpt checkpoints_48000_fixed/token_vae/.../best-epoch=...-val/seq_acc=....ckpt \
    --csv-metadata data/raw/pde_dataset_48000_fixed.csv \
    --split test \
    --outdir experiments/reports/tokenization_eval/ \
    --seed 42
```

### Run with Pre-computed Latents (Faster)

If you have pre-computed latent vectors:

```bash
python scripts/run_tokenization_eval.py \
    --grammar-ckpt <grammar_ckpt> \
    --token-ckpt <token_ckpt> \
    --latent-npz experiments/comparison/latents/grammar_latents.npz \
    --csv-metadata data/raw/pde_dataset_48000_fixed.csv \
    --split test \
    --outdir experiments/reports/tokenization_eval/
```

## 📊 Output Files

After running, you'll find in `experiments/reports/tokenization_eval/`:

### Metrics
- `metrics_representation.json` + `.csv`: Clustering + classification metrics
- `metrics_decoded.json` + `.csv`: Decoded semantics metrics
- `interpolation.json`: Interpolation analysis
- `perturbation.json`: Perturbation analysis
- `sampling.json`: Prior sampling results

### Visualizations
- `plots/grammar_tsne_gt.png`: Grammar VAE t-SNE (GT labels)
- `plots/grammar_tsne_decoded.png`: Grammar VAE t-SNE (decoded labels)
- `plots/token_tsne_gt.png`: Token VAE t-SNE (GT labels)
- `plots/token_tsne_decoded.png`: Token VAE t-SNE (decoded labels)
- `plots/grammar_umap_gt.png`: Grammar VAE UMAP (GT labels)
- `plots/token_umap_gt.png`: Token VAE UMAP (GT labels)
- (and more...)

### Report
- `REPORT.md`: Comprehensive markdown report with all results

## ⏱️ Runtime

- **Full evaluation**: ~30-60 minutes (depends on dataset size)
- **With pre-computed latents**: ~15-30 minutes
- **GPU recommended** for encoding/decoding

## 🔧 Command Line Options

```
--grammar-ckpt      Path to Grammar VAE checkpoint (required)
--token-ckpt        Path to Token VAE checkpoint (required)
--csv-metadata      Path to CSV with PDE strings (required)
--split             Dataset split: train/val/test (default: test)
--latent-npz        Optional: Pre-computed latents (NPZ format)
--n_pairs           Interpolation pairs (default: 50)
--n_steps           Interpolation steps (default: 11)
--sigma             Perturbation sigma (default: 0.1)
--n_samples         Samples for perturbation/sampling (default: 500)
--seed              Random seed (default: 42)
--outdir            Output directory (default: experiments/reports/tokenization_eval)
--device            Device: cuda/cpu (default: cuda)
```

## 📖 Documentation

For detailed documentation, see:
- `COMPREHENSIVE_CODEBASE_REPORT.md`: Full codebase documentation
- `CODEBASE_SUMMARY.md`: Quick summary

## 🐛 Troubleshooting

### GPU Not Available
Use `--device cpu` (will be slower)

### Out of Memory
- Use smaller `--n_samples`
- Use smaller `--n_pairs`
- Process in batches

### UMAP Not Available
Script will skip UMAP plots and continue with t-SNE only.

## 📝 Example Output

After running, check `experiments/reports/tokenization_eval/REPORT.md` for:
- Comparison tables
- Winner indicators (Grammar vs Token)
- All metrics with values
- Summary statistics
