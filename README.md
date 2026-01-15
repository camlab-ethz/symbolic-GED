# Symbolic-GED: PDE VAE Pipeline

This repo contains a clean, reproducible pipeline to generate symbolic PDE datasets, tokenize them (grammar + token/Lample-Charton style), train VAEs, and run latent-space analyses (t-SNE, clustering, interpolation, shared-z prior sampling).

## Repo layout

- `src/`: all pipeline code (dataset creation, tokenizers, VAE training, analysis, SLURM scripts)
- `pyproject.toml`: editable install for clean imports
- `.gitignore`: excludes checkpoints/logs/generated artifacts

## Install

```bash
pip install -e .
```

## Quick start (reporting-ready 48000_fixed pipeline)

Single source of truth paths:
- `src/configs/paths_48000_fixed.yaml`

### 1) Create dataset + splits + tokenized arrays

```bash
bash src/dataset_creation/create_dataset.sh
```

### 2) Train the 4 models (SLURM grid)

```bash
sbatch src/scripts/slurm/vae_train_grid_48000_fixed.sbatch
```

### 3) Run analyses (SLURM)

```bash
sbatch src/scripts/slurm/create_tsne_all_models.sbatch
sbatch src/scripts/slurm/run_clustering_metrics_all_models.sbatch
sbatch src/scripts/slurm/run_interpolation_examples_all_keys.sbatch
sbatch src/scripts/slurm/run_prior_sampling_shared_z_all_models.sbatch
```

Optional (tag outputs):
```bash
export RUN_ID="final"
```

## Documentation

- Reproducibility: `src/REPRODUCIBILITY.md`
- Dataset pipeline: `src/DATASET_PIPELINE.md`
- Training guide: `src/vae/TRAINING_GUIDE.md`
- Evaluation notes: `src/README_EVALUATION.md`

## Notes

- Dataset PDE strings are operator-only (no trailing `= 0`).
- Power syntax is normalized consistently across the pipeline (treat `**` as `^` internally for tokenizers/grammar).
