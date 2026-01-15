# Reproducibility Guide (reporting-ready `48000_fixed`)

This repo is designed as **research code with reproducible pipelines**. The “reporting-ready” setup is the operator-only `48000_fixed` dataset plus 4 VAEs (grammar/token × beta=2e-4/1e-2) and the standard analysis suite (t-SNE, clustering, interpolation, shared-z prior sampling).

## Single source of truth

All “final pipeline” paths live in:
- `configs/paths_48000_fixed.yaml`

Most scripts/sbatch files either read this directly or default to these locations.

## Environment

Run everything from the repo root:

```bash
cd /cluster/work/math/ooikonomou/symbolic-GED/src
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

Optional (cleanest import behavior): install the repo as an editable package from the project root:

```bash
cd /cluster/work/math/ooikonomou/symbolic-GED
pip install -e .
```

## 1) Dataset creation (operator-only strings)

Creates:
- `data/raw/pde_dataset_48000.csv` (raw)
- `data/raw/pde_dataset_48000_fixed.csv` (fixed labels + normalized strings)
- `data/splits_48000_fixed/` (train/val/test indices)
- `data/tokenized_48000_fixed/` (grammar/token ids + masks)

```bash
bash dataset_creation/create_dataset.sh
```

## 2) Training (4 models)

The reporting-ready training config is:
- `configs/config_vae_48000_operator.yaml`

### Local (single model)

```bash
python3 -m vae.train.train \
  --config configs/config_vae_48000_operator.yaml \
  --tokenization token \
  --beta 2e-4 \
  --seed 42 \
  --gpus 1
```

### SLURM grid (all 4)

```bash
sbatch scripts/slurm/vae_train_grid_48000_fixed.sbatch
```

Outputs:
- `checkpoints_48000_fixed/grammar_vae/beta_..._seed_42/`
- `checkpoints_48000_fixed/token_vae/beta_..._seed_42/`

## 3) Evaluation (t-SNE, clustering, interpolation, shared-z prior sampling)

### t-SNE (train/val/test)

```bash
sbatch scripts/slurm/create_tsne_all_models.sbatch
```

### Clustering metrics

```bash
# Optional: isolate outputs under analysis_results/clustering_48000/<RUN_ID>/
export RUN_ID="final"
sbatch scripts/slurm/run_clustering_metrics_all_models.sbatch
```

### Interpolation examples (all common keys)

```bash
sbatch scripts/slurm/run_interpolation_examples_all_keys.sbatch
```

### Shared-z prior sampling

```bash
sbatch scripts/slurm/run_prior_sampling_shared_z_all_models.sbatch
```

## 4) PDE families (reporting-ready set)

The `48000_fixed` pipeline uses **16 families**:
`heat`, `wave`, `poisson`, `advection`, `burgers`, `kdv`, `airy`,
`kuramoto_sivashinsky`, `cahn_hilliard`, `beam_plate`,
`reaction_diffusion_cubic`, `allen_cahn`, `fisher_kpp`, `telegraph`,
`biharmonic`, `sine_gordon`.

All dataset PDE strings are **operator-only** (no trailing `= 0`).

---

## 3. VAE Training

### Configuration

Training configs are in YAML format:
- `config_train_grammar.yaml` - Grammar VAE config
- `config_train_token.yaml` - Token VAE config

### Training Commands

```bash
# Train Grammar VAE
python vae/train.py --config config_train_grammar.yaml

# Train Token VAE
python vae/train.py --config config_train_token.yaml
```

### Key Hyperparameters

| Parameter | Grammar VAE | Token VAE |
|-----------|-------------|-----------|
| latent_dim (z_dim) | 26 | 26 |
| encoder_hidden | 128 | 128 |
| decoder_hidden | 80 | 80 |
| beta (KL weight) | 1e-5 | 1e-5 |
| learning_rate | 1e-3 | 1e-3 |
| epochs | ~200 | ~300 |

### Checkpoints

Best models saved in `checkpoints/`:
- `grammar_vae/best-epoch=XXX-seqacc=val/seq_acc=0.99XX.ckpt`
- `token_vae/best-epoch=XXX-seqacc=val/seq_acc=0.98XX.ckpt`

---

## 4. Analysis and Evaluation

### Extract Latent Vectors

```bash
python scripts/run_tokenization_comparison.py
```

Creates:
- `experiments/comparison/latents/grammar_latents.npz`
- `experiments/comparison/latents/token_latents.npz`

### Physics-Aware Classification

The `analysis/pde_classifier.py` module classifies decoded PDEs by:
- Family (16 classes)
- PDE type (parabolic, hyperbolic, elliptic, dispersive)
- Linearity (linear, nonlinear)
- Temporal order (0, 1, 2)
- Spatial order (1, 2, 3, 4)
- Dimension (1, 2, 3)

**Accuracy on dataset: 100% for all labels**

### Clustering Analysis

```python
from analysis.clustering import compute_all_clustering_metrics

metrics = compute_all_clustering_metrics(latents, labels)
# Returns: NMI, ARI, Purity, Silhouette scores
```

### Interpolation Analysis

```python
from analysis.interpolation_analysis import analyze_interpolation

results = analyze_interpolation(
    model=model,
    z_start=z1,
    z_end=z2,
    n_steps=10,
    tokenization='grammar'
)
```

### t-SNE Visualization

```bash
python scripts/run_tsne.py
```

Creates visualizations in `latent_visualizations/`.

---

## 5. Known Issues and Fixes

### Issue 1: Cahn-Hilliard Nonlinearity Labels

**Problem:** Dataset labels cahn_hilliard as `nonlinear=True`, but the actual PDEs don't contain the nonlinear terms (dxx(u³)).

**Status:** Fixed in `pde_dataset_48444_fixed.csv`

**Fix Script:**
```bash
python dataset_creation/fix_dataset_labels.py
```

### Issue 2: Classifier Dimension Detection

**Problem:** Mixed derivatives (dxxyy, dyyzz) weren't detected for dimension inference.

**Status:** Fixed in `analysis/pde_classifier.py`

### Issue 3: Navier-Stokes vs Burgers

**Problem:** Both have `u*dx(u)` term; distinguished by pressure constant.

**Status:** Fixed in `analysis/pde_classifier.py`

### Issue 4: Klein-Gordon Pattern

**Problem:** Pattern required coefficient before `u` but some PDEs have just `+ u`.

**Status:** Fixed in `analysis/pde_classifier.py`

### Issue 5: Kuramoto-Sivashinsky Nonlinearity

**Problem:** `(dx(u))^2` pattern wasn't recognized as nonlinear.

**Status:** Fixed in `analysis/pde_classifier.py`

---

## Reproducibility Checklist

- [ ] Use `pde_dataset_48444_fixed.csv` (not original)
- [ ] Validate dataset: `python dataset_creation/validate_dataset.py`
- [ ] Verify classifier: 100% accuracy on all labels
- [ ] Use same hyperparameters for VAE training
- [ ] Extract latents with same train/val/test splits
- [ ] Use same random seeds for reproducibility

---

## Citation

If you use this code, please cite:
```
[Your paper citation here]
```

---

## Contact

[Your contact information]
