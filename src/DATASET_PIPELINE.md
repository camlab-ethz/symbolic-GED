# PDE Dataset Creation and Tokenization Pipeline

Complete reproduction guide for generating PDE datasets and preparing tokenized data for VAE training.

---

## Overview

This pipeline generates symbolic PDE strings from 16 physics families, assigns metadata labels, and tokenizes them using two methods: (1) Grammar-based tokenization via Context-Free Grammar parsing, and (2) Character-based tokenization in prefix notation (inspired by Lample & Charton 2019, adapted for PDEs).

**Single source of truth (recommended):**
- `configs/paths_48000_fixed.yaml` defines the dataset CSV, split dir, tokenized dir, and output roots used for the reporting-ready `48000_fixed` pipeline.

**Pipeline stages:**
1. Generate PDE strings → CSV with labels → `data/raw/`
2. Fix and validate labels → `data/raw/pde_dataset_48000_fixed.csv`
3. Create train/val/test splits → `data/splits_48000_fixed/`
4. Tokenize for Grammar VAE → `data/tokenized_48000_fixed/grammar/`
5. Tokenize for Token VAE → `data/tokenized_48000_fixed/token/`

**Data Organization:**
```
data/
├── raw/                           # Raw dataset files
│   ├── pde_dataset_48000.csv
│   └── pde_dataset_48000_fixed.csv
├── splits_48000_fixed/            # Train/val/test indices
│   ├── train_indices.npy
│   ├── val_indices.npy
│   └── test_indices.npy
└── tokenized_48000_fixed/         # Tokenized data for VAE training
    ├── grammar/
    │   ├── train.npy
    │   ├── val.npy
    │   └── test.npy
    └── token/
        ├── train.npy
        ├── val.npy
        └── test.npy
```

---

## Prerequisites

```bash
# Required Python packages
pip install pandas numpy torch scikit-learn sympy
```

---

## Quick Start: Run Dataset Creation Pipeline

```bash
cd /cluster/work/math/ooikonomou/symbolic-GED/src

# Run entire dataset creation pipeline in one command
bash dataset_creation/create_dataset.sh
```

This will:
1. Generate the dataset
2. Fix labels
3. Validate correctness
4. Create train/val/test splits (70/15/15)
5. Tokenize for both Grammar and Token VAEs

**Note**: This script only creates the dataset files. VAE training is done separately (see "Next Steps" section below).

---

## Step-by-Step Guide

### Step 1: Generate PDE Dataset

#### Command

```bash
cd /cluster/work/math/ooikonomou/symbolic-GED/src
python dataset_creation/generator.py \
    --output data/raw/pde_dataset_48000.csv \
    --num_per_family 3000
```

#### What it does

- Reads PDE family definitions from `pde/families.py`
- Generates 16 families × ~3000 PDEs each = ~48,444 total
- Samples random coefficients from predefined ranges
- Creates variants for 1D, 2D, 3D spatial dimensions
- Outputs CSV: `pde`, `family`, `dim`, `temporal_order`, `spatial_order`, `nonlinear`

#### Output Format

**File:** `data/raw/pde_dataset_48000.csv`

| Column | Type | Description |
|--------|------|-------------|
| `pde` | string | PDE string (no "= 0" suffix) |
| `family` | string | One of 16 families (heat, wave, etc.) |
| `dim` | int | Spatial dimension: 1, 2, or 3 |
| `temporal_order` | int | 0 (elliptic), 1 (parabolic), 2 (hyperbolic) |
| `spatial_order` | int | Highest spatial derivative: 1-4 |
| `nonlinear` | bool | True if contains u², u³, u*dx(u), etc. |

#### Example Output

```csv
pde,family,dim,temporal_order,spatial_order,nonlinear
dt(u) - 1.935*dxx(u),heat,1,1,2,False
dtt(u) - 3.005*dxx(u),wave,1,2,2,False
dt(u) + u*dx(u) - 1.310*dxx(u),burgers,1,1,2,True
```

---

### Step 2: Fix Dataset Labels

#### Command

```bash
python dataset_creation/fix_dataset_labels.py \
    --input data/raw/pde_dataset_48000.csv \
    --output data/raw/pde_dataset_48000_fixed.csv
```

#### What it does

- Corrects label inconsistencies (e.g., Cahn-Hilliard nonlinear labels)
- Removes "= 0" suffixes if present
- Outputs `data/raw/pde_dataset_48000_fixed.csv`

#### Verify labels are correct

```bash
python dataset_creation/validate_dataset.py \
    --dataset data/raw/pde_dataset_48000_fixed.csv
```

Expected output: `✓ DATASET IS VALID` with 100% accuracy on all labels

---

### Step 3: Create Train/Validation/Test Splits

#### Command

```bash
python dataset_creation/create_data_splits.py \
    --dataset data/raw/pde_dataset_48000_fixed.csv \
    --output data/splits_48000_fixed \
    --train-ratio 0.70 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --seed 42
```

#### What it does

- Creates stratified splits by family (ensures balanced distribution)
- Default: 70% train / 15% validation / 15% test
- Saves index arrays: `train_indices.npy`, `val_indices.npy`, `test_indices.npy`
- All splits maintain family proportions

#### Output Files

**Directory:** `data/splits_48000_fixed/`

- `train_indices.npy`: Training set indices (70% of dataset)
- `val_indices.npy`: Validation set indices (15%)
- `test_indices.npy`: Test set indices (15%)

#### Example Output

```
Split sizes:
  Train:  33911 ( 70.0%)
  Val:     7267 ( 15.0%)
  Test:    7266 ( 15.0%)

Family distribution:
Family                Train     Val    Test   Total
--------------------------------------------------
advection              2118     454     454    3026
allen_cahn            2102     451     450    3003
...
```

---

### Step 4: Tokenize Dataset

#### Command

```bash
python dataset_creation/create_tokenized_data.py \
    --dataset data/raw/pde_dataset_48000_fixed.csv \
    --splits data/splits_48000_fixed \
    --output data/tokenized_48000_fixed \
    --grammar-max-len 114 \
    --token-max-len 62
```

#### What it does

Tokenizes the entire dataset using both methods, then splits according to train/val/test indices.

**Grammar VAE Tokenization:**
- Converts PDE string to production rule sequence
- Uses Context-Free Grammar from `pde/grammar.py`
- Vocabulary size: **P = 56** production rules
- Max length: **114** tokens
- Saves: `data/tokenized_48000_fixed/grammar/{train,val,test}.npy`

**Token VAE Tokenization:**
- Converts PDE string to prefix notation
- Character-by-character number tokenization (INT+/INT-)
- Vocabulary size: **V = 82** tokens
- Max length: **62** tokens
- Saves: `data/tokenized_48000_fixed/token/{train,val,test}.npy`

#### Tokenizer Details

**Grammar Tokenization:**
- Parses PDE into derivation sequence using CFG
- Each token is a production rule ID (0-55)
- Padded with -1 (invalid production)
- See `pde/grammar.py` for grammar definition

**Token Tokenization:**

- Converts PDE string to prefix notation (e.g., `dt(u) - 1.935*dxx(u)` → `sub dt u mul INT+ 1 . 9 3 5 dxx u`)
- Tokenizes each symbol, number, and operator
- Pads sequences to fixed length (62) with `0` (PAD token)
- Saves as NumPy array: `(N_split, 62)` int16

#### Tokenization Approach

The token tokenization follows the core principles of **Lample & Charton (2019)** "Deep Learning for Symbolic Mathematics" (ICLR 2020):

- **Number tokenization**: Character-by-character with sign tokens
  - Example: `1.935` → `['INT+', '1', '.', '9', '3', '5']`
  - Negative numbers: `-0.1` → `['INT-', '0', '.', '1']`
- **Prefix notation**: Converts infix to Polish notation
  - Example: `dt(u) - 1.5*dxx(u)` → `['sub', 'dt', 'u', 'mul', 'INT+', '1', '.', '5', 'dxx', 'u']`
- **Named operators**: Uses `add`, `sub`, `mul`, `div`, `pow` instead of symbols

**Adaptation for PDEs**: 
- Direct decimal tokenization (Lample & Charton use SymPy Rationals for floats)
- PDE-specific derivative tokens (`dt`, `dtt`, `dx`, `dxx`, etc.)
- Functional notation style (`dt(u)` instead of `∂u/∂t`)

**Reference**: Lample, G., & Charton, F. (2019). Deep Learning for Symbolic Mathematics. *ICLR 2020*. https://arxiv.org/abs/1912.01412

#### Output Files

**Directory:** `data/tokenized_48000_fixed/`

```
data/tokenized_48000_fixed/
├── grammar/
│   ├── train.npy  (..., 114) int16
│   ├── val.npy    (..., 114) int16
│   └── test.npy   (..., 114) int16
├── token/
│   ├── train.npy  (..., 62)  int16
│   ├── val.npy    (..., 62)  int16
│   └── test.npy   (..., 62)  int16
├── grammar_full.npy  (..., 114) - Full dataset (for compatibility)
└── token_full.npy    (..., 62)  - Full dataset (for compatibility)
```

---

## Step 5: Verification

### Verify Splits

```bash
python << 'EOF'
import numpy as np
import pandas as pd

# Load data
df = pd.read_csv('data/raw/pde_dataset_48000_fixed.csv')
train_idx = np.load('data/splits_48000_fixed/train_indices.npy')
val_idx = np.load('data/splits_48000_fixed/val_indices.npy')
test_idx = np.load('data/splits_48000_fixed/test_indices.npy')

print(f"Dataset size: {len(df)}")
print(f"\nSplit sizes:")
print(f"  Train: {len(train_idx)} ({100*len(train_idx)/len(df):.1f}%)")
print(f"  Val:   {len(val_idx)} ({100*len(val_idx)/len(df):.1f}%)")
print(f"  Test:  {len(test_idx)} ({100*len(test_idx)/len(df):.1f}%)")

# Check for overlap (should be none)
overlap_train_val = np.intersect1d(train_idx, val_idx)
overlap_train_test = np.intersect1d(train_idx, test_idx)
overlap_val_test = np.intersect1d(val_idx, test_idx)

print(f"\nOverlaps (should be 0):")
print(f"  Train-Val: {len(overlap_train_val)}")
print(f"  Train-Test: {len(overlap_train_test)}")
print(f"  Val-Test: {len(overlap_val_test)}")
EOF
```

### Verify Tokenization

```bash
python << 'EOF'
import numpy as np
from pde.grammar import PROD_COUNT
from pde.chr_tokenizer import PDETokenizer

# 1. Check grammar tokenization
grammar_train = np.load('data/tokenized_48000_fixed/grammar/train.npy', mmap_mode='r')
print(f"✓ Grammar tokenization:")
print(f"  Shape: {grammar_train.shape}")
print(f"  Vocabulary size (P): {PROD_COUNT}")
print(f"  Data type: {grammar_train.dtype}")
print(f"  Padding value: -1 (invalid production)")

# 2. Check token tokenization
token_train = np.load('data/tokenized_48000_fixed/token/train.npy', mmap_mode='r')
tokenizer = PDETokenizer()
print(f"\n✓ Token tokenization:")
print(f"  Shape: {token_train.shape}")
print(f"  Vocabulary size (V): {tokenizer.vocab.vocab_size}")
print(f"  Data type: {token_train.dtype}")
print(f"  Padding value: 0 (PAD token)")

# 3. Decode example
first_tokens = [int(t) for t in token_train[0] if t > 0]
decoded_token = tokenizer.decode(first_tokens, skip_special_tokens=True)
print(f"\n✓ Example decode:")
print(f"  First 10 token IDs: {first_tokens[:10]}")
print(f"  Decoded: '{decoded_token}'")
EOF
```

---

## Complete Dataset Creation (One Command)

```bash
cd /cluster/work/math/ooikonomou/symbolic-GED/src && \
bash dataset_creation/create_dataset.sh
```

**Or run steps manually:**

```bash
cd /cluster/work/math/ooikonomou/symbolic-GED/src && \
python dataset_creation/generator.py --output data/raw/pde_dataset_48000.csv --num_per_family 3000 && \
python dataset_creation/fix_dataset_labels.py --input data/raw/pde_dataset_48000.csv --output data/raw/pde_dataset_48000_fixed.csv && \
python dataset_creation/validate_dataset.py --dataset data/raw/pde_dataset_48000_fixed.csv && \
python dataset_creation/create_data_splits.py --dataset data/raw/pde_dataset_48000_fixed.csv --output data/splits_48000_fixed && \
python dataset_creation/create_tokenized_data.py
```

---

## File Structure

```
src/
├── data/                           # All data files (NEW)
│   ├── raw/                        # Raw dataset CSV files
│   │   ├── pde_dataset_48000.csv
│   │   └── pde_dataset_48000_fixed.csv
│   ├── splits/                     # Train/val/test indices
│   │   ├── train_indices.npy
│   │   ├── val_indices.npy
│   │   └── test_indices.npy
│   └── tokenized/                  # Tokenized data for VAE
│       ├── grammar/
│       │   ├── train.npy
│       │   ├── val.npy
│       │   └── test.npy
│       └── token/
│           ├── train.npy
│           ├── val.npy
│           └── test.npy
├── dataset_creation/               # Dataset creation scripts
│   ├── fix_dataset_labels.py
│   ├── validate_dataset.py
│   ├── create_data_splits.py       # Creates train/val/test splits
│   ├── create_tokenized_data.py    # Tokenizes dataset
│   ├── create_dataset.sh           # Master script
│   └── generator.py                # Dataset generation script
├── pde/                            # PDE module
│   ├── __init__.py
│   ├── families.py                 # PDE family definitions
│   ├── grammar.py                  # Grammar tokenizer (P=56)
│   └── chr_tokenizer.py            # Character/Token tokenizer (V=82)
```

---

## PDE Family Reference

### Mathematical Definitions

| Family | Standard Form | Our Format | Type |
|--------|---------------|------------|------|
| **Heat** | ∂u/∂t = α∇²u | `dt(u) - α*dxx(u)` | Parabolic |
| **Wave** | ∂²u/∂t² = c²∇²u | `dtt(u) - c²*dxx(u)` | Hyperbolic |
| **Telegraph** | ∂²u/∂t² + α∂u/∂t = c²∇²u | `dtt(u) + α*dt(u) - c²*dxx(u)` | Hyperbolic |
| **Advection** | ∂u/∂t + c·∂u/∂x = 0 | `dt(u) + c*dx(u)` | Hyperbolic |
| **Burgers** | ∂u/∂t + u∂u/∂x = ν∇²u | `dt(u) + u*dx(u) - ν*dxx(u)` | Parabolic |
| **KdV** | ∂u/∂t + u∂u/∂x + ∂³u/∂x³ = 0 | `dt(u) + u*dx(u) + β*dxxx(u)` | Dispersive |
| **Klein-Gordon** | ∂²u/∂t² - c²∇²u + m²u = 0 | `dtt(u) - c²*dxx(u) + m²*u` | Hyperbolic |
| **Sine-Gordon** | ∂²u/∂t² - ∇²u = sin(u) | `dtt(u) - dxx(u) - β*u^3 + u` | Hyperbolic |
| **Allen-Cahn** | ∂u/∂t = ε²∇²u + u - u³ | `dt(u) - ε²*dxx(u) + u^3 - u` | Parabolic |
| **Fisher-KPP** | ∂u/∂t = D∇²u + ru(1-u) | `dt(u) - D*dxx(u) + r*u^2 - r*u` | Parabolic |
| **Cahn-Hilliard** | ∂u/∂t = -γ∇⁴u (linearized) | `dt(u) + γ*dxxxx(u)` | Parabolic |
| **Kuramoto-Sivashinsky** | ∂u/∂t + (∂u/∂x)² + ∇²u + ∇⁴u = 0 | `dt(u) + (dx(u))^2 + dxx(u) + dxxxx(u)` | Parabolic |
| **Schrödinger** | i∂u/∂t = -∇²u + g\|u\|²u | `dt(u) - dxx(u) - β*u^3` | Parabolic |
| **Poisson** | ∇²u = f | `dxx(u) + dyy(u) - f` | Elliptic |
| **Navier-Stokes** | ∂u/∂t + u∂u/∂x = ν∇²u - ∂p/∂x | `dt(u) + u*dx(u) - ν*dxx(u) + p` | Parabolic |
| **Biharmonic** | ∇⁴u = f | `dxxxx(u) - f` | Elliptic |

### Notation Mapping

| Our Notation | Mathematical | Example PDE |
|-------------|--------------|-------------|
| `dt(u)` | ∂u/∂t | `dt(u) - 1.935*dxx(u)` |
| `dtt(u)` | ∂²u/∂t² | `dtt(u) - 2.5*dxx(u)` |
| `dx(u)`, `dy(u)`, `dz(u)` | ∂u/∂x, ∂u/∂y, ∂u/∂z | `dt(u) + 0.8*dx(u)` |
| `dxx(u)`, `dyy(u)`, `dzz(u)` | ∂²u/∂x², ∂²u/∂y², ∂²u/∂z² | `dt(u) - 1.935*dxx(u)` |
| `dxxx(u)` | ∂³u/∂x³ | `dt(u) + 0.646*dxxx(u)` |
| `dxxxx(u)` | ∂⁴u/∂x⁴ | `dt(u) + 0.241*dxxxx(u)` |
| `u^2`, `u^3` | u², u³ | `dt(u) + 1.437*u^2 - 1.437*u` |
| `u*dx(u)` | u·∂u/∂x | `dt(u) + u*dx(u) - 1.310*dxx(u)` |
| `(dx(u))^2` | (∂u/∂x)² | `dt(u) + 0.805*(dx(u))^2` |

### String Format Rules

1. **No "= 0" suffix**: Store as `dt(u) - α*dxx(u)`, not `dt(u) - α*dxx(u) = 0`
2. **Coefficients**: Random floats with 3 decimal places (e.g., `1.935`, `4.759`)
3. **Operators**: Spaces around `+`, `-`, `*`
4. **Function notation**: Always use `dt(u)`, `dxx(u)`, etc., not `u_t`, `u_xx`

---

## Next Steps: VAE Training

After completing this pipeline, you can train VAEs using:

```bash
# Recommended: training entrypoint + 48000_fixed config
python3 -m vae.train.train --config configs/config_vae_48000_operator.yaml --tokenization grammar --beta 2e-4 --seed 42
python3 -m vae.train.train --config configs/config_vae_48000_operator.yaml --tokenization token   --beta 2e-4 --seed 42

# Or submit the SLURM grid for all 4 models:
sbatch scripts/slurm/vae_train_grid_48000_fixed.sbatch
```

See `vae/TRAINING_GUIDE.md` for detailed training instructions.

---

## References

### Tokenization Methods

1. **Lample, G., & Charton, F. (2019)**
   - "Deep Learning for Symbolic Mathematics"
   - *ICLR 2020*
   - URL: https://arxiv.org/abs/1912.01412
   - **Inspiration for**: Character-level tokenization, prefix notation, INT+/INT- number tokenization
   - **Implementation**: `pde/chr_tokenizer.py` follows core principles with PDE-specific adaptations

2. **Gomez et al. (2018)**
   - "Grammar Variational Autoencoder"
   - *ICML 2018*
   - **Inspiration for**: Context-Free Grammar-based tokenization
   - **Implementation**: `pde/grammar.py` defines CFG for PDE expressions

### Verification

All tokenization implementations have been verified for correctness:
- ✅ Mathematical correctness: All PDE templates match canonical forms
- ✅ Tokenization accuracy: 100% roundtrip encoding/decoding verified
- ✅ Dataset consistency: Verified on the reporting-ready ~48k PDEs
- ✅ Split correctness: No overlaps, proper stratification

See `scripts/verify_tokenization_correctness.py` for automated verification and `TOKENIZATION_VERIFICATION_REPORT.md` for detailed analysis.

---

## Troubleshooting

### Issue: "FileNotFoundError: data/raw/pde_dataset_48000.csv"

**Solution**: Create the directory structure first:
```bash
mkdir -p data/raw data/splits_48000_fixed data/tokenized_48000_fixed/grammar data/tokenized_48000_fixed/token
```

### Issue: "ValueError: Ratios must sum to 1.0"

**Solution**: Ensure train_ratio + val_ratio + test_ratio = 1.0

### Issue: Tokenization fails on some PDEs

**Solution**: Check that PDE strings don't have "= 0" suffix (should be stripped in Step 2)

### Issue: Memory error during tokenization

**Solution**: Use `mmap_mode='r'` when loading large arrays, or process in batches
