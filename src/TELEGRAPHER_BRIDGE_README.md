# Telegrapher Bridge Dataset - Quick Reference

## Overview

The **Telegrapher Bridge Dataset** is a specialized continuation experiment for testing symbolic interpolation between diffusion-like and wave-like PDEs.

### Telegraph Equation
```
∂²u/∂t² + a·∂u/∂t - c²·∂²u/∂x² = 0
```

Where `a = 1/τ` (damping parameter):
- **Small τ** (large a) → Strong damping → **Diffusion-like**
- **Large τ** (small a) → Weak damping → **Wave-like**
- **Middle τ** (moderate a) → **Transition regime** (unseen during training)

---

## Quick Start

### 1. Generate Dataset
```bash
cd src/examples
python3 generate_telegrapher_bridge.py
```

**Output:**
- `telegrapher_bridge_default.csv` - Default tau ranges
- `telegrapher_bridge_custom.csv` - Custom tau ranges

### 2. Split Train/Test
```bash
python3 split_telegrapher_data.py telegrapher_bridge_default.csv
```

**Output:**
- `telegrapher_train_endpoints.csv` - Training data (5 PDEs: τ ∈ {0.02, 0.05, 0.1, 5.0, 10.0})
- `telegrapher_test_middle.csv` - Testing data (4 PDEs: τ ∈ {0.2, 0.5, 1.0, 2.0})

---

## Python API

### Basic Usage
```python
from generator import PDEGenerator

gen = PDEGenerator(seed=42)
bridge = gen.generate_telegrapher_bridge()

# Filter by split
train = [e for e in bridge if e['split'] == 'train_endpoints']
test = [e for e in bridge if e['split'] == 'test_middle']
```

### Custom Tau Ranges
```python
bridge = gen.generate_telegrapher_bridge(
    tau_small=[0.01, 0.03, 0.08],      # Diffusion-like
    tau_mid=[0.3, 0.7, 1.5, 3.0],      # Unseen continuation
    tau_large=[8.0, 15.0],             # Wave-like
    c_sq=2.5,                          # Wave speed
)
```

---

## Dataset Structure

### CSV Format
```csv
pde,family,dim,temporal_order,spatial_order,nonlinear,split,tau
dtt(u) + 50.0*dt(u) - 1.0*dxx(u) = 0,telegraph,1,2,2,False,train_endpoints,0.02
dtt(u) + 5.0*dt(u) - 1.0*dxx(u) = 0,telegraph,1,2,2,False,test_middle,0.2
...
```

### Fields
- `pde`: PDE string (canonical format)
- `family`: "telegraph"
- `dim`: Spatial dimension (default: 1)
- `temporal_order`: 2 (second-order in time)
- `spatial_order`: 2 (second-order in space)
- `nonlinear`: False
- `split`: "train_endpoints" or "test_middle"
- `tau`: τ value (damping timescale)

---

## Default Tau Ranges

| Split | Tau Values | a = 1/τ | Regime |
|-------|------------|---------|--------|
| `train_endpoints` | 0.02, 0.05, 0.1 | 50.0, 20.0, 10.0 | Diffusion-like |
| `train_endpoints` | 5.0, 10.0 | 0.2, 0.1 | Wave-like |
| `test_middle` | 0.2, 0.5, 1.0, 2.0 | 5.0, 2.0, 1.0, 0.5 | Continuation (unseen) |

---

## Example PDEs

### Diffusion-like (τ = 0.02, a = 50.0)
```
dtt(u) + 50.0*dt(u) - 1.0*dxx(u) = 0
```
- Strong damping dominates
- Second derivative term `dtt(u)` negligible
- Behaves like heat equation

### Wave-like (τ = 10.0, a = 0.1)
```
dtt(u) + 0.1*dt(u) - 1.0*dxx(u) = 0
```
- Weak damping
- Second derivative term `dtt(u)` dominates
- Behaves like wave equation

### Continuation (τ = 1.0, a = 1.0) - UNSEEN
```
dtt(u) + 1.0*dt(u) - 1.0*dxx(u) = 0
```
- Balanced terms
- Transition regime
- **Key test**: Can VAE interpolate?

---

## Experiment Workflow

### Step 1: Generate and Split Data
```bash
# Generate dataset
python3 examples/generate_telegrapher_bridge.py

# Split into train/test
python3 examples/split_telegrapher_data.py telegrapher_bridge_default.csv
```

### Step 2: Encode Dataset
```bash
# Grammar method
python3 -c "
from pde_grammar import parse_to_productions, pad_production_sequence
import csv, numpy as np

with open('examples/telegrapher_train_endpoints.csv') as f:
    pdes = [row['pde'] for row in csv.DictReader(f)]

prods = [pad_production_sequence(parse_to_productions(p), 114) for p in pdes]
np.save('telegrapher_train_grammar.npy', np.array(prods, dtype=np.int16))
"

# Token method
python3 -c "
from pde_tokenizer import PDETokenizer
import csv, numpy as np

tokenizer = PDETokenizer()
with open('examples/telegrapher_train_endpoints.csv') as f:
    pdes = [row['pde'] for row in csv.DictReader(f)]

batch = tokenizer.encode_batch(pdes, pad=True, max_length=62)
np.save('telegrapher_train_token.npy', batch['input_ids'].numpy())
"
```

### Step 3: Train VAE (endpoints only)
```bash
# Train on endpoints
python3 ../model/train.py \
  --config config_train_grammar.yaml \
  --dataset examples/telegrapher_train_endpoints.csv \
  --epochs 500
```

### Step 4: Evaluate Continuation
```bash
# Test on middle τ values
python3 vae/decode_test_set.py \
  --checkpoint checkpoints/grammar_vae/best.ckpt \
  --csv_path examples/telegrapher_test_middle.csv
```

---

## Key Questions to Answer

1. **Reconstruction accuracy on endpoints** (should be high)
   - Can VAE perfectly reconstruct training PDEs?

2. **Reconstruction accuracy on middle τ** (key test)
   - Can VAE interpolate unseen continuation region?
   - Which tokenization handles symbolic continuation better?

3. **Comparison across tokenizations**
   - **Grammar**: Explicit structure, deterministic rules
   - **Token**: Sequential, character-level
   - **Tag**: Discrete coefficient bins

4. **Latent space smoothness**
   - Is there a smooth path in latent space from diffusion → wave?
   - Can we linearly interpolate latent vectors?

---

## Expected Results

### Hypothesis
- **Grammar**: Better continuation due to structural encoding
  - Grammar rules enforce valid PDE syntax
  - Coefficient changes are explicit in production sequence

- **Token**: May struggle with numeric interpolation
  - Character-level encoding sensitive to coefficient digits
  - No explicit structure for coefficient continuation

### Success Metrics
- **Endpoints**: >95% exact match (sanity check)
- **Middle τ**: Compare exact match rates:
  - Grammar: ?% (hypothesis: higher)
  - Token: ?% (hypothesis: lower)

---

## Files Overview

### Generator Files
- `generator.py` - Main generator class with `generate_telegrapher_bridge()` method
- `pde_families.py` - Telegraph family definition

### Example Scripts
- `examples/generate_telegrapher_bridge.py` - Generate bridge dataset
- `examples/split_telegrapher_data.py` - Split train/test

### Output Files
- `telegrapher_bridge_default.csv` - Full dataset
- `telegrapher_train_endpoints.csv` - Training (endpoints)
- `telegrapher_test_middle.csv` - Testing (continuation)

---

## Tips

### Increase Tau Resolution
```python
bridge = gen.generate_telegrapher_bridge(
    tau_small=[0.01, 0.02, 0.03, 0.05, 0.08, 0.1],  # 6 points
    tau_mid=[0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0],  # 9 points
    tau_large=[4.0, 5.0, 8.0, 10.0, 15.0],  # 5 points
)
# Total: 20 PDEs (11 train, 9 test)
```

### Multiple c² Values
```python
for c_sq in [0.5, 1.0, 2.0, 5.0]:
    bridge = gen.generate_telegrapher_bridge(c_sq=c_sq)
    gen.save_dataset(bridge, f'telegrapher_c{c_sq}.csv')
```

### Verify Tau→PDE Mapping
```python
import csv

with open('telegrapher_bridge.csv') as f:
    for row in csv.DictReader(f):
        tau = float(row['tau'])
        a = round(1.0 / tau, 3)
        expected = f"dtt(u) + {a}*dt(u) - 1.0*dxx(u) = 0"
        assert row['pde'] == expected
        print(f"✓ τ={tau} → a={a} → {row['pde']}")
```

---

## Citation

If you use this dataset for research, please cite:

```bibtex
@misc{telegrapher_bridge_2025,
  title={Telegrapher Bridge Dataset: Diffusion-Wave Continuation for Symbolic PDE Encoding},
  author={Your Name},
  year={2025},
  note={Dataset for testing symbolic continuation in VAE latent spaces}
}
```

---

**Questions?** See `ENCODING_DECODING_GUIDE.md` for encoding/decoding details.
