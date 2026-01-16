---
marp : true
---

# Symbolic-GED: Grammar vs Character VAE for PDE Representation Learning

---

## 1. Project Goal

Learn structured latent representations of symbolic Partial Differential Equations (PDEs) using Variational Autoencoders.

**Core Question:** Does encoding PDEs with a context-free grammar improve latent space quality compared to character-level tokenization?

**Approach:** Train and compare two VAE variants:
- **Grammar VAE**: Encodes PDEs as production rule sequences from a CFG
- **Character VAE**: Encodes PDEs as character sequences (Lample & Charton style)

---

## 2. Dataset: 16 PDE Families

We generate a synthetic dataset of **48,000 unique PDEs** from 16 canonical families spanning different physics types.

### Classification by Physics Type

| Type | Families | Characteristics |
|------|----------|-----------------|
| **Parabolic** | Heat, Allen-Cahn, Fisher-KPP, Cahn-Hilliard, Kuramoto-Sivashinsky, Reaction-Diffusion | 1st-order in time, diffusion-dominated |
| **Hyperbolic** | Wave, Telegraph, Beam/Plate, Sine-Gordon, Advection | 2nd-order in time (or 1st-order transport), wave-like |
| **Elliptic** | Poisson, Biharmonic | Steady-state, no time derivative |
| **Dispersive** | KdV, Airy | Odd-order spatial derivatives |
| **Mixed** | Burgers | Combines advection + diffusion |

---

### Dataset Properties

- **3,000 PDEs per family** (balanced)
- **Dimensions:** 1D, 2D, 3D (balanced where applicable)
- **Coefficients:** Randomly sampled, 3 decimal precision
- **Format:** Operator-only strings (e.g., `dt(u) - 1.5*dxx(u)`)
- **Train/Val/Test Split:** 33,600 / 7,200 / 7,200 (70% / 15% / 15%)

---

### Example PDEs by Family

| Family | Example |
|--------|---------|
| Heat | `dt(u) - 1.935*dxx(u)` |
| Wave | `dtt(u) - 2.5*dxx(u) - 2.5*dyy(u)` |
| Burgers | `dt(u) + u*dx(u) - 0.1*dxx(u)` |
| KdV | `dt(u) + u*dx(u) + 0.022*dxxx(u)` |
| Allen-Cahn | `dt(u) - 0.01*dxx(u) - u + u^3` |
| Sine-Gordon | `dtt(u) - 1.0*dxx(u) + 0.5*sin(u)` |
| Telegraph | `dtt(u) + 1.188*dt(u) - 0.126*dxx(u)` |

---

## 3. Two Tokenization Strategies

### Grammar-Based Tokenization

Encodes PDEs as **production rule sequences** from a context-free grammar.

**How it works:**
1. Parse PDE string into leftmost derivation
2. Each production rule has a unique ID
3. Sequence = list of production IDs applied in order

**Grammar structure (56 rules):**
```
PDE   → PROD SUM_T
SUM   → PROD SUM_T  
SUM_T → '+' PROD SUM_T | '-' PROD SUM_T | ε
ATOM  → dt(u) | dxx(u) | u^3 | NUM | ...
NUM   → DIGIT DIGITS_REST   ← digit-by-digit encoding
```

**Key property:** Numbers are encoded digit-by-digit, so `1.5` and `1.6` produce different sequences.

| Property | Value |
|----------|-------|
| Vocabulary size | 56 productions |
| Max sequence length | 114 |

---

### Character-Based Tokenization

Encodes PDEs as **character sequences** with a learned vocabulary (Lample & Charton style).

| Property | Value |
|----------|-------|
| Vocabulary size | 82 tokens |
| Max sequence length | 62 |

**Advantage:** Shorter sequences, simpler implementation.

---

## 4. VAE Architecture

### Encoder
- Multi-scale 1D convolutions (kernel sizes 3, 5, 7)
- Residual connections
- Global average pooling → μ, log(σ²)

### Decoder  
- 3-layer GRU
- Learnable start embedding
- Output: logits over vocabulary

### Latent Space
- **Dimension:** z = 26
- **Prior:** Standard Gaussian N(0, I)

### Training Objective
```
Loss = Reconstruction + β × KL(q(z|x) || p(z))
```

Where:
- Reconstruction = masked cross-entropy over valid positions
- KL uses **free bits = 0.5** to prevent posterior collapse

---

## 5. The β Contrast: Two Training Regimes

We train each tokenization with two β values to study the reconstruction-regularization tradeoff:

| Regime | β Value | Effect |
|--------|---------|--------|
| **Low β** | 2×10⁻⁴ | Prioritizes reconstruction accuracy |
| **High β** | 0.01 | Stronger latent regularization |

This gives us **4 model variants**:

| Model | Tokenization | β | Seq Accuracy (Val)* |
|-------|--------------|---|---------------------|
| `grammar_beta2e4` | Grammar | 2×10⁻⁴ | 99.69% |
| `grammar_beta1e2` | Grammar | 0.01 | 8.87% |
| `char_beta2e4` | Character | 2×10⁻⁴ | 99.76% |
| `char_beta1e2` | Character | 0.01 | 0.57% |

***Seq Accuracy (Val):** Percentage of validation set PDEs where the decoded sequence exactly matches the input sequence (token-by-token). This is the metric used to select the best checkpoint during training. High β models sacrifice exact reconstruction for better latent structure.*

---

### Understanding Low Reconstruction Accuracy

**Key Question:** When seq_acc drops to 8.87% or 0.57%, what actually goes wrong?

There are **three types of reconstruction errors**:

| Error Type | Description | Example |
|------------|-------------|---------|
| **Coefficient Change** | Same structure, different numbers | Wave `dtt(u) - 2.5*dxx(u)` → `dtt(u) - 3.1*dxx(u)` |
| **Family Change** | Valid PDE, but different family | Wave `dtt(u) - 2.5*dxx(u)` → Heat `dt(u) - 1.9*dxx(u)` |
| **Corruption** | Invalid/garbage output | Wave → `dtt(u) - 2.5*dxx(u + ` |

**Why does this matter?**
- **Coefficient changes** are acceptable for representation learning (structure preserved)
- **Family changes** indicate the model is generalizing across families (could be good or bad)
- **Corruption** indicates the model failed completely

---

### Reconstruction Quality Analysis (Validated)

We decoded the full validation set (7,200 PDEs) with each model and classified each output:

| Category | Definition | Example |
|----------|------------|---------|
| **Exact Match** | Decoded PDE = Input PDE (after whitespace normalization) | `dt(u) - 1.5*dxx(u)` → `dt(u) - 1.5*dxx(u)` ✓ |
| **Structure Preserved** | Same family, same structure, only coefficient values differ | `dt(u) - 1.5*dxx(u)` → `dt(u) - 2.3*dxx(u)` ✓ |
| **Structure Changed** | Classifier labels same family, but structure actually differs (wrong!) | `u^2` → `u^22` (labeled fisher_kpp, but physics is wrong) ✗ |
| **Family Changed** | Classifier labels as different family | Heat → Wave ✗ |
| **Invalid** | Syntax error, missing arguments, or malformed expression | `sin(u)` → `sin()` ✗ |

---

### Results: Reconstruction Quality on Validation Set

| Model | Seq Acc | Exact Match | Struct Preserved | Struct Changed | Family Changed | Invalid | **Family Preserved** |
|-------|---------|-------------|------------------|----------------|----------------|---------|---------------------|
| `grammar_beta2e4` | 99.7% | **99.81%** | 0.15% | 0.04% | 0.0% | **0.0%** | **99.96%** |
| `grammar_beta1e2` | 8.87% | 12.93% | **83.75%** | 2.86% | 0.46% | **0.0%** | **96.68%** |
| `char_beta2e4` | 99.8% | 93.58% | 0.17% | 0.0% | 0.0% | 6.25% | 100%* |
| `char_beta1e2` | 0.57% | 0.44% | **92.94%** | 0.25% | 0.03% | 6.33% | 99.70%* |

*\*Family Preserved = (Exact + Struct Preserved) / Valid. "Struct Changed" is NOT counted as family preserved because the physics is wrong (e.g., `u^2` → `u^22`).*

---

### Key Insights from Reconstruction Quality

1. **Grammar VAE has 0% invalid outputs** - constrained decoding guarantees syntactic validity

2. **Grammar's 8.87% seq_acc actually means 96.7% family preservation:**
   - 12.9% exact match
   - 83.8% structure preserved (same family, different coefficients)
   - Only 2.9% structure changed + 0.5% family changed
   - The low seq_acc is because coefficients rarely match exactly

3. **Character VAE has 6.25% invalid outputs** - mostly `sin(u)` → `sin()` failures:
   - The model struggles with function arguments
   - Grammar constraints prevent this failure mode

4. **Both high-β models preserve family structure well:**
   - Grammar: 96.7% family preserved (of 100% valid)
   - Character: 99.7% family preserved (of 93.7% valid)

---

### Why This Matters

The low seq_acc numbers (8.87%, 0.57%) were misleading. What actually happens:
- **Not corruption** - the outputs are valid PDEs
- **Not random** - they're in the same family 96-99% of the time
- **Just coefficient variation** - structure is preserved, only numbers change

This validates that high-β models learn meaningful latent structure even when exact reconstruction fails

---

## 6. Constrained Decoding (Grammar VAE Only)

**Problem:** Unconstrained decoding from VAE often produces invalid sequences.

**Solution:** Grammar-constrained decoding using a stack:

```
1. Initialize stack with start symbol [PDE]
2. For each timestep:
   a. Pop leftmost nonterminal from stack
   b. Get valid productions for that nonterminal
   c. Mask decoder logits (invalid → -∞)
   d. Select production (greedy argmax)
   e. Push RHS symbols back to stack
3. Repeat until stack empty or max length
```

**Result:** Guarantees syntactically valid PDE strings.

---

## 6.5. What Makes a PDE "Valid"?

A decoded string is considered **valid** if it satisfies all of the following criteria:

### Validity Criteria

1. **Syntactically correct**: Parseable mathematical expression
   - Balanced parentheses
   - No trailing operators
   - No empty function calls

2. **Contains spatial derivatives**: Must have at least one dx, dy, dz, dxx, dyy, etc.
   - PDEs without spatial derivatives are ODEs, not PDEs
   - Example: `dt(u) + u = 0` is rejected (no spatial derivative)

3. **Well-formed operators**: All functions must have proper arguments
   - `sin(u)` is valid
   - `sin()` is **invalid** (missing argument)

---

### Examples of VALID PDEs

| Decoded PDE | Why Valid |
|-------------|-----------|
| `dt(u) - 1.696*dxx(u)` | Has temporal + spatial derivative, proper syntax |
| `dtt(u) + 0.938*dt(u) - 1.772*dxx(u)` | Telegraph equation, all terms well-formed |
| `dt(u) + u*dx(u) - 1.661*dxx(u)` | Burgers equation with nonlinear term |
| `dxx(u) + dyy(u) - 3.565` | Poisson equation (steady-state), has spatial derivs |

### Examples of INVALID PDEs

| Decoded PDE | Why Invalid | Error Type |
|-------------|-------------|------------|
| `dt(u) - 4.035*dxx(u) + 4.956*sin()` | `sin()` has no argument | **Syntax error** |
| `dtt(u) - 4.035*dxx(u) - 3.986*u + ` | Trailing `+` operator | **Incomplete expression** |
| `dt(u) + u*u - u` | No spatial derivative (only dt) | **ODE, not PDE** |
| `dt(u) + dxx(u) - ` | Trailing `-` operator | **Incomplete expression** |
| `dtt(u) - 3.379*dxx(u) + 2.87*sin()` | `sin()` missing argument | **Syntax error** |

---

### Why Grammar VAE Has Higher Validity

The Grammar VAE uses **constrained decoding**:
- At each step, only grammatically valid productions are allowed
- The decoder can't generate `sin()` without an argument
- The decoder can't leave trailing operators

Character VAE has **no such constraints**:
- Decoder freely generates characters
- Can produce any sequence, including malformed ones
- Must learn validity implicitly from training data

---

## 7. Evaluation: Prior Sampling with Shared Z

**Experiment:** Sample the same 20,000 latent vectors Z ~ N(0, I) and decode with all 4 models.

**Why shared Z?** Fair comparison—all models decode the exact same latent points.

### Results (n=20,000, seed=42)

| Model | Valid Count | Valid % | Unique Valid | Unique % | Unique Signatures |
|-------|-------------|---------|--------------|----------|-------------------|
| Grammar β=2e-4 | 19,356 | **96.78%** | 18,909 | 97.69% | 706 |
| Grammar β=0.01 | 19,924 | **99.62%** | 18,071 | 90.70% | 193 |
| Character β=2e-4 | 12,903 | 64.52% | 12,391 | 96.03% | 291 |
| Character β=0.01 | 16,926 | 84.63% | 10,944 | 64.66% | 188 |

*Note: Validity requires at least one spatial derivative (dx, dy, dz, or higher order). PDEs with only temporal derivatives (like `dt(u)` or `dtt(u)` alone) are rejected as invalid ODEs.*

---

### Key Observations from Prior Sampling

1. **Grammar VAE achieves ~97-99% validity** vs 64-84% for Character VAE
   - Constrained decoding ensures grammatical correctness
   - Grammar produces syntactically valid PDE structures

2. **Character VAE validity improves with high β**
   - β=0.01: 84.63% validity (more regularized latent space)
   - β=2e-4: 64.52% validity (less constrained)
   - More regularized latent space → more "safe" decodings

3. **Grammar VAE shows slight validity decrease with low β**
   - β=2e-4: 96.78% (more exploration, some edge cases)
   - β=0.01: 99.62% (tighter latent space, safer outputs)

4. **Diversity vs Validity tradeoff**
   - Low β: More unique signatures (706 vs 193 for Grammar)
   - High β: Fewer unique outputs but higher validity

---

## 8. Evaluation: Latent Space Clustering

**Question:** Does the latent space organize PDEs by physics properties?

### Metrics Explained

| Metric | Measures | Range | Best |
|--------|----------|-------|------|
| **NMI** (Normalized Mutual Information) | Overlap between k-means clusters and true labels | 0-1 | Higher |
| **ARI** (Adjusted Rand Index) | Cluster agreement adjusted for chance | -1 to 1 | Higher |
| **Purity** | Fraction of samples in majority class per cluster | 0-1 | Higher |
| **Silhouette** | How well-separated clusters are in latent space | -1 to 1 | Higher |

---

### Clustering Results on Test Set (n=7,200)

#### Clustering by PDE Family (16 classes) - **Most Important**

| Model | NMI | ARI | Purity | Silhouette |
|-------|-----|-----|--------|------------|
| Grammar β=2e-4 | 0.557 | 0.363 | 0.495 | 0.057 |
| Grammar β=0.01 | 0.463 | 0.243 | 0.391 | 0.182 |
| **Character β=2e-4** | **0.645** | **0.462** | **0.591** | 0.061 |
| Character β=0.01 | 0.460 | 0.212 | 0.369 | 0.259 |

**Winner: Character β=2e-4** for family clustering (NMI=0.645)

---

### Clustering by Other Physics Properties

**Spatial Order (4 classes: 1st, 2nd, 3rd, 4th)**

| Model | NMI | ARI | Purity |
|-------|-----|-----|--------|
| Grammar β=2e-4 | 0.335 | 0.226 | 0.678 |
| Grammar β=0.01 | 0.099 | 0.075 | 0.635 |
| **Character β=2e-4** | **0.386** | **0.259** | **0.688** |
| Character β=0.01 | 0.113 | 0.067 | 0.563 |

**Temporal Order (3 classes: steady, 1st, 2nd)**

| Model | NMI | ARI | Purity |
|-------|-----|-----|--------|
| Grammar β=2e-4 | 0.190 | 0.083 | 0.625 |
| **Grammar β=0.01** | 0.170 | **0.139** | **0.694** |
| Character β=2e-4 | 0.124 | 0.106 | 0.640 |
| Character β=0.01 | 0.085 | 0.058 | 0.625 |

**Dimension (3 classes: 1D, 2D, 3D)**

| Model | NMI | ARI | Purity |
|-------|-----|-----|--------|
| **Character β=2e-4** | **0.218** | **0.198** | **0.622** |
| Grammar β=2e-4 | 0.096 | 0.117 | 0.532 |
| Grammar β=0.01 | 0.046 | 0.046 | 0.510 |
| Character β=0.01 | 0.022 | 0.010 | 0.501 |

---

### Clustering Summary

| Property | Best Model | NMI |
|----------|------------|-----|
| **Family** | Character β=2e-4 | 0.645 |
| **Spatial Order** | Character β=2e-4 | 0.386 |
| **Temporal Order** | Grammar β=2e-4 | 0.190 |
| **Dimension** | Character β=2e-4 | 0.218 |
| **Linearity** | Character β=0.01 | 0.048 |

**Key Finding:** Character VAE with low β shows best clustering by physics properties, suggesting it learns more physics-meaningful representations despite lower validity.

---

## 9. Evaluation: t-SNE Visualization

**Purpose:** Visualize 26-dimensional latent space in 2D, colored by PDE family.

### t-SNE Plots (Placeholders)

#### Grammar VAE β=2e-4
```
[INSERT: tsne_grammar_beta2e4_family.png]
```
*Expected: Clusters for each of 16 families, some overlap between related PDEs*

#### Grammar VAE β=0.01
```
[INSERT: tsne_grammar_beta1e2_family.png]
```
*Expected: More compact clusters due to higher regularization*

#### Character VAE β=2e-4
```
[INSERT: tsne_token_beta2e4_family.png]
```
*Expected: Best separation between families (highest NMI)*

---

#### Character VAE β=0.01
```
[INSERT: tsne_token_beta1e2_family.png]
```
*Expected: Compact but less diverse clusters*

### What to Look For

- **Clear separation** between families → good latent organization
- **Overlapping clusters** → families share latent features (e.g., Heat/Allen-Cahn)
- **Gradient between related families** (e.g., Heat ↔ Wave via Telegraph)
- **Dimension/order organization** within family clusters

---

## 10. Evaluation: Latent Interpolation

**Experiment:** Linear interpolation between two PDEs in latent space.

```
z(α) = (1-α)·z₁ + α·z₂,   α ∈ [0, 0.05, 0.1, ..., 1.0]
```

**Decode at each of 21 interpolation steps and measure:**
- Validity rate (% of steps that decode to valid PDEs)
- Family transition (when does family change?)
- Coefficient evolution (smooth or abrupt?)

---

### Example 1: Telegraph → Wave (Damped → Undamped Hyperbolic)

**Grammar VAE β=2e-4** (100% validity along path)

| α | Decoded PDE | Family |
|---|-------------|--------|
| 0.0 | `dtt(u)+1.188*dt(u)-0.126*dxx(u)` | Telegraph |
| 0.1 | `dtt(u)+1.188*dt(u)-0.126*dxx(u)` | Telegraph |
| 0.2 | `dtt(u)+0.488*dt(u)-0.126*dxx(u)` | Telegraph |
| 0.3 | `dtt(u)+0.468*dt(u)-1.126*dxx(u)` | Telegraph |
| 0.4 | `dtt(u)+0.466*dt(u)-1.126*dxx(u)` | Telegraph |
| 0.5 | `dtt(u)+0.466*dt(u)-1.026*dxx(u)` | Telegraph |
| **0.55** | `dtt(u)+1.126*dxx(u)-1.026*dyy(u)` | **Wave** ← Transition |
| 0.6 | `dtt(u)-1.026*dxx(u)` | Wave |
| 0.7 | `dtt(u)-1.035*dxx(u)` | Wave |
| 0.8 | `dtt(u)-4.035*dxx(u)` | Wave |
| 1.0 | `dtt(u)-4.035*dxx(u)` | Wave |

**Key Observation:** The `dt(u)` damping term smoothly decreases until it disappears at α≈0.55, where the PDE transitions from Telegraph to Wave. Grammar VAE maintains 100% validity throughout.

---

### Same Interpolation: Character VAE Comparison

**Character VAE β=2e-4** (57% validity)

| α | Decoded PDE | Family | Valid |
|---|-------------|--------|-------|
| 0.0 | `dt(u)+2.126*dx(u)+0.968*dy(u)+4.942*dz(u)` | Advection | ✓ (wrong start!) |
| 0.2 | `dt(u)-0.128*dx(u)+0.268*dy(u)+4.942*dz(u)` | Advection | ✓ |
| 0.4 | `dt(u)-0.128*dx(u)-0.268*dy(u)+4.946*dz(u)` | Advection | ✓ |
| 0.5 | `dt(u)-0.128*dxx(u)-0.298*dy(u)+0.646*dz(u)` | Heat | ✓ |
| 0.6 | `dt(u)-4.035*dxx(u)-3.023*dyy(u)+4.956*sin()` | - | **✗ Invalid** |
| 0.7 | `dtt(u)-4.035*dxx(u)-3.083*dyy(u)+0.956*sin()` | - | **✗ Invalid** |
| 0.8 | `dtt(u)-4.035*dxx(u)-3.686*u+u` | Wave | ✓ |
| 0.9 | `dtt(u)-4.035*dxx(u)-0.986*sin()+` | - | **✗ Invalid** |
| 1.0 | `dtt(u)-4.035*dxx(u)-0.986*sin()+` | - | **✗ Invalid** |

**Problems with Character VAE:**
- Starts at wrong family (Advection instead of Telegraph)
- Produces invalid `sin()` outputs in the middle
- Never reaches the correct target (Wave with clean structure)

---

### Character VAE β=0.01 (33% validity)

| α | Decoded PDE | Valid |
|---|-------------|-------|
| 0.0 | `dtt(u)+2.113*dt(u)-3.307*dxx(u)-1.807*dyy(u)-1.127*dzz(u)` | ✓ Telegraph |
| 0.3 | `dtt(u)+1.113*dt(u)-3.306*dxx(u)-3.246*dyy(u)-4.206*dzz(u)` | ✓ Telegraph |
| 0.35 | `dtt(u)-1.113*dt(u)-3.306*dxx(u)-3.226*dyy(u)-2.207*sin()` | **✗ Invalid** |
| 0.5 | `dtt(u)-3.327*dxx(u)-3.207*dyy(u)-3.207*dzz(u)+2.558*sin()` | **✗ Invalid** |
| 0.7 | `dtt(u)-3.879*dxx(u)-3.179*dyy(u)-3.189*dzz(u)+2.855*sin()` | **✗ Invalid** |
| 1.0 | `dtt(u)-0.881*dxx(u)-0.871*dyy(u)-1.689*dzz(u)+1.31*sin()` | **✗ Invalid** |

**Character β=0.01 fails badly:** After α=0.3, almost all outputs have malformed `sin()` - the model can't navigate between families smoothly.

---

### Example 2: Heat → Allen-Cahn (Linear → Nonlinear Parabolic)

**Grammar VAE β=2e-4** (100% validity along path)

| α | Decoded PDE | Family |
|---|-------------|--------|
| 0.0 | `dt(u)-1.935*dxx(u)` | Heat |
| 0.2 | `dt(u)-1.936*dxx(u)` | Heat |
| 0.4 | `dt(u)-1.968*dxx(u)` | Heat |
| **0.45** | `dt(u)-1.968*dxx(u)-u+u^3` | **Allen-Cahn** ← Transition |
| 0.6 | `dt(u)-1.268*dxx(u)-u+u^3` | Allen-Cahn |
| 0.8 | `dt(u)-1.254*dxx(u)-u+u^3` | Allen-Cahn |
| 1.0 | `dt(u)-1.251*dxx(u)-u+u^3` | Allen-Cahn |

**Key Observation:** At α≈0.45, the nonlinear terms `-u+u^3` suddenly appear, marking the transition from linear Heat to nonlinear Allen-Cahn. This shows the model learns the structural difference between linear and nonlinear PDEs.

---

### Interpolation Validity Summary

| Pair | Grammar β=2e-4 | Grammar β=0.01 | Character β=2e-4 | Character β=0.01 |
|------|----------------|----------------|--------------|--------------|
| Telegraph → Wave | **100%** | **100%** | 57% | 33% |
| Heat → Allen-Cahn | **100%** | **100%** | ~80% | ~50% |
| Wave → Beam/Plate | **100%** | **100%** | ~65% | ~40% |
| Wave → Heat | **100%** | **100%** | ~70% | ~50% |

**Key Conclusions:**
1. **Grammar VAE: 100% validity** on all interpolation paths (constrained decoding)
2. **Character VAE: 33-80% validity** - often produces `sin()` errors mid-interpolation
3. **Character VAE may start at wrong family** (e.g., Advection instead of Telegraph)
4. **Grammar interpolations are smooth** - coefficients evolve gradually
5. **Character interpolations are chaotic** - structure jumps unpredictably

---

## 11. Summary: Grammar vs Character VAE

### Quantitative Comparison

| Aspect | Grammar VAE (β=2e-4) | Character VAE (β=2e-4) | Winner |
|--------|----------------------|--------------------| -------|
| **Prior Sampling Validity** | 96.78% | 64.52% | Grammar |
| **Unique Signatures** | 706 | 291 | Grammar |
| **Family NMI** | 0.557 | **0.645** | Character |
| **Spatial Order NMI** | 0.335 | **0.386** | Character |
| **Interpolation Validity** | **100%** | ~70% | Grammar |
| **Sequence Length** | 114 | **62** | Character |

---

### Key Takeaways

1. **Grammar constraints ensure validity**: 97-99% valid outputs vs 64-84% for Character VAE
   - Constrained decoding is the primary driver of high validity
   - Critical for downstream applications that require valid PDEs

2. **Character VAE learns better physics structure**: Higher clustering metrics (NMI=0.645 vs 0.557)
   - Better separation of families in latent space
   - Shorter sequences may force more compact representations

3. **β tradeoff**: 
   - Low β → better reconstruction, more diversity, worse validity
   - High β → smoother latent space, less diversity, better validity

4. **Interpolation reveals latent structure**:
   - Grammar VAE: 100% valid interpolations with smooth transitions
   - Character VAE: Invalid intermediates, abrupt jumps

5. **Recommendation**:
   - Use **Grammar VAE** when validity is critical (generation, downstream tasks)
   - Use **Character VAE** for representation learning (embedding, similarity)

---

## 12. Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA GENERATION                          │
│  16 PDE families → 48,000 unique PDEs → train/val/test     │
│  Split: 33,600 / 7,200 / 7,200 (stratified by family)      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    TOKENIZATION                             │
│  Grammar: PDE → production IDs (P=56, length 114)          │
│  Character: PDE → character IDs (V=82, length 62)          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    VAE TRAINING                             │
│  4 variants: {grammar, token} × {β=2e-4, β=0.01}           │
│  Encoder: Conv1D + ResNet → μ, logσ² (z_dim=26)            │
│  Decoder: GRU (3 layers) → logits                          │
│  Training: 600-1000 epochs, batch=256, lr=0.001            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    EVALUATION                               │
│  • Prior sampling (n=20,000, shared Z, validity/diversity) │
│  • Clustering metrics (NMI, ARI, Purity by physics labels) │
│  • t-SNE visualization (26D → 2D, colored by family)       │
│  • Latent interpolation (physics-motivated pairs)          │
└─────────────────────────────────────────────────────────────┘
```

---

## 13. Extension: From Representation Learning to Verifiable Discovery

---

## 13.1 The Key Insight: Forward vs Inverse Problems

### The Forward Problem (Easy)
Given a PDE operator L and a solution u, compute the forcing term f:

```
L, u  →  f = L(u)
```

This is **easy** because:
- Symbolic differentiation is deterministic
- SymPy computes exact derivatives
- No ambiguity: one L + one u = exactly one f

### The Inverse Problem (Hard)
Given observations of u and f, discover the operator L:

```
u, f  →  L  such that L(u) = f
```

This is **hard** because:
- Infinitely many L could explain finite observations
- Noise in real data obscures the true operator
- Must search a vast space of symbolic expressions
- **Verification is non-trivial**: How do we know L is correct?

---

## 13.2 Why This Matters: Governing Equation Discovery (GED)

**Scientific Machine Learning Goal:** Automatically discover PDEs from data.

**Current approaches (SINDy, PDE-Net, symbolic regression):**
- Take noisy measurements of u(x,t)
- Estimate derivatives numerically
- Fit candidate operators

**The Problem:** How do we evaluate if a discovered L is correct?

### The Verification Gap

| What we have | What we need |
|--------------|--------------|
| Discovered operator L̂ | Ground truth operator L* |
| Noisy observations û | True solution u |
| Estimated residual ∥L̂(û) - f̂∥ | Symbolic equivalence L̂ ≡ L* |

**Key Insight:** Low residual doesn't mean correct discovery!
- L̂ might overfit to noise
- L̂ might be a different operator that happens to fit the data
- We need **symbolic verification**, not just numerical agreement

---

## 13.3 Our Solution: Manufactured Solutions for Verifiable GED

### The Manufactured Solutions Method

Instead of starting with a PDE and solving it (hard), we:

1. **Pick any smooth function u** (the "manufactured solution")
2. **Apply the operator L symbolically** to get f = L(u)
3. **Now we have a verified triplet** (L, u, f) where L(u) = f exactly

```
┌─────────────────────────────────────────────────────────────────┐
│  MANUFACTURED SOLUTIONS PIPELINE                                 │
│                                                                  │
│  Step 1: Sample operator L from dataset                         │
│          L = "dt(u) - 1.935*dxx(u)"  (Heat equation)            │
│                                                                  │
│  Step 2: Sample physically-motivated u                          │
│          u = sin(2πx)·exp(-λt) + 0.3·exp(-α(x-0.5)²)           │
│                                                                  │
│  Step 3: Compute f = L(u) symbolically                          │
│          f = [-2πλ·cos(2πx)·exp(-λt) + ...]  (exact!)           │
│                                                                  │
│  Result: Verified triplet (L, u, f) for benchmarking            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 13.4 Why Manufactured Solutions?

### Advantages for GED Benchmarking

| Property | Benefit |
|----------|---------|
| **Exact f** | No numerical differentiation errors |
| **Known L** | Ground truth for verification |
| **Controllable u** | Can design physically meaningful solutions |
| **Scalable** | Generate millions of (L, u, f) triplets |
| **Diverse** | Cover all 16 PDE families, all dimensions |

### The Verification Protocol

Given a GED algorithm that outputs L̂:

```python
# 1. Algorithm discovers operator from (u, f) observations
L_hat = ged_algorithm(u_samples, f_samples)

# 2. Symbolic verification against ground truth
is_correct = symbolic_equivalent(L_hat, L_true)

# 3. Report discovery accuracy across test set
accuracy = sum(is_correct) / n_test
```

**This is only possible with manufactured solutions** where we know L_true!

---

## 13.5 Dataset Design: Two Tracks for Fair Evaluation

### The Label Leakage Problem

If we sample u to "look like" heat equation solutions for heat operators, a clever model could:
- Recognize heat-like u patterns
- Guess "heat equation" without actually solving the inverse problem

**Solution:** Two evaluation tracks with different u sampling strategies.

---

### Track A: Physics-Guided Sampling (Development/Training)

**Goal:** Generate u that resembles realistic solutions for each PDE family.

**Method:** Bias motif selection towards physics-appropriate behaviors:

| PDE Family | Preferred Motifs | Physical Interpretation |
|------------|------------------|------------------------|
| Heat | M2, M4, M6 | Exponential decay, Gaussian diffusion |
| Wave | M1, M3, M6 | Modal oscillations, traveling packets |
| Burgers | M5, M3, M1 | Shock fronts, advected waves |
| KdV | M3, M1, M6 | Soliton-like packets, dispersive waves |
| Allen-Cahn | M5, M2, M4 | Phase fronts, relaxation dynamics |

**Use case:** Training GED algorithms, development, debugging.

---

### Track B: Shared Prior Sampling (Benchmark/Evaluation)

**Goal:** Eliminate any operator-dependent bias in u sampling.

**Method:** Use identical motif distribution for ALL operators:

```
M1 (modal):     20%
M2 (decay):     10%
M3 (Gabor):     15%
M4 (Gaussian):  15%
M5 (front):     15%
M6 (Fourier):   15%
M7 (separable):  5%
M8 (chirp):      3%
M9 (bump):       2%
```

**Key Property:** The u distribution is independent of the operator family.

**Use case:** Fair benchmarking - models can't exploit u patterns to guess L.

---

## 13.6 The Motif Library: Building Blocks for u

We designed 9 physical motifs (M1-M9) that span canonical solution behaviors:

### M1: Modal Waves
```
u = A·sin(2π(kx - ωt) + φ)     # traveling wave
u = A·sin(2πkx)·cos(2πωt)      # standing wave
```
**Physics:** Eigenmodes of wave/vibration problems.

### M2: Diffusion Decay
```
u = A·sin(2πkx)·exp(-λt)
```
**Physics:** Heat equation eigenmodes with temporal decay.

### M3: Gabor Packet (Localized Wave)
```
u = A·exp(-α(x - x₀ - ct)²)·sin(2πk(x - x₀) - ωt + φ)
```
**Physics:** Wave packet propagating at speed c, used in dispersive PDEs.

---

### M4: Gaussian Load
```
u = A·exp(-α((x - x₀)² + (y - y₀)²))
```
**Physics:** Localized source/forcing, common in Poisson problems.

### M5: Front/Interface (tanh)
```
u = A·tanh((x - ct - x₀)/δ)
```
**Physics:** Traveling fronts in reaction-diffusion, Burgers shocks.

### M6: Multi-scale Fourier
```
u = Σₙ aₙ·sin(2π(kₙx - ωₙt) + φₙ),  where aₙ ~ 1/|kₙ|^p
```
**Physics:** Turbulent/multi-scale fields with spectral decay.

---

### M7: Separable Product
```
u = (1 + a·sin(kₓx))·(1 + b·cos(kᵧy))·(1 + c·cos(ωt))
```
**Physics:** Excites mixed derivatives (∂²/∂x∂y), tests cross-terms.

### M8: Chirp
```
u = A·sin(2π(k₀x + k₁x²) - ωt + φ)
```
**Physics:** Non-stationary frequency, tests robustness to complex spectra.

### M9: Rational Bump
```
u = A / (1 + α((x - x₀)² + (y - y₀)²))
```
**Physics:** Smooth localized feature, alternative to Gaussian.

---

## 13.7 Complexity Levels: From Simple to Challenging

Each u is composed of multiple motifs. Complexity level controls how many:

| Level | Components | Typical ops | Use Case |
|-------|------------|-------------|----------|
| **1** | 1-2 motifs | 17-30 | Easy: algorithm development |
| **2** | 2-4 motifs | 30-65 | Medium: standard benchmark |
| **3** | 3-6 motifs | 50-100+ | Hard: stress testing |

### Example: Heat Equation at Different Complexities

**Level 1 (Simple):**
```
u = 0.216·exp(2.47x)·exp(-3.02x²)·cos(15.6πt)
```

**Level 2 (Medium):**
```
u = 0.29·exp(6.46t)·exp(-5.6t²)·exp(6.07x)·exp(-4.95x²)
  + sin(-7.7πt + 10πx + 4.6)/100
  + sin(-5.4πt + 6πx + 2.9)/50
```

**Level 3 (Complex):**
```
u = -346·sin(10πx)·cos(8.5πt)/211
  + sin(-7.7πt + 10πx + 4.6)/100
  + sin(-5.4πt + 8πx + 4.9)/100
  + sin(-2.3πt + 4πx + 5.6)/40
  + 0.3·tanh((x - 0.5 - 1.2t)/0.08)
  + ...
```

---

## 13.8 The Forcing Term f = L(u)

When we apply the operator L to a complex u, we get a complex f:

### Example: Heat Equation

**Operator:** `L = dt(u) - 1.935·dxx(u)`

**Manufactured u:**
```
u = sin(2πkx)·exp(-λt) + A·exp(-α(x-x₀)²)
```

**Computed f = L(u):**
```
f = -λ·sin(2πkx)·exp(-λt)                    # from dt(u)
  + 1.935·(2πk)²·sin(2πkx)·exp(-λt)          # from -1.935·dxx(u) term 1
  - 1.935·A·(-2α + 4α²(x-x₀)²)·exp(-α(x-x₀)²) # from -1.935·dxx(u) term 2
```

**Key Properties of f:**
- Complex expression (often 100+ operations)
- Exactly satisfies L(u) = f (no numerical error)
- Provides ground truth for GED algorithms

---

## 13.9 Dataset Structure

### What's in the Dataset?

| Component | Description | Count |
|-----------|-------------|-------|
| **PDE Operators (L)** | Symbolic differential operators from 16 families | 48,000 |
| **Solutions (u)** | Manufactured smooth functions | 768,000 |
| **Forcing Terms (f)** | Computed as f = L(u), exact symbolic | 768,000 |
| **Tracks** | A (physics-guided) + B (shared prior) | 2 |
| **Pairs per Operator** | Multiple (u, f) for each L | k = 8 |

### What Each Record Contains

| Field | Type | Example |
|-------|------|---------|
| `track` | string | `"A"` or `"B"` |
| `family` | string | `"heat"`, `"wave"`, `"burgers"`, ... |
| `dim` | int | 1, 2, or 3 |
| `temporal_order` | int | 0 (steady), 1, or 2 |
| `spatial_order` | int | 1, 2, 3, or 4 |
| `operator_str` | string | `"dt(u) - 1.935*dxx(u)"` |
| `u_str_list` | list[string] | k=8 symbolic u expressions |
| `f_str_list` | list[string] | k=8 corresponding f = L(u) |

---

### Dataset Statistics

```
Total Records:     48,000 operators × 2 tracks = 96,000 records
Total (u, f) Pairs: 96,000 records × 8 pairs = 768,000 pairs
Total Triplets:    768,000 verified (L, u, f) triplets

Storage:           ~2-5 GB (JSONL format)
Location:          $SCRATCH/symbolic-GED/datasets/manufactured/
```

### Coverage by PDE Family

| Family | Operators | (u, f) Pairs |
|--------|-----------|--------------|
| Heat | 3,000 | 48,000 |
| Wave | 3,000 | 48,000 |
| Burgers | 3,000 | 48,000 |
| KdV | 3,000 | 48,000 |
| ... (16 families) | ... | ... |
| **Total** | **48,000** | **768,000** |

---

### Output Format (JSONL)

```json
{
  "track": "A",
  "family": "heat",
  "dim": 1,
  "temporal_order": 1,
  "spatial_order": 2,
  "operator_str": "dt(u) - 1.935*dxx(u)",
  "k": 8,
  "u_str_list": [
    "sin(2*pi*x)*exp(-3.14*t) + 0.5*exp(-10*(x-0.3)**2)",
    "0.8*sin(4*pi*x)*cos(2.5*pi*t) + tanh((x-0.5)/0.1)",
    ...
  ],
  "f_str_list": [
    "-3.14*sin(2*pi*x)*exp(-3.14*t) + 76.6*pi**2*sin(2*pi*x)*exp(-3.14*t) + ...",
    ...
  ]
}
```

---

### Concrete Example: Full (L, u, f) Triplet

**Operator L (Heat equation):**
```
L = dt(u) - 1.935·dxx(u)
```

**Manufactured solution u:**
```
u = 0.216·exp(2.47x)·exp(-3.02x²)·cos(15.63πt) 
  + 0.864·exp(2.47x)·exp(-3.02x²)
```

**Computed forcing f = L(u):**
```
f = -15.26x²·exp(2.47x)·exp(-3.02x²)·cos(15.63πt) 
  - 61.04x²·exp(2.47x)·exp(-3.02x²) 
  + 12.50x·exp(2.47x)·exp(-3.02x²)·cos(15.63πt) 
  + 49.99x·exp(2.47x)·exp(-3.02x²) 
  - 3.38π·exp(2.47x)·exp(-3.02x²)·sin(15.63πt) 
  - 0.034·exp(2.47x)·exp(-3.02x²)·cos(15.63πt) 
  - 0.136·exp(2.47x)·exp(-3.02x²)
```

**Observations:**
- u has 2 terms, ~20 operations
- f has 7 terms, ~50 operations  
- f is significantly more complex than u
- f is **exact** — no numerical approximation

---

### Why f is Complex

When we apply L = ∂/∂t - α·∂²/∂x² to u:

1. **Each term in u produces multiple terms in f**
   - exp(ax)·exp(-bx²) → chain rule creates polynomial factors

2. **Derivatives compound**
   - ∂²/∂x² of exp(ax)·exp(-bx²) involves (a - 2bx)² and -2b

3. **No cancellation** (unlike exact eigenfunctions)
   - Manufactured u is not an eigenfunction of L
   - So terms don't simplify to zero

**This complexity is intentional** — it tests whether GED algorithms can discover L from non-trivial observations.

---

## 13.10 Evaluation Protocol for GED Algorithms

### Input to Algorithm
```python
# Give the algorithm k observations
observations = [
    (u_1, f_1),  # u and f as symbolic strings or numerical samples
    (u_2, f_2),
    ...
    (u_k, f_k)
]
```

### Expected Output
```python
# Algorithm discovers operator
L_discovered = "dt(u) - 1.9*dxx(u)"  # hopefully close to ground truth
```

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Exact Match** | L̂ ≡ L symbolically (after simplification) |
| **Family Match** | L̂ is same PDE family as L |
| **Coefficient Error** | ∥coefficients(L̂) - coefficients(L)∥ |
| **Structural Similarity** | Same derivative terms, different coefficients |
| **Residual** | ∥L̂(u) - f∥ on held-out test u |

---

## 13.11 Why This Dataset is Unique

### Comparison with Existing GED Benchmarks

| Dataset | Ground Truth L? | Symbolic u? | Verified f? | Multiple u per L? |
|---------|-----------------|-------------|-------------|-------------------|
| PDE-Net data | ✗ | ✗ | ✗ | ✓ |
| SINDy examples | ✓ (few) | ✗ | ✗ | ✗ |
| DeepONet benchmarks | ✗ | ✗ | ✗ | ✓ |
| **Our Dataset** | **✓ (48k)** | **✓** | **✓** | **✓ (k=8)** |

### What We Provide That Others Don't

1. **Symbolic ground truth** for 48,000 diverse operators
2. **Exact (u, f) pairs** with no numerical error
3. **Two evaluation tracks** to prevent label leakage
4. **Complexity levels** for progressive difficulty
5. **Physics-meaningful u** that looks like real solutions

---

## 13.12 Mathematical Verification of the Pipeline

### We verified that f = L(u) is computed correctly:

**Test 1: Exact eigenfunctions give f = 0**
```python
u = sin(2πx)·exp(-4π²t)   # Heat eigenfunction with α=1
L = dt(u) - 1.0·dxx(u)
f = L(u) = 0  ✓
```

**Test 2: Wave equation standing wave**
```python
u = sin(2πx)·cos(2πt)     # Wave eigenmode with c=1
L = dtt(u) - 1.0·dxx(u)
f = L(u) = 0  ✓
```

**Test 3: Manufactured solution (non-zero f)**
```python
u = x³·exp(-t)
L = dt(u) - 0.5·dxx(u)
f = L(u) = x·(-x² - 3)·exp(-t)

# Manual verification:
dt(u) = -x³·exp(-t)
dxx(u) = 6x·exp(-t)
f = -x³·exp(-t) - 0.5·6x·exp(-t) = x·(-x² - 3)·exp(-t)  ✓
```

---

## 13.13 The Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MANUFACTURED SOLUTIONS PIPELINE                   │
└─────────────────────────────────────────────────────────────────────┘

     48,000 PDE Operators (from VAE dataset)
                    │
                    ▼
    ┌───────────────────────────────────┐
    │  For each operator L:              │
    │                                    │
    │  ┌─────────────┐  ┌─────────────┐ │
    │  │  Track A    │  │  Track B    │ │
    │  │  Physics-   │  │  Shared     │ │
    │  │  guided u   │  │  prior u    │ │
    │  └──────┬──────┘  └──────┬──────┘ │
    │         │                │         │
    │         ▼                ▼         │
    │    Sample k=8 u    Sample k=8 u   │
    │         │                │         │
    │         ▼                ▼         │
    │    Compute f=L(u)  Compute f=L(u) │
    │         │                │         │
    │         ▼                ▼         │
    │    (L, u₁..u₈, f₁..f₈)  (L, u₁..u₈, f₁..f₈) │
    └───────────────────────────────────┘
                    │
                    ▼
         768,000 verified (L, u, f) triplets
                    │
                    ▼
         Saved to $SCRATCH/symbolic-GED/datasets/
```

---


---

## 13.15 Scientific Soundness: Implementation Details

### Continuous Parameters (No Quantization)

All motif parameters are **continuous floats**, not discretized:

| Before (Problematic) | After (Correct) |
|---------------------|-----------------|
| `A = Rational(1.5).limit_denominator(100)` | `A = Float(1.5)` |
| Discretized to ~100 values | True continuous distribution |
| May cause duplicate u | Unique u expressions |

**Why this matters:** Discrete parameters could create spurious patterns that leak operator identity.

---

### Reproducibility Guarantees

**M6 Multi-scale Fourier Fix:**
- Previously: Created internal RNG seeded from amplitude (non-reproducible)
- Now: Uses **only the passed RNG** for all random sampling

```python
# Before (broken)
rng = np.random.default_rng(int(params['A'] * 1e6))

# After (reproducible)  
def M6_multiscale_fourier(params, dim, temporal_order, rng):
    # rng passed explicitly, same seed = same output
```

**Result:** Same seed always produces identical (u, f) pairs.

---

### Canonical Uniqueness

**Problem:** `str(sin(x) + cos(x))` vs `str(cos(x) + sin(x))` may differ.

**Solution:** Canonical printing with lexicographic ordering:

```python
def canonical_print(expr, do_simplify=True):
    canonical = expand(expr)
    if do_simplify:
        canonical = simplify(canonical, ratio=1.5)
    return sstr(canonical, order='lex')  # Deterministic!
```

**Uniqueness tracking:**
- Per-set: No duplicate u within k pairs
- Set-level: No duplicate (u₁...uₖ) sets for same operator

---

### Filter Integration for Track B

**Track B Requirement:** u must be informative for operator identification.

| Filter | Purpose | Track A | Track B |
|--------|---------|---------|---------|
| Complexity | Reject overly complex u/f | ✓ | ✓ |
| Stability | Reject NaN/Inf on grid | ✓ | ✓ |
| Informative | Excites all operator terms | Skip | **Required** |

**Informative filter checks:**
- All derivative terms in L have non-zero response
- Nonlinear terms (u², sin(u)) have sufficient variation
- u is not near-constant

---

### Generation Statistics

Each record includes rejection statistics for transparency:

```json
{
  "meta": {
    "idx": 42,
    "attempts": 156,
    "rejection_stats": {
      "duplicate": 23,
      "complexity": 8,
      "stability": 2,
      "informative": 45,
      "operator_error": 3
    }
  }
}
```

**Typical rejection rates:**
- Track A: ~20% rejected (mostly duplicates)
- Track B: ~50% rejected (informative filter is strict)

## 13.16 Summary: What We Built

### Dataset Specifications

| Property | Value |
|----------|-------|
| **PDE Operators** | 48,000 (16 families × 3,000 each) |
| **Evaluation Tracks** | 2 (A: physics-guided, B: shared prior) |
| **Pairs per Operator** | k = 8 |
| **Total Triplets** | 768,000 |
| **u Complexity** | 3 levels (17-100+ ops) |
| **f Computation** | Exact symbolic (SymPy) |

### Key Contributions

1. **First large-scale verified GED benchmark** with symbolic ground truth
2. **Two-track design** prevents label leakage in evaluation
3. **Physics-motivated motif library** generates meaningful solutions
4. **Scalable pipeline** runs on HPC (SLURM array jobs)
5. **Reproducible** with fixed random seeds

---

## 15. Future Work

1. **Hybrid approach**: Combine Grammar constraints with Character embeddings
2. **Conditional generation**: Generate PDEs with specific properties (dimension, order)
3. **Physics-informed loss**: Add PDE-specific regularization (e.g., symmetry)
4. **Larger grammar**: Extend to more PDE operators and equation types
5. **Downstream tasks**: Use embeddings for PDE classification, similarity search
6. **GED Benchmarking**: Evaluate SINDy, PDE-Net, symbolic regression on our dataset
7. **Noise Robustness**: Add numerical sampling with controlled noise levels

---

## 16. Appendix A: PDE Family Reference

| Family | Equation Form | Temporal | Spatial | Nonlinear |
|--------|---------------|----------|---------|-----------|
| Heat | u_t = k∇²u | 1st | 2nd | No |
| Wave | u_tt = c²∇²u | 2nd | 2nd | No |
| Poisson | ∇²u = f | - | 2nd | No |
| Advection | u_t + v·∇u = 0 | 1st | 1st | No |
| Burgers | u_t + u·u_x = ν·u_xx | 1st | 2nd | Yes |
| KdV | u_t + u·u_x + δ·u_xxx = 0 | 1st | 3rd | Yes |
| Allen-Cahn | u_t = ε²∇²u + u - u³ | 1st | 2nd | Yes |
| Cahn-Hilliard | u_t = -γ∇⁴u + ∇²(u³-u) | 1st | 4th | Yes |
| Fisher-KPP | u_t = D∇²u + r·u(1-u) | 1st | 2nd | Yes |
| Kuramoto-Sivashinsky | u_t + νu_xx + γu_xxxx + αu·u_x = 0 | 1st | 4th | Yes |
| Telegraph | u_tt + a·u_t = b²∇²u | 2nd | 2nd | No |
| Biharmonic | ∇⁴u = f | - | 4th | No |
| Sine-Gordon | u_tt - c²∇²u + β·sin(u) = 0 | 2nd | 2nd | Yes |
| Airy | u_t + α·u_xxx = 0 | 1st | 3rd | No |
| Beam/Plate | u_tt + κ∇⁴u = 0 | 2nd | 4th | No |
| Reaction-Diffusion | u_t - ∇²u ± g·u³ = 0 | 1st | 2nd | Yes |

---

## 16.1 Appendix B: Full Clustering Results (Test Set)

### Grammar VAE β=2e-4

| Label | NMI | ARI | Purity | Silhouette |
|-------|-----|-----|--------|------------|
| Family | 0.557 | 0.363 | 0.495 | 0.057 |
| Type | 0.317 | 0.170 | 0.548 | 0.076 |
| Order | 0.335 | 0.226 | 0.678 | 0.076 |
| Dimension | 0.096 | 0.117 | 0.532 | 0.064 |
| Temporal Order | 0.190 | 0.083 | 0.625 | 0.064 |
| Linearity | 0.022 | 0.031 | 0.588 | 0.077 |

### Grammar VAE β=0.01

| Label | NMI | ARI | Purity | Silhouette |
|-------|-----|-----|--------|------------|
| Family | 0.463 | 0.243 | 0.391 | 0.182 |
| Type | 0.093 | 0.068 | 0.501 | 0.176 |
| Order | 0.099 | 0.075 | 0.635 | 0.176 |
| Dimension | 0.046 | 0.046 | 0.510 | 0.171 |
| Temporal Order | 0.170 | 0.139 | 0.694 | 0.171 |
| Linearity | 0.000 | 0.000 | 0.508 | 0.171 |

---

### Character VAE β=2e-4

| Label | NMI | ARI | Purity | Silhouette |
|-------|-----|-----|--------|------------|
| Family | 0.645 | 0.462 | 0.591 | 0.061 |
| Type | 0.272 | 0.136 | 0.541 | 0.063 |
| Order | 0.386 | 0.259 | 0.688 | 0.063 |
| Dimension | 0.218 | 0.198 | 0.622 | 0.065 |
| Temporal Order | 0.124 | 0.106 | 0.640 | 0.065 |
| Linearity | 0.006 | 0.008 | 0.544 | 0.096 |

### Character VAE β=0.01

| Label | NMI | ARI | Purity | Silhouette |
|-------|-----|-----|--------|------------|
| Family | 0.460 | 0.212 | 0.369 | 0.259 |
| Type | 0.149 | 0.102 | 0.528 | 0.263 |
| Order | 0.113 | 0.067 | 0.563 | 0.263 |
| Dimension | 0.022 | 0.010 | 0.501 | 0.286 |
| Temporal Order | 0.085 | 0.058 | 0.625 | 0.286 |
| Linearity | 0.048 | 0.065 | 0.628 | 0.300 |

---

## 16.2 Appendix C: Interpolation Pairs Tested

| Pair | Physics Transition | Dims Tested |
|------|-------------------|-------------|
| Wave → Heat | Hyperbolic → Parabolic (temporal order 2→1) | 1D, 2D, 3D |
| Telegraph → Wave | Damped → Undamped hyperbolic | 1D, 2D, 3D |
| Heat → Allen-Cahn | Linear → Nonlinear parabolic | 1D, 2D, 3D |
| Wave → Beam/Plate | 2nd → 4th order spatial | 1D, 2D |

All interpolations use 21 steps (α = 0.0, 0.05, 0.1, ..., 1.0).
