# Differential Equation Triplet Generator

## Quick Start: Creating a Dataset

### 1. Prerequisites
- Install Python dependencies:
```bash
pip install sympy tqdm pyyaml
```

### 2. Dataset Generation Workflow

#### Option 1: Comprehensive Dataset (All Combinations)
```bash
# Edit generate_dataset.sh to set parameters
./generate_dataset.sh
```

#### Option 2: Random Dataset
```bash
# Modify generate_dataset.sh
MODE="random"
NUM_PER_OPERATOR=5  # Generate 5 random examples per operator
```

### 3. Customizing Your Dataset

Edit `generate_dataset.sh` to control generation:
```bash
# Filter specific operators
OPERATORS="diffusion wave"

# Filter dimensions
DIMENSIONS="1 2"

# Filter solution types
SOLUTIONS="sine_cosine exp_decay"
```

## Project Structure

### Key Files
- `generate_triplets.py`: Core dataset generation logic
- `config_dt_helpers.py`: Configuration management
- `generate_dataset.sh`: Execution script
- `config_dataset.yaml`: Operator and solution configurations

## Detailed Configuration Guide

### 1. YAML Configuration (`config_dataset.yaml`)

#### Operator Configuration Example
```yaml
operators:
  diffusion:
    valid_dims: [1]  # Supported dimensions
    coefficient_range: [0.1, 1.0]  # Coefficient randomization
    solution_options:
      - type: exp_decay
      - type: sine_cosine
      - type: gaussian
```

### 2. Bash Script Parameters (`generate_dataset.sh`)

| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| `OUTPUT_FILE` | CSV output path | `"../datasets/generated_triplets.csv"` |
| `MODE` | Generation strategy | `"comprehensive"` or `"random"` |
| `NUM_PER_OPERATOR` | Random examples per operator | `1`, `5`, `10` |
| `OPERATORS` | Specific operators | `"diffusion wave"` |
| `DIMENSIONS` | Spatial dimensions | `"1 2"` |
| `SOLUTIONS` | Solution types | `"sine_cosine exp_decay"` |

## Extending the Tool

### Adding New Operators
1. Update `config_dataset.yaml`
   - Add operator configuration
   - Define dimensions, coefficients
2. Modify `generate_triplets.py`
   - Implement solution generation
   - Add operator computation logic

### Output Format
Generated CSV columns:
- `operator_type`: Differential equation type
- `spatial_dimension`: Spatial dimensions
- `solution_type`: Manufactured solution type
- `manufactured_solution_u`: Symbolic solution
- `operator_L`: Differential operator
- `forcing_term_f`: Computed forcing term

## Advanced Usage

### Programmatic Generation
```python
# Example of generating a single triplet
u_str, L_formatted, f_str = generate_triplet(
    operator_type='diffusion', 
    dims=1, 
    solution_type='exp_decay'
)
```

