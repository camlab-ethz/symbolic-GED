
# ACFG‑PDE Toolkit

A modular toolkit to **generate**, **validate**, and **parse** PDE expressions with an **Attributed CFG**.

## Features
- Unambiguous CFG (explicit operator application) → unique parse.
- Knuth‑style attributes (inherited/synthesized) with **family profiles** (heat, wave, Poisson, Allen–Cahn).
- Structural guards (dimension/order/linearity/isotropy/sign constraints).
- Canonicalization + hashing for **uniqueness**.
- CFG‑only parser → **production‑ID sequences** and one‑hot matrices.

## Install (editable)
```bash
pip install -e .
```
Ensure your `terminals.py` is importable (same repo root or on PYTHONPATH).

## Quick start
```python
from acfg_toolkit import preset_parabolic, generate_unique, canon_str, analyze, parse_rule_ids, rule_ids_to_onehot
from acfg_toolkit import profile_heat, check_family

inh = preset_parabolic(dim=2)
asts = generate_unique(inh, n=5)
for ast in asts:
    s = canon_str(ast)               # canonical PDE string
    syn = analyze(ast)               # synthesized attributes
    ids = parse_rule_ids(s)          # production IDs
    onehot, L = rule_ids_to_onehot(ids, L_max=128)
    ok = check_family(ast, 2, profile_heat(2))  # semantic validation
```

## Families
Built‑ins: **heat**, **wave**, **poisson**, **allen_cahn**.
Add more by creating a `FamilyProfile` and adding rules to `check_family`.

## Research vs Production
- **Research**: use **hard guards** in family profiles to generate clean datasets; log rejections.
- **Production**: add soft scoring/prior sampling to broaden diversity.

## Extending operators
- Add to your `terminals.py` (e.g., `dzzz(u)`, `dxxy(u)`), then extend parser recognizers in `parser.py` and availability in `terminals_adapter.py`.
- For composite atoms (e.g., higher‑dim `|∇u|^2`), add constructors in `generator.py` and recognizers in `parser.py`.

## Tests
```bash
pytest -q
```


## NLTK grammar + CSV to one-hot
- `acfg_toolkit/grammar_nltk.py`: NLTK `CFG` grammar + `get_mask` utility.
- `acfg_toolkit/csv_to_onehot.py`: tokenizes expressions, parses with NLTK, saves HDF5 one-hot.

### Usage
```bash
python -m acfg_toolkit.csv_to_onehot /path/to/data.csv operator_L expressions_oh.h5 125
```
