"""
Manufactured Solutions Dataset Module

Generates (u, f=L(u)) pairs for PDE operator identification benchmark.

Two tracks:
- Track A: Physics-guided u sampling (operator-conditioned)
- Track B: Shared prior u sampling (operator-agnostic with identifiability filter)

Components:
- motif_library.py: Physical motifs M1-M9 for u generation
- u_sampler.py: Track A and Track B sampling logic
- operator_apply.py: Apply L to u symbolically using SymPy
- filters.py: Identifiability, complexity, and stability filters
- utils.py: Canonical printing, uniqueness tracking
- generate_chunk.py: SLURM-ready chunk generator
"""

from datasets.manufactured.utils import (
    canonical_print, canonical_hash, UniquenessTracker,
    make_grid, evaluate_on_grid
)
from datasets.manufactured.motif_library import (
    sample_motif, boundary_mask, MOTIF_FUNCTIONS
)
from datasets.manufactured.u_sampler import sample_u
from datasets.manufactured.operator_apply import (
    apply_operator, infer_orders_from_operator, parse_operator_tokens
)
from datasets.manufactured.filters import (
    check_complexity, check_stability, check_informative, apply_all_filters
)

__all__ = [
    # Core functions
    'sample_u',
    'apply_operator',
    # Utilities
    'canonical_print',
    'canonical_hash',
    'UniquenessTracker',
    'make_grid',
    'evaluate_on_grid',
    # Motifs
    'sample_motif',
    'boundary_mask',
    'MOTIF_FUNCTIONS',
    # Operator parsing
    'infer_orders_from_operator',
    'parse_operator_tokens',
    # Filters
    'check_complexity',
    'check_stability',
    'check_informative',
    'apply_all_filters',
]
