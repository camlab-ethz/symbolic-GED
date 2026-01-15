"""Analysis modules for PDE VAE experiments."""

from .physics import (
    PDE_PHYSICS,
    classify_pde_type,
    classify_linearity,
    classify_order,
    classify_spatial_dim,
    classify_temporal_order,
    is_valid_pde,
    parse_pde,
    assign_physics_labels,
)

from .clustering import (
    compute_clustering_metrics,
    compute_all_clustering,
    compute_ood_aware_clustering,
    train_classifier,
    train_ood_aware_classifier,
)

__all__ = [
    # Physics
    'PDE_PHYSICS',
    'classify_pde_type',
    'classify_linearity',
    'classify_order',
    'classify_spatial_dim',
    'classify_temporal_order',
    'is_valid_pde',
    'parse_pde',
    'assign_physics_labels',
    # Clustering
    'compute_clustering_metrics',
    'compute_all_clustering',
    'compute_ood_aware_clustering',
    'train_classifier',
    'train_ood_aware_classifier',
]
