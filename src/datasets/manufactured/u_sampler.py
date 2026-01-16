"""
u Sampler for Manufactured Solutions

Two tracks:
- Track A: Physics-guided (operator-conditioned) u sampling
- Track B: Shared prior (operator-agnostic) u sampling with identifiability filter
"""

import numpy as np
from typing import Dict, List, Optional
from sympy import Expr, expand, S

from datasets.manufactured.motif_library import (
    sample_motif, boundary_mask, MOTIF_FUNCTIONS
)


# Track B: Fixed motif probabilities (shared across all operators)
TRACK_B_WEIGHTS_TEMPORAL = {
    'M1': 0.20,  # Modal waves
    'M2': 0.10,  # Diffusion decay
    'M3': 0.15,  # Gabor packet
    'M4': 0.15,  # Gaussian load
    'M5': 0.15,  # Front
    'M6': 0.15,  # Multi-scale Fourier
    'M7': 0.05,  # Separable product
    'M8': 0.03,  # Chirp
    'M9': 0.02,  # Rational bump
}

# For steady (temporal_order=0), remove time-dependent biases
TRACK_B_WEIGHTS_STEADY = {
    'M1': 0.25,  # Modal waves (static)
    'M2': 0.15,  # Just modes
    'M4': 0.20,  # Gaussian load
    'M5': 0.15,  # Static front
    'M6': 0.10,  # Multi-scale Fourier
    'M7': 0.08,  # Separable product
    'M8': 0.04,  # Chirp
    'M9': 0.03,  # Rational bump
}

# Track A: Family-specific motif preferences
FAMILY_MOTIF_PREFERENCES = {
    # Hyperbolic / wave-like
    'wave': ['M1', 'M3', 'M6'],
    'telegraph': ['M1', 'M3', 'M2'],
    'sine_gordon': ['M1', 'M3', 'M5'],
    
    # Parabolic / diffusion
    'heat': ['M2', 'M4', 'M6'],
    'allen_cahn': ['M5', 'M2', 'M4'],
    'fisher_kpp': ['M5', 'M2', 'M3'],
    'cahn_hilliard': ['M6', 'M4', 'M7'],
    'kuramoto_sivashinsky': ['M6', 'M3', 'M1'],
    'reaction_diffusion_cubic': ['M5', 'M2', 'M4'],
    
    # Transport / advection
    'advection': ['M1', 'M3', 'M5'],
    'burgers': ['M5', 'M3', 'M1'],
    
    # 4th order / plate
    'beam_plate': ['M6', 'M7', 'M4'],
    'biharmonic': ['M4', 'M6', 'M7'],
    
    # Elliptic (steady)
    'poisson': ['M4', 'M7', 'M6'],
    
    # Dispersive
    'kdv': ['M3', 'M1', 'M6'],
    'airy': ['M3', 'M1', 'M8'],
}


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Normalize weights to sum to 1."""
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def sample_u_track_A(family: str, dim: int, temporal_order: int,
                     spatial_order: int, complexity_level: int,
                     rng: np.random.Generator) -> Expr:
    """
    Track A: Physics-guided u sampling.
    
    Selects motifs biased by family to produce physically meaningful u.
    
    Args:
        family: PDE family name
        dim: Spatial dimension (1 or 2)
        temporal_order: 0 (steady), 1, or 2
        spatial_order: Highest spatial derivative order
        complexity_level: 1 (simple), 2 (moderate), 3 (complex)
        rng: NumPy random generator
    
    Returns:
        SymPy expression for u
    """
    # Get family-specific motif preferences
    preferred_motifs = FAMILY_MOTIF_PREFERENCES.get(
        family, ['M1', 'M4', 'M6']  # Default fallback
    )
    
    # Build weights favoring preferred motifs
    weights = {}
    for motif in MOTIF_FUNCTIONS.keys():
        if motif in preferred_motifs:
            weights[motif] = 0.3  # Higher weight for preferred
        else:
            weights[motif] = 0.05  # Lower weight for others
    
    # Remove time-dependent motifs for steady problems
    if temporal_order == 0:
        # M2, M3, M5 still work as static versions
        pass
    
    weights = _normalize_weights(weights)
    
    # Determine number of components based on complexity
    if complexity_level == 1:
        n_components = rng.integers(1, 3)  # 1-2 components
    elif complexity_level == 2:
        n_components = rng.integers(2, 5)  # 2-4 components
    else:
        n_components = rng.integers(3, 7)  # 3-6 components
    
    # Sample motifs
    motif_names = list(weights.keys())
    motif_probs = [weights[m] for m in motif_names]
    
    u = S(0)
    for _ in range(n_components):
        chosen_motif = rng.choice(motif_names, p=motif_probs)
        component = sample_motif(chosen_motif, rng, dim, temporal_order)
        u = u + component
    
    # For 4th order operators, ensure high-k content
    if spatial_order >= 4:
        # Add a high-frequency component
        high_k_motif = sample_motif('M6', rng, dim, temporal_order)
        u = u + high_k_motif
    
    # Apply boundary mask for elliptic/plate families
    if family in ['poisson', 'biharmonic', 'beam_plate']:
        u = boundary_mask(dim) * u
    
    return expand(u)


def sample_u_track_B(dim: int, temporal_order: int, spatial_order: int,
                     complexity_level: int, rng: np.random.Generator) -> Expr:
    """
    Track B: Shared prior u sampling (operator-agnostic).
    
    Uses fixed motif mixture for all operators to avoid label leakage.
    Identifiability filter applied separately after sampling.
    
    Args:
        dim: Spatial dimension (1 or 2)
        temporal_order: 0 (steady), 1, or 2
        spatial_order: Highest spatial derivative order
        complexity_level: 1 (simple), 2 (moderate), 3 (complex)
        rng: NumPy random generator
    
    Returns:
        SymPy expression for u
    """
    # Select weight distribution based on temporal order
    if temporal_order > 0:
        weights = TRACK_B_WEIGHTS_TEMPORAL.copy()
    else:
        weights = TRACK_B_WEIGHTS_STEADY.copy()
    
    weights = _normalize_weights(weights)
    
    # Determine number of components based on complexity
    if complexity_level == 1:
        n_components = rng.integers(1, 3)  # 1-2 components
    elif complexity_level == 2:
        n_components = rng.integers(2, 5)  # 2-4 components
    else:
        n_components = rng.integers(3, 7)  # 3-6 components
    
    # Sample motifs from shared distribution
    motif_names = list(weights.keys())
    motif_probs = [weights[m] for m in motif_names]
    
    u = S(0)
    for _ in range(n_components):
        chosen_motif = rng.choice(motif_names, p=motif_probs)
        component = sample_motif(chosen_motif, rng, dim, temporal_order)
        u = u + component
    
    # For 4th order operators, ensure sufficient curvature
    if spatial_order >= 4:
        # Add high-frequency component with probability
        if rng.random() < 0.5:
            high_k_motif = sample_motif('M6', rng, dim, temporal_order)
            u = u + high_k_motif
    
    # Apply boundary mask with small probability (for variety)
    if rng.random() < 0.15:
        u = boundary_mask(dim) * u
    
    return expand(u)


def sample_u(track: str, family: str, dim: int, temporal_order: int,
             spatial_order: int, complexity_level: int,
             rng: np.random.Generator) -> Expr:
    """
    Main interface for u sampling.
    
    Args:
        track: 'A' (physics-guided) or 'B' (shared prior)
        family: PDE family name
        dim: Spatial dimension (1 or 2)
        temporal_order: 0 (steady), 1, or 2
        spatial_order: Highest spatial derivative order
        complexity_level: 1-3
        rng: NumPy random generator
    
    Returns:
        SymPy expression for u
    """
    if track == 'A':
        return sample_u_track_A(family, dim, temporal_order, 
                                spatial_order, complexity_level, rng)
    elif track == 'B':
        return sample_u_track_B(dim, temporal_order, spatial_order,
                                complexity_level, rng)
    else:
        raise ValueError(f"Unknown track: {track}. Use 'A' or 'B'.")
