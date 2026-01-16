"""Unified PDE physics classification functions.

This module consolidates all physics-based classification and labeling functions
used across the codebase. It provides:

1. Family-based physics properties (from PDE_PHYSICS dictionary)
2. String-based classification functions (for generated/decoded PDEs)
3. Unified labeling pipeline
"""

import re
from typing import Dict, List, Optional, Union
import numpy as np

# =============================================================================
# PDE FAMILY PHYSICS PROPERTIES
# =============================================================================

PDE_PHYSICS = {
    "heat": {
        "type": "parabolic",
        "linearity": "linear",
        "mechanisms": ["diffusion"],
        "order": 2,
        "temporal": "first",
        "example": "u_t = α∇²u",
    },
    "wave": {
        "type": "hyperbolic",
        "linearity": "linear",
        "mechanisms": ["propagation"],
        "order": 2,
        "temporal": "second",
        "example": "u_tt = c²∇²u",
    },
    "advection": {
        "type": "hyperbolic",
        "linearity": "linear",
        "mechanisms": ["advection"],
        "order": 1,
        "temporal": "first",
        "example": "u_t + c·∇u = 0",
    },
    "burgers": {
        "type": "parabolic",
        "linearity": "nonlinear",
        "mechanisms": ["advection", "diffusion"],
        "order": 2,
        "temporal": "first",
        "example": "u_t + u·∇u = ν∇²u",
    },
    "kdv": {
        "type": "dispersive",
        "linearity": "nonlinear",
        "mechanisms": ["advection", "dispersion"],
        "order": 3,
        "temporal": "first",
        "example": "u_t + u·u_x + u_xxx = 0",
    },
    "kuramoto_sivashinsky": {
        "type": "parabolic",
        "linearity": "nonlinear",
        "mechanisms": ["diffusion", "dispersion", "instability"],
        "order": 4,
        "temporal": "first",
        "example": "u_t + u·u_x + u_xx + u_xxxx = 0",
    },
    "allen_cahn": {
        "type": "parabolic",
        "linearity": "nonlinear",
        "mechanisms": ["diffusion", "reaction"],
        "order": 2,
        "temporal": "first",
        "example": "u_t = ε²∇²u + u - u³",
    },
    "cahn_hilliard": {
        "type": "parabolic",
        "linearity": "nonlinear",
        "mechanisms": ["diffusion", "reaction"],
        "order": 4,
        "temporal": "first",
        "example": "u_t = ∇²(u³ - u - γ∇²u)",
    },
    "fisher_kpp": {
        "type": "parabolic",
        "linearity": "nonlinear",
        "mechanisms": ["diffusion", "reaction"],
        "order": 2,
        "temporal": "first",
        "example": "u_t = D∇²u + ru(1-u)",
    },
    "poisson": {
        "type": "elliptic",
        "linearity": "linear",
        "mechanisms": ["source"],
        "order": 2,
        "temporal": "none",
        "example": "∇²u = f",
    },
    "laplace": {
        "type": "elliptic",
        "linearity": "linear",
        "mechanisms": ["source"],
        "order": 2,
        "temporal": "none",
        "example": "∇²u = 0",
    },
    "helmholtz": {
        "type": "elliptic",
        "linearity": "linear",
        "mechanisms": ["source", "reaction"],
        "order": 2,
        "temporal": "none",
        "example": "∇²u + k²u = f",
    },
    "biharmonic": {
        "type": "elliptic",
        "linearity": "linear",
        "mechanisms": ["bending"],
        "order": 4,
        "temporal": "none",
        "example": "∇⁴u = f",
    },
    "reaction_diffusion_cubic": {
        "type": "parabolic",
        "linearity": "nonlinear",
        "mechanisms": ["diffusion", "reaction"],
        "order": 2,
        "temporal": "first",
        "example": "u_t - ∇²u ± g u³ = 0",
    },
    "airy": {
        "type": "dispersive",
        "linearity": "linear",
        "mechanisms": ["dispersion"],
        "order": 3,
        "temporal": "first",
        "example": "u_t + α u_xxx = 0",
    },
    "sine_gordon": {
        "type": "hyperbolic",
        "linearity": "nonlinear",
        "mechanisms": ["propagation", "soliton"],
        "order": 2,
        "temporal": "second",
        "example": "u_tt - c²∇²u = sin(u)",
    },
    "beam_plate": {
        "type": "hyperbolic",
        "linearity": "linear",
        "mechanisms": ["bending", "waves"],
        "order": 4,
        "temporal": "second",
        "example": "u_tt + κ Δ²u = 0",
    },
    "telegraph": {
        "type": "hyperbolic",
        "linearity": "linear",
        "mechanisms": ["propagation", "damping"],
        "order": 2,
        "temporal": "second",
        "example": "u_tt + αu_t = c²∇²u",
    },
    "fokker_planck": {
        "type": "parabolic",
        "linearity": "linear",
        "mechanisms": ["diffusion", "advection"],
        "order": 2,
        "temporal": "first",
        "example": "u_t = -∇·(Au) + ∇·(D∇u)",
    },
    # navier_stokes removed from dataset families
}


# =============================================================================
# STRING-BASED CLASSIFICATION FUNCTIONS
# =============================================================================


def classify_pde_type(pde: str) -> str:
    """Classify PDE string by equation type.

    Args:
        pde: PDE string (e.g., 'dt(u) = dxx(u)')

    Returns:
        'elliptic', 'parabolic', 'hyperbolic', 'dispersive', or 'unknown'
    """
    pde_lower = pde.lower()

    # Check for temporal derivatives
    has_dt = "dt(" in pde_lower or "_t" in pde_lower
    has_dtt = "dtt(" in pde_lower or "_tt" in pde_lower

    # Check for dispersive terms (odd-order spatial derivatives)
    has_dxxx = "dxxx(" in pde_lower or "_xxx" in pde_lower
    has_schrodinger = "i*" in pde_lower or "complex" in pde_lower

    if has_dxxx or has_schrodinger:
        return "dispersive"
    elif has_dtt:
        return "hyperbolic"
    elif has_dt:
        return "parabolic"
    elif not has_dt and not has_dtt:
        return "elliptic"
    else:
        return "unknown"


def classify_linearity(pde: str) -> str:
    """Classify PDE string by linearity.

    Args:
        pde: PDE string

    Returns:
        'linear' or 'nonlinear'
    """
    # Check for common nonlinear patterns
    nonlinear_patterns = [
        r"u\s*\*\s*d[xyz]",  # u * dx(...)
        r"d[xyz][xyz]?\(u\)\s*\*\s*u",  # dxx(u) * u
        r"u\s*\*\s*u",  # u * u
        r"u\s*\*{2}\s*\d",  # u ** 2
        r"pow\(u",  # pow(u, ...)
        r"sin\(u\)",  # sin(u)
        r"cos\(u\)",  # cos(u)
        r"exp\(u\)",  # exp(u)
        r"u\^",  # u^n
    ]

    for pattern in nonlinear_patterns:
        if re.search(pattern, pde, re.IGNORECASE):
            return "nonlinear"

    return "linear"


def classify_order(pde: str) -> int:
    """Classify PDE by highest spatial derivative order.

    Args:
        pde: PDE string

    Returns:
        Integer order (1, 2, 3, or 4)
    """
    pde_lower = pde.lower()

    # Check for 4th order
    if any(
        x in pde_lower for x in ["dxxxx", "dyyyy", "dzzzz", "_xxxx", "_yyyy", "_zzzz"]
    ):
        return 4

    # Check for 3rd order
    if any(x in pde_lower for x in ["dxxx", "dyyy", "dzzz", "_xxx", "_yyy", "_zzz"]):
        return 3

    # Check for 2nd order
    if any(
        x in pde_lower
        for x in [
            "dxx",
            "dyy",
            "dzz",
            "dxy",
            "dxz",
            "dyz",
            "_xx",
            "_yy",
            "_zz",
            "_xy",
            "_xz",
            "_yz",
        ]
    ):
        return 2

    # Check for 1st order
    if any(x in pde_lower for x in ["dx(", "dy(", "dz(", "_x", "_y", "_z"]):
        return 1

    return 0


def classify_spatial_dim(pde: str) -> int:
    """Classify PDE by spatial dimensionality.

    Args:
        pde: PDE string

    Returns:
        1, 2, or 3
    """
    pde_lower = pde.lower()

    has_x = "dx" in pde_lower or "_x" in pde_lower
    has_y = "dy" in pde_lower or "_y" in pde_lower
    has_z = "dz" in pde_lower or "_z" in pde_lower

    if has_z:
        return 3
    elif has_y:
        return 2
    elif has_x:
        return 1
    else:
        return 1  # Default to 1D


def classify_temporal_order(pde: str) -> int:
    """Classify PDE by temporal derivative order.

    Args:
        pde: PDE string

    Returns:
        0 (steady-state), 1 (first-order), or 2 (second-order)
    """
    pde_lower = pde.lower()

    if "dtt(" in pde_lower or "_tt" in pde_lower:
        return 2
    elif "dt(" in pde_lower or "_t" in pde_lower:
        return 1
    else:
        return 0


def is_valid_pde(pde: str, use_sympy: bool = False) -> bool:
    """Check if PDE string is syntactically valid.

    A valid PDE must:
    - Have balanced parentheses
    - Contain at least one SPATIAL derivative (dx, dy, dz, or higher order)
    - Contain the variable u
    - Not have malformed syntax (trailing operators, consecutive operators, etc.)
    - Not have malformed numbers (like ..19 or ..)

    Note: A PDE requires spatial derivatives. Equations with only temporal
    derivatives (like dt(u) or dtt(u) alone) are ODEs, not PDEs.

    Args:
        pde: PDE string (can omit = 0)
        use_sympy: Whether to use sympy for validation (slower but more thorough)

    Returns:
        True if valid
    """
    if not pde or pde == "[INVALID]":
        return False

    # Remove = 0 if present (we allow omitting it)
    pde_clean = pde.split("=")[0].strip() if "=" in pde else pde.strip()

    if not pde_clean:
        return False

    # Basic structural checks
    if pde_clean.count("(") != pde_clean.count(")"):
        return False

    # Check for required components
    # CRITICAL: A PDE must have at least one SPATIAL derivative (not just temporal)
    # Patterns for spatial derivatives: dx, dy, dz (but NOT dt)
    # Also match higher orders: dxx, dxxx, dxxxx, dyy, dzz, dxy, dxxyy, etc.
    has_spatial_derivative = bool(re.search(
        r'd[xyz]+[xyz]*\s*\(', pde_clean.lower()
    )) or any(
        x in pde_clean.lower() for x in ["_x", "_y", "_z", "_xx", "_yy", "_zz"]
    )
    has_variable = "u" in pde_clean.lower()

    if not has_spatial_derivative or not has_variable:
        return False

    # =========================================================================
    # STRICT SYNTAX CHECKS
    # =========================================================================

    # 1. Check for trailing operators (+, -, *, /)
    if re.search(r"[+\-*/]\s*$", pde_clean):
        return False

    # 2. Check for trailing dot (like "dxx(u) + .")
    if re.search(r"\.\s*$", pde_clean):
        return False

    # 3. Check for standalone dot anywhere (like "+ . +", "- .", "+ .")
    #    This catches patterns like "- . + 4" or "+ ."
    if re.search(r"[+\-*/]\s*\.\s*[+\-*/\s]", pde_clean):
        return False
    if re.search(r"[+\-*/]\s*\.\s*\d", pde_clean):  # "- . + 4" becomes "- .4" sometimes
        return False
    if re.search(r"[+\-*/]\s*\.$", pde_clean):  # ends with "+ ." or "- ."
        return False

    # 4. Check for malformed numbers (consecutive dots like ..19 or ..95)
    if re.search(r"\.\.+", pde_clean):
        return False

    # 5. Check for consecutive operators (like "+  *" or "*  +")
    #    Patterns: operator, optional whitespace, operator
    if re.search(r"[+\-*/]\s*[*/]", pde_clean):  # + * or - * or * * etc
        return False
    if re.search(
        r"[*/]\s*[+\-]", pde_clean
    ):  # * + or / - etc (but "* -2" is ok for negative)
        # Allow "* -" only if followed by a digit (negative number)
        if re.search(r"[*/]\s*[+\-]\s*[^\d]", pde_clean):
            return False

    # 6. Check for derivatives with invalid arguments like dxx(0) or dyy()
    #    Valid: dxx(u), dxx(u * u), dxx(2 * u)
    #    Invalid: dxx(0), dxx(), dxx(.)
    if re.search(r"d[xyzt]+\(\s*\)", pde_clean):  # empty derivative dxx()
        return False
    if re.search(r"d[xyzt]+\(\s*0\s*\)", pde_clean):  # derivative of 0
        return False
    if re.search(r"d[xyzt]+\(\s*\.\s*\)", pde_clean):  # derivative of dot
        return False
    
    # 6b. Check for functions with invalid arguments like sin(.), cos(.), exp(.)
    #     Valid: sin(u), cos(2*u), exp(u)
    #     Invalid: sin(.), cos(.), exp(.), sin(), cos()
    if re.search(r"(sin|cos|tan|exp|log|sqrt|pow)\(\s*\)", pde_clean):  # empty function
        return False
    if re.search(r"(sin|cos|tan|exp|log|sqrt|pow)\(\s*\.\s*\)", pde_clean):  # function of dot
        return False

    # 7. Check for leading operators at the very start (but allow leading -)
    if re.search(r"^[+*/]", pde_clean.strip()):
        return False

    # 8. Check for empty parentheses (like "u + ()")
    if re.search(r"\(\s*\)", pde_clean):
        return False

    if use_sympy:
        try:
            import sympy as sp
            from sympy.parsing.sympy_parser import parse_expr

            # Try to parse as sympy expression
            parse_expr(pde_clean.replace("=", "-"))
            return True
        except:
            return False

    return True


def parse_pde(pde: str, family: str = None) -> Dict:
    """Parse PDE string and return all physics labels.

    Args:
        pde: PDE string
        family: Optional family name (uses PDE_PHYSICS if provided)

    Returns:
        Dictionary with all physics labels
    """
    # If family is known, use PDE_PHYSICS
    if family and family in PDE_PHYSICS:
        props = PDE_PHYSICS[family]
        return {
            "valid": True,
            "type": props["type"],
            "linearity": props["linearity"],
            "order": props["order"],
            "dim": classify_spatial_dim(pde),
            "temporal_order": (
                1
                if props["temporal"] == "first"
                else (2 if props["temporal"] == "second" else 0)
            ),
            "mechanisms": props["mechanisms"],
            "family": family,
        }

    # Otherwise, classify from string
    return {
        "valid": is_valid_pde(pde),
        "type": classify_pde_type(pde),
        "linearity": classify_linearity(pde),
        "order": classify_order(pde),
        "dim": classify_spatial_dim(pde),
        "temporal_order": classify_temporal_order(pde),
        "mechanisms": [],
        "family": "unknown",
    }


def assign_physics_labels(families: List[str]) -> Dict[str, List]:
    """Assign physics labels to PDEs based on family names.

    Args:
        families: List of family names

    Returns:
        Dictionary of label lists
    """
    labels = {
        "type": [],
        "linearity": [],
        "has_diffusion": [],
        "has_advection": [],
        "has_dispersion": [],
        "has_reaction": [],
        "order": [],
        "temporal": [],
    }

    for fam in families:
        if fam in PDE_PHYSICS:
            props = PDE_PHYSICS[fam]
            labels["type"].append(props["type"])
            labels["linearity"].append(props["linearity"])
            labels["has_diffusion"].append("diffusion" in props["mechanisms"])
            labels["has_advection"].append("advection" in props["mechanisms"])
            labels["has_dispersion"].append("dispersion" in props["mechanisms"])
            labels["has_reaction"].append("reaction" in props["mechanisms"])
            labels["order"].append(props["order"])
            labels["temporal"].append(props["temporal"])
        else:
            labels["type"].append("unknown")
            labels["linearity"].append("unknown")
            labels["has_diffusion"].append(False)
            labels["has_advection"].append(False)
            labels["has_dispersion"].append(False)
            labels["has_reaction"].append(False)
            labels["order"].append(0)
            labels["temporal"].append("unknown")

    return labels


# =============================================================================
# FAMILY CLASSIFICATION
# =============================================================================


def get_family_type(family: str) -> str:
    """Get PDE type for a family.

    Args:
        family: Family name

    Returns:
        'elliptic', 'parabolic', 'hyperbolic', 'dispersive', or 'unknown'
    """
    if family in PDE_PHYSICS:
        return PDE_PHYSICS[family]["type"]
    return "unknown"


def get_family_linearity(family: str) -> str:
    """Get linearity for a family.

    Args:
        family: Family name

    Returns:
        'linear' or 'nonlinear'
    """
    if family in PDE_PHYSICS:
        return PDE_PHYSICS[family]["linearity"]
    return "unknown"


def get_family_order(family: str) -> int:
    """Get spatial order for a family.

    Args:
        family: Family name

    Returns:
        Integer order
    """
    if family in PDE_PHYSICS:
        return PDE_PHYSICS[family]["order"]
    return 0


def is_dispersive_family(family: str) -> bool:
    """Check if family is dispersive.

    Args:
        family: Family name

    Returns:
        True if dispersive
    """
    return family in {"kdv", "schrodinger"}


def is_nonlinear_family(family: str) -> bool:
    """Check if family is nonlinear.

    Args:
        family: Family name

    Returns:
        True if nonlinear
    """
    if family in PDE_PHYSICS:
        return PDE_PHYSICS[family]["linearity"] == "nonlinear"
    return False


# =============================================================================
# BATCH OPERATIONS
# =============================================================================


def classify_batch(
    pde_strings: List[str], families: List[str] = None
) -> Dict[str, np.ndarray]:
    """Classify a batch of PDEs.

    Args:
        pde_strings: List of PDE strings
        families: Optional list of family names

    Returns:
        Dictionary with label arrays
    """
    n = len(pde_strings)

    types = []
    linearities = []
    orders = []
    dims = []
    temporal_orders = []
    valid = []

    for i, pde in enumerate(pde_strings):
        family = families[i] if families else None
        labels = parse_pde(pde, family)

        types.append(labels["type"])
        linearities.append(labels["linearity"])
        orders.append(labels["order"])
        dims.append(labels["dim"])
        temporal_orders.append(labels["temporal_order"])
        valid.append(labels["valid"])

    return {
        "type": np.array(types),
        "linearity": np.array(linearities),
        "order": np.array(orders, dtype=np.int32),
        "dim": np.array(dims, dtype=np.int32),
        "temporal_order": np.array(temporal_orders, dtype=np.int32),
        "valid": np.array(valid, dtype=bool),
    }
