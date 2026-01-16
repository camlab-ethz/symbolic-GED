"""
Filters for Manufactured Solutions

Includes:
- Identifiability filter (ensure u excites relevant operator terms)
- Complexity filter (reject overly complex expressions)
- Stability filter (reject numerically unstable expressions)
"""

import numpy as np
from typing import Dict, Optional
from sympy import Expr, count_ops, diff

from datasets.manufactured.utils import (
    make_grid, evaluate_on_grid, count_ops_safe, x, y, t
)
from datasets.manufactured.operator_apply import parse_operator_tokens


# Default thresholds
DEFAULT_U_MAX_OPS = 250
DEFAULT_F_MAX_OPS = 1200
DEFAULT_MAX_STR_LEN = 4000
DEFAULT_GRADIENT_MAX = 50
DEFAULT_CURVATURE_MAX = 500
DEFAULT_EPS = 1e-6


def check_complexity(u_expr: Expr, f_expr: Expr,
                     u_max_ops: int = DEFAULT_U_MAX_OPS,
                     f_max_ops: int = DEFAULT_F_MAX_OPS,
                     max_str_len: int = DEFAULT_MAX_STR_LEN) -> bool:
    """
    Check if expressions are within complexity bounds.
    
    Args:
        u_expr: SymPy expression for u
        f_expr: SymPy expression for f
        u_max_ops: Maximum operations in u
        f_max_ops: Maximum operations in f
        max_str_len: Maximum string length for canonical form
    
    Returns:
        True if within bounds, False otherwise
    """
    # Check u complexity
    u_ops = count_ops_safe(u_expr)
    if u_ops > u_max_ops:
        return False
    
    # Check f complexity
    f_ops = count_ops_safe(f_expr)
    if f_ops > f_max_ops:
        return False
    
    # Check string length
    if len(str(u_expr)) > max_str_len:
        return False
    if len(str(f_expr)) > max_str_len:
        return False
    
    return True


def check_stability(u_expr: Expr, dim: int, temporal_order: int,
                    gradient_max: float = DEFAULT_GRADIENT_MAX,
                    curvature_max: float = DEFAULT_CURVATURE_MAX) -> bool:
    """
    Check if u is numerically stable on evaluation grid.
    
    Args:
        u_expr: SymPy expression for u
        dim: Spatial dimension
        temporal_order: Temporal order
        gradient_max: Maximum allowed gradient magnitude
        curvature_max: Maximum allowed second derivative magnitude
    
    Returns:
        True if stable, False otherwise
    """
    try:
        grid = make_grid(dim, temporal_order)
        u_vals = evaluate_on_grid(u_expr, grid, dim, temporal_order)
        
        # Check for NaN/Inf
        if not np.isfinite(u_vals).all():
            return False
        
        # Check gradient bounds
        u_x = np.gradient(u_vals, axis=0)
        if np.max(np.abs(u_x)) > gradient_max:
            return False
        
        # Check curvature bounds (second derivative)
        u_xx = np.gradient(u_x, axis=0)
        if np.max(np.abs(u_xx)) > curvature_max:
            return False
        
        # For 2D, also check y derivatives
        if dim >= 2 and u_vals.ndim >= 2:
            u_y = np.gradient(u_vals, axis=1)
            if np.max(np.abs(u_y)) > gradient_max:
                return False
            u_yy = np.gradient(u_y, axis=1)
            if np.max(np.abs(u_yy)) > curvature_max:
                return False
        
        return True
        
    except Exception:
        return False


def check_informative(L_str: str, u_expr: Expr, 
                      dim: int, temporal_order: int,
                      eps: float = DEFAULT_EPS) -> bool:
    """
    Check if u excites all relevant terms in the operator L.
    
    For Track B, this ensures the (u, f) pair is informative for operator ID.
    
    Args:
        L_str: Operator string
        u_expr: SymPy expression for u
        dim: Spatial dimension
        temporal_order: Temporal order
        eps: Minimum derivative magnitude threshold
    
    Returns:
        True if informative, False otherwise
    """
    try:
        grid = make_grid(dim, temporal_order)
        
        # Evaluate u to get scale
        u_vals = evaluate_on_grid(u_expr, grid, dim, temporal_order)
        u_std = np.std(u_vals)
        if u_std < 1e-10:
            return False  # u is constant
        
        # Adaptive epsilon based on u scale
        adaptive_eps = eps * max(1, u_std)
        
        # Parse tokens from operator
        tokens = parse_operator_tokens(L_str)
        
        # Check each derivative token
        for tok in tokens:
            deriv_expr = _compute_derivative(u_expr, tok)
            if deriv_expr is None:
                continue
            
            try:
                deriv_vals = evaluate_on_grid(deriv_expr, grid, dim, temporal_order)
                deriv_max = np.max(np.abs(deriv_vals))
                
                if deriv_max < adaptive_eps:
                    return False  # This derivative is too small
            except Exception:
                return False
        
        # Check nonlinear terms if present
        if 'u**' in L_str or 'u^' in L_str or 'u*u' in L_str:
            # Ensure u is not near zero everywhere
            if np.max(np.abs(u_vals)) < adaptive_eps:
                return False
        
        if 'sin(u)' in L_str:
            # Ensure sin(u) varies meaningfully
            from sympy import sin as sym_sin
            sin_u = sym_sin(u_expr)
            sin_vals = evaluate_on_grid(sin_u, grid, dim, temporal_order)
            # Check that sin(u) is distinguishable from u (not linearized)
            diff_vals = sin_vals - u_vals
            if np.max(np.abs(diff_vals)) < 0.05 * np.max(np.abs(u_vals)):
                return False
        
        return True
        
    except Exception:
        return False


def _compute_derivative(u_expr: Expr, token: str) -> Optional[Expr]:
    """
    Compute the derivative corresponding to a token.
    
    Args:
        u_expr: SymPy expression
        token: Derivative token like 'dxx', 'dt', etc.
    
    Returns:
        Derivative expression, or None if unknown token
    """
    token_map = {
        'dt': lambda u: diff(u, t),
        'dtt': lambda u: diff(u, t, 2),
        'dx': lambda u: diff(u, x),
        'dxx': lambda u: diff(u, x, 2),
        'dxxx': lambda u: diff(u, x, 3),
        'dxxxx': lambda u: diff(u, x, 4),
        'dy': lambda u: diff(u, y),
        'dyy': lambda u: diff(u, y, 2),
        'dyyy': lambda u: diff(u, y, 3),
        'dyyyy': lambda u: diff(u, y, 4),
        'dxy': lambda u: diff(diff(u, x), y),
        'dxxyy': lambda u: diff(diff(diff(diff(u, x), x), y), y),
    }
    
    if token in token_map:
        return token_map[token](u_expr)
    return None


def apply_all_filters(u_expr: Expr, f_expr: Expr, L_str: str,
                      dim: int, temporal_order: int,
                      track: str = 'A') -> Dict[str, bool]:
    """
    Apply all filters and return detailed results.
    
    Args:
        u_expr: SymPy expression for u
        f_expr: SymPy expression for f
        L_str: Operator string
        dim: Spatial dimension
        temporal_order: Temporal order
        track: 'A' or 'B' (Track B requires informative check)
    
    Returns:
        Dictionary with filter results
    """
    results = {}
    
    results['complexity'] = check_complexity(u_expr, f_expr)
    results['stability'] = check_stability(u_expr, dim, temporal_order)
    
    # Informative check is mandatory for Track B, optional for Track A
    if track == 'B':
        results['informative'] = check_informative(L_str, u_expr, dim, temporal_order)
    else:
        results['informative'] = True  # Skip for Track A
    
    results['passed'] = all(results.values())
    
    return results
