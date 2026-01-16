"""
Utility functions for manufactured solutions generation.
"""

import hashlib
import numpy as np
from typing import List, Union
from sympy import symbols, sstr, Expr, expand, simplify, S


# Define symbolic variables
x, y, t = symbols('x y t', real=True)


class TimeoutError(Exception):
    """Raised when a computation exceeds the allowed time."""
    pass


class timeout:
    """Dummy timeout context manager (signal-based doesn't work in multiprocessing)."""
    def __init__(self, seconds: int = 10):
        self.seconds = seconds
    def __enter__(self):
        return self
    def __exit__(self, type, value, traceback):
        pass


def make_grid(dim: int, temporal_order: int, 
              nx: int = 32, ny: int = 16, nt: int = 8) -> dict:
    """Create evaluation grid."""
    grid = {'x': np.linspace(0.05, 0.95, nx)}
    if dim >= 2:
        grid['y'] = np.linspace(0.05, 0.95, ny)
    if temporal_order > 0:
        grid['t'] = np.linspace(0.05, 0.95, nt)
    return grid


def evaluate_on_grid(expr: Expr, grid: dict, dim: int, temporal_order: int) -> np.ndarray:
    """Evaluate SymPy expression on grid."""
    from sympy import lambdify
    
    if dim == 1 and temporal_order == 0:
        func = lambdify([x], expr, modules=['numpy'])
        return func(grid['x'])
    elif dim == 1 and temporal_order > 0:
        func = lambdify([x, t], expr, modules=['numpy'])
        X, T = np.meshgrid(grid['x'], grid['t'], indexing='ij')
        return func(X, T)
    elif dim == 2 and temporal_order == 0:
        func = lambdify([x, y], expr, modules=['numpy'])
        X, Y = np.meshgrid(grid['x'], grid['y'], indexing='ij')
        return func(X, Y)
    else:
        func = lambdify([x, y, t], expr, modules=['numpy'])
        X, Y, T = np.meshgrid(grid['x'], grid['y'], grid['t'], indexing='ij')
        return func(X, Y, T)


def canonical_print(expr: Expr, do_simplify: bool = True) -> str:
    """
    Generate canonical string representation of a SymPy expression.
    
    Uses expand() and optionally simplify() to normalize, then prints
    with lexicographic ordering for deterministic output.
    
    Args:
        expr: SymPy expression
        do_simplify: If True, apply light simplification after expand
    
    Returns:
        Canonical string representation
    """
    try:
        # Always expand first
        canonical = expand(expr)
        
        # Optional light simplification
        if do_simplify:
            # Use simplify with ratio limit to avoid heavy computation
            canonical = simplify(canonical, ratio=1.5)
        
        # Print with lexicographic ordering for determinism
        return sstr(canonical, order='lex')
    except Exception:
        # Fallback to str if canonicalization fails
        return str(expr)


def canonical_hash(u_list: Union[List[str], List[Expr]]) -> str:
    """
    Generate hash for a set of u expressions for uniqueness checking.
    
    Args:
        u_list: List of canonical u strings or expressions
    
    Returns:
        16-character hash string
    """
    # Convert expressions to canonical strings if needed
    str_list = []
    for u in u_list:
        if isinstance(u, str):
            str_list.append(u)
        else:
            str_list.append(canonical_print(u, do_simplify=False))
    
    # Sort for order-independence, join, and hash
    combined = '|'.join(sorted(str_list))
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


class UniquenessTracker:
    """
    Track uniqueness of u expressions within a generation run.
    
    Tracks:
    - Individual u expressions (per-set uniqueness)
    - Complete sets of u expressions (set-level uniqueness)
    """
    def __init__(self):
        self.seen_u = set()
        self.seen_sets = set()
    
    def is_unique_u(self, canon_u: str) -> bool:
        """Check if a single u is unique within current set."""
        if canon_u in self.seen_u:
            return False
        self.seen_u.add(canon_u)
        return True
    
    def is_unique_set(self, u_list: List[str]) -> bool:
        """Check if a complete set of u's is unique (for same operator)."""
        sig = canonical_hash(u_list)
        if sig in self.seen_sets:
            return False
        self.seen_sets.add(sig)
        return True
    
    def reset_for_new_record(self):
        """Reset per-set tracking for a new (operator, track) pair."""
        self.seen_u.clear()


def count_ops_safe(expr: Expr) -> int:
    """Safely count operations in expression."""
    try:
        from sympy import count_ops
        return count_ops(expr)
    except Exception:
        return len(str(expr))
