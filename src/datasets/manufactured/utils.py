"""
Utility functions for manufactured solutions generation.
"""

import hashlib
import numpy as np
from typing import List
from sympy import symbols, sstr, Expr


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


def canonical_print(expr: Expr, do_simplify: bool = False) -> str:
    """Generate canonical string representation."""
    return str(expr)


def canonical_hash(u_list: List[str]) -> str:
    """Generate hash for uniqueness."""
    combined = '|'.join(sorted(u_list))
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


class UniquenessTracker:
    """Track uniqueness of u expressions."""
    def __init__(self):
        self.seen_u = set()
        self.seen_sets = set()
    
    def is_unique_u(self, canon_u: str) -> bool:
        if canon_u in self.seen_u:
            return False
        self.seen_u.add(canon_u)
        return True
    
    def is_unique_set(self, sig: str) -> bool:
        if sig in self.seen_sets:
            return False
        self.seen_sets.add(sig)
        return True
    
    def reset_record(self):
        self.seen_u.clear()


def count_ops_safe(expr: Expr) -> int:
    try:
        from sympy import count_ops
        return count_ops(expr)
    except:
        return len(str(expr))
