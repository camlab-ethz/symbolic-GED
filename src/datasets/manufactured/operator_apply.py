"""
Operator Application Module - Apply L to u to get f = L(u).
"""

import re
from typing import Optional, Tuple, Literal
from sympy import (
    symbols, diff, sin, cos, tan, tanh, atan, exp, log, cosh, sinh, sqrt,
    pi, expand, simplify, Expr, sympify
)

x, y, t = symbols('x y t', real=True)


def dt(u): return diff(u, t)
def dtt(u): return diff(u, t, 2)
def dx(u): return diff(u, x)
def dxx(u): return diff(u, x, 2)
def dxxx(u): return diff(u, x, 3)
def dxxxx(u): return diff(u, x, 4)
def dy(u): return diff(u, y)
def dyy(u): return diff(u, y, 2)
def dyyy(u): return diff(u, y, 3)
def dyyyy(u): return diff(u, y, 4)
def dxy(u): return diff(diff(u, x), y)
def dxxyy(u): return diff(diff(diff(diff(u, x), x), y), y)
def dz(u): 
    z = symbols('z', real=True)
    return diff(u, z)
def dzz(u):
    z = symbols('z', real=True)
    return diff(u, z, 2)


def parse_operator_tokens(L_str: str) -> list:
    """Extract derivative tokens from operator string."""
    pattern = r'\b(d[txyz]+)\('
    return list(set(re.findall(pattern, L_str)))


def infer_orders_from_operator(L_str: str) -> Tuple[int, int, int]:
    """Infer temporal_order, spatial_order, dim from operator string."""
    temporal_order = 2 if 'dtt(' in L_str else (1 if 'dt(' in L_str else 0)
    
    if any(tok in L_str for tok in ['dxxxx(', 'dyyyy(', 'dxxyy(']):
        spatial_order = 4
    elif any(tok in L_str for tok in ['dxxx(', 'dyyy(']):
        spatial_order = 3
    elif any(tok in L_str for tok in ['dxx(', 'dyy(', 'dxy(']):
        spatial_order = 2
    else:
        spatial_order = 1
    
    dim = 2 if any(tok in L_str for tok in ['dy(', 'dyy(', 'dxy(']) else 1
    
    return temporal_order, spatial_order, dim


def apply_operator(L_str: str, u_expr: Expr, 
                   dim: Optional[int] = None,
                   temporal_order: Optional[int] = None,
                   simplify_level: Literal["none", "light"] = "none",
                   timeout_seconds: int = 15) -> Expr:
    """
    Apply PDE operator L to u. Returns f = L(u).
    
    Args:
        L_str: Operator string like "dt(u) - 1.5*dxx(u)"
        u_expr: SymPy expression for u
        dim: Spatial dimension (inferred if None)
        temporal_order: Temporal order (inferred if None)
        simplify_level: "none" (expand only) or "light" (expand + simplify)
        timeout_seconds: Unused (kept for API compatibility)
    
    Returns:
        f = L(u) as SymPy expression
    """
    if dim is None or temporal_order is None:
        t_ord, s_ord, d = infer_orders_from_operator(L_str)
        temporal_order = temporal_order or t_ord
        dim = dim or d
    
    locals_dict = {
        'x': x, 'y': y, 't': t, 'u': u_expr,
        'dt': dt, 'dtt': dtt,
        'dx': dx, 'dxx': dxx, 'dxxx': dxxx, 'dxxxx': dxxxx,
        'dy': dy, 'dyy': dyy, 'dyyy': dyyy, 'dyyyy': dyyyy,
        'dxy': dxy, 'dxxyy': dxxyy,
        'dz': dz, 'dzz': dzz,
        'sin': sin, 'cos': cos, 'tan': tan, 'tanh': tanh, 'atan': atan,
        'exp': exp, 'log': log, 'cosh': cosh, 'sinh': sinh, 'sqrt': sqrt,
        'pi': pi,
    }
    
    f_expr = sympify(L_str, locals=locals_dict)
    
    # Always expand
    f_expr = expand(f_expr)
    
    # Optional light simplification
    if simplify_level == "light":
        f_expr = simplify(f_expr, ratio=1.5)
    
    return f_expr
