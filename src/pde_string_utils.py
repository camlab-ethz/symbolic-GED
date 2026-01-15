"""Utilities for operator-only PDE strings.

These helpers operate purely on the string form (no SymPy reordering), to support
benchmark checks around:
- inferred derivative orders and minimum spatial dimension
- canonicalized formatting
- coefficient-agnostic signatures
"""

from __future__ import annotations

import re
from typing import Dict, Tuple


_DERIV_TOKENS = [
    "dtt",
    "dt",
    "dxxyy",
    "dxxxx",
    "dyyyy",
    "dzzzz",
    "dxxx",
    "dyyy",
    "dzzz",
    "dxx",
    "dyy",
    "dzz",
    "dx",
    "dy",
    "dz",
]


def infer_orders_and_dim(op_str: str) -> Dict[str, int]:
    """Infer temporal order, spatial order, and minimum dimension from an operator string."""
    s = str(op_str)

    # temporal order
    temporal_order = 0
    if re.search(r"\bdtt\s*\(", s):
        temporal_order = 2
    elif re.search(r"\bdt\s*\(", s):
        temporal_order = 1

    # spatial order
    spatial_order = 0
    # Order 4 tokens (including mixed)
    if re.search(r"\b(dxxyy|dxxxx|dyyyy|dzzzz)\s*\(", s):
        spatial_order = 4
    elif re.search(r"\b(dxxx|dyyy|dzzz)\s*\(", s):
        spatial_order = 3
    elif re.search(r"\b(dxx|dyy|dzz)\s*\(", s):
        spatial_order = 2
    elif re.search(r"\b(dx|dy|dz)\s*\(", s):
        spatial_order = 1

    # inferred minimum dimension (based on presence of y/z derivatives)
    inferred_min_dim = 1
    if re.search(r"\b(dy|dyy|dyyy|dyyyy|dxxyy)\s*\(", s):
        inferred_min_dim = max(inferred_min_dim, 2)
    if re.search(r"\b(dz|dzz|dzzz|dzzzz)\s*\(", s):
        inferred_min_dim = max(inferred_min_dim, 3)

    return {
        "temporal_order": temporal_order,
        "spatial_order": spatial_order,
        "inferred_min_dim": inferred_min_dim,
    }


def canonicalize(op_str: str) -> str:
    """Lightweight deterministic normalization (no algebraic reordering)."""
    s = str(op_str).strip()

    # Normalize whitespace around operators/parens
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*\(\s*", "(", s)
    s = re.sub(r"\s*\)\s*", ")", s)
    s = re.sub(r"\s*\+\s*", " + ", s)
    s = re.sub(r"\s*-\s*", " - ", s)
    s = re.sub(r"\s*\*\s*", "*", s)
    s = re.sub(r"\s*/\s*", "/", s)

    # Collapse "+ -" -> "-"
    s = re.sub(r"\+\s*-\s*", "- ", s)
    # Collapse "- -" -> "+"
    s = re.sub(r"-\s*-\s*", "+ ", s)

    # Collapse multiple spaces again and trim
    s = re.sub(r"\s+", " ", s).strip()
    return s


def signature(op_str: str) -> Tuple[Tuple[str, ...], int, int, bool, bool, bool, bool, bool, bool]:
    """Coefficient-agnostic signature for family-level uniqueness checks.

    Returns:
      (
        sorted_derivative_tokens_used,
        max_temporal_order,
        max_spatial_order,
        has_advective_nonlinearity (u*dx(u)/u*dy(u)/u*dz(u)),
        has_cubic_nonlinearity (u**3 or u^3),
        has_trig_nonlinearity (sin(u)),
        has_mixed_4th (dxxyy),
      )
    """
    s = canonicalize(op_str)

    used = set()
    for tok in _DERIV_TOKENS:
        if re.search(rf"\b{re.escape(tok)}\s*\(", s):
            used.add(tok)

    meta = infer_orders_and_dim(s)

    has_adv = bool(re.search(r"\bu\*(dx|dy|dz)\(u\)", s))
    has_u3 = ("u**3" in s) or ("u^3" in s)
    has_u2 = ("u**2" in s) or ("u^2" in s)
    has_sin = "sin(u)" in s
    has_mixed = "dxxyy(" in s or "dxxyy(u)" in s
    # Linear reaction term presence (helps distinguish heat vs Fisher/AC, etc).
    # Accepts forms like "- u" and "- 0.5*u".
    has_linear_u = bool(
        re.search(r"(^|[+-])\s*(?:\d+(?:\.\d+)?(?:e[+-]?\d+)?)?\*?u(\s|$)", s)
    )

    return (
        tuple(sorted(used)),
        int(meta["temporal_order"]),
        int(meta["spatial_order"]),
        has_adv,
        has_u3,
        has_u2,
        has_sin,
        has_mixed,
        has_linear_u,
    )

