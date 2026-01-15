"""Shared PDE string normalization utilities.

This module is the single source of truth for how we treat PDE strings across:
- dataset creation / label fixing
- grammar tokenization
- Lample & Charton-style tokenization

Key invariants we want everywhere:
- operator-only expressions (no trailing '= 0')
- consistent power operator (treat Python '**' as '^' for tokenizers / signatures)
"""

from __future__ import annotations

import re

_EQ0_RE = re.compile(r"\s*=\s*0\s*$")


def strip_eq0(pde: str) -> str:
    """Remove a trailing '= 0' (with any spacing) if present."""
    return _EQ0_RE.sub("", str(pde)).strip()


def normalize_power(pde: str) -> str:
    """Normalize Python-style exponentiation to caret form used by our grammar/tokenizers."""
    return str(pde).replace("**", "^")


def normalize_pde_string(pde: str) -> str:
    """Apply all cross-pipeline normalizations (safe for all callers)."""
    return normalize_power(strip_eq0(pde))

