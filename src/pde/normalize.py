"""Shared PDE string normalization utilities.

This module is the single source of truth for how we treat PDE strings across:
- dataset creation / label fixing
- grammar tokenization
- Lample & Charton-style tokenization

Key invariants we want everywhere:
- operator-only expressions (no trailing '= 0')
- consistent power operator (treat Python '**' as '^' for tokenizers / signatures)
- canonical formatting (spaces, coefficients, etc.)
"""

from __future__ import annotations

import re
from typing import Callable

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


def fmt_coeff(x: float) -> str:
    """Format coefficient with exactly 3 decimals, handling -0.000.
    
    Args:
        x: Coefficient value
        
    Returns:
        Formatted string with exactly 3 decimals (e.g., "1.500", "0.000")
    """
    if abs(x) < 1e-10:
        return "0.000"
    return f"{x:.3f}"


def canonicalize_operator_str(s: str) -> str:
    """Canonicalize PDE operator string to unique format.
    
    Rules:
    1. Strip trailing '=0' variants
    2. Normalize '**' -> '^'
    3. Normalize spaces: spaces around + and -; no spaces around * and ^
    4. Normalize coefficients: exactly 3 decimals (1.500 not 1.5), -0.000 -> 0.000
    5. Collapse numeric products: 2*1.234*dxx(u) -> 2.468*dxx(u)
    6. Signs via +/- at SUM level; coefficients nonnegative where possible
    
    Args:
        s: PDE operator string (may have =0, **, inconsistent spacing)
        
    Returns:
        Canonical operator string
    """
    if not s:
        return s
    
    # Step 1: Strip =0
    s = strip_eq0(s)
    
    # Step 2: Normalize ** to ^
    s = normalize_power(s)
    
    # Step 3: Collapse numeric products (e.g., 2*1.234 -> 2.468)
    # We need to find patterns like: number * number * derivative
    # Do this iteratively to handle multiple products
    max_iterations = 10
    for _ in range(max_iterations):
        def collapse_numeric_product(match):
            """Collapse numeric product like 2*1.234 to 2.468."""
            num1_str = match.group(1)
            num2_str = match.group(2)
            rest = match.group(3)
            try:
                num1 = float(num1_str)
                num2 = float(num2_str)
                product = num1 * num2
                return f"{fmt_coeff(product)}*{rest}"
            except (ValueError, TypeError):
                return match.group(0)
        
        # Pattern: number * number * (derivative, u^power, or parenthesized expression)
        pattern = r'(\d+\.?\d*)\s*\*\s*(\d+\.?\d*)\s*\*\s*([a-z]+\([^)]*\)|u\^[23]|\([^)]+\))'
        new_s = re.sub(pattern, collapse_numeric_product, s)
        if new_s == s:
            break  # No more changes
        s = new_s
    
    # Step 4: Normalize coefficient formatting to 3 decimals
    def normalize_coeff_before_star(match):
        """Normalize coefficient before * operator."""
        num_str = match.group(1)
        rest = match.group(2)
        try:
            num = float(num_str)
            return f"{fmt_coeff(num)}*{rest}"
        except (ValueError, TypeError):
            return match.group(0)
    
    # Replace coefficients before * (e.g., 1.5*dxx(u) -> 1.500*dxx(u))
    s = re.sub(r'(\d+\.?\d*)\s*\*\s*([a-z]+\([^)]*\)|u\^[23]|u|\([^)]+\))', 
               normalize_coeff_before_star, s)
    
    # Step 5: Normalize spaces
    # Remove all spaces first
    s = re.sub(r'\s+', '', s)
    
    # Add spaces around + and - operators (but preserve them in function names)
    # Strategy: add space before and after + and - that are not part of numbers
    s = re.sub(r'([+-])(?=[^0-9])', r' \1 ', s)  # Space after +/-
    s = re.sub(r'(?<=[^0-9])([+-])', r' \1', s)  # Space before +/-
    
    # Remove spaces around * and ^
    s = re.sub(r'\s*\*\s*', '*', s)
    s = re.sub(r'\s*\^\s*', '^', s)
    
    # Ensure no spaces inside function calls (dt ( u ) -> dt(u))
    s = re.sub(r'([a-z]+)\s*\(\s*', r'\1(', s)
    s = re.sub(r'\s*\)', ')', s)
    
    # Clean up multiple spaces
    s = re.sub(r'\s+', ' ', s)
    s = s.strip()
    
    return s

