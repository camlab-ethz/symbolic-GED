"""Utility functions for PDE string processing."""

import re


def skeletonize_pde(pde: str) -> str:
    """
    Replace all numeric constants with token "C" to get structure-only skeleton.
    
    Handles:
    - Integers: 1, -2, 42
    - Floats: 0.37, -2.0, 1.5
    - Scientific notation: 3e-2, 1.2e+5, -4.5E-3
    
    Note: Exponents (numbers after ^) are NOT replaced, as they are part of structure.
    Signs are preserved: -2.0 becomes -C, but 2.0 becomes C.
    
    Args:
        pde: PDE string (e.g., "dt(u)=0.37*dxx(u)+1.1*u")
        
    Returns:
        Skeletonized string with numbers replaced by "C" (e.g., "dt(u)=C*dxx(u)+C*u")
    """
    # First strip spaces
    pde_no_spaces = pde.replace(' ', '')
    
    # Pattern to match numeric constants (but not exponents after ^)
    # Use negative lookbehind to avoid matching numbers immediately after ^
    # Capture optional minus sign and the number part separately
    # Pattern matches: (optional -)(digits with optional decimal and scientific notation)
    # But exclude if preceded by ^
    def replace_number(match):
        # Get the full match
        full_match = match.group(0)
        # Check if it starts with minus
        if full_match.startswith('-'):
            return '-C'
        else:
            return 'C'
    
    # Pattern: negative lookbehind for ^, then optional minus, then number
    pattern = r'(?<!\^)-?\d+\.?\d*(?:[eE][+-]?\d+)?'
    
    # Replace all numbers (except exponents) with "C" or "-C"
    skeleton = re.sub(pattern, replace_number, pde_no_spaces)
    
    return skeleton
