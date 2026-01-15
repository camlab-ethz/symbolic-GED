"""
Unambiguous PDE context-free grammar (CFG) for canonical PDE strings.

This module defines:
- a set of productions (ordered list) where each production is (LHS, RHS_tokens)
- a simple left-most derivation parser that yields production id sequence for a canonical PDE string
- utilities to get valid productions for the current nonterminal (masking)

The grammar enforces operator precedence and produces deterministic left-most derivations.
"""

from typing import List, Tuple, Dict, Optional
import re

from .normalize import strip_eq0 as _strip_eq0_shared, normalize_power as _normalize_power_shared

# Production: Tuple[LHS: str, RHS: List[str]]
Productions: List[Tuple[str, List[str]]] = []

# Build index maps (kept dynamic so adding/removing productions updates maps)
PROD_COUNT = 0
PROD_ID: Dict[int, Tuple[str, List[str]]] = {}
LHS_TO_PRODS: Dict[str, List[int]] = {}
PAD_PROD_ID: Optional[int] = None


def refresh_grammar() -> None:
    """Rebuild derived grammar maps from `Productions`.

    Call this after adding/removing productions to keep `PROD_COUNT`, `PROD_ID`,
    `LHS_TO_PRODS`, and `PAD_PROD_ID` up to date. This makes the grammar module
    robust to further edits at runtime or during development.
    """
    global PROD_COUNT, PROD_ID, LHS_TO_PRODS, PAD_PROD_ID
    PROD_COUNT = len(Productions)
    PROD_ID = {i: Productions[i] for i in range(PROD_COUNT)}
    LHS_TO_PRODS = {}
    for pid, (lhs, rhs) in PROD_ID.items():
        LHS_TO_PRODS.setdefault(lhs, []).append(pid)
    PAD_PROD_ID = next((i for i, (lhs, _) in PROD_ID.items() if lhs == 'PAD'), None)


# Helper to add productions and maintain order (ids are indices)
def add(lhs: str, rhs: List[str]):
    Productions.append((lhs, rhs))
    refresh_grammar()

# Grammar design (unambiguous, expression precedence):
# PDE -> EXPR '=' '0'
# EXPR -> SUM
# SUM -> SUM '+' PROD | SUM '-' PROD | PROD   (we will encode as left-factored)
# To keep left-most deterministic we rewrite using left recursion elimination:
# SUM -> PROD SUM_T
# SUM_T -> '+' PROD SUM_T | '-' PROD SUM_T | epsilon
# PROD -> PROD '*' FACT | FACT   (again left-factored)
# PROD -> FACT PROD_T
# PROD_T -> '*' FACT PROD_T | epsilon
# FACT -> ATOM | ATOM '^' NUM
# ATOM -> derivative | nonlinear | 'u' | NUMBER | '(' EXPR ')'

# Terminals used: dt(u), dtt(u), dx(u), dxx(u), dxxx(u), dxxxx(u), dxxyy(u), dxxzz(u), dyyzz(u)
#                u, u^2, u^3, u*dx(u), (dx(u))^2, numbers (like 1.234), +, -, '*', '^', '(', ')', '=', '0'

# Nonterminals: PDE, SUM, SUM_T, PROD, PROD_T, FACT, ATOM, NUM
# Simplified: PDE is just the expression, no '= 0' suffix
# Optimized: Removed wrapper nonterminals PDE->EXPR and EXPR->SUM

add('PDE', ['PROD', 'SUM_T'])

add('SUM', ['PROD', 'SUM_T'])
add('SUM_T', ['+', 'PROD', 'SUM_T'])
add('SUM_T', ['-', 'PROD', 'SUM_T'])
add('SUM_T', [])  # epsilon

add('PROD', ['FACT', 'PROD_T'])
add('PROD_T', ['*', 'FACT', 'PROD_T'])
add('PROD_T', [])  # epsilon

add('FACT', ['ATOM'])
add('FACT', ['ATOM', '^', 'NUM'])

# ATOM alternatives
add('ATOM', ['dt', '(', 'u', ')'])
add('ATOM', ['dtt', '(', 'u', ')'])

# spatial derivatives (up to 4th order and some mixed)
add('ATOM', ['dx', '(', 'u', ')'])
add('ATOM', ['dy', '(', 'u', ')'])
add('ATOM', ['dz', '(', 'u', ')'])
add('ATOM', ['dxx', '(', 'u', ')'])
add('ATOM', ['dyy', '(', 'u', ')'])
add('ATOM', ['dzz', '(', 'u', ')'])
add('ATOM', ['dxxx', '(', 'u', ')'])
add('ATOM', ['dyyy', '(', 'u', ')'])
add('ATOM', ['dzzz', '(', 'u', ')'])
add('ATOM', ['dxxxx', '(', 'u', ')'])
add('ATOM', ['dyyyy', '(', 'u', ')'])
add('ATOM', ['dzzzz', '(', 'u', ')'])
add('ATOM', ['dxxyy', '(', 'u', ')'])
add('ATOM', ['dxxzz', '(', 'u', ')'])
add('ATOM', ['dyyzz', '(', 'u', ')'])

# derivatives of nonlinear terms (for Cahn-Hilliard equations)
# IMPORTANT: Add before simpler patterns to ensure correct parsing
add('ATOM', ['dxx', '(', 'u', '^', '3', ')'])  # dxx(u^3)
add('ATOM', ['dyy', '(', 'u', '^', '3', ')'])  # dyy(u^3)

# nonlinear atoms - IMPORTANT: longer patterns FIRST (greedy matching)!
add('ATOM', ['(', 'dx', '(', 'u', ')', ')', '^', '2'])  # (dx(u))^2
add('ATOM', ['u', '*', 'dx', '(', 'u', ')'])
add('ATOM', ['u', '*', 'dy', '(', 'u', ')'])
add('ATOM', ['u', '*', 'dz', '(', 'u', ')'])
add('ATOM', ['sin', '(', 'u', ')'])  # sin(u)
add('ATOM', ['u', '^', '3'])  # u^3 before u^2
add('ATOM', ['u', '^', '2'])  # u^2 before u
add('ATOM', ['u'])  # plain u LAST

# numeric constant as ATOM (we fold NUM separately)
add('ATOM', ['NUM'])

# NUM -> digit-by-digit parsing (following Lample & Charton)
# This ensures different coefficients have different production sequences!
# Format: digits with optional decimal point (no sign - all numbers are positive)
# Signs are handled by the SUM_T operators (+ and -)
add('NUM', ['DIGITS'])

# DIGITS can be:
# - single digit
# - digit followed by more digits
# - digit followed by decimal point and fractional part
# IMPORTANT: List specific/longer matches BEFORE epsilon!
add('DIGITS', ['DIGIT', 'DIGITS_REST'])
add('DIGITS_REST', ['.', 'FRAC'])  # decimal point + fractional part (try first!)
add('DIGITS_REST', ['DIGIT', 'DIGITS_REST'])  # more integer digits
add('DIGITS_REST', [])  # epsilon (just integer) - LAST so we try longer matches first

# Fractional part: at least one digit, possibly more
add('FRAC', ['DIGIT', 'FRAC'])  # recursive case first
add('FRAC', ['DIGIT'])  # base case last

# Individual digit productions (0-9)
add('DIGIT', ['0'])
add('DIGIT', ['1'])
add('DIGIT', ['2'])
add('DIGIT', ['3'])
add('DIGIT', ['4'])
add('DIGIT', ['5'])
add('DIGIT', ['6'])
add('DIGIT', ['7'])
add('DIGIT', ['8'])
add('DIGIT', ['9'])

# explicit PAD production for padded timesteps (helps decoders treat pads as a real class)
add('PAD', [])  # will be the last production id by design

# refresh_grammar() was called by add() so derived maps are already built


class ParseError(Exception):
    pass


def strip_eq0(pde: str) -> str:
    """Remove a trailing '= 0' (with any spacing) if present.

    The grammar itself is for the *expression only* (no '=0' suffix), but many
    datasets store PDEs in the form 'EXPR = 0'. We accept both here.
    """
    # Backward-compatible wrapper (other modules may import pde.grammar.strip_eq0).
    return _strip_eq0_shared(pde)


def tokenize_canonical(pde: str) -> List[str]:
    """Tokenize canonical PDE string into tokens matching grammar terminals.

    This tokenizer expects the canonical format (spaces around + and -)
    and numbers with decimals. It returns tokens split digit-by-digit:
    '0.357' -> ['0', '.', '3', '5', '7']
    
    Special cases:
    - Power exponents (u^2, u^3, etc.) are kept as literal tokens '2', '3'
    - Only coefficient numbers are split digit-by-digit
    
    PDE format is just the expression (no '= 0' suffix).
    
    This ensures different coefficients have different token sequences!
    """
    # Use regex to find all tokens
    # Number pattern: digits, optional decimal + more digits
    pattern = r'(dtt|dt|sin|dxxyy|dxxzz|dyyzz|dxxxx|dyyyy|dzzzz|dxxx|dyyy|dzzz|dxx|dyy|dzz|dx|dy|dz|u|\d+\.\d+|\d+|\^|\*|\+|-|\(|\)|\s+)'
    
    tokens: List[str] = []
    pde = strip_eq0(pde)
    # Normalize python-style exponentiation into grammar exponent token.
    # The grammar supports '^' but operator-only datasets may emit '**' (e.g., u**3).
    pde = _normalize_power_shared(pde)
    parts = re.findall(pattern, pde)
    
    for i, part in enumerate(parts):
        if part.isspace():
            continue
        elif re.match(r'\d+(?:\.\d+)?$', part):
            # Check if previous token is '^' - if so, this is a power exponent, keep as literal
            if tokens and tokens[-1] == '^':
                tokens.append(part)  # Keep '2' or '3' as literal token
            else:
                # Regular coefficient: split into individual digits/decimal point (no sign prefix)
                for char in part:
                    tokens.append(char)
        else:
            # All other tokens (derivatives, operators, parens, etc.)
            tokens.append(part)
    
    return tokens


def match_rhs(tokens: List[str], pos: int, rhs: List[str]) -> Optional[int]:
    """Attempt to match RHS at tokens[pos:]. Return new pos if matches, else None.

    For nonterminals we recursively try to parse them (left-most); for terminals, match exact token.
    """
    cur = pos
    for sym in rhs:
        if sym in LHS_TO_PRODS:  # nonterminal
            # parse nonterminal by trying its productions in order
            parsed = False
            for pid in LHS_TO_PRODS[sym]:
                lhs, subrhs = PROD_ID[pid]
                newpos = match_rhs(tokens, cur, subrhs)
                if newpos is not None:
                    cur = newpos
                    parsed = True
                    break
            if not parsed:
                return None
        else:
            # terminal: exact match
            if cur < len(tokens) and tokens[cur] == sym:
                cur += 1
            else:
                return None
    return cur


def parse_to_productions(pde: str) -> List[int]:
    """Parse canonical PDE string and return left-most production id sequence.

    This is a straightforward recursive left-most parser that tries productions in the order
    they are defined for each nonterminal. The grammar is designed to be unambiguous so this
    deterministic approach should succeed for canonical inputs.
    """
    tokens = tokenize_canonical(pde)

    prod_seq: List[int] = []

    # inner recursive function to expand a nonterminal at current token position
    def expand(sym: str, pos: int) -> Optional[int]:
        # try productions for sym in order
        for pid in LHS_TO_PRODS.get(sym, []):
            lhs, rhs = PROD_ID[pid]
            # attempt match: match_rhs will attempt to match terminals and recursively nonterminals
            newpos = match_rhs(tokens, pos, rhs)
            if newpos is not None:
                # record production id and—but we must also record productions for any nonterminal expansions inside rhs
                # To produce a left-most production sequence, we traverse rhs left to right and whenever we encounter a
                # nonterminal, we recursively expand it and record its productions in order. Since match_rhs already
                # matched them by trying productions in order, here we must re-run the expansion to collect ids.
                prod_seq.append(pid)
                cur = pos
                for sym2 in rhs:
                    if sym2 in LHS_TO_PRODS:
                        # expand and update cur
                        res = expand(sym2, cur)
                        if res is None:
                            raise ParseError(f"Expansion failed for {sym2} at {cur}")
                        cur = res
                    else:
                        # terminal: skip it (already matched in match_rhs)
                        cur += 1
                return newpos
        return None

    endpos = expand('PDE', 0)
    if endpos is None or endpos != len(tokens):
        raise ParseError(f"Failed to parse PDE. Tokens: {tokens}")
    return prod_seq


def valid_productions_for_nonterminal(nonterminal: str) -> List[int]:
    """Return production ids that have LHS == nonterminal (in order)."""
    return LHS_TO_PRODS.get(nonterminal, [])


def build_masks_from_production_sequence(prod_seq: List[int]) -> List[List[int]]:
    """Given a production sequence (left-most), produce masks per step: at each step, mask indicates
    which production ids are valid choices (1) and others 0. The valid productions are those whose LHS
    equals the current left-most nonterminal during the derivation.
    """
    masks: List[List[int]] = []
    # simulate stack of symbols for left-most derivation
    stack: List[str] = ['PDE']
    for pid in prod_seq:
        # pop terminals until we find a nonterminal at left-most position
        while stack and stack[0] not in LHS_TO_PRODS:
            # consume terminal
            stack.pop(0)
        if not stack:
            raise ParseError('Empty stack during derivation')
        # If this step is an explicit PAD production, allow only PAD and do not change stack
        if PAD_PROD_ID is not None and pid == PAD_PROD_ID:
            valid = [1 if i == PAD_PROD_ID else 0 for i in range(PROD_COUNT)]
            masks.append(valid)
            # do not modify the derivation stack for padded timestep
            continue

        cur = stack.pop(0)
        # valid productions are those with this LHS
        valid = [1 if PROD_ID[i][0] == cur else 0 for i in range(PROD_COUNT)]
        masks.append(valid)
        # apply the actual production (prod_seq step)
        lhs, rhs = PROD_ID[pid]
        if lhs != cur:
            raise ParseError(f'Production LHS {lhs} does not match stack symbol {cur}')
        # expand rhs: insert at front (left-most), but since we popped left-most, we insert rhs at front
        # we need to place terminals/nonterminals preserving left-to-right order
        # we'll convert rhs list into symbols and insert at front of stack
        stack = rhs + stack
    return masks


def pad_production_sequence(prod_seq: List[int], max_len: int, pad_value: int = -1) -> List[int]:
    """Pad or truncate a production id sequence to length max_len.

    Padding uses `pad_value` (default -1). Downstream one-hot helpers treat out-of-range
    or negative ids as producing zero vectors, which is compatible with Grammar-VAE padding.
    """
    if len(prod_seq) >= max_len:
        return prod_seq[:max_len]
    # if pad_value is negative, use PAD_PROD_ID if available for explicit padding
    if pad_value is None or pad_value < 0:
        pad_value = PAD_PROD_ID if PAD_PROD_ID is not None else -1
    return prod_seq + [pad_value] * (max_len - len(prod_seq))


def decode_production_sequence(prod_ids: List[int]) -> str:
    """Decode a production id sequence back to a PDE string using stack-based derivation.
    
    This follows the Grammar VAE approach (Kusner et al.): apply productions in order
    using a stack to track which nonterminals need to be expanded. Each production
    expands the leftmost nonterminal on the stack.
    
    Args:
        prod_ids: List of production rule IDs (integers)
        
    Returns:
        The reconstructed PDE string, or empty string if invalid
    """
    # Filter out padding (-1) and invalid IDs
    valid_ids = [pid for pid in prod_ids if 0 <= pid < PROD_COUNT]
    
    if not valid_ids:
        return ""
    
    # Start with PDE nonterminal on stack
    stack = ['PDE']
    result = []  # Collect terminals as we go
    
    # Apply each production
    for pid in valid_ids:
        # Pop terminals from the left until we find a nonterminal
        while stack and stack[0] not in LHS_TO_PRODS:
            terminal = stack.pop(0)
            result.append(terminal)  # Collect terminal
        
        if not stack:
            # Stack is empty, no more nonterminals to expand
            break
        
        # Get leftmost nonterminal
        nonterminal = stack.pop(0)
        
        # Get the production rule
        lhs, rhs = PROD_ID[pid]
        
        # Check if production matches current nonterminal
        if lhs != nonterminal:
            # Invalid sequence - production doesn't match expected nonterminal
            # This can happen with unconstrained decoding from VAE
            # Try to continue with what we have
            break
        
        # Expand: add RHS to front of stack (left-to-right order)
        stack = list(rhs) + stack
    
    # Collect any remaining terminals from stack
    # Skip any remaining nonterminals (incomplete derivation)
    for symbol in stack:
        if symbol not in LHS_TO_PRODS:  # It's a terminal
            result.append(symbol)
    
    return ''.join(result)


if __name__ == '__main__':
    # quick test
    examples = [
        'dt(u) - 1.935*dxx(u) = 0',
        'dtt(u) - 4.759*dxx(u) - 4.759*dyy(u) = 0',
        'dt(u) - 1.467*dxx(u) + u*dx(u) = 0'
    ]
    for e in examples:
        print('\nPDE:', e)
        toks = tokenize_canonical(e)
        print('TOKENS:', toks)
        seq = parse_to_productions(e)
        print('PRODUCTION IDS:', seq)
        
        # Test decode
        decoded = decode_production_sequence(seq)
        print('DECODED:', decoded)
        print('MATCH:', decoded == e.replace(' = 0', '').replace(' ', ''))
        
        masks = build_masks_from_production_sequence(seq)
        print('MASKS (per step, count valid):', [sum(m) for m in masks])
