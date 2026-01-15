"""
PDE Module

Core PDE-related functionality:
- PDE family definitions and templates
- Grammar-based tokenization
- Token-based tokenization (Lample & Charton style)
"""

from .families import PDE_FAMILIES, get_family, list_families
from .grammar import (
    parse_to_productions,
    decode_production_sequence,
    pad_production_sequence,
    build_masks_from_production_sequence,
    PROD_COUNT,
)
from .chr_tokenizer import PDETokenizer, PDEVocabulary, SPECIAL_TOKENS

__all__ = [
    'PDE_FAMILIES',
    'get_family',
    'list_families',
    'parse_to_productions',
    'decode_production_sequence',
    'pad_production_sequence',
    'build_masks_from_production_sequence',
    'PROD_COUNT',
    'PDETokenizer',
    'PDEVocabulary',
    'SPECIAL_TOKENS',
]
