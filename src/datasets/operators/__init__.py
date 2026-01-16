"""
Operator Dataset Module

This module provides tools for creating PDE operator string datasets.

Components:
- generator.py: PDEGenerator class for sampling operator strings
- create_data_splits.py: Split dataset into train/val/test
- create_tokenized_data.py: Tokenize operator strings
- fix_dataset_labels.py: Fix known label issues
- validate_dataset.py: Validate dataset integrity

Usage:
    from datasets.operators import PDEGenerator
    gen = PDEGenerator(seed=42)
    pde = gen.generate_pde("heat", dim=2)
"""

from datasets.operators.generator import PDEGenerator

__all__ = ["PDEGenerator"]
