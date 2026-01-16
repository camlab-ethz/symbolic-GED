"""
Datasets Module

This module provides tools for creating different types of PDE datasets.

Submodules:
- operators/: PDE operator string generation (L) for VAE training
- manufactured/: Manufactured solutions (u, f=L(u)) for operator identification benchmark
"""

from datasets.operators.generator import PDEGenerator

__all__ = ["PDEGenerator"]
