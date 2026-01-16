"""VAE package for PDE sequence encoding.

This package provides a clean, modular VAE implementation for encoding PDE sequences
using either grammar-based (production sequences) or token-based (character sequences) inputs.

Core Components:
    VAEModule: PyTorch Lightning module for VAE training
    VAEConfig: Dataclass-based configuration system
    GrammarVAEDataModule: DataModule for grammar tokenization
    TokenVAEDataModule: DataModule for token tokenization

Example:
    from vae import VAEModule, VAEConfig

    # Load config and create model
    config = VAEConfig.from_yaml('config.yaml')
    model = VAEModule.from_config(config)

    # Or create directly
    model = VAEModule(P=56, max_length=114, z_dim=26)
"""

from .config import (
    VAEConfig,
    ModelConfig,
    EncoderConfig,
    DecoderConfig,
    TrainingConfig,
    KLAnnealingConfig,
    LRSchedulerConfig,
    DataConfig,
    LoggingConfig,
)
from .module import VAEModule, GrammarVAEModule
from .architecture import Encoder, Decoder
from .utils import (
    GrammarVAEDataModule,
    TokenVAEDataModule,
    reparameterize,
    masked_cross_entropy,
    kl_divergence,
)

__all__ = [
    # Config
    "VAEConfig",
    "ModelConfig",
    "EncoderConfig",
    "DecoderConfig",
    "TrainingConfig",
    "KLAnnealingConfig",
    "LRSchedulerConfig",
    "DataConfig",
    "LoggingConfig",
    # Module
    "VAEModule",
    "GrammarVAEModule",  # Backward compatibility
    # Architecture
    "Encoder",
    "Decoder",
    # Data
    "GrammarVAEDataModule",
    "TokenVAEDataModule",
    # Utils
    "reparameterize",
    "masked_cross_entropy",
    "kl_divergence",
]
