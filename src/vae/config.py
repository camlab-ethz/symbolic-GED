"""Configuration dataclasses for VAE training.

Supports YAML loading/saving and CLI overrides.

Example:
    # Load from YAML
    config = VAEConfig.from_yaml('config.yaml')

    # Create programmatically
    config = VAEConfig(
        model=ModelConfig(z_dim=32),
        training=TrainingConfig(lr=0.001)
    )

    # Save to YAML
    config.to_yaml('my_config.yaml')
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Literal, Union
from pathlib import Path
import yaml


@dataclass
class EncoderConfig:
    """Encoder architecture configuration."""

    hidden_size: int = 128
    conv_sizes: List[int] = field(default_factory=lambda: [64, 128, 256])
    kernel_sizes: List[int] = field(default_factory=lambda: [7, 7, 7])
    dropout: float = 0.1

    @property
    def num_layers(self) -> int:
        return len(self.conv_sizes)


@dataclass
class DecoderConfig:
    """Decoder architecture configuration."""

    hidden_size: int = 80
    num_layers: int = 3
    dropout: float = 0.1
    bidirectional: bool = False
    rnn_type: Literal["gru", "lstm"] = "gru"


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    z_dim: int = 26
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)


@dataclass
class KLAnnealingConfig:
    """KL divergence annealing configuration."""

    beta: float = 0.0001  # Final/max KL weight
    free_bits: float = 0.5  # Minimum KL per dimension
    anneal_epochs: int = 0  # Linear annealing epochs (0 = no annealing)
    cyclical: bool = False  # Use cyclical annealing
    cycle_epochs: int = 10  # Epochs per cycle


@dataclass
class LRSchedulerConfig:
    """Learning rate scheduler configuration."""

    type: Optional[Literal["plateau", "cosine", "step", "exponential"]] = "plateau"
    patience: int = 10  # For plateau/step
    factor: float = 0.5  # Reduction factor
    min_lr: float = 1e-6  # Minimum learning rate
    T_max: Optional[int] = None  # For cosine (None = use max_epochs)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    batch_size: int = 256
    max_epochs: int = 1000
    lr: float = 0.001
    gradient_clip: float = 1.0
    early_stopping_patience: int = 20
    num_workers: int = 4

    # Annealing configs
    kl_annealing: KLAnnealingConfig = field(default_factory=KLAnnealingConfig)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)


@dataclass
class DataConfig:
    """Data paths and settings."""

    # Grammar tokenization - using new data/ structure (full files + split indices)
    grammar_ids_path: str = "data/tokenized/grammar_full.npy"
    grammar_masks_path: str = "data/tokenized/grammar_full_masks.npy"

    # Token tokenization - using new data/ structure (full files + split indices)
    token_ids_path: str = "data/tokenized/token_full.npy"
    token_masks_path: str = "data/tokenized/token_full_masks.npy"

    # Splits directory (contains train_indices.npy, val_indices.npy, test_indices.npy)
    split_dir: str = "data/splits"

    # Tokenization-specific
    grammar_max_length: int = 114
    grammar_vocab_size: int = 56
    token_max_length: int = 62
    token_vocab_size: int = 82


@dataclass
class LoggingConfig:
    """Logging and checkpointing configuration."""

    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    save_top_k: int = 1
    log_every_n_steps: int = 50
    print_every: int = 50  # Print detailed stats every N steps (0 = disabled)


@dataclass
class VAEConfig:
    """Complete VAE configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Experiment settings
    tokenization: Literal["grammar", "token"] = "grammar"
    seed: Optional[int] = 42

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "VAEConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "VAEConfig":
        """Create configuration from dictionary."""
        # Handle nested configs
        if "model" in data:
            model_data = data["model"]
            if "encoder" in model_data:
                model_data["encoder"] = EncoderConfig(**model_data["encoder"])
            if "decoder" in model_data:
                model_data["decoder"] = DecoderConfig(**model_data["decoder"])
            data["model"] = ModelConfig(**model_data)

        if "training" in data:
            training_data = data["training"].copy()  # Make a copy to avoid modifying original
            
            # Handle 'beta' at training level - move it to kl_annealing if not already there
            beta_at_top = training_data.pop("beta", None)
            if beta_at_top is not None:
                if "kl_annealing" not in training_data:
                    training_data["kl_annealing"] = {}
                if "beta" not in training_data["kl_annealing"]:
                    training_data["kl_annealing"]["beta"] = beta_at_top
            
            # Map 'epochs' -> 'max_epochs' for backward compatibility
            if "epochs" in training_data and "max_epochs" not in training_data:
                training_data["max_epochs"] = training_data.pop("epochs")
            # Map 'learning_rate' -> 'lr' for backward compatibility
            if "learning_rate" in training_data and "lr" not in training_data:
                training_data["lr"] = training_data.pop("learning_rate")
            # Map 'clip' -> 'gradient_clip' for backward compatibility
            if "clip" in training_data and "gradient_clip" not in training_data:
                training_data["gradient_clip"] = training_data.pop("clip")
            
            # Handle nested configs first (extract them before creating TrainingConfig)
            kl_annealing_data = training_data.pop("kl_annealing", None)
            lr_scheduler_data = training_data.pop("lr_scheduler", None)
            
            # Remove fields that aren't part of TrainingConfig dataclass
            # (device, accelerator, seed are used elsewhere, not in TrainingConfig)
            valid_fields = {"batch_size", "max_epochs", "lr", "gradient_clip", 
                          "early_stopping_patience", "num_workers"}
            filtered_training = {k: v for k, v in training_data.items() if k in valid_fields}
            
            # Create TrainingConfig with only valid fields
            training_config = TrainingConfig(**filtered_training)
            
            # Set nested configs
            if kl_annealing_data:
                training_config.kl_annealing = KLAnnealingConfig(**kl_annealing_data)
            if lr_scheduler_data:
                # Ensure numeric types are correct (YAML might read 1e-6 as string)
                if "min_lr" in lr_scheduler_data:
                    lr_scheduler_data["min_lr"] = float(lr_scheduler_data["min_lr"])
                if "factor" in lr_scheduler_data:
                    lr_scheduler_data["factor"] = float(lr_scheduler_data["factor"])
                if "patience" in lr_scheduler_data:
                    lr_scheduler_data["patience"] = int(lr_scheduler_data["patience"])
                training_config.lr_scheduler = LRSchedulerConfig(**lr_scheduler_data)
            
            data["training"] = training_config

        if "data" in data:
            data["data"] = DataConfig(**data["data"])

        if "logging" in data:
            data["logging"] = LoggingConfig(**data["logging"])

        # Filter out tokenization-specific sections (grammar/token) that aren't part of VAEConfig
        # These are used elsewhere in the training pipeline
        filtered_data = {k: v for k, v in data.items() 
                        if k not in ["grammar", "token"]}
        
        return cls(**filtered_data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @property
    def max_length(self) -> int:
        """Get max sequence length for current tokenization."""
        if self.tokenization == "grammar":
            return self.data.grammar_max_length
        return self.data.token_max_length

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size for current tokenization."""
        if self.tokenization == "grammar":
            return self.data.grammar_vocab_size
        return self.data.token_vocab_size

    @property
    def ids_path(self) -> str:
        """Get data path for current tokenization."""
        if self.tokenization == "grammar":
            return self.data.grammar_ids_path
        return self.data.token_ids_path

    @property
    def masks_path(self) -> str:
        """Get masks path for current tokenization."""
        if self.tokenization == "grammar":
            return self.data.grammar_masks_path
        return self.data.token_masks_path


def get_default_config() -> VAEConfig:
    """Get default configuration."""
    return VAEConfig()


def merge_cli_args(config: VAEConfig, args) -> VAEConfig:
    """Merge CLI arguments into config (CLI takes precedence)."""
    # Training overrides
    if hasattr(args, "batch_size") and args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if hasattr(args, "lr") and args.lr is not None:
        config.training.lr = args.lr
    if hasattr(args, "epochs") and args.epochs is not None:
        config.training.max_epochs = args.epochs
    if hasattr(args, "beta") and args.beta is not None:
        config.training.kl_annealing.beta = args.beta

    # KL annealing overrides
    if hasattr(args, "kl_anneal_epochs") and args.kl_anneal_epochs:
        config.training.kl_annealing.anneal_epochs = args.kl_anneal_epochs
    if hasattr(args, "cyclical_beta") and args.cyclical_beta:
        config.training.kl_annealing.cyclical = args.cyclical_beta
    if hasattr(args, "cycle_epochs") and args.cycle_epochs is not None:
        config.training.kl_annealing.cycle_epochs = args.cycle_epochs

    # LR scheduler overrides
    if hasattr(args, "lr_scheduler") and args.lr_scheduler is not None:
        lr_type = args.lr_scheduler if args.lr_scheduler != "none" else None
        config.training.lr_scheduler.type = lr_type
    if (
        hasattr(args, "lr_scheduler_patience")
        and args.lr_scheduler_patience is not None
    ):
        config.training.lr_scheduler.patience = args.lr_scheduler_patience
    if hasattr(args, "lr_scheduler_factor") and args.lr_scheduler_factor is not None:
        config.training.lr_scheduler.factor = args.lr_scheduler_factor
    if hasattr(args, "lr_scheduler_min") and args.lr_scheduler_min is not None:
        config.training.lr_scheduler.min_lr = args.lr_scheduler_min

    # Tokenization
    if hasattr(args, "tokenization") and args.tokenization is not None:
        config.tokenization = args.tokenization

    # Seed
    if hasattr(args, "seed") and args.seed is not None:
        config.seed = args.seed

    return config
