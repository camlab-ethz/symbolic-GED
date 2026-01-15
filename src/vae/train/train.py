#!/usr/bin/env python3
"""Train VAE with Grammar or Token tokenization.

This script provides a clean CLI for training VAEs on PDE sequences.
Supports both grammar-based (production sequences) and token-based (character sequences) inputs.

Examples:
    # Train with grammar tokenization (using config defaults)
    python -m vae.train --tokenization grammar

    # Train with token tokenization
    python -m vae.train --tokenization token

    # Override hyperparameters
    python -m vae.train --tokenization grammar --lr 0.0005 --batch_size 512

    # Use specific config file
    python -m vae.train --config my_config.yaml --tokenization grammar
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    TQDMProgressBar,
    LearningRateMonitor,
    Callback,
)

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup path to import from parent vae module
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from vae.module import VAEModule
from vae.utils import GrammarVAEDataModule, TokenVAEDataModule
from vae.config import VAEConfig, merge_cli_args

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BestAccuracyLogger(Callback):
    """Callback to log best accuracy to stderr when checkpoint is saved."""

    def __init__(self):
        super().__init__()
        self.best_acc = None
        self.best_epoch = None

    def on_validation_end(self, trainer, pl_module):
        """Log best accuracy after validation if it improved."""
        # Find ModelCheckpoint callback
        checkpoint_cb = None
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                checkpoint_cb = cb
                break

        if checkpoint_cb is None:
            return

        current_acc = checkpoint_cb.best_model_score
        if current_acc is not None:
            current_acc = float(current_acc)
            # Check if this is a new best
            if self.best_acc is None or current_acc > self.best_acc:
                self.best_acc = current_acc
                self.best_epoch = trainer.current_epoch
                # Log to stderr (goes to .err file)
                import sys

                checkpoint_path = checkpoint_cb.best_model_path or "N/A"
                print(
                    f"\n{'='*60}\n"
                    f"🏆 NEW BEST MODEL SAVED!\n"
                    f"   Epoch: {self.best_epoch}\n"
                    f"   Best val/seq_acc: {self.best_acc:.6f} ({self.best_acc*100:.2f}%)\n"
                    f"   Checkpoint: {checkpoint_path}\n"
                    f"{'='*60}\n",
                    file=sys.stderr,
                    flush=True,
                )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train VAE with Grammar or Token tokenization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    parser.add_argument(
        "--tokenization",
        type=str,
        required=True,
        choices=["grammar", "token"],
        help="Tokenization method",
    )

    # Training overrides
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)

    # KL annealing
    parser.add_argument(
        "--kl_anneal_epochs",
        type=int,
        default=None,
        help="Epochs to anneal beta from 0 to target",
    )
    parser.add_argument(
        "--cyclical_beta", action="store_true", help="Use cyclical KL annealing"
    )
    parser.add_argument(
        "--cycle_epochs",
        type=int,
        default=None,
        help="Epochs per cycle for cyclical annealing",
    )

    # LR scheduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default=None,
        choices=["none", "plateau", "cosine", "step", "exponential"],
    )
    parser.add_argument("--lr_scheduler_patience", type=int, default=None)
    parser.add_argument("--lr_scheduler_factor", type=float, default=None)
    parser.add_argument("--lr_scheduler_min", type=float, default=None)

    # System
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every_n_steps", type=int, default=50)

    return parser.parse_args()


def load_config(args) -> VAEConfig:
    """Load configuration from file or defaults, then apply CLI overrides."""
    if args.config:
        logger.info(f"Loading config from {args.config}")
        config = VAEConfig.from_yaml(args.config)
    else:
        # Load default config from config_vae.yaml if it exists
        # Path: vae/train/train.py -> vae/ -> src/ -> config_vae.yaml
        default_config_path = Path(__file__).parent.parent.parent / "config_vae.yaml"
        if default_config_path.exists():
            logger.info(f"Loading default config from {default_config_path}")
            # Use proper config loading method
            try:
                config = VAEConfig.from_yaml(default_config_path)
                config.tokenization = args.tokenization  # Override with CLI arg
                logger.info(
                    f"  ✓ Loaded config via from_yaml(): beta={config.training.kl_annealing.beta}, "
                    f"early_stopping_patience={config.training.early_stopping_patience}"
                )
            except Exception as e:
                logger.warning(f"Failed to load config via from_yaml(): {e}")
                logger.warning(f"Using fallback parser...")
                import traceback

                logger.debug(f"Traceback: {traceback.format_exc()}")
                # Fallback to old parser for backward compatibility
                with open(default_config_path) as f:
                    data = yaml.safe_load(f)
                config = VAEConfig(tokenization=args.tokenization)
                if "training" in data:
                    training_data = data["training"]
                    config.training.batch_size = training_data.get("batch_size", 256)
                    config.training.lr = training_data.get("learning_rate", 0.001)
                    config.training.max_epochs = training_data.get("epochs", 1000)
                    config.training.gradient_clip = training_data.get("clip", 1.0)
                    config.training.early_stopping_patience = training_data.get(
                        "early_stopping_patience", 100  # Updated default
                    )
                    # Handle beta: can be at training.beta or training.kl_annealing.beta
                    if (
                        "kl_annealing" in training_data
                        and isinstance(training_data["kl_annealing"], dict)
                        and "beta" in training_data["kl_annealing"]
                    ):
                        config.training.kl_annealing.beta = training_data[
                            "kl_annealing"
                        ]["beta"]
                        logger.info(
                            f"  ✓ Read beta from kl_annealing.beta: {config.training.kl_annealing.beta}"
                        )
                    elif "beta" in training_data:
                        config.training.kl_annealing.beta = training_data["beta"]
                        logger.info(
                            f"  ✓ Read beta from training.beta: {config.training.kl_annealing.beta}"
                        )
                    else:
                        config.training.kl_annealing.beta = 0.001  # Updated default
                        logger.warning(
                            f"  ⚠️  Using default beta: {config.training.kl_annealing.beta}"
                        )

                logger.info(
                    f"  ✓ Fallback parser loaded: beta={config.training.kl_annealing.beta}, "
                    f"early_stopping_patience={config.training.early_stopping_patience}"
                )

                # LR scheduler from old config
                if "lr_scheduler" in data["training"]:
                    lr_sched = data["training"]["lr_scheduler"]
                    if lr_sched and lr_sched != "null":
                        config.training.lr_scheduler.type = lr_sched
                    else:
                        config.training.lr_scheduler.type = None
                if "lr_scheduler_patience" in data["training"]:
                    config.training.lr_scheduler.patience = data["training"][
                        "lr_scheduler_patience"
                    ]
                if "lr_scheduler_factor" in data["training"]:
                    config.training.lr_scheduler.factor = data["training"][
                        "lr_scheduler_factor"
                    ]
                if "lr_scheduler_min" in data["training"]:
                    config.training.lr_scheduler.min_lr = data["training"][
                        "lr_scheduler_min"
                    ]
        else:
            logger.info("Using default configuration")
            config = VAEConfig(tokenization=args.tokenization)

    # Apply CLI overrides
    config.tokenization = args.tokenization
    config = merge_cli_args(config, args)

    return config


def create_datamodule(config: VAEConfig, num_workers: int):
    """Create appropriate datamodule based on tokenization."""
    base_dir = BASE_DIR  # Use the BASE_DIR from module level

    if config.tokenization == "grammar":
        dm = GrammarVAEDataModule(
            prod_path=str(base_dir / config.ids_path),
            masks_path=str(base_dir / config.masks_path),
            batch_size=config.training.batch_size,
            num_workers=num_workers,
            split_dir=str(base_dir / config.data.split_dir),
        )
    else:
        dm = TokenVAEDataModule(
            token_path=str(base_dir / config.ids_path),
            masks_path=str(base_dir / config.masks_path),
            batch_size=config.training.batch_size,
            num_workers=num_workers,
            split_dir=str(base_dir / config.data.split_dir),
            vocab_size=config.vocab_size,
        )

    return dm


def create_callbacks(config: VAEConfig, checkpoint_dir: str):
    """Create training callbacks."""
    callbacks = []

    # Checkpoint: save best model
    checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-{epoch:03d}-{val/seq_acc:.4f}",
        save_top_k=config.logging.save_top_k,
        monitor="val/seq_acc",
        mode="max",
    )
    callbacks.append(checkpoint_cb)

    # Best accuracy logger (logs to stderr)
    callbacks.append(BestAccuracyLogger())

    # Early stopping
    if config.training.early_stopping_patience > 0:
        early_stop_cb = EarlyStopping(
            monitor="val/seq_acc",
            patience=config.training.early_stopping_patience,
            mode="max",
        )
        callbacks.append(early_stop_cb)

    # Progress bar
    callbacks.append(TQDMProgressBar(refresh_rate=config.logging.log_every_n_steps))

    # LR monitor
    if config.training.lr_scheduler.type:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    return callbacks


def print_config(config: VAEConfig):
    """Print configuration summary."""
    logger.info("=" * 60)
    logger.info(f"VAE Training - {config.tokenization.upper()} tokenization")
    logger.info("=" * 60)
    logger.info(f"Model:")
    logger.info(f"  z_dim: {config.model.z_dim}")
    logger.info(
        f"  encoder: hidden={config.model.encoder.hidden_size}, "
        f"kernels={config.model.encoder.kernel_sizes}"
    )
    logger.info(
        f"  decoder: hidden={config.model.decoder.hidden_size}, "
        f"layers={config.model.decoder.num_layers}"
    )
    logger.info(f"Training:")
    logger.info(f"  batch_size: {config.training.batch_size}")
    logger.info(f"  lr: {config.training.lr}")
    logger.info(f"  max_epochs: {config.training.max_epochs}")
    logger.info(f"  beta: {config.training.kl_annealing.beta}")
    if config.training.kl_annealing.anneal_epochs > 0:
        logger.info(
            f"  KL annealing: linear over {config.training.kl_annealing.anneal_epochs} epochs"
        )
    elif config.training.kl_annealing.cyclical:
        logger.info(
            f"  KL annealing: cyclical ({config.training.kl_annealing.cycle_epochs} epochs/cycle)"
        )
    if config.training.lr_scheduler.type:
        logger.info(f"  LR scheduler: {config.training.lr_scheduler.type}")
    logger.info(f"Data:")
    logger.info(f"  max_length: {config.max_length}")
    logger.info(f"  vocab_size: {config.vocab_size}")
    logger.info("=" * 60)


def main():
    """Main training function."""
    args = parse_args()

    # Load configuration
    config = load_config(args)
    print_config(config)

    # Set seed for reproducibility (before any random operations)
    if config.seed is not None:
        import random
        import numpy as np

        pl.seed_everything(config.seed, workers=True)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        logger.info(
            f"✅ Set random seed: {config.seed} (PyTorch, NumPy, Python random)"
        )
    else:
        logger.warning("⚠️  No seed specified - results may not be reproducible!")

    # Create data module
    dm = create_datamodule(config, args.num_workers)
    logger.info(f"Created datamodule: {dm.N} sequences")

    # Create model
    model = VAEModule.from_config(config)
    logger.info("Created VAE model")

    # Setup checkpoint directory - separate by tokenization type, beta, and seed
    base_dir = BASE_DIR

    # Format beta for filesystem-safe path
    beta_val = config.training.kl_annealing.beta
    if beta_val >= 1.0:
        beta_tag = f"{beta_val:.0f}"
    elif beta_val >= 0.01:
        beta_tag = f"{beta_val:.2f}"
    else:
        beta_tag = f"{beta_val:.0e}".replace("e-0", "e-").replace("e+0", "e+")

    seed_val = config.seed if config.seed is not None else 0

    # Respect config.logging.checkpoint_dir (relative paths are resolved from repo root).
    ckpt_root = (
        Path(config.logging.checkpoint_dir)
        if getattr(config, "logging", None)
        else Path("checkpoints")
    )
    if not ckpt_root.is_absolute():
        ckpt_root = base_dir / ckpt_root

    checkpoint_dir = (
        ckpt_root / f"{config.tokenization}_vae" / f"beta_{beta_tag}_seed_{seed_val}"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoint directory: {checkpoint_dir}")

    # Create callbacks
    callbacks = create_callbacks(config, str(checkpoint_dir))

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else "auto",
        callbacks=callbacks,
        log_every_n_steps=args.log_every_n_steps,
        gradient_clip_val=config.training.gradient_clip,
        gradient_clip_algorithm="norm",
    )

    # Train
    logger.info("Starting training...")
    trainer.fit(model, dm)
    logger.info("Training complete!")

    # Print best checkpoint path and accuracy
    if trainer.checkpoint_callback:
        best_path = trainer.checkpoint_callback.best_model_path
        best_score = trainer.checkpoint_callback.best_model_score
        logger.info(f"Best checkpoint: {best_path}")
        if best_score is not None:
            logger.info(f"Best val/seq_acc: {best_score:.6f} ({best_score*100:.2f}%)")
            # Also log to stderr for easy finding in error files
            import sys

            print(
                f"\n{'='*60}\n"
                f"✅ TRAINING COMPLETE\n"
                f"   Best val/seq_acc: {best_score:.6f} ({best_score*100:.2f}%)\n"
                f"   Best checkpoint: {best_path}\n"
                f"{'='*60}\n",
                file=sys.stderr,
                flush=True,
            )


if __name__ == "__main__":
    main()
