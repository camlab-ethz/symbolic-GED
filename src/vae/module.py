"""VAE Lightning Module for Grammar and Token-based PDE encoding.

This module provides a clean, well-documented VAE implementation using PyTorch Lightning.
Supports both grammar-based (production sequences) and token-based (character sequences) inputs.

Example:
    # Using config (recommended)
    from vae.config import VAEConfig
    config = VAEConfig.from_yaml('config.yaml')
    model = VAEModule.from_config(config)

    # Direct instantiation (backward compatible)
    model = VAEModule(P=53, max_length=114, z_dim=26)
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import pytorch_lightning as pl

from .architecture import Encoder, Decoder
from .utils import (
    reparameterize,
    masked_cross_entropy,
    kl_divergence,
    kl_divergence_raw,
)

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class EpochMetrics:
    """Accumulator for epoch-level metrics."""

    rec_loss_sum: float = 0.0
    kl_sum: float = 0.0
    batch_count: int = 0
    token_correct: float = 0.0
    valid_count: float = 0.0
    seq_acc_sum: float = 0.0

    def reset(self):
        """Reset all accumulators."""
        self.rec_loss_sum = 0.0
        self.kl_sum = 0.0
        self.batch_count = 0
        self.token_correct = 0.0
        self.valid_count = 0.0
        self.seq_acc_sum = 0.0

    def update(
        self,
        rec_loss: float,
        kl: float,
        token_correct: float,
        valid_count: float,
        seq_acc: float,
    ):
        """Update accumulators with batch metrics."""
        self.rec_loss_sum += rec_loss
        self.kl_sum += kl
        self.batch_count += 1
        self.token_correct += token_correct
        self.valid_count += valid_count
        self.seq_acc_sum += seq_acc

    @property
    def avg_rec_loss(self) -> float:
        return self.rec_loss_sum / max(1, self.batch_count)

    @property
    def avg_kl(self) -> float:
        return self.kl_sum / max(1, self.batch_count)

    @property
    def token_acc(self) -> float:
        return self.token_correct / max(1.0, self.valid_count)

    @property
    def seq_acc(self) -> float:
        return self.seq_acc_sum / max(1, self.batch_count)


class VAEModule(pl.LightningModule):
    """Variational Autoencoder for sequence data (Grammar or Token-based).

    This module implements a VAE with:
    - Convolutional encoder with residual connections
    - GRU-based decoder (optionally bidirectional)
    - KL annealing (linear or cyclical)
    - Learning rate scheduling
    - Grammar-constrained decoding (for grammar tokenization)

    Attributes:
        P: Vocabulary size (number of productions or tokens)
        max_length: Maximum sequence length
        z_dim: Latent dimension
    """

    def __init__(
        self,
        P: int,
        max_length: int,
        z_dim: int = 26,
        # Training
        lr: float = 1e-3,
        # KL annealing
        beta: float = 0.0001,
        free_bits: float = 0.5,
        kl_anneal_epochs: int = 0,
        cyclical_beta: bool = False,
        cycle_epochs: int = 10,
        # Encoder
        encoder_hidden: int = 128,
        encoder_conv_layers: int = 3,
        encoder_kernel: Union[int, List[int]] = 7,
        encoder_dropout: float = 0.1,
        # Decoder
        decoder_hidden: int = 80,
        decoder_layers: int = 3,
        decoder_dropout: float = 0.1,
        decoder_bidirectional: bool = False,
        # LR Scheduler
        lr_scheduler: Optional[str] = None,
        lr_scheduler_patience: int = 10,
        lr_scheduler_factor: float = 0.5,
        lr_scheduler_min: float = 1e-6,
        lr_scheduler_T_max: Optional[int] = None,
        # Logging
        print_every: int = 0,  # 0 = disabled
    ):
        """Initialize VAE module.

        Args:
            P: Vocabulary size (productions or tokens)
            max_length: Maximum sequence length
            z_dim: Latent dimension
            lr: Learning rate
            beta: KL weight (final value after annealing)
            free_bits: Minimum total KL (prevents collapse, applied to total KL across all dimensions)
            kl_anneal_epochs: Epochs to linearly anneal beta (0 = no annealing)
            cyclical_beta: Use cyclical KL annealing
            cycle_epochs: Epochs per cycle for cyclical annealing
            encoder_hidden: Encoder hidden dimension
            encoder_conv_layers: Number of conv layers
            encoder_kernel: Kernel size(s) for conv layers
            encoder_dropout: Encoder dropout rate
            decoder_hidden: Decoder GRU hidden dimension
            decoder_layers: Number of GRU layers
            decoder_dropout: Decoder dropout rate
            decoder_bidirectional: Use bidirectional GRU
            lr_scheduler: Type of LR scheduler ('plateau', 'cosine', 'step', 'exponential')
            lr_scheduler_patience: Patience for plateau/step scheduler
            lr_scheduler_factor: LR reduction factor
            lr_scheduler_min: Minimum learning rate
            lr_scheduler_T_max: Max epochs for cosine scheduler
            print_every: Print detailed stats every N steps (0 = disabled)
        """
        super().__init__()
        self.save_hyperparameters()

        # Model dimensions
        self.P = P
        self.max_length = max_length
        self.z_dim = z_dim

        # Training params
        self.lr = lr

        # KL annealing
        self.beta = float(beta)
        self.free_bits = float(free_bits)
        self.kl_anneal_epochs = kl_anneal_epochs
        self.cyclical_beta = cyclical_beta
        self.cycle_epochs = cycle_epochs

        # LR scheduler
        self.lr_scheduler_type = lr_scheduler
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_min = lr_scheduler_min
        self.lr_scheduler_T_max = lr_scheduler_T_max

        # Logging
        self.print_every = print_every

        # Build networks
        self.encoder = Encoder(
            P=P,
            hidden_channels=encoder_hidden,
            z_dim=z_dim,
            conv_layers=encoder_conv_layers,
            kernel_size=encoder_kernel,
            dropout=encoder_dropout,
        )
        self.decoder = Decoder(
            P=P,
            z_dim=z_dim,
            hidden_dim=decoder_hidden,
            max_length=max_length,
            num_layers=decoder_layers,
            dropout=decoder_dropout,
            bidirectional=decoder_bidirectional,
        )

        # Metrics accumulators
        self._train_metrics = EpochMetrics()
        self._val_metrics = EpochMetrics()

        # Store last batch stats for epoch sanity logging (cheap)
        self._last_mu = None
        self._last_logvar = None

        # Grammar cache (for constrained decoding)
        self._grammar_cache = None

        logger.info(f"Initialized VAE: P={P}, T={max_length}, z_dim={z_dim}")

    @classmethod
    def from_config(cls, config: "VAEConfig") -> "VAEModule":
        """Create VAE from configuration object.

        Args:
            config: VAEConfig instance

        Returns:
            Initialized VAEModule
        """
        return cls(
            P=config.vocab_size,
            max_length=config.max_length,
            z_dim=config.model.z_dim,
            lr=config.training.lr,
            beta=config.training.kl_annealing.beta,
            free_bits=config.training.kl_annealing.free_bits,
            kl_anneal_epochs=config.training.kl_annealing.anneal_epochs,
            cyclical_beta=config.training.kl_annealing.cyclical,
            cycle_epochs=config.training.kl_annealing.cycle_epochs,
            encoder_hidden=config.model.encoder.hidden_size,
            encoder_conv_layers=config.model.encoder.num_layers,
            encoder_kernel=config.model.encoder.kernel_sizes,
            encoder_dropout=config.model.encoder.dropout,
            decoder_hidden=config.model.decoder.hidden_size,
            decoder_layers=config.model.decoder.num_layers,
            decoder_dropout=config.model.decoder.dropout,
            decoder_bidirectional=config.model.decoder.bidirectional,
            lr_scheduler=config.training.lr_scheduler.type,
            lr_scheduler_patience=config.training.lr_scheduler.patience,
            lr_scheduler_factor=config.training.lr_scheduler.factor,
            lr_scheduler_min=config.training.lr_scheduler.min_lr,
            lr_scheduler_T_max=config.training.lr_scheduler.T_max,
            print_every=config.logging.print_every,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through encoder and decoder.

        Args:
            x: Input sequences (B, T, P) one-hot encoded

        Returns:
            logits: Reconstruction logits (B, T, P)
            mu: Latent mean (B, z_dim)
            logvar: Latent log variance (B, z_dim)
        """
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        logits = self.decoder(z)
        return logits, mu, logvar

    def _compute_beta(self) -> float:
        """Compute current KL weight based on annealing schedule."""
        if self.cyclical_beta:
            cycle_pos = (self.current_epoch % self.cycle_epochs) / self.cycle_epochs
            return cycle_pos * self.beta
        elif self.kl_anneal_epochs > 0:
            anneal_factor = min(1.0, (self.current_epoch + 1) / self.kl_anneal_epochs)
            return anneal_factor * self.beta
        return self.beta

    def _apply_grammar_masks(
        self, logits: torch.Tensor, masks: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply grammar masks to logits.

        Sets invalid production logits to -inf at non-padding positions.

        Args:
            logits: Raw logits (B, T, P)
            masks: Grammar masks (B, T, P) with 1 for valid productions
            valid_mask: Padding mask (B, T) with 1 for non-padding

        Returns:
            Masked logits with invalid productions set to -inf
        """
        # Optimized: avoid clone if possible, use in-place operations where safe
        # We need to preserve gradients, so we'll use where() instead of masked_fill
        # which can be more efficient

        # Ensure masks are bool (do this once, not per call if possible)
        if masks.dtype != torch.bool:
            masks = masks.bool()

        # valid_mask is bool (B,T). Expand for broadcast.
        non_padding = valid_mask.unsqueeze(-1)  # (B, T, 1) bool

        # Create invalid mask: positions that are non-padding AND invalid productions
        invalid_mask = (~masks) & non_padding  # (B, T, P)

        # Use where() instead of clone + masked_fill for better performance
        # where() can be more efficient as it avoids the clone
        neg = torch.finfo(logits.dtype).min
        logits_masked = torch.where(invalid_mask, neg, logits)

        return logits_masked

    def _compute_metrics(
        self,
        logits_masked: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute accuracy metrics.

        Args:
            logits_masked: Masked logits (B, T, P)
            targets: Target sequences (B, T)
            valid_mask: Padding mask (B, T)

        Returns:
            Dictionary with token_acc, seq_acc, token_correct, valid_count
        """
        preds = logits_masked.argmax(dim=-1)
        valid_bool = valid_mask.bool()
        valid_count = valid_bool.sum().clamp(min=1)
        token_correct = ((preds == targets) & valid_bool).sum().float()
        token_acc = token_correct / valid_count
        seq_acc = (((preds == targets) | (~valid_bool)).all(dim=1)).float().mean()

        return {
            "token_acc": token_acc,
            "seq_acc": seq_acc,
            "token_correct": token_correct,
            "valid_count": valid_count,
        }

    def _shared_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], stage: str
    ) -> Dict[str, torch.Tensor]:
        """Shared forward step for training/validation/test.

        Args:
            batch: Tuple of (input, targets, masks)
            stage: 'train', 'val', or 'test'

        Returns:
            Dictionary with loss and metrics
        """
        x, targets, masks = batch
        logits, mu, logvar = self.forward(x)

        # Compute valid mask (non-padding positions)
        # Grammar VAE: masks is (B, T, P) - sum over P to get (B, T)
        # Token VAE: masks is (B, T) - use directly
        if masks.ndim == 3:
            # Prefer boolean masks throughout (avoid float > 0 comparisons later)
            mask_has_any = masks.bool().any(dim=-1)  # (B, T)

            # If targets are available, also gate by non-padding targets.
            # Convention assumed: padding targets are < 0 (e.g. -1).
            non_pad = targets >= 0  # (B, T)
            valid_mask = mask_has_any & non_pad  # (B, T) bool

            # Debug check: verify padding targets are -1 (non-fatal warning only)
            # Note: This is a sanity check and should not crash training
            # if padding convention differs between tokenizations

            # Apply grammar masks (only for Grammar VAE with 3D masks)
            logits_masked = self._apply_grammar_masks(logits, masks, valid_mask)
        else:
            # Token VAE doesn't use grammar masks; keep as bool and gate by targets
            valid_mask = masks.bool()  # (B, T)
            valid_mask = valid_mask & (targets >= 0)

            # Debug check: verify padding targets are -1 (non-fatal warning only)
            # Note: This is a sanity check and should not crash training
            # if padding convention differs between tokenizations
            # Token VAE doesn't use grammar masks, just use logits as-is
            logits_masked = logits

        # Compute losses
        rec_loss = masked_cross_entropy(logits_masked, targets, valid_mask)
        kl_raw = kl_divergence_raw(mu, logvar)
        kl = kl_divergence(mu, logvar, free_bits=self.free_bits)

        # Get beta (with annealing) - use same beta for all stages for consistency
        beta = self._compute_beta()
        loss = rec_loss + kl * beta

        # Compute metrics
        metrics = self._compute_metrics(logits_masked, targets, valid_mask)

        return {
            "loss": loss,
            "rec_loss": rec_loss,
            "kl": kl,
            "kl_raw": kl_raw,
            "beta": beta,
            **metrics,
            "mu": mu,
            "logvar": logvar,
        }

    def training_step(self, batch, batch_idx):
        """Training step."""
        results = self._shared_step(batch, "train")

        # Log metrics
        self.log(
            "train/loss", results["loss"], prog_bar=True, on_step=True, on_epoch=True
        )
        self.log(
            "train/rec_loss",
            results["rec_loss"],
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        self.log("train/kl", results["kl"], on_step=False, on_epoch=True)
        self.log("train/kl_raw", results["kl_raw"], on_step=False, on_epoch=True)
        self.log("train/beta", results["beta"], on_step=False, on_epoch=True)
        self.log(
            "train/token_acc",
            results["token_acc"],
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train/seq_acc",
            results["seq_acc"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # Monitor per-dimension KL (for collapse detection)
        per_dim_kl = -0.5 * (
            1 + results["logvar"] - results["mu"].pow(2) - results["logvar"].exp()
        )
        self.log("train/kl_per_dim", per_dim_kl.mean(), on_step=False, on_epoch=True)

        # Update epoch metrics
        self._train_metrics.update(
            float(results["rec_loss"]),
            float(results["kl"]),
            float(results["token_correct"]),
            float(results["valid_count"]),
            float(results["seq_acc"]),
        )

        # Store last batch for epoch sanity stats (cheap - overwrites each batch)
        self._last_mu = results["mu"].detach()
        self._last_logvar = results["logvar"].detach()

        # Optional verbose logging
        if self.print_every > 0 and self.global_step % self.print_every == 0:
            logger.info(
                f"step={self.global_step} epoch={self.current_epoch} "
                f"rec={results['rec_loss']:.4f} kl={results['kl']:.4f} "
                f"beta={results['beta']:.4f} token_acc={results['token_acc']:.4f}"
            )

        return results["loss"]

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        results = self._shared_step(batch, "val")

        # Log metrics
        self.log("val/loss", results["loss"])
        self.log("val/rec_loss", results["rec_loss"])
        self.log("val/kl", results["kl"])
        self.log("val/kl_raw", results["kl_raw"])
        self.log("val/beta", results["beta"], on_step=False, on_epoch=True)
        self.log("val/token_acc", results["token_acc"], prog_bar=False)
        self.log(
            "val/seq_acc",
            results["seq_acc"],
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # Update epoch metrics
        self._val_metrics.update(
            float(results["rec_loss"]),
            float(results["kl"]),
            float(results["token_correct"]),
            float(results["valid_count"]),
            float(results["seq_acc"]),
        )

    def test_step(self, batch, batch_idx):
        """Test step."""
        results = self._shared_step(batch, "test")

        self.log("test/loss", results["loss"], prog_bar=True)
        self.log("test/rec_loss", results["rec_loss"], prog_bar=True)
        self.log("test/kl", results["kl"], prog_bar=True)
        self.log("test/token_acc", results["token_acc"], prog_bar=True)
        self.log("test/seq_acc", results["seq_acc"], prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        if self.lr_scheduler_type is None:
            return optimizer

        scheduler_config = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "interval": "epoch",
                "frequency": 1,
            },
        }

        if self.lr_scheduler_type == "plateau":
            # Ensure min_lr is a float (not string from YAML)
            min_lr = (
                float(self.lr_scheduler_min)
                if self.lr_scheduler_min is not None
                else 1e-6
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(self.lr_scheduler_factor),
                patience=int(self.lr_scheduler_patience),
                min_lr=min_lr,
            )
            scheduler_config["lr_scheduler"]["scheduler"] = scheduler
            scheduler_config["lr_scheduler"]["monitor"] = "val/loss"

        elif self.lr_scheduler_type == "cosine":
            T_max = self.lr_scheduler_T_max or self.trainer.max_epochs
            # Ensure min_lr is a float (not string from YAML)
            eta_min = (
                float(self.lr_scheduler_min)
                if self.lr_scheduler_min is not None
                else 1e-6
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=T_max,
                eta_min=eta_min,
            )
            scheduler_config["lr_scheduler"]["scheduler"] = scheduler

        elif self.lr_scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(self.lr_scheduler_patience),
                gamma=float(self.lr_scheduler_factor),
            )
            scheduler_config["lr_scheduler"]["scheduler"] = scheduler

        elif self.lr_scheduler_type == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=float(self.lr_scheduler_factor),
            )
            scheduler_config["lr_scheduler"]["scheduler"] = scheduler

        else:
            raise ValueError(f"Unknown lr_scheduler: {self.lr_scheduler_type}")

        return scheduler_config

    def on_train_epoch_end(self):
        """Log epoch summary at end of training epoch."""
        m = self._train_metrics
        if m.batch_count > 0:
            # Compute epoch sanity stats from last batch (cheap approximation)
            beta = self._compute_beta()
            if self._last_mu is not None and self._last_logvar is not None:
                mean_mu_abs = self._last_mu.abs().mean().item()
                mean_exp_logvar = self._last_logvar.exp().mean().item()
                kl_per_dim = -0.5 * (
                    1
                    + self._last_logvar
                    - self._last_mu.pow(2)
                    - self._last_logvar.exp()
                )
                mean_kl_per_dim = kl_per_dim.mean().item()
            else:
                mean_mu_abs = 0.0
                mean_exp_logvar = 0.0
                mean_kl_per_dim = 0.0

            # Prioritize sequence accuracy - it's the most important metric!
            logger.info(
                f"[Train] epoch={self.current_epoch} "
                f"seq_acc={m.seq_acc:.4f} token_acc={m.token_acc:.4f} "
                f"rec={m.avg_rec_loss:.4f} kl={m.avg_kl:.4f} "
                f"|mu|={mean_mu_abs:.3f} exp(logvar)={mean_exp_logvar:.3f} "
                f"kl_per_dim={mean_kl_per_dim:.4f} beta={beta:.6f}"
            )
        self._train_metrics.reset()
        self._last_mu = None
        self._last_logvar = None

    def on_validation_epoch_end(self):
        """Log epoch summary at end of validation epoch."""
        m = self._val_metrics
        if m.batch_count > 0:
            # Get current learning rate
            try:
                current_lr = self.optimizers().param_groups[0]["lr"]
                self.log("lr", current_lr, on_epoch=True, logger=True)
                lr_str = f" lr={current_lr:.6f}"
            except:
                lr_str = ""

            # Prioritize sequence accuracy - it's the most important metric!
            logger.info(
                f"[Val] epoch={self.current_epoch} "
                f"seq_acc={m.seq_acc:.4f} token_acc={m.token_acc:.4f} "
                f"rec={m.avg_rec_loss:.4f} kl={m.avg_kl:.4f}{lr_str}"
            )
        self._val_metrics.reset()

    # ==================== Generation Methods ====================

    def _build_grammar_tables(self):
        """Build precomputed grammar tables for constrained decoding."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from pde import grammar as pde_grammar

        nonterminals = list(pde_grammar.LHS_TO_PRODS.keys())
        nonterminal_to_id = {nt: i for i, nt in enumerate(nonterminals)}
        num_nt = len(nonterminals)

        # Build mask table
        nonterminal_masks = torch.zeros(num_nt, self.P)
        for nt, nt_id in nonterminal_to_id.items():
            for pid in pde_grammar.LHS_TO_PRODS.get(nt, []):
                if pid < self.P:
                    nonterminal_masks[nt_id, pid] = 1.0

        # Build RHS nonterminal list
        prod_rhs_nonterminals = []
        for pid in range(pde_grammar.PROD_COUNT):
            lhs, rhs = pde_grammar.PROD_ID[pid]
            rhs_nt_ids = [
                nonterminal_to_id[sym] for sym in rhs if sym in nonterminal_to_id
            ]
            prod_rhs_nonterminals.append(rhs_nt_ids)

        start_nt_id = nonterminal_to_id.get("PDE", 0)
        pad_id = pde_grammar.PAD_PROD_ID if pde_grammar.PAD_PROD_ID is not None else 0

        return nonterminal_masks, prod_rhs_nonterminals, start_nt_id, pad_id

    def generate_constrained(
        self, z: torch.Tensor, greedy: bool = True, temperature: float = 1.0
    ) -> torch.Tensor:
        """Generate production sequences using grammar-constrained decoding.

        Args:
            z: Latent vectors (B, z_dim)
            greedy: If True, use argmax; if False, sample
            temperature: Sampling temperature

        Returns:
            Production IDs tensor (B, max_length)
        """
        device = z.device
        batch_size = z.size(0)

        logits = self.decoder(z)

        # Build grammar tables (cached)
        if self._grammar_cache is None:
            self._grammar_cache = self._build_grammar_tables()
        nt_masks_cpu, prod_rhs_nt, start_nt_id, pad_id = self._grammar_cache
        nt_masks = nt_masks_cpu.to(device)
        num_prods = len(prod_rhs_nt)

        # Initialize
        prod_ids = [[pad_id] * self.max_length for _ in range(batch_size)]
        stacks = [deque([start_nt_id]) for _ in range(batch_size)]
        active_set = set(range(batch_size))

        for t in range(self.max_length):
            if not active_set:
                break

            active_list = list(active_set)
            current_nt_list = []
            still_active = []

            for b in active_list:
                if stacks[b]:
                    current_nt_list.append(stacks[b].popleft())
                    still_active.append(b)
                else:
                    active_set.discard(b)

            if not still_active:
                break

            current_nt = torch.tensor(current_nt_list, dtype=torch.long, device=device)
            masks = nt_masks[current_nt]

            active_indices = torch.tensor(still_active, dtype=torch.long, device=device)
            step_logits = logits[active_indices, t, :]
            masked_logits = step_logits.masked_fill(masks == 0, float("-inf"))

            if greedy:
                selected = masked_logits.argmax(dim=-1)
            else:
                probs = torch.softmax(masked_logits / temperature, dim=-1)
                selected = torch.multinomial(probs, 1).squeeze(-1)

            selected_list = selected.tolist()

            for i, b in enumerate(still_active):
                pid = selected_list[i]
                prod_ids[b][t] = pid

                if pid < num_prods:
                    for nt_id in reversed(prod_rhs_nt[pid]):
                        stacks[b].appendleft(nt_id)

                if not stacks[b]:
                    active_set.discard(b)

        return torch.tensor(prod_ids, dtype=torch.long, device=device)

    def sample_from_prior(
        self,
        n_samples: int,
        device: str = "cpu",
        constrained: bool = True,
        greedy: bool = True,
    ) -> torch.Tensor:
        """Sample from prior N(0,1) and decode.

        Args:
            n_samples: Number of samples
            device: Device to use
            constrained: Use grammar-constrained decoding
            greedy: Use greedy decoding

        Returns:
            Production/token IDs (n_samples, max_length)
        """
        self.to(device)
        self.eval()

        with torch.no_grad():
            z = torch.randn(n_samples, self.z_dim, device=device)

            if constrained:
                prod_ids = self.generate_constrained(z, greedy=greedy)
            else:
                logits = self.decoder(z)
                prod_ids = logits.argmax(dim=-1)

        return prod_ids.cpu()

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input sequences to latent space.

        Args:
            x: Input sequences (B, T, P)

        Returns:
            mu: Latent means (B, z_dim)
            logvar: Latent log variances (B, z_dim)
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors to logits.

        Args:
            z: Latent vectors (B, z_dim)

        Returns:
            Logits (B, T, P)
        """
        return self.decoder(z)


# Backward compatibility alias
GrammarVAEModule = VAEModule
