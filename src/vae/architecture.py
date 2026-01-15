"""VAE Encoder and Decoder architectures.

This module provides the neural network architectures for the VAE:
- Encoder: Convolutional encoder with residual connections
- Decoder: GRU-based decoder with optional bidirectional support

Both architectures are designed for sequence-to-sequence encoding of PDE expressions.
"""

import warnings
from typing import List, Tuple, Union

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Convolutional encoder mapping sequences to latent Gaussian parameters.

    Architecture:
    - Multi-scale 1D convolutions with residual connections
    - Progressive channel expansion (doubles each layer)
    - Global average pooling to fixed-size representation
    - Linear projections to mu and logvar

    Args:
        P: Input vocabulary size (sequence depth)
        hidden_channels: Base hidden dimension
        z_dim: Latent space dimension
        conv_layers: Number of convolutional layers
        kernel_size: Kernel size(s) for conv layers (int or list)
        dropout: Dropout rate (currently unused, for future extension)
    """

    def __init__(
        self,
        P: int,
        hidden_channels: int = 256,
        z_dim: int = 64,
        conv_layers: int = 3,
        kernel_size: Union[int, List[int]] = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.P = P
        self.hidden = hidden_channels
        self.z_dim = z_dim

        # Determine kernel sizes for each layer
        kernel_sizes = self._get_kernel_sizes(kernel_size, conv_layers)

        # Progressive channel expansion: hidden//2 → hidden → hidden*2
        channel_progression = self._get_channel_progression(
            hidden_channels, conv_layers
        )
        
        # Build convolutional layers with residual connections
        self.conv_blocks = nn.ModuleList()
        in_ch = P

        for i in range(conv_layers):
            out_ch = channel_progression[i]
            k = kernel_sizes[i]

            # Main convolution block
            block = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2), nn.ReLU()
            )
            self.conv_blocks.append(block)

            # Residual projection (1x1 conv if dimensions change)
            if in_ch != out_ch:
                self.conv_blocks.append(nn.Conv1d(in_ch, out_ch, kernel_size=1))
            else:
                self.conv_blocks.append(None)

            in_ch = out_ch

        self.final_channels = channel_progression[-1]
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_mu = nn.Linear(self.final_channels, z_dim)
        self.fc_logvar = nn.Linear(self.final_channels, z_dim)

    def _get_kernel_sizes(
        self, kernel_size: Union[int, List[int]], conv_layers: int
    ) -> List[int]:
        """Determine kernel sizes for each layer."""
        if isinstance(kernel_size, int):
            # Default multi-scale: [3, 5, 7]
            kernel_sizes = [3, 5, 7][:conv_layers]
            while len(kernel_sizes) < conv_layers:
                kernel_sizes.append(min(kernel_sizes[-1] + 2, 9))
        else:
            if len(kernel_size) != conv_layers:
                warnings.warn(
                    f"kernel_size list length ({len(kernel_size)}) doesn't match "
                    f"conv_layers ({conv_layers}). Using first value for all layers.",
                    UserWarning,
                )
                kernel_sizes = [kernel_size[0]] * conv_layers
            else:
                kernel_sizes = list(kernel_size)
        return kernel_sizes

    def _get_channel_progression(
        self, hidden_channels: int, conv_layers: int
    ) -> List[int]:
        """Compute channel sizes for progressive expansion."""
        channels = [hidden_channels // 2] * conv_layers
        for i in range(1, conv_layers):
            channels[i] = min(channels[i - 1] * 2, hidden_channels * 2)
        return channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input sequences (B, T, P) - one-hot encoded

        Returns:
            mu: Latent means (B, z_dim)
            logvar: Latent log variances (B, z_dim)
        """
        # Permute to (B, P, T) for Conv1d
        x = x.permute(0, 2, 1)

        # Apply conv blocks with residual connections
        for i in range(0, len(self.conv_blocks), 2):
            conv_block = self.conv_blocks[i]
            residual_proj = self.conv_blocks[i + 1]

            identity = x
            out = conv_block(x)

            if residual_proj is not None:
                identity = residual_proj(identity)

            x = out + identity

        # Global pooling and projection
        h = self.pool(x).squeeze(-1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


class Decoder(nn.Module):
    """GRU-based decoder mapping latent vectors to sequence logits.

    Architecture:
    - Linear projection from z to initial hidden state
    - Multi-layer GRU (optionally bidirectional)
    - Layer normalization on GRU output
    - Linear projection to vocabulary logits

    Args:
        P: Output vocabulary size
        z_dim: Latent space dimension
        hidden_dim: GRU hidden dimension
        max_length: Maximum sequence length
        num_layers: Number of GRU layers
        dropout: Dropout rate between GRU layers
        bidirectional: Use bidirectional GRU
    """

    def __init__(
        self,
        P: int,
        z_dim: int = 64,
        hidden_dim: int = 256,
        max_length: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.P = P
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Initialize hidden state from z
        self.init_linear = nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.ELU())

        # Learnable start embedding
        self.embedding = nn.Parameter(torch.randn(1, hidden_dim))

        # GRU
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.act = nn.ELU()

        # Output projection
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.layer_norm = nn.LayerNorm(gru_output_dim)
        self.out = nn.Linear(gru_output_dim, P)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            z: Latent vectors (B, z_dim)

        Returns:
            logits: Sequence logits (B, T, P)
        """
        B = z.size(0)

        # Initialize hidden state
        h0 = self.init_linear(z).unsqueeze(0)  # (1, B, hidden)

        # Expand for multi-layer (and bidirectional)
        num_directions = 2 if self.bidirectional else 1
        h0 = h0.repeat(self.num_layers * num_directions, 1, 1)

        # Prepare input sequence (repeat start embedding)
        start = self.embedding.unsqueeze(0).expand(B, 1, -1)  # (B, 1, hidden)
        inputs = start.repeat(1, self.max_length, 1)  # (B, T, hidden)

        # Run GRU
        out, _ = self.gru(inputs, h0)

        # Process output
        out = self.layer_norm(out)
        out = self.act(out)
        logits = self.out(out)  # (B, T, P)

        return logits
