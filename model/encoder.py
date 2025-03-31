import torch
import torch.nn as nn
from typing import List

class Encoder(nn.Module):
    """
    Compact Encoder for Grammar VAE with backward-compatible state_dict mapping.
    """
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 z_dim: int,
                 max_length: int,
                 conv_sizes: List[int],
                 kernel_sizes: List[int],
                 use_batch_norm: bool = True,
                 dropout: float = 0.2):
        super().__init__()
        if len(conv_sizes) != len(kernel_sizes):
            raise ValueError("conv_sizes and kernel_sizes must have the same length")
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim if i == 0 else conv_sizes[i - 1], conv_sizes[i], kernel_sizes[i]),
                nn.BatchNorm1d(conv_sizes[i]) if use_batch_norm else nn.Identity(),
                nn.ReLU()
            ) for i in range(len(conv_sizes))
        ])
        length = max_length
        for k in kernel_sizes:
            length -= k - 1
        self.output_size = length * conv_sizes[-1]
        self.linear = nn.Sequential(
            nn.Linear(self.output_size, hidden_dim),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.sigma = nn.Sequential(nn.Linear(hidden_dim, z_dim), nn.Softplus())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for conv in self.convs:
            x = conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return self.mu(x), self.sigma(x)

    @staticmethod
    def map_state_dict(state_dict: dict) -> dict:
        """
        Map parameter names from the checkpoint to match the model's expected state_dict keys.
        Handles both encoder and decoder keys.
        """
        mapping = {
            "encoder.conv1.weight": "encoder.convs.0.0.weight",
            "encoder.conv1.bias": "encoder.convs.0.0.bias",
            "encoder.batch_norm1.weight": "encoder.convs.0.1.weight",
            "encoder.batch_norm1.bias": "encoder.convs.0.1.bias",
            "encoder.batch_norm1.running_mean": "encoder.convs.0.1.running_mean",
            "encoder.batch_norm1.running_var": "encoder.convs.0.1.running_var",
            "encoder.conv2.weight": "encoder.convs.1.0.weight",
            "encoder.conv2.bias": "encoder.convs.1.0.bias",
            "encoder.batch_norm2.weight": "encoder.convs.1.1.weight",
            "encoder.batch_norm2.bias": "encoder.convs.1.1.bias",
            "encoder.batch_norm2.running_mean": "encoder.convs.1.1.running_mean",
            "encoder.batch_norm2.running_var": "encoder.convs.1.1.running_var",
            "encoder.conv3.weight": "encoder.convs.2.0.weight",
            "encoder.conv3.bias": "encoder.convs.2.0.bias",
            "encoder.batch_norm3.weight": "encoder.convs.2.1.weight",
            "encoder.batch_norm3.bias": "encoder.convs.2.1.bias",
            "encoder.batch_norm3.running_mean": "encoder.convs.2.1.running_mean",
            "encoder.batch_norm3.running_var": "encoder.convs.2.1.running_var",
            "encoder.linear.weight": "encoder.linear.0.weight",
            "encoder.linear.bias": "encoder.linear.0.bias",
            "encoder.mu.weight": "encoder.mu.weight",
            "encoder.mu.bias": "encoder.mu.bias",
            "encoder.sigma.weight": "encoder.sigma.0.weight",
            "encoder.sigma.bias": "encoder.sigma.0.bias",
            "decoder.init_input": "decoder.init_input",
            "decoder.embedding.weight": "decoder.embedding.weight",
            "decoder.linear_hidden.weight": "decoder.linear_hidden.weight",
            "decoder.linear_hidden.bias": "decoder.linear_hidden.bias",
            "decoder.rnn.weight_ih_l0": "decoder.rnn.weight_ih_l0",
            "decoder.rnn.weight_hh_l0": "decoder.rnn.weight_hh_l0",
            "decoder.rnn.bias_ih_l0": "decoder.rnn.bias_ih_l0",
            "decoder.rnn.bias_hh_l0": "decoder.rnn.bias_hh_l0",
            "decoder.rnn.weight_ih_l1": "decoder.rnn.weight_ih_l1",
            "decoder.rnn.weight_hh_l1": "decoder.rnn.weight_hh_l1",
            "decoder.rnn.bias_ih_l1": "decoder.rnn.bias_ih_l1",
            "decoder.rnn.bias_hh_l1": "decoder.rnn.bias_hh_l1",
            "decoder.output_layer.weight": "decoder.output_layer.weight",
            "decoder.output_layer.bias": "decoder.output_layer.bias",
        }
        new_state_dict = {}
        for old_key, value in state_dict.items():
            new_key = mapping.get(old_key, old_key)  # Map if present, else keep as is
            new_state_dict[new_key] = value

        return new_state_dict
