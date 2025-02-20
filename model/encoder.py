import torch
import torch.nn as nn
from typing import List

class Encoder(nn.Module):
   
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
            nn.Sequential( nn.Conv1d(input_dim if i == 0 else conv_sizes[i - 1], conv_sizes[i], kernel_sizes[i]),
                nn.BatchNorm1d(conv_sizes[i]) if use_batch_norm else nn.Identity(), nn.ReLU()) for i in range(len(conv_sizes))
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

    