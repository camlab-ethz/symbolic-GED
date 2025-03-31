

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Normal, Categorical
from typing import Tuple, Optional, List
from nltk import Nonterminal
from encoder import Encoder
from decoder import Decoder
from stack import Stack
from library.grammar_discovery import GCFG, S, get_mask
import sys

class GrammarVAE(nn.Module):
    """Grammar Variational Autoencoder with complete VAE operations"""
    def __init__(self, config: dict):
        super().__init__()
        # Extract configurations
        enc_config = config['model']['encoder']
        dec_config = config['model']['decoder']
        shared_config = config['model']['shared']
        
        # Initialize encoder
        self.encoder = Encoder(
            input_dim=shared_config['output_size'],
            hidden_dim=enc_config['hidden_size'],
            z_dim=shared_config['z_dim'],
            max_length=shared_config['max_length'],
            conv_sizes=enc_config['conv_sizes'],
            kernel_sizes=enc_config['kernel_sizes'],
            use_batch_norm=enc_config['use_batch_norm']
        )
        
        # Initialize decoder
        self.decoder = Decoder(
            latent_rep_size=shared_config['z_dim'],
            hidden_size=dec_config['hidden_size'],
            output_size=shared_config['output_size'],
            max_length=shared_config['max_length'],
            rnn_type=dec_config['rnn_type'],
            num_layers=dec_config['num_layers']
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = shared_config['max_length']
        self.tightness = config['training']['tightness']

    # def sample(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    #     """
    #     Reparameterization trick from original implementation
    #     """
    #     device = mu.device
    #     normal = Normal(torch.zeros(mu.shape).to(device), 
    #                    torch.ones(sigma.shape).to(device))
    #     eps = Variable(normal.sample()).to(device)
    #     z = mu + eps * torch.sqrt(sigma)*self.tightness
    #     return z

    def sample(self, mu: torch.Tensor, sigma: torch.Tensor, num_samples: int = 10) -> torch.Tensor:
        """
        Reparameterization trick with multiple samples
        Args:
            mu: shape (batch_size, latent_dim)
            sigma: shape (batch_size, latent_dim)
            num_samples: number of samples to draw (m in supervisor's notation)
        Returns:
            z: shape (batch_size, num_samples, latent_dim)
        """
        batch_size = mu.shape[0]
        latent_dim = mu.shape[1]
        device = mu.device
        
        # Expand mu and sigma to match desired output shape
        # (batch_size, 1, latent_dim)
        mu = mu.unsqueeze(1)
        sigma = sigma.unsqueeze(1)
        
        # Sample epsilon: (batch_size, num_samples, latent_dim)
        epsilon = torch.randn(batch_size, num_samples, latent_dim, device=device)
        
        # Reparameterization trick
        z = mu + epsilon * torch.sqrt(sigma) * self.tightness
        
        return z  # shape: (batch_size, num_samples, latent_dim)

    def kl_divergence(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Original KL divergence calculation
        """
        # return torch.mean(-0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp(), dim=1))
        return torch.mean(-0.5 * torch.sum(1 + torch.log(sigma) - mu.pow(2) - sigma, dim=1))

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Complete forward pass through the VAE
        """
        # Ensure input is on correct device
        x = x.to(self.device)
        
        # Encode
        mu, sigma = self.encoder(x)
        
        # Sample latent vector
        z = self.sample(mu, sigma)
        
        # Decode
        logits = self.decoder(z)
        
        return logits, mu, sigma

    def generate(self, z: torch.Tensor, sample: bool = False) -> List:
        torch.manual_seed(42)
        try:
            # print("Entering generate...")
            sys.stdout.flush()
            # print(f"Input z shape: {z.shape}")
            sys.stdout.flush()
            # Log the process ID for debugging
            import os
            # print(f"Process ID: {os.getpid()} at decoder call.")
            sys.stdout.flush()
            stack = Stack(grammar=GCFG, start_symbol=S)
            # print(f"Initial stack initialized with contents: {stack}")
            # print(f"Initial stack: {stack.contents()}")
            sys.stdout.flush()
            # print("Before decoder forward pass...")
            sys.stdout.flush()
            
            try:
                # print("Calling decoder...")
                sys.stdout.flush()
                # print(f"Devices: z on {z.device}, decoder on {self.device}")
                assert z.device == self.device, f"Mismatch: z on {z.device}, decoder on {self.device}"
                # print(f"decoder: {self.decoder}")
                sys.stdout.flush()
                logits = self.decoder(z).squeeze()
                # print("Decoder call successful.")
                sys.stdout.flush()
            except Exception as e:
                print(f"Decoder failed with exception: {e}")
                import traceback
                traceback.print_exc()  # Print full traceback for debugging
                sys.stdout.flush()
                raise

            # print(f"Logits shape: {logits.shape}")
            sys.stdout.flush()
            
            rules = []
            t = 0

            while stack.nonempty and t < self.max_length:
                # print(f"Step {t}, stack: {stack.contents()}")
                alpha = stack.pop()
                # print(f"Popped symbol: {alpha}")
                sys.stdout.flush()
                mask = get_mask(alpha, stack.grammar, as_variable=True).to(z.device)
                # print(f"Mask: {mask}")
                sys.stdout.flush()

                probs = mask * logits[t].exp()
                probs = probs / probs.sum()
                # print(f"Probabilities: {probs}")
                sys.stdout.flush()

                if sample:
                    m = Categorical(probs)
                    i = m.sample()
                else:
                    _, i = probs.max(-1)
                # print(f"Selected rule index: {i}")
                sys.stdout.flush()

                rule = stack.grammar.productions()[i.item()]
                rules.append(rule)
                # print(f"Selected rule: {rule}")
                sys.stdout.flush()

                for symbol in reversed(rule.rhs()):
                    if isinstance(symbol, Nonterminal):
                        stack.push(symbol)
                t += 1

            # print("Exiting generate.")
            sys.stdout.flush()
            return rules
        except Exception as e:
            print(f"Error gen: {e}")
            sys.stdout.flush()
            return None

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent space
        """
        x = x.to(self.device)
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to logits
        """
        return self.decoder(z)

    def to(self, device):
        """
        Ensure proper device movement
        """
        self.device = device
        return super().to(device)