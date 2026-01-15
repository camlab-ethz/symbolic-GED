#!/usr/bin/env python3
"""Discover and enumerate VAE configurations from checkpoints.

This module scans checkpoint directories to find all available VAE configs
with different beta values and annealing settings.
"""

import torch
import glob
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class VAEConfig:
    """Configuration for a VAE checkpoint."""
    name: str
    beta: float
    anneal_epochs: int
    grammar_ckpt: str
    token_ckpt: str
    grammar_seq_acc: float
    token_seq_acc: float

    @property
    def has_annealing(self) -> bool:
        return self.anneal_epochs > 0

    @property
    def anneal_str(self) -> str:
        return f"anneal{self.anneal_epochs}" if self.has_annealing else "no_anneal"

    @property
    def beta_str(self) -> str:
        if self.beta >= 0.01:
            return f"beta{self.beta:.2f}".replace("0.", "0p")
        else:
            # Scientific notation for small betas
            exp = int(abs(round(math.log10(self.beta))))
            return f"beta1e-{exp}"


def parse_checkpoint_info(ckpt_path: str) -> Optional[Dict]:
    """Extract beta, anneal_epochs, and seq_acc from a checkpoint."""
    try:
        data = torch.load(ckpt_path, map_location='cpu')
        hp = data.get('hyper_parameters', {})

        beta = hp.get('beta', None)
        anneal_epochs = hp.get('kl_anneal_epochs', 0)

        # Parse seq_acc from filename
        match = re.search(r'seq_acc=([0-9.]+)', ckpt_path)
        if match:
            acc_str = match.group(1).rstrip('.')
            seq_acc = float(acc_str)
        else:
            seq_acc = 0.0

        return {
            'beta': beta,
            'anneal_epochs': anneal_epochs,
            'seq_acc': seq_acc,
            'path': ckpt_path
        }
    except Exception as e:
        print(f"Warning: Could not parse {ckpt_path}: {e}")
        return None


def discover_checkpoints(base_dir: str = None) -> Tuple[Dict, Dict]:
    """Discover all grammar and token VAE checkpoints.

    Returns:
        (grammar_by_config, token_by_config): Dicts mapping (beta, anneal_epochs) -> list of checkpoint info
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent
    else:
        base_dir = Path(base_dir)

    grammar_ckpts = glob.glob(str(base_dir / 'checkpoints/grammar_vae/**/*.ckpt'), recursive=True)
    token_ckpts = glob.glob(str(base_dir / 'checkpoints/token_vae/**/*.ckpt'), recursive=True)

    from collections import defaultdict
    grammar_by_config = defaultdict(list)
    token_by_config = defaultdict(list)

    for ckpt in grammar_ckpts:
        info = parse_checkpoint_info(ckpt)
        if info and info['beta'] is not None:
            key = (info['beta'], info['anneal_epochs'])
            grammar_by_config[key].append(info)

    for ckpt in token_ckpts:
        info = parse_checkpoint_info(ckpt)
        if info and info['beta'] is not None:
            key = (info['beta'], info['anneal_epochs'])
            token_by_config[key].append(info)

    return dict(grammar_by_config), dict(token_by_config)


def get_best_configs(min_seq_acc: float = 0.5) -> List[VAEConfig]:
    """Get list of best VAE configs where both grammar and token have good checkpoints.

    Args:
        min_seq_acc: Minimum sequence accuracy threshold

    Returns:
        List of VAEConfig objects, sorted by beta (ascending)
    """
    import math

    grammar_by_config, token_by_config = discover_checkpoints()

    configs = []

    # Find configs that exist for both grammar and token
    all_keys = set(grammar_by_config.keys()) & set(token_by_config.keys())

    for key in all_keys:
        beta, anneal_epochs = key

        # Get best checkpoint for each
        grammar_best = max(grammar_by_config[key], key=lambda x: x['seq_acc'])
        token_best = max(token_by_config[key], key=lambda x: x['seq_acc'])

        # Only include if both have reasonable accuracy
        if grammar_best['seq_acc'] >= min_seq_acc and token_best['seq_acc'] >= min_seq_acc:
            # Generate config name
            if beta >= 0.01:
                beta_str = f"beta{beta:.2f}".replace(".", "p")
            else:
                exp = int(abs(round(math.log10(beta))))
                beta_str = f"beta1e-{exp}"

            anneal_str = f"_anneal{anneal_epochs}" if anneal_epochs > 0 else "_noanneal"
            name = beta_str + anneal_str

            config = VAEConfig(
                name=name,
                beta=beta,
                anneal_epochs=anneal_epochs,
                grammar_ckpt=grammar_best['path'],
                token_ckpt=token_best['path'],
                grammar_seq_acc=grammar_best['seq_acc'],
                token_seq_acc=token_best['seq_acc']
            )
            configs.append(config)

    # Sort by beta (ascending)
    configs.sort(key=lambda c: (c.beta, c.anneal_epochs))

    return configs


def print_available_configs():
    """Print all available configs."""
    configs = get_best_configs(min_seq_acc=0.0)

    print("=" * 80)
    print("AVAILABLE VAE CONFIGURATIONS")
    print("=" * 80)
    print(f"\n{'Name':<25} {'Beta':>10} {'Anneal':>8} {'Grammar Acc':>12} {'Token Acc':>12}")
    print("-" * 70)

    for cfg in configs:
        anneal_str = f"{cfg.anneal_epochs}" if cfg.anneal_epochs > 0 else "no"
        print(f"{cfg.name:<25} {cfg.beta:>10.5f} {anneal_str:>8} {cfg.grammar_seq_acc:>11.2%} {cfg.token_seq_acc:>11.2%}")

    print("\n" + "=" * 80)
    print("USABLE CONFIGS (seq_acc >= 50%)")
    print("=" * 80)

    usable = get_best_configs(min_seq_acc=0.5)
    print(f"\n{'Name':<25} {'Beta':>10} {'Anneal':>8} {'Grammar Acc':>12} {'Token Acc':>12}")
    print("-" * 70)

    for cfg in usable:
        anneal_str = f"{cfg.anneal_epochs}" if cfg.anneal_epochs > 0 else "no"
        print(f"{cfg.name:<25} {cfg.beta:>10.5f} {anneal_str:>8} {cfg.grammar_seq_acc:>11.2%} {cfg.token_seq_acc:>11.2%}")


if __name__ == '__main__':
    print_available_configs()
