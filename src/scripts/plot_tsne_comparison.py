#!/usr/bin/env python3
"""
Generate side-by-side t-SNE plots comparing beta=2e-4 vs beta=1e-2
for both Grammar and Token tokenizations on a given split.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.physics import PDE_PHYSICS
from analysis.interpolation_analysis import load_vae_model
from pde import grammar as pde_grammar
from pde.chr_tokenizer import PDETokenizer

# Color palette for families
FAMILY_COLORS = {
    "advection": "#1f77b4",
    "allen_cahn": "#ff7f0e",
    "biharmonic": "#2ca02c",
    "beam_plate": "#2ca02c",
    "burgers": "#d62728",
    "cahn_hilliard": "#9467bd",
    "fisher_kpp": "#8c564b",
    "heat": "#e377c2",
    "kdv": "#7f7f7f",
    "kuramoto_sivashinsky": "#17becf",
    "poisson": "#ffbb78",
    "reaction_diffusion_cubic": "#98df8a",
    "sine_gordon": "#ff9896",
    "telegraph": "#c5b0d5",
    "wave": "#c49c94",
    "airy": "#bcbd22",
}


def encode_dataset_to_latents(
    model,
    tokenization,
    csv_path,
    split,
    split_dir: Path | None = None,
    device="cuda",
    batch_size=256,
):
    """Encode PDE dataset to latent vectors. Returns (latents, valid_families)."""
    df = pd.read_csv(csv_path)

    # Filter by split
    if "split" in df.columns:
        df = df[df["split"] == split]
    elif split != "all":
        # Prefer explicit split_dir, otherwise fallback to legacy location.
        sd = split_dir if split_dir is not None else (csv_path.parent.parent / "splits")
        if sd.exists():
            split_indices = np.load(sd / f"{split}_indices.npy")
            df = df.iloc[split_indices]

    pde_strings = df["pde"].values.tolist()
    families = df["family"].values.tolist() if "family" in df.columns else None

    vocab_size = model.P
    max_length = model.max_length

    all_mu = []
    valid_families = []

    model.eval()
    tokenizer = PDETokenizer() if tokenization == "token" else None

    with torch.no_grad():
        for batch_idx in range(0, len(pde_strings), batch_size):
            batch_pdes = pde_strings[batch_idx : batch_idx + batch_size]
            batch_fams = (
                families[batch_idx : batch_idx + batch_size]
                if families
                else [None] * len(batch_pdes)
            )

            batch_onehot = torch.zeros(
                len(batch_pdes), max_length, vocab_size, dtype=torch.float32
            )
            valid_mask = []

            for b, pde in enumerate(batch_pdes):
                try:
                    if tokenization == "grammar":
                        pde_cleaned = pde.replace(" ", "").replace("=0", "")
                        seq = pde_grammar.parse_to_productions(pde_cleaned)
                        for t, pid in enumerate(seq[:max_length]):
                            if 0 <= pid < vocab_size:
                                batch_onehot[b, t, pid] = 1.0
                    else:
                        ids = tokenizer.encode(pde)
                        for t, tid in enumerate(ids[:max_length]):
                            if 0 <= tid < vocab_size:
                                batch_onehot[b, t, tid] = 1.0

                    valid_mask.append(True)
                    if families:
                        valid_families.append(batch_fams[b])
                except:
                    valid_mask.append(False)

            if any(valid_mask):
                valid_batch_indices = [i for i, v in enumerate(valid_mask) if v]
                if valid_batch_indices:
                    batch_onehot_valid = batch_onehot[valid_batch_indices].to(device)
                    mu, _ = model.encoder(batch_onehot_valid)
                    all_mu.append(mu.cpu().numpy())

    if all_mu:
        latents = np.concatenate(all_mu, axis=0)
    else:
        latents = np.array([]).reshape(0, model.z_dim)

    valid_families = [f for f in valid_families if f is not None]
    return latents, valid_families if families else None


def load_latents_for_beta(
    model_path,
    tokenization,
    csv_path,
    split,
    split_dir: Path | None = None,
    device="cuda",
):
    """Load model and encode dataset to get latents."""
    print(f"  Loading {tokenization} model: {model_path}")
    model, hparams = load_vae_model(model_path, device)

    print(f"  Encoding {tokenization} dataset (split={split})...")
    latents, families = encode_dataset_to_latents(
        model, tokenization, csv_path, split, split_dir=split_dir, device=device
    )

    return latents, families, model, hparams


def create_side_by_side_tsne(
    latents_beta1,
    families_beta1,
    beta1_name,
    latents_beta2,
    families_beta2,
    beta2_name,
    title_prefix,
    output_path,
    seed=42,
):
    """Create side-by-side t-SNE plots with family labels."""
    print(f"    Computing t-SNE for {title_prefix}...")

    # Compute t-SNE for both
    tsne1 = TSNE(n_components=2, random_state=seed, perplexity=30, n_jobs=-1)
    embedding1 = tsne1.fit_transform(latents_beta1)

    tsne2 = TSNE(n_components=2, random_state=seed, perplexity=30, n_jobs=-1)
    embedding2 = tsne2.fit_transform(latents_beta2)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10), dpi=150)

    # Get unique families
    unique_families = sorted(set(families_beta1 + families_beta2))

    # Helper function to add labels for a family
    def add_family_labels(ax, embedding, families, family_name):
        """Add text label for a family at its centroid."""
        mask = np.array(families) == family_name
        if mask.sum() > 0:
            # Use centroid of the cluster
            centroid = embedding[mask].mean(axis=0)
            # Or use the point closest to centroid for better positioning
            distances = np.linalg.norm(embedding[mask] - centroid, axis=1)
            closest_idx = np.argmin(distances)
            label_pos = embedding[mask][closest_idx]

            ax.text(
                label_pos[0],
                label_pos[1],
                family_name,
                fontsize=9,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=0.5,
                ),
                ha="center",
                va="center",
            )

    # Plot left (beta1)
    for family in unique_families:
        mask1 = np.array(families_beta1) == family
        if mask1.sum() > 0:
            color = FAMILY_COLORS.get(family, "#cccccc")
            ax1.scatter(
                embedding1[mask1, 0],
                embedding1[mask1, 1],
                c=color,
                label=family,
                s=10,
                alpha=0.5,
                edgecolors="black",
                linewidths=0.2,
            )
            # Add label
            add_family_labels(ax1, embedding1, families_beta1, family)

    ax1.set_xlabel("t-SNE 1", fontsize=12)
    ax1.set_ylabel("t-SNE 2", fontsize=12)
    ax1.set_title(f"{beta1_name}", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Plot right (beta2)
    for family in unique_families:
        mask2 = np.array(families_beta2) == family
        if mask2.sum() > 0:
            color = FAMILY_COLORS.get(family, "#cccccc")
            ax2.scatter(
                embedding2[mask2, 0],
                embedding2[mask2, 1],
                c=color,
                label=family,
                s=10,
                alpha=0.5,
                edgecolors="black",
                linewidths=0.2,
            )
            # Add label
            add_family_labels(ax2, embedding2, families_beta2, family)

    ax2.set_xlabel("t-SNE 1", fontsize=12)
    ax2.set_ylabel("t-SNE 2", fontsize=12)
    ax2.set_title(f"{beta2_name}", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # Add a single legend for family colors (figure-level, not per-axis)
    # This keeps plots clean and ensures the same color mapping is shown.
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=family,
            markerfacecolor=FAMILY_COLORS.get(family, "#cccccc"),
            markeredgecolor="black",
            markersize=7,
            linewidth=0,
        )
        for family in unique_families
    ]
    # Use multiple columns if the legend gets too tall
    ncol = 1 if len(unique_families) <= 10 else 2
    fig.legend(
        handles=legend_handles,
        title="Family",
        loc="center right",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fontsize=9,
        title_fontsize=10,
        ncol=ncol,
    )

    # Overall title
    fig.suptitle(
        f"{title_prefix} - Side-by-Side t-SNE Comparison",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    # Leave room on the right for the legend
    plt.tight_layout(rect=[0, 0, 0.84, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Saved: {output_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate side-by-side t-SNE plots for beta comparison"
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "val", "test"],
        help="Dataset split to use",
    )
    parser.add_argument(
        "--csv-metadata",
        type=str,
        required=True,
        help="Path to CSV with PDE strings and GT labels",
    )
    parser.add_argument(
        "--split-dir",
        type=str,
        default=None,
        help="Directory containing split indices (train_indices.npy, val_indices.npy, test_indices.npy)",
    )
    parser.add_argument(
        "--grammar-ckpt-beta1",
        type=str,
        required=True,
        help="Grammar VAE checkpoint for beta1 (2e-4)",
    )
    parser.add_argument(
        "--token-ckpt-beta1",
        type=str,
        required=True,
        help="Token VAE checkpoint for beta1 (2e-4)",
    )
    parser.add_argument(
        "--grammar-ckpt-beta2",
        type=str,
        required=True,
        help="Grammar VAE checkpoint for beta2 (1e-2)",
    )
    parser.add_argument(
        "--token-ckpt-beta2",
        type=str,
        required=True,
        help="Token VAE checkpoint for beta2 (1e-2)",
    )
    parser.add_argument("--outdir", type=str, required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.csv_metadata)
    split_dir = Path(args.split_dir) if args.split_dir else None

    print("=" * 80)
    print(f"t-SNE COMPARISON: Beta=2e-4 vs Beta=1e-2")
    print(f"Split: {args.split}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    # Load latents for beta1 (2e-4)
    print("\n[1/4] Loading latents for Beta=2e-4...")
    grammar_latents_beta1, grammar_families_beta1, _, _ = load_latents_for_beta(
        args.grammar_ckpt_beta1,
        "grammar",
        csv_path,
        args.split,
        split_dir=split_dir,
        device=args.device,
    )
    token_latents_beta1, token_families_beta1, _, _ = load_latents_for_beta(
        args.token_ckpt_beta1,
        "token",
        csv_path,
        args.split,
        split_dir=split_dir,
        device=args.device,
    )

    print(f"  Grammar: {grammar_latents_beta1.shape}")
    print(f"  Token:   {token_latents_beta1.shape}")

    # Load latents for beta2 (1e-2)
    print("\n[2/4] Loading latents for Beta=1e-2...")
    grammar_latents_beta2, grammar_families_beta2, _, _ = load_latents_for_beta(
        args.grammar_ckpt_beta2,
        "grammar",
        csv_path,
        args.split,
        split_dir=split_dir,
        device=args.device,
    )
    token_latents_beta2, token_families_beta2, _, _ = load_latents_for_beta(
        args.token_ckpt_beta2,
        "token",
        csv_path,
        args.split,
        split_dir=split_dir,
        device=args.device,
    )

    print(f"  Grammar: {grammar_latents_beta2.shape}")
    print(f"  Token:   {token_latents_beta2.shape}")

    # Create side-by-side plots
    print("\n[3/4] Creating Grammar VAE side-by-side t-SNE...")
    create_side_by_side_tsne(
        grammar_latents_beta1,
        grammar_families_beta1,
        "Beta = 2e-4",
        grammar_latents_beta2,
        grammar_families_beta2,
        "Beta = 1e-2",
        f"Grammar VAE ({args.split} split)",
        output_dir / f"grammar_tsne_comparison_{args.split}.png",
        seed=args.seed,
    )

    print("\n[4/4] Creating Token VAE side-by-side t-SNE...")
    create_side_by_side_tsne(
        token_latents_beta1,
        token_families_beta1,
        "Beta = 2e-4",
        token_latents_beta2,
        token_families_beta2,
        "Beta = 1e-2",
        f"Token VAE ({args.split} split)",
        output_dir / f"token_tsne_comparison_{args.split}.png",
        seed=args.seed,
    )

    print("\n" + "=" * 80)
    print("t-SNE COMPARISON COMPLETE!")
    print("=" * 80)
    print(f"\nPlots saved to: {output_dir}")
    print(f"  - grammar_tsne_comparison_{args.split}.png")
    print(f"  - token_tsne_comparison_{args.split}.png")
    print("=" * 80)


if __name__ == "__main__":
    main()
