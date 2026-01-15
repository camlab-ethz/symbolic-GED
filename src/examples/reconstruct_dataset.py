"""
Load trained Grammar-VAE model and reconstruct the training dataset.
Saves both the original and reconstructed PDEs to text files.
"""

import torch
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.grammar_vae.module import GrammarVAEModule
from src.pde import grammar as pde_grammar


def prod_ids_to_string(prod_ids):
    """Convert production ID sequence to readable PDE string by simulating left-most derivation.

    Note: This implementation doesn't reconstruct actual numeric values (coefficients),
    it only shows the structure. All coefficients appear as 'N'.
    """
    try:
        # Get set of all nonterminals from the grammar
        nonterminals = set(pde_grammar.LHS_TO_PRODS.keys())

        # Simulate left-most derivation
        derivation = ["PDE"]

        for step, pid in enumerate(prod_ids):
            if pid >= len(pde_grammar.Productions):
                return f"[INVALID ID {pid}]"

            lhs, rhs = pde_grammar.Productions[pid]

            # Find leftmost nonterminal matching this production's LHS
            found_idx = None
            for i, sym in enumerate(derivation):
                if sym == lhs:
                    found_idx = i
                    break

            if found_idx is None:
                # Check if we've finished all nonterminals (no more to expand)
                remaining_nts = [s for s in derivation if s in nonterminals]
                if not remaining_nts:
                    # All done, rest is padding - this is OK
                    break
                # This is a real mismatch - production doesn't apply
                return f"[MISMATCH@{step}: prod{pid}={lhs}, have {remaining_nts[:2]}]"

            # Expand the nonterminal
            derivation = (
                derivation[:found_idx] + list(rhs) + derivation[found_idx + 1 :]
            )

        # Convert to string: collect all terminals
        result = []
        for sym in derivation:
            if not sym:  # epsilon
                continue
            elif sym == "NUMBER":
                result.append("N")  # Placeholder for coefficients
            elif sym in nonterminals:
                # Still a nonterminal - incomplete parse
                continue
            else:
                # Terminal symbol
                result.append(sym)

        pde_str = "".join(result).strip()

        # Check if there are unexpanded nonterminals
        remaining_nts = [s for s in derivation if s in nonterminals]
        if remaining_nts:
            return f"{pde_str} [INCOMPLETE: {remaining_nts[:2]}]"

        return pde_str

    except Exception as e:
        import traceback

        return f"[ERROR: {str(e)[:30]}]"


def reconstruct_dataset(
    checkpoint_path: str,
    prod_onehot_path: str,
    masks_path: str,
    output_file: str = "reconstructions.txt",
    max_samples: int = None,
):
    """
    Load trained model and reconstruct dataset.

    Args:
        checkpoint_path: Path to model checkpoint (.ckpt)
        prod_onehot_path: Path to production onehot data (.pt)
        masks_path: Path to masks data (.pt)
        output_file: Output text file name
        max_samples: Maximum number of samples to process (None = all)
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    model = GrammarVAEModule.load_from_checkpoint(checkpoint_path)
    model.eval()

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on {device}")

    # Load data
    print(f"Loading data from {prod_onehot_path}...")
    prod_onehot = torch.load(prod_onehot_path)  # (N, T, P)
    masks = torch.load(masks_path)  # (N, T, P)

    # Get targets (original production sequences)
    targets = prod_onehot.argmax(dim=-1)  # (N, T)

    N = prod_onehot.shape[0]
    if max_samples is not None:
        N = min(N, max_samples)

    print(f"Processing {N} samples...")

    results = []
    correct_reconstructions = 0

    with torch.no_grad():
        for i in range(N):
            # Get single sample
            x = prod_onehot[i : i + 1].to(device)  # (1, T, P)
            target = targets[i].cpu().numpy().tolist()  # List of production IDs

            # Encode and decode through VAE
            logits, mu, logvar = model.forward(x)  # logits: (1, T, P)

            # Get predicted production IDs
            pred = logits.argmax(dim=-1).squeeze(0).cpu().numpy().tolist()

            # Convert to strings
            original_str = prod_ids_to_string(target)
            reconstructed_str = prod_ids_to_string(pred)

            # Check if exact match
            is_match = target == pred
            if is_match:
                correct_reconstructions += 1

            # Store result
            results.append(
                {
                    "idx": i,
                    "original": original_str,
                    "reconstructed": reconstructed_str,
                    "match": is_match,
                    "original_ids": target,
                    "reconstructed_ids": pred,
                }
            )

            # Print progress
            if (i + 1) % 100 == 0:
                print(
                    f"Processed {i+1}/{N} samples, {correct_reconstructions} exact matches ({100*correct_reconstructions/(i+1):.2f}%)"
                )

    # Save to text file
    print(f"\nSaving results to {output_file}...")
    with open(output_file, "w") as f:
        f.write(f"Grammar-VAE Reconstruction Results\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Total samples: {N}\n")
        f.write(
            f"Exact reconstructions: {correct_reconstructions} ({100*correct_reconstructions/N:.2f}%)\n"
        )
        f.write("=" * 80 + "\n\n")

        for res in results:
            f.write(f"Sample {res['idx']}:\n")
            f.write(f"  Original:      {res['original']}\n")
            f.write(f"  Reconstructed: {res['reconstructed']}\n")
            f.write(f"  Match: {'✓' if res['match'] else '✗'}\n")
            if not res["match"]:
                # Show production IDs for debugging
                f.write(f"  Original IDs:      {res['original_ids']}\n")
                f.write(f"  Reconstructed IDs: {res['reconstructed_ids']}\n")
            f.write("\n")

    print(f"Done! Results saved to {output_file}")
    print(
        f"Exact reconstruction rate: {100*correct_reconstructions/N:.2f}% ({correct_reconstructions}/{N})"
    )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Reconstruct dataset using trained Grammar-VAE"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint (.ckpt)"
    )
    parser.add_argument(
        "--prod", type=str, required=True, help="Path to production onehot data (.pt)"
    )
    parser.add_argument(
        "--masks", type=str, required=True, help="Path to masks data (.pt)"
    )
    parser.add_argument(
        "--output", type=str, default="reconstructions.txt", help="Output text file"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Maximum samples to process"
    )

    args = parser.parse_args()

    reconstruct_dataset(
        checkpoint_path=args.checkpoint,
        prod_onehot_path=args.prod,
        masks_path=args.masks,
        output_file=args.output,
        max_samples=args.max_samples,
    )
