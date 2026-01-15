#!/usr/bin/env python3
"""Run latent-space clustering metrics for multiple VAE checkpoints.

This script encodes PDE strings into latent vectors (mu) using each model's
encoder, then computes clustering metrics (especially NMI) for various label
types (family/type/order/dim/linearity/temporal_order).

It is intended to be run on the 4 "current" models (grammar/token × beta),
across dataset splits (train/val/test), and to write results in one place.

Example:
  python -m analysis.run_clustering_metrics \
    --csv-metadata data/raw/pde_dataset_45672.csv \
    --splits train,val,test \
    --ckpt grammar_beta2e4:checkpoints/grammar_vae/beta_2e-4_seed_42/...ckpt \
    --ckpt grammar_beta1e2:checkpoints/grammar_vae/beta_0.01_seed_42/...ckpt \
    --ckpt token_beta2e4:checkpoints/token_vae/beta_2e-4_seed_42/...ckpt \
    --ckpt token_beta1e2:checkpoints/token_vae/beta_0.01_seed_42/...ckpt \
    --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from analysis.clustering import compute_all_clustering
from analysis.physics import parse_pde


@dataclass(frozen=True)
class ModelSpec:
    name: str
    ckpt_path: Path
    tokenization: str  # "grammar" | "token"


def _infer_tokenization_from_path(ckpt_path: Path) -> str:
    p = str(ckpt_path).lower()
    if "token_vae" in p or "/token_" in p or "tokenbeta" in p:
        return "token"
    if "grammar_vae" in p or "/grammar_" in p or "grammarbeta" in p:
        return "grammar"
    # Fallback: heuristics for directory names
    if "token" in p and "grammar" not in p:
        return "token"
    return "grammar"


def _load_vae_from_checkpoint(
    ckpt_path: Path, device: str
) -> Tuple[torch.nn.Module, Dict]:
    """Load `vae.module.VAEModule` from a checkpoint path."""
    from vae.module import VAEModule

    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    hparams = checkpoint.get("hyper_parameters", {})

    model = VAEModule(
        P=hparams["P"],
        max_length=hparams["max_length"],
        z_dim=hparams.get("z_dim", 26),
        lr=hparams.get("lr", 1e-3),
        beta=hparams.get("beta", 1e-5),
        encoder_hidden=hparams.get("encoder_hidden", 128),
        encoder_conv_layers=hparams.get("encoder_conv_layers", 3),
        encoder_kernel=hparams.get("encoder_kernel", [7, 7, 7]),
        decoder_hidden=hparams.get("decoder_hidden", 80),
        decoder_layers=hparams.get("decoder_layers", 3),
        decoder_dropout=hparams.get("decoder_dropout", 0.1),
        free_bits=hparams.get("free_bits", 0.0),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()
    return model, hparams


def _load_split_dataframe(
    csv_path: Path, split: str, split_dir: Optional[Path] = None
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "split" in df.columns:
        df = df[df["split"] == split].copy()
        return df

    # Fallback: saved indices in `splits/` (repo-level)
    if split != "all":
        sd = split_dir if split_dir is not None else (csv_path.parent.parent / "splits")
        if sd.exists():
            idx_path = sd / f"{split}_indices.npy"
            if idx_path.exists():
                split_indices = np.load(idx_path)
                df = df.iloc[split_indices].copy()
                return df

    raise RuntimeError(
        f"Could not filter split='{split}'. Expected a 'split' column in {csv_path} "
        f"or indices file in {csv_path.parent.parent / 'splits'}."
    )


def _encode_pdes_to_latents(
    model: torch.nn.Module,
    tokenization: str,
    pdes: List[str],
    device: str,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode PDE strings to latent means (mu).

    Returns:
      latents: (N_valid, z_dim)
      valid_mask: (N,) boolean mask over input `pdes` indicating which were encoded
    """
    from pde import grammar as pde_grammar
    from pde.chr_tokenizer import PDETokenizer

    vocab_size = int(model.P)
    max_length = int(model.max_length)

    tokenizer = PDETokenizer() if tokenization == "token" else None
    valid_mask = np.zeros(len(pdes), dtype=bool)
    all_mu: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(pdes), batch_size):
            batch = pdes[start : start + batch_size]
            # Build onehot only for items that successfully encode (saves GPU work)
            onehots = torch.zeros(
                len(batch), max_length, vocab_size, dtype=torch.float32
            )
            batch_valid = []

            for i, pde in enumerate(batch):
                try:
                    if tokenization == "grammar":
                        pde_cleaned = pde.replace(" ", "").replace("=0", "")
                        seq = pde_grammar.parse_to_productions(pde_cleaned)
                        for t, pid in enumerate(seq[:max_length]):
                            if 0 <= pid < vocab_size:
                                onehots[i, t, pid] = 1.0
                    else:
                        assert tokenizer is not None
                        ids = tokenizer.encode(pde)
                        for t, tid in enumerate(ids[:max_length]):
                            if 0 <= tid < vocab_size:
                                onehots[i, t, tid] = 1.0

                    batch_valid.append(i)
                except Exception:
                    continue

            if not batch_valid:
                continue

            onehots_valid = onehots[batch_valid].to(device)
            mu, _ = model.encoder(onehots_valid)
            all_mu.append(mu.detach().cpu().numpy())

            # Mark valid indices
            for i in batch_valid:
                valid_mask[start + i] = True

    if not all_mu:
        return (
            np.zeros((0, int(getattr(model, "z_dim", 0))), dtype=np.float32),
            valid_mask,
        )
    return np.concatenate(all_mu, axis=0), valid_mask


def _build_label_dict(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Build label arrays aligned to `df` rows."""
    families = (
        df["family"].astype(str).tolist()
        if "family" in df.columns
        else [None] * len(df)
    )
    pdes = df["pde"].astype(str).tolist()

    out: Dict[str, List] = {
        "family": [],
        "type": [],
        "linearity": [],
        "order": [],
        "dim": [],
        "temporal_order": [],
    }

    for pde, fam in zip(pdes, families):
        labels = parse_pde(pde, family=fam if fam is not None else None)
        out["family"].append(
            fam if fam is not None else labels.get("family", "unknown")
        )
        out["type"].append(labels.get("type", "unknown"))
        out["linearity"].append(labels.get("linearity", "unknown"))
        out["order"].append(labels.get("order", 0))
        out["dim"].append(labels.get("dim", 1))
        out["temporal_order"].append(labels.get("temporal_order", 0))

    return {k: np.array(v) for k, v in out.items()}


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:

        def _json_default(o):  # type: ignore[no-untyped-def]
            # Make numpy / torch / Path objects JSON-serializable.
            if isinstance(o, Path):
                return str(o)
            if isinstance(o, (np.integer, np.floating)):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, torch.Tensor):
                return o.detach().cpu().tolist()
            raise TypeError(
                f"Object of type {o.__class__.__name__} is not JSON serializable"
            )

        json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute latent clustering metrics (NMI, etc.) for VAE checkpoints"
    )
    parser.add_argument(
        "--csv-metadata",
        type=str,
        default="data/raw/pde_dataset_48000_fixed.csv",
        help="CSV with at least columns: pde, family, and optionally split",
    )
    parser.add_argument(
        "--split-dir",
        type=str,
        default=None,
        help="Directory containing split indices (train_indices.npy, val_indices.npy, test_indices.npy).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated splits to evaluate (e.g. train,val,test)",
    )
    parser.add_argument(
        "--ckpt",
        action="append",
        required=True,
        help="Model spec as name:path/to.ckpt (repeatable)",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10000,
        help="Max samples for silhouette computations",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="analysis_results/clustering",
        help="Output directory (relative to src root)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run identifier. If set, outputs go under outdir/<run-id>/",
    )
    args = parser.parse_args()

    lib_root = Path(os.getcwd())
    csv_path = (lib_root / args.csv_metadata).resolve()
    outdir = (lib_root / args.outdir).resolve()
    if args.run_id:
        outdir = outdir / str(args.run_id)
    outdir.mkdir(parents=True, exist_ok=True)
    split_dir = (lib_root / args.split_dir).resolve() if args.split_dir else None

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        raise SystemExit("No splits provided.")

    # Parse checkpoints
    models: List[ModelSpec] = []
    for spec in args.ckpt:
        if ":" not in spec:
            raise SystemExit(f"Bad --ckpt '{spec}'. Expected name:path.")
        name, path_str = spec.split(":", 1)
        ckpt_path = (lib_root / path_str).resolve()
        tok = _infer_tokenization_from_path(ckpt_path)
        models.append(ModelSpec(name=name, ckpt_path=ckpt_path, tokenization=tok))

    summary_rows: List[Dict] = []

    for split in splits:
        df_split = _load_split_dataframe(csv_path, split, split_dir=split_dir)
        pdes = df_split["pde"].astype(str).tolist()
        labels_all = _build_label_dict(df_split)

        for m in models:
            print("=" * 80)
            print(
                f"[Clustering] split={split} model={m.name} tokenization={m.tokenization}"
            )
            print(f"  ckpt: {m.ckpt_path}")
            print(f"  csv:  {csv_path}")

            model, hparams = _load_vae_from_checkpoint(m.ckpt_path, device=args.device)
            latents, valid_mask = _encode_pdes_to_latents(
                model=model,
                tokenization=m.tokenization,
                pdes=pdes,
                device=args.device,
                batch_size=args.batch_size,
            )

            n_valid = int(valid_mask.sum())
            print(f"  Encoded: {n_valid}/{len(pdes)} PDEs -> latents {latents.shape}")

            # Align labels to successfully-encoded rows
            labels_valid = {k: v[valid_mask] for k, v in labels_all.items()}

            results = compute_all_clustering(
                latents=latents,
                labels_dict=labels_valid,
                max_samples=args.max_samples,
                random_state=42,
            )

            # Save per-model per-split JSON
            out_json = outdir / split / f"{m.name}.json"
            payload = {
                "split": split,
                "model": {
                    "name": m.name,
                    "tokenization": m.tokenization,
                    "checkpoint": str(m.ckpt_path),
                },
                "dataset": {
                    "csv": str(csv_path),
                    "n_total": int(len(pdes)),
                    "n_encoded": int(n_valid),
                },
                "metrics": results,
            }
            _write_json(out_json, payload)
            print(f"  Saved: {out_json}")

            # Summarize NMI per label
            for label_name, metrics in results.items():
                summary_rows.append(
                    {
                        "split": split,
                        "model": m.name,
                        "tokenization": m.tokenization,
                        "label": label_name,
                        "nmi": metrics.get("nmi", np.nan),
                        "ari": metrics.get("ari", np.nan),
                        "purity": metrics.get("purity", np.nan),
                        "silhouette": metrics.get("silhouette", np.nan),
                        "silhouette_wrt_labels": metrics.get(
                            "silhouette_wrt_labels", np.nan
                        ),
                        "n_classes": metrics.get("n_classes", np.nan),
                        "n_encoded": n_valid,
                    }
                )

    # Write summary CSV (one table for quick comparisons)
    if summary_rows:
        summary_path = outdir / "summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print("=" * 80)
        print(f"Saved summary table: {summary_path}")


if __name__ == "__main__":
    main()
