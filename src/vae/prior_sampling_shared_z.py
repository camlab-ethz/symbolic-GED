#!/usr/bin/env python3
"""Prior sampling with a SHARED latent matrix Z across models.

This is the "prior generation" companion to dataset-interpolation:
- sample a single fixed Z ~ N(0, I) with a fixed seed
- decode the exact same Z with each model checkpoint
- compute simple stats (validity, uniqueness, signature diversity)
- write per-model CSVs + one summary CSV

Designed to work with operator-only PDE strings (no '= 0' required).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch

# Make `src/` importable when running as `python -m vae...`
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from vae.module import VAEModule
from pde.grammar import PROD_COUNT, decode_production_sequence
from pde.chr_tokenizer import PDETokenizer
from analysis.physics import is_valid_pde
from pde_string_utils import canonicalize, signature


@dataclass
class ModelSpec:
    name: str
    tokenization: str  # "grammar" | "token"
    checkpoint: str


def _format_signature(sig: Tuple) -> str:
    try:
        return json.dumps(sig, ensure_ascii=False)
    except TypeError:
        return repr(sig)


def load_model(checkpoint_path: str, tokenization_type: str, device: str) -> VAEModule:
    model = VAEModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    model = model.to(device)
    model._tokenization_type = tokenization_type
    return model


def _decode_grammar_batch(model: VAEModule, z: torch.Tensor) -> List[str]:
    with torch.no_grad():
        prod_ids = model.generate_constrained(z, greedy=True)  # (B, T)
    out: List[str] = []
    for row in prod_ids.detach().cpu().tolist():
        valid_ids = [pid for pid in row if 0 <= pid < PROD_COUNT]
        out.append(decode_production_sequence(valid_ids))
    return out


def _decode_token_batch(
    model: VAEModule, z: torch.Tensor, tokenizer: PDETokenizer
) -> List[str]:
    with torch.no_grad():
        logits = model.decoder(z)  # (B, T, P)
        token_ids = logits.argmax(dim=-1).detach().cpu().tolist()  # List[List[int]]

    vocab = tokenizer.vocab
    pad_id = getattr(vocab, "pad_id", None)
    eos_id = getattr(vocab, "eos_id", None)

    out: List[str] = []
    for row in token_ids:
        kept: List[int] = []
        for tid in row:
            if tid < 0:
                continue
            if pad_id is not None and tid == pad_id:
                continue
            if eos_id is not None and tid == eos_id:
                break
            kept.append(tid)
        out.append(tokenizer.decode_to_infix(kept, skip_special_tokens=True))
    return out


def iter_batches(n: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield start, end


def main() -> None:
    p = argparse.ArgumentParser(
        description="Prior sampling with shared Z across 4 models"
    )
    p.add_argument("--grammar_beta2e4", type=str, required=True)
    p.add_argument("--grammar_beta1e2", type=str, required=True)
    p.add_argument("--token_beta2e4", type=str, required=True)
    p.add_argument("--token_beta1e2", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--n_samples", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--save_z", action="store_true", help="Save sampled Z to outdir (pt)"
    )
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    specs: List[ModelSpec] = [
        ModelSpec("grammar_beta2e4", "grammar", args.grammar_beta2e4),
        ModelSpec("grammar_beta1e2", "grammar", args.grammar_beta1e2),
        ModelSpec("token_beta2e4", "token", args.token_beta2e4),
        ModelSpec("token_beta1e2", "token", args.token_beta1e2),
    ]

    # Load models
    models: Dict[str, VAEModule] = {}
    for spec in specs:
        print(f"Loading {spec.name} from {spec.checkpoint}")
        models[spec.name] = load_model(spec.checkpoint, spec.tokenization, args.device)

    z_dims = {name: m.z_dim for name, m in models.items()}
    z_dim = next(iter(z_dims.values()))
    if any(d != z_dim for d in z_dims.values()):
        raise ValueError(f"Model z_dim mismatch: {z_dims}")

    # Sample SHARED latent matrix
    print(
        f"Sampling shared Z ~ N(0, I): n={args.n_samples}, z_dim={z_dim}, seed={args.seed}"
    )
    g = torch.Generator(device=args.device)
    g.manual_seed(args.seed)
    z = torch.randn(args.n_samples, z_dim, generator=g, device=args.device)

    if args.save_z:
        z_path = outdir / f"Z_n{args.n_samples}_seed{args.seed}_zdim{z_dim}.pt"
        torch.save(z.detach().cpu(), z_path)
        print(f"Saved Z to {z_path}")

    # Save run metadata
    meta = {
        "n_samples": args.n_samples,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": args.device,
        "z_dim": z_dim,
        "models": [asdict(s) for s in specs],
    }
    (outdir / "RUN_METADATA.json").write_text(json.dumps(meta, indent=2))

    tokenizer = PDETokenizer()
    summary_rows: List[Dict[str, object]] = []

    for spec in specs:
        model = models[spec.name]
        rows_out: List[Dict[str, object]] = []

        valid_canon: List[str] = []
        valid_sigs: List[str] = []
        valid_count = 0

        print(f"\nDecoding {spec.name} ({spec.tokenization}) ...")
        for start, end in iter_batches(args.n_samples, args.batch_size):
            z_b = z[start:end]
            if spec.tokenization == "grammar":
                decoded = _decode_grammar_batch(model, z_b)
            else:
                decoded = _decode_token_batch(model, z_b, tokenizer)

            for i, pde_str in enumerate(decoded, start=start):
                ok = is_valid_pde(pde_str)
                canon = canonicalize(pde_str) if pde_str else ""
                sig = _format_signature(signature(pde_str)) if pde_str else ""

                if ok:
                    valid_count += 1
                    valid_canon.append(canon)
                    valid_sigs.append(sig)

                rows_out.append(
                    {
                        "idx": i,
                        "pde": pde_str,
                        "valid": int(ok),
                        "canonical": canon,
                        "signature": sig,
                    }
                )

            if end % (args.batch_size * 10) == 0 or end == args.n_samples:
                print(f"  progress: {end}/{args.n_samples} decoded")

        unique_valid = len(set(valid_canon))
        unique_sig = len(set(valid_sigs))
        valid_ratio = valid_count / max(1, args.n_samples)
        unique_ratio = unique_valid / max(1, valid_count)

        summary_rows.append(
            {
                "model": spec.name,
                "tokenization": spec.tokenization,
                "checkpoint": spec.checkpoint,
                "n_samples": args.n_samples,
                "valid_count": valid_count,
                "valid_ratio": valid_ratio,
                "unique_valid_canonical": unique_valid,
                "unique_valid_ratio": unique_ratio,
                "unique_signature_valid": unique_sig,
            }
        )

        csv_path = outdir / f"{spec.name}_decoded_n{args.n_samples}_seed{args.seed}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["idx", "pde", "valid", "canonical", "signature"]
            )
            w.writeheader()
            w.writerows(rows_out)
        print(f"  wrote: {csv_path}")
        print(
            f"  valid: {valid_count}/{args.n_samples} ({100*valid_ratio:.2f}%) | "
            f"unique(valid): {unique_valid}/{valid_count} ({100*unique_ratio:.2f}%) | "
            f"unique(signature): {unique_sig}"
        )

    summary_path = outdir / f"summary_n{args.n_samples}_seed{args.seed}.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "tokenization",
                "checkpoint",
                "n_samples",
                "valid_count",
                "valid_ratio",
                "unique_valid_canonical",
                "unique_valid_ratio",
                "unique_signature_valid",
            ],
        )
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nWrote summary: {summary_path}")


if __name__ == "__main__":
    main()
