#!/usr/bin/env python3
"""Interpolate between PDE families in latent space and decode along the path.

This script is meant for *human inspection* of latent interpolation quality.
It:
  - selects endpoint PDEs from the metadata CSV by family + matching label keys
  - encodes endpoints to latent means (mu) for each VAE checkpoint
  - linearly interpolates z with a configurable step (default 0.05)
  - decodes each interpolated point to a *human-readable infix PDE string*
  - validates syntax (analysis.physics.is_valid_pde)
  - optionally classifies decoded PDEs with the rigorous PDEClassifier

Outputs:
  analysis_results/interpolation_examples/<pair_name>/<key_name>/<model_name>.csv
  analysis_results/interpolation_examples/<pair_name>/<key_name>/ENDPOINTS.json
  analysis_results/interpolation_examples/<pair_name>/<key_name>/ALL_MODELS.csv
  analysis_results/interpolation_examples/INDEX.csv

Config:
  You can run purely from CLI flags (backwards compatible), or provide a YAML
  config via --config. The YAML is intentionally simple:

    csv_metadata: data/raw/pde_dataset_48000_fixed.csv
    out_dir: analysis_results/interpolation_examples
    device: cuda
    step: 0.05
    classify: true
    models:
      - name: grammar_beta2e4
        ckpt: /abs/path/to.ckpt
        tokenization: grammar   # optional (auto-inferred)
      - ...
    pairs:
      - name: wave_to_heat
        family_a: wave
        family_b: heat
        match_cols: [dim, spatial_order]
        cover_all_keys: true
      - name: telegraph_to_wave
        family_a: telegraph
        family_b: wave
        match_cols: [dim, spatial_order, temporal_order]
        cover_all_keys: true
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.physics import is_valid_pde
from analysis.pde_classifier import PDEClassifier
from pde.grammar import PROD_COUNT, decode_production_sequence
from pde import PDETokenizer
from vae.module import VAEModule


@dataclass(frozen=True)
class ModelSpec:
    name: str
    ckpt: Path
    tokenization: str  # "grammar" | "token"


@dataclass(frozen=True)
class EndpointPick:
    family_a: str
    family_b: str
    match_cols: Tuple[str, ...]
    key: Dict[str, str]
    row_a: Dict
    row_b: Dict


def _infer_tokenization_from_ckpt(ckpt: Path) -> str:
    p = str(ckpt).lower()
    if "token_vae" in p or "/token_" in p:
        return "token"
    if "grammar_vae" in p or "/grammar_" in p:
        return "grammar"
    # fallback
    return "token" if "token" in p and "grammar" not in p else "grammar"


def _load_vae_from_checkpoint(ckpt: Path, device: str) -> VAEModule:
    checkpoint = torch.load(str(ckpt), map_location="cpu")
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
    return model


def _clean_for_grammar_parser(pde: str) -> str:
    # Grammar parser used in clustering strips "=0" after removing spaces.
    s = pde.replace(" ", "")
    s = re.sub(r"=0(\.0+)?$", "", s)
    return s


def _onehot_from_pde(
    pde: str, tokenization: str, max_length: int, vocab_size: int
) -> Optional[torch.Tensor]:
    """Return (T,P) onehot or None if encoding fails."""
    if tokenization == "grammar":
        from pde import grammar as pde_grammar

        try:
            seq = pde_grammar.parse_to_productions(_clean_for_grammar_parser(pde))
        except Exception:
            return None
        x = torch.zeros(max_length, vocab_size, dtype=torch.float32)
        for t, pid in enumerate(seq[:max_length]):
            if 0 <= pid < vocab_size:
                x[t, pid] = 1.0
        return x

    tokenizer = PDETokenizer()
    try:
        ids = tokenizer.encode(pde)
    except Exception:
        return None

    x = torch.zeros(max_length, vocab_size, dtype=torch.float32)
    for t, tid in enumerate(ids[:max_length]):
        if 0 <= tid < vocab_size:
            x[t, tid] = 1.0
    return x


def _encode_mu(
    model: VAEModule, tokenization: str, pde: str, device: str
) -> torch.Tensor:
    x = _onehot_from_pde(
        pde=pde,
        tokenization=tokenization,
        max_length=int(model.max_length),
        vocab_size=int(model.P),
    )
    if x is None:
        raise RuntimeError(
            f"Failed to encode PDE for tokenization='{tokenization}': {pde}"
        )
    x = x.unsqueeze(0).to(device)
    with torch.no_grad():
        mu, _ = model.encoder(x)
    return mu  # (1, z_dim)


def _prod_ids_to_string(prod_ids: Sequence[int]) -> str:
    try:
        valid = [int(pid) for pid in prod_ids if 0 <= int(pid) < PROD_COUNT]
        if not valid:
            return "[INVALID: No valid productions]"
        s = decode_production_sequence(valid)
        return s if s else "[INVALID: Empty sequence]"
    except Exception as e:
        return f"[ERROR: {e}]"


def _token_ids_to_string(token_ids: Sequence[int], tokenizer: PDETokenizer) -> str:
    try:
        vocab = tokenizer.vocab
        pad_id = vocab.pad_id
        eos_id = vocab.eos_id if hasattr(vocab, "eos_id") else None
        valid: List[int] = []
        for tid in token_ids:
            tid = int(tid)
            if tid < 0 or tid == pad_id:
                continue
            if eos_id is not None and tid == eos_id:
                break
            if tid < len(vocab.id2word):
                valid.append(tid)
        if not valid:
            return "[INVALID: No valid tokens]"
        s = tokenizer.decode_to_infix(valid)
        return s if s else "[INVALID: Empty sequence]"
    except Exception as e:
        return f"[ERROR: {e}]"


def _decode_batch(model: VAEModule, tokenization: str, z: torch.Tensor) -> List[str]:
    """Decode (B,z_dim) tensor into list of PDE strings."""
    tokenizer = PDETokenizer() if tokenization == "token" else None
    with torch.no_grad():
        if tokenization == "grammar":
            prod_ids = model.generate_constrained(z, greedy=True)  # (B,T)
            prod_ids = prod_ids.detach().cpu().numpy()
            return [_prod_ids_to_string(row.tolist()) for row in prod_ids]
        logits = model.decoder(z)  # (B,T,P)
        token_ids = logits.argmax(dim=-1).detach().cpu().numpy()
        assert tokenizer is not None
        return [_token_ids_to_string(row.tolist(), tokenizer) for row in token_ids]


def _alphas(step: float) -> np.ndarray:
    n = int(round(1.0 / step))
    # ensure 0..1 inclusive and stable against floating error
    a = np.linspace(0.0, 1.0, n + 1, dtype=np.float64)
    return a


def _pick_pair(
    df: pd.DataFrame,
    family_a: str,
    family_b: str,
    match_cols: Sequence[str],
    prefer: Optional[Dict[str, str]] = None,
) -> EndpointPick:
    """Pick one PDE from each family that matches on `match_cols`.

    `prefer` can pin specific values, e.g. {"dim": "1", "spatial_order": "2"}.
    """
    if prefer is None:
        prefer = {}

    for c in ["pde", "family"]:
        if c not in df.columns:
            raise RuntimeError(f"CSV missing required column '{c}'.")

    # Normalize columns to strings for stable matching
    work = df.copy()
    for c in match_cols:
        if c not in work.columns:
            raise RuntimeError(
                f"CSV missing match column '{c}'. Available: {list(df.columns)}"
            )
        work[c] = work[c].astype(str)

    a = work[work["family"].astype(str) == str(family_a)].copy()
    b = work[work["family"].astype(str) == str(family_b)].copy()
    if a.empty or b.empty:
        raise RuntimeError(
            f"Could not find families in CSV: {family_a=} ({len(a)}), {family_b=} ({len(b)})"
        )

    # Apply prefer constraints
    for k, v in prefer.items():
        if k not in work.columns:
            continue
        a = a[a[k].astype(str) == str(v)]
        b = b[b[k].astype(str) == str(v)]

    common_keys = _common_keys(a, b, match_cols=match_cols)
    if not common_keys:
        raise RuntimeError(
            f"No common keys for families {family_a} vs {family_b} on {match_cols}. "
            f"Try loosening match_cols or prefer constraints."
        )

    key = common_keys[0]

    # Deterministic selection: first row matching key
    a_row = a[(a[list(match_cols)] == pd.Series(key)).all(axis=1)].iloc[0]
    b_row = b[(b[list(match_cols)] == pd.Series(key)).all(axis=1)].iloc[0]

    return EndpointPick(
        family_a=family_a,
        family_b=family_b,
        match_cols=tuple(match_cols),
        key=key,
        row_a=a_row.to_dict(),
        row_b=b_row.to_dict(),
    )


def _common_keys(
    a_df: pd.DataFrame, b_df: pd.DataFrame, match_cols: Sequence[str]
) -> List[Dict[str, str]]:
    """Return sorted list of common key dicts between a_df and b_df over match_cols.

    Assumes match_cols are present and already cast to string.
    """
    a_keys = set(tuple(row[c] for c in match_cols) for _, row in a_df.iterrows())
    b_keys = set(tuple(row[c] for c in match_cols) for _, row in b_df.iterrows())
    common = sorted(a_keys & b_keys)
    out: List[Dict[str, str]] = []
    for tup in common:
        out.append({c: str(tup[i]) for i, c in enumerate(match_cols)})
    return out


def _key_to_dirname(key: Dict[str, str]) -> str:
    parts = []
    for k in sorted(key.keys()):
        v = str(key[k]).replace("/", "_")
        parts.append(f"{k}{v}")
    return "__".join(parts) if parts else "default"


def _load_yaml_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise RuntimeError(f"Config must be a mapping/dict. Got: {type(cfg)}")
    return cfg


def _models_from_config(cfg: Dict, base_dir: Path) -> List[ModelSpec]:
    models_cfg = cfg.get("models", None)
    if not models_cfg:
        raise RuntimeError("Config must include 'models:' list.")
    out: List[ModelSpec] = []
    for m in models_cfg:
        if "name" not in m or "ckpt" not in m:
            raise RuntimeError(f"Each model must have name and ckpt. Got: {m}")
        ckpt = Path(str(m["ckpt"]))
        if not ckpt.is_absolute():
            ckpt = base_dir / ckpt
        tokenization = str(m.get("tokenization") or _infer_tokenization_from_ckpt(ckpt))
        out.append(ModelSpec(name=str(m["name"]), ckpt=ckpt, tokenization=tokenization))
    return out


def _pairs_from_config(cfg: Dict) -> List[Dict]:
    pairs_cfg = cfg.get("pairs", None)
    if not pairs_cfg:
        raise RuntimeError("Config must include 'pairs:' list.")
    out: List[Dict] = []
    for p in pairs_cfg:
        for k in ["name", "family_a", "family_b", "match_cols"]:
            if k not in p:
                raise RuntimeError(f"Each pair must include {k}. Got: {p}")
        if not isinstance(p["match_cols"], (list, tuple)) or not p["match_cols"]:
            raise RuntimeError(
                f"pair.match_cols must be a non-empty list. Got: {p['match_cols']}"
            )
        out.append(
            {
                "name": str(p["name"]),
                "family_a": str(p["family_a"]),
                "family_b": str(p["family_b"]),
                "match_cols": [str(c) for c in p["match_cols"]],
                "cover_all_keys": bool(p.get("cover_all_keys", False)),
                # Optional: pin some columns (e.g. dim=1 to reduce)
                "prefer": {
                    str(k): str(v) for k, v in (p.get("prefer", {}) or {}).items()
                },
            }
        )
    return out


def _coerce_prefer(
    prefer: Dict[str, str], cli_prefer: Dict[str, str]
) -> Dict[str, str]:
    # CLI prefer applies as a global default; per-pair prefer overrides it
    merged = dict(cli_prefer)
    merged.update(prefer or {})
    return merged


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):  # type: ignore[no-untyped-def]
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

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_default)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Latent interpolation examples for 4 VAE models"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config file for interpolation suite (recommended for full coverage runs).",
    )
    parser.add_argument(
        "--csv-metadata",
        type=str,
        default=None,
        help="CSV with columns: pde, family, dim, temporal_order, spatial_order, ...",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory root",
    )
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    parser.add_argument(
        "--step", type=float, default=None, help="Interpolation alpha step (e.g. 0.05)"
    )
    parser.add_argument(
        "--no-classify",
        action="store_true",
        help="Skip PDEClassifier labeling (faster; still saves decoded strings + validity)",
    )
    parser.add_argument(
        "--prefer",
        type=str,
        default="",
        help="Comma-separated preferred constraints for selecting endpoints (e.g. dim=1,spatial_order=2)",
    )
    parser.add_argument(
        "--ckpt",
        action="append",
        default=[],
        help="Model spec as name:/abs/path/to.ckpt (repeatable). If omitted, uses the 4 default checkpoints.",
    )

    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    cfg: Dict = {}
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.is_absolute():
            cfg_path = base_dir / cfg_path
        cfg = _load_yaml_config(cfg_path)

    # Resolve config vs CLI (CLI wins only if explicitly provided).
    default_csv = "data/raw/pde_dataset_48000_fixed.csv"
    csv_path = Path(args.csv_metadata or cfg.get("csv_metadata") or default_csv)
    if not csv_path.is_absolute():
        csv_path = base_dir / csv_path

    default_out = "analysis_results/interpolation_examples"
    out_root = Path(args.out_dir or cfg.get("out_dir") or default_out)
    if not out_root.is_absolute():
        out_root = base_dir / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    device = str(args.device or cfg.get("device") or "cuda")
    step = float(args.step if args.step is not None else cfg.get("step", 0.05))
    # CLI: --no-classify should always disable classification, regardless of config
    classify = False if args.no_classify else bool(cfg.get("classify", True))

    # Defaults: 4 current checkpoints
    if args.config:
        model_specs = _models_from_config(cfg, base_dir=base_dir)
    elif not args.ckpt:
        defaults = {
            "grammar_beta2e4": base_dir
            / "checkpoints/grammar_vae/beta_2e-4_seed_42/best-epoch=380-val/seq_acc=0.9978.ckpt",
            "grammar_beta1e2": base_dir
            / "checkpoints/grammar_vae/beta_0.01_seed_42/best-epoch=517-val/seq_acc=0.0755.ckpt",
            "token_beta2e4": base_dir
            / "checkpoints/token_vae/beta_2e-4_seed_42/best-epoch=433-val/seq_acc=0.9962.ckpt",
            "token_beta1e2": base_dir
            / "checkpoints/token_vae/beta_0.01_seed_42/best-epoch=136-val/seq_acc=0.0016.ckpt",
        }
        model_specs = [
            ModelSpec(name=k, ckpt=v, tokenization=_infer_tokenization_from_ckpt(v))
            for k, v in defaults.items()
        ]
    else:
        model_specs = []
        for spec in args.ckpt:
            if ":" not in spec:
                raise SystemExit(f"Bad --ckpt '{spec}'. Expected name:/abs/path.ckpt")
            name, p = spec.split(":", 1)
            ckpt = Path(p)
            model_specs.append(
                ModelSpec(
                    name=name,
                    ckpt=ckpt,
                    tokenization=_infer_tokenization_from_ckpt(ckpt),
                )
            )

    prefer: Dict[str, str] = {}
    if args.prefer.strip():
        for part in args.prefer.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise SystemExit(f"Bad --prefer '{args.prefer}'. Expected k=v pairs.")
            k, v = part.split("=", 1)
            prefer[k.strip()] = v.strip()

    # Pair specifications
    if args.config:
        pairs = _pairs_from_config(cfg)
    else:
        # Backward-compatible defaults (single example per pair)
        pairs = [
            {
                "name": "wave_to_heat",
                "family_a": "wave",
                "family_b": "heat",
                "match_cols": ["dim", "spatial_order"],
                "cover_all_keys": False,
                "prefer": {},
            },
            {
                "name": "telegraph_to_wave",
                "family_a": "telegraph",
                "family_b": "wave",
                "match_cols": ["dim", "spatial_order", "temporal_order"],
                "cover_all_keys": False,
                "prefer": {},
            },
            {
                "name": "heat_to_allen_cahn",
                "family_a": "heat",
                "family_b": "allen_cahn",
                "match_cols": ["dim", "spatial_order", "temporal_order"],
                "cover_all_keys": False,
                "prefer": {},
            },
            {
                "name": "wave_to_klein_gordon",
                "family_a": "wave",
                "family_b": "klein_gordon",
                "match_cols": ["dim", "spatial_order", "temporal_order"],
                "cover_all_keys": False,
                "prefer": {},
            },
        ]

    df = pd.read_csv(csv_path)

    classifier = PDEClassifier() if classify else None
    alphas = _alphas(step)

    index_rows: List[Dict] = []

    for pair in pairs:
        pair_name = pair["name"]
        fam_a = pair["family_a"]
        fam_b = pair["family_b"]
        match_cols = pair["match_cols"]
        cover_all = bool(pair.get("cover_all_keys", False))
        pair_prefer = _coerce_prefer(pair.get("prefer", {}), prefer)

        pair_root = out_root / pair_name
        pair_root.mkdir(parents=True, exist_ok=True)

        # Normalize columns to strings for stable matching
        work = df.copy()
        for c in match_cols:
            if c not in work.columns:
                raise RuntimeError(
                    f"CSV missing match column '{c}' required by pair '{pair_name}'."
                )
            work[c] = work[c].astype(str)

        a_df = work[work["family"].astype(str) == str(fam_a)].copy()
        b_df = work[work["family"].astype(str) == str(fam_b)].copy()
        if a_df.empty or b_df.empty:
            raise RuntimeError(
                f"Pair '{pair_name}' missing family rows: {fam_a}({len(a_df)}), {fam_b}({len(b_df)})"
            )

        # Apply prefer constraints (global + per-pair)
        for k, v in pair_prefer.items():
            if k in work.columns:
                a_df = a_df[a_df[k].astype(str) == str(v)]
                b_df = b_df[b_df[k].astype(str) == str(v)]

        keys = _common_keys(a_df, b_df, match_cols=match_cols)
        if not keys:
            raise RuntimeError(
                f"Pair '{pair_name}': no common keys for families {fam_a} vs {fam_b} on {match_cols} with prefer={pair_prefer}."
            )
        if not cover_all:
            keys = keys[:1]

        for key in keys:
            key_dirname = _key_to_dirname(key)
            out_dir = pair_root / key_dirname
            out_dir.mkdir(parents=True, exist_ok=True)

            a_row = (
                a_df[(a_df[list(match_cols)] == pd.Series(key)).all(axis=1)]
                .iloc[0]
                .to_dict()
            )
            b_row = (
                b_df[(b_df[list(match_cols)] == pd.Series(key)).all(axis=1)]
                .iloc[0]
                .to_dict()
            )
            pde_a = str(a_row["pde"])
            pde_b = str(b_row["pde"])

            endpoints_payload = {
                "pair": {
                    "name": pair_name,
                    "family_a": fam_a,
                    "family_b": fam_b,
                    "match_cols": list(match_cols),
                    "key": key,
                    "cover_all_keys": cover_all,
                    "prefer": pair_prefer,
                },
                "row_a": a_row,
                "row_b": b_row,
            }
            _write_json(out_dir / "ENDPOINTS.json", endpoints_payload)

            all_rows: List[Dict] = []
            model_paths: Dict[str, str] = {}

            for ms in model_specs:
                model = _load_vae_from_checkpoint(ms.ckpt, device=device)
                z_a = _encode_mu(model, ms.tokenization, pde_a, device=device)  # (1,z)
                z_b = _encode_mu(model, ms.tokenization, pde_b, device=device)  # (1,z)

                z_path = []
                for a in alphas:
                    z_path.append((1.0 - float(a)) * z_a + float(a) * z_b)
                z_path_t = torch.cat(z_path, dim=0)  # (K,z)

                decoded = _decode_batch(model, ms.tokenization, z_path_t)

                rows = []
                for i, a in enumerate(alphas):
                    s = decoded[i]
                    valid = bool(is_valid_pde(s))
                    rec: Dict = {
                        "pair": pair_name,
                        "key": json.dumps(key, sort_keys=True),
                        "model": ms.name,
                        "tokenization": ms.tokenization,
                        "alpha": float(a),
                        "decoded_pde": s,
                        "valid_syntax": valid,
                    }
                    if classifier is not None and valid:
                        try:
                            labels = classifier.classify(s)
                            rec.update(
                                {
                                    "pred_family": labels.family,
                                    "pred_type": labels.pde_type,
                                    "pred_linearity": labels.linearity,
                                    "pred_dim": labels.dimension,
                                    "pred_spatial_order": labels.spatial_order,
                                    "pred_temporal_order": labels.temporal_order,
                                    "pred_confidence": float(labels.confidence),
                                }
                            )
                        except Exception:
                            pass
                    rows.append(rec)

                df_out = pd.DataFrame(rows)
                model_csv = out_dir / f"{ms.name}.csv"
                df_out.to_csv(model_csv, index=False)
                model_paths[ms.name] = str(model_csv)
                all_rows.extend(rows)

            all_csv = out_dir / "ALL_MODELS.csv"
            pd.DataFrame(all_rows).to_csv(all_csv, index=False)

            index_rows.append(
                {
                    "pair": pair_name,
                    "family_a": fam_a,
                    "family_b": fam_b,
                    "match_cols": ",".join(match_cols),
                    "key_dir": key_dirname,
                    "key_json": json.dumps(key, sort_keys=True),
                    "endpoint_a": pde_a,
                    "endpoint_b": pde_b,
                    "all_models_csv": str(all_csv),
                    **{f"model_csv__{k}": v for k, v in model_paths.items()},
                }
            )

    if index_rows:
        index_path = out_root / "INDEX.csv"
        pd.DataFrame(index_rows).to_csv(index_path, index=False)

    print(f"✓ Wrote interpolation examples to: {out_root}")


if __name__ == "__main__":
    main()
