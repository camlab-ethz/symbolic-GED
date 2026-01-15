"""Encode canonical PDEs for two model types:
- Token-level (Lample & Charton char-digit): char tokens with digits split -> token one-hot (B, T, V)
- Grammar-production level (Kusner-style): production id sequences + per-step masks -> one-hot (B, T, P) and masks (B, T, P)

Saves tensors to `examples_out/` using `src.io.save_tensor`.

Usage: run as script from repo root with PYTHONPATH set, or from within package.
"""

from __future__ import annotations
import os
from typing import List
import torch

from src import tokenizer, vocab, onehot, io
from src.pde import grammar as pde_grammar


def load_pdes(path: str, max_items: int = 200) -> List[str]:
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines[:max_items]


def build_token_vocab(token_lists: List[List[str]]):
    t2i, i2t = vocab.build_vocab(token_lists)
    return t2i, i2t


def encode_tokens_batch(
    token_lists: List[List[str]], token2id: dict, add_sos_eos: bool = False
):
    # returns torch.Tensor (B, T, V)
    return onehot.tokens_list_to_onehot_torch(
        token_lists, token2id, add_sos_eos=add_sos_eos
    )


def encode_productions_batch(prod_seqs: List[List[int]]):
    P = pde_grammar.PROD_COUNT
    prod_onehot = onehot.batch_productions_to_onehot_torch(prod_seqs, P)
    masks = [pde_grammar.build_masks_from_production_sequence(s) for s in prod_seqs]
    masks_t = onehot.batch_masks_to_tensor(masks)
    return prod_onehot, masks_t


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(repo_root, "pde_dataset.txt")
    out_dir = os.path.join(repo_root, "examples_out")
    os.makedirs(out_dir, exist_ok=True)

    pdes = load_pdes(data_path, max_items=200)
    print(f"Loaded {len(pdes)} PDEs")

    # Tokenization: Lample & Charton style -> char tokens with digits split
    token_lists = [tokenizer.num_tokenize_as_digits(s.replace(" ", "")) for s in pdes]
    print("Sample tokenized (char-digit) example:", token_lists[0][:50])

    token2id, id2token = build_token_vocab(token_lists)
    print("Token vocab size:", len(token2id))

    token_onehot = encode_tokens_batch(token_lists, token2id, add_sos_eos=False)
    print("Token one-hot shape:", tuple(token_onehot.shape))
    io.save_tensor(token_onehot, os.path.join(out_dir, "token_onehot_char.pt"))

    # Grammar productions + masks
    prod_seqs = []
    failed = []
    for i, s in enumerate(pdes):
        try:
            seq = pde_grammar.parse_to_productions(s)
            prod_seqs.append(seq)
        except Exception as e:
            failed.append((i, s, str(e)))
    print(
        f"Parsed {len(prod_seqs)} / {len(pdes)} PDEs into production sequences; failed {len(failed)}"
    )

    prod_onehot, masks = encode_productions_batch(prod_seqs)
    print("Production one-hot shape:", tuple(prod_onehot.shape))
    print("Masks shape:", tuple(masks.shape))
    io.save_tensor(prod_onehot, os.path.join(out_dir, "prod_onehot.pt"))
    io.save_tensor(masks, os.path.join(out_dir, "prod_masks.pt"))

    # Also save the token vocab for later decoding
    io.save_tensor(
        {"token2id": token2id, "id2token": id2token},
        os.path.join(out_dir, "token_vocab.pt"),
    )

    print("Saved tensors to", out_dir)


if __name__ == "__main__":
    main()
