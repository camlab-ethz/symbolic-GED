"""
Example: tokenize PDE strings and produce one-hot token encodings and production masks.
"""

from pathlib import Path
from src.tokenizer import (
    operator_tokenize,
    char_tokenize,
    num_tokenize_as_digits,
    to_prefix,
)
from src.vocab import build_vocab, tokens_to_ids
from src.onehot import (
    tokens_to_onehot,
    batch_to_onehot,
    tokens_to_onehot_torch,
    productions_to_onehot_torch,
)
from src.mask_adapter import string_to_production_sequence_and_masks
from src.pde import grammar as pde_grammar
from src.io import save_tensor
import torch


def main():
    data_file = Path(__file__).parent.parent / "pde_dataset.txt"
    pdes = [l.strip() for l in open(data_file) if l.strip()][:20]

    print("Sample PDEs:")
    for p in pdes[:5]:
        print("  ", p)

    # Build char-level vocab
    char_lists = [char_tokenize(p) for p in pdes]
    char_v2i, char_i2v = build_vocab(char_lists)

    # Build operator-level vocab
    op_lists = [operator_tokenize(p) for p in pdes]
    op_v2i, op_i2v = build_vocab(op_lists)

    print("\nChar vocab size:", len(char_v2i))
    print("Op vocab size  :", len(op_v2i))

    # Example: take first PDE
    p = pdes[0]
    op_tokens = operator_tokenize(p)
    print("\nOperator tokens:", op_tokens)
    op_ids = tokens_to_ids(op_tokens, op_v2i)
    onehot = tokens_to_onehot(op_ids, len(op_v2i))
    onehot_t = tokens_to_onehot_torch(op_ids, len(op_v2i))
    print("One-hot shape (L,V):", onehot.shape)
    print("Torch one-hot shape:", tuple(onehot_t.shape))
    # save token-level one-hot tensor
    save_tensor(onehot_t, "examples_out/token_onehot.pt")

    # Production sequence and masks
    seq, masks = string_to_production_sequence_and_masks(p)
    print("Production seq length:", len(seq))
    print("Masks shape:", masks.shape)
    # convert production sequence to torch one-hot
    P = pde_grammar.PROD_COUNT
    prod_onehot_t = productions_to_onehot_torch(seq, P)
    print("Production one-hot shape (T,P):", tuple(prod_onehot_t.shape))
    save_tensor(
        {
            "prod_seq": torch.tensor(seq, dtype=torch.long),
            "prod_onehot": prod_onehot_t,
            "masks": torch.tensor(masks),
        },
        "examples_out/prod_seq_and_masks.pt",
    )


if __name__ == "__main__":
    main()
