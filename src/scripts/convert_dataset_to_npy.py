#!/usr/bin/env python3
"""Convert Grammar-VAE dataset tensors to NumPy formats optimized for memory-mapped loading.

Usage:
    python convert_dataset_to_npy.py \
        --ids src/examples_out/prod_ids_48444.pt \
        --masks src/examples_out/prod_masks_48444.pt \
        --out-prefix src/examples_out/prod_48444

Outputs:
    {out-prefix}_ids_int16.npy   # int16 tensor of shape (N, T)
    {out-prefix}_masks_bool.npy  # bool tensor of shape (N, T, P)
    {out-prefix}_meta.json       # metadata with N, T, P
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Grammar-VAE dataset to NumPy memmap-friendly format"
    )
    parser.add_argument(
        "--ids", required=True, type=Path, help="Path to production ids .pt file"
    )
    parser.add_argument(
        "--masks", required=True, type=Path, help="Path to production masks .pt file"
    )
    parser.add_argument(
        "--out-prefix",
        required=True,
        type=Path,
        help="Output prefix (without extension)",
    )
    args = parser.parse_args()

    out_ids = args.out_prefix.with_name(args.out_prefix.name + "_ids_int16.npy")
    out_masks = args.out_prefix.with_name(args.out_prefix.name + "_masks_bool.npy")
    out_meta = args.out_prefix.with_name(args.out_prefix.name + "_meta.json")

    print(f"[convert] Loading ids tensor from {args.ids} ...")
    prod_ids = torch.load(args.ids).cpu()
    if prod_ids.dim() != 2:
        raise ValueError(
            f"Expected ids tensor with 2 dims (N, T), got shape {tuple(prod_ids.shape)}"
        )

    print(f"[convert] Loading masks tensor from {args.masks} ...")
    masks = torch.load(args.masks).cpu()
    if masks.dim() != 3:
        raise ValueError(
            f"Expected mask tensor with 3 dims (N, T, P), got shape {tuple(masks.shape)}"
        )

    if prod_ids.shape[0] != masks.shape[0] or prod_ids.shape[1] != masks.shape[1]:
        raise ValueError("Ids and masks tensor shapes do not align")

    N, T = prod_ids.shape
    P = masks.shape[2]
    print(f"[convert] Dataset size: N={N}, T={T}, P={P}")

    # Convert ids to int16 for compact storage (values include -1 padding and production ids < P)
    if prod_ids.dtype != torch.int16:
        ids_np = prod_ids.to(torch.int16).numpy()
    else:
        ids_np = prod_ids.numpy()

    print(f"[convert] Writing ids to {out_ids} (dtype=int16)...")
    np.save(out_ids, ids_np)

    # Pack masks into bits for 8x compression (56 productions -> 7 bytes per timestep)
    mask_bool = masks.bool().numpy()
    # Pad to multiple of 8 for packing
    P_padded = ((P + 7) // 8) * 8
    if P < P_padded:
        mask_bool_padded = np.zeros((N, T, P_padded), dtype=bool)
        mask_bool_padded[:, :, :P] = mask_bool
        mask_bool = mask_bool_padded
    # Pack along production dimension
    mask_packed = np.packbits(mask_bool, axis=-1)
    print(f"[convert] Writing packed masks to {out_masks} (dtype=uint8, packed)...")
    np.save(out_masks, mask_packed)

    meta = {
        "N": int(N),
        "T": int(T),
        "P": int(P),
        "P_padded": int(P_padded),
        "ids_path": str(out_ids.name),
        "masks_path": str(out_masks.name),
        "masks_packed": True,
    }
    print(f"[convert] Writing metadata to {out_meta} ...")
    out_meta.write_text(json.dumps(meta, indent=2))

    print("[convert] Done.")


if __name__ == "__main__":
    main()
