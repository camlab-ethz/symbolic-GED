#!/usr/bin/env python3
"""Pack existing bool masks to uint8 for 8x compression."""
import numpy as np
import sys

if len(sys.argv) != 3:
    print("Usage: python pack_masks.py input_bool.npy output_packed.npy")
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2]

print(f"[pack] Loading {input_path}...")
masks = np.load(input_path)
N, T, P = masks.shape
print(f"[pack] Shape: {masks.shape}, dtype: {masks.dtype}")

# Pad P to multiple of 8
P_padded = ((P + 7) // 8) * 8
if P < P_padded:
    masks_padded = np.zeros((N, T, P_padded), dtype=bool)
    masks_padded[:, :, :P] = masks
    masks = masks_padded
    print(f"[pack] Padded P from {P} to {P_padded}")

# Pack along last axis
masks_packed = np.packbits(masks, axis=-1)
print(f"[pack] Packed shape: {masks_packed.shape}, dtype: {masks_packed.dtype}")

print(f"[pack] Saving to {output_path}...")
np.save(output_path, masks_packed)

print(f"[pack] Done. Original: {masks.nbytes / 1e6:.1f} MB, Packed: {masks_packed.nbytes / 1e6:.1f} MB")
