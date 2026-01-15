from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class ProductionDataset(Dataset):
    """Dataset that lazily converts production IDs to one-hot vectors.

    Supports two storage backends:
      • In-memory torch tensors (legacy .pt files)
      • Memory-mapped NumPy arrays (.npy files)
    """

    def __init__(self, ids_source, masks_source, P: int, use_memmap: bool, start_idx: int = 0, end_idx: Optional[int] = None, indices: Optional[np.ndarray] = None):
        self.P = P
        self.use_memmap = use_memmap
        self.targets_dtype = torch.long
        self.indices = indices  # Custom indices for train/val/test split

        if self.use_memmap:
            # Defer opening memmaps to worker processes to avoid fd duplication issues
            self.ids_path = ids_source
            self.masks_path = masks_source
            self._ids_memmap = None
            self._masks_memmap = None
            self._masks_packed = None  # Will detect packing on first load

            ids_tmp = np.load(self.ids_path, mmap_mode="r")
            total_N, self.T = ids_tmp.shape
            if hasattr(ids_tmp, "_mmap"):
                ids_tmp._mmap.close()
            del ids_tmp

            masks_tmp = np.load(self.masks_path, mmap_mode="r")
            # Detect if masks are packed (uint8) or unpacked (bool)
            if masks_tmp.dtype == np.uint8 and masks_tmp.ndim == 3:
                # Packed format: (N, T, bytes)
                if masks_tmp.shape[:2] != (total_N, self.T):
                    raise ValueError("Mask shape does not match ids shape")
                self._masks_packed = True
                # When packed, masks are padded to multiple of 8, but actual vocab is 55
                # Use the P parameter passed to __init__ (which should be 55)
                # Don't override self.P here - it was already set from the parameter
            elif masks_tmp.dtype == bool and masks_tmp.ndim == 3:
                # Unpacked format: (N, T, P)
                if masks_tmp.shape[:2] != (total_N, self.T):
                    raise ValueError("Mask shape does not match ids shape")
                self._masks_packed = False
                self.P = masks_tmp.shape[-1]
            else:
                raise ValueError(f"Unexpected mask format: dtype={masks_tmp.dtype}, shape={masks_tmp.shape}")
            
            if hasattr(masks_tmp, "_mmap"):
                masks_tmp._mmap.close()
            del masks_tmp
            self.start = start_idx
            self.end = end_idx if end_idx is not None else total_N
            
            # If custom indices provided, use them; otherwise use range
            if self.indices is not None:
                self.N = len(self.indices)
            else:
                self.N = self.end - self.start
            self._total_N = total_N
        else:
            self.prod_ids = ids_source  # (N, T) torch tensor
            self.masks = masks_source.bool()  # (N, T, P)
            self.N = self.prod_ids.shape[0]
            self.T = self.prod_ids.shape[1]
            self.start = 0

    def __len__(self):
        return self.N

    def _ensure_memmaps(self):
        if self._ids_memmap is None:
            self._ids_memmap = np.load(self.ids_path, mmap_mode="r")
        if self._masks_memmap is None:
            self._masks_memmap = np.load(self.masks_path, mmap_mode="r")

    def __getitem__(self, idx):
        # Map idx to actual global index
        if self.indices is not None:
            global_idx = self.indices[idx]
        else:
            global_idx = self.start + idx

        if self.use_memmap:
            self._ensure_memmaps()
            prod_id_np = np.asarray(self._ids_memmap[global_idx], dtype=np.int16)
            
            if self._masks_packed:
                # Unpack bits for this sample
                mask_packed_np = np.asarray(self._masks_memmap[global_idx], dtype=np.uint8)
                mask_np = np.unpackbits(mask_packed_np, axis=-1).astype(bool)
                # Trim to actual P if it was padded
                actual_P = self.P
                if mask_np.shape[-1] > actual_P:
                    mask_np = mask_np[:, :actual_P]
            else:
                mask_np = np.asarray(self._masks_memmap[global_idx], dtype=np.bool_)
            
            # Copy to make arrays writable (silences PyTorch warning)
            prod_id = torch.from_numpy(prod_id_np.copy()).to(self.targets_dtype)
            mask = torch.from_numpy(mask_np.copy())
        else:
            prod_id = self.prod_ids[idx].to(self.targets_dtype)
            mask = self.masks[idx]
        
        # Convert to one-hot encoding on the fly
        prod_id_clamped = prod_id.clamp_min(0)
        prod_onehot = torch.nn.functional.one_hot(prod_id_clamped, num_classes=self.P).to(torch.float32)

        padding_mask = prod_id < 0
        if padding_mask.any():
            prod_onehot[padding_mask] = 0.0

        return prod_onehot, prod_id, mask


class GrammarVAEDataModule(pl.LightningDataModule):
    def __init__(self, prod_path: str, masks_path: str, batch_size: int = 32, num_workers: int = 2, pin_memory: bool = False, persistent_workers: bool = False, prefetch_factor: Optional[int] = None, split_dir: Optional[str] = None):
        super().__init__()
        self.prod_path = prod_path
        self.masks_path = masks_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = 2 if prefetch_factor is None else prefetch_factor
        self.split_dir = split_dir  # If provided, use pre-defined train/val/test splits

        self.use_memmap = prod_path.endswith(".npy")

        if self.use_memmap:
            ids_mm = np.load(self.prod_path, mmap_mode="r")
            masks_mm = np.load(self.masks_path, mmap_mode="r")
            if ids_mm.ndim != 2:
                raise ValueError(f"Expected ids .npy with shape (N, T), got {ids_mm.shape}")
            
            # Detect packed or unpacked masks
            if masks_mm.dtype == np.uint8 and masks_mm.ndim == 3:
                # Packed: (N, T, bytes)
                self._masks_packed = True
                # When packed, P is padded to multiple of 8, but actual vocab is 55
                # Calculate actual P from packed size: 7 bytes = 56 bits, but actual P=55
                self.P = masks_mm.shape[-1] * 8  # This is P_padded (56)
                # We'll trim to actual P=55 in __getitem__
                # But for now, we need to know the actual vocab size
                # For grammar, actual vocab_size is 55 (hardcoded - matches config)
                self.P = 55  # Actual vocab size (not padded)
            elif masks_mm.dtype == bool and masks_mm.ndim == 3:
                # Unpacked: (N, T, P)
                self._masks_packed = False
                self.P = masks_mm.shape[-1]
            else:
                raise ValueError(f"Expected masks .npy with dtype uint8 (packed) or bool, got {masks_mm.dtype} with shape {masks_mm.shape}")
            
            self.N, self.T = ids_mm.shape
            if hasattr(ids_mm, "_mmap"):
                ids_mm._mmap.close()
            if hasattr(masks_mm, "_mmap"):
                masks_mm._mmap.close()
            del ids_mm, masks_mm
            print(f"[DataModule] Using NumPy memmap backend ({self.N} sequences, T={self.T}, P={self.P}, masks_packed={self._masks_packed})")
            self.prod_ids = None
            self.masks = None
        else:
            print(f"[DataModule] Loading data from {prod_path} (torch tensors)...")
            self.prod_ids = torch.load(self.prod_path)
            self.masks = torch.load(self.masks_path).bool()

            if self.prod_ids.dim() == 3:
                print("[DataModule] Received one-hot tensor, converting to ids")
                self.P = self.prod_ids.shape[-1]
                self.prod_ids = self.prod_ids.argmax(dim=-1)
            elif self.prod_ids.dim() == 2:
                self.P = self.masks.shape[-1]
            else:
                raise ValueError(f"Unexpected prod_ids shape {self.prod_ids.shape}")

            self.N = self.prod_ids.shape[0]
            self.T = self.prod_ids.shape[1]
            print(f"[DataModule] Loaded {self.N} sequences, max length {self.T}, productions {self.P}")

    def setup(self, stage: Optional[str] = None):
        # Use pre-defined splits if provided
        if self.split_dir is not None:
            import os
            train_idx_path = os.path.join(self.split_dir, 'train_indices.npy')
            val_idx_path = os.path.join(self.split_dir, 'val_indices.npy')
            test_idx_path = os.path.join(self.split_dir, 'test_indices.npy')
            
            if not os.path.exists(train_idx_path):
                raise FileNotFoundError(f"Split indices not found in {self.split_dir}. Run create_train_val_test_split.py first!")
            
            train_idx = np.load(train_idx_path)
            val_idx = np.load(val_idx_path)
            test_idx = np.load(test_idx_path)
            
            if self.use_memmap:
                self.train_ds = ProductionDataset(self.prod_path, self.masks_path, self.P, use_memmap=True, start_idx=0, end_idx=self.N, indices=train_idx)
                self.val_ds = ProductionDataset(self.prod_path, self.masks_path, self.P, use_memmap=True, start_idx=0, end_idx=self.N, indices=val_idx)
                self.test_ds = ProductionDataset(self.prod_path, self.masks_path, self.P, use_memmap=True, start_idx=0, end_idx=self.N, indices=test_idx)
            else:
                self.train_ds = ProductionDataset(self.prod_ids[train_idx], self.masks[train_idx], self.P, use_memmap=False)
                self.val_ds = ProductionDataset(self.prod_ids[val_idx], self.masks[val_idx], self.P, use_memmap=False)
                self.test_ds = ProductionDataset(self.prod_ids[test_idx], self.masks[test_idx], self.P, use_memmap=False)
            
            print(f"[DataModule] Using pre-defined splits from {self.split_dir}")
            print(f"  Train: {len(train_idx)} sequences")
            print(f"  Val:   {len(val_idx)} sequences")
            print(f"  Test:  {len(test_idx)} sequences")
        else:
            # Fallback to simple 90/10 split (no test set) - LEGACY BEHAVIOR
            split = int(0.9 * self.N)
            if self.use_memmap:
                self.train_ds = ProductionDataset(self.prod_path, self.masks_path, self.P, use_memmap=True, start_idx=0, end_idx=split)
                self.val_ds = ProductionDataset(self.prod_path, self.masks_path, self.P, use_memmap=True, start_idx=split, end_idx=self.N)
            else:
                self.train_ds = ProductionDataset(self.prod_ids[:split], self.masks[:split], self.P, use_memmap=False)
                self.val_ds = ProductionDataset(self.prod_ids[split:], self.masks[split:], self.P, use_memmap=False)

            print(f"[DataModule] Train: {split} sequences, Val: {self.N - split}")
            print(f"[WARNING] No test set! For fair comparison, use --split_dir argument.")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def test_dataloader(self):
        """Return test dataloader if test set exists."""
        if hasattr(self, 'test_ds'):
            return DataLoader(
                self.test_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers and self.num_workers > 0,
                prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            )
        else:
            raise ValueError("No test set available. Use --split_dir to enable test set.")
