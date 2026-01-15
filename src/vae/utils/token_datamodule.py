"""Token VAE DataModule - uses same structure as Grammar VAE but with token sequences."""
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pytorch_lightning as pl
import numpy as np
import os


class TokenVAEDataModule(pl.LightningDataModule):
    """DataModule for token-based VAE (Lample & Charton style)."""
    
    def __init__(self, token_path: str, masks_path: str, batch_size: int = 64,
                 num_workers: int = 2, split_dir: str = None, vocab_size: int = 82):
        super().__init__()
        self.token_path = token_path
        self.masks_path = masks_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_dir = split_dir
        self.vocab_size = vocab_size
        
        # Load data using memmap (same as grammar VAE)
        self.tokens = np.load(token_path, mmap_mode='r')  # (N, T)
        self.masks = np.load(masks_path, mmap_mode='r')   # (N, T) boolean
        
        self.N = len(self.tokens)
        self.T = self.tokens.shape[1]  # max_length
        
        print(f"[TokenDataModule] Loaded {self.N} token sequences, max_length={self.T}")
    
    def setup(self, stage=None):
        """Load train/val/test splits."""
        if self.split_dir:
            print(f"[TokenDataModule] Using pre-defined splits from {self.split_dir}")
            train_idx = np.load(os.path.join(self.split_dir, 'train_indices.npy'))
            val_idx = np.load(os.path.join(self.split_dir, 'val_indices.npy'))
            test_idx = np.load(os.path.join(self.split_dir, 'test_indices.npy'))
            
            # Pass paths instead of memmap objects for proper multiprocessing
            self.train_dataset = Subset(TokenDataset(self.token_path, self.masks_path, self.vocab_size), train_idx)
            self.val_dataset = Subset(TokenDataset(self.token_path, self.masks_path, self.vocab_size), val_idx)
            self.test_dataset = Subset(TokenDataset(self.token_path, self.masks_path, self.vocab_size), test_idx)
            
            print(f"  Train: {len(train_idx)} sequences")
            print(f"  Val:   {len(val_idx)} sequences")
            print(f"  Test:  {len(test_idx)} sequences")
        else:
            # Random split
            n_val = int(0.1 * self.N)
            n_test = int(0.1 * self.N)
            n_train = self.N - n_val - n_test
            
            full_dataset = TokenDataset(self.token_path, self.masks_path, self.vocab_size)
            self.train_dataset, self.val_dataset, self.test_dataset = \
                torch.utils.data.random_split(full_dataset, [n_train, n_val, n_test])
    
    def train_dataloader(self):
        # Disable pin_memory when using multiprocessing to avoid CUDA init errors in workers
        pin_mem = self.num_workers == 0
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                         shuffle=True, num_workers=self.num_workers, pin_memory=pin_mem)
    
    def val_dataloader(self):
        pin_mem = self.num_workers == 0
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                         shuffle=False, num_workers=self.num_workers, pin_memory=pin_mem)
    
    def test_dataloader(self):
        pin_mem = self.num_workers == 0
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                         shuffle=False, num_workers=self.num_workers, pin_memory=pin_mem)


class TokenDataset(Dataset):
    """Dataset for token sequences - converts to one-hot encoding.
    
    Handles memmap files properly for multiprocessing DataLoader workers.
    """
    
    def __init__(self, tokens, masks, vocab_size):
        # Store paths if memmap, or arrays if already loaded
        if isinstance(tokens, np.memmap) or isinstance(tokens, str):
            self.tokens_path = tokens if isinstance(tokens, str) else tokens.filename
            self.masks_path = masks if isinstance(masks, str) else masks.filename
            self._tokens_memmap = None
            self._masks_memmap = None
        else:
            self.tokens_path = None
            self.masks_path = None
            self._tokens_memmap = tokens
            self._masks_memmap = masks
        self.vocab_size = vocab_size
        self._len = len(tokens) if not isinstance(tokens, (str, np.memmap)) else None
    
    def __len__(self):
        if self._len is not None:
            return self._len
        # Lazy load to get length
        if self._tokens_memmap is None and self.tokens_path:
            self._tokens_memmap = np.load(self.tokens_path, mmap_mode='r')
        return len(self._tokens_memmap)
    
    def __getitem__(self, idx):
        # Lazy load memmap in worker process (each worker gets its own memmap)
        if self._tokens_memmap is None:
            if self.tokens_path:
                self._tokens_memmap = np.load(self.tokens_path, mmap_mode='r')
                self._masks_memmap = np.load(self.masks_path, mmap_mode='r')
            else:
                raise RuntimeError("Tokens and masks not properly initialized")
        
        tokens_idx = torch.from_numpy(np.array(self._tokens_memmap[idx])).long()
        masks = torch.from_numpy(np.array(self._masks_memmap[idx])).bool()
        
        # Convert token indices to one-hot encoding (B, T, V)
        tokens_onehot = torch.nn.functional.one_hot(tokens_idx, num_classes=self.vocab_size).float()
        
        # Return same format as grammar: (input, target, mask)
        return tokens_onehot, tokens_idx, masks
