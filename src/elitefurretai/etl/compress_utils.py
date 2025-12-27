"""
Compression utilities for saving and loading PyTorch tensors with zstd compression.

These utilities are used throughout the ETL pipeline for efficient storage of
preprocessed training data.
"""

import io

import torch
import zstd


def save_compressed(obj, out_path, level=3):
    """
    Save a PyTorch object with zstd compression.

    Args:
        obj: Object to save (typically dict of tensors)
        out_path: Path to save to (typically .pt.zst extension)
        level: Compression level (1-22, default 3 for speed/size balance)
    """
    buf = io.BytesIO()
    torch.save(obj, buf)
    compressed = zstd.compress(buf.getvalue(), level)
    with open(out_path, "wb") as f:
        f.write(compressed)


def load_compressed(in_path, map_location="cpu"):
    """
    Load a zstd-compressed PyTorch object.

    Args:
        in_path: Path to compressed file
        map_location: Device to load tensors to (default "cpu")

    Returns:
        Decompressed PyTorch object
    """
    with open(in_path, "rb") as f:
        compressed = f.read()
    raw = zstd.decompress(compressed)
    return torch.load(io.BytesIO(raw), map_location=map_location, weights_only=False)
