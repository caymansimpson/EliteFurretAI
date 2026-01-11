# -*- coding: utf-8 -*-
"""
Unit tests for compression utilities.

These tests verify:
1. save_compressed correctly compresses and saves PyTorch objects
2. load_compressed correctly decompresses and loads PyTorch objects
3. Roundtrip preservation of tensor data and dtypes
4. Handling of different object types (dicts, tensors, nested structures)
"""

import os
import tempfile

import pytest
import torch

from elitefurretai.etl.compress_utils import load_compressed, save_compressed

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_dir():
    """
    Create a temporary directory for test files.

    Automatically cleaned up after test completes.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_tensor():
    """
    Create a sample tensor for testing.

    Uses random values to ensure compression is actually tested.
    """
    return torch.randn(100, 50)


@pytest.fixture
def sample_dict():
    """
    Create a sample dictionary of tensors.

    This mimics the structure of actual training data:
    - states: float32 features
    - actions: int64 action indices
    - masks: bool action masks
    """
    return {
        "states": torch.randn(32, 10, 100, dtype=torch.float32),
        "actions": torch.randint(0, 2025, (32, 10), dtype=torch.long),
        "masks": torch.ones(32, 10, 2025, dtype=torch.bool),
        "rewards": torch.randn(32, 10, dtype=torch.float32),
    }


# =============================================================================
# BASIC ROUNDTRIP TESTS
# =============================================================================


def test_save_load_tensor_roundtrip(temp_dir, sample_tensor):
    """
    Test that a tensor survives compression roundtrip.

    Save a tensor with save_compressed, load with load_compressed,
    verify the loaded tensor equals the original.

    Expected: Loaded tensor should be identical to original.
    """
    filepath = os.path.join(temp_dir, "tensor.pt.zst")

    save_compressed(sample_tensor, filepath)
    loaded = load_compressed(filepath)

    assert torch.allclose(loaded, sample_tensor)
    assert loaded.dtype == sample_tensor.dtype
    assert loaded.shape == sample_tensor.shape


def test_save_load_dict_roundtrip(temp_dir, sample_dict):
    """
    Test that a dictionary of tensors survives compression roundtrip.

    Training data is stored as dicts with keys like 'states', 'actions', etc.

    Expected: All tensors in dict should be identical after roundtrip.
    """
    filepath = os.path.join(temp_dir, "batch.pt.zst")

    save_compressed(sample_dict, filepath)
    loaded = load_compressed(filepath)

    # Verify all keys present
    assert set(loaded.keys()) == set(sample_dict.keys())

    # Verify each tensor
    for key in sample_dict:
        assert torch.allclose(loaded[key], sample_dict[key]), f"Mismatch in {key}"
        assert loaded[key].dtype == sample_dict[key].dtype, f"Dtype mismatch in {key}"


def test_save_load_preserves_dtypes(temp_dir):
    """
    Test that different tensor dtypes are preserved.

    PyTorch tensors can have various dtypes (float32, float16, int64, bool).
    All should be preserved through compression.

    Expected: Loaded tensors have same dtype as originals.
    """
    data = {
        "float32": torch.randn(10, dtype=torch.float32),
        "float64": torch.randn(10, dtype=torch.float64),
        "int32": torch.randint(0, 100, (10,), dtype=torch.int32),
        "int64": torch.randint(0, 100, (10,), dtype=torch.int64),
        "bool": torch.ones(10, dtype=torch.bool),
    }

    filepath = os.path.join(temp_dir, "dtypes.pt.zst")
    save_compressed(data, filepath)
    loaded = load_compressed(filepath)

    for key in data:
        assert loaded[key].dtype == data[key].dtype, f"Dtype mismatch for {key}"


# =============================================================================
# COMPRESSION TESTS
# =============================================================================


def test_compressed_file_smaller_than_uncompressed(temp_dir, sample_dict):
    """
    Test that compressed file is smaller than uncompressed.

    Zstd compression should significantly reduce file size for tensor data.

    Expected: Compressed size < uncompressed size.
    """
    compressed_path = os.path.join(temp_dir, "batch.pt.zst")
    uncompressed_path = os.path.join(temp_dir, "batch.pt")

    # Save compressed
    save_compressed(sample_dict, compressed_path)

    # Save uncompressed for comparison
    torch.save(sample_dict, uncompressed_path)

    compressed_size = os.path.getsize(compressed_path)
    uncompressed_size = os.path.getsize(uncompressed_path)

    assert compressed_size < uncompressed_size, (
        f"Compressed ({compressed_size}) should be smaller than uncompressed ({uncompressed_size})"
    )


def test_compression_level_affects_size(temp_dir, sample_dict):
    """
    Test that higher compression levels produce smaller files.

    Zstd supports levels 1-22. Higher levels = slower but smaller.
    Level 3 is default (balance of speed/size).

    Expected: Level 10 file should be smaller than level 1 file.
    """
    path_level1 = os.path.join(temp_dir, "level1.pt.zst")
    path_level10 = os.path.join(temp_dir, "level10.pt.zst")

    save_compressed(sample_dict, path_level1, level=1)
    save_compressed(sample_dict, path_level10, level=10)

    size_level1 = os.path.getsize(path_level1)
    size_level10 = os.path.getsize(path_level10)

    # Higher compression should generally produce smaller files
    # (may not always be true for small data, but usually is)
    assert size_level10 <= size_level1, (
        f"Level 10 ({size_level10}) should be <= level 1 ({size_level1})"
    )


# =============================================================================
# MAP_LOCATION TESTS
# =============================================================================


def test_load_to_cpu(temp_dir, sample_tensor):
    """
    Test loading tensors to CPU explicitly.

    map_location='cpu' ensures tensors load to CPU even if saved from GPU.

    Expected: Loaded tensor should be on CPU device.
    """
    filepath = os.path.join(temp_dir, "tensor.pt.zst")
    save_compressed(sample_tensor, filepath)

    loaded = load_compressed(filepath, map_location="cpu")

    assert loaded.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_load_to_cuda(temp_dir, sample_tensor):
    """
    Test loading tensors directly to CUDA.

    map_location='cuda' loads tensors directly to GPU.

    Expected: Loaded tensor should be on CUDA device.
    """
    filepath = os.path.join(temp_dir, "tensor.pt.zst")
    save_compressed(sample_tensor, filepath)

    loaded = load_compressed(filepath, map_location="cuda")

    assert loaded.device.type == "cuda"


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


def test_empty_tensor(temp_dir):
    """
    Test handling of empty tensors.

    Empty tensors (shape with 0 dimension) should survive roundtrip.

    Expected: Empty tensor preserved through compression.
    """
    empty_tensor = torch.tensor([])
    filepath = os.path.join(temp_dir, "empty.pt.zst")

    save_compressed(empty_tensor, filepath)
    loaded = load_compressed(filepath)

    assert loaded.shape == empty_tensor.shape
    assert len(loaded) == 0


def test_nested_dict(temp_dir):
    """
    Test handling of nested dictionaries.

    Complex nested structures should survive compression.

    Expected: Nested structure preserved.
    """
    nested = {
        "level1": {"level2": {"tensor": torch.randn(10)}},
        "list_of_tensors": [torch.randn(5), torch.randn(3)],
    }
    filepath = os.path.join(temp_dir, "nested.pt.zst")

    save_compressed(nested, filepath)
    loaded = load_compressed(filepath)

    assert torch.allclose(
        loaded["level1"]["level2"]["tensor"],
        nested["level1"]["level2"]["tensor"],  # type: ignore[index]
    )
    assert len(loaded["list_of_tensors"]) == 2  # type: ignore[arg-type]


def test_large_tensor(temp_dir):
    """
    Test handling of large tensors.

    Training batches can be quite large (e.g., 512 trajectories × 17 steps × 15000 features).
    This tests a moderately large tensor.

    Expected: Large tensor preserved through compression.
    """
    # ~60MB uncompressed (1000 * 1000 * 15 * 4 bytes)
    large_tensor = torch.randn(1000, 1000, 15)
    filepath = os.path.join(temp_dir, "large.pt.zst")

    save_compressed(large_tensor, filepath)
    loaded = load_compressed(filepath)

    assert loaded.shape == large_tensor.shape
    # For large tensors, use allclose with tolerance
    assert torch.allclose(loaded, large_tensor)


def test_special_float_values(temp_dir):
    """
    Test handling of special float values (inf, nan).

    Models can produce inf/nan during training. These should be preserved.

    Expected: Special values preserved through compression.
    """
    special = torch.tensor([float("inf"), float("-inf"), float("nan"), 0.0, 1.0])
    filepath = os.path.join(temp_dir, "special.pt.zst")

    save_compressed(special, filepath)
    loaded = load_compressed(filepath)

    # Check inf values
    assert torch.isinf(loaded[0]) and loaded[0] > 0
    assert torch.isinf(loaded[1]) and loaded[1] < 0

    # Check nan (nan != nan, so use isnan)
    assert torch.isnan(loaded[2])

    # Check normal values
    assert loaded[3] == 0.0
    assert loaded[4] == 1.0


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


def test_load_nonexistent_file():
    """
    Test that loading nonexistent file raises appropriate error.

    Expected: FileNotFoundError when file doesn't exist.
    """
    with pytest.raises(FileNotFoundError):
        load_compressed("/nonexistent/path/file.pt.zst")


def test_load_corrupted_file(temp_dir):
    """
    Test that loading corrupted file raises appropriate error.

    If file is corrupted (not valid zstd), should raise error.

    Expected: Error when decompressing corrupted data.
    """
    filepath = os.path.join(temp_dir, "corrupted.pt.zst")

    # Write invalid data
    with open(filepath, "wb") as f:
        f.write(b"this is not valid zstd compressed data")

    with pytest.raises(Exception):  # zstd.ZstdError or similar
        load_compressed(filepath)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
