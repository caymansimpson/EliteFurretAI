import pytest
import torch

from elitefurretai.etl.cuda_prefetcher import CudaStreamPrefetcher


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_prefetcher_yields_same_batches_in_order():
    batches = [
        {"x": torch.arange(4, dtype=torch.float32).view(2, 2) + i} for i in range(5)
    ]
    prefetcher = CudaStreamPrefetcher(iter(batches), device="cuda")
    out = list(prefetcher)
    assert len(out) == 5
    for i, batch in enumerate(out):
        assert batch["x"].device.type == "cuda"
        expected = torch.arange(4, dtype=torch.float32).view(2, 2) + i
        assert torch.allclose(batch["x"].cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_prefetcher_handles_empty_iterable():
    prefetcher = CudaStreamPrefetcher(iter([]), device="cuda")
    assert list(prefetcher) == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_prefetcher_preserves_keys():
    batch = {"a": torch.zeros(2), "b": torch.ones(3), "c": torch.full((1,), 7.0)}
    prefetcher = CudaStreamPrefetcher(iter([batch]), device="cuda")
    out = next(iter(prefetcher))
    assert set(out.keys()) == {"a", "b", "c"}
