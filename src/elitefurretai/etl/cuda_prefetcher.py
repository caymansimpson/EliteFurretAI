"""
Overlap host-to-device tensor transfer with GPU compute.

The main-thread .cuda() call in supervised training is synchronous (we
cannot use pin_memory=True on WSL2). This wrapper issues each batch's
H2D copy on a side CUDA stream so the compute stream can run the
previous batch's forward/backward concurrently.

Usage:
    for batch in CudaStreamPrefetcher(dataloader, device="cuda"):
        # batch tensors are already on `device`; H2D for batch N+1 is
        # already in flight on the side stream
        ...
"""

from typing import Dict, Iterable, Iterator, Optional

import torch


class CudaStreamPrefetcher:
    def __init__(
        self,
        loader: Iterable[Dict[str, torch.Tensor]],
        device: str = "cuda",
    ):
        self._loader = loader
        self._device = torch.device(device)
        self._stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=self._device) if self._device.type == "cuda" else None
        )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        it = iter(self._loader)

        if self._stream is None:
            # Non-CUDA fallback: just move tensors to the device.
            for batch in it:
                yield {k: v.to(self._device) for k, v in batch.items()}
            return

        next_batch = self._prefetch(it)
        while next_batch is not None:
            # Make the compute stream wait until the H2D finishes for
            # the batch we are about to hand out.
            torch.cuda.current_stream(self._device).wait_stream(self._stream)
            current = next_batch
            next_batch = self._prefetch(it)
            yield current

    def _prefetch(
        self, it: Iterator[Dict[str, torch.Tensor]]
    ) -> Optional[Dict[str, torch.Tensor]]:
        try:
            host = next(it)
        except StopIteration:
            return None
        assert self._stream is not None
        with torch.cuda.stream(self._stream):
            return {k: v.to(self._device, non_blocking=True) for k, v in host.items()}
