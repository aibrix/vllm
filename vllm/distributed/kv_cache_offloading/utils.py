# SPDX-License-Identifier: Apache-2.0
import os
import time
from contextlib import contextmanager

import torch


def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Convert a PyTorch tensor (CPU/GPU) to raw bytes."""
    if tensor.is_cuda:
        tensor = tensor.cpu()  # Move to CPU if on GPU
    return tensor.view(torch.uint8).numpy().tobytes()


def bytes_to_tensor(data: bytes) -> torch.Tensor:
    """Convert raw bytes to a PyTorch tensor."""
    return torch.frombuffer(data, dtype=torch.uint8)


def in_place_pin_memory(x: torch.Tensor) -> torch.Tensor:
    """Pin a tensor in-place."""
    if x.is_pinned():
        return x
    cudart = torch.cuda.cudart()
    flags = 2  # cudaHostRegisterMapped
    if cudart.cudaHostRegister(x.data_ptr(),
                               x.numel() * x.element_size(), flags) == 0:
        return x
    else:
        # could not register, fallback to torch pin_memory
        return x.pin_memory()


@contextmanager
def cpu_perf_timer(enabled: bool = True):
    if not enabled:
        yield lambda: 0
    else:
        start = time.perf_counter()
        end = start
        yield lambda: (end - start) * 1000
        end = time.perf_counter()


if torch.cuda.is_available():

    @contextmanager
    def perf_timer():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        yield lambda: start.elapsed_time(end)
        end.record()

        end.synchronize()
else:
    perf_timer = cpu_perf_timer


def ensure_dir_exist(path: str) -> None:
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
