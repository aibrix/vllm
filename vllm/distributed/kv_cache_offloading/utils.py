import os
import time
import torch

from contextlib import contextmanager
from typing import Any, List, Tuple
from .spec import KVCacheBlockLayout


def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Convert a PyTorch tensor (CPU/GPU) to raw bytes."""
    if tensor.is_cuda:
        tensor = tensor.cpu()  # Move to CPU if on GPU
    return tensor.numpy().tobytes()


def get_block_num_layers(shape: Tuple[int, ...],
                         layout: KVCacheBlockLayout) -> int:
    if layout == KVCacheBlockLayout.NLD:
        return shape[1]
    elif layout == KVCacheBlockLayout.LND:
        return shape[0]


def get_block_num_tokens(shape: Tuple[int, ...],
                         layout: KVCacheBlockLayout) -> int:
    if layout == KVCacheBlockLayout.NLD:
        return shape[0]
    elif layout == KVCacheBlockLayout.LND:
        return shape[1]


def split_list(lst: List[Any], step: int):
    return [lst[i:i + step] for i in range(0, len(lst), step)]


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

    @contextmanager
    def perf_timer():
        start = time.perf_counter()
        end = start
        yield lambda: (end - start) * 1000
        end = time.perf_counter()


def ensure_dir_exist(path: str) -> None:
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
