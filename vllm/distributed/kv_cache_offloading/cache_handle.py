import torch

from abc import ABC, abstractmethod
from typing import Iterable, Tuple

from .memory import MemoryRegion


class KVCacheHandle(ABC):
    """Cache handle to support zero-copy APIs.
    """

    def __init__(self, block_dtype: torch.dtype,
                 block_shape: Tuple[int, ...]) -> None:
        self._block_dtype = block_dtype
        self._block_shape = block_shape

    @abstractmethod
    def to_tensors(self) -> Iterable[torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def release(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError


class BaseKVCacheHandle(KVCacheHandle):

    def __init__(self, block_dtype: torch.dtype, block_shape: Tuple[int, ...],
                 mrs: Iterable[MemoryRegion]) -> None:
        super().__init__(block_dtype, block_shape)
        self._mrs = mrs

    def to_tensors(self) -> Iterable[torch.Tensor]:
        return MemoryRegion.to_tensors(self._mrs, self._block_dtype,
                                       self._block_shape)

    def release(self) -> None:
        for mr in self._mrs:
            mr.ref_down()

    def __len__(self) -> int:
        return len(self._mrs)
