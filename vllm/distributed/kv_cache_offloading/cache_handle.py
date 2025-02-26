import torch

from abc import ABC, abstractmethod
from typing import Iterable

from .memory import MemoryRegion


class KVCacheHandle(ABC):
    """Cache handle to support zero-copy APIs.
    """

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

    def __init__(self, mrs: Iterable[MemoryRegion]) -> None:
        self._mrs = mrs

    def to_tensors(self) -> Iterable[torch.Tensor]:
        return MemoryRegion.to_tensors(self._mrs)

    def release(self) -> None:
        for mr in self._mrs:
            mr.ref_down()

    def __len__(self) -> int:
        return len(self._mrs)
