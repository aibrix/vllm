# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Iterable, Tuple

import torch

from .common import ObjectPool
from .memory import MemoryRegion


class KVCacheHandle(ABC):
    """Cache handle to support zero-copy APIs."""

    @abstractmethod
    def to_tensors(self) -> Iterable[torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def release(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError


class MemoryRegionKVCacheHandle(KVCacheHandle):

    def __init__(
        self,
        block_dtype: torch.dtype,
        block_shape: Tuple[int, ...],
        mrs: Iterable[MemoryRegion],
    ) -> None:
        self._block_dtype = block_dtype
        self._block_shape = block_shape
        self._mrs = mrs

    def to_tensors(self) -> Iterable[torch.Tensor]:
        return MemoryRegion.to_tensors(self._mrs, self._block_dtype,
                                       self._block_shape)

    def release(self) -> None:
        for mr in self._mrs:
            mr.ref_down()

    def __len__(self) -> int:
        return len(self._mrs)


class ObjectPoolKVCacheHandle(KVCacheHandle):

    def __init__(
        self,
        tensors: Iterable[torch.Tensor],
        obj_pool: ObjectPool,
    ) -> None:
        self._tensors = tensors
        self._obj_pool = obj_pool

    def to_tensors(self) -> Iterable[torch.Tensor]:
        return self._tensors

    def release(self) -> None:
        self._obj_pool.put(self._tensors)

    def __len__(self) -> int:
        return len(self._tensors)
