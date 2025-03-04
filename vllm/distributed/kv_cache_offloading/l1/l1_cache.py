import logging
import torch

from typing import Iterable, Iterator, Tuple
from .eviction_policy import BaseEvictionPolicy, Functor
from ..common import nvtx_range
from ..common.absl_logging import log_every_n_seconds, getLogger
from ..memory import MemoryRegion, TensorPoolAllocator
from ..spec import KVCacheBlockLayout, KVCacheBlockSpec
from ..status import Status, StatusCodes
from ..utils import split_list, perf_timer

logger = getLogger(__name__)


class L1Cache:

    def __init__(
        self,
        eviction_policy: str,
        capacity: int,
        allocator: TensorPoolAllocator,
        block_spec: KVCacheBlockSpec,
        evict_size: int = 1,
        on_put: Functor | None = None,
        on_evict: Functor | None = None,
        on_hot_access: Functor | None = None,
    ) -> None:
        """Create a cache object.
        Args:
            eviction_policy (str): The name of the eviction policy.
            capacity (int): The capacity of the cache in terms of number of blocks.
            allocator (TensorPoolAllocator): The allocator to allocate cache block.
            evict_size (int, optional): The number of items to evict at a time. Defaults to 1.
            on_put(Functor, optional): The callback function to call when putting new items.
                                       Defaults to None.
            on_evict(Functor, optional): The evict function to call when evicting items. Defaults to None.
            on_hot_access(Functor, optional): The callback function to call when a cache item becomes hot.
                                              Defaults to None.
        """

        self.capacity: int = capacity
        self.allocator: TensorPoolAllocator = allocator
        self.block_spec: KVCacheBlockSpec = block_spec
        self.block_shape: Tuple[int, ...] = self.block_spec.block_shape
        self.block_dtype: torch.dtype = self.block_spec.block_dtype
        self.block_ntokens: int = self.block_spec.block_ntokens
        self.block_nbytes: int = self.block_spec.block_nbytes
        self.block_shape_token_dim: int = self.block_spec.block_shape_token_dim

        self._eviction_policy: BaseEvictionPolicy = BaseEvictionPolicy.create(
            eviction_policy,
            capacity,
            evict_size,
            on_put=on_put,
            on_evict=on_evict,
            on_hot_access=on_hot_access,
        )

        assert (
            self.allocator.capacity_nbytes == self.capacity * self.block_nbytes
        ), f"Allocator capacity {self.allocator.capacity_nbytes} is not equal to cache capacity {self.capacity * self.block_nbytes}."

        if not self.block_spec.is_homogeneous():
            raise NotImplementedError(
                "Heterogeneous block is not supported yet.")

        logger.info(f"{str(self)} is initialized.")

    def __len__(self) -> int:
        """Return the number of cache blocks in the cache."""
        return len(self.allocator) // self.block_nbytes

    def __repr__(self) -> str:
        return f"L1Cache(policy={self._eviction_policy.name}, capacity={self.capacity}, size={len(self)})"

    def __str__(self) -> str:
        return self.__repr__()

    def set_on_put_callback(self, functor: Functor) -> None:
        """Set the callback function to call when putting new items."""
        self._eviction_policy.set_on_put_callback(functor)

    def set_on_evict_callback(self, on_evict: Functor) -> None:
        """Set the callback function to call when evicting items."""
        self._eviction_policy.set_on_evict_callback(on_evict)

    def set_on_hot_access_callback(self, on_hot_access: Functor) -> None:
        """Set the callback function to call when a cache item becomes hot."""
        self._eviction_policy.set_on_hot_access_callback(on_hot_access)

    def allocate(
        self,
        num_blocks: int,
    ) -> Status[Iterable[MemoryRegion]]:
        """Allocate a set of memory regions that have the capapcity to hold `nblocks`.
        
        Args:
            num_blocks: The number of blocks to allocate.
        Returns:
            The memory regions.
        """
        if self.capacity - len(self) < num_blocks:
            self._eviction_policy.evict(num_blocks)

        num_blocks = min(num_blocks, self.capacity - len(self))

        if num_blocks == 0:
            return Status(StatusCodes.OUT_OF_MEMORY)

        status = self.allocator.alloc(self.block_nbytes * num_blocks)
        if status.is_ok():
            return Status(
                value=MemoryRegion.split(status.value, self.block_nbytes))
        else:
            # failed to allocate one MR for all the blocks due to fragmentation,
            # try to allocate one MR for each block
            block_mrs = []
            for _ in range(num_blocks):
                status = self.allocator.alloc(self.block_nbytes)
                if status.is_ok():
                    block_mrs.append(status.value)
                else:
                    break
            return Status(value=block_mrs)

    @nvtx_range("put", "kv_cache_ol.L1Cache")
    def put(
        self,
        prefix: Iterable[int] | None,
        tokens: Iterable[int],
        kv_tensors: torch.Tensor | Iterable[MemoryRegion],
    ) -> Status[int]:
        """Put kv tensors to the cache.
        Args:
            prefix (Iterable[int] | None): The prefix tokens of the kv tensors.
            tokens (Iterable[int]): The tokens of the kv tensors.
            kv_tensors (torch.Tensor | Iterable[MemoryRegion]): The kv tensors.
        Returns:
            The status of the put operation and the number of tokens.
        """
        if isinstance(kv_tensors, torch.Tensor):
            return self._put_tensors_impl(prefix, tokens, kv_tensors)
        else:
            return self._put_mrs_impl(prefix, tokens, kv_tensors)

    def _put_tensors_impl(
        self,
        prefix: Iterable[int] | None,
        tokens: Iterable[int],
        kv_tensors: torch.Tensor,
    ) -> Status[int]:
        """Put kv tensors to the cache.
        Args:
            prefix (Iterable[int] | None): The prefix tokens of the kv tensors.
            tokens (Iterable[int]): The tokens of the kv tensors.
            kv_tensors (torch.Tensor): The kv tensors.
        Returns:
            The status of the put operation and the number of tokens.
        """
        if prefix is not None and len(prefix) % self.block_ntokens != 0:
            return Status(
                StatusCodes.INVALID,
                f"Prefix tokens {prefix} is not aligned to block size {self.block_ntokens}."
            )

        if len(tokens) != kv_tensors.shape[self.block_shape_token_dim]:
            return Status(
                StatusCodes.INVALID,
                f"Number of tokens {len(tokens)} is not equal to the number of tokens "
                f"in key tensors {kv_tensors.shape[self.block_shape_token_dim]}."
            )

        # If it is not a full block, we don't need to cache it.
        if len(tokens) // self.block_ntokens == 0:
            return Status(StatusCodes.OK, 0)

        num_tokens = len(tokens)
        num_blocks = num_tokens // self.block_ntokens

        status = self.allocate(num_blocks)
        if not status.is_ok():
            return status

        block_mrs = status.value
        offset = 0
        block_mr_shape = [s for s in self.block_shape]
        block_mr_shape[self.block_shape_token_dim] = self.block_ntokens
        slices = [slice(None)] * len(self.block_shape)
        with perf_timer() as get_copy_dur_ms:
            for block_mr in block_mrs:
                cached_tensors = (block_mr.to_tensor(self.block_dtype,
                                                     block_mr_shape))

                slices[self.block_shape_token_dim] = slice(
                    offset, offset + self.block_ntokens)
                cached_tensors.copy_(kv_tensors[tuple(slices)])
                offset += self.block_ntokens
        log_every_n_seconds(
            logger,
            logging.INFO,
            f"Copying kv tensors takes {get_copy_dur_ms():.4f} ms",
            n_seconds=10,
        )

        return self._put_mrs_impl(prefix, tokens, block_mrs, with_check=False)

    def _put_mrs_impl(
        self,
        prefix: Iterable[int] | None,
        tokens: Iterable[int],
        kv_mrs: Iterable[MemoryRegion],
        with_check: bool = True,
    ) -> Status[int]:
        """Put kv mrs to the cache.
        Args:
            prefix (Iterable[int] | None): The prefix tokens of the kv mrs.
            tokens (Iterable[int]): The tokens of the kv mrs.
            kv_mrs (Iterable[MemoryRegion]): The kv memory regions.
            with_check (bool): Whether to check the validity of the kv mrs.
        Returns:
            The status of the put operation and the number of tokens.
        """
        num_tokens = len(tokens)
        num_blocks = sum(mr.length for mr in kv_mrs) // self.block_nbytes

        if with_check:
            if prefix is not None and len(prefix) % self.block_ntokens != 0:
                [mr.ref_down() for mr in kv_mrs]
                return Status(
                    StatusCodes.INVALID,
                    f"Prefix tokens {prefix} is not aligned to block size {self.block_ntokens}."
                )

            if num_tokens != num_blocks * self.block_ntokens:
                [mr.ref_down() for mr in kv_mrs]
                return Status(
                    StatusCodes.INVALID,
                    f"Number of tokens {num_tokens} is not equal to the number of tokens "
                    f"in key tensors {num_blocks * self.block_ntokens}.")

            # If it is not a full block, we don't need to cache it.
            if num_tokens // self.block_ntokens == 0:
                [mr.ref_down() for mr in kv_mrs]
                return Status(StatusCodes.OK, 0)

            for mr in kv_mrs:
                assert mr.ref_count == 1

        assert len(kv_mrs) == num_blocks

        bi = 0
        for cache_key in self._cache_block_keys(
                prefix, tokens[:num_blocks * self.block_ntokens]):
            if bi >= len(kv_mrs):
                break
            block_mr = kv_mrs[bi]
            if not self._eviction_policy.put(cache_key, block_mr).is_ok():
                break
            bi += 1

        [mr.ref_down() for mr in kv_mrs[bi:]]
        return Status(StatusCodes.OK, bi * self.block_ntokens)

    @nvtx_range("get", "kv_cache_ol.L1Cache")
    def get(
        self,
        prefix: Iterable[int] | None,
        tokens: Iterable[int],
    ) -> Status[Iterable[torch.Tensor]]:
        """Get kv tensors from the cache.
        Args:
            prefix (Iterable[int] | None): The prefix tokens of the kv tensors.
            tokens (Iterable[int]): The tokens of the kv tensors.
        Returns:
            The kv tensors corresponding to the tokens.
        """
        return self._get_impl("get", prefix, tokens)

    @nvtx_range("peak", "kv_cache_ol.L1Cache")
    def peak(
        self,
        prefix: Iterable[int] | None,
        tokens: Iterable[int],
    ) -> Status[Iterable[torch.Tensor]]:
        """Peak the kv tensors from the cache. Peak does not update the eviction policy.
        Args:
            prefix (Iterable[int] | None): The prefix tokens of the kv tensors.
            tokens (Iterable[int]): The tokens of the kv tensors.
        Returns:
            The kv tensors corresponding to the tokens.
        """
        return self._get_impl("peak", prefix, tokens)

    @nvtx_range("acquire", "kv_cache_ol.L1Cache")
    def acquire(
        self,
        prefix: Iterable[int] | None,
        tokens: Iterable[int],
    ) -> Status[Iterable[MemoryRegion]]:
        """Acquire cache handle pointing to the kv tensors such that the upper layer
        can access these tensors in a zero-copy way.
        Args:
            prefix (Iterable[int] | None): The prefix tokens of the kv tensors.
            tokens (Iterable[int]): The tokens of the kv tensors.
        Returns:
            The memory regions corresponding to the tokens.
        """
        return self._get_impl("get", prefix, tokens, zero_copy=True)

    def _get_impl(
        self,
        name: str,
        prefix: Iterable[int] | None,
        tokens: Iterable[int],
        zero_copy: bool = False,
    ) -> Status[Iterable[torch.Tensor | MemoryRegion]]:
        """Get/peak the kv tensors from the cache. Peak does not update the eviction policy.
        Args:
            name (str): get or peak.
            prefix (Iterable[int] | None): The prefix tokens of the kv tensors.
            tokens (Iterable[int]): The tokens of the kv tensors.
            zero_copy (bool): whether to return cache handle to support zero-copy.
        Returns:
            The kv tensors / memory regions corresponding to the tokens.
        """
        if prefix is not None and len(prefix) % self.block_ntokens != 0:
            return Status(StatusCodes.INVALID)

        mrs = []
        for key in self._cache_block_keys(prefix, tokens):
            status = getattr(self._eviction_policy, name)(key)
            if status.is_ok():
                mrs.append(status.value)
            else:
                break

        if len(mrs) == 0:
            return Status(StatusCodes.NOT_FOUND)

        if zero_copy:
            return Status(value=mrs)

        tensors = []
        with perf_timer() as get_tensor_clone_dur_ms:
            block_mr_shape = [s for s in self.block_shape]
            block_mr_shape[self.block_shape_token_dim] = self.block_ntokens
            for mr in mrs:
                tensor = (mr.to_tensor(self.block_dtype,
                                       block_mr_shape).clone())
                tensors.append(tensor)
        log_every_n_seconds(
            logger,
            logging.INFO,
            f"Cloning tensor takes {get_tensor_clone_dur_ms():.4f} ms",
            n_seconds=10,
        )

        [mr.ref_down() for mr in mrs]
        return Status(value=tensors)

    @nvtx_range("delete", "kv_cache_ol.L1Cache")
    def delete(self, prefix: Iterable[int] | None,
               tokens: Iterable[int]) -> Status:
        """Delete kv tensors from the cache.
        Args:
            prefix (Iterable[int] | None): The prefix tokens of the kv tensors.
            tokens (Iterable[int]): The tokens of the kv tensors.
        """
        if prefix is not None and len(prefix) % self.block_ntokens != 0:
            return Status(StatusCodes.INVALID)

        for key in self._cache_block_keys(prefix, tokens):
            self._eviction_policy.delete(key)
        return Status(StatusCodes.OK)

    def _cache_block_keys(
        self, prefix: Iterable[int] | None, tokens: Iterable[int]
    ) -> Iterator[Tuple[Iterable[int], Iterable[int]]]:
        """Get the cache block keys of the kv tensors.
        Args:
            prefix (Iterable[int] | None): The prefix tokens of the kv tensors.
            tokens (Iterable[int]): The tokens of the kv tensors.
        Returns:
            The cache block keys of the kv tensors.
        """
        cache_key = [] if prefix is None else [x for x in prefix]
        block_tokens = split_list(tokens, self.block_ntokens)
        num_tokens = len(tokens)
        if num_tokens % self.block_ntokens != 0:
            block_tokens = block_tokens[:-1]

        for i in range(len(block_tokens)):
            yield (tuple(cache_key), tuple(block_tokens[i]))
            cache_key += block_tokens[i]
