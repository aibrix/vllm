import asyncio
import logging
import threading
import torch

import torch.distributed as dist

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Hashable, Iterable, Iterator, Tuple

from . import envs
from .cache_handle import KVCacheHandle, BaseKVCacheHandle
from .common import nvtx_range
from .common.absl_logging import log_every_n_seconds, getLogger
from .config import KVCacheConfig
from .l1 import L1Cache
from .l2 import (
    L2Cache,
    MD5Hasher,
    RollingHashKeyBuilder,
    StringSerializer,
    TensorSerializer,
    ZstdCompressor,
)
from .memory import MemoryRegion, TensorPoolAllocator
from .spec import KVCacheBlockLayout, KVCacheBlockSpec
from .status import Status, StatusCodes
from .utils import split_list, perf_timer

logger = getLogger(__name__)


@dataclass
class KVCacheFeature:
    """The features of the kv cache.
    Args:
        zero_copy: Whether the kv cache supports zero-copy.
        non_blocking_put: Whether the kv cache uses non-blocking put.
    """
    zero_copy: bool = False
    non_blocking_put: bool = False


class KVCacheManager(ABC):
    """The KV cache manager.

    Args:
        config: The KV cache manager configuration.
    """

    def __init__(self, config: KVCacheConfig) -> None:
        self.config: KVCacheConfig = config
        self.block_spec: KVCacheBlockSpec = self.config.block_spec
        self.block_layout: KVCacheBlockLayout = self.config.block_spec.block_layout
        self.block_shape: Tuple[int, ...] = self.config.block_spec.block_shape
        self.block_dtype: torch.dtype = self.config.block_spec.block_dtype
        self.block_ntokens: int = self.config.block_spec.block_ntokens
        self.block_nbytes: int = self.config.block_spec.block_nbytes
        self.block_shape_token_dim: int = self.block_spec.block_shape_token_dim

    @property
    @abstractmethod
    def feature(self) -> KVCacheFeature:
        """Get the feature of the kv cache.
        Returns:
            The feature of the kv cache.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def chunk_size(self) -> int:
        """Get the chunk size of the kv cache.
        Returns:
            The chunk size of the kv cache.
        """
        raise NotImplementedError

    @classmethod
    def prefetch(self, prefix: Iterable[int] | None,
                 tokens: Iterable[int]) -> None:
        """(Optional) Prefetch the kv cache for the given prefix and tokens.
        Args:
            prefix: The prefix of the kv cache. E.g., [1, 2, 3]
            tokens: The tokens of the kv cache. E.g., [4, 5, 6, 7]
        """
        pass

    @classmethod
    def allocate(
        self,
        nblocks: int,
    ) -> Status[KVCacheHandle]:
        """(Optional) Allocate a cache handle that points to buffers owned by the kv
        cache service.
        
        Only the kv cache services supporting zero-copy need to implement this method.

        Args:
            nblocks: The number of blocks to allocate.
        Returns:
            The cache handle.
        """
        raise NotImplementedError

    @classmethod
    def acquire(
        self,
        prefix: Iterable[int] | None,
        tokens: Iterable[int],
    ) -> Status[Tuple[int, Iterable[KVCacheHandle]]]:
        """(Optional) Acquire cache handle of the kv tensors for the given prefix
        and tokens. Only the kv cache services supporting zero-copy need to implement
        this method.
        
        The returned cache handle pointing to buffers owned by the kv cache service.
        We can use "KVCacheHandle.to_tensors()" to get tensors sharing the same storage.
        After the kv tensors are used, we need to explicitly `ref_down()` the cache handle
        to let the kv cache service know that these buffers are not referenced anymore.
        
        Args:
            prefix: The prefix of the kv cache. E.g., [1, 2, 3]
            tokens: The tokens of the kv cache. E.g., [4, 5, 6, 7]
        Returns:
            Number of tokens have been fetched from the kv cache service.
            The cache handles corresponding to the given tokens.
        """
        raise NotImplementedError

    @abstractmethod
    def get(
        self,
        prefix: Iterable[int] | None,
        tokens: Iterable[int],
    ) -> Status[Tuple[int, torch.Tensor]]:
        """Get kv tensors from the kv cache service.

        Args:
            prefix: The prefix of the kv cache. E.g., [1, 2, 3]
            tokens: The tokens of the kv cache. E.g., [4, 5, 6, 7]
        Returns:
            Number of tokens have been fetched from the kv cache service.
            The kv tensors corresponding to the tokens:
                Its layout matches the layout of the kv cache service.

                For example, if the layout is NCLD, then:
                The k, v tensors for i-th token at the j-th layer are kv_tensors[i][0[j]
                and kv_tensors[i][1[j], respectively.

        """
        raise NotImplementedError

    @abstractmethod
    def put(
        self,
        prefix: Iterable[int] | None,
        tokens: Iterable[int],
        kv_tensors: torch.Tensor | KVCacheHandle,
    ) -> Status[int]:
        """Put kv tensors to the kv cache service.

        Args:
            prefix: The prefix of the kv cache. E.g., [1, 2, 3]
            tokens: The tokens of the kv cache. E.g., [4, 5, 6, 7]
            kv_tensors:
                The kv tensors to put into the kv cache.
                
                The layout of kv_tensors must match the layout of the kv cache service.

                For example, if the layout is NCLD, then:
                The k, v tensors for i-th token at the j-th layer are kv_tensors[i][0[j]
                and kv_tensors[i][1[j], respectively.

        Returns:
            The status of the put operation and the number of tokens have been put or
            scheduled to put into the kv cache service.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(
        self,
        prefix: Iterable[int] | None,
        tokens: Iterable[int],
    ) -> Status:
        """Delete kv tensors from the kv cache service.
        Args:
            prefix: The prefix of the kv cache. E.g., [1, 2, 3]
            tokens: The tokens of the kv cache. E.g., [4, 5, 6, 7]
        Returns:
            The status of the delete operation.
        """
        raise NotImplementedError

    def flush(self) -> Status:
        """Flush the kv cache service.
        
        Returns:
            The status of the flush operation.
        """
        return Status(StatusCodes.OK)

    @abstractmethod
    def cache_chunk_keys(
        self, prefix: Iterable[int] | None, tokens: Iterable[int]
    ) -> Iterator[Iterator[Tuple[Iterable[int], Iterable[int],
                                 Iterable[int]]]]:
        """Get the cache chunk keys.
        Args:
            prefix (Iterable[int] | None): The prefix tokens of the kv tensors.
            tokens (Iterable[int]): The tokens of the kv tensors.
        Returns:
            chunk prefix tokens, chunk tokens, next chunk tokens
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Close the kv cache service."""
        raise NotImplementedError


class BaseKVCacheManager(KVCacheManager):
    """Base KV cache manager.

    Args:
        config: The KV cache manager configuration.
    """

    def __init__(self, config: KVCacheConfig) -> None:
        super().__init__(config)

        if not self.block_spec.is_homogeneous():
            raise NotImplementedError(
                "Heterogeneous block is not supported yet.")

        self._l1_cache: L1Cache = None
        self._l2_cache: L2Cache = None
        self._event_loop: asyncio.AbstractEventLoop = None
        self._thread: threading.Thread = None
        self._lock: threading.Lock = None
        self._infight_cv: threading.Condition = None
        self._l2_inflight_writes: int = 0
        self._l2_inflight_quota: int = 0

        self._double_get_threshold: Tuple[
            int, float] = envs.VLLM_KV_CACHE_OL_DOUBLE_GET_THRESHOLD
        self._l2_cache_per_token_timeout_ms: int = envs.VLLM_KV_CACHE_OL_L2_CACHE_PER_TOKEN_TIMEOUT_MS

        self._chunk_size: int = envs.VLLM_KV_CACHE_OL_CHUNK_SIZE

        if self._chunk_size % self.block_ntokens != 0:
            self._chunk_size = self._chunk_size - self._chunk_size % self.block_ntokens
            logger.warning(
                f"VLLM_KV_CACHE_OL_CHUNK_SIZE={envs.VLLM_KV_CACHE_OL_CHUNK_SIZE} "
                f"is not divisible by block_ntokens={self.block_ntokens}, aligned "
                f"to {self._chunk_size}")

        if self._chunk_size < 4 * self.block_ntokens:
            logger.warning(
                f"chunk_size is too small, using {4 * self.block_ntokens} instead"
            )
            self._chunk_size = 4 * self.block_ntokens

        if envs.VLLM_KV_CACHE_OL_L1_CACHE_ENABLED:
            eviction_policy: str = envs.VLLM_KV_CACHE_OL_L1_CACHE_EVICTION_POLICY
            capacity_nbytes: int = int(
                envs.VLLM_KV_CACHE_OL_L1_CACHE_CAPACITY_GB * 1024**3)
            capacity: int = capacity_nbytes // self.block_nbytes
            evict_size: int = envs.VLLM_KV_CACHE_OL_L1_CACHE_EVICT_SIZE

            device: str = envs.VLLM_KV_CACHE_OL_L1_CACHE_DEVICE
            pin_memory: bool = envs.VLLM_KV_CACHE_OL_L1_CACHE_PIN_MEMORY
            allocator: TensorPoolAllocator = TensorPoolAllocator(
                capacity * self.block_nbytes,
                self.block_nbytes,
                device=device,
                pin_memory=pin_memory,
            )

            self._l1_cache = L1Cache(eviction_policy, capacity, allocator,
                                     self.block_spec, evict_size)

        if len(envs.VLLM_KV_CACHE_OL_L2_CACHE_BACKEND) > 0:
            backend_name: str = envs.VLLM_KV_CACHE_OL_L2_CACHE_BACKEND
            namespace: str = envs.VLLM_KV_CACHE_OL_L2_CACHE_NAMESPACE
            compression: str = envs.VLLM_KV_CACHE_OL_L2_CACHE_COMPRESSION
            ingestion_type: str = envs.VLLM_KV_CACHE_OL_L2_CACHE_INGESTION_TYPE
            op_batch: int = envs.VLLM_KV_CACHE_OL_L2_CACHE_OP_BATCH
            self._l2_inflight_quota = (
                envs.VLLM_KV_CACHE_OL_L2_CACHE_INGESTION_MAX_INFLIGHT_TOKENS //
                self.block_ntokens)

            key_builder = RollingHashKeyBuilder(MD5Hasher(),
                                                self.block_ntokens)
            key_serializer = StringSerializer()
            tensor_serializer = TensorSerializer()
            if len(compression) > 0:
                if compression == "ZSTD":
                    tensor_serializer = ZstdCompressor(tensor_serializer)

            self._l2_cache = L2Cache(
                backend_name=backend_name,
                namespace=namespace,
                block_spec=self.block_spec,
                key_builder=key_builder,
                key_serializer=key_serializer,
                tensor_serializer=tensor_serializer,
                op_batch=op_batch,
            )

            # new an event loop to carry out L2Cache ops
            self._event_loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=self._event_loop.run_forever)
            self._lock = threading.Lock()
            self._infight_cv = threading.Condition(self._lock)
            self._thread.start()

            # launch L2Cache
            status = self._l2_cache.open()
            status.raise_if_has_exception()

            # register l1 cache callback
            if self._l1_cache is not None:
                if ingestion_type == "HOT":
                    self._l1_cache.set_on_hot_access_callback(
                        self._l2_ingestion_callback)
                elif ingestion_type == "ALL":
                    self._l1_cache.set_on_put_callback(
                        self._l2_ingestion_callback)
                else:
                    self._l1_cache.set_on_evict_callback(
                        self._l2_ingestion_callback)

        assert self._l1_cache is not None or self._l2_cache is not None, \
            "At least one cache service must be enabled."

    @property
    def feature(self) -> KVCacheFeature:
        """Get the feature of the kv cache.
        Returns:
            The feature of the kv cache.
        """
        if self._l1_cache is not None:
            return KVCacheFeature(zero_copy=True)
        # TODO: enable zero-copy if the L2Cache supports it.
        # elif self._l2_cache._backend.feature.zero_copy:
        #     return KVCacheFeature(zero_copy=True)
        if self._l2_inflight_quota > 0:
            return KVCacheFeature(non_blocking_put=True)
        return KVCacheFeature()

    @property
    def chunk_size(self) -> int:
        """Get the chunk size of the kv cache.
        Returns:
            The chunk size of the kv cache.
        """
        return self._chunk_size

    def __repr__(self) -> str:
        return f"BaseKVCacheManager(l1_cache={self._l1_cache}, l2_cache={self._l2_cache})"

    def __str__(self) -> str:
        return self.__repr__()

    def close(self) -> None:
        # flush
        self.flush()

        # terminate event loop and thread
        if self._event_loop is not None and self._event_loop.is_running():
            try:
                self._event_loop.call_soon_threadsafe(self._event_loop.stop)
            except:
                # ignore the exception
                pass

        if self._thread is not None and self._thread.is_alive():
            self._thread.join()

        if self._l1_cache is not None:
            del self._l1_cache
            self._l1_cache = None

        if self._l2_cache is not None:
            self._l2_cache.close()
            del self._l2_cache
            self._l2_cache = None

        self._thread = None
        self._event_loop = None

    def _l2_ingestion_callback(
            self,
            key_pair: Hashable,
            value: (torch.Tensor | MemoryRegion),
    ) -> Status:
        """Ingest the kv tensors to the L2Cache.
        Args:
            key_pair: E.g., (prefix, tokens)
            value: The kv tensors.
        Returns:
            The status of the ingestion operation and the number of tokens have
            been ingested or scheduled.
        """
        assert self._l2_cache is not None, "l2_cache is not initialized."

        if isinstance(value, tuple):
            raise NotImplementedError("Tuple kv tensors is not supported yet.")

        status = None
        if self._l2_inflight_quota == 0:
            # sync write
            status = self._l2_ingestion_sync_callback(key_pair, value)
        else:
            # async write
            status = self._l2_ingestion_async_callback(key_pair, value)

        self._release_mrs([value])
        return status

    def _l2_ingestion_async_callback(
            self,
            key_pair: Hashable,
            value: (torch.Tensor | MemoryRegion),
    ) -> Status:
        """Ingest the kv tensors to the L2Cache.
        Args:
            key_pair: E.g., (prefix, tokens)
            value: The kv tensors.
        Returns:
            The status of the ingestion operation and the number of tokens have
            been scheduled.
        """
        with self._lock:
            log_every_n_seconds(
                logger,
                logging.INFO,
                f"l2_cache infight writes {self._l2_inflight_writes}/quota {self._l2_inflight_quota}",
                n_seconds=3,
            )
            if self._l2_inflight_quota <= self._l2_inflight_writes:
                log_every_n_seconds(
                    logger,
                    logging.WARNING,
                    f"There are too many infight writes, skip writing to l2_cache. " \
                    f"infight writes {self._l2_inflight_writes}/quota {self._l2_inflight_quota}",
                    n_seconds=10,
                )
                return Status(StatusCodes.DENIED)
            self._l2_inflight_writes += 1

        prefix, tokens = key_pair
        tensor = value

        def _done_callback(future: asyncio.Future) -> None:
            with self._infight_cv:
                self._l2_inflight_writes -= 1
                self._infight_cv.notify_all()
            if not future.result().is_ok():
                log_every_n_seconds(
                    logger,
                    logging.WARNING,
                    f"Failed to write to l2_cache, error: {future.result().value}",
                    n_seconds=10,
                )

        # Async write to L2Cache
        future = asyncio.run_coroutine_threadsafe(
            self._l2_cache.put(prefix, tokens, tensor), self._event_loop)
        future.add_done_callback(_done_callback)
        return Status(StatusCodes.OK, len(tokens))

    def _l2_ingestion_sync_callback(
            self,
            key_pair: Hashable,
            value: (torch.Tensor | MemoryRegion),
    ) -> Status:
        """Ingest the kv tensors to the L2Cache.
        Args:
            key_pair: E.g., (prefix, tokens)
            value: The kv tensors.
        Returns:
            The status of the ingestion operation and the number of tokens have
            been ingested.
        """
        prefix, tokens = key_pair
        tensor = value
        future = asyncio.run_coroutine_threadsafe(
            self._l2_cache.put(prefix, tokens, tensor), self._event_loop)
        # wait until the write is done
        return future.result()

    def prefetch(self, prefix: Iterable[int] | None,
                 tokens: Iterable[int]) -> None:
        """(Optional) Prefetch the kv cache for the given prefix and tokens.
        Args:
            prefix: The prefix of the kv cache. E.g., [1, 2, 3]
            tokens: The tokens of the kv cache. E.g., [4, 5, 6, 7]
        """
        # TODO: implement background prefetching that loads kv cache from L2Cache to L1Cache.
        pass

    @nvtx_range("acquire", "KVCacheManager")
    def acquire(
        self,
        prefix: Iterable[int] | None,
        tokens: Iterable[int],
    ) -> Status[Tuple[int, Iterable[KVCacheHandle]]]:
        """(Optional) Acquire cache handle of the kv tensors for the given prefix
        and tokens. Only the kv cache services supporting zero-copy need to implement
        this method.
        
        The returned cache handle pointing to buffers owned by the kv cache service.
        We can use "KVCacheHandle.to_tensors()" to get tensors sharing the same storage.
        After the kv tensors are used, we need to explicitly `del` the cache handle to
        let the kv cache service know that these buffers are not referenced anymore.
        
        Args:
            prefix: The prefix of the kv cache. E.g., [1, 2, 3]
            tokens: The tokens of the kv cache. E.g., [4, 5, 6, 7]
        Returns:
            Number of tokens have been fetched from the kv cache service.
            The cache handles corresponding to the given tokens.
        """
        if self._l1_cache is None:
            raise NotImplementedError

        status = self._get_impl(prefix, tokens, zero_copy=True)
        if not status.is_ok():
            return status

        return Status(value=(len(status.value) * self.block_ntokens,
                             BaseKVCacheHandle(self.block_dtype, self.
                                               block_shape, status.value)))

    @nvtx_range("get", "KVCacheManager")
    def get(
        self,
        prefix: Iterable[int] | None,
        tokens: Iterable[int],
    ) -> Status[Tuple[int, torch.Tensor]]:
        """Get kv tensors from the kv cache service.

        Args:
            prefix: The prefix of the kv cache. E.g., [1, 2, 3]
            tokens: The tokens of the kv cache. E.g., [4, 5, 6, 7]
        Returns:
            Number of tokens have been fetched from the kv cache service.
            The kv tensors corresponding to the tokens:
                Its layout matches the layout of the kv cache service.

                For example, if the layout is NCLD, then:
                The k, v tensors for i-th token at the j-th layer are kv_tensors[i][0[j]
                and kv_tensors[i][1[j], respectively.
        """
        status = self._get_impl(prefix, tokens)
        if not status.is_ok():
            return status

        mrs_and_tensors = status.value
        result = None
        with perf_timer() as get_tensor_cat_dur_ms:
            result = self._merge_kv_tensors(*mrs_and_tensors)

        log_every_n_seconds(
            logger,
            logging.INFO,
            f"Concatenating tensors takes {get_tensor_cat_dur_ms():.4f} ms",
            n_seconds=10,
        )
        return Status(value=(len(mrs_and_tensors) * self.block_ntokens,
                             result))

    def _get_impl(
        self,
        prefix: Iterable[int] | None,
        tokens: Iterable[int],
        zero_copy: bool = False,
    ) -> Status[Iterable[torch.Tensor | MemoryRegion]]:
        """Get kv tensors from the kv cache service.

        Args:
            prefix: The prefix of the kv cache. E.g., [1, 2, 3]
            tokens: The tokens of the kv cache. E.g., [4, 5, 6, 7]
            zero_copy: Whether to use zero-copy.
        Returns:
            The kv tensors / memory regions corresponding to the tokens.
        """
        if prefix is not None and len(prefix) % self.block_ntokens != 0:
            return Status(StatusCodes.INVALID)

        num_blocks = len(tokens) // self.block_ntokens

        # If it is not a full block, return
        if num_blocks == 0:
            return Status(StatusCodes.NOT_FOUND)

        fetched_mrs = []
        num_fetched_blocks = 0
        num_missing_blocks = num_blocks
        l1_status = Status(StatusCodes.NOT_FOUND)
        if self._l1_cache is not None:
            l1_status = self._l1_cache.acquire(prefix, tokens)

            fetched_mrs = l1_status.value if l1_status.is_ok() else []
            num_fetched_blocks = len(fetched_mrs)
            num_missing_blocks = num_blocks - num_fetched_blocks

            if num_missing_blocks == 0 or not self._use_double_get(
                    num_missing_blocks, num_blocks):
                # 1. fully hit on L1Cache, return the result directly
                # 2. L2Cache is not enabled, return the result directly
                # 3. num of missing blocks is less than the threshold,
                #    return the result directly
                return l1_status

        assert self._l2_cache is not None
        # fetch missing kv tensors from L2Cache
        prefix_curr = [t for t in prefix] if prefix is not None else []
        prefix_curr.extend(tokens[:num_fetched_blocks * self.block_ntokens])
        tokens_curr = tokens[num_fetched_blocks * self.block_ntokens:]
        timeout_s = (num_missing_blocks * self.block_ntokens *
                     self._l2_cache_per_token_timeout_ms) / 1000

        future = asyncio.run_coroutine_threadsafe(
            self._l2_cache.get(prefix_curr, tokens_curr), self._event_loop)
        try:
            status = future.result(timeout=timeout_s)
            if not status.is_ok():
                return status if num_fetched_blocks == 0 else l1_status

            # put the fetched kv tensors to L1Cache
            if self._l1_cache is not None:
                # TODO: remove this tensor concat
                cat = torch.cat(status.value, dim=self.block_shape_token_dim)
                # discard its status
                self._l1_cache.put(prefix_curr, tokens_curr, cat)

            if zero_copy:
                assert self._l1_cache is not None
                # get the cache handles
                acquire_status = self._l1_cache.acquire(
                    prefix_curr, tokens_curr)
                if acquire_status.is_ok():
                    return Status(value=fetched_mrs + acquire_status.value)
                else:
                    return Status(value=fetched_mrs)

            return Status(value=fetched_mrs + status.value)
        except asyncio.CancelledError:
            # cancelled
            return Status(StatusCodes.CANCELLED
                          ) if num_fetched_blocks == 0 else l1_status
        except asyncio.TimeoutError:
            # timed out
            return Status(
                StatusCodes.TIMEOUT) if num_fetched_blocks == 0 else l1_status
        except Exception as e:
            # other exceptions
            return Status(StatusCodes.ERROR,
                          e) if num_fetched_blocks == 0 else l1_status
        finally:
            if not future.done():
                future.cancel()

    def _use_double_get(self, num_missing_blocks: int,
                        num_total_blocks: int) -> bool:
        """Whether to use double get.
        Args:
            num_missing_blocks: The number of missing blocks.
            num_total_blocks: The total number of blocks.
        Returns:
            Whether to use double get.
        """
        if self._l2_cache is None:
            return False
        if len(self._double_get_threshold) == 1:
            # only num is set
            return num_missing_blocks >= self._double_get_threshold[0]
        elif len(self._double_get_threshold) == 2:
            # both ratio and num are set
            return (num_missing_blocks / num_total_blocks
                    >= self._double_get_threshold[1]
                    and num_missing_blocks >= self._double_get_threshold[0])
        return False

    def _merge_kv_tensors(
        self,
        *args: torch.Tensor | MemoryRegion,
    ) -> torch.Tensor:
        """Merge two or more kv tensors.
        Args:
            *args: KV tensors.
        Returns:
            The merged kv tensor.
        """
        if args is None:
            return None

        tensors_or_tuples = []
        for mr_or_tensor in args:
            if isinstance(mr_or_tensor, MemoryRegion):
                tensors_or_tuples.append(
                    mr_or_tensor.to_tensor(self.block_dtype, self.block_shape))
            else:
                tensors_or_tuples.append(mr_or_tensor)

        if len(tensors_or_tuples) == 1:
            self._release_mrs(args)
            return tensors_or_tuples[0]

        self._release_mrs(args)
        return torch.cat(tensors_or_tuples, dim=self.block_shape_token_dim)

    def _release_mrs(
        self,
        mr_or_tensors: Iterable[torch.Tensor | MemoryRegion],
    ) -> None:
        if mr_or_tensors is None:
            return
        [mr.ref_down() for mr in mr_or_tensors if isinstance(mr, MemoryRegion)]

    @nvtx_range("allocate", "KVCacheManager")
    def allocate(
        self,
        nblocks: int,
    ) -> Status[KVCacheHandle]:
        """(Optional) Allocate a cache handle that points to buffers owned by the kv
        cache service.
        
        Only the kv cache services supporting zero-copy need to implement this method.

        Args:
            nblocks: The number of blocks to allocate.
        Returns:
            The cache handle.
        """
        if self._l1_cache is None:
            raise NotImplementedError

        status = self._l1_cache.allocate(nblocks)
        if not status.is_ok():
            return status
        return Status(value=BaseKVCacheHandle(self.block_dtype,
                                              self.block_shape, status.value))

    @nvtx_range("put", "KVCacheManager")
    def put(
        self,
        prefix: Iterable[int] | None,
        tokens: Iterable[int],
        kv_tensors: torch.Tensor | KVCacheHandle,
    ) -> Status[int]:
        """Put kv tensors to the kv cache service.

        Args:
            prefix: The prefix of the kv cache. E.g., [1, 2, 3]
            tokens: The tokens of the kv cache. E.g., [4, 5, 6, 7]
            kv_tensors:
                The kv tensors to put into the kv cache.
                
                The layout of kv_tensors must match the layout of the kv cache service.

                For example, if the layout is NCLD, then:
                The k, v tensors for i-th token at the j-th layer are kv_tensors[i][0[j]
                and kv_tensors[i][1[j], respectively.

        Returns:
            The status of the put operation and the number of tokens have been put or
            scheduled to put into the kv cache service.
        """
        # If L1Cache is enabled, we put kv tensors to L1Cache and leverage its
        # eviction policy to asynchronously ingest kv tensors to L2Cache.
        # Otherwise, we ingest kv tensors to L2Cache directly.
        if self._l1_cache is not None:
            return self._l1_cache.put(prefix, tokens, kv_tensors)
        else:
            assert isinstance(kv_tensors, torch.Tensor)
            return self._l2_ingestion_callback((prefix, tokens), kv_tensors)

    @nvtx_range("delete", "KVCacheManager")
    def delete(
        self,
        prefix: Iterable[int] | None,
        tokens: Iterable[int],
    ) -> Status:
        """Delete kv tensors from the kv cache service.
        Args:
            prefix: The prefix of the kv cache. E.g., [1, 2, 3]
            tokens: The tokens of the kv cache. E.g., [4, 5, 6, 7]
        Returns:
            The status of the delete operation.
        """
        if self._l1_cache is not None:
            status = self._l1_cache.delete(prefix, tokens)
            if not status.is_ok():
                return status

        if self._l2_cache is not None:
            future = asyncio.run_coroutine_threadsafe(
                self._l2_cache.delete(prefix, tokens), self._event_loop)
            status = future.result()
            if not status.is_ok():
                return status

        return Status(StatusCodes.OK)

    @nvtx_range("flush", "KVCacheManager")
    def flush(self) -> Status:
        """Flush the kv cache service.
        
        Returns:
            The status of the flush operation.
        """
        if self._infight_cv is None:
            return Status(StatusCodes.OK)

        try:
            with self._infight_cv:
                while self._l2_inflight_writes > 0:
                    self._infight_cv.wait(timeout=60)
        except TimeoutError:
            # timed out
            return Status(StatusCodes.TIMEOUT)
        except Exception as e:
            # other exceptions
            return Status(StatusCodes.ERROR, value=e)
        return Status(StatusCodes.OK)

    def cache_chunk_keys(
        self, prefix: Iterable[int] | None, tokens: Iterable[int]
    ) -> Iterator[Iterator[Tuple[Iterable[int], Iterable[int],
                                 Iterable[int]]]]:
        """Get the cache chunk keys.
        Args:
            prefix (Iterable[int] | None): The prefix tokens of the kv tensors.
            tokens (Iterable[int]): The tokens of the kv tensors.
        Returns:
            chunk prefix tokens, chunk tokens, next chunk tokens
        """
        cache_key = [] if prefix is None else [x for x in prefix]
        chunk_tokens = split_list(tokens, self._chunk_size)

        for i in range(len(chunk_tokens)):
            yield (
                tuple(cache_key),
                tuple(chunk_tokens[i]),
                tuple(chunk_tokens[i + 1] if i +
                      1 < len(chunk_tokens) else []),
            )
            cache_key += chunk_tokens[i]


class GroupAwareKVCacheManager(BaseKVCacheManager):
    """Group-aware KV cache manager.
    
    GroupAwareKVCacheManager uses collectives to ensure all participants
    have the same view towards cache operations.

    Args:
        config: The KV cache config.
        process_group: The process group.
    """

    def __init__(self, config: KVCacheConfig,
                 process_group: dist.ProcessGroup) -> None:
        super().__init__(config)

        assert dist.is_initialized(), f"torch.distributed must be initialized"
        assert process_group is not None, f"process_group must be set"

        self.process_group = process_group
        self.world_size = dist.get_world_size(group=process_group)
        self.rank = dist.get_rank(group=process_group)

    def __repr__(self) -> str:
        return super().__repr__().replace("BaseKVCacheManager",
                                          "GroupAwareKVCacheManager")

    def __str__(self) -> str:
        return self.__repr__()

    @nvtx_range("get", "GroupAwareKVCacheManager")
    def get(
        self,
        prefix: Iterable[int] | None,
        tokens: Iterable[int],
    ) -> Status[Tuple[int, torch.Tensor]]:
        """Get kv tensors from the kv cache service.

        Args:
            prefix: The prefix of the kv cache. E.g., [1, 2, 3]
            tokens: The tokens of the kv cache. E.g., [4, 5, 6, 7]
        Returns:
            Number of tokens have been fetched from the kv cache service.
            The kv tensors corresponding to the tokens:
                Its layout matches the layout of the kv cache service.

                For example, if the layout is NCLD, then:
                The k, v tensors for i-th token at the j-th layer are kv_tensors[i][0[j]
                and kv_tensors[i][1[j], respectively.
        """
        status = self._get_impl(prefix, tokens)
        if not status.is_ok():
            return status

        with perf_timer() as get_tensor_cat_dur_ms:
            result = self._merge_kv_tensors(*status.value[1])

        log_every_n_seconds(
            logger,
            logging.INFO,
            f"Concatenating tensors takes {get_tensor_cat_dur_ms():.4f} ms",
            n_seconds=10,
        )
        return Status(value=(status.value[0], result))

    @nvtx_range("acquire", "GroupAwareKVCacheManager")
    def acquire(
        self,
        prefix: Iterable[int] | None,
        tokens: Iterable[int],
    ) -> Status[Tuple[int, Iterable[KVCacheHandle]]]:
        """(Optional) Acquire cache handle of the kv tensors for the given prefix
        and tokens. Only the kv cache services supporting zero-copy need to implement
        this method.
        
        The returned cache handle pointing to buffers owned by the kv cache service.
        We can use "KVCacheHandle.to_tensors()" to get tensors sharing the same storage.
        After the kv tensors are used, we need to explicitly `ref_down()` the cache handle
        to let the kv cache service know that these buffers are not referenced anymore.
        
        Args:
            prefix: The prefix of the kv cache. E.g., [1, 2, 3]
            tokens: The tokens of the kv cache. E.g., [4, 5, 6, 7]
        Returns:
            Number of tokens have been fetched from the kv cache service.
            The cache handles corresponding to the given tokens.
        """
        status = self._get_impl(prefix, tokens, zero_copy=True)
        if not status.is_ok():
            return status
        handle = BaseKVCacheHandle(self.block_dtype, self.block_shape,
                                   status.value[1])
        return Status(value=(status.value[0], handle))

    def _get_impl(
        self,
        prefix: Iterable[int] | None,
        tokens: Iterable[int],
        zero_copy: bool = False,
    ) -> Status[Tuple[int, Iterable[torch.Tensor | KVCacheHandle]]]:
        """Get kv tensors / cache handles.
        
        Args:
            prefix: The prefix of the kv cache. E.g., [1, 2, 3]
            tokens: The tokens of the kv cache. E.g., [4, 5, 6, 7]
            zero_copy: Whether to use zero-copy.
        Returns:
            Number of tokens have been fetched from the kv cache service.
            The kv tensors / cache handles corresponding to the given tokens.
        """
        if prefix is not None and len(prefix) % self.block_ntokens != 0:
            return Status(StatusCodes.INVALID)

        # If it is not a full block, return
        if len(tokens) // self.block_ntokens == 0:
            return Status(StatusCodes.NOT_FOUND)

        start = 0
        results = []
        for chunk_prefix, chunk_tokens, next_tokens in self.cache_chunk_keys(
                prefix, tokens):
            if next_tokens and len(next_tokens) >= 0:
                # prefetch
                super().prefetch(chunk_prefix + next_tokens, next_tokens)
            status = super()._get_impl(chunk_prefix,
                                       chunk_tokens,
                                       zero_copy=zero_copy)
            # we only care about the error code and num of blocks
            coll_status = (Status(
                value=len(status.value)) if status.is_ok() else status)
            # check if all participants have the same status
            pg_statuses = [None] * self.world_size
            dist.all_gather_object(pg_statuses,
                                   coll_status,
                                   group=self.process_group)
            # if any participant encountered an error
            if not all([s.is_ok() for s in pg_statuses]):
                self._release_mrs(status.value)
                if start > 0:
                    # we have already got some tokens, return success
                    return Status(value=(start, results))
                else:
                    # return the first error
                    return next(s for s in pg_statuses if not s.is_ok())
            elif not all([
                    s.value * self.block_ntokens == len(chunk_tokens)
                    for s in pg_statuses
            ]):
                # some participants have got less tokens than others
                num = min(s.value for s in pg_statuses)
                results.extend(status.value[:num])
                self._release_mrs(status.value[num:])
                return Status(value=(start + num * self.block_ntokens,
                                     results))

            assert len(status.value) * self.block_ntokens == len(chunk_tokens)
            start += len(chunk_tokens)
            results.extend(status.value)

        return (Status(
            value=(start,
                   results)) if start > 0 else Status(StatusCodes.NOT_FOUND))
