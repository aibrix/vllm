# SPDX-License-Identifier: Apache-2.0
import asyncio
import contextlib
import functools
import logging
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Hashable, Iterable, Iterator, Tuple

import torch
import torch.distributed as dist
import uvloop

from . import envs
from .cache_handle import (KVCacheHandle, MemoryRegionKVCacheHandle,
                           ObjectPoolKVCacheHandle)
from .common import nvtx_range
from .common.absl_logging import getLogger, log_every_n_seconds
from .config import KVCacheConfig
from .l1 import L1Cache
from .l2 import L2Cache
from .memory import MemoryRegion, TensorPoolAllocator
from .metrics import KVCacheMetrics, MeasurableBase, MetricRecorder
from .spec import KVCacheBlockLayout, KVCacheBlockSpec
from .status import Status, StatusCodes
from .utils import cpu_perf_timer, in_place_pin_memory

logger = getLogger(__name__)

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


@dataclass
class KVCacheFeature:
    """The features of the kv cache.
    Args:
        zero_copy: Whether the kv cache supports zero-copy.
    """

    zero_copy: bool = False


class KVCacheManager(ABC):
    """The KV cache manager.

    Args:
        config: The KV cache manager configuration.
    """

    def __init__(self, config: KVCacheConfig) -> None:
        self.config: KVCacheConfig = config
        self.block_spec: KVCacheBlockSpec = self.config.block_spec
        self.block_layout: KVCacheBlockLayout = (
            self.config.block_spec.block_layout)
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

    @classmethod  # noqa: B027
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
        """(Optional) Allocate a cache handle that points to buffers owned
        by the kv cache service.

        Only the kv cache services supporting zero-copy need to implement
        this method.

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
        """(Optional) Acquire cache handle of the kv tensors for the given
        prefix and tokens. Only the kv cache services supporting zero-copy
        need to implement this method.

        The returned cache handle pointing to buffers owned by the kv cache
        service. We can use "KVCacheHandle.to_tensors()" to get tensors sharing
        the same storage. After the kv tensors are used, we need to explicitly
        `ref_down()` the cache handle to let the kv cache service know that
        these buffers are not referenced anymore.

        Args:
            prefix: The prefix of the kv cache. E.g., [1, 2, 3]
            tokens: The tokens of the kv cache. E.g., [4, 5, 6, 7]
        Returns:
            Number of tokens have been fetched from the kv cache service.
            The cache handles corresponding to the given tokens.
        """
        raise NotImplementedError

    @abstractmethod
    def exists(self, prefix: Iterable[int] | None,
               tokens: Iterable[int]) -> Status[int]:
        """Check if the kv cache corresponding to given prefix and
        tokens exists.

        Args:
            prefix: The prefix of the kv cache. E.g., [1, 2, 3]
            tokens: The tokens of the kv cache. E.g., [4, 5, 6, 7]
        Returns:
            Number of tokens existing in the kv cache service.
        """
        raise NotImplementedError

    @abstractmethod
    def get(
        self,
        prefix: Iterable[int] | None,
        tokens: Iterable[int],
        concat: bool = True,
    ) -> Status[Tuple[int, torch.Tensor | Iterable[torch.Tensor]]]:
        """Get kv tensors from the kv cache service.

        Args:
            prefix: The prefix of the kv cache. E.g., [1, 2, 3]
            tokens: The tokens of the kv cache. E.g., [4, 5, 6, 7]
            concat: Whether to concatenate the kv tensors.
        Returns:
            Number of tokens have been fetched from the kv cache service.
            The kv tensors corresponding to the tokens:
                Its layout matches the layout of the kv cache service.

                For example, if the layout is NCLD, then:
                The k, v tensors for i-th token at the j-th layer are
                kv_tensors[i][0[j] and kv_tensors[i][1[j], respectively.

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

                The layout of kv_tensors must match the layout of the
                kv cache service.

                For example, if the layout is NCLD, then:
                The k, v tensors for i-th token at the j-th layer are
                kv_tensors[i][0[j] and kv_tensors[i][1[j], respectively.

        Returns:
            The status of the put operation and the number of tokens have
            been put or scheduled to put into the kv cache service.
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
    ) -> Iterator[Iterator[Tuple[Iterable[int], Iterable[int], Iterable[int],
                                 Iterable[int]]]]:
        """Get the cache chunk keys.
        Args:
            prefix (Iterable[int] | None): The prefix tokens of the kv tensors.
            tokens (Iterable[int]): The tokens of the kv tensors.
        Returns:
            chunk prefix tokens, chunk tokens, next chunk tokens, and all tokens
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Close the kv cache service."""
        raise NotImplementedError

    @property
    def metrics(self) -> KVCacheMetrics:
        """Get the metrics of the kv cache service."""
        raise NotImplementedError


class BaseKVCacheManager(KVCacheManager, MeasurableBase):
    """Base KV cache manager.

    Args:
        config: The KV cache manager configuration.
    """

    def __init__(self, config: KVCacheConfig) -> None:
        KVCacheManager.__init__(self, config)

        self._l1_cache: L1Cache = None
        self._l2_cache: L2Cache = None
        self._executor: asyncio.Executor = None
        self._event_loop: asyncio.AbstractEventLoop = None
        self._thread: threading.Thread = None
        self._lock: threading.Lock = None
        self._infight_cv: threading.Condition = None
        self._l2_inflight_writes: int = 0
        self._l2_inflight_quota: int = 0
        self._allocator: TensorPoolAllocator = None
        self._metrics: KVCacheMetrics = None

        self._double_get_threshold: Tuple[
            int, float] = envs.VLLM_KV_CACHE_OL_DOUBLE_GET_THRESHOLD
        self._l2_cache_per_token_timeout_ms: int = (
            envs.VLLM_KV_CACHE_OL_L2_CACHE_PER_TOKEN_TIMEOUT_MS)

        self._chunk_size: int = envs.VLLM_KV_CACHE_OL_CHUNK_SIZE

        if self._chunk_size % self.block_ntokens != 0:
            self._chunk_size = (self._chunk_size -
                                self._chunk_size % self.block_ntokens)
            logger.warning(
                ("VLLM_KV_CACHE_OL_CHUNK_SIZE=%d is not divisible by "
                 "block_ntokens=%d, aligned to %d"),
                envs.VLLM_KV_CACHE_OL_CHUNK_SIZE, self.block_ntokens,
                self._chunk_size)

        if self._chunk_size < 4 * self.block_ntokens:
            logger.warning("chunk_size is too small, using %d instead",
                           4 * self.block_ntokens)
            self._chunk_size = 4 * self.block_ntokens

        device: str = envs.VLLM_KV_CACHE_OL_DEVICE
        pin_memory: bool = (device == "cpu")

        enable_l1: bool = envs.VLLM_KV_CACHE_OL_L1_CACHE_ENABLED
        enable_l2: bool = len(envs.VLLM_KV_CACHE_OL_L2_CACHE_BACKEND) > 0
        capacity_nbytes: int = int(envs.VLLM_KV_CACHE_OL_L1_CACHE_CAPACITY_GB *
                                   1024**3)
        capacity: int = capacity_nbytes // self.block_nbytes
        if envs.VLLM_KV_CACHE_OL_METRICS_ENABLED:
            enable_time_measurement = (
                envs.VLLM_KV_CACHE_OL_TIME_MEASUREMENT_ENABLED)
            enable_breakdown_measurement = (
                envs.VLLM_KV_CACHE_OL_BREAKDOWN_MEASUREMENT_ENABLED)
            self._metrics = KVCacheMetrics(
                block_ntokens=self.block_ntokens,
                capacity=capacity,
                enable_l1=enable_l1,
                enable_l2=enable_l2,
                enable_time_measurement=enable_time_measurement,
                enable_breakdown_measurement=enable_breakdown_measurement)

        # init MeasurableBase
        MeasurableBase.__init__(self, self._metrics.mgr)

        self._allocator = TensorPoolAllocator(
            self.block_nbytes,
            device=device,
            pin_memory=pin_memory,
        )

        if enable_l1:
            eviction_policy: str = (
                envs.VLLM_KV_CACHE_OL_L1_CACHE_EVICTION_POLICY)
            evict_size: int = envs.VLLM_KV_CACHE_OL_L1_CACHE_EVICT_SIZE
            self._allocator.increase(capacity * self.block_nbytes)

            self._l1_cache = L1Cache(
                eviction_policy,
                capacity,
                self._allocator,
                self.block_spec,
                evict_size,
                metrics=self._metrics.l1,
            )

        if enable_l2:
            backend_name: str = envs.VLLM_KV_CACHE_OL_L2_CACHE_BACKEND
            namespace: str = envs.VLLM_KV_CACHE_OL_L2_CACHE_NAMESPACE
            # compression: str = envs.VLLM_KV_CACHE_OL_L2_CACHE_COMPRESSION
            ingestion_type: str = envs.VLLM_KV_CACHE_OL_L2_CACHE_INGESTION_TYPE
            op_batch: int = envs.VLLM_KV_CACHE_OL_L2_CACHE_OP_BATCH
            self._executor = ThreadPoolExecutor(
                envs.VLLM_KV_CACHE_OL_L2_CACHE_NUM_ASYNC_WORKERS,
                thread_name_prefix="l2_cache_")
            self._l2_inflight_quota = (
                envs.VLLM_KV_CACHE_OL_L2_CACHE_INGESTION_MAX_INFLIGHT_TOKENS //
                self.block_ntokens)

            self._l2_cache = L2Cache(
                backend_name=backend_name,
                namespace=namespace,
                block_spec=self.block_spec,
                executor=self._executor,
                op_batch=op_batch,
                metrics=self._metrics.l2,
            )

            if self._l2_inflight_quota > 0:
                more_capacity = (
                    self._l2_inflight_quota * self.block_nbytes \
                        // self.block_ntokens
                )
            else:
                more_capacity = self.block_nbytes

            if self._l2_cache._backend.feature.rdma:
                more_capacity += (
                    self._chunk_size * self.block_nbytes \
                        // self.block_nbytes
                )

            self._allocator.increase(more_capacity)

            if self._l2_cache._backend.feature.rdma:
                for slab in self._allocator._slabs:
                    status = self._l2_cache.register_mr(
                        slab.data_ptr(), slab.numel())
                    if not status.is_ok():
                        logger.fatal(
                            f"Failed to register slab with "
                            f"{self._l2_cache._backend.name}'s register func")

            # new an event loop to carry out L2Cache ops
            self._event_loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=self._event_loop.run_forever, daemon=True)
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

        assert (self._l1_cache is not None or self._l2_cache
                is not None), "At least one cache service must be enabled."

        logger.info("%s is initialized", self)

    @property
    def feature(self) -> KVCacheFeature:
        """Get the feature of the kv cache.
        Returns:
            The feature of the kv cache.
        """
        if self._l1_cache is not None or \
            self._l2_cache._backend.feature.rdma:
            return KVCacheFeature(zero_copy=True)
        return KVCacheFeature()

    @property
    def chunk_size(self) -> int:
        """Get the chunk size of the kv cache.
        Returns:
            The chunk size of the kv cache.
        """
        return self._chunk_size

    @property
    def metrics(self) -> KVCacheMetrics:
        return self._metrics

    def __repr__(self) -> str:
        return (f"BaseKVCacheManager(l1_cache={self._l1_cache}, "
                f"l2_cache={self._l2_cache})")

    def __str__(self) -> str:
        return self.__repr__()

    def close(self) -> None:
        # flush
        self.flush()

        # terminate event loop and thread
        if self._event_loop is not None and self._event_loop.is_running():
            with contextlib.suppress(Exception):
                # ignore the exception
                self._event_loop.call_soon_threadsafe(self._event_loop.stop)

        if self._thread is not None and self._thread.is_alive():
            self._thread.join()

        if self._executor is not None:
            self._executor.shutdown(wait=True)

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
            value: (torch.Tensor | MemoryRegion | KVCacheHandle),
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

        status = None
        if self._l2_inflight_quota == 0:
            # sync write
            status = self._l2_ingestion_sync_callback(key_pair, value)
        else:
            # async write
            status = self._l2_ingestion_async_callback(key_pair, value)

        return status

    def _l2_ingestion_async_callback(
            self,
            key_pair: Hashable,
            value: (torch.Tensor | MemoryRegion | KVCacheHandle),
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
            log_every_n_seconds(logger, logging.INFO,
                                "l2_cache infight writes %d/quota %d", 3,
                                self._l2_inflight_writes,
                                self._l2_inflight_quota)
            if self._l2_inflight_quota <= self._l2_inflight_writes:
                log_every_n_seconds(
                    logger, logging.WARNING,
                    ("There are too many infight writes, skip writing to "
                     "l2_cache. infight writes %d/quota %d"), 10,
                    self._l2_inflight_writes, self._l2_inflight_quota)
                return Status(StatusCodes.DENIED)
            self._l2_inflight_writes += 1

        prefix, tokens = key_pair
        tensor = value

        def _done_callback(
            future: asyncio.Future,
            value: (torch.Tensor | MemoryRegion | KVCacheHandle)
        ) -> None:
            self._release([value])

            with self._infight_cv:
                self._l2_inflight_writes -= 1
                self._infight_cv.notify_all()
            if not future.result().is_ok():
                log_every_n_seconds(logger, logging.WARNING,
                                    "Failed to write to l2_cache, error: %s",
                                    10,
                                    future.result().value)

        # Async write to L2Cache
        future = asyncio.run_coroutine_threadsafe(
            self._l2_cache.put(prefix, tokens, tensor), self._event_loop)
        future.add_done_callback(functools.partial(_done_callback,
                                                   value=value))
        return Status(StatusCodes.OK, len(tokens))

    def _l2_ingestion_sync_callback(
            self,
            key_pair: Hashable,
            value: (torch.Tensor | MemoryRegion | KVCacheHandle),
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
        status = future.result()

        self._release([value])

        if not status.is_ok():
            return status
        return Status(value=status.value * self.block_ntokens)

    def prefetch(self, prefix: Iterable[int] | None,
                 tokens: Iterable[int]) -> None:
        """(Optional) Prefetch the kv cache for the given prefix and tokens.
        Args:
            prefix: The prefix of the kv cache. E.g., [1, 2, 3]
            tokens: The tokens of the kv cache. E.g., [4, 5, 6, 7]
        """
        # TODO: implement background prefetching that loads kv cache
        # from L2Cache to L1Cache.
        pass

    @nvtx_range("acquire", "KVCacheManager")
    @MeasurableBase.measure(MetricRecorder.OP.ACQUIRE)
    def acquire(
        self,
        prefix: Iterable[int] | None,
        tokens: Iterable[int],
    ) -> Status[Tuple[int, Iterable[KVCacheHandle]]]:
        """(Optional) Acquire cache handle of the kv tensors for the given
        prefix and tokens. Only the kv cache services supporting zero-copy
        need to implement this method.

        The returned cache handle pointing to buffers owned by the kv cache
        service. We can use "KVCacheHandle.to_tensors()" to get tensors
        sharing the same storage.  After the kv tensors are used, we need to
        explicitly `del` the cache handle to let the kv cache service know that
        these buffers are not referenced anymore.

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

        return Status(value=(
            len(status.value) * self.block_ntokens,
            MemoryRegionKVCacheHandle(self.block_dtype, self.block_shape,
                                      status.value),
        ))

    @MeasurableBase.measure(MetricRecorder.OP.EXISTS)
    def exists(self, prefix: Iterable[int] | None,
               tokens: Iterable[int]) -> Status[int]:
        """Check if the kv cache corresponding to given prefix and
        tokens exists.

        Args:
            prefix: The prefix of the kv cache. E.g., [1, 2, 3]
            tokens: The tokens of the kv cache. E.g., [4, 5, 6, 7]
        Returns:
            Number of tokens existing in the kv cache service.
        """
        status = self._exists_impl(prefix, tokens)
        if not status.is_ok():
            return status

        return Status(value=status.value * self.block_ntokens)

    def _exists_impl(self, prefix: Iterable[int] | None,
                     tokens: Iterable[int]) -> Status[int]:
        """Check if the kv cache corresponding to given prefix and
        tokens exists.

        Args:
            prefix: The prefix of the kv cache. E.g., [1, 2, 3]
            tokens: The tokens of the kv cache. E.g., [4, 5, 6, 7]
        Returns:
            Number of tokens existing in the kv cache service.
        """
        if prefix is not None and len(prefix) % self.block_ntokens != 0:
            return Status(StatusCodes.INVALID)

        num_blocks = len(tokens) // self.block_ntokens

        # If it is not a full block, return
        if num_blocks == 0:
            return Status(StatusCodes.NOT_FOUND)

        num_existing_blocks = 0
        num_missing_blocks = num_blocks
        l1_status = Status(StatusCodes.NOT_FOUND)
        if self._l1_cache is not None:
            l1_status = self._l1_cache.exists(prefix, tokens)

            num_existing_blocks = l1_status.value if l1_status.is_ok() else 0
            num_missing_blocks -= num_existing_blocks

            if num_missing_blocks == 0 or not self._use_double_get(
                    num_missing_blocks, num_blocks):
                # 1. fully exist in L1Cache, return the result directly
                # 2. L2Cache is not enabled, return the result directly
                # 3. num of missing blocks is less than the threshold,
                #    return the result directly
                return l1_status

        assert self._l2_cache is not None
        num_existing_tokens = num_existing_blocks * self.block_ntokens
        # check if missing kv tensors are in L2Cache
        prefix_curr = [t for t in prefix] if prefix is not None else []
        prefix_curr.extend(tokens[:num_existing_tokens])
        tokens_curr = tokens[num_existing_tokens:]
        timeout_s = (num_missing_blocks * self.block_ntokens *
                     self._l2_cache_per_token_timeout_ms) / 1000

        future = asyncio.run_coroutine_threadsafe(
            self._l2_cache.exists(prefix_curr, tokens_curr), self._event_loop)
        try:
            status = future.result(timeout=timeout_s)
            if not status.is_ok():
                return status if num_existing_blocks == 0 else l1_status

            return Status(value=num_existing_blocks + status.value)
        except asyncio.CancelledError:
            # cancelled
            return (Status(StatusCodes.CANCELLED)
                    if num_existing_blocks == 0 else l1_status)
        except asyncio.TimeoutError:
            # timed out
            return (Status(StatusCodes.TIMEOUT)
                    if num_existing_blocks == 0 else l1_status)
        except Exception as e:
            # other exceptions
            return (Status(StatusCodes.ERROR, e)
                    if num_existing_blocks == 0 else l1_status)
        finally:
            if not future.done():
                future.cancel()

    @nvtx_range("get", "KVCacheManager")
    @MeasurableBase.measure(MetricRecorder.OP.GET)
    def get(
        self,
        prefix: Iterable[int] | None,
        tokens: Iterable[int],
        concat: bool = True,
    ) -> Status[Tuple[int, torch.Tensor | Iterable[torch.Tensor]]]:
        """Get kv tensors from the kv cache service.

        Args:
            prefix: The prefix of the kv cache. E.g., [1, 2, 3]
            tokens: The tokens of the kv cache. E.g., [4, 5, 6, 7]
            concat: Whether to concatenate the kv tensors.
        Returns:
            Number of tokens have been fetched from the kv cache service.
            The kv tensors corresponding to the tokens:
                Its layout matches the layout of the kv cache service.

                For example, if the layout is NCLD, then:
                The k, v tensors for i-th token at the j-th layer are
                kv_tensors[i][0[j] and kv_tensors[i][1[j], respectively.
        """
        status = self._get_impl(prefix, tokens)
        if not status.is_ok():
            return status

        if not concat:
            for i, x in enumerate(status.value):
                status.value[i] = in_place_pin_memory(x)

            return Status(value=(len(status.value) * self.block_ntokens,
                                 status.value))

        mrs_and_tensors = status.value
        result = None
        with cpu_perf_timer() as get_tensor_cat_dur_ms:
            result = self._merge_kv_tensors(*mrs_and_tensors)
            result = in_place_pin_memory(result)

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
        use_rdma = self._l2_cache._backend.feature.rdma
        # fetch missing kv tensors from L2Cache
        prefix_curr = [t for t in prefix] if prefix is not None else []
        prefix_curr.extend(tokens[:num_fetched_blocks * self.block_ntokens])
        tokens_curr = tokens[num_fetched_blocks * self.block_ntokens:]
        timeout_s = (num_missing_blocks * self.block_ntokens *
                     self._l2_cache_per_token_timeout_ms) / 1000

        mrs = None
        if use_rdma:
            nblocks_curr = len(tokens_curr) // self.block_ntokens
            status = self._allocator.alloc(nblocks_curr * self.block_nbytes)
            if not status.is_ok():
                return status if num_fetched_blocks == 0 else l1_status
            mrs = status.value
        future = asyncio.run_coroutine_threadsafe(
            self._l2_cache.get(prefix_curr, tokens_curr, mrs),
            self._event_loop)
        try:
            status = future.result(timeout=timeout_s)
            if not status.is_ok():
                if use_rdma:
                    self._release(mrs)
                return status if num_fetched_blocks == 0 else l1_status

            if use_rdma:
                self._release(mrs[len(status.value):])

            # put the fetched kv tensors to L1Cache
            if self._l1_cache is not None:
                # discard its status
                self._l1_cache.put(prefix_curr, tokens_curr, status.value)

            if zero_copy:
                if use_rdma:
                    self._release(mrs[len(status.value):])
                    return Status(value=fetched_mrs + status.value)
                else:
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
            return (Status(StatusCodes.CANCELLED)
                    if num_fetched_blocks == 0 else l1_status)
        except asyncio.TimeoutError:
            # timed out
            return (Status(StatusCodes.TIMEOUT)
                    if num_fetched_blocks == 0 else l1_status)
        except Exception as e:
            # other exceptions
            return (Status(StatusCodes.ERROR, e)
                    if num_fetched_blocks == 0 else l1_status)
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
            self._release(args)
            return tensors_or_tuples[0]

        self._release(args)
        return torch.cat(tensors_or_tuples, dim=self.block_shape_token_dim)

    def _release(
        self,
        mr_or_handle_or_tensors: Iterable[torch.Tensor | MemoryRegion
                                          | KVCacheHandle],
    ) -> None:
        if mr_or_handle_or_tensors is None:
            return
        for x in mr_or_handle_or_tensors:
            if isinstance(x, MemoryRegion):
                x.ref_down()
            elif isinstance(x, KVCacheHandle):
                x.release()

    @nvtx_range("allocate", "KVCacheManager")
    def allocate(
        self,
        nblocks: int,
    ) -> Status[KVCacheHandle]:
        """(Optional) Allocate a cache handle that points to buffers
        owned by the kv cache service.

        Only the kv cache services supporting zero-copy need to implement
        this method.

        Args:
            nblocks: The number of blocks to allocate.
        Returns:
            The cache handle.
        """
        if self._l1_cache is not None:
            status = self._l1_cache.allocate(nblocks)
        elif self._l2_inflight_quota > 0:
            # l2 cache only, async put: returns OOM if no inflight quota
            status = None
            with self._lock:
                if self._l2_inflight_writes >= self._l2_inflight_quota:
                    status = Status(StatusCodes.OUT_OF_MEMORY)
            if status is None:
                status = self._allocator.alloc(self.block_nbytes * nblocks)
        else:
            # l2 cache only, sync put
            status = self._allocator.alloc(self.block_nbytes * nblocks)

        if not status.is_ok():
            return status
        return Status(value=MemoryRegionKVCacheHandle(
            self.block_dtype, self.block_shape, status.value))

    @nvtx_range("put", "KVCacheManager")
    @MeasurableBase.measure(MetricRecorder.OP.PUT)
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

                The layout of kv_tensors must match the layout of the
                kv cache service.

                For example, if the layout is NCLD, then:
                The k, v tensors for i-th token at the j-th layer are
                kv_tensors[i][0[j] and kv_tensors[i][1[j], respectively.

        Returns:
            The status of the put operation and the number of tokens have
            been put or scheduled to put into the kv cache service.
        """
        # If L1Cache is enabled, we put kv tensors to L1Cache and leverage its
        # eviction policy to asynchronously ingest kv tensors to L2Cache.
        # Otherwise, we ingest kv tensors to L2Cache directly.
        if self._l1_cache is not None:
            if isinstance(kv_tensors, ObjectPoolKVCacheHandle):
                handle = kv_tensors
                kv_tensors = kv_tensors._tensors
                status = self._l1_cache.put(prefix, tokens, kv_tensors)
                # return all tensors back to the object pool
                handle.release()
            elif isinstance(kv_tensors, MemoryRegionKVCacheHandle):
                handle = kv_tensors
                kv_tensors = kv_tensors._mrs
                status = self._l1_cache.put(prefix, tokens, kv_tensors)
                # release mrs that are not put to L1Cache
                self._release(handle._mrs[status.value:])
            else:
                status = self._l1_cache.put(prefix, tokens, kv_tensors)

            if not status.is_ok():
                return status
            return Status(value=status.value * self.block_ntokens)
        else:
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
    ) -> Iterator[Iterator[Tuple[Iterable[int], Iterable[int], Iterable[int],
                                 Iterable[int]]]]:
        """Get the cache chunk keys.
        Args:
            prefix (Iterable[int] | None): The prefix tokens of the kv tensors.
            tokens (Iterable[int]): The tokens of the kv tensors.
        Returns:
            chunk prefix tokens, chunk tokens, next chunk tokens, and all tokens
        """
        not_none_prefix = tuple() if prefix is None else tuple(prefix)
        all = tuple(not_none_prefix + tuple(tokens))

        cache_key_len = len(not_none_prefix)
        num_tokens = len(tokens)
        aligned_num_tokens = num_tokens - num_tokens % self.block_ntokens
        num_chunks = -(-aligned_num_tokens // self._chunk_size)
        for i in range(num_chunks):
            yield (
                all[:cache_key_len],
                all[cache_key_len:cache_key_len + self._chunk_size],
                all[cache_key_len + self._chunk_size:cache_key_len +
                    2 * self._chunk_size] if cache_key_len +
                2 * self._chunk_size < len(all) else tuple(),
                all,
            )
            cache_key_len += self._chunk_size


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
        assert dist.is_initialized(), "torch.distributed must be initialized"
        assert process_group is not None, "process_group must be set"

        self.process_group = process_group
        self.world_size = dist.get_world_size(group=process_group)
        self.rank = dist.get_rank(group=process_group)

        super().__init__(config)

    def __repr__(self) -> str:
        return (super().__repr__().replace(
            "BaseKVCacheManager",
            f"GroupAwareKVCacheManager[rank={self.rank}, "
            f"world_size={self.world_size}]",
        ))

    def __str__(self) -> str:
        return self.__repr__()

    @nvtx_range("get", "GroupAwareKVCacheManager")
    @MeasurableBase.measure(MetricRecorder.OP.GET)
    def get(
        self,
        prefix: Iterable[int] | None,
        tokens: Iterable[int],
        concat: bool = True,
    ) -> Status[Tuple[int, torch.Tensor | Iterable[torch.Tensor]]]:
        """Get kv tensors from the kv cache service.

        Args:
            prefix: The prefix of the kv cache. E.g., [1, 2, 3]
            tokens: The tokens of the kv cache. E.g., [4, 5, 6, 7]
            concat: Whether to concatenate the kv tensors.
        Returns:
            Number of tokens have been fetched from the kv cache service.
            The kv tensors corresponding to the tokens:
                Its layout matches the layout of the kv cache service.

                For example, if the layout is NCLD, then:
                The k, v tensors for i-th token at the j-th layer are
                kv_tensors[i][0[j] and kv_tensors[i][1[j], respectively.
        """
        status = self._get_impl(prefix, tokens)
        if not status.is_ok():
            return status

        if not concat:
            return status

        with cpu_perf_timer() as get_tensor_cat_dur_ms:
            result = self._merge_kv_tensors(*status.value[1])

        log_every_n_seconds(
            logger,
            logging.INFO,
            f"Concatenating tensors takes {get_tensor_cat_dur_ms():.4f} ms",
            n_seconds=10,
        )
        return Status(value=(status.value[0], result))

    @nvtx_range("acquire", "GroupAwareKVCacheManager")
    @MeasurableBase.measure(MetricRecorder.OP.ACQUIRE)
    def acquire(
        self,
        prefix: Iterable[int] | None,
        tokens: Iterable[int],
    ) -> Status[Tuple[int, Iterable[KVCacheHandle]]]:
        """(Optional) Acquire cache handle of the kv tensors for the given
        prefix and tokens. Only the kv cache services supporting zero-copy
        need to implement this method.

        The returned cache handle pointing to buffers owned by the kv cache
        service. We can use "KVCacheHandle.to_tensors()" to get tensors sharing
        the same storage. After the kv tensors are used, we need to explicitly
        `ref_down()` the cache handle to let the kv cache service know that
        these buffers are not referenced anymore.

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
        handle = MemoryRegionKVCacheHandle(self.block_dtype, self.block_shape,
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
        for chunk_prefix, chunk_tokens, next_tokens, _ in self.cache_chunk_keys(
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
                self._release(status.value)
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
                self._release(status.value[num:])
                return Status(value=(start + num * self.block_ntokens,
                                     results))

            assert len(status.value) * self.block_ntokens == len(chunk_tokens)
            start += len(chunk_tokens)
            results.extend(status.value)

        return (Status(
            value=(start,
                   results)) if start > 0 else Status(StatusCodes.NOT_FOUND))
