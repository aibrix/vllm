# SPDX-License-Identifier: Apache-2.0
import asyncio
import itertools
from concurrent.futures import Executor
from typing import Iterable, Iterator, Tuple

import torch

from ..cache_handle import KVCacheHandle, MemoryRegionKVCacheHandle
from ..common import nvtx_range
from ..common.absl_logging import getLogger
from ..memory import MemoryRegion
from ..metrics import L2CacheMetrics, MeasurableBase, MetricRecorder
from ..spec import KVCacheBlockLayout, KVCacheBlockSpec
from ..status import Status, StatusCodes
from .connectors import Connector
from .key_builders import KeyBuilder, RawKeyBuilder

logger = getLogger(__name__)


class L2Cache(MeasurableBase):

    def __init__(
        self,
        backend_name: str,
        namespace: str,
        block_spec: KVCacheBlockSpec,
        executor: Executor,
        op_batch: int = 8,
        metrics: L2CacheMetrics | None = None,
    ) -> None:
        """Create a cache object.
        Args:
            backend_name (str): The name of cache backend.
            namespace (str): Namespace.
            block_spec (KVCacheBlockSpec): The block spec.
            executor (Executor): The executor.
            op_batch (int): The number of ops in a batch.
            metrics (L2CacheMetrics): metrics recorder.
        """
        super().__init__(metrics)
        self.block_spec: KVCacheBlockSpec = block_spec
        self.block_layout: KVCacheBlockLayout = self.block_spec.block_layout
        self.block_shape: Tuple[int, ...] = self.block_spec.block_shape
        self.block_dtype: torch.dtype = self.block_spec.block_dtype
        self.block_ntokens: int = self.block_spec.block_ntokens
        self.block_shape_token_dim: int = self.block_spec.block_shape_token_dim
        self.key_builder: KeyBuilder = RawKeyBuilder(self.block_ntokens)
        self.op_batch: int = op_batch
        self._backend: Connector = None
        self._executor: Executor = executor

        cat_head_ids = "_".join([
            str(self.block_spec.tensor_spec.heads[0]),
            str(self.block_spec.tensor_spec.heads[-1])
        ])
        cat_layer_ids = "_".join([
            str(self.block_spec.tensor_spec.layers[0]),
            str(self.block_spec.tensor_spec.layers[-1]),
        ])
        partition_id = f"h{cat_head_ids}_l{cat_layer_ids}"
        self._backend = Connector.create(backend_name, namespace, partition_id,
                                         executor)
        self._register_descs = []

        logger.info("%s is initialized. Using partition_id=%s.", str(self),
                    partition_id)

    def __repr__(self) -> str:
        return f"L2Cache(backend={self._backend.name})"

    def __str__(self) -> str:
        return self.__repr__()

    def __del__(self) -> None:
        self.close()
        logger.info("%s is closed.", str(self))

    def open(self) -> Status:
        """Open the cache."""
        return self._backend.open()

    def close(self) -> Status:
        """Close the cache."""
        if self._backend is not None:
            for desc in self._register_descs:
                self._backend.deregister_mr(desc)
            return self._backend.close()
        return Status(StatusCodes.OK)

    def register_mr(self, addr: int, length: int) -> Status:
        if not self._backend.feature.rdma:
            raise NotImplementedError
        status = self._backend.register_mr(addr, length)
        if not status.is_ok():
            return status
        self._register_descs.append(status.value)

    @nvtx_range("prefetch", "kv_cache_ol.L2Cache")
    async def prefetch(
        self,
        prefix: Iterable[int] | None,
        tokens: Iterable[int],
    ) -> Status:
        """Prefetch kv tensors from the cache.
        Args:
            prefix (Iterable[int] | None): The prefix tokens of the kv tensors.
            tokens (Iterable[int]): The tokens of the kv tensors.
        Returns:
            The status of the prefetch operation.
        """
        if not self._backend.feature.prefetch:
            return Status(StatusCodes.OK)

        if prefix is not None and len(prefix) % self.block_ntokens != 0:
            return Status(StatusCodes.INVALID)

        await asyncio.gather(
            *(self._prefetch_impl(cache_key)
              for _, cache_key in self._cache_block_keys(prefix, tokens)),
            return_exceptions=False,  # backend returns exception as status
        )
        return Status(StatusCodes.OK)

    async def _prefetch_impl(self, cache_key: str) -> Status:
        return await self._backend.prefetch(cache_key)

    @nvtx_range("exists", "kv_cache_ol.L2Cache")
    @MeasurableBase.measure(MetricRecorder.OP.EXISTS)
    async def exists(
        self,
        prefix: Iterable[int] | None,
        tokens: Iterable[int],
    ) -> Status[int]:
        """Check if kv tensors exist in the cache.
        Args:
            prefix (Iterable[int] | None): The prefix tokens of the kv tensors.
            tokens (Iterable[int]): The tokens of the kv tensors.
        Returns:
            The number of blocks that exist in the cache.
        """
        if prefix is not None and len(prefix) % self.block_ntokens != 0:
            return Status(StatusCodes.INVALID)

        total = 0
        for key_batch in self._cache_block_key_batchs(prefix, tokens):
            tasks = []
            async with asyncio.TaskGroup() as tg:
                for real_key, cache_key in key_batch:
                    tasks.append(
                        tg.create_task(self._backend.exists(cache_key)))

            if len(tasks) == 0:
                break

            should_break = False
            for i in range(len(tasks)):
                if not tasks[i].done() or not tasks[i].result().is_ok():
                    should_break = True
                    break
                total += 1

            if should_break:
                break

        if total == 0:
            return Status(StatusCodes.NOT_FOUND)

        return Status(value=total)

    @nvtx_range("put", "kv_cache_ol.L2Cache")
    @MeasurableBase.measure(MetricRecorder.OP.PUT)
    async def put(
        self,
        prefix: Iterable[int] | None,
        tokens: Iterable[int],
        kv_tensors: (torch.Tensor | MemoryRegion | KVCacheHandle),
    ) -> Status[int]:
        """Put kv tensors to the cache.
        Args:
            prefix (Iterable[int] | None): The prefix tokens of the kv tensors.
            tokens (Iterable[int]): The tokens of the kv tensors.
            kv_tensors: kv tensors or cache handles.
        Returns:
            The status of the put operation and the number of blocks.
        """
        if prefix is not None and len(prefix) % self.block_ntokens != 0:
            return Status(StatusCodes.INVALID)

        # If it is not a full block, we don't need to cache it.
        if len(tokens) // self.block_ntokens == 0:
            return Status(StatusCodes.OK, 0)

        num_tokens = len(tokens)
        num_blocks = num_tokens // self.block_ntokens

        if isinstance(kv_tensors, MemoryRegion):
            # `kv_tensors` comes from L1Cache and should be only one block
            assert len(tokens) // self.block_ntokens == 1
            kv_tensors = kv_tensors.to_tensor(self.block_dtype,
                                              self.block_shape)
            if len(tokens) != kv_tensors.shape[self.block_shape_token_dim]:
                return Status(
                    StatusCodes.INVALID,
                    (f"Number of tokens {len(tokens)} is not equal to the "
                     f"number of tokens in key tensors "
                     f"{kv_tensors.shape[self.block_shape_token_dim]}."),
                )

        if isinstance(kv_tensors, torch.Tensor):
            # split to kv blocks
            slices = [slice(None)] * len(self.block_shape)
            slices[self.block_shape_token_dim] = slice(
                0, num_blocks * self.block_ntokens)
            blocks = torch.split(
                kv_tensors[tuple(slices)],
                self.block_ntokens,
                dim=self.block_shape_token_dim,
            )
        elif isinstance(kv_tensors, KVCacheHandle):
            blocks = kv_tensors.to_tensors()
            if len(tokens) != len(blocks) * self.block_ntokens:
                return Status(
                    StatusCodes.INVALID,
                    (f"Number of tokens {len(tokens)} is not equal to the "
                     f"number of tokens in key tensors "
                     f"{len(blocks) * self.block_ntokens}."),
                )
            if isinstance(kv_tensors, MemoryRegionKVCacheHandle) \
                and self._backend.feature.gather_scatter:
                return await self._scatter(prefix, tokens, kv_tensors._mrs)
        else:
            raise ValueError("Unsupported kv tensors type")

        # TODO: use mput if backend's mput_mget feature is enabled.
        num_processed_blocks = 0
        block_idx = 0
        for key_batch in self._cache_block_key_batchs(prefix, tokens):
            tasks = []
            num_blocks_in_batch = len(key_batch)
            async with asyncio.TaskGroup() as tg:
                for real_key, cache_key in key_batch:
                    block = blocks[block_idx]
                    block_idx += 1
                    tasks.append(
                        tg.create_task(self._backend.put(cache_key, block)))

            if len(tasks) == 0:
                return Status(StatusCodes.ERROR)
            elif all(task.done() and task.result().is_ok() for task in tasks):
                # all success, continue to the next batch
                num_processed_blocks += num_blocks_in_batch
                continue
            elif num_processed_blocks > 0:
                # current batch is not the first one.
                # at least one batch is done successfully, return success.
                return Status(StatusCodes.OK, num_processed_blocks)
            else:
                # this is the first batch and at least one block in
                # current batch is failed, return error.
                failures = [
                    task for task in tasks
                    if task.done() and not task.result().is_ok()
                ]
                if len(failures) > 0:
                    return failures[0].result()
                return Status(StatusCodes.ERROR)

        return Status(StatusCodes.OK, num_processed_blocks)

    async def _scatter(
        self,
        prefix: Iterable[int] | None,
        tokens: Iterable[int],
        mrs: Iterable[MemoryRegion],
    ) -> Status[int]:
        assert mrs is not None
        keys = self._cache_block_keys(prefix, tokens)
        cache_keys = [cache_key for _, cache_key in keys]
        sge_lists = self._backend.get_sge_list(cache_keys, mrs)

        num_processed_blocks = 0
        for sge_list_batch in itertools.batched(sge_lists, self.op_batch):
            tasks = []
            num_blocks_in_batch = 0
            async with asyncio.TaskGroup() as tg:
                for sge_list in sge_list_batch:
                    num_blocks_in_batch += len(sge_list)
                    tasks.append(
                        tg.create_task(self._backend.scatter(sge_list)))

            if len(tasks) == 0:
                return Status(StatusCodes.ERROR)
            elif all(task.done() and task.result().is_ok() for task in tasks):
                # all success, continue to the next batch
                num_processed_blocks += num_blocks_in_batch
                continue
            elif num_processed_blocks > 0:
                # current batch is not the first one.
                # at least one batch is done successfully, return success.
                return Status(StatusCodes.OK, num_processed_blocks)
            else:
                # this is the first batch and at least one block in
                # current batch is failed, return error.
                failures = [
                    task for task in tasks
                    if task.done() and not task.result().is_ok()
                ]
                if len(failures) > 0:
                    return failures[0].result()
                return Status(StatusCodes.ERROR)

        return Status(StatusCodes.OK, num_processed_blocks)

    @nvtx_range("get", "kv_cache_ol.L2Cache")
    @MeasurableBase.measure(MetricRecorder.OP.GET)
    async def get(
        self,
        prefix: Iterable[int] | None,
        tokens: Iterable[int],
        mrs: Iterable[MemoryRegion] = None,
    ) -> Status[Iterable[torch.Tensor | MemoryRegion]]:
        """Get kv tensors from the cache.
        Args:
            prefix (Iterable[int] | None): The prefix tokens of the kv tensors.
            tokens (Iterable[int]): The tokens of the kv tensors.
            mrs (Iterable[MemoryRegion]): Memory regions to place the fetched
                                          kv tensors.
        Returns:
            The kv tensors corresponding to the tokens.
        """
        if prefix is not None and len(prefix) % self.block_ntokens != 0:
            return Status(StatusCodes.INVALID)

        if mrs is not None:
            assert len(mrs) == len(tokens) // self.block_ntokens
            if self._backend.feature.gather_scatter:
                return await self._gather(prefix, tokens, mrs)

        use_rdma = self._backend.feature.rdma
        # TODO: use mget if backend's mput_mget feature is enabled.
        tensors = []
        offset = 0
        for key_batch in self._cache_block_key_batchs(prefix, tokens):
            tasks = []
            async with asyncio.TaskGroup() as tg:
                tid = 0
                for real_key, cache_key in key_batch:
                    mr = mrs[offset + tid] if mrs is not None else None
                    tid += 1  # noqa: SIM113

                    tasks.append(
                        tg.create_task(self._get_impl(real_key, cache_key,
                                                      mr)))

            if len(tasks) == 0:
                break

            should_break = False
            for i in range(len(tasks)):
                if not tasks[i].done() or not tasks[i].result().is_ok():
                    should_break = True
                    break
                if use_rdma:
                    tensors.append(mrs[offset + i])
                else:
                    tensors.append(tasks[i].result().value)

            if should_break:
                break

            offset += self.op_batch

        if len(tensors) == 0:
            return Status(StatusCodes.NOT_FOUND)

        return Status(value=tensors)

    async def _gather(
        self,
        prefix: Iterable[int] | None,
        tokens: Iterable[int],
        mrs: Iterable[MemoryRegion],
    ) -> Status[Iterable[MemoryRegion]]:
        assert mrs is not None
        keys = self._cache_block_keys(prefix, tokens)
        cache_keys = [cache_key for _, cache_key in keys]
        sge_lists = self._backend.get_sge_list(cache_keys, mrs)

        succ_total = 0
        for sge_list_batch in itertools.batched(sge_lists, self.op_batch):
            tasks = []
            async with asyncio.TaskGroup() as tg:
                for sge_list in sge_list_batch:
                    tasks.append(tg.create_task(
                        self._backend.gather(sge_list)))

            if len(tasks) == 0:
                break

            should_break = False
            for i in range(len(tasks)):
                if not tasks[i].done() or not tasks[i].result().is_ok():
                    should_break = True
                    break
                succ_total += len(sge_list_batch[i])

            if should_break:
                break

        if succ_total == 0:
            return Status(StatusCodes.NOT_FOUND)

        return Status(value=mrs[:succ_total])

    async def _get_impl(
        self,
        real_key: Iterable[int],
        cache_key: str,
        mr: MemoryRegion,
    ) -> Status[torch.Tensor]:
        """Get kv tensors from the backend.
        Args:
            real_key (Iterable[int]): The key of the kv tensors.
            cache_key (str): The cache key of the kv tensors.
            mr (MemoryRegion): The memory region to place the fetched
                               kv tensor.
        Returns:
            The kv tensors corresponding to the key.
        """
        if mr is None:
            status = await self._backend.get(cache_key)
            if status.is_ok():
                tensor = status.value
                return Status(
                    value=tensor.view(self.block_dtype).view(self.block_shape))
            else:
                return status
        else:
            return await self._backend.get(cache_key, mr)

    @nvtx_range("delete", "kv_cache_ol.L2Cache")
    async def delete(self, prefix: Iterable[int] | None,
                     tokens: Iterable[int]) -> Status:
        """Delete kv tensors from the cache.
        Args:
            prefix (Iterable[int] | None): The prefix tokens of the kv tensors.
            tokens (Iterable[int]): The tokens of the kv tensors.
        Returns:
            The status of the delete operation.
        """
        if prefix is not None and len(prefix) % self.block_ntokens != 0:
            return Status(StatusCodes.INVALID)

        for _, cache_key in self._cache_block_keys(prefix, tokens):
            await self._backend.delete(cache_key)
        return Status(StatusCodes.OK)

    def _cache_block_keys(
            self, prefix: Iterable[int] | None,
            tokens: Iterable[int]) -> Iterator[Tuple[Iterable[int], str]]:
        """Get the cache block keys of the kv tensors.
        Args:
            prefix (Iterable[int] | None): The prefix tokens of the kv tensors.
            tokens (Iterable[int]): The tokens of the kv tensors.
        Returns:
            The cache block keys of the kv tensors.
        """
        return iter(self.key_builder.build(prefix, tokens))

    def _cache_block_key_batchs(
        self, prefix: Iterable[int] | None, tokens: Iterable[int]
    ) -> Iterator[Iterator[Tuple[Iterable[int], str]]]:
        """Get the cache block key batchs.
        Args:
            prefix (Iterable[int] | None): The prefix tokens of the kv tensors.
            tokens (Iterable[int]): The tokens of the kv tensors.
        Returns:
            The cache block key batchs of the kv tensors.
        """
        return itertools.batched(self._cache_block_keys(prefix, tokens),
                                 self.op_batch)
