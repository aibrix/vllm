import asyncio
import itertools
import logging
import torch

from typing import Iterable, Iterator, Tuple
from .connectors import Connector
from .key_builders import MD5Hasher, KeyBuilder, RollingHashKeyBuilder
from .marshallers import Marshaller, StringSerializer, TensorSerializer
from ..common import nvtx_range
from ..common.absl_logging import log_every_n_seconds, getLogger
from ..config import KVCacheRetentionType
from ..memory import MemoryRegion
from ..spec import KVCacheBlockLayout, KVCacheBlockSpec
from ..status import Status, StatusCodes

logger = getLogger(__name__)


class L2Cache:

    def __init__(
        self,
        backend_name: str,
        namespace: str,
        block_spec: KVCacheBlockSpec,
        retention_type: KVCacheRetentionType,
        key_builder: KeyBuilder | None = None,
        key_serializer: Marshaller | None = None,
        tensor_serializer: Marshaller | None = None,
        op_batch: int = 8,
    ) -> None:
        """Create a cache object.
        Args:
            backend_name (str): The name of cache backend.
            namespace (str): Namespace.
            block_spec (KVCacheBlockSpec): The block spec.
            retention_type (KVCacheRetentionType): The retention type of the cache.
            key_builder (KeyBuilder | None): Key builder for cache blocks.
            key_serializer (Marshaller | None): Key serializer
            tensor_serializer (Marshaller | None): Tensor serializer
            op_batch (int): The number of ops in a batch.
        """

        self.block_spec: KVCacheBlockSpec = block_spec
        self.retention_type: KVCacheRetentionType = retention_type
        self.block_layout: KVCacheBlockLayout = self.block_spec.block_layout
        self.block_shape: Tuple[int, ...] = self.block_spec.block_shape
        self.block_dtype: torch.dtype = self.block_spec.block_dtype
        self.block_ntokens: int = self.block_spec.block_ntokens
        self.key_builder: KeyBuilder = key_builder
        self.key_serializer: Marshaller = key_serializer
        self.tensor_serializer: Marshaller = tensor_serializer
        self.op_batch: int = op_batch
        self._backend: Connector = None

        cat_head_ids = "_".join(
            str(x) for x in self.block_spec.tensor_spec.heads)
        cat_layer_ids = "_".join(
            str(x) for x in self.block_spec.tensor_spec.layers)
        partition_id = f"h{cat_head_ids}_l{cat_layer_ids}"
        self._backend = Connector.create(backend_name, retention_type,
                                         namespace, partition_id)

        if self.key_builder is None:
            self.key_builder = RollingHashKeyBuilder(MD5Hasher(),
                                                     self.block_ntokens)
        if self.key_serializer is None:
            self.key_serializer = StringSerializer()
        if self.tensor_serializer is None:
            self.tensor_serializer = TensorSerializer()

        if self.block_layout == KVCacheBlockLayout.LND:
            raise NotImplementedError("LND layout is not supported yet.")

        if not self.block_spec.is_homogeneous():
            raise NotImplementedError(
                "Heterogeneous block shape is not supported yet.")

        logger.info(
            f"{str(self)} is initialized. Using partition_id={partition_id}.")

    def __repr__(self) -> str:
        return f"L2Cache(backend={self._backend.name})"

    def __str__(self) -> str:
        return self.__repr__()

    def __del__(self) -> None:
        self.close()
        logger.info(f"{str(self)} is closed.")

    def open(self) -> Status:
        """Open the cache."""
        return self._backend.open()

    def close(self) -> Status:
        """Close the cache."""
        if self._backend is not None:
            return self._backend.close()
        return Status(StatusCodes.OK)

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

        asyncio.gather(
            *(self._prefetch_impl(cache_key)
              for _, cache_key in self._cache_block_keys(prefix, tokens)),
            return_exceptions=False,  # backend returns exception as status
        )
        return Status(StatusCodes.OK)

    async def _prefetch_impl(self, cache_key: str) -> Status:
        return await self._backend.prefetch(
            self.key_serializer.marshal(cache_key))

    @nvtx_range("put", "kv_cache_ol.L2Cache")
    async def put(
        self,
        prefix: Iterable[int] | None,
        tokens: Iterable[int],
        kv_tensors: (torch.Tensor
                     | Tuple[torch.Tensor, torch.Tensor]
                     | MemoryRegion
                     | Tuple[MemoryRegion, MemoryRegion]),
    ) -> Status[int]:
        """Put kv tensors to the cache.
        Args:
            prefix (Iterable[int] | None): The prefix tokens of the kv tensors.
            tokens (Iterable[int]): The tokens of the kv tensors.
            kv_tensors: kv tensors or cache handles.
        Returns:
            The status of the put operation.
        """
        if isinstance(kv_tensors, tuple):
            raise NotImplementedError("Tuple kv tensors is not supported yet.")

        if prefix is not None and len(prefix) % self.block_ntokens != 0:
            return Status(StatusCodes.INVALID)

        if isinstance(kv_tensors, MemoryRegion):
            kv_tensors = kv_tensors.to_tensor().view(self.block_dtype).view(
                self.block_shape)

        if len(tokens) != kv_tensors.shape[0]:
            return Status(StatusCodes.INVALID)

        # If it is not a full block, we don't need to cache it.
        if len(tokens) // self.block_ntokens == 0:
            return Status(StatusCodes.OK, 0)

        num_tokens = len(tokens)
        num_blocks = num_tokens // self.block_ntokens

        blocks = torch.split(kv_tensors[0:num_blocks * self.block_ntokens],
                             self.block_ntokens)

        # TODO: use mput if backend's mput_mget feature is enabled.
        num_processed_tokens = 0
        block_idx = 0
        for key_batch in self._cache_block_key_batchs(prefix, tokens):
            tasks = []
            num_tokens_in_batch = len(key_batch) * self.block_ntokens
            async with asyncio.TaskGroup() as tg:
                for real_key, cache_key in key_batch:
                    block = blocks[block_idx]
                    block_idx += 1
                    tasks.append(
                        tg.create_task(
                            self._put_impl(real_key, cache_key, block)))

            if len(tasks) == 0:
                return Status(StatusCodes.ERROR)
            elif all(task.done() and task.result().is_ok() for task in tasks):
                # all success, continue to the next batch
                num_processed_tokens += num_tokens_in_batch
                continue
            elif num_processed_tokens > 0:
                # current batch is not the first one.
                # at least one batch is done successfully, return success.

                # delete current batch
                for _, cache_key in key_batch:
                    await self._delete_impl(cache_key)
                return Status(StatusCodes.OK, num_processed_tokens)
            else:
                # this is the first batch and at least one block in
                # current batch is failed, return error.

                # delete current batch
                for _, cache_key in key_batch:
                    await self._delete_impl(cache_key)

                failures = [
                    task for task in tasks
                    if task.done() and not task.result().is_ok()
                ]
                if len(failures) > 0:
                    return failures[0].result()
                return Status(StatusCodes.ERROR)

        return Status(StatusCodes.OK, num_processed_tokens)

    async def _put_impl(
        self,
        real_key: Iterable[int],
        cache_key: str,
        value: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
    ) -> Status:
        """Put kv tensors to the cache.
        Args:
            key (Iterable[int]): The key of the kv tensors.
            value (torch.Tensor | Tuple[torch.Tensor, torch.Tensor]): The kv tensors.
        Returns:
            The status of the put operation.
        """
        return await self._backend.put(
            self.key_serializer.marshal(cache_key),
            # cache value = (tokens, tensor)
            self.tensor_serializer.marshal((real_key, value)),
        )

    @nvtx_range("get", "kv_cache_ol.L2Cache")
    async def get(
        self,
        prefix: Iterable[int] | None,
        tokens: Iterable[int],
    ) -> Status[Iterable[torch.Tensor | Tuple[torch.Tensor, torch.Tensor]]]:
        """Get kv tensors from the cache.
        Args:
            prefix (Iterable[int] | None): The prefix tokens of the kv tensors.
            tokens (Iterable[int]): The tokens of the kv tensors.
        Returns:
            The kv tensors corresponding to the tokens.
        """
        if prefix is not None and len(prefix) % self.block_ntokens != 0:
            return Status(StatusCodes.INVALID)

        # TODO: use mget if backend's mput_mget feature is enabled.
        tensors = []
        for key_batch in self._cache_block_key_batchs(prefix, tokens):
            tasks = []
            async with asyncio.TaskGroup() as tg:
                for real_key, cache_key in key_batch:
                    tasks.append(
                        tg.create_task(self._get_impl(real_key, cache_key)))

            if len(tasks) == 0:
                break

            should_break = False
            for i in range(len(tasks)):
                if not tasks[i].done() or not tasks[i].result().is_ok():
                    should_break = True
                    break
                tensors.append(tasks[i].result().value)

            if should_break:
                break

        if len(tensors) == 0:
            return Status(StatusCodes.NOT_FOUND)

        return Status(value=tensors)

    async def _get_impl(self, real_key: Iterable[int],
                        cache_key: str) -> Status[torch.Tensor]:
        """Get kv tensors from the backend.
        Args:
            real_key (Iterable[int]): The key of the kv tensors.
            cache_key (str): The cache key of the kv tensors.
        Returns:
            The kv tensors corresponding to the key.
        """
        status = await self._backend.get(self.key_serializer.marshal(cache_key)
                                         )
        if status.is_ok():
            fetched_key, tensor = self.tensor_serializer.unmarshal(
                status.value)
            if real_key != fetched_key:
                # key not match
                return Status(StatusCodes.NOT_FOUND)
            return Status(
                value=tensor.view(self.block_dtype).view(self.block_shape))
        else:
            return Status(StatusCodes.NOT_FOUND)

    @nvtx_range("delte", "kv_cache_ol.L2Cache")
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
            await self._delete_impl(cache_key)
        return Status(StatusCodes.OK)

    async def _delete_impl(self, cache_key: str) -> Status:
        return await self._backend.delete(
            self.key_serializer.marshal(cache_key))

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
    ) -> Iterator[Iterator[Tuple[Iterable[int], Iterable[str]]]]:
        """Get the cache block key batchs.
        Args:
            prefix (Iterable[int] | None): The prefix tokens of the kv tensors.
            tokens (Iterable[int]): The tokens of the kv tensors.
        Returns:
            The cache block key batchs of the kv tensors.
        """
        return itertools.batched(self._cache_block_keys(prefix, tokens),
                                 self.op_batch)
