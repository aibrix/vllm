# SPDX-License-Identifier: Apache-2.0
import functools
from concurrent.futures import Executor
from typing import Iterable

import infinistore
import torch

from ... import envs
from ...common import AsyncBase
from ...memory import MemoryRegion
from ...status import Status, StatusCodes
from . import (Connector, ConnectorFeature, ConnectorRegisterDescriptor,
               ConnectorSGEntry)


@AsyncBase.async_wrap(delete="_delete")
class InfiniStoreConnector(Connector[bytes, torch.Tensor], AsyncBase):
    """InfiniStore connector."""

    def __init__(
        self,
        config: infinistore.ClientConfig,
        key_suffix: str,
        executor: Executor,
    ):
        super().__init__(executor)
        self.config = config
        self.key_suffix = key_suffix
        self.conn = None

    @classmethod
    def from_envs(cls, conn_id: str,
                  executor: Executor) -> "InfiniStoreConnector":
        """Create a connector from environment variables."""
        config = infinistore.ClientConfig(
            host_addr=envs.VLLM_KV_CACHE_OL_INFINISTORE_HOST_ADDR,
            service_port=envs.VLLM_KV_CACHE_OL_INFINISTORE_SERVICE_PORT,
            log_level="info",
            connection_type=envs.VLLM_KV_CACHE_OL_INFINISTORE_CONNECTION_TYPE,
            ib_port=envs.VLLM_KV_CACHE_OL_INFINISTORE_IB_PORT,
            link_type=envs.VLLM_KV_CACHE_OL_INFINISTORE_LINK_TYPE,
            dev_name=envs.VLLM_KV_CACHE_OL_INFINISTORE_DEV_NAME,
        )
        return cls(config, conn_id, executor)

    @property
    def name(self) -> str:
        return "InfiniStore"

    @property
    def feature(self) -> ConnectorFeature:
        feature = ConnectorFeature()
        if self.config is not None and \
            self.config.connection_type is infinistore.TYPE_RDMA:
            feature.gather_scatter = True
            feature.rdma = True
        return feature

    def _key(self, key: bytes) -> str:
        return key.hex() + self.key_suffix

    @Status.capture_exception
    def open(self) -> Status:
        """Open a connection."""
        if self.conn is None:
            self.conn = infinistore.InfinityConnection(self.config)
            self.conn.connect()
        return Status(StatusCodes.OK)

    @Status.capture_exception
    def close(self) -> Status:
        """Close a connection."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None
        return Status(StatusCodes.OK)

    @Status.capture_exception
    def register_mr(self, addr: int, length: int):
        ret = self.conn.register_mr(addr, length)
        if ret != 0:
            return Status(StatusCodes.INVALID)
        return Status(value=ConnectorRegisterDescriptor)

    @Status.capture_exception
    def deregister_mr(self, desc: ConnectorRegisterDescriptor):
        # InfiniStore does not expose deregister function
        pass

    @Status.capture_exception
    async def exists(self, key: bytes) -> Status:
        """Check if key is in the store."""
        if self.conn.check_exist(self._key(key)):
            return Status(StatusCodes.OK)
        return Status(StatusCodes.NOT_FOUND)

    def get_sge_list(
            self, keys: Iterable[bytes], mrs: Iterable[MemoryRegion]
    ) -> Iterable[Iterable[ConnectorSGEntry]]:
        lists = []
        for key, mr in zip(keys, mrs):
            if len(lists) == 0 or lists[-1][0].base_addr != mr.slab.data_ptr():
                lists.append([
                    ConnectorSGEntry(key, mr.slab.data_ptr(), mr.addr,
                                     mr.length)
                ])
            else:
                lists[-1].append(
                    ConnectorSGEntry(key, mr.slab.data_ptr(), mr.addr,
                                     mr.length))
        return lists

    @Status.capture_exception
    async def gather(self, sge_list: Iterable[ConnectorSGEntry]) -> Status:
        base_addr = sge_list[0].base_addr
        block_size = sge_list[0].length
        blocks = [None] * len(sge_list)
        for i, sge in enumerate(sge_list):
            blocks[i] = (self._key(sge.key), sge.offset)

        try:
            await self.conn.rdma_read_cache_async(blocks, block_size,
                                                  base_addr)
        except infinistore.InfiniStoreKeyNotFound:
            return Status(StatusCodes.NOT_FOUND)
        return Status(StatusCodes.OK)

    @Status.capture_exception
    async def scatter(self, sge_list: Iterable[ConnectorSGEntry]) -> Status:
        base_addr = sge_list[0].base_addr
        block_size = sge_list[0].length
        blocks = [None] * len(sge_list)
        for i, sge in enumerate(sge_list):
            blocks[i] = (self._key(sge.key), sge.offset)

        await self.conn.rdma_write_cache_async(blocks, block_size, base_addr)
        return Status(StatusCodes.OK)

    @Status.capture_exception
    async def get(self,
                  key: bytes,
                  mr: MemoryRegion = None) -> Status[torch.Tensor]:
        """Get a value."""
        if self.config.connection_type is infinistore.TYPE_RDMA:
            assert mr is not None
            return await self._rdma_get(key, mr)
        else:
            assert mr is None
            tcp_get = functools.partial(self._tcp_get, key)
            return await self.event_loop.run_in_executor(
                self._executor, tcp_get)

    def _tcp_get(self, key: bytes) -> Status[torch.Tensor]:
        """Get a value via TCP."""
        val = self.conn.tcp_read_cache(self._key(key))
        if val is None or len(val) == 0:
            return Status(StatusCodes.NOT_FOUND)
        return Status(value=torch.from_numpy(val))

    async def _rdma_get(self, key: bytes, mr: MemoryRegion) -> Status:
        """Get a value via RDMA."""
        try:
            await self.conn.rdma_read_cache_async([(self._key(key), 0)],
                                                  mr.length, mr.data_ptr())
        except infinistore.InfiniStoreKeyNotFound:
            return Status(StatusCodes.NOT_FOUND)
        return Status(StatusCodes.OK)

    @Status.capture_exception
    async def put(self, key: bytes, value: torch.Tensor) -> Status:
        """Put a key value pair"""
        if self.config.connection_type is infinistore.TYPE_RDMA:
            return await self._rdma_put(key, value)
        else:
            tcp_put = functools.partial(self._tcp_put, key, value)
            return await self.event_loop.run_in_executor(
                self._executor, tcp_put)

    async def _rdma_put(self, key: bytes, value: torch.Tensor) -> Status:
        """Put a value via RDMA."""
        nbytes = value.numel() * value.element_size()
        await self.conn.rdma_write_cache_async([(self._key(key), 0)], nbytes,
                                               value.data_ptr())
        return Status(StatusCodes.OK)

    def _tcp_put(self, key: bytes,
                 value: torch.Tensor) -> Status[torch.Tensor]:
        """Put a value via TCP."""
        nbytes = value.numel() * value.element_size()
        self.conn.tcp_write_cache(self._key(key), value.data_ptr(), nbytes)
        return Status(StatusCodes.OK)

    @Status.capture_exception
    def _delete(self, key: bytes) -> Status:
        """Delete a key."""
        self.conn.delete_keys(self._key(key))
        return Status(StatusCodes.OK)
