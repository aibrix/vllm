# SPDX-License-Identifier: Apache-2.0
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from ... import envs
from ...common import AsyncBase
from ...memory import MemoryRegion
from ...status import Status, StatusCodes
from ...utils import bytes_to_tensor, tensor_to_bytes
from . import (Connector, ConnectorFeature, ConnectorRegisterDescriptor,
               ConnectorSGEntry)


@dataclass
class MockConfig:
    use_rdma: bool = False
    use_gather_scatter: bool = False


class MockRegisterDescriptor(ConnectorRegisterDescriptor):
    addr: int


@AsyncBase.async_wrap(exists="_exists",
                      get="_get",
                      put="_put",
                      delete="_delete")
class MockConnector(Connector[bytes, torch.Tensor], AsyncBase):
    """Mock connector."""

    def __init__(
        self,
        config: MockConfig,
        executor: Executor,
    ):
        super().__init__(executor)
        self.config = config
        self.store = None

    @classmethod
    def from_envs(cls, conn_id: str, executor: Executor) -> "MockConnector":
        """Create a connector from environment variables."""
        config = MockConfig(
            use_rdma=envs.VLLM_KV_CACHE_OL_MOCK_USE_RDMA,
            use_gather_scatter=envs.VLLM_KV_CACHE_OL_MOCK_USE_GATHER_SCATTER,
        )
        return cls(config, executor)

    @property
    def name(self) -> str:
        return "Mock"

    @property
    def feature(self) -> ConnectorFeature:
        feature = ConnectorFeature()
        if self.config.use_gather_scatter:
            feature.gather_scatter = True
        if self.config.use_rdma:
            feature.rdma = True
        return ConnectorFeature()

    @Status.capture_exception
    def open(self) -> Status:
        """Open a connection."""
        if self.store is None:
            self.store = {}
        return Status(StatusCodes.OK)

    @Status.capture_exception
    def close(self) -> Status:
        """Close a connection."""
        if self.store is not None:
            self.store.clear()
            self.store = None
        return Status(StatusCodes.OK)

    @Status.capture_exception
    def register_mr(self, addr: int, length: int):
        self.register_cache[addr] = length
        desc = MockRegisterDescriptor(addr=addr)
        return Status(value=desc)

    @Status.capture_exception
    def deregister_mr(self, desc: ConnectorRegisterDescriptor):
        assert desc.addr in self.register_cache
        del self.register_cache[desc.addr]

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
        for sge in sge_list:
            val = self.conn.get(self._key(sge.key), None)
            if val is None or len(val) == 0:
                return Status(StatusCodes.NOT_FOUND)
            arr = np.frombuffer(base_addr + sge.offset, block_size)
            arr[:] = np.frombuffer(val)

        return Status(StatusCodes.OK)

    @Status.capture_exception
    async def scatter(self, sge_list: Iterable[ConnectorSGEntry]) -> Status:
        base_addr = sge_list[0].base_addr
        block_size = sge_list[0].length
        for sge in sge_list:
            self.conn[self._key(sge.key)] = np.frombuffer(
                base_addr + sge.offset, block_size).tobytes()

        return Status(StatusCodes.OK)

    @Status.capture_exception
    def _exists(self, key: bytes) -> Status:
        """Check if key is in the store."""
        if key in self.store:
            return Status(StatusCodes.OK)
        return Status(StatusCodes.NOT_FOUND)

    @Status.capture_exception
    def _get(self, key: bytes) -> Status[torch.Tensor]:
        """Get a value."""
        val = self.store.get(key, None)
        if val is None:
            return Status(StatusCodes.NOT_FOUND)
        tensor = bytes_to_tensor(val)
        return Status(value=tensor)

    @Status.capture_exception
    def _put(self, key: bytes, value: torch.Tensor) -> Status:
        """Put a key value pair"""
        self.store[key] = tensor_to_bytes(value)
        return Status(StatusCodes.OK)

    @Status.capture_exception
    def _delete(self, key: bytes) -> Status:
        """Delete a key."""
        self.store.pop(key, None)
        return Status(StatusCodes.OK)
