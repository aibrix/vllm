# SPDX-License-Identifier: Apache-2.0
from concurrent.futures import Executor
from dataclasses import dataclass
from threading import Lock
from typing import Iterable, Tuple

import torch

from ... import envs
from ...common import AsyncBase
from ...memory import MemoryRegion
from ...status import Status, StatusCodes
from . import Connector, ConnectorFeature, ConnectorRegisterDescriptor


@dataclass
class MockConfig:
    use_rdma: bool = False
    use_mput_mget: bool = False


class MockRegisterDescriptor(ConnectorRegisterDescriptor):
    addr: int


@AsyncBase.async_wrap(exists="_exists",
                      get="_get",
                      put="_put",
                      delete="_delete")
class MockConnector(Connector[str, torch.Tensor], AsyncBase):
    """Mock connector."""

    def __init__(
        self,
        config: MockConfig,
        executor: Executor,
    ):
        super().__init__(executor)
        self.config = config
        self.lock = Lock()
        self.store = None

    @classmethod
    def from_envs(cls, conn_id: str, executor: Executor) -> "MockConnector":
        """Create a connector from environment variables."""
        config = MockConfig(
            use_rdma=envs.AIBRIX_KV_CACHE_OL_MOCK_USE_RDMA,
            use_mput_mget=envs.AIBRIX_KV_CACHE_OL_MOCK_USE_MPUT_MGET,
        )
        return cls(config, executor)

    @property
    def name(self) -> str:
        return "Mock"

    @property
    def feature(self) -> ConnectorFeature:
        feature = ConnectorFeature()
        if self.config.use_mput_mget:
            feature.mput_mget = True
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
    def register_mr(self, addr: int,
                    length: int) -> Status[ConnectorRegisterDescriptor]:
        self.register_cache[addr] = length
        desc = MockRegisterDescriptor(addr=addr)
        return Status(value=desc)

    @Status.capture_exception
    def deregister_mr(self, desc: ConnectorRegisterDescriptor) -> Status:
        assert desc.addr in self.register_cache
        del self.register_cache[desc.addr]
        return Status(StatusCodes.OK)

    def get_batches(
        self,
        keys: Iterable[str],
        mrs: Iterable[MemoryRegion],
        batch_size: int,
    ) -> Iterable[Iterable[Tuple[str, MemoryRegion]]]:
        lists = []
        for key, mr in zip(keys, mrs):
            if len(lists) == 0 or len(lists[-1]) >= batch_size:
                lists.append([(key, mr)])
            else:
                lists[-1].append((key, mr))
        return lists

    @Status.capture_exception
    async def mget(self, keys: Iterable[str],
                   mrs: Iterable[MemoryRegion]) -> Iterable[Status]:
        statuses = []
        for i, mr in enumerate(mrs):
            statuses.append(self._get(keys[i], mr))
        return statuses

    @Status.capture_exception
    async def mput(self, keys: Iterable[str],
                   mrs: Iterable[MemoryRegion]) -> Iterable[Status]:
        statuses = []
        for i, mr in enumerate(mrs):
            statuses.append(self._put(keys[i], mr))
        return statuses

    @Status.capture_exception
    def _exists(self, key: str) -> Status:
        """Check if key is in the store."""
        with self.lock:
            if key in self.store:
                return Status(StatusCodes.OK)
        return Status(StatusCodes.NOT_FOUND)

    @Status.capture_exception
    def _get(self, key: str, mr: MemoryRegion) -> Status:
        """Get a value."""
        with self.lock:
            val = self.store.get(key, None)
        if val is None:
            return Status(StatusCodes.NOT_FOUND)
        mr.fill(val)
        return Status(StatusCodes.OK)

    @Status.capture_exception
    def _put(self, key: str, mr: MemoryRegion) -> Status:
        """Put a key value pair"""
        with self.lock:
            self.store[key] = mr.tobytes()
        return Status(StatusCodes.OK)

    @Status.capture_exception
    def _delete(self, key: str) -> Status:
        """Delete a key."""
        with self.lock:
            self.store.pop(key, None)
        return Status(StatusCodes.OK)
