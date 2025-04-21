# SPDX-License-Identifier: Apache-2.0
from abc import abstractmethod
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import Generic, Iterable, TypeVar

from ...cache_handle import KVCacheHandle
from ...memory import MemoryRegion
from ...status import Status

K = TypeVar("K")
V = TypeVar("V")


@dataclass
class ConnectorFeature:
    """The features of the kv cache connector.
    Args:
        mput_mget: Whether the kv cache connector supports mput/mget
        prefetch: Whether the kv cache connector supports prefetch.
        acquire: Whether the kv cache connector supports acquire.
        rdma: Whether the kv cache connector supports RDMA.
        gather_scatter: Whether the kv cache connector supports gather_scatter.
    """

    mput_mget: bool = False
    prefetch: bool = False
    acquire: bool = False
    rdma: bool = False
    gather_scatter: bool = False


@dataclass
class ConnectorRegisterDescriptor:
    """The register descriptor"""
    pass


@dataclass
class ConnectorSGEntry:
    """The entry of scatter-gather list."""
    key: K
    base_addr: int
    offset: int
    length: int


class Connector(Generic[K, V]):
    """Connector interface."""

    @staticmethod
    def create(
        backend_name: str,
        namespace: str,
        partition_id: str,
        executor: Executor,
    ) -> "Connector":
        """Create a connector."""
        conn_id = f"{namespace}_{partition_id}"
        if backend_name == "ROCKSDB":
            from .rocksdb import RocksDBConnector

            return RocksDBConnector.from_envs(conn_id, executor)
        elif backend_name == "INFINISTORE":
            from .infinistore import InfiniStoreConnector

            return InfiniStoreConnector.from_envs(conn_id, executor)
        elif backend_name == "MOCK":
            from .mock import MockConnector

            return MockConnector.from_envs(conn_id, executor)
        else:
            raise ValueError(f"Unknown connector type: {backend_name}")

    @classmethod
    @abstractmethod
    def from_envs(cls, conn_id: str, executor: Executor):
        """Create a connector from environment variables."""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def feature(self) -> ConnectorFeature:
        """Get the feature of the connector.
        Returns:
            The feature of the kv cache service.
        """
        raise NotImplementedError

    @abstractmethod
    def open(self) -> Status:
        """Open a connection."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> Status:
        """Close a connection."""
        raise NotImplementedError

    async def prefetch(self, keys: Iterable[K]) -> None:
        """Prefetch a list of keys.
        Args:
            keys: The keys of the kv tensors.
        """
        pass

    @abstractmethod
    async def exists(self, key: K) -> Status:
        """Check if key is in the store."""
        raise NotImplementedError

    @abstractmethod
    async def get(self, key: K, mr: MemoryRegion = None) -> Status[V]:
        """Get a value.
        Args:
            key: The key of the kv tensor.
            mr: The memory region to place the fetched kv tensor. Only
                backends support RDMA will use this parameter.
        Returns:
            The fetched kv tensor.
        """
        raise NotImplementedError

    @abstractmethod
    async def put(self, key: K, value: V) -> Status:
        """Put a key value pair.
        Args:
            key: The key of the kv cache.
            value: The value of the kv cache.
        Returns:
            The status of the put operation.
        """
        raise NotImplementedError

    def register_mr(self, addr: int,
                    length: int) -> Status[ConnectorRegisterDescriptor]:
        """Register an memory region with backend-specific register function.
        Args:
            addr: memory region's address
            length: memory region's length
        Returns:
            Status of the register operation.
            The register descriptor.
        """
        raise NotImplementedError

    def deregister_mr(self, desc: ConnectorRegisterDescriptor) -> Status:
        """Deregister an memory region.
        Args:
            desc: the register descriptor returned by `register_mr`.
        Returns:
            Status of the deregister operation.
        """
        raise NotImplementedError

    def get_sge_list(
            self, keys: Iterable[K], mrs: Iterable[MemoryRegion]
    ) -> Iterable[Iterable[ConnectorSGEntry]]:
        """Convert a list of keys and mrs to a list of scatter-gather entries.
        The upper-layer will call gather/scatter on each returned list of
        entries.
        This function is optional and only connectors have gather_scatter
        feature enabled can implement this function.
        Args:
            keys: The keys of the kv tensors.
            mrs: The memory regions to gather/scatter.
        """
        raise NotImplementedError

    async def gather(
            self,
            sge_list: Iterable[ConnectorSGEntry]) -> Status | Iterable[Status]:
        """Gather a list of values. This function is optional and only
        connectors have gather_scatter feature enabled can implement this
        function.
        Args:
            sge_list: A list of scatter-gather entries.
        Returns:
            Status of the gather operation.
            Or, a list of operation status on each entry.
        """
        raise NotImplementedError

    async def scatter(
            self,
            sge_list: Iterable[ConnectorSGEntry]) -> Status | Iterable[Status]:
        """Scatter a list of values. This function is optional and only
        connectors have gather_scatter feature enabled can implement this
        function.
        Args:
            sge_list: A list of scatter-gather entries.
        Returns:
            Status of the scatter operation.
            Or, a list of operation status on each entry.
        """
        raise NotImplementedError

    async def mget(self, keys: Iterable[K]) -> Iterable[Status[V]]:
        """MGet a list of values. This function is optional and only connectors
        have mput_mget feature enabled can implement this function.
        Args:
            keys: The keys of the kv tensors.
        Returns:
            List of values.
        """
        raise NotImplementedError

    async def mput(self, keys: Iterable[K],
                   values: Iterable[V]) -> Iterable[Status]:
        """MPut a list of key value pairs. This function is optional and only
        connectors have mput_mget feature enabled can implement this function.
        Args:
            keys: The keys of the kv tensors.
            values: The values of the kv tensors.
        Returns:
            List of statuses.
        """
        raise NotImplementedError

    async def acquire(self, key: K) -> Status[KVCacheHandle]:
        """Acquire a kv cache handle pointing to the kv tensors. This function
        is optional and only connectors have acquire feature enabled can
        implement this function.
        Args:
            key: The key of the kv cache.
        Returns:
            The kv cache handle.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete(self, key: K) -> Status:
        """Delete a key.
        Args:
            key: The key of the kv cache.
        Returns:
            The status of the delete operation.
        """
        raise NotImplementedError
