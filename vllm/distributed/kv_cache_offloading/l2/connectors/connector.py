# SPDX-License-Identifier: Apache-2.0
from abc import abstractmethod
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import Generic, Iterable, Tuple, TypeVar

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
        rdma: Whether the kv cache connector supports RDMA.
    """

    mput_mget: bool = False
    prefetch: bool = False
    rdma: bool = False


@dataclass
class ConnectorRegisterDescriptor:
    """The register descriptor"""
    pass


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
        elif backend_name == "HPKV":
            from .hpkv import HPKVConnector

            return HPKVConnector.from_envs(conn_id, executor)
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
    async def get(self, key: K, mr: MemoryRegion) -> Status:
        """Get a value.
        Args:
            key: The key of the kv tensor.
            mr: The memory region to place the fetched kv tensor.
        Returns:
            The status of the get operation.
        """
        raise NotImplementedError

    @abstractmethod
    async def put(self, key: K, mr: MemoryRegion) -> Status:
        """Put a key value pair.
        Args:
            key: The key of the kv cache.
            mr: The memory region holding the kv tensors.
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

    def get_batches(
        self,
        keys: Iterable[K],
        mrs: Iterable[MemoryRegion],
        batch_size: int,
    ) -> Iterable[Iterable[Tuple[K, MemoryRegion]]]:
        """Get a list of key MR batches that is used for mput and mget
        operations.

        Args:
            keys: The keys of the kv tensors.
            mrs: Memory regions holding the kv tensors.
            batch_size: The maximum number of key MR pairs in a batch.
        Returns:
            List of key MR batches.
        """
        raise NotImplementedError

    async def mget(self, keys: Iterable[K],
                   mrs: Iterable[MemoryRegion]) -> Iterable[Status]:
        """MGet a list of values. This function is optional and only connectors
        have mput_mget feature enabled can implement this function.
        Args:
            keys: The keys of the kv tensors.
            mrs: Memory regions to hold the fetched kv tensors.
        Returns:
            List of statuses.
        """
        raise NotImplementedError

    async def mput(self, keys: Iterable[K],
                   mrs: Iterable[MemoryRegion]) -> Iterable[Status]:
        """MPut a list of key value pairs. This function is optional and only
        connectors have mput_mget feature enabled can implement this function.
        Args:
            keys: The keys of the kv tensors.
            mrs: Memory regions holding the kv tensors.
        Returns:
            List of statuses.
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
