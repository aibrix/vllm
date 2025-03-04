from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, Iterable, TypeVar

from ...cache_handle import KVCacheHandle
from ...status import Status

K = TypeVar('K')
V = TypeVar('V')


@dataclass
class ConnectorFeature:
    """The features of the kv cache connector.
    Args:
        mput_mget: Whether the kv cache connector supports mput/mget
        prefetch: Whether the kv cache connector supports prefetch.
        zero_copy: Whether the kv cache connector supports zero-copy.
    """
    mput_mget: bool = False
    prefetch: bool = False
    zero_copy: bool = False


class Connector(Generic[K, V]):
    """Connector interface."""

    @staticmethod
    def create(
        backend_name: str,
        namespace: str,
        partition_id: str,
    ) -> "Connector":
        """Create a connector."""
        conn_id = f"{namespace}_{partition_id}"
        if backend_name == "ROCKSDB":
            from .rocksdb import RocksDBConnector
            return RocksDBConnector.from_envs(conn_id)
        else:
            raise ValueError(f"Unknown connector type: {backend_name}")

    @classmethod
    @abstractmethod
    def from_envs(cls, conn_id: str):
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
    async def get(self, key: K) -> Status[V]:
        """Get a value.
        Args:
            key: The key of the kv tensor.
        Returns:
            The value of the kv tensor.
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
        """MPut a list of key value pairs. This function is optional and only connectors
        have mput_mget feature enabled can implement this function.
        Args:
            keys: The keys of the kv tensors.
            values: The values of the kv tensors.
        Returns:
            List of statuses.
        """
        raise NotImplementedError

    async def acquire(self, key: K) -> Status[KVCacheHandle]:
        """Acquire a kv cache handle pointing to the kv tensors. This function is
        optional and only connectors have zero_copy feature enabled can implement
        this function.
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
