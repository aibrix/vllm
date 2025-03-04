from abc import abstractmethod
from typing import Callable, Dict, Hashable, Iterator, Generic, Tuple, TypeVar
from ...memory import RefCountedObj
from ...status import Status, StatusCodes

N = TypeVar('N')
V = TypeVar('V')

Functor = Callable[
    [Hashable, V],
    None,
]


class BaseEvictionPolicyNode(Generic[V]):
    __slots__ = ("key", "value", "hotness")

    def __init__(self, key: Hashable, value: V | None):
        self.key: Hashable = key
        self.value: V = value
        self.hotness: int = 0

    def __repr__(self) -> str:
        derived_members = ", ".join(
            {f"{slot}={getattr(self, slot, None)}"
             for slot in self.__slots__})
        return f"Node(key={self.key}, value={self.value}, hotness={self.hotness}, {derived_members})"

    def __str__(self) -> str:
        return self.__repr__()


class BaseEvictionPolicy(Generic[N, V]):
    """Base class for eviction policies."""

    def __init__(
        self,
        name: str,
        capacity: int,
        evict_size: int = 1,
        on_put: Functor | None = None,
        on_evict: Functor | None = None,
        on_hot_access: Functor | None = None,
    ) -> None:
        """Initialize the eviction policy.
        Args:
            name (str): The name of the eviction policy.
            capacity(int): The capacity of the eviction policy in terms of number of items.
            evict_size (int, optional): The number of items to evict at a time. Defaults to 1.
            on_put (Functor, optional): The put function to call when putting new items. Defaults to None.
            on_evict (Functor, optional): The evict function to call when evicting items. Defaults to None.
            on_hot_access (Functor, optional): The callback function to call when a cache item becomes hot.
                                               Defaults to None.
        """

        self._name: str = name
        self._capacity: int = capacity
        self._evict_size: int = evict_size
        self._on_put: Functor = on_put
        self._on_evict: Functor = on_evict
        self._on_hot_access: Functor = on_hot_access

        self._hashmap: Dict[Hashable, N[V]] = {}

    @staticmethod
    def create(name: str, *args, **kwargs) -> 'BaseEvictionPolicy':
        """Return the eviction policy with the given name."""
        if name == "LRU":
            from .lru import LRU
            return LRU(*args, **kwargs)
        elif name == "FIFO":
            from .fifo import FIFO
            return FIFO(*args, **kwargs)
        elif name == "S3FIFO":
            from .s3fifo import S3FIFO
            return S3FIFO(*args, **kwargs)
        else:
            raise ValueError(f"Unknown eviction policy: {name}")

    @property
    def name(self) -> str:
        """Return the name of the eviction policy."""
        return self._name

    @property
    def evict_size(self) -> int:
        """Return the number of items to evict at a time."""
        return self._evict_size

    @property
    def capacity(self) -> int:
        """Return the capacity of the eviction policy in terms of number of items."""
        return self._capacity

    def __del__(self) -> None:
        for _, node in self._hashmap.items():
            if isinstance(node.value, RefCountedObj):
                node.value.ref_down()

    def __len__(self) -> int:
        """Return the number of items in the eviction policy."""
        return len(self._hashmap)

    def __contains__(self, key: Hashable) -> bool:
        """Return True if the key is in the eviction policy."""
        return key in self._hashmap

    def __getitem__(self, key: Hashable) -> V:
        """Return the value of the key."""
        value = self.get(key, None)
        if value is None:
            raise KeyError(key)
        return value

    def __setitem__(self, key: Hashable, value: V) -> None:
        """Set the value of the key."""
        self.put(key, value)

    def __delitem__(self, key: Hashable) -> None:
        """Delete the key."""
        self.delete(key)

    def __iter__(self) -> Iterator[Hashable]:
        """Return an iterator over the keys in the eviction policy."""
        return iter(self._hashmap.keys())

    def set_on_put_callback(self, functor: Functor) -> None:
        """Set the callback function to call when putting new items."""
        self._on_put = functor

    def set_on_evict_callback(self, functor: Functor) -> None:
        """Set the callback function to call when evicting items."""
        self._on_evict = functor

    def set_on_hot_access_callback(self, functor: Functor) -> None:
        """Set the callback function to call when a cache item becomes hot."""
        self._on_hot_access = functor

    def items(self) -> Iterator[Tuple[Hashable, V]]:
        """Return an iterator over the key-value pairs in the eviction policy."""
        return iter({(key, node.value) for key, node in self._hashmap.items()})

    def keys(self) -> Iterator[Hashable]:
        """Return an iterator over the keys in the eviction policy."""
        return iter(self._hashmap.keys())

    def values(self) -> Iterator[V]:
        """Return an iterator over the values in the eviction policy."""
        return iter({node.value for node in self._hashmap.values()})

    def __repr__(self) -> str:
        return f"{self._name}(capacity={self._capacity}, size={len(self)})"

    def __str__(self) -> str:
        return self.__repr__()

    @abstractmethod
    def put(self, key: Hashable, value: V) -> Status:
        """Put a key into the eviction policy.
        Args:
            key (Hashable): The key of the item.
            value: The value of the item.
        Returns:
            Status: The status of the operation.
        """
        raise NotImplementedError

    @abstractmethod
    def get(
        self,
        key: Hashable,
    ) -> Status[V]:
        """Get the value of key from the eviction policy.
        Args:
            key (Hashable): The key of the item.
        Returns:
            Status: The status of the operation.
        """
        raise NotImplementedError

    @abstractmethod
    def peak(
        self,
        key: Hashable,
    ) -> Status[V]:
        """Peak the value of key from the eviction policy. Peak does not update the eviction policy.
        Args:
            key (Hashable): The key of the item.
        Returns:
            Status: The status of the operation.
        """
        if key in self._hashmap:
            node = self._hashmap[key]
            return Status(value=node.value)
        return Status(StatusCodes.NOT_FOUND)

    @abstractmethod
    def delete(self, key: Hashable) -> Status:
        """Delete a key-value pair from the eviction policy.
        Args:
            key (Hashable): The key of the item.
        Returns:
            Status: The status of the operation.
        """
        raise NotImplementedError

    @abstractmethod
    def evict(self, size: int = 1) -> Status:
        """Evict a key-value pair from the eviction policy.
        Args:
            size (int, optional): The number of items to evict. Defaults to 1.
        """
        raise NotImplementedError

    @abstractmethod
    def assert_consistency(self) -> None:
        """Check the consistency of the eviction policy. Only for test purpose."""
        raise NotImplementedError
