import enum

from dataclasses import dataclass

from .spec import KVCacheBlockSpec


class KVCacheRetentionType(enum.Enum):
    """The retention type of the KV cache block.
    Args:
        STORE: Data is retained indefinitely, the upper layer is responsible for
               managing the life cycle of the data in the KV cache.
        CACHING: KV cache manages the life cycle of the data in the KV cache. When
                 kv cache is full, the data is removed from the kv cache based on
                 the eviction policy or the specified TTL.
    """
    STORE = enum.auto()
    CACHING = enum.auto()


@dataclass
class KVCacheConfig:
    """Configuration for the KV cache manager.

    Args:
        block_spec: The specification of the kv cache block.
    """
    block_spec: KVCacheBlockSpec
    retention_type: KVCacheRetentionType
