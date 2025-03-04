from dataclasses import dataclass

from .spec import KVCacheBlockSpec


@dataclass
class KVCacheConfig:
    """Configuration for the KV cache manager.

    Args:
        block_spec: The specification of the kv cache block.
    """
    block_spec: KVCacheBlockSpec
