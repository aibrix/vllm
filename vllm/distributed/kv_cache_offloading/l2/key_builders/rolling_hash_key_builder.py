import struct

from typing import Iterable, Tuple

from .hasher import Hasher
from .key_builder import KeyBuilder


class RollingHashKeyBuilder(KeyBuilder):

    def __init__(self, hasher: Hasher, block_size: int):
        super().__init__()
        self.hasher = hasher
        self.block_size = block_size

    def build(self, prefix: Iterable[int] | None,
              tokens: Iterable[int]) -> Tuple[str, ...]:
        assert prefix is None or len(prefix) % self.block_size == 0

        token_size = len(tokens) - len(tokens) % self.block_size
        if token_size < self.block_size:
            return []

        results = []
        prev_hash = None

        if prefix is not None:
            for i in range(0, len(prefix), self.block_size):
                candidates = [prev_hash] if i > 0 else []
                candidates.extend(prefix[i:i + self.block_size])

                # Split into low 64 bits and high 64 bits
                split_candidates = [(c & 0xFFFFFFFFFFFFFFFF,
                                     (c >> 64) & 0xFFFFFFFFFFFFFFFF)
                                    for c in candidates]

                # Convert to byte representation
                data = b"".join(
                    struct.pack("QQ", low, high)
                    for low, high in split_candidates)

                prev_hash = self.hasher.hash(data)

        for i in range(0, token_size, self.block_size):
            keys = [x for x in prefix] if prefix is not None else []
            candidates = [prev_hash] if prev_hash is not None else []
            candidates.extend(tokens[i:i + self.block_size])
            keys.extend(tokens[0:i + self.block_size])

            # Split into low 64 bits and high 64 bits
            split_candidates = [(c & 0xFFFFFFFFFFFFFFFF,
                                 (c >> 64) & 0xFFFFFFFFFFFFFFFF)
                                for c in candidates]

            # Convert to byte representation
            data = b"".join(
                struct.pack("QQ", low, high) for low, high in split_candidates)

            curr_hash = self.hasher.hash(data)

            # Format hash as a 32-character hexadecimal string
            hash_hex = f"{curr_hash:032x}"
            results.append((keys, hash_hex))
            prev_hash = curr_hash

        return results
