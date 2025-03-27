# SPDX-License-Identifier: Apache-2.0
import array
from typing import Iterable, Tuple

from .key_builder import KeyBuilder


class RawKeyBuilder(KeyBuilder):

    def __init__(self, block_size: int):
        super().__init__()
        self.block_size = block_size

    def build(
            self, prefix: Iterable[int] | None,
            tokens: Iterable[int]) -> Tuple[Tuple[Iterable[int], bytes], ...]:
        assert prefix is None or len(prefix) % self.block_size == 0

        token_size = len(tokens) - len(tokens) % self.block_size
        if token_size < self.block_size:
            return []

        results = []

        not_none_prefix = tuple() if prefix is None else tuple(prefix)
        all = tuple(not_none_prefix + tuple(tokens[:token_size]))
        assert len(all) % self.block_size == 0
        prefix_len = len(not_none_prefix)

        all_bytes = array.array('I', all).tobytes()
        itemsize = array.array('I').itemsize
        for i in range(0, token_size, self.block_size):
            keys = all[:prefix_len + i + self.block_size]
            block_bytes = all_bytes[:(prefix_len + i + self.block_size) *
                                    itemsize]
            results.append((keys, block_bytes))

        return results
