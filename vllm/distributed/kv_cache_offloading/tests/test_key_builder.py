# SPDX-License-Identifier: Apache-2.0
import random
from typing import Iterable

import pytest

from ..l2.key_builders import (HexKeyBuilder, MD5Hasher, RawKeyBuilder,
                               RollingHashKeyBuilder, SimpleHashKeyBuilder)

# Fixed tokens length
TOKENS_LENGTH = 512
BLOCK_SIZE = 16


@pytest.fixture(params=[
    HexKeyBuilder(BLOCK_SIZE),
    RawKeyBuilder(BLOCK_SIZE),
    RollingHashKeyBuilder(MD5Hasher(), BLOCK_SIZE),
    SimpleHashKeyBuilder(MD5Hasher(), BLOCK_SIZE),
])
def key_builder(request):
    return request.param


@pytest.fixture(params=[512, 4 * 1024, 32 * 1024])
def prefix_length(request):
    return request.param


def test_key_builder(benchmark, key_builder, prefix_length):
    prefix = [random.randint(0, 99999999) for _ in range(prefix_length)]
    tokens = [random.randint(0, 99999999) for _ in range(TOKENS_LENGTH)]

    # Run benchmark
    benchmark(key_builder.build, prefix, tokens)

    result = key_builder.build(prefix, tokens)
    assert len(result) > 0
    for key_tuple in result:
        assert len(key_tuple) == 2
        assert isinstance(key_tuple[0], Iterable)
        assert isinstance(key_tuple[1], (str, bytes))
