import copy
import pytest
import random
import torch

from .conftest import cache_conf_fixture, CACHE_DTYPE
from ..l1 import L1Cache
from ..memory import TensorPoolAllocator


def get_allocator(capacity, shape, dtype):
    mr_nbytes = torch.Size(shape).numel() * dtype.itemsize
    capacity_nbytes = capacity * mr_nbytes
    allocator = TensorPoolAllocator(capacity_nbytes=capacity_nbytes,
                                    mr_nbytes=mr_nbytes)
    return allocator


def test_cache_initialization(cache_conf_fixture):
    capacity = 10
    shape, spec = cache_conf_fixture
    cache = L1Cache(
        eviction_policy="LRU",
        capacity=capacity,
        allocator=get_allocator(capacity, shape, CACHE_DTYPE),
        block_spec=spec,
    )

    assert cache.capacity == capacity
    assert cache.block_shape == tuple(shape)


def test_put_and_get_aligned(cache_conf_fixture):
    capacity = 10
    shape, spec = cache_conf_fixture

    cache = L1Cache(
        eviction_policy="LRU",
        capacity=capacity,
        allocator=get_allocator(capacity, shape, CACHE_DTYPE),
        block_spec=spec,
    )

    tokens = [i for i in range(32)]
    origin_tokens = copy.deepcopy(tokens)
    shape[spec.block_shape_token_dim] = 32
    kv_tensors = torch.randn(*shape, dtype=CACHE_DTYPE)

    put_status = cache.put(None, tokens, kv_tensors)
    assert tokens == origin_tokens
    assert put_status.is_ok()
    assert put_status.value == kv_tensors.shape[spec.block_shape_token_dim]

    get_status = cache.get(None, tokens)
    assert tokens == origin_tokens
    assert get_status.is_ok()
    assert len(get_status.value) == 2
    cat = torch.cat(get_status.value, dim=spec.block_shape_token_dim)
    assert cat.shape == kv_tensors.shape
    assert torch.equal(cat, kv_tensors)


def test_put_and_get_unaligned(cache_conf_fixture):
    capacity = 10
    shape, spec = cache_conf_fixture

    cache = L1Cache(
        eviction_policy="LRU",
        capacity=capacity,
        allocator=get_allocator(capacity, shape, CACHE_DTYPE),
        block_spec=spec,
    )

    tokens = [i for i in range(35)]
    shape[spec.block_shape_token_dim] = len(tokens)
    kv_tensors = torch.randn(*shape, dtype=CACHE_DTYPE)

    put_status = cache.put(None, tokens, kv_tensors)
    assert put_status.is_ok()
    assert put_status.value == kv_tensors.shape[
        spec.block_shape_token_dim] // spec.block_ntokens * spec.block_ntokens

    get_status = cache.get(None, tokens)
    assert get_status.is_ok()
    assert len(get_status.value) == 2
    slices = [slice(None)] * len(shape)
    slices[spec.block_shape_token_dim] = slice(0, 32)
    assert torch.equal(
        torch.cat(get_status.value, dim=spec.block_shape_token_dim),
        kv_tensors[tuple(slices)])


@pytest.mark.parametrize("eviction_policy", ["FIFO", "LRU", "S3FIFO"])
def test_put_and_get_with_prefix(cache_conf_fixture, eviction_policy):
    capacity = 10
    shape, spec = cache_conf_fixture

    cache = L1Cache(
        eviction_policy=eviction_policy,
        capacity=capacity,
        allocator=get_allocator(capacity, shape, CACHE_DTYPE),
        block_spec=spec,
    )

    tokens0 = [i for i in range(32)]
    shape[spec.block_shape_token_dim] = len(tokens0)
    kv_tensors0 = torch.randn(*shape, dtype=CACHE_DTYPE)

    put_status = cache.put(None, tokens0, kv_tensors0)
    assert put_status.is_ok()
    assert put_status.value == kv_tensors0.shape[spec.block_shape_token_dim]

    tokens1 = [i for i in range(100, 135)]
    shape[spec.block_shape_token_dim] = len(tokens1)
    kv_tensors1 = torch.randn(*shape, dtype=CACHE_DTYPE)

    put_status = cache.put(tokens0, tokens1, kv_tensors1)
    assert put_status.is_ok()
    assert put_status.value == kv_tensors1.shape[
        spec.block_shape_token_dim] // spec.block_ntokens * spec.block_ntokens

    get_status = cache.get(None, tokens0)
    assert get_status.is_ok()
    assert torch.equal(
        torch.cat(get_status.value, dim=spec.block_shape_token_dim),
        kv_tensors0)

    get_status = cache.get(tokens0, tokens1)
    assert get_status.is_ok()
    slices = [slice(None)] * len(shape)
    slices[spec.block_shape_token_dim] = slice(0, 32)
    assert torch.equal(
        torch.cat(get_status.value, dim=spec.block_shape_token_dim),
        kv_tensors1[tuple(slices)])

    get_status = cache.get(None, tokens0 + tokens1)
    assert get_status.is_ok()
    chunks = torch.chunk(torch.cat(get_status.value,
                                   dim=spec.block_shape_token_dim),
                         2,
                         dim=spec.block_shape_token_dim)
    assert torch.equal(chunks[0], kv_tensors0)
    assert torch.equal(chunks[1], kv_tensors1[tuple(slices)])


@pytest.mark.parametrize("eviction_policy", ["FIFO", "LRU", "S3FIFO"])
def test_duplicated_puts(cache_conf_fixture, eviction_policy):
    capacity = 10
    shape, spec = cache_conf_fixture

    cache = L1Cache(
        eviction_policy=eviction_policy,
        capacity=capacity,
        allocator=get_allocator(capacity, shape, CACHE_DTYPE),
        block_spec=spec,
    )

    for _ in range(10):
        tokens = [i for i in range(32)]
        shape[spec.block_shape_token_dim] = len(tokens)
        kv_tensors = torch.randn(*shape, dtype=CACHE_DTYPE)

        put_status = cache.put(None, tokens, kv_tensors)
        assert put_status.is_ok()
        assert put_status.value == kv_tensors.shape[spec.block_shape_token_dim]

        get_status = cache.get(None, tokens)
        assert get_status.is_ok()
        assert torch.equal(
            torch.cat(get_status.value, dim=spec.block_shape_token_dim),
            kv_tensors)
        assert len(cache) == 2


@pytest.mark.parametrize("eviction_policy", ["FIFO", "LRU", "S3FIFO"])
def test_cache_eviction(cache_conf_fixture, eviction_policy):
    capacity = 10
    shape, spec = cache_conf_fixture

    cache = L1Cache(
        eviction_policy=eviction_policy,
        capacity=capacity,
        allocator=get_allocator(capacity, shape, CACHE_DTYPE),
        block_spec=spec,
    )

    for i in range(0, capacity, 2):
        tokens = [i * 64 + j for j in range(32)]
        shape[spec.block_shape_token_dim] = len(tokens)
        kv_tensors = torch.randn(*shape, dtype=CACHE_DTYPE)

        put_status = cache.put(None, tokens, kv_tensors)
        assert put_status.is_ok(), f"i={i}, len(cache)={len(cache)}"
        assert put_status.value == kv_tensors.shape[spec.block_shape_token_dim]
        assert len(cache) == (i // 2 + 1) * 2

    assert len(cache) == 10
    tokens = [640 + j for j in range(32)]
    shape[spec.block_shape_token_dim] = len(tokens)
    kv_tensors = torch.randn(*shape, dtype=CACHE_DTYPE)
    put_status = cache.put(None, tokens, kv_tensors)
    assert put_status.is_ok()
    assert put_status.value == kv_tensors.shape[spec.block_shape_token_dim]


@pytest.mark.parametrize("eviction_policy", ["FIFO", "LRU", "S3FIFO"])
def test_stress_cache(cache_conf_fixture, eviction_policy):
    capacity = 10000
    shape, spec = cache_conf_fixture

    cache = L1Cache(
        eviction_policy=eviction_policy,
        capacity=capacity,
        allocator=get_allocator(capacity, shape, CACHE_DTYPE),
        block_spec=spec,
    )

    query = {}
    for i in range(500):
        num_prefix_blocks = random.randint(0, 10)
        prefix_tokens = [
            j for j in range(num_prefix_blocks * spec.block_ntokens)
        ]
        shape[spec.block_shape_token_dim] = len(prefix_tokens)
        prefix_kv_tensors = torch.randn(*shape, dtype=CACHE_DTYPE)
        put_status = cache.put(None, prefix_tokens, prefix_kv_tensors)
        if put_status.is_out_of_memory():
            continue

        assert put_status.is_ok()
        assert put_status.value >= 0 and put_status.value <= prefix_kv_tensors.shape[
            spec.block_shape_token_dim]
        cache.get(None, prefix_tokens)

        ntokens = random.randint(16, 1024)
        tokens = [j for j in range(ntokens)]
        random.shuffle(tokens)
        shape[spec.block_shape_token_dim] = len(tokens)
        kv_tensors = torch.randn(*shape, dtype=CACHE_DTYPE)
        put_status = cache.put(prefix_tokens, tokens, kv_tensors)
        if put_status.is_out_of_memory():
            continue

        assert put_status.is_ok()
        assert put_status.value >= 0 and put_status.value <= kv_tensors.shape[
            spec.block_shape_token_dim]
        cache.get(prefix_tokens, tokens)
        query[i] = (prefix_tokens, tokens, kv_tensors)

    # check if fragmentation ratio is acceptable
    assert len(cache) > capacity * 0.8

    results = []
    for i in range(500):
        if i not in query:
            continue

        prefix_tokens, tokens, kv_tensors = query[i]
        j = 0
        while j < len(tokens):
            length = (random.randint(1,
                                     (len(tokens) - j) // spec.block_ntokens) *
                      spec.block_ntokens if len(tokens) -
                      j > spec.block_ntokens else spec.block_ntokens)

            get_status = cache.get(prefix_tokens, tokens[j:j + length])
            if get_status.is_ok():
                assert len(get_status.value) > 0
                slices = [slice(None)] * len(shape)
                slices[spec.block_shape_token_dim] = slice(
                    j, j + len(get_status.value) * spec.block_ntokens)
                assert torch.equal(
                    torch.cat(get_status.value,
                              dim=spec.block_shape_token_dim),
                    kv_tensors[tuple(slices)])
                results.append(1)
            else:
                results.append(0)
            prefix_tokens += tokens[j:j + length]
            j += length

    num_oks = sum(results)
    assert num_oks > 250
