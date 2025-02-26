import copy
import pytest
import random
import torch

from ..l1 import L1Cache
from ..memory import TensorPoolAllocator
from ..spec import KVCacheBlockLayout, KVCacheBlockSpec, KVCacheTensorSpec, KVCacheLayerSpec


def get_allocator(capacity, shape, dtype):
    mr_nbytes = torch.Size(shape).numel() * dtype.itemsize
    capacity_nbytes = capacity * mr_nbytes
    allocator = TensorPoolAllocator(capacity_nbytes=capacity_nbytes,
                                    mr_nbytes=mr_nbytes)
    return allocator


def get_block_spec(shape, dtype):
    return KVCacheBlockSpec(
        block_ntokens=shape[0],
        block_dtype=dtype,
        block_layout=KVCacheBlockLayout.NLD,
        tensor_spec=KVCacheTensorSpec(
            heads=[1, 2],
            layers=list(range(shape[1])),
            layer_specs=[
                KVCacheLayerSpec(size=shape[2]) for _ in range(shape[1])
            ],
        ),
    )


def test_cache_initialization():
    capacity = 10
    shape = (16, 8, 64)
    dtype = torch.bfloat16
    cache = L1Cache(
        eviction_policy="LRU",
        capacity=capacity,
        allocator=get_allocator(capacity, shape, dtype),
        block_spec=get_block_spec(shape, dtype),
    )

    assert cache.capacity == capacity
    assert cache.block_shape == shape


def test_put_and_get_aligned():
    capacity = 10
    shape = (16, 8, 64)
    dtype = torch.bfloat16

    cache = L1Cache(
        eviction_policy="LRU",
        capacity=capacity,
        allocator=get_allocator(capacity, shape, dtype),
        block_spec=get_block_spec(shape, dtype),
    )

    tokens = [i for i in range(32)]
    origin_tokens = copy.deepcopy(tokens)
    kv_tensors = torch.randn(32, 8, 64, dtype=torch.bfloat16)

    put_status = cache.put(None, tokens, kv_tensors)
    assert tokens == origin_tokens
    assert put_status.is_ok()
    assert put_status.value == kv_tensors.shape[0]

    get_status = cache.get(None, tokens)
    assert tokens == origin_tokens
    assert get_status.is_ok()
    assert len(get_status.value) == 2
    assert torch.equal(torch.cat(get_status.value), kv_tensors)


def test_put_and_get_unaligned():
    capacity = 10
    shape = (16, 8, 64)
    dtype = torch.bfloat16

    cache = L1Cache(
        eviction_policy="LRU",
        capacity=capacity,
        allocator=get_allocator(capacity, shape, dtype),
        block_spec=get_block_spec(shape, dtype),
    )

    tokens = [i for i in range(35)]
    kv_tensors = torch.randn(len(tokens), 8, 64, dtype=torch.bfloat16)

    put_status = cache.put(None, tokens, kv_tensors)
    assert put_status.is_ok()
    assert put_status.value == kv_tensors.shape[0] // shape[0] * shape[0]

    get_status = cache.get(None, tokens)
    assert get_status.is_ok()
    assert len(get_status.value) == 2
    assert torch.equal(torch.cat(get_status.value), kv_tensors[0:32])


@pytest.mark.parametrize("eviction_policy", ["FIFO", "LRU", "S3FIFO"])
def test_put_and_get_with_prefix(eviction_policy):
    capacity = 10
    shape = (16, 8, 64)
    dtype = torch.bfloat16

    cache = L1Cache(
        eviction_policy=eviction_policy,
        capacity=capacity,
        allocator=get_allocator(capacity, shape, dtype),
        block_spec=get_block_spec(shape, dtype),
    )

    tokens0 = [i for i in range(32)]
    kv_tensors0 = torch.randn(len(tokens0), 8, 64, dtype=torch.bfloat16)

    put_status = cache.put(None, tokens0, kv_tensors0)
    assert put_status.is_ok()
    assert put_status.value == kv_tensors0.shape[0]

    tokens1 = [i for i in range(100, 135)]
    kv_tensors1 = torch.randn(len(tokens1), 8, 64, dtype=torch.bfloat16)

    put_status = cache.put(tokens0, tokens1, kv_tensors1)
    assert put_status.is_ok()
    assert put_status.value == kv_tensors1.shape[0] // shape[0] * shape[0]

    get_status = cache.get(None, tokens0)
    assert get_status.is_ok()
    assert torch.equal(torch.cat(get_status.value), kv_tensors0)

    get_status = cache.get(tokens0, tokens1)
    assert get_status.is_ok()
    assert torch.equal(torch.cat(get_status.value), kv_tensors1[0:32])

    get_status = cache.get(None, tokens0 + tokens1)
    assert get_status.is_ok()
    chunks = torch.chunk(torch.cat(get_status.value), 2)
    assert torch.equal(chunks[0], kv_tensors0)
    assert torch.equal(chunks[1], kv_tensors1[0:32])


@pytest.mark.parametrize("eviction_policy", ["FIFO", "LRU", "S3FIFO"])
def test_duplicated_puts(eviction_policy):
    capacity = 10
    shape = (16, 8, 64)
    dtype = torch.bfloat16

    cache = L1Cache(
        eviction_policy=eviction_policy,
        capacity=capacity,
        allocator=get_allocator(capacity, shape, dtype),
        block_spec=get_block_spec(shape, dtype),
    )

    for _ in range(10):
        tokens = [i for i in range(32)]
        kv_tensors = torch.randn(32, 8, 64, dtype=torch.bfloat16)

        put_status = cache.put(None, tokens, kv_tensors)
        assert put_status.is_ok()
        assert put_status.value == kv_tensors.shape[0]

        get_status = cache.get(None, tokens)
        assert get_status.is_ok()
        assert torch.equal(torch.cat(get_status.value), kv_tensors)
        assert len(cache) == 2


@pytest.mark.parametrize("eviction_policy", ["FIFO", "LRU", "S3FIFO"])
def test_cache_eviction(eviction_policy):
    capacity = 10
    shape = (16, 8, 64)
    dtype = torch.bfloat16

    cache = L1Cache(
        eviction_policy=eviction_policy,
        capacity=capacity,
        allocator=get_allocator(capacity, shape, dtype),
        block_spec=get_block_spec(shape, dtype),
    )

    for i in range(0, capacity, 2):
        tokens = [i * 64 + j for j in range(32)]
        kv_tensors = torch.randn(32, 8, 64, dtype=torch.bfloat16)

        put_status = cache.put(None, tokens, kv_tensors)
        assert put_status.is_ok(), f"i={i}, len(cache)={len(cache)}"
        assert put_status.value == kv_tensors.shape[0]
        assert len(cache) == (i // 2 + 1) * 2

    assert len(cache) == 10
    tokens = [640 + j for j in range(32)]
    kv_tensors = torch.randn(32, 8, 64, dtype=torch.bfloat16)
    put_status = cache.put(None, tokens, kv_tensors)
    assert put_status.is_ok()
    assert put_status.value == kv_tensors.shape[0]


@pytest.mark.parametrize("eviction_policy", ["FIFO", "LRU", "S3FIFO"])
def test_stress_cache(eviction_policy):
    block_ntokens = 16
    capacity = 10000
    shape = (block_ntokens, 8, 64)
    dtype = torch.bfloat16

    cache = L1Cache(
        eviction_policy=eviction_policy,
        capacity=capacity,
        allocator=get_allocator(capacity, shape, dtype),
        block_spec=get_block_spec(shape, dtype),
    )

    query = {}
    for i in range(500):
        num_prefix_blocks = random.randint(0, 10)
        prefix_tokens = [j for j in range(num_prefix_blocks * block_ntokens)]
        prefix_kv_tensors = torch.randn(len(prefix_tokens),
                                        8,
                                        64,
                                        dtype=torch.bfloat16)
        put_status = cache.put(None, prefix_tokens, prefix_kv_tensors)
        if put_status.is_out_of_memory():
            continue

        assert put_status.is_ok()
        assert put_status.value >= 0 and put_status.value <= prefix_kv_tensors.shape[
            0]
        cache.get(None, prefix_tokens)

        ntokens = random.randint(16, 1024)
        tokens = [j for j in range(ntokens)]
        random.shuffle(tokens)
        kv_tensors = torch.randn(len(tokens), 8, 64, dtype=torch.bfloat16)
        put_status = cache.put(prefix_tokens, tokens, kv_tensors)
        if put_status.is_out_of_memory():
            continue

        assert put_status.is_ok()
        assert put_status.value >= 0 and put_status.value <= kv_tensors.shape[0]
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
            length = (random.randint(1, (len(tokens) - j) // 16) *
                      16 if len(tokens) - j > 16 else 16)

            get_status = cache.get(prefix_tokens, tokens[j:j + length])
            if get_status.is_ok():
                assert len(get_status.value) > 0
                assert torch.equal(
                    torch.cat(get_status.value),
                    kv_tensors[j:j + len(get_status.value) * 16])
                results.append(1)
            else:
                results.append(0)
            prefix_tokens += tokens[j:j + length]
            j += length

    num_oks = sum(results)
    assert num_oks > 250
