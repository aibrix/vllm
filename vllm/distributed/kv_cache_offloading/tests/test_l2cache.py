import asyncio
import copy
import os
import pytest
import random
import shutil
import torch

from ..config import KVCacheRetentionType
from ..l2 import L2Cache
from ..spec import KVCacheBlockLayout, KVCacheBlockSpec, KVCacheTensorSpec, KVCacheLayerSpec

TEMP_ROOT = os.path.join(os.path.expanduser("."), ".test_rocksdb")

# rocksdb envs
os.environ["VLLM_KV_CACHE_OL_ROCKSDB_ROOT"] = TEMP_ROOT


@pytest.fixture
def l2cache():
    if os.path.exists(TEMP_ROOT):
        shutil.rmtree(TEMP_ROOT, ignore_errors=True)

    shape = (16, 8, 64)
    dtype = torch.bfloat16

    block_spec = KVCacheBlockSpec(
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
    cache = None
    try:
        cache = L2Cache(
            backend_name="ROCKSDB",
            namespace="test",
            block_spec=block_spec,
            retention_type=KVCacheRetentionType.CACHING,
        )
        yield cache
    finally:
        if cache is not None:
            cache.close()
            del cache
        if os.path.exists(TEMP_ROOT):
            shutil.rmtree(TEMP_ROOT, ignore_errors=True)


@pytest.mark.asyncio
async def test_put_and_get_aligned(l2cache):
    open_status = l2cache.open()
    open_status.raise_if_has_exception()

    tokens = [i for i in range(32)]
    origin_tokens = copy.deepcopy(tokens)
    kv_tensors = torch.randn(32, 8, 64, dtype=torch.bfloat16)

    put_status = await l2cache.put(None, tokens, kv_tensors)
    assert tokens == origin_tokens
    assert put_status.is_ok()
    assert put_status.value == kv_tensors.shape[0]

    get_status = await l2cache.get(None, tokens)
    assert tokens == origin_tokens
    assert get_status.is_ok()
    assert len(get_status.value) == 2
    assert torch.equal(torch.cat(get_status.value), kv_tensors)


@pytest.mark.asyncio
async def test_put_and_get_unaligned(l2cache):
    open_status = l2cache.open()
    open_status.raise_if_has_exception()

    tokens = [i for i in range(35)]
    kv_tensors = torch.randn(len(tokens), 8, 64, dtype=torch.bfloat16)

    put_status = await l2cache.put(None, tokens, kv_tensors)
    assert put_status.is_ok()
    assert put_status.value == 32

    get_status = await l2cache.get(None, tokens)
    assert get_status.is_ok()
    assert len(get_status.value) == 2
    assert torch.equal(torch.cat(get_status.value), kv_tensors[0:32])


@pytest.mark.asyncio
async def test_put_and_get_with_prefix(l2cache):
    open_status = l2cache.open()
    open_status.raise_if_has_exception()

    tokens0 = [i for i in range(32)]
    kv_tensors0 = torch.randn(len(tokens0), 8, 64, dtype=torch.bfloat16)

    put_status = await l2cache.put(None, tokens0, kv_tensors0)
    assert put_status.is_ok()
    assert put_status.value == kv_tensors0.shape[0]

    tokens1 = [i for i in range(100, 135)]
    kv_tensors1 = torch.randn(len(tokens1), 8, 64, dtype=torch.bfloat16)

    put_status = await l2cache.put(tokens0, tokens1, kv_tensors1)
    assert put_status.is_ok()
    assert put_status.value == 32

    get_status = await l2cache.get(None, tokens0)
    assert get_status.is_ok()
    assert torch.equal(torch.cat(get_status.value), kv_tensors0)

    get_status = await l2cache.get(tokens0, tokens1)
    assert get_status.is_ok()
    assert torch.equal(torch.cat(get_status.value), kv_tensors1[0:32])

    get_status = await l2cache.get(None, tokens0 + tokens1)
    assert get_status.is_ok()
    chunks = torch.chunk(torch.cat(get_status.value), 2)
    assert torch.equal(chunks[0], kv_tensors0)
    assert torch.equal(chunks[1], kv_tensors1[0:32])


@pytest.mark.asyncio
async def test_duplicated_puts(l2cache):
    open_status = l2cache.open()
    open_status.raise_if_has_exception()

    for _ in range(10):
        tokens = [i for i in range(32)]
        kv_tensors = torch.randn(32, 8, 64, dtype=torch.bfloat16)

        put_status = await l2cache.put(None, tokens, kv_tensors)
        assert put_status.is_ok()
        assert put_status.value == kv_tensors.shape[0]

        get_status = await l2cache.get(None, tokens)
        assert get_status.is_ok()
        assert torch.equal(torch.cat(get_status.value), kv_tensors)


@pytest.mark.asyncio
async def test_delete(l2cache):
    open_status = l2cache.open()
    open_status.raise_if_has_exception()

    tokens = [i for i in range(32)]
    origin_tokens = copy.deepcopy(tokens)
    kv_tensors = torch.randn(32, 8, 64, dtype=torch.bfloat16)

    put_status = await l2cache.put(None, tokens, kv_tensors)
    assert tokens == origin_tokens
    assert put_status.is_ok()
    assert put_status.value == kv_tensors.shape[0]

    del_status = await l2cache.delete(tokens[:16], tokens[16:])
    assert del_status.is_ok()

    get_status = await l2cache.get(None, tokens[:16])
    assert get_status.is_ok()
    assert len(get_status.value) == 1
    assert torch.equal(torch.cat(get_status.value), kv_tensors[:16])

    get_status = await l2cache.get(tokens[:16], tokens[16:])
    assert get_status.is_not_found()


@pytest.mark.asyncio
async def test_stress_cache(l2cache):
    open_status = l2cache.open()
    open_status.raise_if_has_exception()

    query = {}
    for i in range(200):
        num_prefix_blocks = random.randint(0, 10)
        prefix_tokens = [j for j in range(num_prefix_blocks * 16)]
        prefix_kv_tensors = torch.randn(len(prefix_tokens),
                                        8,
                                        64,
                                        dtype=torch.bfloat16)
        put_status = await l2cache.put(None, prefix_tokens, prefix_kv_tensors)
        if put_status.is_out_of_memory() or put_status.is_denied():
            continue

        assert put_status.is_ok()
        assert put_status.value >= 0 and put_status.value <= prefix_kv_tensors.shape[
            0]
        await l2cache.get(None, prefix_tokens)

        ntokens = random.randint(16, 1024)
        tokens = [j for j in range(ntokens)]
        random.shuffle(tokens)
        kv_tensors = torch.randn(len(tokens), 8, 64, dtype=torch.bfloat16)
        put_status = await l2cache.put(prefix_tokens, tokens, kv_tensors)
        if put_status.is_out_of_memory() or put_status.is_denied():
            continue

        assert put_status.is_ok()
        assert put_status.value >= 0 and put_status.value <= kv_tensors.shape[0]
        await l2cache.get(prefix_tokens, tokens)
        query[i] = (prefix_tokens, tokens, kv_tensors)

    results = []
    for i in range(200):
        if i not in query:
            continue

        prefix_tokens, tokens, kv_tensors = query[i]
        j = 0
        while j < len(tokens):
            length = (random.randint(1, (len(tokens) - j) // 16) *
                      16 if len(tokens) - j > 16 else 16)

            get_status = await l2cache.get(prefix_tokens, tokens[j:j + length])
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
    assert num_oks > 50
