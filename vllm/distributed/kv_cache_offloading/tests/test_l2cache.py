import copy
import os
import pytest
import random
import shutil
import torch

from .conftest import cache_conf_fixture, CACHE_DTYPE, TEMP_ROOT
from ..l2 import L2Cache

# rocksdb envs
os.environ["VLLM_KV_CACHE_OL_ROCKSDB_ROOT"] = TEMP_ROOT


@pytest.fixture
def l2cache_fixture(cache_conf_fixture):
    if os.path.exists(TEMP_ROOT):
        shutil.rmtree(TEMP_ROOT, ignore_errors=True)

    shape, spec = cache_conf_fixture

    cache = None
    try:
        cache = L2Cache(
            backend_name="ROCKSDB",
            namespace="test",
            block_spec=spec,
        )
        yield shape, spec, cache
    finally:
        if cache is not None:
            cache.close()
            del cache
        if os.path.exists(TEMP_ROOT):
            shutil.rmtree(TEMP_ROOT, ignore_errors=True)


@pytest.mark.asyncio
async def test_put_and_get_aligned(l2cache_fixture):
    shape, spec, l2cache = l2cache_fixture
    open_status = l2cache.open()
    open_status.raise_if_has_exception()

    tokens = [i for i in range(32)]
    origin_tokens = copy.deepcopy(tokens)
    shape[spec.block_shape_token_dim] = 32
    kv_tensors = torch.randn(*shape, dtype=CACHE_DTYPE)

    put_status = await l2cache.put(None, tokens, kv_tensors)
    assert tokens == origin_tokens
    assert put_status.is_ok()
    assert put_status.value == kv_tensors.shape[spec.block_shape_token_dim]

    get_status = await l2cache.get(None, tokens)
    assert tokens == origin_tokens
    assert get_status.is_ok()
    assert len(get_status.value) == 2
    assert torch.equal(
        torch.cat(get_status.value, dim=spec.block_shape_token_dim),
        kv_tensors)


@pytest.mark.asyncio
async def test_put_and_get_unaligned(l2cache_fixture):
    shape, spec, l2cache = l2cache_fixture
    open_status = l2cache.open()
    open_status.raise_if_has_exception()

    tokens = [i for i in range(35)]
    shape[spec.block_shape_token_dim] = len(tokens)
    kv_tensors = torch.randn(*shape, dtype=CACHE_DTYPE)

    put_status = await l2cache.put(None, tokens, kv_tensors)
    assert put_status.is_ok()
    assert put_status.value == 32

    get_status = await l2cache.get(None, tokens)
    assert get_status.is_ok()
    assert len(get_status.value) == 2
    slices = [slice(None)] * len(shape)
    slices[spec.block_shape_token_dim] = slice(0, 32)
    assert torch.equal(
        torch.cat(get_status.value, dim=spec.block_shape_token_dim),
        kv_tensors[tuple(slices)])


@pytest.mark.asyncio
async def test_put_and_get_with_prefix(l2cache_fixture):
    shape, spec, l2cache = l2cache_fixture
    open_status = l2cache.open()
    open_status.raise_if_has_exception()

    tokens0 = [i for i in range(32)]
    shape[spec.block_shape_token_dim] = len(tokens0)
    kv_tensors0 = torch.randn(*shape, dtype=CACHE_DTYPE)

    put_status = await l2cache.put(None, tokens0, kv_tensors0)
    assert put_status.is_ok()
    assert put_status.value == kv_tensors0.shape[spec.block_shape_token_dim]

    tokens1 = [i for i in range(100, 135)]
    shape[spec.block_shape_token_dim] = len(tokens1)
    kv_tensors1 = torch.randn(*shape, dtype=CACHE_DTYPE)

    put_status = await l2cache.put(tokens0, tokens1, kv_tensors1)
    assert put_status.is_ok()
    assert put_status.value == 32

    get_status = await l2cache.get(None, tokens0)
    assert get_status.is_ok()
    assert torch.equal(
        torch.cat(get_status.value, dim=spec.block_shape_token_dim),
        kv_tensors0)

    get_status = await l2cache.get(tokens0, tokens1)
    assert get_status.is_ok()
    slices = [slice(None)] * len(shape)
    slices[spec.block_shape_token_dim] = slice(0, 32)
    assert torch.equal(
        torch.cat(get_status.value, dim=spec.block_shape_token_dim),
        kv_tensors1[tuple(slices)])

    get_status = await l2cache.get(None, tokens0 + tokens1)
    assert get_status.is_ok()
    chunks = torch.chunk(torch.cat(get_status.value,
                                   dim=spec.block_shape_token_dim),
                         2,
                         dim=spec.block_shape_token_dim)
    assert torch.equal(chunks[0], kv_tensors0)
    slices = [slice(None)] * len(shape)
    slices[spec.block_shape_token_dim] = slice(0, 32)
    assert torch.equal(chunks[1], kv_tensors1[tuple(slices)])


@pytest.mark.asyncio
async def test_duplicated_puts(l2cache_fixture):
    shape, spec, l2cache = l2cache_fixture
    open_status = l2cache.open()
    open_status.raise_if_has_exception()

    for _ in range(10):
        tokens = [i for i in range(32)]
        shape[spec.block_shape_token_dim] = len(tokens)
        kv_tensors = torch.randn(*shape, dtype=CACHE_DTYPE)

        put_status = await l2cache.put(None, tokens, kv_tensors)
        assert put_status.is_ok()
        assert put_status.value == kv_tensors.shape[spec.block_shape_token_dim]

        get_status = await l2cache.get(None, tokens)
        assert get_status.is_ok()
        assert torch.equal(
            torch.cat(get_status.value, dim=spec.block_shape_token_dim),
            kv_tensors)


@pytest.mark.asyncio
async def test_delete(l2cache_fixture):
    shape, spec, l2cache = l2cache_fixture
    open_status = l2cache.open()
    open_status.raise_if_has_exception()

    tokens = [i for i in range(32)]
    shape[spec.block_shape_token_dim] = len(tokens)
    origin_tokens = copy.deepcopy(tokens)
    kv_tensors = torch.randn(*shape, dtype=CACHE_DTYPE)

    put_status = await l2cache.put(None, tokens, kv_tensors)
    assert tokens == origin_tokens
    assert put_status.is_ok()
    assert put_status.value == kv_tensors.shape[spec.block_shape_token_dim]

    del_status = await l2cache.delete(tokens[:16], tokens[16:])
    assert del_status.is_ok()

    get_status = await l2cache.get(None, tokens[:16])
    assert get_status.is_ok()
    assert len(get_status.value) == 1
    slices = [slice(None)] * len(shape)
    slices[spec.block_shape_token_dim] = slice(0, 16)
    assert torch.equal(
        torch.cat(get_status.value, dim=spec.block_shape_token_dim),
        kv_tensors[tuple(slices)])

    get_status = await l2cache.get(tokens[:16], tokens[16:])
    assert get_status.is_not_found()


@pytest.mark.asyncio
async def test_stress_cache(l2cache_fixture):
    shape, spec, l2cache = l2cache_fixture
    open_status = l2cache.open()
    open_status.raise_if_has_exception()

    query = {}
    for i in range(200):
        num_prefix_blocks = random.randint(0, 10)
        prefix_tokens = [j for j in range(num_prefix_blocks * 16)]
        shape[spec.block_shape_token_dim] = len(prefix_tokens)
        prefix_kv_tensors = torch.randn(*shape, dtype=CACHE_DTYPE)
        put_status = await l2cache.put(None, prefix_tokens, prefix_kv_tensors)
        if put_status.is_out_of_memory() or put_status.is_denied():
            continue

        assert put_status.is_ok()
        assert put_status.value >= 0 and put_status.value <= prefix_kv_tensors.shape[
            spec.block_shape_token_dim]
        await l2cache.get(None, prefix_tokens)

        ntokens = random.randint(16, 1024)
        tokens = [j for j in range(ntokens)]
        random.shuffle(tokens)
        shape[spec.block_shape_token_dim] = len(tokens)
        kv_tensors = torch.randn(*shape, dtype=CACHE_DTYPE)
        put_status = await l2cache.put(prefix_tokens, tokens, kv_tensors)
        if put_status.is_out_of_memory() or put_status.is_denied():
            continue

        assert put_status.is_ok()
        assert put_status.value >= 0 and put_status.value <= kv_tensors.shape[
            spec.block_shape_token_dim]
        await l2cache.get(prefix_tokens, tokens)
        query[i] = (prefix_tokens, tokens, kv_tensors)

    results = []
    for i in range(200):
        if i not in query:
            continue

        prefix_tokens, tokens, kv_tensors = query[i]
        slices = [slice(None)] * len(shape)
        j = 0
        while j < len(tokens):
            length = (random.randint(1, (len(tokens) - j) // 16) *
                      16 if len(tokens) - j > 16 else 16)

            get_status = await l2cache.get(prefix_tokens, tokens[j:j + length])
            if get_status.is_ok():
                assert len(get_status.value) > 0
                slices[spec.block_shape_token_dim] = slice(
                    j, j + len(get_status.value) * 16)
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
    assert num_oks > 50
