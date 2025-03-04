import asyncio
import copy
import os
import pytest
import random
import shutil
import torch

from .conftest import cache_conf_fixture, discard_all_vllm_envs, TEMP_ROOT
from .. import BaseKVCacheManager, KVCacheConfig


@pytest.fixture(params=["l1", "l2_sync", "l2_async", "l1_l2_sync"],
                scope="function")
def cache_mgr_fixture(cache_conf_fixture, request):
    discard_all_vllm_envs()

    os.environ["VLLM_KV_CACHE_OL_L1_CACHE_CAPACITY_GB"] = "1"

    if request.param == "l1":
        # enable l1 and disable l2
        os.environ["VLLM_KV_CACHE_OL_L1_CACHE_ENABLED"] = "1"
        os.environ["VLLM_KV_CACHE_OL_L2_CACHE_BACKEND"] = ""

        # let allocator use host memory
        os.environ["VLLM_KV_CACHE_OL_L1_CACHE_DEVICE"] = "cpu"
        os.environ["VLLM_KV_CACHE_OL_L1_CACHE_PIN_MEMORY"] = "0"
    elif request.param == "l2_sync":
        # enable l2 and disable l1
        os.environ["VLLM_KV_CACHE_OL_L1_CACHE_ENABLED"] = "0"

        os.environ["VLLM_KV_CACHE_OL_L2_CACHE_BACKEND"] = "ROCKSDB"
        os.environ[
            "VLLM_KV_CACHE_OL_L2_CACHE_INGESTION_MAX_INFLIGHT_TOKENS"] = "0"

        # rocksdb envs
        os.environ["VLLM_KV_CACHE_OL_ROCKSDB_ROOT"] = TEMP_ROOT
    elif request.param == "l2_async":
        # enable l2 and disable l1
        os.environ["VLLM_KV_CACHE_OL_L1_CACHE_ENABLED"] = "0"
        os.environ["VLLM_KV_CACHE_OL_L2_CACHE_BACKEND"] = "ROCKSDB"

        # rocksdb envs
        os.environ["VLLM_KV_CACHE_OL_ROCKSDB_ROOT"] = TEMP_ROOT

    elif request.param == "l1_l2_sync":
        # enable both l1 and l2
        os.environ["VLLM_KV_CACHE_OL_L1_CACHE_ENABLED"] = "1"
        os.environ["VLLM_KV_CACHE_OL_L2_CACHE_BACKEND"] = "ROCKSDB"

        # let allocator use host memory
        os.environ["VLLM_KV_CACHE_OL_L1_CACHE_DEVICE"] = "cpu"
        os.environ["VLLM_KV_CACHE_OL_L1_CACHE_PIN_MEMORY"] = "0"
        os.environ["VLLM_KV_CACHE_OL_L1_CACHE_CAPACITY_GB"] = "0.01"

        os.environ["VLLM_KV_CACHE_OL_L2_CACHE_INGESTION_TYPE"] = "EVICTED"
        os.environ[
            "VLLM_KV_CACHE_OL_L2_CACHE_INGESTION_MAX_INFLIGHT_TOKENS"] = "0"
        # always use double get
        os.environ["VLLM_KV_CACHE_OL_DOUBLE_GET_THRESHOLD"] = "0"

        # rocksdb envs
        os.environ["VLLM_KV_CACHE_OL_ROCKSDB_ROOT"] = TEMP_ROOT

    if os.path.exists(TEMP_ROOT):
        shutil.rmtree(TEMP_ROOT, ignore_errors=True)

    shape, spec = cache_conf_fixture

    cache = None
    try:
        config = KVCacheConfig(block_spec=spec)
        cache = BaseKVCacheManager(config=config)
        yield shape, spec, cache, request.param
    finally:
        if cache is not None:
            cache.close()
        if os.path.exists(TEMP_ROOT):
            shutil.rmtree(TEMP_ROOT, ignore_errors=True)


def test_cache_initialization(cache_mgr_fixture):
    _, _, cache_mgr, _ = cache_mgr_fixture
    request_param = getattr(cache_mgr, "__test_request_param__", "")
    if "l1" in request_param:
        assert cache_mgr._l1_cache is not None
    if "l2" in request_param:
        assert cache_mgr._l2_cache is not None


def test_put_and_get_aligned(cache_mgr_fixture):
    shape, spec, cache_mgr, param = cache_mgr_fixture
    tokens = [i for i in range(32)]
    origin_tokens = copy.deepcopy(tokens)
    shape[spec.block_shape_token_dim] = len(tokens)
    kv_tensors = torch.randn(*shape, dtype=torch.bfloat16)

    put_status = cache_mgr.put(None, tokens, kv_tensors)
    assert tokens == origin_tokens
    assert put_status.is_ok()

    if param.endswith("async"):
        cache_mgr.flush()

    get_status = cache_mgr.get(None, tokens)
    assert tokens == origin_tokens
    assert get_status.is_ok()
    assert get_status.value[0] == 32
    assert torch.equal(get_status.value[1], kv_tensors)


def test_put_and_get_unaligned(cache_mgr_fixture):
    shape, spec, cache_mgr, param = cache_mgr_fixture
    tokens = [i for i in range(35)]
    shape[spec.block_shape_token_dim] = len(tokens)
    kv_tensors = torch.randn(*shape, dtype=torch.bfloat16)

    put_status = cache_mgr.put(None, tokens, kv_tensors)
    assert put_status.is_ok()

    if param.endswith("async"):
        cache_mgr.flush()

    get_status = cache_mgr.get(None, tokens)
    assert get_status.is_ok()
    assert get_status.value[0] == 32
    slices = [slice(None)] * len(shape)
    slices[spec.block_shape_token_dim] = slice(0, 32)
    assert torch.equal(get_status.value[1], kv_tensors[tuple(slices)])


def test_put_and_get_with_prefix(cache_mgr_fixture):
    shape, spec, cache_mgr, param = cache_mgr_fixture
    tokens0 = [i for i in range(32)]
    shape[spec.block_shape_token_dim] = len(tokens0)
    kv_tensors0 = torch.randn(*shape, dtype=torch.bfloat16)

    put_status = cache_mgr.put(None, tokens0, kv_tensors0)
    assert put_status.is_ok()

    tokens1 = [i for i in range(100, 135)]
    shape[spec.block_shape_token_dim] = len(tokens1)
    kv_tensors1 = torch.randn(*shape, dtype=torch.bfloat16)

    put_status = cache_mgr.put(tokens0, tokens1, kv_tensors1)
    assert put_status.is_ok()

    if param.endswith("async"):
        cache_mgr.flush()

    get_status = cache_mgr.get(None, tokens0)
    assert get_status.is_ok()
    assert torch.equal(get_status.value[1], kv_tensors0)

    get_status = cache_mgr.get(tokens0, tokens1)
    assert get_status.is_ok()
    slices = [slice(None)] * len(shape)
    slices[spec.block_shape_token_dim] = slice(0, 32)
    assert torch.equal(get_status.value[1], kv_tensors1[tuple(slices)])

    get_status = cache_mgr.get(None, tokens0 + tokens1)
    assert get_status.is_ok()
    chunks = torch.chunk(get_status.value[1],
                         2,
                         dim=spec.block_shape_token_dim)
    assert torch.equal(chunks[0], kv_tensors0)
    slices = [slice(None)] * len(shape)
    slices[spec.block_shape_token_dim] = slice(0, 32)
    assert torch.equal(chunks[1], kv_tensors1[tuple(slices)])


def test_duplicated_puts(cache_mgr_fixture):
    shape, spec, cache_mgr, param = cache_mgr_fixture
    for _ in range(10):
        tokens = [i for i in range(32)]
        shape[spec.block_shape_token_dim] = len(tokens)
        kv_tensors = torch.randn(*shape, dtype=torch.bfloat16)

        put_status = cache_mgr.put(None, tokens, kv_tensors)
        assert put_status.is_ok()

        if param.endswith("async"):
            cache_mgr.flush()

        get_status = cache_mgr.get(None, tokens)
        assert get_status.is_ok()
        assert torch.equal(get_status.value[1], kv_tensors)


def test_delete(cache_mgr_fixture):
    shape, spec, cache_mgr, param = cache_mgr_fixture
    tokens = [i for i in range(32)]
    origin_tokens = copy.deepcopy(tokens)
    shape[spec.block_shape_token_dim] = len(tokens)
    kv_tensors = torch.randn(*shape, dtype=torch.bfloat16)

    put_status = cache_mgr.put(None, tokens, kv_tensors)
    assert tokens == origin_tokens
    assert put_status.is_ok()
    assert put_status.value == kv_tensors.shape[spec.block_shape_token_dim]

    if param.endswith("async"):
        cache_mgr.flush()

    del_status = cache_mgr.delete(tokens[:16], tokens[16:])
    assert del_status.is_ok()

    get_status = cache_mgr.get(None, tokens[:16])
    assert get_status.is_ok()
    assert get_status.value[0] == 16
    slices = [slice(None)] * len(shape)
    slices[spec.block_shape_token_dim] = slice(0, 16)
    assert torch.equal(get_status.value[1], kv_tensors[tuple(slices)])

    get_status = cache_mgr.get(tokens[:16], tokens[16:])
    assert get_status.is_not_found()


def test_stress_cache(cache_mgr_fixture):
    shape, spec, cache_mgr, param = cache_mgr_fixture
    query = {}
    for i in range(200):
        num_prefix_blocks = random.randint(0, 30)
        prefix_tokens = [j for j in range(num_prefix_blocks * 16)]
        shape[spec.block_shape_token_dim] = len(prefix_tokens)
        prefix_kv_tensors = torch.randn(*shape, dtype=torch.bfloat16)
        put_status = cache_mgr.put(None, prefix_tokens, prefix_kv_tensors)
        if put_status.is_out_of_memory() or put_status.is_denied():
            continue

        assert put_status.is_ok()
        cache_mgr.get(None, prefix_tokens)

        ntokens = random.randint(128, 1024)
        tokens = [j for j in range(ntokens)]
        random.shuffle(tokens)
        shape[spec.block_shape_token_dim] = len(tokens)
        kv_tensors = torch.randn(*shape, dtype=torch.bfloat16)
        put_status = cache_mgr.put(prefix_tokens, tokens, kv_tensors)
        if put_status.is_out_of_memory() or put_status.is_denied():
            continue

        assert put_status.is_ok()
        cache_mgr.get(prefix_tokens, tokens)
        query[i] = (prefix_tokens, tokens, kv_tensors)

    if param.endswith("async"):
        cache_mgr.flush()

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

            if cache_mgr._l1_cache is not None and cache_mgr._l2_cache is not None:
                l1_get_status = cache_mgr._l1_cache.get(
                    prefix_tokens, tokens[j:j + length])
                l1_got = len(
                    l1_get_status.value) if l1_get_status.is_ok() else 0
                l2_get_status = asyncio.run_coroutine_threadsafe(
                    cache_mgr._l2_cache.get(prefix_tokens,
                                            tokens[j:j + length]),
                    cache_mgr._event_loop).result()
                l2_got = len(
                    l2_get_status.value) if l2_get_status.is_ok() else 0
                get_status = cache_mgr.get(prefix_tokens, tokens[j:j + length])
                if l1_got + l2_got > 0:
                    assert get_status.is_ok() and get_status.value[0] == max(
                        l1_got, l2_got
                    ) * 16, f"l1_got={l1_got}, l2_got={l2_got}, get_status={get_status}"
            else:
                get_status = cache_mgr.get(prefix_tokens, tokens[j:j + length])

            if get_status.is_ok():
                assert get_status.value[0] > 0 and get_status.value[
                    0] <= length, f"{get_status.value[0]} vs {length}"
                slices[spec.block_shape_token_dim] = slice(
                    j, j + get_status.value[0])
                assert torch.equal(get_status.value[1],
                                   kv_tensors[tuple(slices)])
                results.append(1)
            else:
                results.append(0)
            prefix_tokens += tokens[j:j + length]
            j += length

    num_oks = sum(results)
    assert num_oks > 50
