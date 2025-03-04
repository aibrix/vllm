import copy
import os
import pytest
import random
import shutil
import torch

import torch.distributed as dist
import torch.multiprocessing as mp

from contextlib import contextmanager
from tqdm import tqdm
from typing import Any, List
from .conftest import get_cache_conf, discard_all_vllm_envs, CACHE_DTYPE, TEMP_ROOT

from .. import (
    GroupAwareKVCacheManager,
    KVCacheBlockLayout,
    KVCacheConfig,
)


@pytest.fixture(params=["l1", "l2_sync", "l2_async", "l1_l2_sync"],
                scope="function")
def envs(request):
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

    yield request.param

    if os.path.exists(TEMP_ROOT):
        shutil.rmtree(TEMP_ROOT, ignore_errors=True)


# dist utils
def dist_run(func, envs_name, world_size, layout):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    processes: List[mp.Process] = []
    for i in range(world_size):
        p = mp.Process(target=func, args=(envs_name, i, world_size, layout))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    for p in processes:
        assert p.exitcode == 0


@contextmanager
def process_group(rank: int, world_size: int):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # each process use a different root for rocksdb
    if "VLLM_KV_CACHE_OL_ROCKSDB_ROOT" in os.environ:
        os.environ["VLLM_KV_CACHE_OL_ROCKSDB_ROOT"] += f"/{rank}"
    dist.barrier()
    yield
    dist.barrier()
    dist.destroy_process_group()


# cache utils
def my_get_cache_conf(rank: int, world_size: int, layout: KVCacheBlockLayout):
    heads = list(range(world_size * 4))

    shape, spec = get_cache_conf(layout)
    spec.tensor_spec.heads = heads[rank * 4:(rank + 1) * 4],
    return shape, spec


@contextmanager
def cache_conf(rank: int, world_size: int, layout: KVCacheBlockLayout):
    cache = None
    try:
        shape, spec = my_get_cache_conf(rank, world_size, layout)
        config = KVCacheConfig(block_spec=spec)
        cache = GroupAwareKVCacheManager(config=config,
                                         process_group=dist.group.WORLD)
        yield shape, spec, cache
    finally:
        if cache is not None:
            cache.close()


def _test_put_and_get_aligned(envs_name: str, rank: int, world_size: int,
                              layout: KVCacheBlockLayout):
    with process_group(rank, world_size), cache_conf(rank, world_size,
                                                     layout) as cache_config:
        shape, spec, cache = cache_config
        tokens = [i for i in range(32)]
        origin_tokens = copy.deepcopy(tokens)
        shape[spec.block_shape_token_dim] = len(tokens)
        kv_tensors = torch.randn(*shape, dtype=CACHE_DTYPE)

        put_status = cache.put(None, tokens, kv_tensors)
        assert tokens == origin_tokens, f"{tokens}!= {origin_tokens}"
        assert put_status.is_ok(), f"{put_status}"
        assert put_status.value == len(
            tokens), f"{put_status.value}!= {len(tokens)}"

        if envs_name.endswith("async"):
            cache.flush()

        get_status = cache.get(None, tokens)
        assert tokens == origin_tokens, f"{tokens}!= {origin_tokens}"
        assert get_status.is_ok(), f"{get_status}"
        assert get_status.value[0] == 32, f"{get_status.value[0]}!= 32"
        assert torch.equal(get_status.value[1], kv_tensors)


@pytest.mark.parametrize("layout",
                         [KVCacheBlockLayout.NCLD, KVCacheBlockLayout.LCND])
def test_put_and_get_aligned(envs, layout):
    dist_run(_test_put_and_get_aligned, envs, 8, layout)


def _test_put_and_get_unaligned(envs_name: str, rank: int, world_size: int,
                                layout: KVCacheBlockLayout):
    with process_group(rank, world_size), cache_conf(rank, world_size,
                                                     layout) as cache_config:
        shape, spec, cache = cache_config
        tokens = [i for i in range(35 + world_size * 700)]
        shape[spec.block_shape_token_dim] = len(tokens)
        kv_tensors = torch.randn(*shape, dtype=CACHE_DTYPE)

        put_status = cache.put(None, tokens, kv_tensors)
        assert put_status.is_ok(), f"{put_status}"

        if envs_name.endswith("async"):
            cache.flush()

        num_blocks = len(tokens) // 16
        num_blocks_per_rank = num_blocks // world_size
        num_tokens_per_rank = num_blocks_per_rank * 16
        my_token_len = num_tokens_per_rank * (rank + 1)
        # we delete some portion of tokens on different ranks to mimic
        # the scenario of different ranks have different cache hits
        del_status = cache.delete(tokens[:my_token_len], tokens[my_token_len:])
        assert del_status.is_ok(), f"{del_status}"

        get_status = cache.get(None, tokens)
        assert get_status.is_ok(), f"{get_status}"
        assert get_status.value[
            0] == num_tokens_per_rank, f"{get_status.value[0]}!={num_tokens_per_rank}"
        slices = [slice(None)] * len(shape)
        slices[spec.block_shape_token_dim] = slice(0, num_tokens_per_rank)
        assert torch.equal(get_status.value[1], kv_tensors[tuple(slices)])


@pytest.mark.parametrize("layout",
                         [KVCacheBlockLayout.NCLD, KVCacheBlockLayout.LCND])
def test_put_and_get_unaligned(envs, layout):
    dist_run(_test_put_and_get_unaligned, envs, 8, layout)


def _test_stress_cache(envs_name: str, rank: int, world_size: int,
                       layout: KVCacheBlockLayout):

    def _bcast_object(obj: Any) -> Any:
        obj_list = [obj]
        dist.broadcast_object_list(obj_list, src=0, group=dist.group.WORLD)
        return obj_list[0]

    with process_group(rank, world_size), cache_conf(rank, world_size,
                                                     layout) as cache_config:
        shape, spec, cache = cache_config
        random.seed(rank)
        query = {}
        for i in tqdm(range(200), desc="putting cache"):
            num_prefix_blocks = random.randint(0, 10)
            num_prefix_blocks = _bcast_object(num_prefix_blocks)
            prefix_tokens = [j for j in range(num_prefix_blocks * 16)]
            prefix_tokens = _bcast_object(prefix_tokens)
            shape[spec.block_shape_token_dim] = len(prefix_tokens)
            prefix_kv_tensors = torch.randn(*shape, dtype=CACHE_DTYPE)
            cache.put(None, prefix_tokens, prefix_kv_tensors)

            ntokens = random.randint(256, 10240)
            ntokens = _bcast_object(ntokens)
            tokens = [j for j in range(ntokens)]
            random.shuffle(tokens)
            tokens = _bcast_object(tokens)
            shape[spec.block_shape_token_dim] = len(tokens)
            kv_tensors = torch.randn(*shape, dtype=CACHE_DTYPE)
            cache.put(prefix_tokens, tokens, kv_tensors)
            query[i] = (prefix_tokens, tokens, kv_tensors)

        if envs_name.endswith("async"):
            cache.flush()

        results = []
        slices = [slice(None)] * len(shape)
        for i in tqdm(range(200), desc="getting cache"):
            prefix_tokens, tokens, kv_tensors = query[i]

            ntokens_to_del = random.randint(128, len(tokens))
            ntokens_left = (len(tokens) - ntokens_to_del) // 16 * 16
            # we delete some portion of tokens on different ranks to mimic
            # the scenario of different ranks have different cache hits
            del_status = cache.delete(prefix_tokens + tokens[:ntokens_left],
                                      tokens[ntokens_left:])
            assert del_status.is_ok(), f"{del_status}"

            get_status = cache.get(prefix_tokens, tokens)
            if get_status.is_ok():
                assert get_status.value[0] > 0, f"{get_status.value[0]}<=0"
                assert get_status.value[
                    0] <= ntokens_left, f"{get_status.value[0]}>{ntokens_left}"
                slices[spec.block_shape_token_dim] = slice(
                    0, get_status.value[0])
                assert torch.equal(get_status.value[1],
                                   kv_tensors[tuple(slices)])
                results.append(1)
            else:
                results.append(0)

        num_oks = sum(results)
        assert num_oks > 0, f"{num_oks}<=0"


@pytest.mark.parametrize("layout",
                         [KVCacheBlockLayout.NCLD, KVCacheBlockLayout.LCND])
def test_stress_cache(envs, layout):
    dist_run(_test_stress_cache, envs, 8, layout)
