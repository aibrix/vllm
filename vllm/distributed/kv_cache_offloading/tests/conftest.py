import os
import pytest
import torch

from ..spec import KVCacheBlockLayout, KVCacheBlockSpec, KVCacheTensorSpec, KVCacheLayerSpec

CACHE_SHAPE_NCLD = (16, 2, 8, 64)
CACHE_SHAPE_LCND = (8, 2, 16, 64)
CACHE_DTYPE = torch.bfloat16
TEMP_ROOT = os.path.join(os.path.expanduser("."), ".test_dir")


def discard_all_vllm_envs():
    # Find all environment variables that start with "VLLM_"
    vllm_keys = [key for key in os.environ if key.startswith("VLLM_")]

    # Remove them from the environment
    for key in vllm_keys:
        del os.environ[key]


def get_cache_conf(layout):
    if layout == KVCacheBlockLayout.NCLD:
        shape = CACHE_SHAPE_NCLD
        return list(shape), KVCacheBlockSpec(
            block_ntokens=shape[0],
            block_dtype=CACHE_DTYPE,
            block_layout=layout,
            tensor_spec=KVCacheTensorSpec(
                heads=[1, 2],
                layers=list(range(shape[2])),
                layer_specs=[
                    KVCacheLayerSpec(size=shape[3]) for _ in range(shape[2])
                ],
            ),
        )
    elif layout == KVCacheBlockLayout.LCND:
        shape = CACHE_SHAPE_LCND
        return list(shape), KVCacheBlockSpec(
            block_ntokens=shape[2],
            block_dtype=CACHE_DTYPE,
            block_layout=layout,
            tensor_spec=KVCacheTensorSpec(
                heads=[1, 2],
                layers=list(range(shape[0])),
                layer_specs=[
                    KVCacheLayerSpec(size=shape[3]) for _ in range(shape[0])
                ],
            ),
        )
    return None, None


@pytest.fixture(params=[KVCacheBlockLayout.NCLD, KVCacheBlockLayout.LCND],
                scope="function")
def cache_conf_fixture(request):
    layout = request.param
    return get_cache_conf(layout)
