import asyncio
import os
import pytest
import shutil
import torch

from ..config import KVCacheRetentionType
from ..l2.connectors.rocksdb import RocksDBConnector
from ..l2.marshallers import StringSerializer, TensorSerializer, ZstdCompressor

TEMP_ROOT = os.path.join(os.path.expanduser("."), ".test_rocksdb")
TEMP_FILE = os.path.join(os.path.expanduser(TEMP_ROOT), "test")

key_serializer_g = StringSerializer()
tensor_serializer_g = ZstdCompressor(TensorSerializer())


@pytest.fixture
def rocksdb():
    if os.path.exists(TEMP_ROOT):
        shutil.rmtree(TEMP_ROOT, ignore_errors=True)

    import rocksdict
    opts = rocksdict.Options(raw_mode=True)
    opts.create_if_missing(True)
    opts.create_missing_column_families(True)
    opts.set_wal_dir(os.path.join(os.path.expanduser(TEMP_ROOT), "wal"))
    opts.set_db_log_dir(os.path.join(os.path.expanduser(TEMP_ROOT), "db"))
    # access_type = rocksdict.AccessType.read_write().with_ttl(10)
    access_type = rocksdict.AccessType.with_ttl(10)
    cache = None
    try:
        cache = RocksDBConnector(TEMP_FILE, opts, access_type,
                                 KVCacheRetentionType.CACHING)
        yield cache
    finally:
        if cache is not None:
            cache.close()
            del cache
        if os.path.exists(TEMP_ROOT):
            shutil.rmtree(TEMP_ROOT, ignore_errors=True)


@pytest.mark.asyncio
async def test_put_and_get(rocksdb):
    cache = rocksdb
    assert cache.open().is_ok()

    key = "-".join(str(i) for i in range(32))
    kv_tensors = torch.randn(32, 8, 64, dtype=torch.bfloat16)

    put_status = await cache.put(key_serializer_g.marshal(key),
                                 tensor_serializer_g.marshal(kv_tensors))
    assert put_status.is_ok()

    get_status = await cache.get(key_serializer_g.marshal(key))
    assert get_status.is_ok()
    assert torch.equal(
        tensor_serializer_g.unmarshal(get_status.value).view(
            kv_tensors.dtype).view(kv_tensors.shape),
        kv_tensors,
    )


@pytest.mark.asyncio
async def test_put_and_get_with_key(rocksdb):
    cache = rocksdb
    assert cache.open().is_ok()

    key = [i for i in range(32)]
    key_str = "-".join(str(i) for i in key)
    kv_tensors = torch.randn(32, 8, 64, dtype=torch.bfloat16)

    put_status = await cache.put(
        key_serializer_g.marshal(key_str),
        tensor_serializer_g.marshal((key, kv_tensors)))
    assert put_status.is_ok()

    get_status = await cache.get(key_serializer_g.marshal(key_str))
    assert get_status.is_ok()
    fetched_key, fetched_kv_tensors = tensor_serializer_g.unmarshal(
        get_status.value)
    assert fetched_key == key
    assert torch.equal(
        fetched_kv_tensors.view(kv_tensors.dtype).view(kv_tensors.shape),
        kv_tensors)


@pytest.mark.asyncio
async def test_put_update_existing(rocksdb):
    cache = rocksdb
    assert cache.open().is_ok()

    key, value = "key1", torch.tensor([1, 2, 3])
    new_value = torch.tensor([9, 10, 11])
    put_status = await cache.put(key_serializer_g.marshal(key),
                                 tensor_serializer_g.marshal(value))
    put_status = await cache.put(key_serializer_g.marshal(key),
                                 tensor_serializer_g.marshal(new_value))
    assert put_status.is_ok()
    get_status = await cache.get(key_serializer_g.marshal(key))
    assert get_status.is_ok()
    assert torch.equal(
        tensor_serializer_g.unmarshal(get_status.value).view(
            new_value.dtype).view(new_value.shape),
        new_value,
    )


@pytest.mark.asyncio
async def test_delete(rocksdb):
    cache = rocksdb
    assert cache.open().is_ok()

    key, value = "key1", torch.tensor([1, 2, 3])
    put_status = await cache.put(key_serializer_g.marshal(key),
                                 tensor_serializer_g.marshal(value))
    assert put_status.is_ok()
    delete_status = await cache.delete(key_serializer_g.marshal(key))
    assert delete_status.is_ok()
    get_status = await cache.get(key_serializer_g.marshal(key))
    assert get_status.is_not_found()


@pytest.mark.asyncio
async def test_delete_empty(rocksdb):
    cache = rocksdb
    assert cache.open().is_ok()

    key = "nonexistent"
    await cache.delete(key_serializer_g.marshal(key)
                       )  # Should not raise an error
