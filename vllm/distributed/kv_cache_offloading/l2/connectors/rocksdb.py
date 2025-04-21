# SPDX-License-Identifier: Apache-2.0
import os
from concurrent.futures import Executor

import rocksdict
import torch

from ... import envs
from ...common import AsyncBase
from ...status import Status, StatusCodes
from ...utils import bytes_to_tensor, ensure_dir_exist, tensor_to_bytes
from . import Connector, ConnectorFeature


@AsyncBase.async_wrap(exists="_exists",
                      get="_get",
                      put="_put",
                      delete="_delete")
class RocksDBConnector(Connector[bytes, torch.Tensor], AsyncBase):
    """RocksDB connector."""

    def __init__(
        self,
        path: str,
        opts: rocksdict.Options,
        access_type: rocksdict.AccessType,
        executor: Executor,
    ):
        super().__init__(executor)
        self.path = path
        self.opts = opts
        self.access_type = access_type
        self.store = None

    @classmethod
    def from_envs(cls, conn_id: str, executor: Executor) -> "RocksDBConnector":
        """Create a connector from environment variables."""
        root = envs.VLLM_KV_CACHE_OL_ROCKSDB_ROOT
        root = os.path.join(os.path.expanduser(root), conn_id)
        opts = rocksdict.Options(raw_mode=True)
        opts.create_if_missing(True)
        opts.create_missing_column_families(True)
        opts.set_write_buffer_size(
            envs.VLLM_KV_CACHE_OL_ROCKSDB_WRITE_BUFFER_SIZE)
        opts.set_target_file_size_base(
            envs.VLLM_KV_CACHE_OL_ROCKSDB_TARGET_FILE_SIZE_BASE)
        opts.set_max_write_buffer_number(
            envs.VLLM_KV_CACHE_OL_ROCKSDB_MAX_WRITE_BUFFER_NUMBER)
        opts.set_max_background_jobs(
            envs.VLLM_KV_CACHE_OL_ROCKSDB_MAX_BACKGROUND_JOBS)
        opts.set_max_total_wal_size(
            envs.VLLM_KV_CACHE_OL_ROCKSDB_MAX_TOTAL_WAL_SIZE)
        opts.set_wal_dir(os.path.join(os.path.expanduser(root), "wal"))
        opts.set_db_log_dir(os.path.join(os.path.expanduser(root), "db"))
        access_type = rocksdict.AccessType.read_write()
        # use TTL to manage the life cycle of the data in the KV cache
        access_type = access_type.with_ttl(envs.VLLM_KV_CACHE_OL_ROCKSDB_TTL_S)
        return cls(root, opts, access_type, executor)

    @property
    def name(self) -> str:
        return "RocksDB"

    @property
    def feature(self) -> ConnectorFeature:
        return ConnectorFeature()

    @Status.capture_exception
    def open(self) -> Status:
        """Open a connection."""
        if self.store is None:
            ensure_dir_exist(self.path)
            self.store = rocksdict.Rdict(self.path,
                                         self.opts,
                                         access_type=self.access_type)
        return Status(StatusCodes.OK)

    @Status.capture_exception
    def close(self) -> Status:
        """Close a connection."""
        if self.store is not None:
            self.store.close()
            self.store = None
        return Status(StatusCodes.OK)

    @Status.capture_exception
    def _exists(self, key: bytes) -> Status:
        """Check if key is in the store."""
        if key in self.store:
            return Status(StatusCodes.OK)
        return Status(StatusCodes.NOT_FOUND)

    @Status.capture_exception
    def _get(self, key: bytes) -> Status[torch.Tensor]:
        """Get a value."""
        val = self.store.get(key)
        if val is None:
            return Status(StatusCodes.NOT_FOUND)
        tensor = bytes_to_tensor(val)
        return Status(value=tensor)

    @Status.capture_exception
    def _put(self, key: bytes, value: torch.Tensor) -> Status:
        """Put a key value pair"""
        val = tensor_to_bytes(value)
        self.store.put(key, val)
        return Status(StatusCodes.OK)

    @Status.capture_exception
    def _delete(self, key: bytes) -> Status:
        """Delete a key."""
        self.store.delete(key)
        return Status(StatusCodes.OK)
