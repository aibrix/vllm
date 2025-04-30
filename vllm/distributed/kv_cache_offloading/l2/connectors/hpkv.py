# SPDX-License-Identifier: Apache-2.0
from concurrent.futures import Executor
from dataclasses import dataclass

import torch
from hpkv.hpkv_client import HPKVClient

from ... import envs
from ...common import AsyncBase
from ...memory import MemoryRegion
from ...status import Status, StatusCodes
from . import Connector, ConnectorFeature, ConnectorRegisterDescriptor


@dataclass
class HPKVConfig:
    """HPKV config.
    Args:
        remote_addr (str): remote address
        remote_port (int): remote port
        local_addr (str): local address
        local_port (int): local port
        num_queues (int): number of queues, default 0
    """
    remote_addr: str
    remote_port: int
    local_addr: str
    local_port: int
    num_queues: int = 0


@dataclass
class HPKVRegisterDescriptor(ConnectorRegisterDescriptor):
    """HPKV register descriptor."""
    reg_buf: int


@AsyncBase.async_wrap(exists="_exists",
                      get="_get",
                      put="_put",
                      delete="_delete")
class HPKVConnector(Connector[str, torch.Tensor], AsyncBase):
    """HPKV connector."""

    def __init__(
        self,
        config: HPKVConfig,
        key_suffix: str,
        executor: Executor,
    ):
        super().__init__(executor)
        self.config = config
        self.key_suffix = key_suffix
        self.conn = None

    @classmethod
    def from_envs(cls, conn_id: str, executor: Executor) -> "HPKVConnector":
        """Create a connector from environment variables."""
        config = HPKVConfig(
            remote_addr=envs.AIBRIX_KV_CACHE_OL_HPKV_REMOTE_ADDR,
            remote_port=envs.AIBRIX_KV_CACHE_OL_HPKV_REMOTE_PORT,
            local_addr=envs.AIBRIX_KV_CACHE_OL_HPKV_LOCAL_ADDR,
            local_port=envs.AIBRIX_KV_CACHE_OL_HPKV_LOCAL_PORT,
        )
        return cls(config, conn_id, executor)

    @property
    def name(self) -> str:
        return "HPKV"

    @property
    def feature(self) -> ConnectorFeature:
        feature = ConnectorFeature(rdma=True, )
        return feature

    def _key(self, key: str) -> str:
        return key.hex() + self.key_suffix

    @Status.capture_exception
    def open(self) -> Status:
        """Open a connection."""
        if self.conn is None:
            self.conn = HPKVClient(
                raddr=self.config.remote_addr,
                rport=self.config.remote_port,
                laddr=self.config.local_addr,
                lport=self.config.local_port,
                nqueue=self.config.num_queues,
            )
        return Status(StatusCodes.OK)

    @Status.capture_exception
    def close(self) -> Status:
        """Close a connection."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None
        return Status(StatusCodes.OK)

    @Status.capture_exception
    def register_mr(self, addr: int,
                    length: int) -> Status[ConnectorRegisterDescriptor]:
        reg_buf = self.conn.reg_memory(addr, length)
        if reg_buf == 0:
            return Status(StatusCodes.INVALID)
        desc = HPKVRegisterDescriptor(reg_buf)
        return Status(value=desc)

    @Status.capture_exception
    def deregister_mr(self, desc: ConnectorRegisterDescriptor) -> Status:
        if desc.reg_buf != 0:
            self.conn.dereg_memory(desc.reg_buf)
        desc.reg_buf = 0
        return Status(StatusCodes.OK)

    @Status.capture_exception
    def _exists(self, key: str) -> Status:
        """Check if key is in the store."""
        if self.conn.test(self._key(key)):
            return Status(StatusCodes.OK)
        return Status(StatusCodes.NOT_FOUND)

    @Status.capture_exception
    def _get(self, key: str, mr: MemoryRegion) -> Status[torch.Tensor]:
        """Get a value."""
        desc = mr.register_descriptor()
        if desc is None:
            return Status(StatusCodes.INVALID)
        sgl = self.conn.SGL(mr.data_ptr(), mr.length, desc)
        if self.conn.get(self._key(key), sgl, mr.length) != 0:
            return Status(StatusCodes.ERROR)
        return Status(StatusCodes.OK)

    @Status.capture_exception
    def _put(self, key: str, mr: MemoryRegion) -> Status:
        """Put a key value pair"""
        desc = mr.register_descriptor()
        if desc is None:
            return Status(StatusCodes.INVALID)
        sgl = self.conn.SGL(mr.data_ptr(), mr.length, desc)
        if self.conn.set(self._key(key), sgl) != 0:
            return Status(StatusCodes.ERROR)
        return Status(StatusCodes.OK)

    @Status.capture_exception
    def _delete(self, key: str) -> Status:
        """Delete a key."""
        self.conn.delete_keys(self._key(key))
        return Status(StatusCodes.OK)
