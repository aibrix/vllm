# SPDX-License-Identifier: Apache-2.0
from .base_eviction_policy import BaseEvictionPolicy, Functor
from .fifo import FIFO
from .lru import LRU
from .s3fifo import S3FIFO

__all__ = ["BaseEvictionPolicy", "Functor", "FIFO", "LRU", "S3FIFO"]
