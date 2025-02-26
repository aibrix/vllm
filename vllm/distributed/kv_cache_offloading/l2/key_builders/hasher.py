import hashlib

from abc import ABC, abstractmethod
from typing import Iterable


class Hasher(ABC):

    @abstractmethod
    def hash(self, data: bytes) -> int:
        """Compute a 128-bit hash from the given data.
        
        Args:
            data (bytes): Input data to hash.
        
        Returns:
            int: 128-bit unsigned integer hash value.
        """
        raise NotImplementedError


class MD5Hasher(Hasher):

    def hash(self, data: bytes) -> int:
        return int(hashlib.md5(data).hexdigest(), 16)
