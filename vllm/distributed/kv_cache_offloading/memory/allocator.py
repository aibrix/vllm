import torch

from abc import ABC
from dataclasses import dataclass
from sortedcontainers import SortedDict, SortedList
from threading import Lock
from typing import Iterable, Tuple
from . import RefCountedObj
from ..status import Status, StatusCodes


@dataclass
class MemoryRegionIntl:
    addr: int
    length: int

    @staticmethod
    def is_appendable(src: 'MemoryRegionIntl',
                      dst: 'MemoryRegionIntl') -> bool:
        """
        Check if the src MR can be appended to the dst MR.
        """
        if src is None or dst is None:
            return False

        return src.addr == dst.addr + dst.length

    @staticmethod
    def expand(mr: 'MemoryRegionIntl', expand_size: int) -> 'MemoryRegionIntl':
        """Expand the MR by the given size.
        Args:
            expand_size (int): The size to be expanded.
        Returns:
            The expanded MR.
        """
        if mr is None:
            return mr

        mr.length += expand_size
        return mr


class MemoryRegion(RefCountedObj):
    """A memory region representation used by Allocator."""

    def __init__(
        self,
        allocator: "TensorPoolAllocator",
        addr: int,
        len: int,
    ) -> None:
        super().__init__()
        assert allocator is not None
        self.allocator = allocator
        self.addr = addr
        self.length = len
        self._use_finalizer = True

    def __repr__(self) -> str:
        return f"MemoryRegion(addr={self.addr}, length={self.length}, ref={self.ref_count})"

    def __str__(self) -> str:
        return self.__repr__()

    @staticmethod
    def split(mr: 'MemoryRegion',
              split_size: int) -> Tuple['MemoryRegion', ...]:
        """Split the MR into sub MRs.
        Note:
            Need to delete mr after split to ensure the same memory buffer has no
            more than two MRs referencing it.
        Args:
            split_size (int): size of a single sub MR
        Returns:
            The chunks.
        """
        if mr is None:
            return mr

        with mr._lock:
            mr._use_finalizer = False
            return tuple(
                MemoryRegion(mr.allocator, mr.addr + offset, split_size)
                for offset in range(0, mr.length, split_size))

    def destroy_unsafe(self):
        if self._use_finalizer:
            self.allocator._finalize_mr(self.addr, self.length)

    def to_tensor(self, mr_dtype: torch.dtype,
                  mr_shape: Tuple[int, ...]) -> torch.Tensor:
        """Convert MR to tensor"""
        return self.allocator._buffer[self.addr:self.addr +
                                      self.length].view(mr_dtype).view(
                                          *mr_shape)

    @staticmethod
    def to_tensors(mrs: Iterable['MemoryRegion'], mr_dtype: torch.dtype,
                   mr_shape: Tuple[int, ...]) -> Iterable[torch.Tensor]:
        """Convert MRs to tensors. Contiguous MRs are supposed to form a single tensor."""
        if mrs is None or len(mrs) == 0:
            return mrs

        return [mr.to_tensor(mr_dtype, mr_shape) for mr in mrs]


class TensorPoolAllocator(ABC):

    def __init__(
        self,
        capacity_nbytes: int,
        mr_nbytes: int,
        device: str = "cpu",
        pin_memory: bool = False,
    ) -> None:
        """Initialize the tensor pool allocator.
        Args:
            capacity_nbytes: The capacity of the allocator in bytes.
            mr_nbytes: The size of the memory region in bytes.
            device: The device to allocate the memory on.
            pin_memory: Whether to pin the memory.
        """
        assert capacity_nbytes > 0, f"capacity_nbytes must be greater than 0"
        assert mr_nbytes > 0, f"mr_nbytes must be greater than 0"
        assert mr_nbytes % 2 == 0, f"mr_nbytes must be a multiple of 2"
        assert capacity_nbytes % mr_nbytes == 0, f"capacity_nbytes must be a mutiple of mr_nbytes"

        self.capacity_nbytes: int = capacity_nbytes
        self._used_nbytes: int = 0
        self.mr_nbytes: int = mr_nbytes
        self.device: str = 'cpu' if device is None else device
        self.pin_memory: bool = pin_memory

        # Internal buffer
        self._buffer: torch.Tensor = torch.empty(
            self.capacity_nbytes,
            dtype=torch.uint8,
            device=self.device,
            pin_memory=self.pin_memory,
        )

        init_mr = MemoryRegionIntl(addr=0, length=self._buffer.numel())
        self._mr_list = SortedList([init_mr], key=lambda x: x.addr)

        # Each item is a list of memory regions having the same length
        self._lookup_table = SortedDict()
        self._lookup_table[self._buffer.numel()] = self._mr_list

        self._lock: Lock = Lock()

    def __len__(self) -> int:
        """Return nbytes allocated by the allocator."""
        with self._lock:
            return self._used_nbytes

    def __repr__(self) -> str:
        return f"TensorPoolAllocator(capacity_nbytes={self.capacity_nbytes}, used={self._used_nbytes}, mr_nbytes={self.mr_nbytes}, device={self.device}, pin_memory={self.pin_memory})"

    def __str__(self) -> str:
        return self.__repr__()

    def alloc(self, size: int) -> Status[MemoryRegion]:
        assert size % self.mr_nbytes == 0
        with self._lock:
            if self.capacity_nbytes - self._used_nbytes < size:
                return Status(StatusCodes.OUT_OF_MEMORY)

            # Find the first length that is greater than or equal to size
            idx = self._lookup_table.bisect_left(size)
            if idx >= len(self._lookup_table):
                return Status(StatusCodes.OUT_OF_MEMORY)

            target_mr_len = self._lookup_table.keys()[idx]
            target_mr_list = self._lookup_table[target_mr_len]
            # Get the first memory region from the list
            target_mr = target_mr_list.pop()
            self._mr_list.discard(target_mr)

            # Remove the list if it is empty
            if len(target_mr_list) == 0:
                del self._lookup_table[target_mr_len]

            # Split the memory region if needed
            if target_mr_len > size:
                left_over_mr = MemoryRegionIntl(
                    addr=target_mr.addr + size,
                    length=target_mr.length - size,
                )
                self._mr_list.add(left_over_mr)
                self._lookup_table.setdefault(
                    left_over_mr.length,
                    SortedList(key=lambda x: x.addr)).add(left_over_mr)

            mr = MemoryRegion(self, target_mr.addr, size)
            self._used_nbytes += size
            return Status(value=mr)

    def _finalize_mr(self, addr: int, length: int) -> None:
        with self._lock:
            mr = MemoryRegionIntl(addr, length)
            self._used_nbytes -= mr.length
            assert self._used_nbytes >= 0, f"double free memory region"
            # Find the index of the memory region in the list
            idx = self._mr_list.bisect_right(mr)
            prev = self._mr_list[idx - 1] if idx > 0 else None
            next = self._mr_list[idx] if idx < len(self._mr_list) else None

            curr = mr
            # 1. append mr to prev if possible
            if MemoryRegionIntl.is_appendable(curr, prev):
                # Remove prev from the list and lookup table
                self._mr_list.discard(prev)
                prev_len_list = self._lookup_table[prev.length]
                prev_len_list.discard(prev)
                # Remove the list if it is empty
                if len(prev_len_list) == 0:
                    del self._lookup_table[prev.length]
                # Append curr to prev
                curr = MemoryRegionIntl.expand(prev, curr.length)

            # curr = prev + mr if append happened
            # 2. append next to curr if possible
            if MemoryRegionIntl.is_appendable(next, curr):
                # Remove next from the list and lookup table
                self._mr_list.discard(next)
                next_len_list = self._lookup_table[next.length]
                next_len_list.discard(next)
                # Remove the list if it is empty
                if len(next_len_list) == 0:
                    del self._lookup_table[next.length]
                # Append next to curr
                curr = MemoryRegionIntl.expand(curr, next.length)

            # 3. insert curr into the list and lookup table
            self._mr_list.add(curr)
            self._lookup_table.setdefault(
                curr.length, SortedList(key=lambda x: x.addr)).add(curr)

    @property
    def num_memory_regions(self) -> int:
        """Return the number of memory regions."""
        with self._lock:
            return len(self._mr_list)

    def assert_consistency(self) -> None:
        """Assert that the allocator is consistent. For test purpose."""
        with self._lock:
            # 1. check mr list
            mr_list_total_nbytes = 0
            for i in range(len(self._mr_list)):
                mr_i = self._mr_list[i]
                mr_list_total_nbytes += mr_i.length
                assert mr_i.length > 0, f"{mr_i.length} <= 0"
                assert (mr_i.length in self._lookup_table
                        ), f"len={mr_i.length} not in lookup_table"
                assert (mr_i in self._lookup_table[mr_i.length]
                        ), f"{mr_i} not in lookup_table[{mr_i.length}]"
                if i > 0:
                    mr_i_prev = self._mr_list[i - 1]
                    assert (mr_i_prev.addr + mr_i_prev.length < mr_i.addr
                            ), f"{mr_i_prev} and {mr_i} are not disjoint"
            assert (
                mr_list_total_nbytes == self.capacity_nbytes -
                self._used_nbytes
            ), f"{mr_list_total_nbytes} != {self.capacity_nbytes} - {self._used_nbytes}"
            # 2. check lookup table
            lookup_table_total_nbytes = 0
            for mr_len, mr_list in self._lookup_table.items():
                assert mr_len > 0, f"{mr_len} <= 0"
                for mr in mr_list:
                    assert mr_len == mr.length, f"{mr_len} != {mr.length}"
                    assert mr in self._mr_list, f"{mr} not in mr_list"
                lookup_table_total_nbytes += mr_len * len(mr_list)
            assert (lookup_table_total_nbytes == mr_list_total_nbytes
                    ), f"{lookup_table_total_nbytes} != {mr_list_total_nbytes}"
