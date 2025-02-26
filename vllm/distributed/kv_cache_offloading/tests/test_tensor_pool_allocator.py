import pytest
import random
from ..memory import MemoryRegion, TensorPoolAllocator


@pytest.fixture
def allocator():
    return TensorPoolAllocator(1024 * 1024, 16)


def test_basic_allocation(allocator):
    """Test basic allocation and deallocation."""
    size = 1024
    status = allocator.alloc(size)
    allocator.assert_consistency()
    assert status.is_ok()
    assert len(allocator) == size
    mr = status.value
    assert mr.length == size
    assert allocator.num_memory_regions == 1
    mr.ref_down()  # Trigger garbage collection
    assert len(allocator) == 0
    allocator.assert_consistency()
    assert allocator.num_memory_regions == 1


def test_coalescing_mechanism(allocator):
    """Test memory coalescing when MRs are deallocated."""
    size1, size2, size3 = 1024, 2048, 1024

    # Allocate three MRs
    status1 = allocator.alloc(size1)
    assert allocator.num_memory_regions == 1
    status2 = allocator.alloc(size2)
    assert allocator.num_memory_regions == 1
    status3 = allocator.alloc(size3)
    assert allocator.num_memory_regions == 1

    assert status1.is_ok()
    assert status2.is_ok()
    assert status3.is_ok()
    assert len(allocator) == size1 + size2 + size3

    mr1 = status1.value
    assert mr1.length == size1
    mr2 = status2.value
    assert mr2.length == size2
    mr3 = status3.value
    assert mr3.length == size3

    # Free the middle allocation first
    mr2.ref_down()
    allocator.assert_consistency()
    assert allocator.num_memory_regions == 2

    # Free the first and last allocations
    mr1.ref_down()
    allocator.assert_consistency()
    # mr1 got merged with mr2
    assert allocator.num_memory_regions == 2
    mr3.ref_down()
    allocator.assert_consistency()
    # all memory regions got merged into one
    assert allocator.num_memory_regions == 1

    assert len(allocator) == 0


def test_split(allocator):
    """Test split."""
    size = 1024
    status = allocator.alloc(size)
    allocator.assert_consistency()
    assert status.is_ok()
    assert len(allocator) == size
    mr = status.value
    assert mr.length == size
    assert allocator.num_memory_regions == 1

    mrs = list(MemoryRegion.split(mr, 16))
    assert len(allocator) == 1024

    used = 1024
    addr = mr.addr
    for i in range(len(mrs)):
        assert addr == mrs[i].addr
        assert mrs[i].length == 16
        mrs[i].ref_down()  # Trigger garbage collection
        used -= 16
        addr += 16
        assert len(allocator) == used

        allocator.assert_consistency()
        if i < len(mrs) - 1:
            assert allocator.num_memory_regions == 2
        else:
            assert allocator.num_memory_regions == 1


def test_out_of_memory(allocator):
    """Test allocator behavior when requesting more memory than available."""
    max_size = 1 << 30  # 1GB for testing
    status = allocator.alloc(max_size)
    assert status.is_out_of_memory()


@pytest.mark.parametrize("rseed", [i * 100 + 43 for i in range(13)])
def test_stress_allocation(allocator, rseed):
    """Stress test: allocate and free many small blocks to test fragmentation and coalescing."""
    random.seed(rseed)

    num_allocations = 1000
    sizes = [16, 64, 128, 512]
    mrs = []

    allocated_size = 0
    for i in range(num_allocations):
        random.shuffle(sizes)
        size = sizes[i % len(sizes)]
        status = allocator.alloc(size)
        assert status.is_ok()
        mr = status.value
        mrs.append(mr)
        allocated_size += size
        assert mr.length == size
        assert len(allocator) == allocated_size
        allocator.assert_consistency()

    random.shuffle(mrs)
    for i in range(len(mrs)):
        mr = mrs[i]
        mr.ref_down()

        size = mr.length
        allocated_size -= size
        assert len(allocator) == allocated_size
        allocator.assert_consistency()

    assert len(allocator) == 0
