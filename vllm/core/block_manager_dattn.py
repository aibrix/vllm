'''
 Copyright (c) ByteDance Inc.
 Authors:
  - Tongping Liu (tongping.liu@bytedance.com)

  Adopted from https://github.com/vllm-project/vllm/pull/6102/commits
'''
from collections import deque
from typing import Dict, List, Optional, Tuple

from vllm.core.block.utils import check_no_caching_or_swa_for_blockmgr_encdec
from vllm.core.evictor_v1 import EvictionPolicy, Evictor, make_evictor
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.logger import init_logger
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device, Counter
from collections import deque

logger = init_logger(__name__)


class CacheBufferAllocator:
    def __init__(self, num_cache_buffers: int):
        self.num_cache_buffers = num_cache_buffers
        self.free_buffers = deque(range(num_cache_buffers))

    def allocate(self) -> int:
        buffer_id = self.free_buffers.popleft()
        return buffer_id

    def free(self, buffer_id: int):
        self.free_buffers.append(buffer_id)

    def reset(self):
        self.free_buffers = deque(range(self.num_cache_buffers))

    def get_num_free_buffers(self):
        return len(self.free_buffers)

    def get_num_total_buffers(self):
        return self.num_cache_buffers


class BlockSpaceManagerDAttn(BlockSpaceManager):
    """Manages the mapping between logical and physical token blocks."""

    def __init__(
            self,
            block_size: int,
            num_gpu_blocks: int,
            num_cpu_blocks: int,
            watermark: float = 0.01,
            sliding_window: Optional[int] = None,
            enable_caching: bool = False,
            num_cache_buffers: int = 0,
    ) -> None:

        if enable_caching or (sliding_window is not None):
            raise NotImplementedError("Prefix Caching or Sliding window is not supported in VMM now.")

        self.enable_caching = enable_caching

        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks

        self.num_free_gpu_blocks = num_gpu_blocks
        self.num_free_cpu_blocks = num_cpu_blocks

        self.num_cache_buffers = num_cache_buffers  # == self.scheduler_config.max_num_seqs

        print(f"self.num_cache_buffers:{self.num_cache_buffers} inside BlockSpaceManagerDAttn")
        # use to alloc cache buffer id for seq
        self.gpu_allocator = CacheBufferAllocator(num_cache_buffers)

        # Watermark indicates that the least amount of blocks should be free.
        self.watermark = watermark
        assert watermark >= 0.0

        self.watermark_blocks = int(watermark * num_gpu_blocks)

        # Mapping from cache buffer ID to the number of allocated blocks.
        self.allocated_block_counts: Dict[int, int] = {}
        self.modified_block_counts: Dict[int, int] = {}
        self.waiting_free_buffers: List[Tuple[int, int]] = []
        self.waiting_free_blocks: int = 0
        self.free_buffer_ids: List[int] = []
        self.free_latency: int = 10

        # TODO: this is very confusing
        self.iter_counter = Counter()

        self._init_alloc()

    def _init_alloc(self) -> None:
        # we init alloc one block for warp in cache_engine_vmm
        self.allocated_block_counts[0] = 1
        self.num_free_gpu_blocks -= 1

    def _predict_gen_len(self, seq: Sequence) -> int:
        # TODO:this function is used to predict the generated content length,
        # which can used to pre allocate the memory handles
        return 1

    def _get_seq_num_required_blocks(self, seq: Sequence) -> int:
        return 0 if seq is None else seq.n_blocks

    # This will be invoked in the prefill phase
    def can_allocate(self, seq_group: SequenceGroup) -> AllocStatus:
        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.

        check_no_caching_or_swa_for_blockmgr_encdec(self, seq_group)

        # get_seqs will collect a list of sequence with status equalling to SequenceStatus.WAITING
        # then we will get the first sequence in this group
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        num_required_blocks = self._get_seq_num_required_blocks(seq)
        num_required_blocks += self._predict_gen_len(seq)

        # If the sequence is not allocated yet, its cache_buffer_id must be -1.
        assert seq.cache_buffer_id == -1

        num_free_gpu_blocks = self.num_free_gpu_blocks + \
                              self.waiting_free_blocks

        # Ensure that one request should not use more than 90% or 99% of memory
        # This can avoid frequent cache eviction
        if (self.num_total_gpu_blocks - num_required_blocks <
                self.watermark_blocks):
            return AllocStatus.NEVER
        if num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks:
            # Make sure that we are not holding more than schedule_config.max_num_seqs
            # TODO: is there a potential issue? If self.gpu_allocator.get_num_free_buffers()=0, and
            # waiting_free_buffer exists (only one will exit soon), then can multiple requests get admitted?
            if self.gpu_allocator.get_num_free_buffers() > 0 or self.waiting_free_buffers:
                return AllocStatus.OK
            else:
                return AllocStatus.LATER
        else:
            return AllocStatus.LATER

    # This function is only invoked by _allocate_and_set_running (invoked by _schedule_prefills)
    # That is, it is allocated when admitting a new request in prefill phase.
    # Therefore, it will invoke self._allocate_buffer() to allocate a request and then
    # update the seq.cache_buffer_id, seq.data.cache_buffer_id, self.allocated_block_counts[buffer_id]
    # TODO: for instance, if there is a request with 26 tokens, then it will need two
    # blocks??
    def allocate(self, seq_group: SequenceGroup) -> None:
        # No need to do this, as we have checked before
        # if seq_group.is_encoder_decoder():
        #    raise NotImplementedError("Encoder-decoder is not supported in VMM now.")

        # check_no_caching_or_swa_for_blockmgr_encdec(self, seq_group)

        # Allocate decoder sequences
        #
        # NOTE: Here we assume that all sequences in the group have the same
        # decoder prompt.
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]

        need_blocks_num = self._get_seq_num_required_blocks(seq)

        # TODO: Don't know why we will need this _predict_gen_len??
        need_blocks_num += self._predict_gen_len(seq)
        buffer_id, allocated_num = self._allocate_buffer(need_blocks_num)

        seq.cache_buffer_id = buffer_id
        seq.data.cache_buffer_id = buffer_id
        self.allocated_block_counts[buffer_id] = allocated_num
        self.modified_block_counts[buffer_id] = allocated_num
        # predict generate content length and pre allocate the blocks
        # need_blocks_num += self._predict_gen_len(seq)

    def _allocate_buffer(self, need_blocks_num: int) -> Tuple[int, int]:
        if self.waiting_free_buffers:
            return self._allocate_from_waiting_buffer(need_blocks_num)
        else:
            assert self.num_free_gpu_blocks >= need_blocks_num
            buffer_id = self.gpu_allocator.allocate()
            self.num_free_gpu_blocks -= need_blocks_num
            return buffer_id, need_blocks_num

    """
    Allocate need_blocks_num from waiting buffer that holds freed sequences 
    """

    def _allocate_from_waiting_buffer(self,
                                      need_blocks_num: int) -> Tuple[int, int]:
        buffer_id, _ = self.waiting_free_buffers.pop(0)
        allocated_num = self.allocated_block_counts[buffer_id]
        self.waiting_free_blocks -= allocated_num

        # If the number of blocks is not sufficient, let's allocate more blocks.
        # However, I don't know whether these new blocks are related to the given req_id
        if allocated_num < need_blocks_num:
            # TODO:  this allocation has the issue, as it can't guarantee that
            # the blocks are allocated from the specified request id.
            self._allocate_extra_blocks(need_blocks_num - allocated_num)
            allocated_num = need_blocks_num
        # If we reuse a buffer that's too long, we may need to free the memory
        # that's more than we currently need (need_blocks_num)
        # But now, frequent frees are an overhead, so we don't do it.
        # TODO: Reduced overhead or asynchronous free
        # else:
        #     self.num_free_gpu_blocks += (allocated_num - need_blocks_num)
        #     allocated_num = need_blocks_num

        return buffer_id, allocated_num

    def _allocate_extra_blocks(self, extra_blocks: int) -> None:
        if self.num_free_gpu_blocks >= extra_blocks:
            # It is actually deducted free_gpu_blocks.
            self.num_free_gpu_blocks -= extra_blocks
        else:
            extra_need_blocks = extra_blocks - self.num_free_gpu_blocks
            self.num_free_gpu_blocks = 0

            self._allocate_from_waiting_buffers(extra_need_blocks)

    # free some blocks from waiting buffers to allocate
    # The name is very confusing, as it is similar to _allocate_from_waiting_buffer
    def _allocate_from_waiting_buffers(self, blocks_to_alloc: int) -> None:
        while self.waiting_free_buffers and blocks_to_alloc > 0:
            free_id, _ = self.waiting_free_buffers.pop(0)
            free_blocks = self.allocated_block_counts[free_id]
            self.waiting_free_blocks -= free_blocks
            self.free_buffer_ids.append(free_id)
            self.allocated_block_counts[free_id] = 0
            blocks_to_alloc -= free_blocks

        assert blocks_to_alloc <= 0
        self.num_free_gpu_blocks -= blocks_to_alloc

    # Invoked by _schedule_running in running phase.
    def can_append_slots(self,
                         seq_group: SequenceGroup,
                         num_lookahead_slots: int = 0) -> bool:
        assert (num_lookahead_slots == 0
                ), "lookahead allocation not supported in BlockSpaceManagerDAttn."

        # FIXME: this is wrong for vAttention, as it requires many blocks for
        # a token (unless its num_free_gpu_blocks already consider the number of layers )
        # Simple heuristic: If there is at least one free block
        # for each sequence, we can append.
        num_seqs = seq_group.num_seqs(status=SequenceStatus.RUNNING)
        num_free_gpu_blocks = self.num_free_gpu_blocks + \
                              self.waiting_free_blocks
        return num_seqs <= num_free_gpu_blocks

    # FIXME: there is no handling on num_lookahead_slots, which should be handled.
    def append_slots(
            self,
            seq: Sequence,
            num_lookahead_slots: int = 0,
    ) -> List[Tuple[int, int]]:
        """Allocate a physical slot for a new token."""

        buffer_id = seq.cache_buffer_id

        # If the sequence is allocated, its cache_buffer_id must >= 0.
        assert buffer_id >= 0

        logical_blocks_num = seq.n_blocks
        allocated_num = self.allocated_block_counts[buffer_id]

        # If we need to allocate a new physical block
        if allocated_num < logical_blocks_num:
            # Currently this code only supports adding one physical block
            assert allocated_num == logical_blocks_num - 1

            # Added one new block??? Why, this is confusing?
            self._allocate_extra_blocks(1)
            self.allocated_block_counts[buffer_id] = logical_blocks_num
            self.modified_block_counts[buffer_id] = logical_blocks_num
            return []

        else:
            # the last block is not full, no need to allocate a new block
            return []

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        raise NotImplementedError("Forking is not supported in BlockSpaceManagerVMM now.")

    def can_swap_in(self, seq_group: SequenceGroup,
                    num_lookahead_slots: int) -> AllocStatus:
        raise NotImplementedError("Swap-in is not supported in BlockSpaceManagerVMM now.")

    def swap_in(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        raise NotImplementedError("Swap-in is not supported in BlockSpaceManagerVMM now.")

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        raise NotImplementedError("Swap-out is not supported in BlockSpaceManagerVMM now.")

    def swap_out(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        raise NotImplementedError("Swap-out is not supported in BlockSpaceManagerVMM now.")

    """
    Free a sequence. We will append the seq to waiting_free_buffers. 
    Initially, we did this inside the memory management library. Maybe we should do it here as well. 
    """

    def free(self, seq: Sequence) -> None:
        # Here, we just append free seq to waiting_free_buffers.
        waiting_free_id = seq.cache_buffer_id

        # If no blocks are allocated in the sequence, then this sequence may be deallocated.
        if waiting_free_id not in self.allocated_block_counts or \
                self.allocated_block_counts[waiting_free_id] == 0:
            # Already freed or haven't been scheduled yet.
            return

        # Get free_blocks of this sequence
        free_blocks = self.allocated_block_counts[waiting_free_id]
        self.waiting_free_buffers.append((waiting_free_id,
                                          self.iter_counter.counter))
        self.waiting_free_blocks += free_blocks

    def reset(self) -> None:
        # Free decoder block tables
        self.allocated_block_counts.clear()
        self.num_free_gpu_blocks = self.num_total_gpu_blocks
        self.num_free_cpu_blocks = self.num_total_cpu_blocks

        self.waiting_free_buffers = []
        self.modified_block_counts = {}
        self.free_buffer_ids = []
        self.gpu_allocator.reset()

    def get_block_table(self, seq: Sequence) -> List[int]:
        # logger.warning("block table is not used in BlockSpaceManagerVMM now.")
        return []

    def get_num_free_gpu_blocks(self) -> int:
        return self.num_free_gpu_blocks

    def get_num_free_cpu_blocks(self) -> int:
        return self.num_free_cpu_blocks

    def access_all_blocks_in_seq(
            self,
            seq: Sequence,
            access_time: float,
    ) -> None:
        # logger.warning("Access all blocks in seq is not supported in BlockSpaceManagerVMM now.")
        pass

    def get_common_computed_block_ids(self,
                                      seq_group: SequenceGroup) -> List[int]:
        # logger.warning("Common computed block ids is not supported in BlockSpaceManagerVMM now.")
        return None  # type: ignore

    def mark_blocks_as_computed(self, seq_group: SequenceGroup) -> None:
        # logger.warning("Mark blocks as computed is not supported in BlockSpaceManagerVMM now.")
        pass

    def get_allocated_block_count(self, seq_id: int) -> int:
        return self.allocated_block_counts[seq_id]

    def check_and_free_waiting_buffers(self, now_iter: int) -> None:
        while self.waiting_free_buffers and \
                self.waiting_free_buffers[0][1] - now_iter >= self.free_latency:
            free_id, _ = self.waiting_free_buffers.pop(0)
            free_blocks = self.allocated_block_counts[free_id]
            self.waiting_free_blocks -= free_blocks
            self.num_free_gpu_blocks += free_blocks
            self.free_buffer_ids.append(free_id)
            self.allocated_block_counts[free_id] = 0

    def step(self) -> Tuple[Dict[int, int], List[int]]:
        # next() is a built-in function for the iterator, which will execute __next__()
        iter = next(self.iter_counter)
        modified_block_counts = self.modified_block_counts
        free_buffer_ids = self.free_buffer_ids

        # Whether we need to invoke this before returning free_buffer_ids??
        self.check_and_free_waiting_buffers(iter)

        # step() is invoked once after _schedule() inside Scheduler::schedule(). It is invoked once for every decode or prefill
        # We actually uses self.free_buffer_ids and self.modified_block_counts to track all requests
        # checked by the whole _schedule(). This is a hacky solution but may work correctly.
        self.modified_block_counts = {}
        self.free_buffer_ids = []
        return modified_block_counts, free_buffer_ids
