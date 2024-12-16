'''
 Copyright (c) ByteDance Inc.
 Authors: 
  - Tongping Liu (tongping.liu@bytedance.com)

  This file will manage blocks and cache ids for both CPU and GPU memory. 
  However, the address related to each cache id will be tracked and managed by CacheEngineDattn

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
from dataclasses import dataclass, field
import sys

logger = init_logger(__name__)

class CacheAllocator:
    def __init__(self, name: str, num_caches: int):
        self.num_caches = num_caches
        self.type = name 
        # kv_caches: tracking the available cache ids
        self.kv_caches = deque(range(num_caches))

    def allocate(self) -> int:
        cache_id = self.kv_caches.popleft() 
        #if self.type == "cpu" and len(self.kv_caches) == 0:
        #    print(f"ERROR: self.kv_caches is 000000000 NOW")
        #elif self.type == "cpu":
        #    print(f"ERROR checking: allocated a cpu cache:{cache_id}, remaining cache:{len(self.kv_caches)}") 
        return cache_id

    def free(self, cache_id: int):
        # FIXME
        self.kv_caches.appendleft(cache_id)
        #self.kv_caches.append(cache_id)

    def get_free_caches(self):
        return len(self.kv_caches)

class SwappedCPUCache:
    def __init__(self, cache_id, blocks):
        self.cache_id = cache_id
        self.blocks = blocks

class BlockSpaceManagerDAttn(BlockSpaceManager):
    """Manages the mapping between logical and physical token blocks."""

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.03,
        sliding_window: Optional[int] = None, # Not supported
        enable_caching: bool = False, # Not supported
        vmm_frequency: int = 16, 
        num_caches: int = 0,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks

        # For every 16 steps, we will perform vmm updates by invoking update_cache_blocks
        self.vmm_frequency = vmm_frequency
        self.vmm_frequency_mask = vmm_frequency - 1 
        
        # Useful when admitting new requests or swapping in some requests. 
        # Then we prefer those requests that just exit.  
        self.cached_free_gpu_blocks: int = 0

        # Tracking the number of gpu_blocks (including self.cached_free_gpu_blocks) 
        self.num_free_gpu_blocks = num_gpu_blocks
        self.num_free_cpu_blocks = num_cpu_blocks
        
        num_gpu_caches = num_caches
        num_cpu_caches = int(num_caches/20)

        # use to alloc cache buffer id for seq
        self.gpu_allocator = CacheAllocator("cuda", num_gpu_caches)
        self.cpu_allocator = CacheAllocator("cpu", num_cpu_caches)

        # Watermark indicates that the least amount of blocks should be free. 
        assert watermark >= 0.0
        self.watermark_blocks = int(watermark * num_gpu_blocks)

        # Mapping from cache_id to the number of allocated blocks.
        # The information is more persitent across different steps
        self.allocated_gpu_blocks: Dict[int, int] = {} 
        # Pre-allocate one block for the first cache,  to support graph capture
        self.allocated_gpu_blocks[0] = 1 

        # Temporary buffer for each step. self.step() will collect these information and freed all 
        # caches of to_free_gpu_caches
        self.to_allocate_blocks: Dict[int, int] = {} 

        # to_free_gpu_caches keeps the requests that are freed in the current step
        self.to_free_gpu_caches: Dict[int, int] = {}
        self.immediate_allocate = False

        # Maintain the mapping between seq.req_id and SwappedCPUCache (cache_id, blocks)
        self.swapped_caches: Dict[int, SwappedCPUCache] = {}

        self.step_index = 0
    
    def _predict_n_blocks(self, tokens: int, is_prefill: bool = False) -> int:
        if tokens == 0:
            return 0
        
        if is_prefill:
            return (tokens + self.vmm_frequency - (self.step_index & self.vmm_frequency_mask) + self.block_size - 1) // self.block_size 
        else:
            return (tokens + self.vmm_frequency + self.block_size - 1) // self.block_size

    def _check_availability(self, need_blocks) -> AllocStatus:
        num_free_gpu_blocks = self.num_free_gpu_blocks + self.cached_free_gpu_blocks

        # Ensure that one request should not use more than 90% or 99% of memory
        # This can avoid frequent cache eviction 
        if (self.num_total_gpu_blocks - need_blocks < self.watermark_blocks):
            return AllocStatus.NEVER
        
        if num_free_gpu_blocks - need_blocks >= self.watermark_blocks:
            # Make sure that we are not holding more than schedule_config.max_num_seqs
            if self.gpu_allocator.get_free_caches() > 0 or len(self.to_free_gpu_caches) > 0:
                return AllocStatus.OK
            else:
                return AllocStatus.LATER
        else:
            return AllocStatus.LATER

    # This function is invoked only in the prefill phase
    def can_allocate(self, seq_group: SequenceGroup) -> AllocStatus:
        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.
        check_no_caching_or_swa_for_blockmgr_encdec(self, seq_group)

        # get_seqs will collect a list of sequence with status equalling to SequenceStatus.WAITING
        # then we will get the first sequence in this group 
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]

        self_num_required_blocks = self._predict_n_blocks(tokens=seq.get_len(), is_prefill=True)
        cross_seq = seq_group.get_encoder_seq()
        cross_num_required_blocks = 0 
        if cross_seq:
            cross_num_required_blocks = self._predict_n_blocks(
                    tokens = cross_seq.get_len(), is_prefill=True)

        num_required_blocks = self_num_required_blocks + \
                              cross_num_required_blocks

        return self._check_availability(num_required_blocks)

    # This is to swap_in an pre-existing block, which is slightly different 
    # from can_allocate(). 
    def can_swap_in(self, seq_group: SequenceGroup,
                    num_lookahead_slots: int) -> AllocStatus:
        need_blocks = num_lookahead_slots
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            if seq.is_finished():
                continue
            
            req_id = seq.seq_id

            need_blocks += self.swapped_caches[req_id].blocks
        
        # Adopted from the block_manager_v1. 
        need_blocks += seq_group.num_seqs(status=SequenceStatus.SWAPPED)
        if seq_group.is_encoder_decoder():
            need_blocks += 1
        
        return self._check_availability(need_blocks) 

    # This function is only invoked by _allocate_and_set_running (invoked by _schedule_prefills)
    # Allocate a GPU cache when admitting a new request in prefill phase.
    def allocate(self, seq_group: SequenceGroup) -> None:
        # Allocate decoder sequences
        #
        # NOTE: Here we assume that all sequences in the group have the same
        # decoder prompt.
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        
        need_blocks = self._predict_n_blocks(tokens=seq.get_len(), is_prefill=True)
        
        cache_id = self._allocate_gpu_cache(need_blocks = need_blocks, allocate_now = True)
        
        #print(f"step_index-{self.step_index}, allocate cache_id: {cache_id}, need_blocks:{need_blocks}, tokens:{seq.get_len()}") 
        seq.cache_id = cache_id
        seq.data.cache_id = cache_id
        
    #  Allocate a new GPU cache, when the available GPU blocks are sufficient
    def _allocate_gpu_cache(self, need_blocks: int, allocate_now: bool) -> Tuple[int, int]:
        cache_id = -1
        to_allocate_num = need_blocks
        allocated_block_num = need_blocks

        # Prefer to reuse the to_free_gpu_caches at first, as some pages have been allocated already. 
        if self.cached_free_gpu_blocks > 0:
            # Make it block_diff a big number for the better comparison
            block_diff = need_blocks*100
        
            # Find one kv_cache with the smallest difference on the number of blocks
            # The found cache can have more or less available blocks.   
            for id, num_blocks in self.to_free_gpu_caches.items():
                diff = abs(num_blocks - need_blocks)
                
                # kv_cache : cache_id, blocks 
                if diff < block_diff:
                    cache_id = id
                    block_diff = diff

                    allocated_block_num = num_blocks

                    # No need to check anymore if we already found a perfect one
                    if diff == 0:
                        break 
            
            # Remove this item from the to_free_gpu_caches
            del self.to_free_gpu_caches[cache_id]

            # After the loop, allocated_block_num will be the number of blocks for the best cache id 
            # We will assign all blocks for the request now.
            # FIXME: we may use a smaller number if allocated_block_num is too big
            self.cached_free_gpu_blocks -= allocated_block_num

            if allocated_block_num < need_blocks:
                to_allocate_num = need_blocks - allocated_block_num
                
                # update the allocated number
                allocated_block_num = need_blocks
            else:
                # allocated_block_num == num_blocks 
                # No need to allocate more blocks
                to_allocate_num = 0   
        else:
            # Check whether the can_allocate or can_swap_in has a bug
            #if self.num_free_gpu_blocks < need_blocks: 
            #    print(f"Error: self.num_free_gpu_blocks:{self.num_free_gpu_blocks}, need_blocks:{need_blocks}")
            assert self.num_free_gpu_blocks >= need_blocks

            cache_id = self.gpu_allocator.allocate()
            
            #print(f"_allocate_buffer new, need_blocks:{need_blocks}, cache_id:{cache_id}", file=sys.stderr)
        self.num_free_gpu_blocks -= to_allocate_num
        self.allocated_gpu_blocks[cache_id] = allocated_block_num
        if to_allocate_num > 0: 
            self.to_allocate_blocks[cache_id] = allocated_block_num 

        # When admitting a new request or swap in a request, we require the immediate allocation.         
        self.immediate_allocate = allocate_now

        return cache_id

    # Invoked by _schedule_running in running phase.  
    def can_append_slots(self,
                         seq_group: SequenceGroup,
                         num_lookahead_slots: int = 0) -> bool:

        # Simple heuristic: If there is at least one free block
        # for each sequence, we can append.
        num_seqs = seq_group.num_seqs(status=SequenceStatus.RUNNING)

        if num_seqs > self.num_free_gpu_blocks + self.cached_free_gpu_blocks:
            print(f"STOP now!!!!!!Cannot append slots, num_seqs:{num_seqs}, self.num_free_gpu_blocks:{self.num_free_gpu_blocks}, self.cached_free_gpu_blocks:{self.cached_free_gpu_blocks}") 
        return num_seqs < self.num_free_gpu_blocks + self.cached_free_gpu_blocks

    # FIXME: there is no handling on num_lookahead_slots, which should be handled.  
    def append_slots(
        self,
        seq: Sequence,
        num_lookahead_slots: int = 0,
    ) -> List[Tuple[int, int]]:
        if self.step_index & self.vmm_frequency_mask:
            return []

        """Allocate a physical token/slot for a new token."""
        cache_id = seq.cache_id

        # If the sequence is allocated, its cache_id must >= 0.
        assert cache_id >= 0
        
        logical_blocks_num = self._predict_n_blocks(seq.get_len())
        allocated_block_num = self.allocated_gpu_blocks[cache_id]

        #if (self.step_index & self.vmm_frequency_mask) == 0:
        #    print(f"seq_id:{seq.seq_id}, cache_id:{cache_id}, tokens:{seq.get_len()}, logical_blocks_num:{logical_blocks_num}, allocated_block_num:{allocated_block_num}, free_blocks:{self.num_free_gpu_blocks}")

        # If we need to allocate a new physical block
        if allocated_block_num < logical_blocks_num:
            if allocated_block_num != logical_blocks_num - 1: 
                print(f"append_slots cache_id:{cache_id}, logical_blocks_num:{logical_blocks_num} - {allocated_block_num}, real tokens:{seq.get_len()}", file=sys.stderr) 
            
            # Currently this code only supports adding one physical block in the decoding phase
            assert allocated_block_num == logical_blocks_num - 1

            free_blocks = self.num_free_gpu_blocks 
            self.num_free_gpu_blocks -= logical_blocks_num - allocated_block_num 
            if self.num_free_gpu_blocks <= 0:
                print(f"ERROR: self.num_free_gpu_blocks:{self.num_free_gpu_blocks}, cache_id:{cache_id}, free_blocks:{free_blocks}, logical_blocks_num:{logical_blocks_num}, allocated_block:{allocated_block_num}")

            self.allocated_gpu_blocks[cache_id] = logical_blocks_num

            # Note that to_allocate_blocks actually hold the logic blocks number, not a bug. 
            self.to_allocate_blocks[cache_id] = logical_blocks_num 

        # No need to return anything here, since step() will collect all information
        # related to the current scheduling phase.            
        return []

    # Collect the number of physical blocks used by this sequence group 
    def _get_physical_blocks(
            self, seq_group: SequenceGroup):
        
        blocks = 0
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                continue

            cache_id = seq.cache_id
            blocks += self.allocated_gpu_blocks[cache_id]
        
        return blocks

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        raise NotImplementedError("Forking is not supported in BlockSpaceManagerDAttn now.")

    # A fucntion is invoked to figure out the blocks that need to be allocated. 
    def swap_in(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        to_swap_in_caches = []

        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            cpu_cache = self.swapped_caches[seq.seq_id] 
            need_blocks = cpu_cache.blocks
            cpu_cache_id = cpu_cache.cache_id

            # Free cpu cache id and update the counter
            
            self.cpu_allocator.free(cpu_cache_id)
            self.num_free_cpu_blocks += need_blocks  
            
            # Allocate a gpu cache id, based on the need_blocks. 
            # Note that we specifically request one more block in order to accomodate vmm_frequency's memory management
            gpu_cache_id = self._allocate_gpu_cache(need_blocks=need_blocks+1, allocate_now = False)

            seq.cache_id = gpu_cache_id
            seq.data.cache_id = gpu_cache_id
            # NOTE: we may not need the allocation, if gpu_cache_id 
            #print(f"SWAPIN seq_id:{seq.seq_id} with tokens:{seq.get_len()}, cpu_cache_id:{cpu_cache_id}, gpu_cache_id:{gpu_cache_id}, allocated_blocks:{self.allocated_gpu_blocks[gpu_cache_id]}, free_gpu_blocks:{self.num_free_gpu_blocks}")
            to_swap_in_caches.append([cpu_cache_id, gpu_cache_id, need_blocks])
            

        return to_swap_in_caches

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        return self._get_physical_blocks(seq_group) <= self.num_free_cpu_blocks

    def swap_out(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
    
        to_swap_out_caches = []

        for seq in seq_group.get_seqs():
            if seq.is_finished():
                continue

            # Find the cache id and gpu_blocks        
            gpu_cache_id = seq.cache_id
            num_gpu_blocks = self.allocated_gpu_blocks[gpu_cache_id] 

            # Free the cache related to gpu_cache_id
            self._free_cache(cache_id =gpu_cache_id, immediate_free = True)

            # Allocate the cpu cache id
            cpu_cache_id = self.cpu_allocator.allocate()
            cpu_cache = SwappedCPUCache(cpu_cache_id, num_gpu_blocks) 
            self.swapped_caches[seq.seq_id] = cpu_cache

            # After the swapped out, num_free_cpu_blocks should be decremented 
            self.num_free_cpu_blocks -= num_gpu_blocks
            
            #print(f"SWAPOUT {seq.seq_id} with tokens-{seq.get_len()}, cpu_cache_id:{cpu_cache_id}, gpu_cache_id:{gpu_cache_id}, blocks:{num_gpu_blocks} at step-{self.step_index}")
            
            to_swap_out_caches.append([gpu_cache_id, cpu_cache_id, num_gpu_blocks]) 

        #print(f"to_swap_out_caches:{to_swap_out_caches}")
        return to_swap_out_caches

    # When immediate_free is False, we will add the cache
    def _free_cache(self, cache_id: int, immediate_free: bool) -> None:
        # Check whether cache_id is in the list
        if cache_id in self.to_free_gpu_caches:
            # Already freed yet, no need to do anything.
            return

        # Get blocks of this cache
        free_blocks = self.allocated_gpu_blocks[cache_id]
        #print(f"FREE cache_id:{cache_id}, free_blocks:{free_blocks}, step:{self.step_index}")
       
        self.allocated_gpu_blocks[cache_id] = 0

        if immediate_free:
            # Only in swapping out case
            self.num_free_gpu_blocks += free_blocks
        else:
            self.to_free_gpu_caches[cache_id] = free_blocks
            self.cached_free_gpu_blocks += free_blocks

    """
    Free a sequence. We will append the seq to to_free_gpu_caches. 
    Initially, we did this inside the memory management library. Maybe we should do it here as well. 
    """
    def free(self, seq: Sequence) -> None:
        self._free_cache(cache_id=seq.cache_id, immediate_free=False)

    def reset(self) -> None:
        # Free decoder block tables
        self.allocated_gpu_blocks.clear()
        self.num_free_gpu_blocks = self.num_total_gpu_blocks
        self.num_free_cpu_blocks = self.num_total_cpu_blocks
        
        self.to_free_gpu_caches = {}
        self.to_allocate_blocks = {}

    # A dummy function that will be never invoked
    def get_block_table(self, seq: Sequence) -> List[int]:
        # logger.warning("block table is not used in BlockSpaceManagerDAttn now.")
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
        # logger.warning("Access all blocks in seq is not supported in BlockSpaceManagerDAttn now.")
        pass

    def get_common_computed_block_ids(self,
                                      seq_group: SequenceGroup) -> List[int]:
        # logger.warning("Common computed block ids is not supported in BlockSpaceManagerDAttn now.")
        return None  # type: ignore

    def mark_blocks_as_computed(self, seq_group: SequenceGroup, token_chunk_size: int) -> None:
        # logger.warning("Mark blocks as computed is not supported in BlockSpaceManagerDAttn now.")
        pass

    # In the end of each step's scheduling, this function is invoked to 
    # collect the information of allocation and deallocation  
    def step(self) -> Tuple[Dict[int, int], List[int], bool]:
        to_allocate_blocks = {}
        to_free_gpu_caches = []

        immediate_allocate = self.immediate_allocate
        self.immediate_allocate = False

        #print(f"in the end step-{self.step_index} now!") 
        # We will perform virtual memory management once for every self.vmm_frequency 
        if (self.step_index & self.vmm_frequency_mask) != 0 and immediate_allocate != True:
            # No need to invoke virtual memory management
            #print(f"step-{self.step_index} no need to do memory management") 
            self.step_index += 1
            return to_allocate_blocks, to_free_gpu_caches, immediate_allocate

        to_allocate_blocks = self.to_allocate_blocks.copy()

        # Update the immediate_allocate so that we could perform some
        # synchronized allocation in admitting new requests or swapping in requests
        #print(f"self.to_free_gpu_caches has length:{len(self.to_free_gpu_caches)}, cached_free_gpu_blocks:{self.cached_free_gpu_blocks}, total_available:{self.num_free_gpu_blocks}", file=sys.stderr)

        if len(to_free_gpu_caches) > 0:
            print(f"self.num_free_gpu_blocks: {self.num_free_gpu_blocks}") 
        
        to_free_blocks = 0
        # Check whether there is a need to free kv caches
        for cache_id, num_blocks in self.to_free_gpu_caches.items():
            to_free_gpu_caches.append(cache_id)
            self.gpu_allocator.free(cache_id)
            self.num_free_gpu_blocks += num_blocks
            to_free_blocks += num_blocks 

        #if len(to_allocate_blocks) > 0 or len(to_free_gpu_caches) > 0:         
        #print(f"step-{self.step_index} of updating, to_allocate_blocks:{len(to_allocate_blocks)}, to_free_gpu_caches:{len(to_free_gpu_caches)}({to_free_blocks} blocks), self.num_free_gpu_blocks:{self.num_free_gpu_blocks}")
        
        # step() is invoked once after _schedule() inside Scheduler::schedule(). It is invoked once for every decode or prefill
        self.to_free_gpu_caches.clear()
        self.to_allocate_blocks.clear()
        self.cached_free_gpu_blocks = 0

        self.step_index += 1    
        return to_allocate_blocks, to_free_gpu_caches, immediate_allocate

    def get_prefix_cache_hit_rate(self, device: Device) -> float:
        return 0