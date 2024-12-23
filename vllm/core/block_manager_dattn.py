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
        assert len(self.kv_caches) > 0, f"Please set self.num_caches to a bigger value"
        cache_id = self.kv_caches.popleft() 
        #    print(f"ERROR: self.kv_caches is 000000000 NOW", file=sys.stderr)
        #elif self.type == "cpu":
        #    print(f"ERROR checking: allocated a cpu cache:{cache_id}, remaining cache:{len(self.kv_caches)}", file=sys.stderr) 
        return cache_id

    def free(self, cache_id: int):
        self.kv_caches.appendleft(cache_id)

        #assert cache_id == self.kv_caches[0]
        #print(f"after free-{cache_id} of {self.type}, the left item:{self.kv_caches[0]} ", file=sys.stderr)
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
        num_cpu_caches: int = 40, 
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks

        # For every 16 steps, we will perform vmm updates by invoking update_cache_blocks
        self.vmm_frequency = vmm_frequency
        self.vmm_frequency_mask = vmm_frequency - 1 
        
        # Tracking the number of gpu_blocks (including self.cached_free_gpu_blocks) 
        self.num_free_gpu_blocks = num_gpu_blocks
        self.num_free_cpu_blocks = num_cpu_blocks
        
        
        num_gpu_caches = num_caches
        num_cpu_caches = num_cpu_caches 

        print(f"self.num_free_cpu_blocks-{self.num_free_cpu_blocks}, num_cpu_caches:{num_cpu_caches}", file=sys.stderr)
        # use to alloc cache buffer id for seq
        self.gpu_allocator = CacheAllocator("cuda", num_gpu_caches)
        self.cpu_allocator = CacheAllocator("cpu", num_cpu_caches)

        # Watermark indicates that the least amount of blocks should be free. 
        assert watermark >= 0.0
        self.watermark_blocks = 2
        #int(watermark * num_gpu_blocks)
        
        # Mapping from cache_id to the number of allocated blocks.
        # The information is more persitent across different steps
        self.allocated_gpu_blocks: Dict[int, int] = {} 
        # Pre-allocate one block for the first cache,  to support graph capture
        self.allocated_gpu_blocks[0] = 1 

        # Temporary buffer for each step. self.step() will collect these information and freed all 
        # caches of to_free_gpu_caches
        self.to_allocate_blocks: Dict[int, int] = {} 

        # Useful when admitting new requests or swapping in some requests. 
        # Then we prefer those requests that just exit.
        # Making cached_free_gpu_blocks a part of num_free_gpu_blocks
        self.cached_free_gpu_blocks: int = 0

        # to_free_gpu_caches keeps the requests that are freed in the current step
        self.to_free_gpu_caches: Dict[int, int] = {}
        self.immediate_allocate = False

        # Maintain the mapping between seq.req_id and SwappedCPUCache (cache_id, blocks)
        self.swapped_out_caches: Dict[int, SwappedCPUCache] = {}
        self.swapping_reqs: List[int] = []

        # number of active requests (which will be used to improve the scheduler)
        self.total_active_reqs = 0
        self.allocated_active_reqs = 0

        # Track the step information, used for periodical memory management
        self.step_index = 0
    
    def _predict_n_blocks(self, tokens: int, is_prefill: bool = False) -> int:
        if tokens == 0:
            return 0
        
        if is_prefill:
            return (tokens + self.vmm_frequency - (self.step_index & self.vmm_frequency_mask) + self.block_size - 1) // self.block_size 
        else:
            return (tokens + self.vmm_frequency + self.block_size - 1) // self.block_size

    def _get_n_blocks(self, tokens: int) -> int:
        return (tokens + self.block_size - 1) // self.block_size 

    def _check_availability(self, need_blocks) -> AllocStatus:
        # Ensure that one request should not use more than 90% or 99% of memory
        # This can avoid frequent cache eviction 
        if (self.num_total_gpu_blocks - need_blocks < self.watermark_blocks):
            return AllocStatus.NEVER
        
        if self.num_free_gpu_blocks - need_blocks >= self.watermark_blocks:
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


    # This function is only invoked by _allocate_and_set_running (invoked by _schedule_prefills)
    # Allocate a GPU cache when admitting a new request in prefill phase.
    def allocate(self, seq_group: SequenceGroup) -> None:
        # Allocate decoder sequences
        #
        # NOTE: Here we assume that all sequences in the group have the same
        # decoder prompt.
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        
        need_blocks = self._predict_n_blocks(tokens=seq.get_len(), is_prefill=True)

        self.immediate_allocate = True 
        cache_id = self._allocate_gpu_cache(need_blocks)
        
        print(f"NNOOOOOOWWW step_index-{self.step_index}, allocate cache_id: {cache_id}, need_blocks:{need_blocks}, tokens:{seq.get_len()}", file=sys.stderr) 
        seq.cache_id = cache_id
        seq.data.cache_id = cache_id

    #  Allocate a new GPU cache, when the available GPU blocks are sufficient
    def _allocate_gpu_cache(self, need_blocks: int) -> Tuple[int, int]:
        cache_id = -1
        to_allocate = True
        
        # update total_active_reqs and num_free_gpu_blocks
        self.total_active_reqs +=1
        self.num_free_gpu_blocks -= need_blocks

        # Prefer to reuse the to_free_gpu_caches at first, as some pages have been allocated already. 
        if self.cached_free_gpu_blocks > 0:
            # Make it block_diff a big number for the better comparison
            block_diff = need_blocks*100
            allocated_block_num = None

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
            self.cached_free_gpu_blocks -= allocated_block_num

            # If the cache has too many blocks, then we will release some blocks to other requests
            if allocated_block_num > need_blocks + self.watermark_blocks:
                #free_blocks_per_req = (int)((allocated_block_num-need_blocks)/self.total_active_reqs)

                # Compute the number of free blocks for each requst
                #free_blocks_per_req = self._compute_free_blocks(allocated_block_num, need_blocks) 
                #print(f"Reuse cache-{cache_id}: allocated_blocks-{allocated_block_num}, self.num_free_gpu_blocks:{self.num_free_gpu_blocks}, need_blocks:{need_blocks}", file=sys.stderr)

                # Now the current request will keep free_blocks_per_req blocks, while others 
                # are still need to be freed 
                need_blocks += self.watermark_blocks
                self.num_free_gpu_blocks -= self.watermark_blocks
            elif allocated_block_num > need_blocks:
                # when allocated_block_num <= need_blocks + self.watermark_blocks, no need to allocate
                to_allocate = False 
        else:
            # Check whether the can_allocate or can_swap_in has a bug
            if self.num_free_gpu_blocks < 0: 
                print(f"Error: self.num_free_gpu_blocks:{self.num_free_gpu_blocks}, need_blocks:{need_blocks}", file=sys.stderr)
            assert self.num_free_gpu_blocks >= 0

            cache_id = self.gpu_allocator.allocate()

        self.allocated_gpu_blocks[cache_id] = need_blocks 

        if to_allocate == True:
            self.to_allocate_blocks[cache_id] = need_blocks

        return cache_id

    # Invoked by _schedule_running in running phase.  
    def can_append_slots(self,
                         seq_group: SequenceGroup,
                         num_lookahead_slots: int = 0) -> bool:
        
        # Only check periodically in asynchronous memory management, not each step
        if self.step_index & self.vmm_frequency_mask:
            return True

        # Do not evict a request that have at least 16 slots to extend (at least we could do it next step)
        cache_blocks, real_blocks = self._get_physical_blocks(seq_group)
        if real_blocks < cache_blocks:
            return True

        to_allocate_reqs = self.total_active_reqs - self.allocated_active_reqs 
        # Simple heuristic: at least one free block for each request.
        # Since we will perform the actual allocation in the next epoch (16 steps), where 
        # each request can allocate one block successfully, then there
        # is no need to preempt. Note that self.cache_free_gpu_blocks 
        # should be included as they will be allocated first in the next epoch 
        #if free_blocks < to_allocate_reqs:
        #    print(f"STOP now!!!!!!Cannot append slots for seq-{seq_group.request_id}, self.total_active_reqs:{self.total_active_reqs}, self.num_free_gpu_blocks:{self.num_free_gpu_blocks}, self.cached_free_gpu_blocks:{self.cached_free_gpu_blocks} at step-{self.step_index}", file=sys.stderr) 
        
        return self.num_free_gpu_blocks >= to_allocate_reqs
        
    # FIXME: there is no handling on num_lookahead_slots, which should be handled.  
    def append_slots(
        self,
        seq: Sequence,
        num_lookahead_slots: int = 0,
    ) -> List[Tuple[int, int]]:

        # We only need to check periodically, not each step
        if self.step_index & self.vmm_frequency_mask:
            return []

        """Allocate a physical token/slot for a new token."""
        cache_id = seq.cache_id

        # If the sequence is allocated, its cache_id must >= 0.
        assert cache_id >= 0
        
        logical_blocks_num = self._predict_n_blocks(seq.get_len())
        allocated_block_num = self.allocated_gpu_blocks[cache_id]

        
        #print(f"seq_id:{seq.seq_id}, gpu_cache_id:{cache_id}, tokens:{seq.get_len()}, logical_blocks_num:{logical_blocks_num}, allocated_block_num:{allocated_block_num}, free_blocks:{self.num_free_gpu_blocks}", file=sys.stderr)

        # If we need to allocate a new physical block
        if allocated_block_num < logical_blocks_num:
            if allocated_block_num != logical_blocks_num - 1: 
                print(f"append_slots cache_id:{cache_id}, logical_blocks_num:{logical_blocks_num} - {allocated_block_num}, real tokens:{seq.get_len()}", file=sys.stderr) 
            
            # Currently this code only supports adding one physical block in the decoding phase
            assert allocated_block_num == logical_blocks_num - 1

            self.num_free_gpu_blocks -= 1
          
            if self.num_free_gpu_blocks < 0:
                print(f"ERROR: append_slots for cache_id:{cache_id}, self.num_free_gpu_blocks:{self.num_free_gpu_blocks}, cache_id:{cache_id}, logical_blocks_num:{logical_blocks_num}, allocated_block:{allocated_block_num}, self.cached_free_gpu_blocks:{self.cached_free_gpu_blocks}", file=sys.stderr)

            self.allocated_gpu_blocks[cache_id] = logical_blocks_num

            # Note that to_allocate_blocks actually hold the logic blocks number, not a bug. 
            self.to_allocate_blocks[cache_id] = logical_blocks_num 

        # Update that one active request have already been allocated
        self.allocated_active_reqs += 1 

        # No need to return anything here, since step() will collect all information
        # related to the current scheduling phase.            
        return []

    # Collect the number of physical blocks used by this sequence group 
    def _get_physical_blocks(
            self, seq_group: SequenceGroup):
        
        cache_blocks = 0
        real_blocks = 0
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                continue

            cache_id = seq.cache_id
            cache_blocks += self.allocated_gpu_blocks[cache_id]
            real_blocks += self._get_n_blocks(seq.get_len())
        
        return cache_blocks, real_blocks

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        raise NotImplementedError("Forking is not supported in BlockSpaceManagerDAttn now.")

    # This is to swap_in an pre-existing block, which is slightly different 
    # from can_allocate(). 
    def can_swap_in(self, seq_group: SequenceGroup,
                    num_lookahead_slots: int) -> AllocStatus:
        
        if self.step_index & self.vmm_frequency_mask:
            return AllocStatus.LATER

        # For those swapping requests, now it will return  
        req_id = seq_group.request_id
        if req_id in self.swapping_reqs:
            return AllocStatus.OK

        need_blocks = num_lookahead_slots
        req_id = None
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            if seq.is_finished():
                continue
            
            req_id = seq.seq_id
            need_blocks += self.swapped_out_caches[req_id].blocks

        to_allocate_reqs = self.total_active_reqs - self.allocated_active_reqs + 1
        
        # Make sure that the number of free blocks is sufficient for at least 
        # one block for each request
        need_blocks += to_allocate_reqs
        
        result = self._check_availability(need_blocks) 

        return result

    # A fucntion is invoked to figure out the blocks that need to be allocated. 
    def swap_in(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        to_swap_in_caches = []

        #print(f"SWAP IN NOW with sequence-{seq_group.request_id}, number-{seq_group.num_seqs(status=SequenceStatus.SWAPPED)} at step-{self.step_index}", file=sys.stderr)
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            cpu_cache = self.swapped_out_caches[seq.seq_id] 

            need_blocks = cpu_cache.blocks
            cpu_cache_id = cpu_cache.cache_id

            # Free cpu cache id and update the counter
            self.cpu_allocator.free(cpu_cache_id)
            self.num_free_cpu_blocks += need_blocks  
            
            # Allocate a gpu cache id, based on the need_blocks. 
            # Note that we specifically request one more block in order to accomodate vmm_frequency's memory management
            gpu_cache_id = self._allocate_gpu_cache(need_blocks + 1)

            seq.cache_id = gpu_cache_id
            seq.data.cache_id = gpu_cache_id

            # NOTE: we may not need the allocation, if gpu_cache_id 
            print(f"SWAPIN seq_id:{seq.seq_id} with tokens:{seq.get_len()}, cpu_cache_id:{cpu_cache_id}, gpu_cache_id:{gpu_cache_id}, need_blocks:{need_blocks}, allocated_blocks:{self.allocated_gpu_blocks[gpu_cache_id]}, free_gpu_blocks:{self.num_free_gpu_blocks} freeCPUBlocks:{self.num_free_cpu_blocks}, at step:{self.step_index}, active requests:{self.total_active_reqs}", file=sys.stderr)
            to_swap_in_caches.append([cpu_cache_id, gpu_cache_id, need_blocks])
            
            # Delete this entry
            del self.swapped_out_caches[seq.seq_id]

        #print(f"in the end of swapin checking", file=sys.stderr)
        return to_swap_in_caches

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        cache_blocks, real_blocks = self._get_physical_blocks(seq_group)
        if real_blocks > self.num_free_cpu_blocks:
            print(f"req:{seq_group.request_id}, real_blocks:{real_blocks}, self.num_free_cpu_blocks:{self.num_free_cpu_blocks}", file=sys.stderr)
        return real_blocks <= self.num_free_cpu_blocks

    def swap_out(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
    
        to_swap_out_caches = []

        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            # Find the cache id and gpu_blocks        
            gpu_cache_id = seq.cache_id

            # Since this cache may have more blocks than its necessity, we only record the 
            # real_gpu_blocks here in order to reduce the overhead involved in copy in swapping
            real_gpu_blocks = self._get_n_blocks(seq.get_len())

            # Free the cache related to gpu_cache_id
            self._free_cache(cache_id=gpu_cache_id)

            # Allocate the cpu cache id
            cpu_cache_id = self.cpu_allocator.allocate()
            cpu_cache = SwappedCPUCache(cpu_cache_id, real_gpu_blocks) 
            self.swapped_out_caches[seq.seq_id] = cpu_cache

            # After the swapped out, num_free_cpu_blocks should be decremented 
            self.num_free_cpu_blocks -= real_gpu_blocks
            
            print(f"SWAPOUT request-{seq.seq_id} with tokens-{seq.get_len()}, cpu_cache_id:{cpu_cache_id}, freeCPUBlocks:{self.num_free_cpu_blocks}, freeGPUBlocks:{self.num_free_gpu_blocks},  gpu_cache_id:{gpu_cache_id}, blocks:{real_gpu_blocks} at step-{self.step_index}, requests:{self.total_active_reqs}", file=sys.stderr)
            
            to_swap_out_caches.append([gpu_cache_id, cpu_cache_id, real_gpu_blocks]) 

        #print(f"to_swap_out_caches:{to_swap_out_caches}", file=sys.stderr)
        return to_swap_out_caches

    def _free_cache(self, cache_id: int) -> None:
        # Check whether cache_id is in the list
        if cache_id in self.to_free_gpu_caches:
            # Already freed yet, no need to do anything.
            return

        # Get blocks of this cache
        free_blocks = self.allocated_gpu_blocks[cache_id]
        #print(f"FREE gpu cache_id:{cache_id}, free_blocks:{free_blocks}, step:{self.step_index}", file=sys.stderr)
       
        # Note that we update self.total_active_reqs here, as free_cache() is invoked twice for every request
        self.total_active_reqs -=1
        self.allocated_gpu_blocks[cache_id] = 0

        self.to_free_gpu_caches[cache_id] = free_blocks
        self.cached_free_gpu_blocks += free_blocks
        self.num_free_gpu_blocks += free_blocks

    """
    Free a sequence. We will append the seq to to_free_gpu_caches. 
    Initially, we did this inside the memory management library. Maybe we should do it here as well. 
    """
    def free(self, seq: Sequence) -> None:
        #print(f"free sequence:{seq.seq_id}, cache_id:{seq.cache_id}, data_cache_id:{seq.data.cache_id}", file=sys.stderr)
        self._free_cache(cache_id=seq.cache_id)
        
        #print(f"After free, self.total_active_reqs: {self.total_active_reqs}", file=sys.stderr)

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

        #print(f"in the end step-{self.step_index} with requests:{self.total_active_reqs}, allocate_blocks:{len(self.to_allocate_blocks)} now!", file=sys.stderr) 
        # We will perform virtual memory management once for every self.vmm_frequency 
        if (self.step_index & self.vmm_frequency_mask) != 0 and immediate_allocate != True:
            # No need to invoke virtual memory management
            #print(f"step-{self.step_index} no need to do memory management", file=sys.stderr) 
            self.step_index += 1
            return to_allocate_blocks, to_free_gpu_caches, immediate_allocate

        for cache_id, num_blocks in self.to_allocate_blocks.items():
            #print(f"step-{self.step_index} toallocate cache_id:{cache_id}, num_blocks:{num_blocks}", file=sys.stderr)
            to_allocate_blocks[cache_id] = num_blocks

        to_free_blocks = 0
        # Check whether there is a need to free kv caches
        for cache_id, num_blocks in self.to_free_gpu_caches.items():
            #print(f"step-{self.step_index} free cache_id:{cache_id}, num_blocks:{num_blocks}", file=sys.stderr)
            to_free_gpu_caches.append(cache_id)
            self.gpu_allocator.free(cache_id)
            to_free_blocks += num_blocks 

        #if len(to_allocate_blocks) > 0 or len(to_free_gpu_caches) > 0:         
        #print(f"step-{self.step_index}, immediate_allocate:{immediate_allocate}, to_allocate_blocks:{len(to_allocate_blocks)}, to_free_gpu_caches:{len(to_free_gpu_caches)}({to_free_blocks} blocks), self.num_free_gpu_blocks:{self.num_free_gpu_blocks}, requests:{self.total_active_reqs}, swapped requests:{len(self.swapped_out_caches)}", file=sys.stderr)
        
        # step() is invoked once after _schedule() inside Scheduler::schedule(). It is invoked once for every decode or prefill
        self.to_free_gpu_caches.clear()
        self.to_allocate_blocks.clear()
        self.cached_free_gpu_blocks = 0
        self.allocated_active_reqs = 0

        self.step_index += 1    
        return to_allocate_blocks, to_free_gpu_caches, immediate_allocate

    def get_prefix_cache_hit_rate(self, device: Device) -> float:
        return 0