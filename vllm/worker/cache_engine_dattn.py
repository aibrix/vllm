'''
 Copyright (c) ByteDance Inc.
 Authors: 
  - Tongping Liu (tongping.liu@bytedance.com)
  - https://github.com/vllm-project/vllm/pull/6102/commits
'''
"""CacheEngine class for managing the KV cache."""
from typing import List, Dict, Tuple

import torch

from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig, DeviceConfig
from vllm.logger import init_logger
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, is_pin_memory_available, get_dtype_size

from vllm import _dattn_ops as dattn
import subprocess
from enum import Enum
import re
import mmap
import ctypes
import sys

logger = init_logger(__name__)

class CacheEngineDAttn:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
    ) -> None:
        if device_config.device_type != "cuda":
            raise RuntimeError("DATTN only support cuda device.")

        self.num_layers = model_config.get_num_layers(parallel_config)
        
        head_size = model_config.get_head_size()
        num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        self.block_size = cache_config.block_size

        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        dtype_size = get_dtype_size(dtype)
        self.block_bytes_size = (head_size * num_kv_heads * dtype_size * self.block_size) * self.num_layers  * 2 

        max_batch_size = scheduler_config.max_num_seqs
        max_seq_len = scheduler_config.max_model_len
        
        # If max_seq_len is not divisible by self.block_size,
        # round up to the nearest value that is.
        if max_seq_len % self.block_size != 0:
            logger.warning("Note: max_seq_len mod self.block_size != 0")
            exit(0)

        token_size = num_kv_heads * head_size
        sequence_buffer_size = max_seq_len * token_size
        sequence_buffer_bytes_size = sequence_buffer_size * dtype_size
        cache_space_size = max_batch_size * sequence_buffer_bytes_size
        cache_space_bytes_size = cache_space_size * 2

        cache_space_per_req = sequence_buffer_bytes_size * self.num_layers * 2
        assert (cache_space_bytes_size) % self.block_bytes_size == 0, "cache_space_bytes_size must be divisible by block_bytes_size"
        
        cache_space_page_num = cache_space_bytes_size // self.block_bytes_size

        logger.info("CacheEngineDAttn basic info: { block_size: %d, dtype_size: %d, head_size: %d, "
                    "num_kv_heads: %d, max_seq_len: %d, max_batch_size: %d, self.num_layers: %d,"
                    "token_size: %d, sequence_buffer_size: %d, cache_space_size: %d, "
                    "cache_space_bytes_size: %d, cache_space_page_num: %d, cache_space_per_req: %d, cache_block_size: %x}",
                    self.block_size, dtype_size, head_size,
                    num_kv_heads, max_seq_len, max_batch_size, self.num_layers, 
                    token_size, sequence_buffer_size, cache_space_size,
                    cache_space_bytes_size, cache_space_page_num, cache_space_per_req, self.block_bytes_size)

        self.device_cache_allocator = dattn.kvCacheAllocator(max_seq_len, self.num_layers, num_kv_heads,
                                                             head_size, self.block_size, dtype_size)

        # Let's use 1/20 of max_batch_size for cpu caches
        # NOTE: make sure that is same as BlockSpaceManagerDAttn::num_cpu_caches
        cpu_cache_num = int(max_batch_size/20) 

        # Get attention backend.
        self.attn_backend = get_attn_backend(
            model_config.get_num_attention_heads(parallel_config),
            head_size,
            num_kv_heads,
            model_config.get_sliding_window(),
            model_config.dtype,
            cache_config.cache_dtype,
            self.block_size,
        )

        # A dummy mmap to hold cpu cache's addresses
        self.mmap = []

        self.kv_cache_ptrs = self._reserve_gpu_kv_cache(max_batch_size)
        self.gpu_cache = self._create_fake_kv_cache(self.num_layers)
        self.MADV_COLD = self._find_macro_value("MADV_COLD", "/usr/include/asm-generic/mman-common.h") 

        self.cpu_cache = [None] * cpu_cache_num

        self._reserve_cpu_kv_cache(cpu_cache_num, cache_space_per_req) 

    def _find_macro_value(self, macro_name, header_file):
        try:
            # Run grep to find the macro definition in the specified header file
            result = subprocess.run(
                ['grep', f'#define {macro_name}', header_file],
                text=True,
                capture_output=True,
                check=True
            )
            # Extract the macro value using regex
            match = re.search(rf'#define {macro_name}\s+(\d+)', result.stdout)
            if match:
                return int(match.group(1))
            else:
                print(f"{macro_name} not found in {header_file}.", file=sys.stderr)
                return None
        except subprocess.CalledProcessError:
            print(f"Failed to find {macro_name} in {header_file}.", file=sys.stderr)
            return None

    def get_n_blocks(num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) % self.block_size
    
    """
    In dAttention's design, we are required to pass the layer index so
    that CUDA kernel could use it to get the kv_cache. For other mechanisms, like
    PagedAttention or vAttention, they are passing different kv_vache for different layers.
    """
    def _create_fake_kv_cache(self, num_layers: int) -> List[torch.Tensor]: 
        fake_kv_caches = []

        for i in range(num_layers):
            fake_kv_caches.append(torch.tensor(i))

        return fake_kv_caches
    
    def _reserve_gpu_kv_cache(self, max_batch_size:int) -> List[int]:
        kv_cache_ptrs = []

        for i in range(max_batch_size):
            # Invoke gpu region one by one, which returns the cuda address
            kv_cache_ptr = self.device_cache_allocator.reserve_cache_region(i)
            kv_cache_ptrs.append(kv_cache_ptr)

        return kv_cache_ptrs

    def _reserve_cpu_kv_cache(self, cache_num: int, cache_space_per_req: int) -> List[int]:

        for i in range(cache_num):
            # Using mmap to reserve space for cpu cache. Multiple*2 in order to hold K/V cache
            mm = mmap.mmap(-1, cache_space_per_req)
            self.mmap.append(mm)
            address = ctypes.addressof(ctypes.c_char.from_buffer(mm))
            
            # record the address for ith cache
            self.cpu_cache[i] = address
            #print(f"{i}-th address-{hex(address)}, cache_space_per_req:{hex(cache_space_per_req)}", file=sys.stderr)

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        to_swap_in_caches = []

        for pair in src_to_dst:
            item = pair.flatten()

            cpu_cache_id = item[0]
            gpu_cache_id = item[1]
            blocks = item[2]

            gpu_cache_address = self.kv_cache_ptrs[gpu_cache_id]
            cpu_cache_address = self.cpu_cache[cpu_cache_id]

            size = blocks * self.block_bytes_size
            print(f"swapin src:{cpu_cache_id} - address:{hex(cpu_cache_address)}, dest:{gpu_cache_id} - address:{hex(gpu_cache_address)}, blocks:{blocks}, size:{hex(size)}", file=sys.stderr)
            to_swap_in_caches.append([cpu_cache_address, gpu_cache_id, blocks])

        return to_swap_in_caches
        #src_to_dests = torch.tensor(to_swap_in_caches, dtype=torch.int64)
        #self.device_cache_allocator.swap_in_cache(to_swap_in_caches)

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        
        #print(f"CacheEngineDAttn swap_out with src_to_dst:{src_to_dst}", file=sys.stderr)
        to_swap_out_caches = []

        for pair in src_to_dst:
            item = pair.flatten()

            gpu_cache_id = item[0]
            cpu_cache_id = item[1]
            blocks = item[2]

            gpu_cache_address = self.kv_cache_ptrs[gpu_cache_id]
            cpu_cache_address = self.cpu_cache[cpu_cache_id]
            size = blocks * self.block_bytes_size 
            
            print(f"Engine swapout src:{gpu_cache_id} - address:{hex(gpu_cache_address)}, dest:{cpu_cache_id} - address:{hex(cpu_cache_address)}, blocks:{blocks}, size:{hex(size)}", file=sys.stderr)
            to_swap_out_caches.append([gpu_cache_id, cpu_cache_address, size])

        return to_swap_out_caches
        #src_to_dests = torch.tensor(to_swap_out_caches, dtype=torch.int64)
        #self.device_cache_allocator.swap_out_cache(to_swap_out_caches)

    # TODO: we need to implement the copy_blocks 
    def copy(self, src_to_dsts: torch.Tensor) -> None:
        self.device_cache_allocator.copy_blocks(self.gpu_cache, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_attention_layers = model_config.get_num_attention_layers(
            parallel_config)

        key_cache_block = cache_config.block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_attention_layers * (key_cache_block + value_cache_block)
        #print(f"CacheEngineDAttn:head_size:{head_size}, num_heads:{num_heads}, num_attention_layers:{num_attention_layers}, self.block_size: {cache_config.block_size}, key_cache_block:{key_cache_block},total:{total/1024}KB", file=sys.stderr)
        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        dtype_size = get_dtype_size(dtype)
        #print(f"CacheEngineDAttn:cache_config.block_bytes_size:{dtype_size * total}", file=sys.stderr)
        return dtype_size * total

    def update_cache_blocks(self, immediate_allocate: bool, free_kv_caches: List[int], to_allocate_blocks: Dict[int, int], 
                            to_swap_out: List[List[int]], to_swap_in: List[List[int]]):
        to_alloc_list = []
        for cache_id, blocks in to_allocate_blocks.items():
            to_alloc_list.append([cache_id, blocks])

        self.device_cache_allocator.update_cache_blocks(immediate_allocate, free_kv_caches, to_alloc_list, to_swap_out, to_swap_in)
        

def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()
