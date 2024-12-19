import logging
import numpy as np
import time
import threading
from functools import partial
from queue import Queue, Full
from collections import deque
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

import torch
import torch.distributed

import vllm.envs as envs
from vllm.config import ModelConfig, ParallelConfig
from vllm.distributed.parallel_state import (get_tensor_model_parallel_rank,
                                             get_tensor_model_parallel_world_size)
from vllm.distributed.communication_op import (tensor_model_parallel_all_reduce,
                                               tensor_model_parallel_broadcast_object_list,
                                               tensor_model_parallel_broadcast,)
from vllm.sequence import (SequenceData, SequenceGroupMetadata)
from vllm.utils import init_logger, ObjectPool

try:
    from vineyard.llm import KVCache as VineyardKVCache
    from vineyard.llm import KVTensor as VineyardKVTensor
    from vineyard.llm import FileCacheConfig, VineyardCacheConfig
except ImportError:
    raise
    VineyardKVCache = None
    VineyardKVTensor = None

logger = init_logger(__name__)

class CacheServiceMetrics: 

    def __init__(self):
        # Instance variables
        self.hit_tokens: int = 0  # Total number of tokens hit.
        self.total_tokens: int = 0  # Total number of tokens requested.
        self.hit_blocks: int = 0  # Total number of blocks hit.
        self.total_blocks: int = 0  # Total number of blocks requested.

        self.time_query: List[float] = []  # Times used query cache from cache service.
        self.time_load: List[float] = []  # Times used load fetched cache to device memory.
        self.time_reshape: List[float] = []  # Times used reshaping tensors for flash attention KV format.
        self.time_unload: List[float] = []  # Times used move computed KV from device memory.
        self.time_update: List[float] = []  # Times used update computed KV to cache service.
        
        self.err_query: int = 0 # Number of query errors.
        self.err_async_update_task_queue_full: int = 0  # Number of exceptions for async update tasks.

        self.lock: threading.Lock = threading.Lock()
        # The following metrics need to be protected by `lock`
        self.time_async_update_queue: List[float] = []  # Queuing delays of async update tasks.
        self.time_async_update_exec: List[float] = []  # Execution times of async update tasks.
        self.counter_async_update_updated:  List = []  # Number of updated tokens.
        self.err_update: int = 0 # Number of update errors.
        
    def __getstate__(self):
        # Create a state dictionary excluding the lock
        state = self.__dict__.copy()
        del state['lock']
        return state

    def __setstate__(self, state):
        # Restore the instance attributes
        self.__dict__.update(state)
        # Reinitialize the lock
        self.lock = threading.Lock()

    def add_time_query(self, value):
        self.time_query.append(value)

    def add_time_load(self, value):
        self.time_load.append(value)

    def add_time_reshape(self, value):
        self.time_reshape.append(value)

    def add_time_unload(self, value):
        self.time_unload.append(value)

    def add_time_update(self, value):
        self.time_update.append(value)
    
    def get_tokens_hit_rate(self):
        return 0 if self.total_tokens == 0 else self.hit_tokens / float(self.total_tokens)

    def get_blocks_hit_rate(self):
        return 0 if self.total_blocks == 0 else self.hit_blocks / float(self.total_blocks)

    def update_async_metrics(self, queue_duration, exec_duration, updated):
        with self.lock:
            self.time_async_update_queue.append(queue_duration)
            self.time_async_update_exec.append(exec_duration)
            self.counter_async_update_updated.append(updated)

    def get_async_metrics(self):
        with self.lock:
            return self.time_async_update_queue, self.time_async_update_exec, self.counter_async_update_updated
    
    def reset_async_metrics(self):
        with self.lock:
            self.time_async_update_queue = []
            self.time_async_update_exec = [] 
            self.counter_async_update_updated = []

class VineyardLLMCache:
    def __init__(
        self,
        head_size: int,
        num_kv_heads: int,
        max_num_batched_tokens: int,
        cache_capacity: int = 1024,
        layer: int = 2,
        kv_cache_dtype: str = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        metrics: CacheServiceMetrics = None,
        metrics_enabled: bool = False,
        enable_async_update: bool = False,
        min_inflight_tasks: int = 1,
        max_inflight_tasks: int = 1,
    ):
        self._init_vineyard_logger()

        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.max_num_batched_tokens = max_num_batched_tokens
        self.cache_capacity = cache_capacity
        self.layer = layer
        self.kv_cache_dtype = kv_cache_dtype
        self.torch_dtype = torch_dtype
        self.tensor_nbytes = head_size * num_kv_heads * 2  # float16/bfloat16
        self.cache = VineyardKVCache(
            tensor_nbytes=self.tensor_nbytes,
            cache_capacity=self.cache_capacity,
            layer=self.layer,
            rank=get_tensor_model_parallel_rank(),
            world_size=get_tensor_model_parallel_world_size(),
        )
        self.chunk_size = self.cache.chunk_size

        if self.max_num_batched_tokens % self.chunk_size != 0:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must" \
                f"be a multiple of chunk_size ({self.chunk_size})"
            )
        self.fetch_buffer, self.fetch_tensors = self._pinned_tensor_creator()
        self.cuda_buffer = self.fetch_buffer.cuda()
        self.enable_async_update = enable_async_update

        if self.enable_async_update:
            # we use an object pool to reuse the pinned tensors and restrict the number of
            # inflight tasks. if an update operation cannot get a tensor from the pool,
            # meaning we already have max_inflight_tasks tasks issued, it then simply skips
            # the update. A completed task will return the used tensor back to the pool.
            self.tensor_pool = ObjectPool(
                min_inflight_tasks,
                max_inflight_tasks,
                self._pinned_tensor_creator,
            )

            # `_update_tasks` is a task queue being accessed by both the main thread
            # and the background thread.
            self._update_tasks = Queue(maxsize=max_inflight_tasks)
            # The cache backend is designed to drop updates whose prefix chunks are
            # not already present in the cache, which imposes an ordering requirement
            # on updates: we must perform updates in the issued order. For simplicity,
            # we use a single thread to process all updates sequentially.
            self._background_loop = threading.Thread(
                target=self._run_background_loop, daemon=True
            )
            self._background_loop.start()

        self.metrics = metrics
        self.metrics_enabled = metrics_enabled
        logger.info(f"VineyardLLMCache init {metrics} metrics_enabled {metrics_enabled}")
        logger.info(self)

    def _pinned_tensor_creator(
        self,
    ) -> Tuple[torch.Tensor, List[List[Tuple[VineyardKVTensor, VineyardKVTensor]]]]:
        '''Create a pinned tensor and a list of tensors to hold the KV tensors.
        '''
        buffer = torch.empty(
            (2, self.layer, self.max_num_batched_tokens, self.num_kv_heads, self.head_size),
            dtype=self.torch_dtype, device='cpu',
        ).pin_memory()
        tensors = []
        for i in range(self.max_num_batched_tokens):
            tensors.append([])
            for j in range(self.layer):
                k_tensor = buffer[0, j, i]
                v_tensor = buffer[1, j, i]
                tensors[-1].append((
                    VineyardKVTensor(k_tensor.data_ptr(), k_tensor.numel() * k_tensor.element_size()),
                    VineyardKVTensor(v_tensor.data_ptr(), v_tensor.numel() * v_tensor.element_size()),
                ))
        return buffer, tensors

    def _init_vineyard_logger(self):
        import vineyard
        logging.basicConfig()

        vineyard.logger.setLevel(logger.getEffectiveLevel())
        vineyard.logger.handlers.clear()
        for handler in logger.handlers:
            vineyard.logger.addHandler(handler)

    def _run_background_loop(self):
        '''Start a background loop to process the update tasks.
        '''
        logger.info("VineyardKVCache background loop is running")
        while True:
            # Wait until there is a task in the queue
            update_fn = self._update_tasks.get()
            # Run the task
            try:
                update_fn()
                logger.debug(
                    f"Completed an update op, current task queue size={self._update_tasks.qsize()}"
                )
            except Exception:
                pass

    @staticmethod
    def from_envs(
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        kv_cache_dtype: str,
        max_num_batched_tokens: int,
        torch_dtype: torch.dtype = torch.bfloat16,
        metrics: CacheServiceMetrics = None,
    ) -> Optional["VineyardLLMCache"]:
        if VineyardKVCache is None:
            logger.warn("VineyardKVCache module is not available")
            return None

        if not envs.VLLM_USE_FLASH_ATTN_DECODING:
            logger.warn("VineyardLLMCache requires flash attention decoding")
            return None

        cpu_mem_limit = int(envs.VINEYARD_CACHE_CPU_MEM_LIMIT_GB * 1024**3)
        head_size = model_config.get_head_size()
        num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)
        token_nbytes = num_layers * num_kv_heads * head_size * 4  # sizeof(float16) * 2 (kv)

        # we will use one temp buffer to hold the kv tensors fetched from v6d cache
        # , i.e., `fetch_buffer`
        num_temp_cpu_buffer = 1
        kwargs = {}

        # if async update is enabled, we will use a portion of cpu memory as temp
        # buffers to hold the kv tensors being updated into the v6d cache
        if envs.VINEYARD_CACHE_ENABLE_ASYNC_UPDATE:
            # get the mem limit
            async_update_cpu_mem_util = envs.VINEYARD_CACHE_ASYNC_UPDATE_CPU_MEM_UTIL
            async_update_cpu_mem_limit = async_update_cpu_mem_util * cpu_mem_limit
            max_inflight_tasks = int(
                async_update_cpu_mem_limit // (max_num_batched_tokens * token_nbytes)
            )
            max_inflight_tasks = min(max_inflight_tasks, envs.VINEYARD_CACHE_MAX_INFLIGHT_TASKS)
            num_temp_cpu_buffer += max_inflight_tasks
            kwargs["enable_async_update"] = True
            kwargs["min_inflight_tasks"] = min(envs.VINEYARD_CACHE_MIN_INFLIGHT_TASKS, max_inflight_tasks)
            kwargs["max_inflight_tasks"] = max_inflight_tasks
            logger.info(f"VineyardLLMCache async update: {kwargs}")
        
        metrics_enabled = False
        if envs.VINEYARD_CACHE_METRICS_ENABLED:
            metrics_enabled = True

        # convert cache capacity to number of tokens
        cache_capacity = (
            cpu_mem_limit
            - num_temp_cpu_buffer * max_num_batched_tokens * token_nbytes
        ) // token_nbytes
        logger.info(f"VineyardLLMCache from_envs {metrics}")
        return VineyardLLMCache(
            head_size=head_size,
            num_kv_heads=num_kv_heads,
            max_num_batched_tokens=max_num_batched_tokens,
            cache_capacity=cache_capacity,
            layer=num_layers,
            kv_cache_dtype=kv_cache_dtype,
            torch_dtype=torch_dtype,
            metrics = metrics,
            metrics_enabled = metrics_enabled,
            **kwargs,
        )

    def _update_seq_group_metadata(
        self, seq_group_metadata: SequenceGroupMetadata, value: int
    ) -> None:
        '''Update sequence group's metadata
        '''
        if seq_group_metadata is not None:
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]
            seq_data = seq_group_metadata.seq_data[seq_id]
            seq_data.update_num_computed_tokens(value)
            seq_group_metadata.token_chunk_size -= value

    def prefetch_seq_kv_caches(
        self,
        seq_group_metadata: SequenceGroupMetadata,
        kv_caches: List[torch.Tensor],
        block_size: int,
    ) -> Tuple[str, int]:
        from vllm._custom_ops import reshape_and_cache_flash
        if get_tensor_model_parallel_rank() == 0:
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]
            seq_data = seq_group_metadata.seq_data[seq_id]

            context_len = seq_data.get_num_computed_tokens()
            token_chunk_size = seq_group_metadata.token_chunk_size
            tokens = seq_data.get_prompt_token_ids()

            # Previously, at least one token is left unmatched to always trigger sampling.
            # However, when there is full KV cache hit, there is no need to sample
            # unless it is explicitly required.
            # # leave at least one token unmatched
            # token_chunk_size -= 1

            # alignment `context_len` to `self.chunk_size`
            query_context_len = context_len - context_len % self.chunk_size
            query_token_size = context_len + token_chunk_size - query_context_len
            query_prefix = tokens[:query_context_len]
            query_tokens = tokens[query_context_len:query_context_len + query_token_size]
            query_args = [
                seq_id,
                context_len,
                token_chunk_size,
                query_context_len,
                query_token_size,
                query_prefix,
                query_tokens,
            ]
            tensor_model_parallel_broadcast_object_list(query_args, src=0)
        else:
            query_args = [None, None, None, None, None, None, None]
            tensor_model_parallel_broadcast_object_list(query_args, src=0)
            (seq_id,
             context_len,
             token_chunk_size,
             query_context_len,
             query_token_size,
             query_prefix,
             query_tokens
            ) = query_args

        if query_token_size <= 0:
            return seq_id, 0
        
        self.metrics.total_tokens += query_token_size
        self.metrics.total_blocks += ((-query_token_size) // (-block_size))
        if self.metrics_enabled:
            start_time = time.perf_counter()
        matched = 0
        try:
            matched = self.cache.query(
                prefix=query_prefix,
                tokens=query_tokens,
                kv_cache_list=self.fetch_tensors[:query_token_size],
            )
        except Exception:
            if self.metrics_enabled:
                self.metrics.err_query += 1
        if self.metrics_enabled:
            duration = time.perf_counter() - start_time
            self.metrics.add_time_query(duration)
        # If sampling is required, we need to leave one token unmatched
        # to trigger the following sampling step in engine worker's workflow.
        if seq_group_metadata is not None and seq_group_metadata.is_sampling_enabled:
            matched = min(matched, token_chunk_size - 1)
        # synchronized across tensor parallel ranks
        matched_tensor = torch.tensor([matched], dtype=torch.long, device='cuda')
        tensor_model_parallel_all_reduce(input_=matched_tensor, op=torch.distributed.ReduceOp.MIN)
        matched = matched_tensor[0].item()

        # shift
        offset = context_len % self.chunk_size
        matched -= offset
        if matched <= 0:
            return seq_id, 0
        if get_tensor_model_parallel_rank() == 0:
            block_table = seq_group_metadata.block_tables[seq_id]
            slot_mapping = []
            for i in range(context_len, context_len + matched):
                block_number = block_table[i // block_size]
                block_offset = i % block_size
                slot = block_number * block_size + block_offset
                slot_mapping.append(slot)
            slot_mapping = torch.tensor(slot_mapping, dtype=torch.long, device='cuda')
            tensor_model_parallel_broadcast(slot_mapping, src=0)
        else:
            slot_mapping = torch.zeros((matched,), dtype=torch.long, device='cuda')
            tensor_model_parallel_broadcast(slot_mapping, src=0)
        self.metrics.hit_tokens += matched
        self.metrics.hit_blocks += (matched // block_size)
        if self.metrics_enabled:
            # save to GPU kv cache
            torch.cuda.synchronize()
            copy_start = torch.cuda.Event(enable_timing=True)
            copy_end = torch.cuda.Event(enable_timing=True)
            copy_start.record()
        # Copying the entire buffer to the GPU in a single operation and then
        # slicing it into smaller, non-contiguou chunks on the GPU is more
        # efficient than performing multiple smaller copy operations. This
        # approach reduces the number of transfers between CPU and GPU,
        # leading to faster overall performance.
        buffer = self.cuda_buffer.copy_(self.fetch_buffer)[:, :, :matched]
        if self.metrics_enabled:
            copy_end.record()
            copy_end.synchronize()
            duration = copy_start.elapsed_time(copy_end) / 1000.0
            self.metrics.add_time_unload(duration)

            torch.cuda.synchronize()
            reshape_start = torch.cuda.Event(enable_timing=True)
            reshape_end = torch.cuda.Event(enable_timing=True)
            reshape_start.record()
        for j in range(self.layer):
            # use `reshape_and_cache_flash` rather than `copy_` as
            # the target kv cache slots is not contingous.
            reshape_and_cache_flash(
                buffer[0][j],
                buffer[1][j],
                kv_caches[j][0],
                kv_caches[j][1],
                slot_mapping,
                self.kv_cache_dtype,
                1.0,
                1.0
            )
        if self.metrics_enabled:
            reshape_end.record()
            reshape_end.synchronize()
            duration = reshape_start.elapsed_time(reshape_end) / 1000.0
            self.metrics.add_time_reshape(duration)

        # update the seq_group_metadata's and seq's metadata
        self._update_seq_group_metadata(seq_group_metadata, matched)

        return seq_id, matched

    def prefetch_kv_caches(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        kv_caches: List[torch.Tensor],
        block_size: int,
    ) -> Dict[str, int]:
        ''' Returns a dict to indicate the matched kv cache lengths
            for each sequence group metadata.
        '''
        if block_size is None or kv_caches[0] is None:  # profile run
            return {}
        if get_tensor_model_parallel_rank() == 0:
            prefill_requests = []
            if seq_group_metadata_list is not None:
                for seq_group_meta in seq_group_metadata_list:
                    if seq_group_meta.is_prompt:
                        prefill_requests.append(seq_group_meta)
            num_prefill_requests = [len(prefill_requests)]
            tensor_model_parallel_broadcast_object_list(num_prefill_requests, src=0)
        else:
            num_prefill_requests = [None]
            tensor_model_parallel_broadcast_object_list(num_prefill_requests, src=0)
            prefill_requests = [None] *  num_prefill_requests[0]
        num_prefill_requests = num_prefill_requests[0]
        matched = {}
        for seq_group_meta in prefill_requests:
            seq_id, seq_matched = self.prefetch_seq_kv_caches(
                seq_group_meta, kv_caches, block_size,
            )
            matched[seq_id] = seq_matched
        if matched:
            logger.debug(f"prefetch_kv_caches: matched=%r", matched)
        return matched

    def _update_kv_cache(
        self,
        prefix: List[int],
        tokens: List[int],
        buffer_tensors_tuple: Tuple[
            torch.Tensor, List[List[Tuple[VineyardKVTensor, VineyardKVTensor]]]
        ],
        scheduled_time: float,
    ) -> None:
        '''Update the KV cache.

        Args:
            prefix: Prefix tokens.
            tokens: Tokens to be cached.
            buffer_tensors_tuple: Within the tuple, the first element is a continugous,
                                  pinned buffer, and the second element is a logical view
                                  of the buffer that is used to let v6d to know about the
                                  actual layout of the KV tensors.
                                  If async update is enabled, the `buffer_tensors_tuple`
                                  is allocated from the object pool, and thus we need to
                                  return it back to the pool after completing the update
                                  operation.
            scheduled_time: The timestamp that the task is scheduled.
        '''
        try:
            if self.metrics_enabled:
                start_time = time.perf_counter()
                queue_duration = start_time - scheduled_time
            update_token_size = len(tokens)
            kv_cache_list = buffer_tensors_tuple[1][:update_token_size]
            updated = self.cache.update(prefix, tokens, kv_cache_list)
            if self.metrics_enabled:
                exec_duration = time.perf_counter() - start_time
            if self.enable_async_update:
                if self.metrics_enabled:
                    logger.debug(
                        f"update kv cache: #prefix={len(prefix)}, #tokens={len(tokens)}, updated={updated}, "
                        f"queue_duration={queue_duration:.4f}, exec_duration={exec_duration:.4f}"
                    )
                    self.metrics.update_async_metrics(queue_duration, exec_duration, updated)
                else:
                    logger.debug(
                        f"update kv cache: #prefix={len(prefix)}, #tokens={len(tokens)}, updated={updated}")
            else:
                logger.debug(
                    f"update kv cache: #prefix={len(prefix)}, #tokens={len(tokens)}, updated={updated}"
                )
        except Exception:
            if self.metrics_enabled:
                with self.metrics.lock:
                    self.metrics.err_update += 1
        finally:
            if self.enable_async_update:
                self.tensor_pool.put(buffer_tensors_tuple)

    def update_seq_kv_caches(
        self,
        matched: Dict[str, int],
        seq_group_metadata: SequenceGroupMetadata,
        kv_caches: List[torch.Tensor],
        block_size: int,
    ) -> None:
        if get_tensor_model_parallel_rank() == 0:
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]
            seq_data = seq_group_metadata.seq_data[seq_id]

            context_len = seq_data.get_num_computed_tokens()
            token_chunk_size = seq_group_metadata.token_chunk_size
            tokens = seq_data.get_prompt_token_ids()

            # alignment `context_len` to `self.chunk_size`
            update_context_len = context_len - context_len % self.chunk_size
            update_token_size = context_len + token_chunk_size - update_context_len
            update_token_size -= update_token_size % self.chunk_size
            update_prefix = tokens[:update_context_len]
            update_tokens = tokens[update_context_len:update_context_len+update_token_size]

            update_args = [
                seq_id,
                update_context_len,
                update_token_size,
                update_prefix,
                update_tokens,
            ]
            tensor_model_parallel_broadcast_object_list(update_args, src=0)
        else:
            update_args = [None, None, None, None, None]
            tensor_model_parallel_broadcast_object_list(update_args, src=0)
            (seq_id,
             update_context_len,
             update_token_size,
             update_prefix,
             update_tokens,
            ) = update_args
        if update_token_size <= 0:
            # restore the seq_group_metadata's and seq's metadata
            self._update_seq_group_metadata(seq_group_metadata, -matched[seq_id])
            return
        
        if get_tensor_model_parallel_rank() == 0:
            block_table = seq_group_metadata.block_tables[seq_id]
            slot_mapping = []
            for i in range(update_context_len, update_context_len + update_token_size):
                block_number = block_table[i // block_size]
                block_offset = i % block_size
                slot = block_number * block_size + block_offset
                slot_mapping.append(slot)
            slot_mapping = torch.tensor(slot_mapping, dtype=torch.long, device='cuda')
            tensor_model_parallel_broadcast(slot_mapping, src=0)
        else:
            slot_mapping = torch.zeros((update_token_size,), dtype=torch.long, device='cuda')
            tensor_model_parallel_broadcast(slot_mapping, src=0)

        if self.enable_async_update:
            buffer_tensors_tuple = self.tensor_pool.get(block=False)
            # buffer_tensors_tuple is None means that we have max number of async
            # updates in the flight. Right now, we just skip updates if we have no
            # buffer.
            if buffer_tensors_tuple is None:
                # restore the seq_group_metadata's and seq's metadata
                self._update_seq_group_metadata(seq_group_metadata, -matched[seq_id])
                return
        else:
            # if async update is disabled, its safe to reuse the same buffer and
            # tensors used in the fetch operation
            buffer_tensors_tuple = self.fetch_buffer, self.fetch_tensors

        update_buffer, _ = buffer_tensors_tuple


        # fetch from GPU kv cache
        if self.metrics_enabled:
            torch.cuda.synchronize()
            start_unload = torch.cuda.Event(enable_timing=True)
            end_unload = torch.cuda.Event(enable_timing=True)
            start_unload.record()
        # using a cuda staging buffer to avoid the inefficiency of non-contiguous HBM->DRAM memcpy
        for j in range(self.layer):
            self.cuda_buffer[:, j, :update_token_size].copy_(
                kv_caches[j][:, slot_mapping // block_size, slot_mapping % block_size])
        update_buffer.copy_(self.cuda_buffer)
        if self.metrics_enabled:
            end_unload.record()
            end_unload.synchronize()
            duration = start_unload.elapsed_time(end_unload) / 1000.0
            self.metrics.add_time_unload(duration)   

        start_time = time.perf_counter()

        update_task = partial(self._update_kv_cache,
            prefix=update_prefix,
            tokens=update_tokens,
            buffer_tensors_tuple=buffer_tensors_tuple,
            scheduled_time=start_time,
        )
        if self.enable_async_update:
            # async update
            try:
                logger.debug(
                    f"submit update task: #prefix={len(update_prefix)}, #tokens={len(update_tokens)}"
                )
                self._update_tasks.put_nowait(update_task)
                logger.debug(
                    f"task queue size={self._update_tasks.qsize()}, tensor pool size={self.tensor_pool.size}"
                )
            except Full:
                logger.warning(f"update_seq_kv_caches: queue is full, skip this update")
                if self.metrics_enabled:
                    self.metrics.err_async_update_task_queue_full += 1
                self.tensor_pool.put(buffer_tensors_tuple)
        else:
            update_task()
            
        if self.metrics_enabled:
            duration = time.perf_counter() - start_time
            self.metrics.add_time_update(duration)   

        # restore the seq_group_metadata's and seq's metadata
        self._update_seq_group_metadata(seq_group_metadata, -matched[seq_id])

    def update_kv_caches(
        self,
        matched: Dict[int, int],
        seq_group_metadata_list: List[SequenceGroupMetadata],
        kv_caches: List[torch.Tensor],
        block_size: int,
    ) -> None:
        if block_size is None or kv_caches[0] is None:  # profile run
            return

        if get_tensor_model_parallel_rank() == 0:
            prefill_requests = []
            for seq_group_meta in seq_group_metadata_list:
                if seq_group_meta.is_prompt:
                    prefill_requests.append(seq_group_meta)
            num_prefill_requests = [len(prefill_requests)]
            tensor_model_parallel_broadcast_object_list(num_prefill_requests, src=0)
        else:
            num_prefill_requests = [None]
            tensor_model_parallel_broadcast_object_list(num_prefill_requests, src=0)
            prefill_requests = [None] * num_prefill_requests[0]
        num_prefill_requests = num_prefill_requests[0]

        for seq_group_meta in prefill_requests:
            self.update_seq_kv_caches(
                matched, seq_group_meta, kv_caches, block_size,
            )

    def __repr__(self):
        return (
            f'VineyardLLMCache('
            f'tensor_nbytes={self.tensor_nbytes}, '
            f'cache_capacity={self.cache_capacity}, '
            f'layer={self.layer}, '
            f'kv_cache_dtype={self.kv_cache_dtype}, '
            f'torch_dtype={self.torch_dtype}, '
            f'cache={self.cache})'
        )
