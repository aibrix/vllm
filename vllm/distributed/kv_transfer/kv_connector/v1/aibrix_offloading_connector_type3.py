# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import contextlib
import logging
import queue
import threading
import time
from typing import TYPE_CHECKING, Any, Optional

import aibrix_kvcache.envs
import msgspec
import torch
import torch.distributed as dist
import zmq
from aibrix_kvcache import KVCacheBlockLayout, KVCacheHandle, TokenListView
from aibrix_kvcache._custom_ops import (reshape_and_cache_multi_layer,
                                        reshape_and_offload_multi_layer)
from aibrix_kvcache.common.absl_logging import (getLogger, log_every_n_seconds,
                                                log_if)
from aibrix_kvcache.profiling import tag_wrapper
from aibrix_kvcache.utils import perf_timer

import vllm.envs
# from vllm.v1.attention.backends.flashinfer import FlashInferBackend
# from vllm.v1.attention.backends.flex_attention import FlexAttentionBackend
# from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.utils import (get_ip, make_zmq_path, make_zmq_socket, round_down,
                        round_up)
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

from .aibrix_offloading_connector_type1 import (
    AIBrixOffloadingConnectorMetadata,
    AIBrixOffloadingConnectorRequestMetadata,
    AIBrixOffloadingConnectorRequestState)
from .aibrix_offloading_connector_type1 import (
    AIBrixOffloadingConnectorScheduler as AIBrixOffloadingConnectorSchedulerType1)
from .aibrix_offloading_connector_type1 import (
    AIBrixOffloadingConnectorWorker as AIBrixOffloadingConnectorWorkerType1)
from .aibrix_offloading_connector_type1 import delegate_to

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

logger = getLogger(__name__)

OFFLOADING_CONNECTOR_SKIP_THRESHOLD = 8
OFFLOADING_CONNECTOR_SUPPORTED_ATTN_BACKENDS = {
    FlashAttentionBackend.get_name(): KVCacheBlockLayout.LCND,
}
OFFLOADING_CONNECTOR_TIMEOUT_S = 0.1
OFFLOADING_CONNECTOR_TEST_BATCH_MAX = 128


class AIBrixOffloadingConnectorWorkerMeta(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        # required for @cached_property.
        dict=True):
    tp_rank: int


class AIBrixOffloadingConnectorTestRequest(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        # required for @cached_property.
        dict=True):
    req_id: str
    seq_token_ids: list[int]


class AIBrixOffloadingConnectorTestResponse(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        # required for @cached_property.
        dict=True):
    req_id: str
    seq_existing_len: int


class AIBrixOffloadingConnectorScheduler(
        AIBrixOffloadingConnectorSchedulerType1):

    def __init__(self, config: "VllmConfig"):
        super().__init__(config)

        if config.parallel_config.data_parallel_backend == "ray":
            sched_ip = get_ip()
        else:
            sched_ip = config.parallel_config.data_parallel_master_ip

        self.tp_size = config.parallel_config.tensor_parallel_size
        self.side_channel_host = sched_ip
        self.side_channel_port = (vllm.envs.VLLM_AIBRIX_SIDE_CHANNEL_PORT +
                                  config.parallel_config.data_parallel_rank)
        self.dp_rank = config.parallel_config.data_parallel_rank

        self._alloc_block_ids: dict[str, list[int]] = {}
        self.cache_block_ntokens = \
            aibrix_kvcache.envs.AIBRIX_KV_CACHE_OL_BLOCK_SIZE
        if self.cache_block_ntokens < 0:
            self.cache_block_ntokens = self.engine_block_ntokens

        self._testing_requests: dict[str, int] = {}
        self._pending_requests = queue.Queue()
        self._testing_result_queue = queue.Queue()

        # init side channels
        self._zmq_ctx = zmq.Context()

        self._sidechannel_path = make_zmq_path(
            "tcp",
            self.side_channel_host,
            self.side_channel_port,
        )
        self._sidechannel_sock = make_zmq_socket(
            ctx=self._zmq_ctx,
            path=self._sidechannel_path,
            socket_type=zmq.ROUTER,
            bind=True,
        )

        self._worker_identities: dict[int, Any] = {}
        self._handshake_listener_thread = threading.Thread(
            target=self._handshake_listener,
            args=(),
            daemon=True,
            name="sched_sidechannel")
        self._handshake_listener_thread.start()
        self._sidechannels_ready = False

    def __del__(self):
        if hasattr(self, "_zmq_ctx") and self._zmq_ctx is not None:
            self._zmq_ctx.destroy(linger=0)

    def _handshake_listener(self):
        logger.debug("Starting listening on path: %s", self._sidechannel_path)
        decoder = msgspec.msgpack.Decoder(AIBrixOffloadingConnectorWorkerMeta)
        while True:
            identity, _, msg = self._sidechannel_sock.recv_multipart()
            metadata = decoder.decode(msg)
            worker_tp_rank = metadata.tp_rank
            if 0 <= worker_tp_rank < self.tp_size:
                if worker_tp_rank in self._worker_identities:
                    logger.warning("Got duplicated worker tp rank: %s",
                                   worker_tp_rank)
                else:
                    logger.info(
                        "Worker %d in DP group %d has established side "
                        "channel",
                        worker_tp_rank,
                        self.dp_rank,
                    )
                self._worker_identities[worker_tp_rank] = identity
            else:
                logger.error("Got invalid worker tp rank: %s", worker_tp_rank)

            if len(self._worker_identities) == self.tp_size:
                logger.info(
                    "Side channels for all workers in DP group %d have "
                    "been established",
                    self.dp_rank,
                )
                break
            else:
                logger.info(
                    "Established %d side channels for DP group %d",
                    len(self._worker_identities),
                    self.dp_rank,
                )

        # start polling
        self._sidechannel_run()

    def _sidechannel_run(self):
        logger.debug(
            "Starting scheduler side channel on path: %s",
            self._sidechannel_path,
        )
        decoder = msgspec.msgpack.Decoder(
            list[AIBrixOffloadingConnectorTestResponse])
        poller = zmq.Poller()
        poller.register(self._sidechannel_sock, zmq.POLLIN)
        while True:
            test_requests: list[AIBrixOffloadingConnectorTestRequest] = []
            while len(test_requests) < OFFLOADING_CONNECTOR_TEST_BATCH_MAX:
                req = self._pending_requests.get()
                test_requests.append(req)
                if self._pending_requests.empty():
                    break
            if len(test_requests) == 0:
                continue

            encoder = msgspec.msgpack.Encoder()
            encoded_data = encoder.encode(test_requests)
            # only test requests on worker-0 to alleviate comm overhead
            id0 = self._worker_identities[0]
            self._sidechannel_sock.send_multipart((id0, b"", encoded_data))

            ready = dict(poller.poll(OFFLOADING_CONNECTOR_TIMEOUT_S * 1000))
            if ready.get(self._sidechannel_sock) != zmq.POLLIN:
                logger.error("Timeout waiting for responses from worker %s",
                             id0.hex())
                continue

            id, _, msg = self._sidechannel_sock.recv_multipart()
            responses = decoder.decode(msg)
            for response in responses:
                seq_existing_len = response.seq_existing_len
                logger.debug(
                    "Test Request[id=%s] %s returns %d",
                    response.req_id,
                    id.hex(),
                    seq_existing_len,
                )
                self._testing_result_queue.put(
                    (response.req_id, seq_existing_len))

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        req_id = request.request_id
        prompt_len = len(request.prompt_token_ids)

        threshold = max(
            OFFLOADING_CONNECTOR_SKIP_THRESHOLD * self.engine_block_ntokens,
            self.cache_block_ntokens,
        )
        if prompt_len < threshold:
            return

        (block_ids, ) = blocks.get_block_ids()
        self._alloc_block_ids.setdefault(req_id, []).extend(block_ids)

    def load_and_prefetch(
        self,
        request: "Request",
        num_to_load_tokens: int,
        num_to_prefetch_tokens: int,
    ) -> int:
        req_id = request.request_id

        prompt_len = len(request.prompt_token_ids)
        threshold = max(
            OFFLOADING_CONNECTOR_SKIP_THRESHOLD * self.engine_block_ntokens,
            self.cache_block_ntokens,
        )
        if prompt_len < threshold:
            logger.debug(
                "Skip Request[id=%s, prompt_len=%d]: prompt len too short",
                req_id,
                prompt_len,
            )
            return 0

        is_new_request = req_id not in self._scheduler_meta

        loaded_len = request.num_computed_tokens
        block_ids = self._alloc_block_ids.get(req_id, None)

        if block_ids is not None:
            aligned_loaded_len = round_up(loaded_len,
                                          self.engine_block_ntokens)
            num_loaded_blocks = aligned_loaded_len // self.engine_block_ntokens
            if len(block_ids) > num_loaded_blocks:
                seq_slot_mapping = (
                    aligned_loaded_len,
                    self._block_ids_to_slot_mapping(block_ids)\
                        [aligned_loaded_len:],
                )
            else:
                # this is a hybrid request, the block ids in _alloc_block_ids
                # are for compute not for loading
                seq_slot_mapping = None
        else:
            seq_slot_mapping = None

        context_len = request.num_computed_tokens + num_to_load_tokens
        query_len = round_down(num_to_prefetch_tokens,
                               self.cache_block_ntokens)

        if num_to_load_tokens == 0 and query_len == 0:
            logger.debug(
                "Skip Request[id=%s, prompt_len=%d, computed_len=%d]: no tokens"
                " to load or prefetch",
                req_id,
                prompt_len,
                request.num_computed_tokens,
            )
            return 0

        seq_len = min(context_len + query_len, prompt_len)

        if not is_new_request:
            seq_token_ids = (
                context_len,
                request.prompt_token_ids[context_len:seq_len],
            )
        else:
            seq_token_ids = (0, request.prompt_token_ids[:seq_len])

        self._scheduler_meta.upsert_request(
            req_id,
            prompt_len=prompt_len,
            context_len=context_len,
            query_len=query_len,
            load_len=num_to_load_tokens,
            seq_token_ids=seq_token_ids,
            seq_slot_mapping=seq_slot_mapping,
            state=AIBrixOffloadingConnectorRequestState.WAITING_FOR_RECV,
        )
        return query_len

    def test_request(
        self,
        request: "Request",
    ) -> bool:
        if not self._sidechannels_ready:
            return False

        req_id = request.request_id
        if req_id not in self._testing_requests:
            return False

        while self._testing_requests[req_id] == -1:
            try:
                (seq_req_id,
                 seq_existing_len) = self._testing_result_queue.get(
                     timeout=OFFLOADING_CONNECTOR_TIMEOUT_S)
            except Exception:
                self._testing_requests.pop(req_id, None)
                return False
            if seq_req_id in self._testing_requests:
                self._testing_requests[seq_req_id] = seq_existing_len

        return self._testing_requests[req_id] > 0

    def add_requests(
        self,
        requests: list["Request"],
    ) -> None:
        num_to_test_tokens = max(
            OFFLOADING_CONNECTOR_SKIP_THRESHOLD * self.engine_block_ntokens,
            self.cache_block_ntokens,
        )

        for request in requests:
            req_id = request.request_id
            if req_id in self._testing_requests:
                continue

            prompt_len = len(request.prompt_token_ids)
            if prompt_len < num_to_test_tokens:
                logger.debug(
                    "Skip testing Request[id=%s, prompt_len=%d]: too short",
                    req_id,
                    prompt_len,
                )
                self._testing_requests[req_id] = 0
                continue

            logger.debug(
                "Test Request[id=%s, prompt_len=%d]",
                req_id,
                prompt_len,
            )

            seq_token_ids = request.prompt_token_ids[:num_to_test_tokens]
            test = AIBrixOffloadingConnectorTestRequest(
                req_id,
                seq_token_ids=seq_token_ids,
            )
            self._testing_requests[req_id] = -1
            self._pending_requests.put(test)

    def build_connector_meta(
            self, scheduler_output: "SchedulerOutput") -> KVConnectorMetadata:
        # 1. new requests
        for req in scheduler_output.scheduled_new_reqs:
            req_id = req.req_id

            prompt_len = len(req.prompt_token_ids)
            context_len = req.num_computed_tokens
            query_len = scheduler_output.num_scheduled_tokens[req_id]

            if context_len >= prompt_len:
                continue

            seq_token_ids = (0, req.prompt_token_ids)
            (block_ids, ) = req.block_ids
            seq_slot_mapping = (0, self._block_ids_to_slot_mapping(block_ids))

            self._scheduler_meta.upsert_request(
                req_id,
                prompt_len=prompt_len,
                context_len=context_len,
                load_len=0,
                query_len=query_len,
                seq_token_ids=seq_token_ids,
                seq_slot_mapping=seq_slot_mapping,
                state=AIBrixOffloadingConnectorRequestState.WAITING_FOR_SEND,
            )

        # 2. cached requests
        cached_reqs = scheduler_output.scheduled_cached_reqs
        req_ids = cached_reqs.req_ids
        for i in range(len(req_ids)):
            req_id = req_ids[i]

            if (req_id not in self._scheduler_meta
                    or req_id not in scheduler_output.num_scheduled_tokens):
                continue

            req_meta = self._scheduler_meta[req_id]

            prompt_len = req_meta.prompt_len
            context_len = min(cached_reqs.num_computed_tokens[i], prompt_len)
            query_len = min(
                scheduler_output.num_scheduled_tokens[req_id],
                prompt_len - context_len,
            )

            if context_len >= prompt_len:
                continue

            if cached_reqs.resumed_from_preemption[i]:
                logger.debug(
                    "Got preempt Request[id=%s, context_len=%d, query_len=%d]",
                    req_id,
                    context_len,
                    query_len,
                )
                (block_ids, ) = cached_reqs.new_block_ids[i]
                seq_slot_mapping = (0,
                                    self._block_ids_to_slot_mapping(block_ids))
            elif cached_reqs.new_block_ids[i] is not None:
                (block_ids, ) = cached_reqs.new_block_ids[i]
                nblocks = round_down(query_len, self.engine_block_ntokens)
                if nblocks > 0:
                    seq_slot_mapping = (
                        round_up(context_len, self.engine_block_ntokens),
                        self._block_ids_to_slot_mapping(block_ids[-nblocks:]),
                    )
                else:
                    seq_slot_mapping = None
            else:
                seq_slot_mapping = None

            self._scheduler_meta.upsert_request(
                req_id,
                prompt_len=prompt_len,
                context_len=context_len,
                load_len=0,
                query_len=query_len,
                seq_token_ids=None,
                seq_slot_mapping=seq_slot_mapping,
                state=AIBrixOffloadingConnectorRequestState.WAITING_FOR_SEND,
                resumed_from_preemption=cached_reqs.resumed_from_preemption[i],
            )

        # 3. keep requests that are in the WAITING_FOR_RECV/SEND state
        meta = self._scheduler_meta.get(lambda req: req.state in [
            AIBrixOffloadingConnectorRequestState.WAITING_FOR_RECV,
            AIBrixOffloadingConnectorRequestState.WAITING_FOR_SEND,
        ])

        # 4. update scheduled requests
        for req_id in meta:
            if (self._scheduler_meta[req_id].state ==
                    AIBrixOffloadingConnectorRequestState.WAITING_FOR_RECV):
                self._scheduler_meta.upsert_request(
                    req_id,
                    state=AIBrixOffloadingConnectorRequestState.RECEIVING,
                )
            else:
                self._scheduler_meta.upsert_request(
                    req_id,
                    state=AIBrixOffloadingConnectorRequestState.SENDING,
                )

        # 5. attach finished requests
        meta.finished_requests_ids = self._scheduler_meta.finished_requests_ids
        self._scheduler_meta.finished_requests_ids = set()

        # 6. attach total_num_scheduled_tokens
        meta.total_num_scheduled_tokens = \
            scheduler_output.total_num_scheduled_tokens

        # 7. side channel host and port
        if not self._sidechannels_ready:
            meta.side_channel_host = self.side_channel_host
            meta.side_channel_port = self.side_channel_port
            self._sidechannels_ready = True
        else:
            meta.side_channel_host = ""

        logger.debug("SCHEDULER: build_connector_meta, meta=%s", meta)
        if len(meta.requests) > 0 or len(meta.finished_requests_ids) > 0:
            logger.debug(
                "Num. of scheduled requests: %s",
                len(
                    self._scheduler_meta.get(lambda req: req.state in [
                        AIBrixOffloadingConnectorRequestState.RECEIVING,
                        AIBrixOffloadingConnectorRequestState.SENDING,
                    ])),
            )
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        req_id = request.request_id
        self._alloc_block_ids.pop(req_id, None)
        self._testing_requests.pop(req_id, None)
        return super().request_finished(request, block_ids)

    def request_preempted(
        self,
        request: "Request",
    ) -> None:
        req_id = request.request_id
        self._alloc_block_ids.pop(req_id, None)
        logger.debug("SCHEDULER: Request[id=%s] preempted", req_id)

        self._scheduler_meta.finish_request(req_id)


class AIBrixOffloadingConnectorWorker(AIBrixOffloadingConnectorWorkerType1):
    """AIBrixOffloadingConnectorWorker carries out the data-plane operations.
    """

    def __init__(self, config: "VllmConfig"):
        self._init_worker(
            config,
            max_num_batched_tokens=config.scheduler_config.
            max_num_batched_tokens,
            multi_threaded=True,
        )
        self.dp_rank = config.parallel_config.data_parallel_rank
        # create streams on current device that are dedicated for sends and
        # recvs
        self._send_stream = torch.cuda.Stream()
        self._load_stream = torch.cuda.Stream()
        self._send_slot_mapping = torch.empty(
            config.scheduler_config.max_num_batched_tokens,
            dtype=torch.long,
            device="cuda",
        )
        self._recv_slot_mapping = torch.empty(
            config.scheduler_config.max_num_batched_tokens,
            dtype=torch.long,
            device="cuda",
        )

        self._kv_event_loop, self._kv_thread = self._create_event_loop()

        self._prefetch_reqs: list[
            AIBrixOffloadingConnectorRequestMetadata] = []
        self._load_reqs: list[AIBrixOffloadingConnectorRequestMetadata] = []
        self._send_reqs: list[AIBrixOffloadingConnectorRequestMetadata] = []

        self._allocated_kvcache_handles: dict[str, KVCacheHandle] = {}
        self._acquired_kvcache_handles: dict[str, list[KVCacheHandle]] = {}
        self._prefetch_future: asyncio.Future | None = None

        # side channel
        self._zmq_ctx = zmq.Context()
        self._sidechannel_thread: Optional[threading.Thread] = None

    def __del__(self):
        self._destroy_event_loop(
            self._kv_event_loop,
            self._kv_thread,
        )
        del self._kv_event_loop
        del self._kv_thread

        if (hasattr(self, "_sidechannel_thread")
                and self._sidechannel_thread is not None
                and self._sidechannel_thread.is_alive()):
            self._sidechannel_thread.join()

        if hasattr(self, "_zmq_ctx") and self._zmq_ctx is not None:
            self._zmq_ctx.destroy(linger=0)

    def _create_event_loop(self) \
        -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
        event_loop = asyncio.new_event_loop()
        thread = threading.Thread(target=event_loop.run_forever, daemon=True)
        thread.start()
        return event_loop, thread

    def _destroy_event_loop(self, event_loop: asyncio.AbstractEventLoop,
                            thread: threading.Thread):
        # terminate event loop and thread
        if event_loop is not None and event_loop.is_running():
            with contextlib.suppress(Exception):
                # ignore the exception
                event_loop.call_soon_threadsafe(event_loop.stop)

        if thread is not None and thread.is_alive():
            thread.join()

    def _get_block_layout(self) -> KVCacheBlockLayout:

        if self.attn_backend.get_name() in \
            OFFLOADING_CONNECTOR_SUPPORTED_ATTN_BACKENDS:
            return KVCacheBlockLayout(
                OFFLOADING_CONNECTOR_SUPPORTED_ATTN_BACKENDS[
                    self.attn_backend.get_name()])
        raise NotImplementedError(
            f"Only support attn backends in "
            f"{list(OFFLOADING_CONNECTOR_SUPPORTED_ATTN_BACKENDS.keys())}. "
            f"{self.attn_backend.get_name()} is used.")

    def _release_allocated_kvcache_handles(self):
        for seq_request_id in self._allocated_kvcache_handles:
            self._allocated_kvcache_handles[seq_request_id].release()
        self._allocated_kvcache_handles.clear()

    def _release_allocated_kvcache_handle(self, req_id):
        handle = self._allocated_kvcache_handles.pop(req_id, None)
        if handle is not None:
            handle.release()

    def _pop_acquired_kvcache_handle(self, req_id: str):
        handles = self._acquired_kvcache_handles.get(req_id, None)
        if handles is None or len(handles) == 0:
            return

        handles[0].release()
        self._acquired_kvcache_handles[req_id] = handles[1:]

    def _release_acquired_kvcache_handles(self, req_id: str):
        handles = self._acquired_kvcache_handles.pop(req_id, None)
        if handles is None:
            return

        for handle in handles:
            handle.release()

    @tag_wrapper({
        "connector": "AIBrixOffloadingConnectorV1Type3",
        "func": "start_load_kv"
    })
    def start_load_kv(
        self,
        metadata: AIBrixOffloadingConnectorMetadata,
    ) -> None:
        if (metadata.side_channel_host is not None
                and len(metadata.side_channel_host) > 0):
            self._handshake(metadata)

        if len(metadata) == 0 and len(metadata.finished_requests_ids) == 0:
            return

        logger.debug("start_load_kv %s", metadata)
        self._update_meta_cache(metadata)

        # release kvcache handles for finished/preempted reqs
        for req_id in metadata.finished_requests_ids:
            self._release_acquired_kvcache_handles(req_id)
            self._release_allocated_kvcache_handle(req_id)

        # split prefetch, load and send reqs
        for seq_request_id, seq_request_meta in metadata.items():
            if seq_request_meta.load_len > 0:
                req_id = seq_request_meta.req_id
                handles = self._acquired_kvcache_handles.get(req_id, None)
                if handles is None or len(handles) == 0:
                    logger.debug("Skip loading %s", req_id)
                self._load_reqs.append(seq_request_meta)

            if (seq_request_meta.query_len > 0 and seq_request_meta.state
                    == AIBrixOffloadingConnectorRequestState.RECEIVING):
                self._prefetch_reqs.append(seq_request_meta)

            if (seq_request_meta.query_len > 0 and seq_request_meta.state
                    == AIBrixOffloadingConnectorRequestState.SENDING):
                prompt_len = seq_request_meta.prompt_len
                context_len = seq_request_meta.context_len
                query_len = seq_request_meta.query_len
                # align to block boundary
                aligned_context_len = round_down(context_len,
                                                 self.cache_block_ntokens)
                actual_query_len = context_len + query_len - \
                    aligned_context_len
                aligned_query_len = round_down(actual_query_len,
                                               self.cache_block_ntokens)
                aligned_prompt_len = round_down(prompt_len,
                                                self.cache_block_ntokens)

                if aligned_query_len == 0 or not self._need_to_process(
                        aligned_prompt_len,
                        aligned_context_len,
                ):
                    logger.debug("Skip sending %s", seq_request_meta.req_id)
                else:
                    self._send_reqs.append(seq_request_meta)

        if len(self._prefetch_reqs) > 0:
            self._prefetch_future = asyncio.run_coroutine_threadsafe(
                self._start_prefetch_kv_async(self._prefetch_reqs),
                self._kv_event_loop,
            )

        if len(self._load_reqs) > 0:
            self._start_load_kv(self._load_reqs)

    def _handshake(self, metadata: AIBrixOffloadingConnectorMetadata):
        sidechannel_path = make_zmq_path("tcp", metadata.side_channel_host,
                                         metadata.side_channel_port)
        # init side channel
        self._sidechannel_thread = threading.Thread(
            target=self._sidechannel_run,
            args=(sidechannel_path, ),
            daemon=True,
            name="worker_sidechannel_run")
        self._sidechannel_thread.start()

    def _sidechannel_run(self, sidechannel_path: str):
        sidechannel_sock = make_zmq_socket(
            ctx=self._zmq_ctx,
            path=sidechannel_path,
            socket_type=zmq.REQ,
            bind=False,
        )
        metadata = AIBrixOffloadingConnectorWorkerMeta(tp_rank=self.rank, )
        encoder = msgspec.msgpack.Encoder()
        encoded_data = encoder.encode(metadata)
        sidechannel_sock.send(encoded_data)
        logger.info(
            "Worker %d in DP group %d is establishing side channel",
            self.rank,
            self.dp_rank,
        )

        logger.debug(
            "Starting worker side channel on path: %s",
            sidechannel_path,
        )
        decoder = msgspec.msgpack.Decoder(
            list[AIBrixOffloadingConnectorTestRequest])
        poller = zmq.Poller()
        poller.register(sidechannel_sock, zmq.POLLIN)
        while True:
            for sock, _ in poller.poll():
                msg = sock.recv()
                requests = decoder.decode(msg)
                responses: list[AIBrixOffloadingConnectorTestResponse] = []
                for request in requests:
                    seq_existing_len = self._test_kv_impl(request)
                    response = AIBrixOffloadingConnectorTestResponse(
                        req_id=request.req_id,
                        seq_existing_len=seq_existing_len,
                    )
                    responses.append(response)
                encoded_data = encoder.encode(responses)
                sock.send(encoded_data)

    async def _start_prefetch_kv_async(
        self, prefetch_reqs: list[AIBrixOffloadingConnectorRequestMetadata]
    ) -> dict[str, tuple[int, KVCacheHandle]]:
        logger.debug("Start prefetching %s", prefetch_reqs)
        stats: dict[str, tuple[int, KVCacheHandle]] = {}
        if self.kv_group is not None:
            for seq_request_meta in prefetch_reqs:
                num_fetched_tokens, handle = \
                    self._recv_kv_impl(seq_request_meta)
                if num_fetched_tokens > 0:
                    seq_request_id = seq_request_meta.req_id
                    stats[seq_request_id] = num_fetched_tokens, handle

            if len(stats) > 0:
                for idx, seq_request_meta in enumerate(prefetch_reqs):
                    seq_request_id = seq_request_meta.req_id
                    self._coll_tensor[idx], _ = stats.get(seq_request_id, \
                        (0, None))
                dist.all_reduce(self._coll_tensor[:idx], dist.ReduceOp.MIN,
                                self.kv_group)

                for idx, seq_request_meta in enumerate(prefetch_reqs):
                    seq_request_id = seq_request_meta.req_id
                    if self._coll_tensor[idx] > 0:
                        num_fetched_tokens = self._coll_tensor[idx].item()
                        num_mrs = round_up(
                            num_fetched_tokens,
                            self.cache_block_ntokens,
                        )
                        handle = stats[seq_request_id][1]
                        handle.truncate(num_mrs)
                        stats[seq_request_id] = num_fetched_tokens, handle
                    else:
                        stats[seq_request_id][1].release()
                        stats.pop(seq_request_id, None)

        else:
            for seq_request_meta in prefetch_reqs:
                seq_request_id = seq_request_meta.req_id
                num_fetched_tokens, handle = \
                    self._recv_kv_impl(seq_request_meta)
                if num_fetched_tokens > 0:
                    stats[seq_request_id] = num_fetched_tokens, handle
        return stats

    def _recv_kv_impl(
        self,
        seq_request_meta: AIBrixOffloadingConnectorRequestMetadata,
    ) -> tuple[int, Optional[KVCacheHandle]]:
        logger.debug("_recv_kv_impl: %s", seq_request_meta)
        seq_request_id = seq_request_meta.req_id
        seq_cached_meta = self._meta_cache[seq_request_id]
        seq_all_tokens = seq_cached_meta.get_context_tokens_view()
        assert seq_all_tokens is not None, "seq_all_tokens is None"
        seq_context_len = seq_request_meta.context_len

        prompt_len = seq_request_meta.prompt_len
        query_len = seq_request_meta.query_len

        threshold = max(
            OFFLOADING_CONNECTOR_SKIP_THRESHOLD * self.engine_block_ntokens,
            self.cache_block_ntokens,
        )
        if query_len < threshold:
            logger.debug(
                "Skip Request[id=%s, context_len=%d, query_len=%d]",
                seq_request_id,
                seq_context_len,
                query_len,
            )
            return 0, None

        prefix = seq_all_tokens[:seq_context_len]
        tokens = seq_all_tokens[seq_context_len:seq_context_len + query_len]

        if self._metrics.time_measurement_enabled:
            start = time.perf_counter()

        seq_recv_len = 0

        # get KV caches from offloading service
        status = self.cache.acquire(prefix, tokens)

        if not status.is_ok():
            if not status.is_not_found():
                log_every_n_seconds(
                    logger,
                    logging.ERROR,
                    "Failed to get from offloading service: %s",
                    3,
                    str(status),
                )
            if self._metrics.time_measurement_enabled:
                end = time.perf_counter()
                lat_ms = (end - start) * 1000
                self._metrics._recv_metrics.add(seq_context_len, query_len, 0,
                                                lat_ms)
            return 0, None

        num_fetched_tokens, handle = status.value
        assert isinstance(num_fetched_tokens, int)

        # update recv_len
        seq_recv_len += num_fetched_tokens

        log_if(
            logger,
            logging.INFO,
            "Request[id=%s, prompt_len=%d, context_len=%d] reused %d tokens",
            seq_recv_len > 0,
            seq_request_id,
            prompt_len,
            seq_context_len,
            seq_recv_len,
        )

        if self._metrics.time_measurement_enabled:
            end = time.perf_counter()
            lat_ms = (end - start) * 1000
            self._metrics._recv_metrics.add(seq_context_len, query_len,
                                            seq_recv_len, lat_ms)

        return seq_recv_len, handle

    def _test_kv_impl(
        self,
        request: AIBrixOffloadingConnectorTestRequest,
    ) -> int:
        logger.debug("_test_kv_impl: %s", request)
        seq_request_id = request.req_id
        if (request.seq_token_ids is None or len(request.seq_token_ids) == 0):
            return 0

        prefix = None
        tokens = TokenListView(request.seq_token_ids)

        status = self.cache.exists(prefix, tokens)

        if not status.is_ok():
            return 0

        seq_existing_len = status.value

        log_if(
            logger,
            logging.DEBUG,
            "Request[id=%s] exists %d tokens",
            seq_existing_len > 0,
            seq_request_id,
            seq_existing_len,
        )

        return seq_existing_len

    def _start_load_kv(
        self,
        load_reqs: list[AIBrixOffloadingConnectorRequestMetadata],
    ):
        logger.debug("Start non-layer-wise loading %s", load_reqs)
        self._start_load_kv_impl(load_reqs)

    def _start_load_kv_impl(
        self,
        load_reqs: list[AIBrixOffloadingConnectorRequestMetadata],
    ):
        tensors: list[torch.Tensor] = []
        slot_mapping_offset = 0

        for req in load_reqs:
            seq_request_id = req.req_id
            handles = self._acquired_kvcache_handles.get(seq_request_id, None)
            if handles is None or len(handles) == 0:
                continue

            seq_cached_meta = self._meta_cache[seq_request_id]
            seq_context_len = req.context_len - req.load_len
            seq_load_len = round_up(req.load_len, self.cache_block_ntokens)

            assert seq_context_len >= 0
            logger.debug(
                "Request[id=%s, context_len=%d, load_len=%d]",
                seq_request_id,
                seq_context_len,
                seq_load_len,
            )

            handle = handles[0]
            kv_blocks = handle.to_tensors()
            num_fetched_tokens = len(kv_blocks) * self.cache_block_ntokens
            assert num_fetched_tokens == seq_load_len, \
                f"{num_fetched_tokens}!={seq_load_len}, handles={len(handles)}"

            offset = seq_context_len
            length = num_fetched_tokens

            if slot_mapping_offset + length >= len(self._recv_slot_mapping):
                new_recv_slot_mapping = torch.empty(
                    len(self._recv_slot_mapping) * 2,
                    dtype=torch.long,
                    device="cuda",
                )
                new_recv_slot_mapping[:slot_mapping_offset].copy_(
                    self._recv_slot_mapping[:slot_mapping_offset])
                self._recv_slot_mapping = new_recv_slot_mapping

            self._recv_slot_mapping[slot_mapping_offset:slot_mapping_offset +
                                    length].copy_(
                                        seq_cached_meta.
                                        context_slot_mapping[offset:offset +
                                                             length])
            slot_mapping_offset += length
            tensors.extend(kv_blocks)

        if slot_mapping_offset == 0:
            return

        with torch.cuda.stream(self._load_stream):
            reshape_and_cache_multi_layer(
                tensors,
                self.layers_kv_caches,
                self._recv_slot_mapping[:slot_mapping_offset],
                self.engine_block_ntokens,
                self.kv_cache_dtype,
                self.k_scales,
                self.v_scales,
                self.block_layout.name,
            )

    def save_kv_layer(
        self,
        metadata: AIBrixOffloadingConnectorMetadata,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
    ) -> None:
        if len(self._send_reqs) == 0:
            return

        if not vllm.envs.VLLM_AIBRIX_TYPE3_USE_LAYER_WISE_SAVE:
            return

        logger.debug("save_kv_layer %s %s", layer_name, self._send_reqs)

        is_first_layer = self.layer_name_idx_mapping[layer_name] == 0
        lid = self.layer_name_idx_mapping[layer_name]
        layer_tensors: list[torch.Tensor] = []
        slot_mapping_offset = 0

        if is_first_layer:
            for req in self._send_reqs:
                seq_req_id = req.req_id
                # align to block boundary
                aligned_context_len = round_down(req.context_len,
                                                 self.cache_block_ntokens)
                actual_query_len = req.context_len + req.query_len - \
                    aligned_context_len
                aligned_query_len = round_down(actual_query_len,
                                               self.cache_block_ntokens)
                aligned_prompt_len = round_down(req.prompt_len,
                                                self.cache_block_ntokens)

                if aligned_query_len == 0 or not self._need_to_process(
                        aligned_prompt_len,
                        aligned_context_len,
                ):
                    continue

                seq_cached_meta = self._meta_cache[seq_req_id]
                seq_all_tokens = seq_cached_meta.get_context_tokens_view()
                assert seq_all_tokens is not None, "seq_all_tokens is None"

                prefix = seq_all_tokens[:aligned_context_len]
                tokens = seq_all_tokens[
                    aligned_context_len:aligned_context_len +
                    aligned_query_len + self.cache_block_ntokens]
                exists_status = self.cache.exists(prefix, tokens)
                if exists_status.is_ok():
                    num_existing_tokens = exists_status.value
                    logger.debug(
                        "Request[id=%s] send(%d) encounters %d existing tokens",
                        seq_req_id, aligned_query_len, num_existing_tokens)
                    if (aligned_query_len <= num_existing_tokens
                            or not self._need_to_process(
                                aligned_prompt_len,
                                aligned_context_len + num_existing_tokens,
                            )):
                        continue
                    else:
                        # partially exists
                        aligned_context_len += num_existing_tokens
                        aligned_query_len -= num_existing_tokens

                # update lengths
                metadata[seq_req_id].context_len = aligned_context_len
                metadata[seq_req_id].query_len = aligned_query_len

                # allocate staging buffers that can hold kvcache for all layers
                handle = self._allocate_for_request(seq_req_id,
                                                    aligned_context_len,
                                                    aligned_query_len)
                if handle is None:
                    continue
                else:
                    self._allocated_kvcache_handles[seq_req_id] = handle

        for seq_req_id, seq_allocated_handle in \
            self._allocated_kvcache_handles.items():
            if seq_allocated_handle is None:
                return

            seq_cached_meta = self._meta_cache[seq_req_id]
            seq_context_len = metadata[seq_req_id].context_len

            seq_allocated_tensors = seq_allocated_handle.to_tensors()
            seq_num_tokens = len(
                seq_allocated_tensors) * self.cache_block_ntokens
            self._send_slot_mapping[slot_mapping_offset:slot_mapping_offset +
                                    seq_num_tokens].copy_(
                                        seq_cached_meta.context_slot_mapping[
                                            seq_context_len:seq_context_len +
                                            seq_num_tokens])
            slot_mapping_offset += seq_num_tokens

            # We are using LCND layout, so the tensors can be split by layers
            layer_tensors.extend(
                [t[lid:lid + 1] for t in seq_allocated_tensors])

        if slot_mapping_offset == 0:
            return

        # wait for compute on current stream
        curr_stream = torch.cuda.current_stream()
        self._send_stream.wait_stream(curr_stream)

        # use send stream to carry out async copy from HBM to DRAM
        with torch.cuda.stream(self._send_stream):
            reshape_and_offload_multi_layer(
                layer_tensors,
                [kv_layer],
                self._send_slot_mapping[:slot_mapping_offset],
                self.engine_block_ntokens,
                self.kv_cache_dtype,
                [self.k_scales[lid]],
                [self.v_scales[lid]],
                self.block_layout.name,
            )

    def _allocate_for_request(
        self,
        seq_request_id: str,
        aligned_context_len: int,
        aligned_query_len: int,
    ) -> Optional[KVCacheHandle]:
        logger.debug("_allocate_for_request %s", seq_request_id)

        seq_cached_meta = self._meta_cache[seq_request_id]
        seq_all_tokens = seq_cached_meta.get_context_tokens_view()
        assert seq_all_tokens is not None, "seq_all_tokens is None"

        prefix = seq_all_tokens[:aligned_context_len]
        tokens = seq_all_tokens[aligned_context_len:aligned_context_len +
                                aligned_query_len]
        status = self.cache.allocate_for(prefix, tokens)
        if not status.is_ok():
            log_every_n_seconds(logger, logging.ERROR,
                                "Failed to allocate : %s", 3, str(status))
            return None
        return status.get()

    @tag_wrapper({
        "connector": "AIBrixOffloadingConnectorV1Type3",
        "func": "wait_for_save"
    })
    def wait_for_save(
        self,
        metadata: AIBrixOffloadingConnectorMetadata,
    ) -> None:
        if len(self._send_reqs) == 0:
            return

        logger.debug("wait_for_save %s", metadata)
        if vllm.envs.VLLM_AIBRIX_TYPE3_USE_LAYER_WISE_SAVE:
            # wait for send stream to finish
            self._send_stream.synchronize()

        for req in self._send_reqs:
            self._send_kv_impl(req)

        self._send_reqs.clear()

        # release all allocated handles
        self._release_allocated_kvcache_handles()

    def _send_kv_impl(
        self,
        seq_request_meta: AIBrixOffloadingConnectorRequestMetadata,
    ) -> None:
        logger.debug("_send_kv_impl: %s", seq_request_meta)
        seq_request_id = seq_request_meta.req_id
        seq_context_len = seq_request_meta.context_len
        query_len = seq_request_meta.query_len
        prompt_len = seq_request_meta.prompt_len

        # align to block boundary
        aligned_context_len = round_down(seq_context_len,
                                         self.cache_block_ntokens)
        actual_query_len = seq_context_len + query_len - aligned_context_len
        aligned_query_len = round_down(actual_query_len,
                                       self.cache_block_ntokens)
        aligned_prompt_len = round_down(prompt_len, self.cache_block_ntokens)

        if aligned_query_len == 0 or not self._need_to_process(
                aligned_prompt_len,
                aligned_context_len,
        ):
            return

        assert prompt_len >= aligned_context_len + aligned_query_len, \
            f"{prompt_len}<{aligned_context_len}+{aligned_query_len}"

        seq_cached_meta = self._meta_cache[seq_request_id]
        seq_all_tokens = seq_cached_meta.get_context_tokens_view()
        assert seq_all_tokens is not None, "seq_all_tokens is None"

        if not vllm.envs.VLLM_AIBRIX_TYPE3_USE_LAYER_WISE_SAVE:
            prefix = seq_all_tokens[:aligned_context_len]
            tokens = seq_all_tokens[aligned_context_len:aligned_context_len +
                                    aligned_query_len +
                                    self.cache_block_ntokens]
            exists_status = self.cache.exists(prefix, tokens)
            if exists_status.is_ok():
                num_existing_tokens = exists_status.value
                logger.debug(
                    "Request[id=%s] send(%d) encounters %d existing tokens",
                    seq_request_id, aligned_query_len, num_existing_tokens)
                aligned_prompt_len = round_down(prompt_len,
                                                self.cache_block_ntokens)
                if (aligned_query_len <= num_existing_tokens
                        or not self._need_to_process(
                            aligned_prompt_len,
                            aligned_context_len + num_existing_tokens,
                        )):
                    return
                else:
                    # partially exists
                    aligned_context_len += num_existing_tokens
                    aligned_query_len -= num_existing_tokens
            seq_allocated_handle = self._allocate_for_request(
                seq_request_id, aligned_context_len, aligned_query_len)
        else:
            seq_allocated_handle = self._allocated_kvcache_handles.pop(
                seq_request_id, None)

        if seq_allocated_handle is None:
            return

        seq_allocated_tensors = seq_allocated_handle.to_tensors()
        aligned_query_len = min(
            aligned_query_len,
            len(seq_allocated_tensors) * self.cache_block_ntokens,
        )

        prefix = seq_all_tokens[:aligned_context_len]
        tokens = seq_all_tokens[aligned_context_len:aligned_context_len +
                                aligned_query_len]

        if self._metrics.time_measurement_enabled:
            start = time.perf_counter()

        total_sent = 0
        length = aligned_query_len

        if not vllm.envs.VLLM_AIBRIX_TYPE3_USE_LAYER_WISE_SAVE:
            offset = len(prefix)
            slot_mapping = seq_cached_meta.context_slot_mapping[offset:offset +
                                                                length]

            # wait for compute on current stream
            curr_stream = torch.cuda.current_stream()
            self._send_stream.wait_stream(curr_stream)

            with perf_timer() as get_kernel_offload_dur_ms:
                with torch.cuda.stream(self._send_stream):
                    reshape_and_offload_multi_layer(
                        seq_allocated_tensors,
                        self.layers_kv_caches,
                        slot_mapping,
                        self.engine_block_ntokens,
                        self.kv_cache_dtype,
                        self.k_scales,
                        self.v_scales,
                        self.block_layout.name,
                    )
                self._send_stream.synchronize()

            logger.info("Request[id=%s] offloads %d tokens in %.4f ms",
                        seq_request_id, length, get_kernel_offload_dur_ms())

        # put KV caches to offloading service
        status = self.cache.put(prefix, tokens, seq_allocated_handle)
        if not status.is_ok():
            log_every_n_seconds(logger, logging.ERROR,
                                "Failed to put to offloading service: %s", 3,
                                str(status))
        else:
            total_sent += length

            log_if(
                logger,
                logging.INFO,
                "Request[id=%s, prompt_len=%d, context_len=%d] sent %d tokens",
                total_sent > 0,
                seq_request_id,
                prompt_len,
                seq_context_len,
                total_sent,
            )

        if self._metrics.time_measurement_enabled:
            end = time.perf_counter()
            lat_ms = (end - start) * 1000
            self._metrics._send_metrics.add(seq_context_len, query_len,
                                            total_sent, lat_ms)

    def _need_to_process(self, aligned_prompt_len: int,
                         aligned_context_len: int) -> bool:
        return aligned_prompt_len - aligned_context_len >= max(
            OFFLOADING_CONNECTOR_SKIP_THRESHOLD * self.engine_block_ntokens,
            self.cache_block_ntokens,
        )

    def get_finished(
        self, metadata: AIBrixOffloadingConnectorMetadata,
        finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str | tuple[str, int]]]]:
        if len(metadata) == 0:
            return None, None

        # wait until prefetching is done
        prefetch_output: set[tuple[str, int]] = set()
        req_ids: set[str] = set()
        for req in self._prefetch_reqs:
            req_ids.add(req.req_id)
        for req in self._load_reqs:
            req_ids.add(req.req_id)

        if self._prefetch_future is not None:
            stats = self._prefetch_future.result()
            for req_id in req_ids:
                (num_fetched_tokens, handle) = stats.get(req_id, (0, None))
                if handle is not None:
                    self._acquired_kvcache_handles.setdefault(req_id, [])\
                        .append(handle)
                    prefetch_output.add((req_id, num_fetched_tokens))
                elif num_fetched_tokens > 0:
                    prefetch_output.add((req_id, num_fetched_tokens))
                else:
                    prefetch_output.add((req_id, 0))
            self._prefetch_future = None
        else:
            for req_id in req_ids:
                prefetch_output.add((req_id, 0))

        # wait until loading is done
        if len(self._load_reqs) > 0:
            self._load_stream.synchronize()
            # release curr handle once loading is complete
            for req in self._load_reqs:
                seq_request_id = req.req_id
                self._pop_acquired_kvcache_handle(seq_request_id)

        self._prefetch_reqs.clear()
        self._load_reqs.clear()

        logger.debug("get_finished: %s", prefetch_output)

        if self._metrics.time_measurement_enabled:
            log_every_n_seconds(
                logger,
                logging.INFO,
                self._metrics.log_str(),
                10,
            )

        return None, prefetch_output


class AIBrixOffloadingConnector(KVConnectorBase_V1):
    """AIBrixOffloadingConnector is a KVConnector that offloads KV caches
    to the kv cache offloading service.
    """

    def __init__(self, config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=config, role=role)

        self.connector_scheduler: Optional[
            AIBrixOffloadingConnectorScheduler] = None

        self.connector_worker: Optional[AIBrixOffloadingConnectorWorker] = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = AIBrixOffloadingConnectorScheduler(
                config)
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = AIBrixOffloadingConnectorWorker(config)

    # ==============================
    # Worker-side methods
    # ==============================

    @delegate_to("connector_worker")
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """
        Initialize with the KV caches. Useful for pre-registering the
        KV Caches in the KVConnector (e.g. for NIXL).

        Args: kv_caches:
            dictionary of layer names, kv cache
        """
        pass

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.

        """
        assert self.connector_worker is not None
        assert isinstance(
            self._connector_metadata,
            AIBrixOffloadingConnectorMetadata,
        )
        return self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        pass

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """
        Start saving a layer of KV cache from vLLM's paged buffer
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        assert self.connector_worker is not None
        assert isinstance(
            self._connector_metadata,
            AIBrixOffloadingConnectorMetadata,
        )
        self.connector_worker.save_kv_layer(self._connector_metadata,
                                            layer_name, kv_layer,
                                            attn_metadata)

    def wait_for_save(self) -> None:
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        assert self.connector_worker is not None
        assert isinstance(
            self._connector_metadata,
            AIBrixOffloadingConnectorMetadata,
        )
        self.connector_worker.wait_for_save(self._connector_metadata)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str | tuple[str, int]]]]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer
            (requests that previously returned True from request_finished()),
            tuple of (sending/saving ids, recving/loading ids or
            (recving/loading id, num. of recv'ed/loaded tokens) pairs).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """
        assert self.connector_worker is not None
        assert isinstance(
            self._connector_metadata,
            AIBrixOffloadingConnectorMetadata,
        )
        return self.connector_worker.get_finished(self._connector_metadata,
                                                  finished_req_ids)

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - The number of tokens that can be loaded from the
                  external KV cache beyond what is already computed.
                - `True` if external KV cache tokens will be loaded
                  asynchronously (between scheduler steps). Must be
                  'False' if the first element is 0.
        """
        return 0, False

    @delegate_to("connector_scheduler")
    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        """
        Update KVConnector state after block allocation.

        If get_num_new_matched_tokens previously returned True for a
        request, this function may be called twice for that same request -
        first when blocks are allocated for the connector tokens to be
        asynchronously loaded into, and second when any additional blocks
        are allocated, after the load/transfer is complete.

        Args:
            request (Request): the request object.
            blocks (KVCacheBlocks): the blocks allocated for the request.
            num_external_tokens (int): the number of tokens that will be
                loaded from the external KV cache.
        """
        return

    @delegate_to("connector_scheduler")
    def load_and_prefetch(
        self,
        request: "Request",
        num_to_load_tokens: int,
        num_to_prefetch_tokens: int,
    ) -> int:
        """
        Get num of tokens that can be scheduled for prefetching.

        Args:
            request (Request): the request object.
            num_to_load_tokens (int): the number of tokens that will be
                scheduled to load to GPU memory.
            num_to_prefetch_tokens (int): the number of tokens that will be
                scheduled to prefetch from the external KV cache.
        Returns:
            Num of tokens to prefetch.
        """
        return 0

    @delegate_to("connector_scheduler")
    def add_requests(
        self,
        requests: list["Request"],
    ) -> None:
        """
        Add requests.

        Args:
            requests (list[Request]): list of request objects.
        """
        return

    @delegate_to("connector_scheduler")
    def test_request(
        self,
        request: "Request",
    ) -> bool:
        """
        Test if request has matched tokens.

        Args:
            request (Request): the request object.
        Returns:
            True if the request has matched tokens.
        """
        return False

    @delegate_to("connector_scheduler")
    def build_connector_meta(
            self, scheduler_output: "SchedulerOutput") -> KVConnectorMetadata:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        pass

    @delegate_to("connector_scheduler")
    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        pass

    @delegate_to("connector_scheduler")
    def request_preempted(
        self,
        request: "Request",
    ) -> None:
        """
        Called when a request has been preempted.
        """
        return
