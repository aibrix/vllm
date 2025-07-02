# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import concurrent
import contextlib
import copy
import dataclasses
import enum
import logging
import os
import threading
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar

import torch
import uvloop
from aibrix_kvcache import (BaseKVCacheManager, GroupAwareKVCacheManager,
                            KVCacheBlockLayout, KVCacheBlockSpec,
                            KVCacheConfig, KVCacheMetrics, KVCacheTensorSpec,
                            ModelSpec, TokenListView)
from aibrix_kvcache._custom_ops import (reshape_and_cache_multi_layer,
                                        reshape_and_offload_multi_layer)
from aibrix_kvcache.common.absl_logging import (getLogger, log_every_n_seconds,
                                                log_if)
from aibrix_kvcache.common.cached_pyobject import CachedPyObjectBase
from aibrix_kvcache.metrics import (MS_BUCKETS, TOKEN_BUCKETS,
                                    BaseMetricsExporter,
                                    KVCacheMetricsExporter, Metrics)
from aibrix_kvcache.utils import perf_timer

from vllm.attention import get_attn_backend
# from vllm.v1.attention.backends.flashinfer import FlashInferBackend
# from vllm.v1.attention.backends.flex_attention import FlexAttentionBackend
# from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
from vllm.distributed import (get_tp_group, get_world_group,
                              init_model_parallel_group)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.kv_transfer.kv_transfer_metrics import (
    KVTransferMetrics, KVTransferMetricsExporter)
from vllm.utils import get_kv_cache_torch_dtype, round_down
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

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
OFFLOADING_CONNECTOR_MAX_PENDING_REQUESTS_DEFAULT = 32

OFFLOADING_CONNECTOR_ENABLE_ASYNC_SEND = os.getenv(
    "OFFLOADING_CONNECTOR_ENABLE_ASYNC_SEND", False)

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

T = TypeVar('T')


def delegate_to(
        member_name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that delegates a method call to a member object.
    
    Args:
        member_name: The name of the member attribute to delegate to.
    """

    def decorator(method: Callable[..., T]) -> Callable[..., T]:

        @wraps(method)
        def wrapper(self, *args, **kwargs) -> T:
            member = getattr(self, member_name)
            assert member is not None, f"{member_name} is not set"
            member_method = getattr(member, method.__name__)
            return member_method(*args, **kwargs)

        return wrapper

    return decorator


class AIBrixOffloadingConnectorOpMetrics(Metrics):
    """Op metrics."""

    class OP(enum.Enum):
        SEND = enum.auto()
        RECV = enum.auto()

    num_ops: int = 0
    total_tokens: int = 0
    total_sent_or_recved_tokens: int = 0
    num_prefixes: list[int] = []
    num_tokens: list[int] = []
    num_sent_or_recved_tokens: list[int] = []
    op_lat_ms: Optional[list[int]] = None

    def __init__(
        self,
        op: OP,
        enable_time_measurement: bool = True,
    ) -> None:
        self._op = op
        self._enable_time_measurement = enable_time_measurement
        self._init_optionals()

    def _init_optionals(self) -> None:
        if self._enable_time_measurement:
            self.op_lat_ms = []

    def add(
        self,
        num_prefix: int,
        num_tokens: int,
        num_sent_or_recved_tokens: int,
        lat_ms: int,
    ) -> None:
        self.num_ops += 1
        self.total_tokens += num_tokens
        self.total_sent_or_recved_tokens += num_sent_or_recved_tokens
        self.num_sent_or_recved_tokens.append(num_sent_or_recved_tokens)
        self.num_prefixes.append(num_prefix)
        self.num_tokens.append(num_tokens)
        if self._enable_time_measurement:
            self.op_lat_ms.append(lat_ms)

    def reset(self) -> None:
        self.num_prefixes = []
        self.num_tokens = []
        self.num_sent_or_recved_tokens = []
        self._init_optionals()

    def summary(self) -> str:
        iter_len = len(self.num_prefixes)
        total_prefixes = sum(self.num_prefixes)
        avg_prefixes = total_prefixes / iter_len if iter_len > 0 else 0
        total_tokens = sum(self.num_tokens)
        avg_tokens = total_tokens / iter_len if iter_len > 0 else 0
        total_sent_or_recved_tokens = sum(self.num_sent_or_recved_tokens)
        avg_sent_or_recved_tokens = (total_sent_or_recved_tokens /
                                     iter_len if iter_len > 0 else 0)
        summary = f"{self._op.name}: Num. of ops: {self.num_ops}, " \
                  f"Total num. of tokens: {self.total_tokens}, "
        if self._op is AIBrixOffloadingConnectorOpMetrics.OP.SEND:
            summary += f"Total num. of sent tokens: " \
                       f"{self.total_sent_or_recved_tokens}, "
        else:
            summary += f"Total num. of received tokens: " \
                       f"{self.total_sent_or_recved_tokens}, "
        summary += f"Num. of prefixes (iter): total={total_prefixes}, " \
                   f"avg={avg_prefixes:.2f}, " \
                   f"Num. of tokens (iter): total={total_tokens}, " \
                   f"avg={avg_tokens:.2f}"
        if self._op is AIBrixOffloadingConnectorOpMetrics.OP.SEND:
            summary += f", Num. of sent tokens (iter): " \
                       f"total={total_sent_or_recved_tokens}, " \
                       f"avg={avg_sent_or_recved_tokens:.2f}"
        else:
            summary += f", Num. of received tokens (iter): " \
                       f"total={total_sent_or_recved_tokens}, " \
                       f"avg={avg_sent_or_recved_tokens:.2f}"
        if self._enable_time_measurement:
            total_lat_ms = sum(self.op_lat_ms)
            avg_lat_ms = total_lat_ms / iter_len if iter_len > 0 else 0
            summary += f", Latency (iter, ms): total={total_lat_ms:.2f}, " \
                       f"avg={avg_lat_ms:.2f}"
        if self._op is AIBrixOffloadingConnectorOpMetrics.OP.RECV:
            hit_rate = (self.total_sent_or_recved_tokens * 100 /
                        self.total_tokens) if self.total_tokens > 0 else 0
            summary += f", Hit rate: {hit_rate:.2f}%"
        return summary


class AIBrixOffloadingConnectorOpMetricsExporter(BaseMetricsExporter):
    OP_TYPE_LABELNAME = "op_type"

    def __init__(self, *, prefix, labelnames, counter_cls, gauge_cls,
                 histogram_cls) -> None:
        labelnames = labelnames or []
        labelnames.append(self.OP_TYPE_LABELNAME)

        super().__init__(
            prefix=f"{prefix}ol_connector_",
            labelnames=labelnames,
            counter_cls=counter_cls,
            gauge_cls=gauge_cls,
            histogram_cls=histogram_cls,
        )

        self._init_exporter_fields()

    def _init_exporter_fields(self) -> None:
        self.counter_num_ops = self._counter_cls(
            name=f"{self._prefix}num_ops",
            documentation="Cumulative number of operations.",
            labelnames=self._labelnames)
        self.histogram_iteration_prefixes = self._histogram_cls(
            name=f"{self._prefix}iteration_prefixes",
            documentation="Histogram of number of prefixes per iteration.",
            labelnames=self._labelnames,
            buckets=TOKEN_BUCKETS)
        self.histogram_iteration_tokens = self._histogram_cls(
            name=f"{self._prefix}iteration_tokens",
            documentation="Histogram of number of tokens per iteration.",
            labelnames=self._labelnames,
            buckets=TOKEN_BUCKETS)
        self.histogram_iteration_sent_tokens = self._histogram_cls(
            name=f"{self._prefix}iteration_sent_tokens",
            documentation=("Histogram of number of sent tokens "
                           "per iteration."),
            labelnames=self._labelnames,
            buckets=TOKEN_BUCKETS)
        self.histogram_iteration_received_tokens = self._histogram_cls(
            name=f"{self._prefix}iteration_received_tokens",
            documentation=("Histogram of number of received tokens "
                           "per iteration."),
            labelnames=self._labelnames,
            buckets=TOKEN_BUCKETS)
        self.histogram_iteration_op_lat_ms = self._histogram_cls(
            name=f"{self._prefix}iteration_op_lat_ms",
            documentation=("Histogram of operation latencies "
                           "per iteration in ms."),
            labelnames=self._labelnames,
            buckets=MS_BUCKETS)

    def export(
        self,
        labels: dict[str, str],
        metrics: AIBrixOffloadingConnectorOpMetrics,
    ) -> None:
        labels = labels.copy()

        labels[self.OP_TYPE_LABELNAME] = metrics._op.name.lower()
        self._export_op_metrics(labels, metrics)

    def _export_op_metrics(
        self,
        labels: dict[str, str],
        metrics: AIBrixOffloadingConnectorOpMetrics,
    ) -> None:
        self._export_counter(self.counter_num_ops, labels,
                             len(metrics.num_prefixes))
        self._export_histogram(self.histogram_iteration_prefixes, labels,
                               metrics.num_prefixes)
        self._export_histogram(self.histogram_iteration_tokens, labels,
                               metrics.num_tokens)
        if metrics._op is AIBrixOffloadingConnectorOpMetrics.OP.SEND:
            self._export_histogram(self.histogram_iteration_sent_tokens,
                                   labels, metrics.num_sent_or_recved_tokens)
        else:
            self._export_histogram(self.histogram_iteration_received_tokens,
                                   labels, metrics.num_sent_or_recved_tokens)
        if metrics._enable_time_measurement:
            self._export_histogram(self.histogram_iteration_op_lat_ms, labels,
                                   metrics.op_lat_ms)


class AIBrixOffloadingConnectorMetrics(KVTransferMetrics):

    def __init__(self, metrics: KVCacheMetrics) -> None:
        self._cache_metrics = metrics
        self._time_measurement_enabled = (
            self._cache_metrics.time_measurement_enabled)
        self._send_metrics = AIBrixOffloadingConnectorOpMetrics(
            AIBrixOffloadingConnectorOpMetrics.OP.SEND,
            enable_time_measurement=self._time_measurement_enabled,
        )
        self._recv_metrics = AIBrixOffloadingConnectorOpMetrics(
            AIBrixOffloadingConnectorOpMetrics.OP.RECV,
            enable_time_measurement=self._time_measurement_enabled,
        )

    @property
    def time_measurement_enabled(self) -> bool:
        return self._time_measurement_enabled

    def reset(self) -> None:
        self._cache_metrics.reset()
        self._send_metrics.reset()
        self._recv_metrics.reset()

    def __str__(self) -> str:
        return f"AIBrixOffloadingConnector metrics: " \
               f"{self._send_metrics.summary()}" \
               f"\n\t{self._recv_metrics.summary()}" \
               f"\n\t{self._cache_metrics.summary()}"

    # NOTE: Functions `log` and `findCaller` are used as a workaround to
    # log metrics on the worker side, will be removed once the worker is
    # able to transfer metrics to the scheduler.
    def log(self, level, msg, *args) -> None:
        logger.log(level, str(self))
        self.reset()

    def findCaller(self) -> tuple[str, int, str, str | None]:
        return logger.findCaller()


class AIBrixOffloadingConnectorMetricsExporter(KVTransferMetricsExporter):
    """Metrics for AIBrixOffloadingConnector."""

    def __init__(self, *, prefix, labelnames, gauge_cls, counter_cls,
                 histogram_cls):
        self.kv_cache_metrics_exporter = KVCacheMetricsExporter(
            prefix=prefix,
            labelnames=labelnames,
            gauge_cls=gauge_cls,
            counter_cls=counter_cls,
            histogram_cls=histogram_cls,
        )
        self.ol_connector_op_metrics_exporter = \
            AIBrixOffloadingConnectorOpMetricsExporter(
            prefix=prefix,
            labelnames=labelnames,
            gauge_cls=gauge_cls,
            counter_cls=counter_cls,
            histogram_cls=histogram_cls,
        )

    def export(
        self,
        *,
        metrics: KVTransferMetrics,
        labels: dict[str, str],
    ):
        self.kv_cache_metrics_exporter.export(
            metrics=metrics._cache_metrics,
            labels=labels,
        )
        self.ol_connector_op_metrics_exporter.export(
            metrics=metrics._send_metrics,
            labels=labels,
        )
        self.ol_connector_op_metrics_exporter.export(
            metrics=metrics._recv_metrics,
            labels=labels,
        )


class AIBrixOffloadingConnectorRequestState(enum.IntEnum):
    INIT = enum.auto()
    WAITING_FOR_ALLOC = enum.auto()
    WAITING_FOR_SEND = enum.auto()
    WAITING_FOR_RECV = enum.auto()
    SENDING = enum.auto()
    RECEIVING = enum.auto()


@dataclasses.dataclass
class AIBrixOffloadingConnectorRequestMetadata(CachedPyObjectBase):
    req_id: str = ""
    prompt_len: int = -1
    context_len: int = -1
    query_len: int = -1  # num of tokens to send/recv
    seq_token_ids: list[int] = dataclasses.field(default_factory=list)
    seq_slot_mapping: Optional[torch.Tensor] = None
    gen_completion: bool = True
    state: AIBrixOffloadingConnectorRequestState = (
        AIBrixOffloadingConnectorRequestState.INIT)

    def __str__(self) -> str:
        return (f"AIBrixOffloadingConnectorRequestMetadata["
                f"req_id={self.req_id}, "
                f"prompt_len={self.prompt_len}, "
                f"context_len={self.context_len}, "
                f"query_len={self.query_len}, "
                f"len(seq_token_ids)={len(self.seq_token_ids)}, "
                f"seq_slot_mapping.shape={self.seq_slot_mapping.shape}, "
                f"gen_completion={self.gen_completion}, "
                f"state={self.state.name}]")

    def __repr__(self):
        return self.__str__()


class AIBrixOffloadingConnectorMetadata(KVConnectorMetadata):

    def __init__(self,
                 requests: dict[str,
                                AIBrixOffloadingConnectorRequestMetadata]):
        # Requests that need to load from external kvcache.
        self.requests = requests

    def __getitem__(self,
                    key: str) -> AIBrixOffloadingConnectorRequestMetadata:
        return self.requests[key]

    def __contains__(self, key: str) -> bool:
        return key in self.requests

    def __iter__(self):
        return iter(self.requests)

    def __next__(self):
        return next(self.requests)

    def __len__(self):
        return len(self.requests)

    def items(self):
        return self.requests.items()

    def upsert_request(
        self,
        request_id: str,
        **kwargs,
    ) -> None:
        self.requests.setdefault(
            request_id,
            AIBrixOffloadingConnectorRequestMetadata()).__dict__.update(
                req_id=request_id, **kwargs)

    def pop_request(
        self,
        request_id: str,
    ) -> AIBrixOffloadingConnectorRequestMetadata | None:
        return self.requests.pop(request_id, None)

    def get(self, predicate: Callable) -> "AIBrixOffloadingConnectorMetadata":
        requests = {k: v for k, v in self.requests.items() if predicate(v)}
        return AIBrixOffloadingConnectorMetadata(requests=requests)

    def filter_requests(self, predicate: Callable) -> None:
        self.requests = {
            k: v
            for k, v in self.requests.items() if not predicate(v)
        }

    def extend(self, other: "AIBrixOffloadingConnectorMetadata") -> None:
        self.requests.update(other.requests)

    def clear(self) -> None:
        self.requests.clear()


class AIBrixOffloadingConnectorScheduler:

    def __init__(self, config: "VllmConfig"):
        self.kv_role = config.kv_transfer_config.kv_role
        self.block_ntokens = config.cache_config.block_size

        self._scheduler_meta = AIBrixOffloadingConnectorMetadata({})

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        # NOTE: right now it should only be invoked for requests with zero
        # num_computed_tokens
        assert request.num_computed_tokens == 0

        # NOTE: in current v1 scheduler, the num_computed_tokens is aligned
        # with the block granularity. And it expects the returned number of
        # matched tokens to also be aligned with the block granularity.
        assert num_computed_tokens % self.block_ntokens == 0

        req_id = request.request_id
        self._scheduler_meta.pop_request(req_id)

        if (len(
                self._scheduler_meta.get(
                    lambda req: req.state ==
                    AIBrixOffloadingConnectorRequestState.RECEIVING))
                > OFFLOADING_CONNECTOR_MAX_PENDING_REQUESTS_DEFAULT):
            logger.debug("Skip Request[id=%s]", req_id)
            return 0, False

        seq_len = len(request.prompt_token_ids)

        logger.debug(
            "SCHEDULER: Request[id=%s] context_len=%s, seq_len=%s",
            req_id,
            num_computed_tokens,
            seq_len,
        )

        # Trying to get all prompt blocks from external kv cache.
        aligned_num_prompt_tokens = round_down(seq_len, self.block_ntokens)

        # Skip receiving if we don't have enough blocks
        if (aligned_num_prompt_tokens
                < OFFLOADING_CONNECTOR_SKIP_THRESHOLD * self.block_ntokens):
            return 0, False

        count = max(aligned_num_prompt_tokens - num_computed_tokens, 0)

        self._scheduler_meta.upsert_request(
            req_id,
            prompt_len=seq_len,
            context_len=num_computed_tokens,
            seq_token_ids=request.prompt_token_ids[:seq_len],
            state=AIBrixOffloadingConnectorRequestState.WAITING_FOR_ALLOC,
        )

        return count, count > 0

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        # NOTE: If get_num_new_matched_tokens previously returned True for a
        # request, this function may be called twice for that same request -
        # first when blocks are allocated for the connector tokens to be
        # asynchronously loaded into, and second when any additional blocks
        # are allocated, after the recv is complete.
        # NOTE: We rely on num_external_tokens to determine whether we need to
        # recv. Recv if num_external_tokens > 0.

        req_id = request.request_id

        if num_external_tokens > 0:
            # recv
            if (req_id not in self._scheduler_meta
                    or self._scheduler_meta[req_id].state !=
                    AIBrixOffloadingConnectorRequestState.WAITING_FOR_ALLOC):
                self._scheduler_meta.pop_request(req_id)
                return

            # Skip receiving if we don't have enough allocated blocks
            if (num_external_tokens // self.block_ntokens
                    < OFFLOADING_CONNECTOR_SKIP_THRESHOLD):
                logger.debug("Skip recving Request[id=%s]", req_id)
                self._scheduler_meta.pop_request(req_id)
                return

            (block_ids, ) = blocks.get_block_ids()
            slot_mapping = self._block_ids_to_slot_mapping(block_ids)

            # For recv request, query_len is the tokens to be recv'ed
            context_len = self._scheduler_meta[req_id].context_len
            seq_len = context_len + num_external_tokens
            self._scheduler_meta.upsert_request(
                req_id,
                seq_token_ids=request.prompt_token_ids[:seq_len],
                query_len=num_external_tokens,
                seq_slot_mapping=slot_mapping[:seq_len],
                state=AIBrixOffloadingConnectorRequestState.WAITING_FOR_RECV,
            )

    def build_connector_meta(
            self, scheduler_output: "SchedulerOutput") -> KVConnectorMetadata:
        # NOTE: Since we are using async load, the requests waiting for xfer
        # are not in neither scheduled_new_reqs nor scheduled_cached_reqs.
        # Therefore, tokens of these requests are not counted in
        # total_num_scheduled_tokens.
        # NOTE: At this point, request.num_computed_tokens is not updated yet
        # and it will be updated in `Scheduler._update_waiting_for_remote_kv`
        # once its xfer is completed.

        # 1. remove finished requests
        for req_id in scheduler_output.finished_req_ids:
            self._scheduler_meta.pop_request(req_id)

        # 2. new requests
        for req in scheduler_output.scheduled_new_reqs:
            req_id = req.req_id

            prompt_len = len(req.prompt_token_ids)
            context_len = req.num_computed_tokens
            query_len = scheduler_output.num_scheduled_tokens[req_id]
            seq_len = min(context_len + query_len, prompt_len)

            aligned_prompt_len = round_down(prompt_len, self.block_ntokens)
            aligned_context_len = round_down(context_len, self.block_ntokens)
            aligned_query_len = round_down(max(seq_len - context_len, 0),
                                           self.block_ntokens)
            aligned_seq_len = aligned_context_len + aligned_query_len
            aligned_remaining_len = max(aligned_prompt_len - aligned_seq_len,
                                        0)

            if (context_len < prompt_len and aligned_query_len
                    < OFFLOADING_CONNECTOR_SKIP_THRESHOLD *
                    self.block_ntokens):
                logger.debug(
                    "Skip sending Request[id=%s, context_len=%d, query_len=%d]",
                    req_id,
                    aligned_context_len,
                    aligned_query_len,
                )
                continue

            need_gen_completion = False
            if (aligned_remaining_len < OFFLOADING_CONNECTOR_SKIP_THRESHOLD *
                    self.block_ntokens):
                need_gen_completion = True

            (block_ids, ) = req.block_ids
            slot_mapping = self._block_ids_to_slot_mapping(block_ids)

            self._scheduler_meta.upsert_request(
                req_id,
                prompt_len=prompt_len,
                context_len=context_len,
                query_len=query_len,
                seq_token_ids=req.prompt_token_ids[:seq_len],
                seq_slot_mapping=slot_mapping,
                gen_completion=need_gen_completion,
                state=AIBrixOffloadingConnectorRequestState.WAITING_FOR_SEND,
            )

        # 3. cached requests
        for req in scheduler_output.scheduled_cached_reqs:
            req_id = req.req_id

            if req_id not in self._scheduler_meta:
                continue

            req_meta = self._scheduler_meta[req_id]

            prompt_len = req_meta.prompt_len
            context_len = min(req.num_computed_tokens, prompt_len)
            query_len = min(
                scheduler_output.num_scheduled_tokens[req_id],
                prompt_len - context_len,
            )
            seq_len = min(context_len + query_len, prompt_len)

            aligned_prompt_len = round_down(prompt_len, self.block_ntokens)
            aligned_context_len = round_down(context_len, self.block_ntokens)
            aligned_query_len = round_down(max(seq_len - context_len, 0),
                                           self.block_ntokens)
            aligned_seq_len = aligned_context_len + aligned_query_len
            aligned_remaining_len = max(aligned_prompt_len - aligned_seq_len,
                                        0)

            if (context_len < prompt_len and aligned_query_len
                    < OFFLOADING_CONNECTOR_SKIP_THRESHOLD *
                    self.block_ntokens):
                logger.debug(
                    "Skip sending Request[id=%s, context_len=%d, query_len=%d]",
                    req_id,
                    aligned_context_len,
                    aligned_query_len,
                )
                continue

            need_gen_completion = False
            if (aligned_remaining_len < OFFLOADING_CONNECTOR_SKIP_THRESHOLD *
                    self.block_ntokens):
                need_gen_completion = True

            (block_ids, ) = req.new_block_ids
            new_slot_mapping = self._block_ids_to_slot_mapping(block_ids)

            if req.resumed_from_preemption:
                logger.debug(
                    "Got preempt Request[id=%s, context_len=%d, query_len=%d]",
                    req_id,
                    aligned_context_len,
                    aligned_query_len,
                )
                seq_token_ids = req.new_token_ids
                seq_slot_mapping = new_slot_mapping
            else:
                seq_token_ids = req_meta.seq_token_ids + req.new_token_ids
                if new_slot_mapping.shape[0] > 0:
                    seq_slot_mapping = torch.cat(
                        [req_meta.seq_slot_mapping, new_slot_mapping])
                else:
                    seq_slot_mapping = req_meta.seq_slot_mapping

            self._scheduler_meta.upsert_request(
                req_id,
                prompt_len=prompt_len,
                context_len=context_len,
                query_len=query_len,
                seq_token_ids=seq_token_ids[:seq_len],
                seq_slot_mapping=seq_slot_mapping,
                gen_completion=need_gen_completion,
                state=AIBrixOffloadingConnectorRequestState.WAITING_FOR_SEND,
            )

        # 4. keep requests that are in the WAITING_FOR_RECV/SEND state
        meta = copy.deepcopy(
            self._scheduler_meta.get(lambda req: req.state in [
                AIBrixOffloadingConnectorRequestState.WAITING_FOR_RECV,
                AIBrixOffloadingConnectorRequestState.WAITING_FOR_SEND,
            ]))

        logger.debug("SCHEDULER: build_connector_meta, meta=%s", meta.__dict__)

        # 5. update scheduled requests
        for req_id in meta:
            if self._scheduler_meta[req_id].state == \
                AIBrixOffloadingConnectorRequestState.WAITING_FOR_RECV:
                self._scheduler_meta.upsert_request(
                    req_id,
                    state=AIBrixOffloadingConnectorRequestState.RECEIVING,
                )
            else:
                self._scheduler_meta.upsert_request(
                    req_id,
                    state=AIBrixOffloadingConnectorRequestState.SENDING,
                )

        logger.debug(
            "Num. of scheduled requests: %s",
            len(
                self._scheduler_meta.get(lambda req: req.state in [
                    AIBrixOffloadingConnectorRequestState.SENDING,
                    AIBrixOffloadingConnectorRequestState.RECEIVING,
                ])),
        )
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        req_id = request.request_id
        logger.debug("SCHEDULER: Request[id=%s] finished", req_id)

        if req_id not in self._scheduler_meta:
            return False, None

        if OFFLOADING_CONNECTOR_ENABLE_ASYNC_SEND:
            if self._scheduler_meta[req_id].state != \
                AIBrixOffloadingConnectorRequestState.SENDING:
                self._scheduler_meta.pop_request(req_id)
                return False, None
            else:
                self._scheduler_meta.pop_request(req_id)
                # Let scheduler wait until the async sending is done
                return True, None
        else:
            # use sync sending
            self._scheduler_meta.pop_request(req_id)
            return False, None

    def _block_ids_to_slot_mapping(self, block_ids: list[int]) -> torch.Tensor:
        block_ids_tensor = torch.tensor(block_ids)
        num_blocks = block_ids_tensor.shape[0]
        block_offsets = torch.arange(0, self.block_ntokens)
        slot_mapping = block_offsets.reshape((1, self.block_ntokens)) + \
                block_ids_tensor.reshape((num_blocks, 1)) * self.block_ntokens
        return slot_mapping.flatten()


class AIBrixOffloadingConnectorWorker:
    """AIBrixOffloadingConnectorWorker carries out the data-plane operations.
    """

    def __init__(self, config: "VllmConfig"):

        cache_config = config.cache_config
        model_config = config.model_config
        parallel_config = config.parallel_config

        tp_size = parallel_config.tensor_parallel_size
        num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        hidden_size = model_config.get_hidden_size()
        num_attention_heads = model_config.get_num_attention_heads(
            parallel_config) * tp_size
        head_size = int(hidden_size / num_attention_heads)

        rank = get_tp_group().rank_in_group
        kv_head_ids = list(
            range(num_kv_heads * rank, num_kv_heads * (rank + 1)))
        layer_ids = list(
            range(*model_config.get_layers_start_end_indices(parallel_config)))
        num_layers = len(layer_ids)

        block_ntokens = cache_config.block_size
        block_dtype = get_kv_cache_torch_dtype(cache_config.cache_dtype,
                                               model_config.dtype)

        kv_cache_dtype = cache_config.cache_dtype

        self.attn_backend = get_attn_backend(
            model_config.get_head_size(),
            model_config.dtype,
            kv_cache_dtype,
            block_ntokens,
            model_config.is_attention_free,
            use_mla=model_config.use_mla,
        )

        block_spec = KVCacheBlockSpec(
            block_ntokens=block_ntokens,
            block_dtype=block_dtype,
            block_layout=self._get_block_layout(),
            tensor_spec=KVCacheTensorSpec(
                heads=kv_head_ids,
                layers=layer_ids,
                head_size=head_size,
            ),
        )

        kv_config = KVCacheConfig(block_spec=block_spec,
                                  model_spec=ModelSpec(
                                      model_config.max_model_len),
                                  multi_threading=True)

        if parallel_config.tensor_parallel_size == 1:
            self.cache = BaseKVCacheManager(config=kv_config)
        else:
            backend = torch.distributed.get_backend(get_world_group()\
                .device_group)
            world_size = parallel_config.world_size
            dp_size = parallel_config.data_parallel_size
            pp_size = parallel_config.pipeline_parallel_size
            # the layout order is: ExternalDP x DP x PP x TP
            # ExternalDP is the data parallel group that is not part of the
            # model, every dp rank can generate independently (in verl
            # integration).
            # DP is the data parallel group that is part of the model,
            # all the ranks in the same DP group should generate simultaneously,
            # i.e. the `generate` call in the same DP group should be called
            # together, otherwise it will cause deadlock.
            # to get group_ranks for each dimension, transpose that dimension to
            # the last dimension, then reshape to 2D, then unbind the last
            # dimension
            all_ranks = torch.arange(world_size).reshape(
                -1, dp_size, pp_size, tp_size)

            # Build the kv model-parallel groups.
            group_ranks = all_ranks.view(-1, tp_size).unbind(0)
            group_ranks = [x.tolist() for x in group_ranks]

            kv_group = init_model_parallel_group(
                group_ranks,
                get_world_group().local_rank,
                backend,
                group_name="kvcache",
            )
            send_group = init_model_parallel_group(
                group_ranks,
                get_world_group().local_rank,
                backend,
                group_name="kv_send",
            )
            assert rank == kv_group.rank_in_group
            self.cache = GroupAwareKVCacheManager(
                config=kv_config, process_group=kv_group.cpu_group)
            self.send_group = send_group.cpu_group

        self.rank = rank
        self.head_size = head_size
        self.tp_size = tp_size
        self.num_kv_heads = num_kv_heads
        self.num_layers = num_layers
        self.kv_head_ids = kv_head_ids
        self.layer_ids = layer_ids
        self.block_ntokens = block_ntokens
        self.block_dtype = block_dtype
        self.block_shape = block_spec.block_shape
        self.block_spec = block_spec
        self.block_layout = block_spec.block_layout
        self.chunk_size = self.cache.chunk_size
        self.cache_feature = self.cache.feature
        self.kv_cache_dtype = kv_cache_dtype

        # KV caches and kv scales will be init'ed later
        self.no_compile_layers = config.compilation_config.\
            static_forward_context
        self.kv_caches: dict[str, torch.Tensor] | None = None
        self.layers_kv_caches: list[torch.Tensor] | None = None
        self.k_scales: list[torch.Tensor] | None = None
        self.v_scales: list[torch.Tensor] | None = None

        # Worker thread and cuda stream for transferring kvcaches
        self._send_stream = torch.cuda.Stream()
        self._recv_stream = torch.cuda.Stream()
        self._send_loop = asyncio.new_event_loop()
        self._recv_loop = asyncio.new_event_loop()
        self._send_thread = threading.Thread(
            target=self._send_loop.run_forever, daemon=True)
        self._recv_thread = threading.Thread(
            target=self._recv_loop.run_forever, daemon=True)
        self._send_thread.start()
        self._recv_thread.start()
        self._recving_futures: dict[str, asyncio.Future] = {}
        self._sending_futures: dict[str, asyncio.Future] = {}

        # metrics
        self._metrics = AIBrixOffloadingConnectorMetrics(self.cache.metrics)

    @property
    def metrics(self) -> KVTransferMetrics:
        return self._metrics

    def get_metrics_exporter_cls(self):
        return AIBrixOffloadingConnectorMetricsExporter

    def __del__(self) -> None:
        if getattr(self, "cache", None) is not None:
            self.cache.close()
            self.cache = None

        # terminate event loop and thread
        if (getattr(self, "_send_loop", None) is not None
                and self._send_loop.is_running()):
            with contextlib.suppress(Exception):
                # ignore the exception
                self._send_loop.call_soon_threadsafe(self._send_loop.stop)

        if (getattr(self, "_recv_loop", None) is not None
                and self._recv_loop.is_running()):
            with contextlib.suppress(Exception):
                # ignore the exception
                self._recv_loop.call_soon_threadsafe(self._recv_loop.stop)

        if (getattr(self, "_send_thread", None) is not None
                and self._send_thread.is_alive()):
            self._send_thread.join()

        if (getattr(self, "_recv_thread", None) is not None
                and self._send_thread.is_alive()):
            self._recv_thread.join()

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

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self.kv_caches = kv_caches
        layer_names = self.no_compile_layers.keys()
        self.layers_kv_caches = [
            self.kv_caches[layer_name] for layer_name in layer_names
        ]
        layers = self.no_compile_layers.values()
        self.k_scales = [layer._k_scale for layer in layers]
        self.v_scales = [layer._v_scale for layer in layers]

    def start_load_kv(
        self,
        metadata: AIBrixOffloadingConnectorMetadata,
    ) -> None:
        # NOTE: `start_load_kv` asynchronously loads the KV cache from the
        # external kvcache for the requests in `metadata`, which are not
        # requested for the current computation corresponding to
        # `forward_context`. Therefore, we use another worker thread and cuda
        # stream to asynchronously load the KV cache to GPU.

        assert self._recv_loop is not None, "recv loop is not initialized"
        for seq_request_id, seq_request_meta in metadata.items():
            if (seq_request_meta.state
                    != AIBrixOffloadingConnectorRequestState.WAITING_FOR_RECV):
                continue
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._recv_kv_async(seq_request_meta), self._recv_loop)
                seq_request_meta.state = \
                    AIBrixOffloadingConnectorRequestState.RECEIVING
                if seq_request_meta.gen_completion:
                    self._recving_futures[seq_request_id] = future
                logger.debug("Request[id=%s] start recv kv", seq_request_id)
            except Exception as e:
                if seq_request_meta.gen_completion:
                    future = concurrent.futures.Future()
                    self._recving_futures[seq_request_id] = future
                    future.set_exception(e)

    async def _recv_kv_async(
        self,
        seq_request_meta: AIBrixOffloadingConnectorRequestMetadata,
    ) -> int:
        with torch.cuda.stream(self._recv_stream):
            return self._recv_kv_sync_impl(seq_request_meta)

    def _recv_kv_sync_impl(
        self,
        seq_request_meta: AIBrixOffloadingConnectorRequestMetadata,
    ) -> int:
        logger.debug("_recv_kv_sync_impl: %s", seq_request_meta)
        seq_request_id = seq_request_meta.req_id
        seq_context_len = seq_request_meta.context_len
        seq_all_tokens = seq_request_meta.seq_token_ids
        assert seq_all_tokens is not None, "seq_all_tokens is None"

        prompt_len = seq_request_meta.prompt_len
        query_len = seq_request_meta.query_len

        seq_slot_mapping = seq_request_meta.seq_slot_mapping.cuda(
            non_blocking=True)

        # align to block boundary
        aligned_context_len = round_down(seq_context_len, self.block_ntokens)
        actual_query_len = seq_context_len + query_len - aligned_context_len
        aligned_query_len = round_down(actual_query_len, self.block_ntokens)
        shift_len = seq_context_len - aligned_context_len

        assert prompt_len >= aligned_context_len + aligned_query_len, \
            f"{prompt_len}<{aligned_context_len}+{aligned_query_len}"

        seq_all_tokens_view = TokenListView(seq_all_tokens)
        prefix = seq_all_tokens_view[:aligned_context_len]
        tokens = seq_all_tokens_view[aligned_context_len:aligned_context_len +
                                     aligned_query_len]

        if self._metrics.time_measurement_enabled:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        seq_recv_len = 0
        for (
                chunk_prefix,
                chunk_tokens,
                next_tokens,
                _,
        ) in self.cache.cache_chunk_keys(prefix, tokens):
            if next_tokens and len(next_tokens) > 0:
                # prefetch
                self.cache.prefetch(chunk_prefix + chunk_tokens, next_tokens)

            # get KV caches from offloading service
            status = self.cache.acquire(chunk_prefix, chunk_tokens)

            if not status.is_ok():
                if not status.is_not_found():
                    log_every_n_seconds(
                        logger,
                        logging.ERROR,
                        "Failed to get from offloading service: %s",
                        3,
                        str(status),
                    )
                break

            num_fetched_tokens, handle = status.value
            kv_blocks = handle.to_tensors()

            offset = len(chunk_prefix)
            length = num_fetched_tokens

            chunk_slot_mapping = seq_slot_mapping[offset:offset + length]

            with perf_timer() as get_kernel_onload_dur_ms:
                reshape_and_cache_multi_layer(
                    kv_blocks,
                    self.layers_kv_caches,
                    chunk_slot_mapping,
                    self.block_ntokens,
                    self.kv_cache_dtype,
                    self.k_scales,
                    self.v_scales,
                    self.block_layout.name,
                )

            logger.info(
                "Request[id=%s] onloads %d tokens in %.4f ms",
                seq_request_id,
                length,
                get_kernel_onload_dur_ms(),
            )

            # update recv_len
            seq_recv_len += num_fetched_tokens - shift_len
            # reset shift_len
            shift_len = 0

            # release handle
            handle.release()

            if num_fetched_tokens < len(chunk_tokens):
                # didn't receive all tokens for current chunk, break
                break

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
            end.record()
            end.synchronize()
            lat_ms = start.elapsed_time(end)
            self._metrics._recv_metrics.add(aligned_context_len,
                                            aligned_query_len, seq_recv_len,
                                            lat_ms)

        return seq_recv_len

    def wait_for_save(
        self,
        metadata: AIBrixOffloadingConnectorMetadata,
    ) -> None:
        # Computation is done, safe to kick off async sends.
        assert self.layers_kv_caches is not None, "layers_kv_caches is None"

        if OFFLOADING_CONNECTOR_ENABLE_ASYNC_SEND:
            assert self._send_loop is not None, "send loop is not initialized"
            for seq_request_id, seq_request_meta in metadata.items():
                if (seq_request_meta.state
                        != AIBrixOffloadingConnectorRequestState.
                        WAITING_FOR_SEND):
                    continue
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        self._send_kv_async(seq_request_meta), self._send_loop)
                    seq_request_meta.state = \
                        AIBrixOffloadingConnectorRequestState.SENDING
                    if seq_request_meta.gen_completion:
                        self._sending_futures[seq_request_id] = future
                    logger.debug("Request[id=%s] start send kv",
                                 seq_request_id)
                except Exception as e:
                    if seq_request_meta.gen_completion:
                        future = concurrent.futures.Future()
                        self._sending_futures[seq_request_id] = future
                        future.set_exception(e)
        else:
            for seq_request_id, seq_request_meta in metadata.items():
                if (seq_request_meta.state
                        != AIBrixOffloadingConnectorRequestState.
                        WAITING_FOR_SEND):
                    continue
                self._send_kv_sync_impl(seq_request_meta)

    async def _send_kv_async(
        self,
        seq_request_meta: AIBrixOffloadingConnectorRequestMetadata,
    ) -> None:
        with torch.cuda.stream(self._send_stream):
            ret = self._send_kv_sync_impl(seq_request_meta)
            if self.tp_size > 1:
                self.send_group.barrier()
            return ret

    def _send_kv_sync_impl(
        self,
        seq_request_meta: AIBrixOffloadingConnectorRequestMetadata,
    ) -> None:
        logger.debug("_send_kv_sync_impl: %s", seq_request_meta)
        seq_request_id = seq_request_meta.req_id
        seq_context_len = seq_request_meta.context_len
        seq_slot_mapping = seq_request_meta.seq_slot_mapping.cuda(
            non_blocking=True)
        seq_all_tokens = seq_request_meta.seq_token_ids
        assert seq_all_tokens is not None, "seq_all_tokens is None"
        prompt_len = seq_request_meta.prompt_len
        query_len = seq_request_meta.query_len

        # align to block boundary
        aligned_context_len = round_down(seq_context_len, self.block_ntokens)
        actual_query_len = seq_context_len + query_len - aligned_context_len
        aligned_query_len = round_down(actual_query_len, self.block_ntokens)

        assert prompt_len >= aligned_context_len + aligned_query_len, \
            f"{prompt_len}<{aligned_context_len}+{aligned_query_len}"

        seq_all_tokens_view = TokenListView(seq_all_tokens)
        prefix = seq_all_tokens_view[:aligned_context_len]
        tokens = seq_all_tokens_view[aligned_context_len:aligned_context_len +
                                     aligned_query_len]

        if self._metrics.time_measurement_enabled:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        total_sent = 0
        for (
                chunk_prefix,
                chunk_tokens,
                _,
                all,
        ) in self.cache.cache_chunk_keys(prefix, tokens):
            # |----------- seq_context_len ---------|- offset -|
            # |- aligned_context_len -|- shift_len -|
            #                         |--- n * chunk_size -----|
            # |----------------- chunk_prefix -----------------|- tokens -|
            #                         |-------- (n + 1) * chunk_size -----|
            chunk_size = len(chunk_tokens)
            offset = len(chunk_prefix)
            length = chunk_size

            exists_status = self.cache.exists(chunk_prefix, chunk_tokens)
            if exists_status.is_ok():
                num_existing_tokens = exists_status.value
                logger.info(
                    "Request[id=%s] send(%d) encounters %d existing tokens",
                    seq_request_id, length, num_existing_tokens)
                if chunk_size - num_existing_tokens < self.block_ntokens:
                    continue
                else:
                    # partially exists
                    offset += num_existing_tokens
                    length -= num_existing_tokens
                    new_chunk_prefix_len = len(
                        chunk_prefix) + num_existing_tokens
                    chunk_prefix = all[:new_chunk_prefix_len]
                    chunk_tokens = all[
                        new_chunk_prefix_len:new_chunk_prefix_len + length]

            # allocate space for KV caches
            status = self.cache.allocate_for(chunk_prefix, chunk_tokens)
            if not status.is_ok():
                log_every_n_seconds(logger, logging.ERROR,
                                    "Failed to allocate : %s", 3, str(status))
                break
            handle = status.value
            tensors = handle.to_tensors()
            length = len(tensors) * self.block_ntokens

            chunk_slot_mapping = seq_slot_mapping[offset:offset + length]

            with perf_timer() as get_kernel_offload_dur_ms:
                reshape_and_offload_multi_layer(
                    tensors,
                    self.layers_kv_caches,
                    chunk_slot_mapping,
                    self.block_ntokens,
                    self.kv_cache_dtype,
                    self.k_scales,
                    self.v_scales,
                    self.block_layout.name,
                )

            logger.info("Request[id=%s] offloads %d tokens in %.4f ms",
                        seq_request_id, length, get_kernel_offload_dur_ms())

            # put KV caches to offloading service
            status = self.cache.put(chunk_prefix, chunk_tokens[:length],
                                    handle)
            if not status.is_ok():
                # TODO: notify other ranks in the group to stop sending
                # if this is a fatal error
                log_every_n_seconds(logger, logging.ERROR,
                                    "Failed to put to offloading service: %s",
                                    3, str(status))
                break
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
            end.record()
            end.synchronize()
            lat_ms = start.elapsed_time(end)
            self._metrics._send_metrics.add(aligned_context_len,
                                            aligned_query_len, total_sent,
                                            lat_ms)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str | tuple[str, int]]]]:
        """Returns finished_sending, (finished_recving, num of recv'ed tokens).
        """
        if self._metrics.time_measurement_enabled:
            log_every_n_seconds(self._metrics, logging.INFO, "UNUSED", 10)

        finished_sending = set()
        for req_id, fut in list(self._sending_futures.items()):
            if fut.done():
                try:
                    fut.result()
                    logger.debug("Request %s finished sending", req_id)
                except Exception as e:
                    logger.error("Failed to send kv cache for request %s: %r",
                                 req_id, e)
                finally:
                    finished_sending.add(req_id)
                    del self._sending_futures[req_id]

        finished_recving = set()
        for req_id, fut in list(self._recving_futures.items()):
            if fut.done():
                try:
                    num_recv_tokens = fut.result()
                    finished_recving.add((req_id, num_recv_tokens))
                    logger.debug("Request %s finished recving", req_id)
                except Exception as e:
                    logger.error("Failed to recv kv cache for request %s: %r",
                                 req_id, e)
                    finished_recving.add((req_id, 0))
                finally:
                    del self._recving_futures[req_id]

        return finished_sending or None, finished_recving or None


class AIBrixOffloadingConnector(KVConnectorBase_V1):
    """AIBrixOffloadingConnector is a KVConnector that offloads KV caches
    to the kv cache offloading service.
    """

    def __init__(self, config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=config, role=role)

        self.connector_scheduler: Optional[
            AIBrixOffloadingConnectorScheduler] = None
        if OFFLOADING_CONNECTOR_ENABLE_ASYNC_SEND:
            logger.info("AIBrixOffloadingConnector using async send")
        else:
            logger.info("AIBrixOffloadingConnector using sync send")
        self.connector_worker: Optional[AIBrixOffloadingConnectorWorker] = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = AIBrixOffloadingConnectorScheduler(
                config)
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = AIBrixOffloadingConnectorWorker(config)

    @delegate_to("connector_worker")
    @property
    def metrics(self) -> 'KVTransferMetrics':
        """
        Get the metrics object associated with the connector.
        Returns:
            KVTransferMetrics: The metrics object.
        """
        pass

    @delegate_to("connector_worker")
    def get_metrics_exporter_cls(self) -> 'KVTransferMetricsExporter':
        """
        Get the metrics exporter class associated with the connector.
        """
        pass

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
        self.connector_worker.start_load_kv(self._connector_metadata, )

    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.
        
        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        """Not supported yet"""
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
        """Not supported yet"""
        pass

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
        # Need to invoke start_load_kv() before get_finished() to ensure recving
        # tasks within the metadata have been scheduled.
        self.connector_worker.start_load_kv(self._connector_metadata)
        return self.connector_worker.get_finished(finished_req_ids)

    # ==============================
    # Scheduler-side methods
    # ==============================

    @delegate_to("connector_scheduler")
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
        pass

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
        pass

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
