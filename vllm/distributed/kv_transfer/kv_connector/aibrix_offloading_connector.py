# SPDX-License-Identifier: Apache-2.0
import copy
import dataclasses
import enum
import itertools
import logging
import time
from typing import TYPE_CHECKING, Optional, Union

import torch
from aibrix_kvcache import (BaseKVCacheManager, GroupAwareKVCacheManager,
                            KVCacheBlockLayout, KVCacheBlockSpec,
                            KVCacheConfig, KVCacheMetrics, KVCacheTensorSpec,
                            ModelSpec, TokenListView)
from aibrix_kvcache._custom_ops import (reshape_and_cache_multi_layer,
                                        reshape_and_offload_multi_layer)
from aibrix_kvcache.common.absl_logging import (getLogger, log_every_n_seconds,
                                                log_if)
from aibrix_kvcache.metrics import (MS_BUCKETS, TOKEN_BUCKETS,
                                    BaseMetricsExporter,
                                    KVCacheMetricsExporter, Metrics)
from aibrix_kvcache.profiling import tag_wrapper
from aibrix_kvcache.utils import perf_timer

from vllm.attention import get_attn_backend
from vllm.attention.backends.flash_attn import FlashAttentionBackend
# from vllm.attention.backends.flashinfer import FlashInferBackend
from vllm.attention.backends.xformers import XFormersBackend
from vllm.distributed import broadcast_tensor_dict, get_tp_group
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.distributed.kv_transfer.kv_transfer_metrics import (
    KVTransferMetrics, KVTransferMetricsExporter)
from vllm.model_executor import SamplingMetadata
from vllm.utils import get_kv_cache_torch_dtype, round_down

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.sequence import IntermediateTensors
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = getLogger(__name__)

OFFLOADING_CONNECTOR_SKIP_THRESHOLD = 8
OFFLOADING_CONNECTOR_SUPPORTED_ATTN_BACKENDS = {
    FlashAttentionBackend.get_name():
    KVCacheBlockLayout.LCND,
    # TODO: support FlashInferBackend.
    # FlashInferBackend.get_name(): KVCacheBlockLayout.LCND,
    XFormersBackend.get_name():
    KVCacheBlockLayout.LCND,
}


class AIBrixOffloadingConnectorComputeMetrics(Metrics):
    """Compute metrics."""

    num_tokens: list[int] = []
    op_lat_ms: Optional[list[int]] = None

    def __init__(
        self,
        enable_time_measurement: bool = True,
    ) -> None:
        self._enable_time_measurement = enable_time_measurement
        self._init_optionals()

    def _init_optionals(self) -> None:
        if self._enable_time_measurement:
            self.op_lat_ms = []

    def add(
        self,
        num_tokens: int,
        lat_ms: int,
    ) -> None:
        self.num_tokens.append(num_tokens)
        if self._enable_time_measurement:
            self.op_lat_ms.append(lat_ms)

    def reset(self) -> None:
        self.num_tokens = []
        self._init_optionals()

    def summary(self) -> str:
        iter_len = len(self.num_tokens)
        total_tokens = sum(self.num_tokens)
        avg_tokens = total_tokens / iter_len if iter_len > 0 else 0
        summary = f"COMPUTE: Num. of tokens (iter): " \
                  f"total={total_tokens}, avg={avg_tokens:.2f}"
        if self._enable_time_measurement:
            total_lat_ms = sum(self.op_lat_ms)
            avg_lat_ms = total_lat_ms / iter_len if iter_len > 0 else 0
            summary += f", Latency (iter, ms): total={total_lat_ms:.2f}, " \
                       f"avg={avg_lat_ms:.2f}"
        return summary


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
    # tracks the latency of rebuilding model_input
    rebuild_lat_ms: Optional[list[int]] = None

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
            self.rebuild_lat_ms = []

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

    def record_rebuild_latency(self, lat_ms: int) -> None:
        if self._enable_time_measurement:
            self.rebuild_lat_ms.append(lat_ms)

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
            if len(self.rebuild_lat_ms) > 0:
                iter_len = len(self.rebuild_lat_ms)
                total_rebuild_lat_ms = sum(self.rebuild_lat_ms)
                avg_rebuild_lat_ms = total_rebuild_lat_ms / iter_len \
                    if iter_len > 0 else 0
                summary += f", model_input rebuild latency (iter, ms): " \
                           f"total={total_rebuild_lat_ms:.2f}, " \
                           f"avg={avg_rebuild_lat_ms:.2f}"
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
        self.histogram_iteration_rebuild_lat_ms = self._histogram_cls(
            name=f"{self._prefix}iteration_rebuild_lat_ms",
            documentation=("Histogram of rebuilding model_input latencies "
                           "per iteration in ms."),
            labelnames=self._labelnames,
            buckets=MS_BUCKETS)

    def export(
        self,
        labels: dict[str, str],
        metrics: AIBrixOffloadingConnectorOpMetrics
        | AIBrixOffloadingConnectorComputeMetrics,
    ) -> None:
        labels = labels.copy()

        if isinstance(metrics, AIBrixOffloadingConnectorOpMetrics):
            labels[self.OP_TYPE_LABELNAME] = metrics._op.name.lower()
            self._export_op_metrics(labels, metrics)
        else:
            labels[self.OP_TYPE_LABELNAME] = "compute"
            self._export_compute_metrics(labels, metrics)

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
            self._export_histogram(self.histogram_iteration_rebuild_lat_ms,
                                   labels, metrics.rebuild_lat_ms)

    def _export_compute_metrics(
        self,
        labels: dict[str, str],
        metrics: AIBrixOffloadingConnectorComputeMetrics,
    ) -> None:
        self._export_histogram(self.histogram_iteration_tokens, labels,
                               metrics.num_tokens)
        if metrics._enable_time_measurement:
            self._export_histogram(self.histogram_iteration_op_lat_ms, labels,
                                   metrics.op_lat_ms)


class AIBrixOffloadingConnectorMetrics(KVTransferMetrics):

    def __init__(self, metrics: KVCacheMetrics) -> None:
        self._cache_metrics = metrics
        self._time_measurement_enabled = (
            self._cache_metrics.time_measurement_enabled)
        self._compute_metrics = AIBrixOffloadingConnectorComputeMetrics(
            enable_time_measurement=self._time_measurement_enabled)
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
        self._compute_metrics.reset()
        self._send_metrics.reset()
        self._recv_metrics.reset()

    def __str__(self) -> str:
        return f"AIBrixOffloadingConnector metrics: " \
               f"{self._compute_metrics.summary()}" \
               f"\n\t{self._send_metrics.summary()}" \
               f"\n\t{self._recv_metrics.summary()}" \
               f"\n\t{self._cache_metrics.summary()}"


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
            metrics=metrics._compute_metrics,
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


@dataclasses.dataclass
class AIBrixOffloadingConnectorCachedMeta:
    context_tokens: list[int] = dataclasses.field(default_factory=list)
    context_tokens_offset: int = 0
    context_tokens_view: Optional[TokenListView] = None

    # When chunked prefill is enabled, the query length in each iteration may
    # not be divisible by block_size. Consider the following example, assuming
    # recv always returns nothing for simplicity. Suppose query_len = 511:
    #
    # - In the first iteration, since no tokens are received, we need to send
    #   all 511 tokens. However, in AIBrixOffloadingConnector, the length is
    #   aligned to 496, and the remaining 15 tokens are ignored.
    # - In the second iteration, query_len is likely 511 again, while
    #   context_len is 511. To prevent a gap in the storage layer caused by
    #   skipping 15 tokens in the first iteration, AIBrixOffloadingConnector
    #   shifts context_len to 496 and attempts to send (15 + 511) tokens.
    #
    # To handle this correctly, we need to determine the slot mapping for those
    # 15 tokens. The actual number varies from 1 to 15 but is always less than
    # block_size. Therefore, we store the slot mapping of the last block in the
    # metadata.
    last_block_slot_mapping: Optional[torch.Tensor] = None

    def __init__(self, seq_len: int) -> None:
        self.context_tokens = [-1] * seq_len

    def get_context_tokens(self) -> list[int]:
        return self.context_tokens[:self.context_tokens_offset]

    def get_context_tokens_view(self) -> TokenListView:
        if self.context_tokens_view is None:
            self.context_tokens_view = TokenListView(self.get_context_tokens())
        return self.context_tokens_view

    def extend_context_tokens(self, tokens: list[int]) -> None:
        offset = self.context_tokens_offset
        length = len(tokens)
        self.context_tokens[offset:offset + length] = tokens
        self.context_tokens_offset += length
        self.context_tokens_view = None

    def clear_context_tokens(self) -> None:
        self.context_tokens.clear()
        self.context_tokens_offset = 0
        self.context_tokens_view = None


class AIBrixOffloadingConnector(KVConnectorBase):
    """AIBrixOffloadingConnector is a KVConnector that offloads KV caches
    and hidden states to the kv cache offloading service.
    """

    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: "VllmConfig",
    ):
        cache_config = config.cache_config
        model_config = config.model_config
        parallel_config = config.parallel_config

        tp_size = parallel_config.tensor_parallel_size
        num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        hidden_size = model_config.get_hidden_size()
        num_attention_heads = model_config.get_num_attention_heads(
            parallel_config) * tp_size
        head_size = int(hidden_size / num_attention_heads)
        tp_rank = get_tp_group().rank_in_group

        kv_head_ids = list(
            range(num_kv_heads * tp_rank, num_kv_heads * (tp_rank + 1)))
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

        config = KVCacheConfig(block_spec=block_spec,
                               model_spec=ModelSpec(
                                   model_config.max_model_len))

        if parallel_config.tensor_parallel_size == 1:
            self.cache = BaseKVCacheManager(config=config)
        else:
            pg = get_tp_group().cpu_group
            self.cache = GroupAwareKVCacheManager(config=config,
                                                  process_group=pg)

        self.rank = tp_rank
        self.head_size = head_size
        self.tp_size = tp_size
        self.num_kv_heads = num_kv_heads
        self.num_layers = num_layers
        self.kv_head_ids = kv_head_ids
        self.layer_ids = layer_ids
        self.engine_block_ntokens = block_ntokens
        self.cache_block_ntokens = self.cache.block_size
        self.block_dtype = block_dtype
        self.block_shape = block_spec.block_shape
        self.block_spec = block_spec
        self.block_layout = block_spec.block_layout
        self.chunk_size = self.cache.chunk_size
        self.cache_feature = self.cache.feature
        self.kv_cache_dtype = kv_cache_dtype

        self._connector_cache: dict[str,
                                    AIBrixOffloadingConnectorCachedMeta] = {}
        self._metrics = AIBrixOffloadingConnectorMetrics(self.cache.metrics)
        # meta to track compute perf
        self._compute_start_event = torch.cuda.Event(enable_timing=True)
        self._compute_end_event = torch.cuda.Event(enable_timing=True)
        self._compute_total_tokens = 0

        self.k_scales: Optional[list[torch.Tensor]] = None
        self.v_scales: Optional[list[torch.Tensor]] = None

        logger.info(
            "AIBrixOffloadingConnector is initialized, "
            "engine_block_ntokens=%d, cache_block_ntokens=%d",
            self.engine_block_ntokens,
            self.cache_block_ntokens,
        )

    @property
    def metrics(self) -> KVTransferMetrics:
        return self._metrics

    def get_metrics_exporter_cls(self):
        return AIBrixOffloadingConnectorMetricsExporter

    def close(self) -> None:
        if self.cache:
            self.cache.close()
            self.cache = None

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

    def _prepare_request_cache(
        self,
        model_input: "ModelInputForGPUWithSamplingMetadata",
    ) -> None:
        # remove finished requests
        for req_id in model_input.finished_requests_ids:
            if req_id in self._connector_cache:
                self._connector_cache.pop(req_id)

        # remove decode requests
        seq_lens = model_input.seq_lens
        num_prefills = model_input.attn_metadata.num_prefills
        request_ids = list(model_input.request_ids_to_seq_ids.keys())
        for seq_idx, _ in enumerate(seq_lens):
            if seq_idx >= num_prefills:
                seq_request_id = request_ids[seq_idx]
                if seq_request_id in self._connector_cache:
                    self._connector_cache.pop(seq_request_id)

        # add new requests
        for req_id in model_input.request_ids_to_seq_ids:
            if req_id not in self._connector_cache:
                self._connector_cache[req_id] = None

    def _update_request_cache(
        self,
        seq_request_id: str,
        seq_len: int,
        seq_input_tokens: list[int],
        seq_slot_mapping: torch.Tensor,
        kv_transfer_context_tokens: list[int],
    ) -> None:
        if self._connector_cache[seq_request_id] is None:
            self._connector_cache[
                seq_request_id] = AIBrixOffloadingConnectorCachedMeta(seq_len)
        seq_request_cache = self._connector_cache[seq_request_id]
        kv_transfer_context_tokens = kv_transfer_context_tokens or []
        if seq_len == len(kv_transfer_context_tokens):
            # when using prefix caching, if all tokens are cached,
            # we will leave a token in `seq_input_tokens` to avoid
            # erroneous behavior. In this case, we need to ignore
            # the token in `seq_input_tokens`.
            # See `_compute_for_prefix_cache_hit` for more details.
            seq_input_tokens = []
        if seq_len != len(seq_request_cache.get_context_tokens()) + len(
                kv_transfer_context_tokens) + len(seq_input_tokens):
            # this is a preempted request to be recomputed, let's drop
            # the context tokens
            seq_request_cache.clear_context_tokens()
        if len(kv_transfer_context_tokens) > 0:
            assert seq_request_cache.context_tokens_offset == 0
            seq_request_cache.extend_context_tokens(kv_transfer_context_tokens)
        seq_request_cache.extend_context_tokens(seq_input_tokens)
        last_block_len = min(self.cache_block_ntokens,
                             seq_slot_mapping.shape[0])
        seq_request_cache.last_block_slot_mapping = seq_slot_mapping[
            -last_block_len:]

    def _get_chunk_slot_mapping(
        self,
        seq_request_id: str,
        seq_slot_mapping: torch.Tensor,
        offset: int,
        length: int,
    ) -> torch.Tensor:
        if offset < 0:
            chunk_slot_mapping = seq_slot_mapping[:offset + length]
            # attach more to slot mapping
            seq_last_block_slot_mapping = self._connector_cache[
                seq_request_id].last_block_slot_mapping
            assert seq_last_block_slot_mapping is not None
            assert -offset <= seq_last_block_slot_mapping.shape[0]
            prepend_slot_mapping = seq_last_block_slot_mapping[offset:]
            chunk_slot_mapping = torch.cat(
                (prepend_slot_mapping, chunk_slot_mapping))
        else:
            chunk_slot_mapping = seq_slot_mapping[offset:offset + length]

        assert chunk_slot_mapping.shape[0] == length
        return chunk_slot_mapping

    def _ensure_kv_scales(self, model_executable: torch.nn.Module) -> None:
        if self.k_scales is None:
            start_layer = model_executable.model.start_layer
            end_layer = model_executable.model.end_layer
            layers = model_executable.model.layers[start_layer:end_layer]
            self.k_scales = [layer.self_attn.attn._k_scale for layer in layers]
            self.v_scales = [layer.self_attn.attn._v_scale for layer in layers]

    @tag_wrapper({
        "connector": "AIBrixOffloadingConnector",
        "func": "send_kv_caches_and_hidden_states"
    })
    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: list[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             "IntermediateTensors"],
    ) -> None:
        attn_metadata = model_input.attn_metadata
        kv_transfer_metadata = model_input.kv_transfer_metadata

        if kv_transfer_metadata is None:
            return

        assert attn_metadata is not None

        self._prepare_request_cache(model_input)

        # only send prompt KV caches
        if attn_metadata.num_prefills <= 0:
            return

        self._ensure_kv_scales(model_executable)
        num_prefills = attn_metadata.num_prefills
        seq_lens = model_input.seq_lens[:num_prefills]
        query_lens = model_input.query_lens[:num_prefills]
        slot_mapping = model_input.attn_metadata.slot_mapping.flatten()
        start_layer = model_executable.model.start_layer
        end_layer = model_executable.model.end_layer
        request_ids = list(model_input.request_ids_to_seq_ids.keys())

        if self._metrics.time_measurement_enabled:
            self._compute_end_event.record()
            self._compute_end_event.synchronize()
            compute_lat_ms = self._compute_start_event.elapsed_time(
                self._compute_end_event)
            self._metrics._compute_metrics.add(self._compute_total,
                                               compute_lat_ms)

        # query_lens contains new KV caches that need to be offloaded
        for seq_idx, query_len in enumerate(query_lens):
            start_pos = sum(query_lens[:seq_idx])
            end_pos = start_pos + query_len
            seq_request_id = request_ids[seq_idx]
            seq_context_len = seq_lens[seq_idx] - query_len
            seq_slot_mapping = slot_mapping[start_pos:end_pos]
            seq_cached_meta = self._connector_cache[seq_request_id]
            assert seq_cached_meta is not None
            seq_all_tokens = seq_cached_meta.get_context_tokens_view()
            assert seq_all_tokens is not None
            assert len(seq_all_tokens) == seq_lens[
                seq_idx], f"{len(seq_all_tokens)}!={seq_lens[seq_idx]}"
            prompt_len = kv_transfer_metadata.seq_groups[seq_idx].prompt_len

            # align to block boundary
            aligned_context_len = round_down(seq_context_len,
                                             self.cache_block_ntokens)
            actual_query_len = seq_context_len + query_len - aligned_context_len
            aligned_query_len = round_down(actual_query_len,
                                           self.cache_block_ntokens)

            # skip if there are not enough tokens to send after alignment
            if prompt_len == seq_lens[seq_idx]:
                # If chunked prefill is not enabled or this is the last
                # chunk, we use a larger skip threshold
                skip_threshold = OFFLOADING_CONNECTOR_SKIP_THRESHOLD
            else:
                # This is an intermediate chunk, only skip if this is not
                # a full block
                skip_threshold = 1
            if aligned_query_len <= skip_threshold * self.engine_block_ntokens:
                continue

            assert len(
                seq_all_tokens) >= aligned_context_len + aligned_query_len

            prefix = seq_all_tokens[:aligned_context_len]
            tokens = seq_all_tokens[aligned_context_len:aligned_context_len +
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
                offset = len(chunk_prefix) - seq_context_len
                length = chunk_size

                exists_status = self.cache.exists(chunk_prefix, chunk_tokens)
                if exists_status.is_ok():
                    num_existing_tokens = exists_status.value
                    logger.info(
                        "Request[id=%s] send(%d) encounters %d existing tokens",
                        seq_request_id, length, num_existing_tokens)
                    if chunk_size - num_existing_tokens < \
                        self.cache_block_ntokens:
                        continue
                    else:
                        # partially exists
                        offset += num_existing_tokens
                        length -= num_existing_tokens
                        new_chunk_prefix_len = (len(chunk_prefix) +
                                                num_existing_tokens)
                        chunk_prefix = all[:new_chunk_prefix_len]
                        chunk_tokens = all[
                            new_chunk_prefix_len:new_chunk_prefix_len + length]

                # allocate space for KV caches
                status = self.cache.allocate_for(chunk_prefix, chunk_tokens)
                if not status.is_ok():
                    log_every_n_seconds(logger, logging.ERROR,
                                        "Failed to allocate : %s", 3,
                                        str(status))
                    break
                handle = status.value
                tensors = handle.to_tensors()
                length = len(tensors) * self.cache_block_ntokens

                chunk_slot_mapping = self._get_chunk_slot_mapping(
                    seq_request_id, seq_slot_mapping, offset, length)

                with perf_timer() as get_kernel_offload_dur_ms:
                    reshape_and_offload_multi_layer(
                        tensors,
                        kv_caches[start_layer:end_layer],
                        chunk_slot_mapping,
                        self.engine_block_ntokens,
                        self.kv_cache_dtype,
                        self.k_scales,
                        self.v_scales,
                        self.block_layout.name,
                    )

                logger.info("Request[id=%s] offloads %d tokens in %.4f ms",
                            seq_request_id, length,
                            get_kernel_offload_dur_ms())

                # put KV caches to offloading service
                status = self.cache.put(chunk_prefix, chunk_tokens[:length],
                                        handle)
                if not status.is_ok():
                    # TODO: notify other ranks in the group to stop sending
                    # if this is a fatal error
                    log_every_n_seconds(
                        logger, logging.ERROR,
                        "Failed to put to offloading service: %s", 3,
                        str(status))
                    break

                put_ntokens = status.get()
                total_sent += put_ntokens
                if put_ntokens != length:
                    break

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

    @tag_wrapper({
        "connector": "AIBrixOffloadingConnector",
        "func": "recv_kv_caches_and_hidden_states"
    })
    def recv_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: list[torch.Tensor],
    ) -> tuple[
            Union[torch.Tensor, "IntermediateTensors"],
            bool,
            "ModelInputForGPUWithSamplingMetadata",
    ]:
        attn_metadata = model_input.attn_metadata
        kv_transfer_metadata = model_input.kv_transfer_metadata

        hidden_or_intermediate_states = None

        # TODO:
        # When using KV cache offloading with chunk-prefill enabled, for
        # the first `n` chunks (except the last chunk before the decode
        # phase), we can bypass the model execution if there is a full
        # hit in the KV cache.
        bypass_model_exec = False

        if kv_transfer_metadata is None:
            return hidden_or_intermediate_states, bypass_model_exec, model_input

        assert attn_metadata is not None

        self._compute_total = 0
        self._prepare_request_cache(model_input)

        # only recv prompt KV caches
        if attn_metadata.num_prefills <= 0:
            return hidden_or_intermediate_states, bypass_model_exec, model_input

        self._ensure_kv_scales(model_executable)
        num_prefills = attn_metadata.num_prefills
        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.seq_lens[:num_prefills]
        query_lens = model_input.query_lens[:num_prefills]
        slot_mapping = model_input.attn_metadata.slot_mapping.flatten()
        start_layer = model_executable.model.start_layer
        end_layer = model_executable.model.end_layer
        request_ids = list(model_input.request_ids_to_seq_ids.keys())

        if kv_transfer_metadata.seq_group_metadata_list is not None:
            assert len(request_ids) == len(
                kv_transfer_metadata.seq_group_metadata_list)

        model_config = model_executable.model.config
        num_kv_heads = int(model_config.num_key_value_heads / self.tp_size)
        hidden_size = model_config.hidden_size
        num_attention_heads = model_config.num_attention_heads
        head_size = int(hidden_size / num_attention_heads)

        assert num_kv_heads == self.num_kv_heads
        assert head_size == self.head_size

        reused_lens = []
        rebuild_model_input = False
        # query_lens contains new KV caches to be received
        for seq_idx, query_len in enumerate(query_lens):
            start_pos = sum(query_lens[:seq_idx])
            end_pos = start_pos + query_len
            seq_request_id = request_ids[seq_idx]
            seq_context_len = seq_lens[seq_idx] - query_len
            seq_input_tokens = (
                input_tokens_tensor[start_pos:end_pos].cpu().tolist())
            prompt_len = kv_transfer_metadata.seq_groups[seq_idx].prompt_len

            seq_slot_mapping = slot_mapping[start_pos:end_pos]
            # will update these lens later if needed
            reused_lens.append(0)

            # align to block boundary
            aligned_context_len = round_down(seq_context_len,
                                             self.cache_block_ntokens)
            actual_query_len = seq_context_len + query_len - aligned_context_len
            aligned_query_len = round_down(actual_query_len,
                                           self.cache_block_ntokens)
            shift_len = seq_context_len - aligned_context_len

            self._update_request_cache(
                seq_request_id,
                seq_lens[seq_idx],
                seq_input_tokens,
                seq_slot_mapping,
                kv_transfer_metadata.seq_groups[seq_idx].context_tokens,
            )

            # skip if there are not enough tokens to receive after alignment
            if prompt_len == seq_lens[seq_idx]:
                # If chunked prefill is not enabled or this is the last
                # chunk, we use a larger skip threshold
                skip_threshold = OFFLOADING_CONNECTOR_SKIP_THRESHOLD
            else:
                # This is an intermediate chunk, only skip if this is not
                # a full block
                skip_threshold = 1
            if aligned_query_len <= skip_threshold * self.engine_block_ntokens:
                continue

            seq_cached_meta = self._connector_cache[seq_request_id]
            assert seq_cached_meta is not None
            seq_all_tokens = seq_cached_meta.get_context_tokens_view()
            assert seq_all_tokens is not None
            assert len(seq_all_tokens) == seq_lens[
                seq_idx], f"{len(seq_all_tokens)}!={seq_lens[seq_idx]}"

            assert len(
                seq_all_tokens) >= aligned_context_len + aligned_query_len

            prefix = seq_all_tokens[:aligned_context_len]
            tokens = seq_all_tokens[aligned_context_len:aligned_context_len +
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
                    self.cache.prefetch(chunk_prefix + chunk_tokens,
                                        next_tokens)

                # get KV caches from offloading service
                status = self.cache.acquire(chunk_prefix, chunk_tokens)

                if not status.is_ok():
                    if not status.is_not_found():
                        log_every_n_seconds(
                            logger, logging.ERROR,
                            "Failed to get from offloading service: %s", 3,
                            str(status))
                    break

                num_fetched_tokens, handle = status.value
                kv_blocks = handle.to_tensors()

                offset = len(chunk_prefix) - seq_context_len
                length = num_fetched_tokens

                chunk_slot_mapping = self._get_chunk_slot_mapping(
                    seq_request_id, seq_slot_mapping, offset, length)

                with perf_timer() as get_kernel_onload_dur_ms:
                    reshape_and_cache_multi_layer(
                        kv_blocks,
                        kv_caches[start_layer:end_layer],
                        chunk_slot_mapping,
                        self.engine_block_ntokens,
                        self.kv_cache_dtype,
                        self.k_scales,
                        self.v_scales,
                        self.block_layout.name,
                    )

                logger.info("Request[id=%s] onloads %d tokens in %.4f ms",
                            seq_request_id, length, get_kernel_onload_dur_ms())

                # update recv_len
                seq_recv_len += num_fetched_tokens - shift_len
                rebuild_model_input = True
                # reset shift_len
                shift_len = 0

                # release handle
                handle.release()

                if num_fetched_tokens < len(chunk_tokens):
                    # didn't receive all tokens for current chunk, break
                    break

            reused_lens[-1] = seq_recv_len
            log_if(
                logger,
                logging.INFO,
                ("Request[id=%s, prompt_len=%d, context_len=%d] "
                 "reused %d tokens"),
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
                                                aligned_query_len,
                                                seq_recv_len, lat_ms)

        if rebuild_model_input:
            if self._metrics.time_measurement_enabled:
                start = time.perf_counter()
            model_input = self._rebuild_model_input(model_input, reused_lens)
            if self._metrics.time_measurement_enabled:
                end = time.perf_counter()
                lat_ms = (end - start) * 1000
                self._metrics._recv_metrics.record_rebuild_latency(lat_ms)

        if self._metrics.time_measurement_enabled:
            self._compute_start_event.record()
            self._compute_total = sum(query_lens) - sum(reused_lens)

        return hidden_or_intermediate_states, bypass_model_exec, model_input

    def _rebuild_model_input(
        self,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        reused_lens: list[int],
    ) -> "ModelInputForGPUWithSamplingMetadata":
        seq_group_metadata_list = (
            model_input.kv_transfer_metadata.seq_group_metadata_list)
        if seq_group_metadata_list:
            new_model_input = self._driver_rebuild_model_input(
                model_input, reused_lens)
            if self.tp_size > 1:
                broadcast_data = new_model_input.as_broadcastable_tensor_dict()
                broadcast_tensor_dict(broadcast_data, src=0)
        else:
            # worker
            runner = model_input.kv_transfer_metadata.runner
            broadcast_data = broadcast_tensor_dict(src=0)
            new_model_input = (
                runner.make_model_input_from_broadcasted_tensor_dict(
                    broadcast_data))

        # replace fields
        return dataclasses.replace(
            model_input,
            input_tokens=new_model_input.input_tokens,
            input_positions=new_model_input.input_positions,
            token_types=new_model_input.token_types,
            attn_metadata=new_model_input.attn_metadata,
            seq_lens=new_model_input.seq_lens,
            query_lens=new_model_input.query_lens,
            lora_mapping=new_model_input.lora_mapping,
            lora_requests=new_model_input.lora_requests,
            multi_modal_kwargs=new_model_input.multi_modal_kwargs,
            prompt_adapter_mapping=new_model_input.prompt_adapter_mapping,
            prompt_adapter_requests=new_model_input.prompt_adapter_requests,
            sampling_metadata=new_model_input.sampling_metadata,
        )

    def _driver_rebuild_model_input(
        self,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        reused_lens: list[int],
    ) -> "ModelInputForGPUWithSamplingMetadata":
        seq_group_metadata_list = (
            model_input.kv_transfer_metadata.seq_group_metadata_list)
        runner = model_input.kv_transfer_metadata.runner
        finished_requests_ids = model_input.finished_requests_ids

        request_ids = list(model_input.request_ids_to_seq_ids.keys())
        assert len(request_ids) == len(seq_group_metadata_list)

        seq_lens = model_input.seq_lens
        query_lens = model_input.query_lens

        backup = [None] * len(seq_group_metadata_list)
        for offset, seq_group_metadata in enumerate(seq_group_metadata_list):
            if not seq_group_metadata.is_prompt:
                break
            if reused_lens[offset] == 0:
                continue
            # Prefill has only 1 sequence
            backup[offset] = seq_group_metadata.computed_block_nums
            context_len = seq_lens[offset] - query_lens[offset]
            context_len += reused_lens[offset]
            num_blocks = context_len // self.engine_block_ntokens
            seq_group_metadata.computed_block_nums = copy.deepcopy(
                seq_group_metadata.computed_block_nums)
            seq_group_metadata.computed_block_nums.extend(
                itertools.repeat(0, num_blocks))

        new_model_input = runner._prepare_model_input_tensors(
            seq_group_metadata_list, finished_requests_ids)

        if model_input.sampling_metadata is not None:
            generators = runner.get_generators(finished_requests_ids)
            sampling_metadata = SamplingMetadata.prepare(
                seq_group_metadata_list,
                new_model_input.seq_lens,
                new_model_input.query_lens,
                runner.device,
                runner.pin_memory,
                generators,
                runner.sampling_metadata_cache,
            )
            new_model_input = dataclasses.replace(
                new_model_input, sampling_metadata=sampling_metadata)

        # revert changes on seq_group_metadata_list
        for offset, seq_group_metadata in enumerate(seq_group_metadata_list):
            if not seq_group_metadata.is_prompt:
                break
            if reused_lens[offset] == 0:
                continue
            seq_group_metadata.computed_block_nums = backup[offset]

        return new_model_input
