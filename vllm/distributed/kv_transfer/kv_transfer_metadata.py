# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from vllm.sequence import SequenceGroupMetadata
from vllm.utils import PyObjectCache

if TYPE_CHECKING:
    from vllm.worker.model_runner import GPUModelRunnerBase


@dataclass
class SequenceGroupToKVTransfer:
    """
    Metadata mainly used by kv_both scenarios.
    """
    prompt_len: Optional[int]
    # Context tokens in prefix cache
    context_tokens: Optional[List[int]]


def seq_group_to_kv_transfer_builder():
    return SequenceGroupToKVTransfer(
        prompt_len=0,
        context_tokens=None,
    )


class KVTransferMetadataCache:
    """Used to cache SequenceGroupToKVTransfer objects between
    scheduler iterations.
    """

    def __init__(self):
        self._seq_group_to_kv_transfer_cache: PyObjectCache = PyObjectCache(
            seq_group_to_kv_transfer_builder)

    def get_cached_seq_group_to_kv_transfer(self):
        obj = self._seq_group_to_kv_transfer_cache.get_object()
        obj.prompt_len = 0
        obj.context_tokens = None
        return obj

    def reset(self):
        self._seq_group_to_kv_transfer_cache.reset()


class KVTransferMetadata:
    """Metadata for input sequences. Used in KV transfer.

    Args:
        seq_groups: List of batched sequence groups.
        model_input_builder: Model input builder for rebuilding model input.
    """

    def __init__(
        self,
        seq_groups: List[SequenceGroupToKVTransfer],
    ) -> None:
        self.seq_groups = seq_groups
        # only driver has seq_group_metadata_list and runner
        self.seq_group_metadata_list: List[SequenceGroupMetadata] = None
        self.runner: GPUModelRunnerBase = None

    @staticmethod
    def prepare(
        seq_group_metadata_list: List[SequenceGroupMetadata],
        cache: Optional[KVTransferMetadataCache] = None,
    ) -> "KVTransferMetadata":
        """
        context_lens include num of tokens in prefix cache.
        """
        seq_groups: List[SequenceGroupToKVTransfer] = []
        ctx_idx = 0
        for seq_group_metadata in seq_group_metadata_list:
            seq_ids = list(seq_group_metadata.seq_data.keys())
            for i in range(len(seq_ids)):
                if cache is not None:
                    seq_group_obj = cache.get_cached_seq_group_to_kv_transfer()
                else:
                    seq_group_obj = seq_group_to_kv_transfer_builder()

                seq_id = seq_ids[i]
                seq_data = seq_group_metadata.seq_data[seq_id]

                seq_group_obj.prompt_len = seq_data.get_prompt_len()

                seq_groups.append(seq_group_obj)

            # prefill sequence group only has one sequence
            if seq_group_metadata.is_prompt:
                seq_id = seq_ids[0]
                nblocks = len(seq_group_metadata.computed_block_nums or [])
                seq_data = seq_group_metadata.seq_data[seq_id]
                context_len = seq_data.get_num_cached_tokens()
                # prefix caching:
                #     context_len > 0 and nblocks > 0
                # chunked prefill intermediate steps:
                #     context_len > 0 and nblocks == 0
                #
                # We only carry context_tokens for prefix caching since
                # chunked prefill intermediate steps can use the cached
                # context tokens in the offloading connector.
                if context_len > 0 and nblocks > 0:
                    seq_groups[-1].context_tokens = seq_data.get_token_ids(
                    )[:context_len]

            ctx_idx += len(seq_ids)

        if cache is not None:
            cache.reset()

        metadata = KVTransferMetadata(seq_groups=seq_groups)
        return metadata

    def __repr__(self) -> str:
        return f"KVTransferMetadata(seq_groups={self.seq_groups})"
