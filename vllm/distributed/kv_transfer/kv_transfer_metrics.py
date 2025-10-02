# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod


class KVTransferMetrics(ABC):
    """
    KVTransferMetrics is used to collect metrics for KVTransferAgent.
    """

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError


class KVTransferMetricsExporter(ABC):
    """
    KVTransferMetrics is used to collect metrics for KVTransferAgent.
    """

    def __init__(
        self,
        *,
        prefix,
        labelnames,
        gauge_cls,
        counter_cls,
        histogram_cls,
    ):
        self._prefix = prefix
        self._labelnames = labelnames
        self._gauge_cls = gauge_cls
        self._counter_cls = counter_cls
        self._histogram_cls = histogram_cls

    @abstractmethod
    def export(
        self,
        *,
        metrics: KVTransferMetrics,
        labels: dict[str, str],
    ) -> None:
        """Export metrics to external systems."""
        raise NotImplementedError
