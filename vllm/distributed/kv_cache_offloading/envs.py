# SPDX-License-Identifier: Apache-2.0
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Tuple

if TYPE_CHECKING:
    # (minimum number of blocks, ratio)
    #
    # If both L1 and L2 caches are enabled, we will issue a second get request
    # (i.e., double get) to L2 cache to fetch the missing cache blocks if:
    # 1. The number of missing blocks is greater than or equal to
    #    VLLM_KV_CACHE_OL_DOUBLE_GET_THRESHOLD[0]
    # 2. The ratio of missing blocks to the total number of blocks is greater
    #    than or equal to VLLM_KV_CACHE_OL_DOUBLE_GET_THRESHOLD[1]
    # Otherwise, we will not issue a second get request to L2 cache.
    #
    # Note that the second rule is only applicable if the ratio threshold is
    # set, otherwise we only consider the first rule. For example, if
    # VLLM_KV_CACHE_OL_DOUBLE_GET_THRESHOLD is set to "4", we will ignore the
    # second rule.
    VLLM_KV_CACHE_OL_DOUBLE_GET_THRESHOLD: Tuple[int, float] = (4, 0.1)
    VLLM_KV_CACHE_OL_CHUNK_SIZE: int = 512
    VLLM_KV_CACHE_OL_METRICS_ENABLED: bool = True
    VLLM_KV_CACHE_OL_TIME_MEASUREMENT_ENABLED: bool = True
    VLLM_KV_CACHE_OL_BREAKDOWN_MEASUREMENT_ENABLED: bool = True

    VLLM_KV_CACHE_OL_L1_CACHE_ENABLED: bool = True
    VLLM_KV_CACHE_OL_L1_CACHE_EVICTION_POLICY: str = "S3FIFO"
    VLLM_KV_CACHE_OL_L1_CACHE_CAPACITY_GB: float = 10
    VLLM_KV_CACHE_OL_DEVICE: str = "cpu"
    VLLM_KV_CACHE_OL_L1_CACHE_EVICT_SIZE: int = 16

    # S3FIFO Env Vars
    VLLM_KV_CACHE_OL_S3FIFO_SMALL_TO_MAIN_PROMO_THRESHOLD: int = 1
    VLLM_KV_CACHE_OL_S3FIFO_SMALL_FIFO_CAPACITY_RATIO: float = 0.3

    VLLM_KV_CACHE_OL_L2_CACHE_BACKEND: str = ""
    VLLM_KV_CACHE_OL_L2_CACHE_NAMESPACE: str = "dummy"
    VLLM_KV_CACHE_OL_L2_CACHE_COMPRESSION: str = ""
    VLLM_KV_CACHE_OL_L2_CACHE_OP_BATCH: int = 32
    VLLM_KV_CACHE_OL_L2_CACHE_PER_TOKEN_TIMEOUT_MS: int = 20

    # Ingestion type, only applicable if L1 cache is enabled. Defaults to "HOT".
    # "ALL": all newly put data in L1 cache will be put onto L2 cache.
    # "HOT": hot data in L1 cache will be put onto L2 cache.
    # "EVICTED": evicted data in L1 cache will be put onto L2 cache.
    VLLM_KV_CACHE_OL_L2_CACHE_INGESTION_TYPE: str = "HOT"
    # Max number of inflight writes to L2 cache in terms of tokens.
    # Defaults to 2048.
    # If the number of inflight writes reaches the limit, new writes
    # will be discarded. Set it to zero to use synchronous writes.
    VLLM_KV_CACHE_OL_L2_CACHE_INGESTION_MAX_INFLIGHT_TOKENS: int = 2048

    VLLM_KV_CACHE_OL_L2_CACHE_NUM_ASYNC_WORKERS: int = 8

    # Mock Connector
    VLLM_KV_CACHE_OL_MOCK_USE_RDMA: bool = False
    VLLM_KV_CACHE_OL_MOCK_USE_GATHER_SCATTER: bool = False

    # RocksDB Env Vars
    VLLM_KV_CACHE_OL_ROCKSDB_ROOT: str = os.path.expanduser(
        os.path.join(os.path.expanduser("~"), ".kv_cache_ol", "rocksdb"))
    VLLM_KV_CACHE_OL_ROCKSDB_TTL_S: int = 600
    VLLM_KV_CACHE_OL_ROCKSDB_WRITE_BUFFER_SIZE: int = 64 * 1024 * 1024
    VLLM_KV_CACHE_OL_ROCKSDB_TARGET_FILE_SIZE_BASE: int = 64 * 1024 * 1024
    VLLM_KV_CACHE_OL_ROCKSDB_MAX_WRITE_BUFFER_NUMBER: int = 3
    VLLM_KV_CACHE_OL_ROCKSDB_MAX_TOTAL_WAL_SIZE: int = 128 * 1024 * 1024
    VLLM_KV_CACHE_OL_ROCKSDB_MAX_BACKGROUND_JOBS: int = 8

    # InfiniStore Env Vars
    VLLM_KV_CACHE_OL_INFINISTORE_HOST_ADDR: str = "127.0.0.1"
    VLLM_KV_CACHE_OL_INFINISTORE_SERVICE_PORT: int = 12345
    VLLM_KV_CACHE_OL_INFINISTORE_CONNECTION_TYPE: str = "RDMA"
    VLLM_KV_CACHE_OL_INFINISTORE_IB_PORT: int = 1
    VLLM_KV_CACHE_OL_INFINISTORE_LINK_TYPE: str = "Ethernet"
    VLLM_KV_CACHE_OL_INFINISTORE_DEV_NAME: str = "mlx5_0"
    VLLM_KV_CACHE_OL_INFINISTORE_USE_GDR: bool = True

# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition

kv_cache_ol_environment_variables: Dict[str, Callable[[], Any]] = {
    "VLLM_KV_CACHE_OL_DOUBLE_GET_THRESHOLD":
    lambda: tuple(
        map(
            lambda x, t: t(x),
            map(
                str.strip,
                os.getenv("VLLM_KV_CACHE_OL_DOUBLE_GET_THRESHOLD", "4,0.1").
                split(","),
            ),
            (int, float),
        )),
    "VLLM_KV_CACHE_OL_CHUNK_SIZE":
    lambda: int(os.getenv("VLLM_KV_CACHE_OL_CHUNK_SIZE", "512")),
    "VLLM_KV_CACHE_OL_METRICS_ENABLED":
    lambda:
    (os.getenv("VLLM_KV_CACHE_OL_METRICS_ENABLED", "1").strip().lower() in
     ("1", "true")),
    "VLLM_KV_CACHE_OL_TIME_MEASUREMENT_ENABLED":
    lambda: (os.getenv("VLLM_KV_CACHE_OL_TIME_MEASUREMENT_ENABLED", "1").strip(
    ).lower() in ("1", "true")),
    "VLLM_KV_CACHE_OL_BREAKDOWN_MEASUREMENT_ENABLED":
    lambda: (os.getenv("VLLM_KV_CACHE_OL_BREAKDOWN_MEASUREMENT_ENABLED", "1").
             strip().lower() in ("1", "true")),
    # ================== L1Cache Env Vars ==================
    "VLLM_KV_CACHE_OL_L1_CACHE_ENABLED":
    lambda:
    (os.getenv("VLLM_KV_CACHE_OL_L1_CACHE_ENABLED", "1").strip().lower() in
     ("1", "true")),
    "VLLM_KV_CACHE_OL_L1_CACHE_EVICTION_POLICY":
    lambda: (os.getenv("VLLM_KV_CACHE_OL_L1_CACHE_EVICTION_POLICY", "S3FIFO").
             strip().upper()),
    "VLLM_KV_CACHE_OL_L1_CACHE_CAPACITY_GB":
    lambda: float(os.getenv("VLLM_KV_CACHE_OL_L1_CACHE_CAPACITY_GB", "10")),
    "VLLM_KV_CACHE_OL_DEVICE":
    lambda: (os.getenv("VLLM_KV_CACHE_OL_DEVICE", "cpu").strip().lower()),
    "VLLM_KV_CACHE_OL_L1_CACHE_EVICT_SIZE":
    lambda: int(os.getenv("VLLM_KV_CACHE_OL_L1_CACHE_EVICT_SIZE", "16")),
    # ================== S3FIFO Env Vars ==================
    # Promotion threshold of small fifo to main fifo
    "VLLM_KV_CACHE_OL_S3FIFO_SMALL_TO_MAIN_PROMO_THRESHOLD":
    lambda: int(
        os.getenv("VLLM_KV_CACHE_OL_S3FIFO_SMALL_TO_MAIN_PROMO_THRESHOLD", "1")
    ),
    # Small fifo capacity ratio
    "VLLM_KV_CACHE_OL_S3FIFO_SMALL_FIFO_CAPACITY_RATIO":
    lambda: float(
        os.getenv("VLLM_KV_CACHE_OL_S3FIFO_SMALL_FIFO_CAPACITY_RATIO", "0.3")),
    # ================== L2Cache Env Vars ==================
    "VLLM_KV_CACHE_OL_L2_CACHE_BACKEND":
    lambda:
    (os.getenv("VLLM_KV_CACHE_OL_L2_CACHE_BACKEND", "").strip().upper()),
    "VLLM_KV_CACHE_OL_L2_CACHE_NAMESPACE":
    lambda: (os.getenv("VLLM_KV_CACHE_OL_L2_CACHE_NAMESPACE", "aibrix").strip(
    ).lower()),
    "VLLM_KV_CACHE_OL_L2_CACHE_COMPRESSION":
    lambda:
    (os.getenv("VLLM_KV_CACHE_OL_L2_CACHE_COMPRESSION", "").strip().upper()),
    "VLLM_KV_CACHE_OL_L2_CACHE_OP_BATCH":
    lambda: int(os.getenv("VLLM_KV_CACHE_OL_L2_CACHE_OP_BATCH", "32")),
    "VLLM_KV_CACHE_OL_L2_CACHE_PER_TOKEN_TIMEOUT_MS":
    lambda: int(
        os.getenv("VLLM_KV_CACHE_OL_L2_CACHE_PER_TOKEN_TIMEOUT_MS", "20")),
    "VLLM_KV_CACHE_OL_L2_CACHE_INGESTION_TYPE":
    lambda: (os.getenv("VLLM_KV_CACHE_OL_L2_CACHE_INGESTION_TYPE", "HOT").
             strip().upper()),
    "VLLM_KV_CACHE_OL_L2_CACHE_INGESTION_MAX_INFLIGHT_TOKENS":
    lambda: int(
        os.getenv("VLLM_KV_CACHE_OL_L2_CACHE_INGESTION_MAX_INFLIGHT_TOKENS",
                  "2048")),
    "VLLM_KV_CACHE_OL_L2_CACHE_NUM_ASYNC_WORKERS":
    lambda: int(os.getenv("VLLM_KV_CACHE_OL_L2_CACHE_NUM_ASYNC_WORKERS", "8")),
    # ================== Mock Connector Env Vars ==================
    "VLLM_KV_CACHE_OL_MOCK_USE_RDMA":
    lambda:
    (os.getenv("VLLM_KV_CACHE_OL_MOCK_USE_RDMA", "0").strip().lower() in
     ("1", "true")),
    "VLLM_KV_CACHE_OL_MOCK_USE_GATHER_SCATTER":
    lambda: (os.getenv("VLLM_KV_CACHE_OL_MOCK_USE_GATHER_SCATTER", "0").strip(
    ).lower() in ("1", "true")),
    # ================== RocksDB Env Vars ==================
    "VLLM_KV_CACHE_OL_ROCKSDB_ROOT":
    lambda: os.path.expanduser(
        os.getenv(
            "VLLM_KV_CACHE_OL_ROCKSDB_ROOT",
            os.path.join(os.path.expanduser("~"), ".kv_cache_ol", "rocksdb"),
        )),
    "VLLM_KV_CACHE_OL_ROCKSDB_TTL_S":
    lambda: int(os.getenv("VLLM_KV_CACHE_OL_ROCKSDB_TTL_S", "600")),
    "VLLM_KV_CACHE_OL_ROCKSDB_WRITE_BUFFER_SIZE":
    lambda: int(
        os.getenv("VLLM_KV_CACHE_OL_ROCKSDB_WRITE_BUFFER_SIZE",
                  f"{64 * 1024 * 1024}")),
    "VLLM_KV_CACHE_OL_ROCKSDB_TARGET_FILE_SIZE_BASE":
    lambda: int(
        os.getenv(
            "VLLM_KV_CACHE_OL_ROCKSDB_TARGET_FILE_SIZE_BASE",
            f"{64 * 1024 * 1024}",
        )),
    "VLLM_KV_CACHE_OL_ROCKSDB_MAX_WRITE_BUFFER_NUMBER":
    lambda: int(
        os.getenv("VLLM_KV_CACHE_OL_ROCKSDB_MAX_WRITE_BUFFER_NUMBER", "3")),
    "VLLM_KV_CACHE_OL_ROCKSDB_MAX_TOTAL_WAL_SIZE":
    lambda: int(
        os.getenv(
            "VLLM_KV_CACHE_OL_ROCKSDB_MAX_TOTAL_WAL_SIZE",
            f"{128 * 1024 * 1024}",
        )),
    "VLLM_KV_CACHE_OL_ROCKSDB_MAX_BACKGROUND_JOBS":
    lambda: int(os.getenv("VLLM_KV_CACHE_OL_ROCKSDB_MAX_BACKGROUND_JOBS", "8")
                ),
    # ================== InfiniStore Env Vars ==================
    "VLLM_KV_CACHE_OL_INFINISTORE_HOST_ADDR":
    lambda:
    (os.getenv("VLLM_KV_CACHE_OL_INFINISTORE_HOST_ADDR", "127.0.0.1").strip()),
    "VLLM_KV_CACHE_OL_INFINISTORE_SERVICE_PORT":
    lambda: int(os.getenv("VLLM_KV_CACHE_OL_INFINISTORE_SERVICE_PORT", "12345")
                ),
    "VLLM_KV_CACHE_OL_INFINISTORE_CONNECTION_TYPE":
    lambda: (os.getenv("VLLM_KV_CACHE_OL_INFINISTORE_CONNECTION_TYPE", "RDMA").
             strip().upper()),
    "VLLM_KV_CACHE_OL_INFINISTORE_IB_PORT":
    lambda: int(os.getenv("VLLM_KV_CACHE_OL_INFINISTORE_IB_PORT", "12345")),
    "VLLM_KV_CACHE_OL_INFINISTORE_LINK_TYPE":
    lambda:
    (os.getenv("VLLM_KV_CACHE_OL_INFINISTORE_LINK_TYPE", "Ethernet").strip()),
    "VLLM_KV_CACHE_OL_INFINISTORE_DEV_NAME":
    lambda:
    (os.getenv("VLLM_KV_CACHE_OL_INFINISTORE_DEV_NAME", "mlx5_0").strip()),
    "VLLM_KV_CACHE_OL_INFINISTORE_USE_GDR":
    lambda:
    (os.getenv("VLLM_KV_CACHE_OL_INFINISTORE_USE_GDR", "1").strip().lower() in
     ("1", "true")),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in kv_cache_ol_environment_variables:
        return kv_cache_ol_environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(kv_cache_ol_environment_variables.keys())
