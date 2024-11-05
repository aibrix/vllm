import os
from io import BytesIO
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import boto3
import torch
from botocore.config import Config

from . import SUPPORTED_STREAM_STORAGE


def read_to_bytes_io(content, chunk_size=None):
    chunk_size = int(os.getenv("STREAM_READ_CHUNK_SIZE", 8388608))  # 8MB
    bytes_io = BytesIO()
    buf = content.read(chunk_size)
    while buf:
        bytes_io.write(buf)
        buf = content.read(chunk_size)
    bytes_io.seek(0)
    return bytes_io


def filter_suffix_files(files: List[str], suffix: str) -> List[str]:
    return [file for file in files if file.endswith(suffix)]


def get_dtype(dtype_str: str):
    # torch.float8 formats require 2.1; we do not support these dtypes on earlier versions
    _float8_e4m3fn = getattr(torch, "float8_e4m3fn", None)
    _float8_e5m2 = getattr(torch, "float8_e5m2", None)
    _TYPES = {
        "F64": torch.float64,
        "F32": torch.float32,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "I64": torch.int64,
        # "U64": torch.uint64,
        "I32": torch.int32,
        # "U32": torch.uint32,
        "I16": torch.int16,
        # "U16": torch.uint16,
        "I8": torch.int8,
        "U8": torch.uint8,
        "BOOL": torch.bool,
        "F8_E4M3": _float8_e4m3fn,
        "F8_E5M2": _float8_e5m2,
    }
    return _TYPES[dtype_str]


def _parse_bucket_info_from_uri(uri: str) -> Tuple[str, str, str]:
    parsed = urlparse(uri)
    scheme = parsed.scheme
    # uri is local path when scheme is empty
    scheme = "local" if scheme == "" else scheme
    if scheme not in SUPPORTED_STREAM_STORAGE:
        raise ValueError(f"{scheme} not supported, 
                         only {SUPPORTED_STREAM_STORAGE} supported")

    bucket_name = parsed.netloc
    bucket_path = parsed.path.lstrip("/") if scheme != "" else parsed.path
    return scheme, bucket_name, bucket_path


def _create_s3_client(ak, sk, endpoint, region):
    ak = ak or os.getenv("AWS_ACCESS_KEY_ID")
    sk = sk or os.getenv("AWS_SECRET_ACCESS_KEY")
    endpoint = endpoint or os.getenv("AWS_ENDPOINT_URL")
    region = region or os.getenv("AWS_REGION")

    my_config = Config(
        # signature_version = 'v4',
        s3={"addressing_style": "virtual"}
    )
    return boto3.client(
        service_name="s3",
        region_name=region,
        endpoint_url=endpoint,
        aws_access_key_id=ak,
        aws_secret_access_key=sk,
        config=my_config
    )


class TensorMeta:
    def __init__(self, name: str, base_offset: int, dtype: str, shape: List[int], data_offsets: List[int]) -> None:
        self._name = name
        self._base_offset = base_offset
        self._dtype = get_dtype(dtype)
        self._shape = shape
        self._data_offsets = data_offsets
        self._tensor = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def shape(self) -> List[int]:
        return self._shape

    @property
    def data_offsets(self) -> List[int]:
        return self._data_offsets

    @property
    def real_offset(self) -> List[int]:
        return self._data_offsets[0] + self._base_offset

    @property
    def count(self) -> List[int]:
        return self._data_offsets[1] - self._data_offsets[0]

    @property
    def tensor(self) -> Optional[torch.Tensor]:
        return self._tensor

    def set_tensor(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def __str__(self) -> str:
        return str(
            {
                "name": self._name,
                "dtype": self._dtype,
                "shape": self._shape,
                "data_offsets": self._data_offsets,
            }
        )

    def __repr__(self) -> str:
        return self.__str__()


def split_continue_tensors(tensor_metas: List[TensorMeta], num_readers:int) -> List[Tuple[TensorMeta]]:
    """
    Note: Usually, the number of groups for splitting tensors
          is greater than num_deaders.
    """
    assert len(tensor_metas) > 0, "tensor_metas should not be empty"
    assert num_readers > 0, "num_readers should be greater than 0"

    if len(tensor_metas) <= num_readers:
        return [(item,) for item in tensor_metas]

    max_offset = tensor_metas[-1].data_offsets[1]
    avg_size = max_offset // num_readers
    group = []
    groups = []
    group_size = 0
    for tensor_meta in tensor_metas:
        if len(group) == 0 or group_size + tensor_meta.count <= avg_size:
            group.append(tensor_meta)
            group_size += tensor_meta.count
        else:
            groups.append(tuple(group))
            group = [tensor_meta]
            group_size = tensor_meta.count

    if len(group) != 0:
        groups.append(tuple(group))
    return groups


def split_continue_tensors_v1(tensor_metas: List[TensorMeta], num_readers:int) -> List[Tuple[TensorMeta]]:
    assert len(tensor_metas) > 0, "tensor_metas should not be empty"
    assert num_readers > 0, "num_readers should be greater than 0"

    if len(tensor_metas) <= num_readers:
        return [(item,) for item in tensor_metas]

    max_offset = tensor_metas[-1].data_offsets[1]
    avg_size = max_offset // num_readers
    group = []
    groups = []
    current_max_offset = avg_size
    for tensor_meta in tensor_metas:
        start, end = tensor_meta.data_offsets
        while start >= current_max_offset:
            current_max_offset += avg_size

        if end <= current_max_offset:
            group.append(tensor_meta)
        else:
            if len(group) != 0:
                groups.append(tuple(group))
            group = [tensor_meta]

            current_max_offset += avg_size
    if len(group) != 0:
        groups.append(tuple(group))
    return groups
