# Copyright 2024 The Aibrix Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import concurrent.futures
import json
import queue
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

import torch

from vllm.logger import init_logger

from .file import LoadFile, LocalFile, S3File
from .utils import (TensorMeta, _create_s3_client, _parse_bucket_info_from_uri,
                    split_continue_tensors)

logger = init_logger(__name__)


def get_safetensors_metas(file: LoadFile):
    LENTH_COUNT = 8
    length_bytes = file.load_to_bytes(offset=0, count=LENTH_COUNT)
    length_of_header = struct.unpack("<Q", length_bytes.read())[0]
    base_offset = length_of_header + LENTH_COUNT

    meta_bytes = file.load_to_bytes(offset=LENTH_COUNT, count=length_of_header)

    tensors_meta = json.loads(meta_bytes.read().decode("utf-8"))
    del tensors_meta["__metadata__"]

    metas: List[TensorMeta] = []
    for name, tensor_meta in tensors_meta.items():
        metas.append(
            TensorMeta(
                name=name,
                base_offset=base_offset,
                dtype=tensor_meta["dtype"],
                shape=tensor_meta["shape"],
                data_offsets=tensor_meta["data_offsets"],
            ))
    # Ensure tensors chunks could be split continuously
    return sorted(metas, key=lambda obj: obj.real_offset)


class StreamLoader:

    def __init__(
        self,
        file: LoadFile,
        num_thread: int = 32,
        use_pinmem: bool = False,
        use_direct_io: bool = False,
    ) -> None:
        self.file = file
        self.num_thread = num_thread
        self.use_pinmem = use_pinmem
        self.use_direct_io = use_direct_io
        # TODO assert file type is safetensors
        self.tensors_metas: List[TensorMeta] = get_safetensors_metas(file)

    def load_safetensors(self, device: Union[torch.device, str] = "cpu"):
        return dict(self.get_weights_iterator(device=device))

    def _tensors_reader(
        self,
        thread_idx,
        barrier,
        device: Union[torch.device, str],
        tensor_metas: Tuple[TensorMeta, ...],
        transfer_out_queue: queue.SimpleQueue[Union[Exception, TensorMeta]],
    ) -> None:
        device = torch.device(device)
        is_cuda = device.type == "cuda"
        # TODO use stream nonblocking IO
        for tensor_meta in tensor_metas:
            tensor_buffer = self.file.load_to_buffer(
                offset=tensor_meta.real_offset, count=tensor_meta.count)
            tensor = torch.frombuffer(
                tensor_buffer, dtype=tensor_meta.dtype).view(tensor_meta.shape)
            if is_cuda:
                tensor = tensor.to(device, non_blocking=True)
            tensor_meta.set_tensor(tensor)
            transfer_out_queue.put(tensor_meta)

    def get_weights_iterator(
        self,
        device: Union[torch.device, str] = "cpu"
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        tensors_per_reader: List[Tuple[TensorMeta,
                                       ...]] = (split_continue_tensors(
                                           self.tensors_metas,
                                           self.num_thread))

        effective_num_readers = len(tensors_per_reader)
        self._reader_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=effective_num_readers,
            thread_name_prefix="SafetensorsReader",
        )
        transfer_out_queue: queue.SimpleQueue[Union[Exception, TensorMeta]] = (
            queue.SimpleQueue())  # type: ignore
        futures: List[concurrent.futures.Future] = []

        barrier = threading.Barrier(effective_num_readers)

        for thread_idx, tensor_metas in enumerate(tensors_per_reader):
            future = self._reader_pool.submit(
                self._tensors_reader,
                thread_idx,
                barrier,
                device,
                tensor_metas,
                transfer_out_queue,
            )
            futures.append(future)

        try:
            for _ in range(len(self.tensors_metas)):
                tensor_meta: Union[TensorMeta,
                                   Exception] = (transfer_out_queue.get(
                                       timeout=3600))
                if isinstance(tensor_meta, Exception):
                    raise tensor_meta
                yield tensor_meta.name, tensor_meta.tensor
        except BaseException:
            raise

    def get_weights_iterator_wo_threads(
        self,
        device: Union[torch.device, str] = "cpu"
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        device = torch.device(device)
        is_cuda = device.type == "cuda"
        # TODO use stream nonblocking IO
        for tensor_meta in self.tensors_metas:
            tensor_buffer = self.file.load_to_bytes(
                offset=tensor_meta.real_offset, count=tensor_meta.count)
            tensor = torch.frombuffer(
                tensor_buffer, dtype=tensor_meta.dtype).view(tensor_meta.shape)

            if is_cuda:
                tensor = tensor.to(device, non_blocking=True)
            # tensor_meta.set_tensor(tensor)
            yield tensor_meta.name, tensor


@dataclass
class StreamModel:
    model_uri: str
    num_threads: int = 16
    s3_access_key_id: Optional[str] = None
    s3_secret_access_key: Optional[str] = None
    s3_region: Optional[str] = None
    s3_endpinit: Optional[str] = None
    use_pinmem: bool = False
    use_direct_io: bool = False

    def __post_init__(self):
        scheme, bucket_name, bucket_path = _parse_bucket_info_from_uri(
            self.model_uri)
        if not bucket_path.endswith("/"):
            bucket_path += "/"
        self.model_source_type = scheme
        self.bucket_name = bucket_name

        # list config and safetensors files in model_uri
        files: List[str] = []
        if self.model_source_type == "local":
            local_dir = Path(self.model_uri)
            if not local_dir.exists():
                raise ValueError(f"local path {local_dir} not exist")
            files = [
                str(file) for file in local_dir.iterdir() if file.is_file()
            ]
        else:
            self.s3_client = _create_s3_client(
                ak=self.s3_access_key_id,
                sk=self.s3_secret_access_key,
                endpoint=self.s3_endpinit,
                region=self.s3_region,
                num_threads=self.num_threads,
            )
            objects_out = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name, Delimiter="/", Prefix=bucket_path)
            files = [
                str(content.get("Key"))
                for content in objects_out.get("Contents", [])
            ]

        self.config_files = [file for file in files if file.endswith(".json")]
        self.safetensors_files = [
            file for file in files if file.endswith(".safetensors")
        ]

        if len(self.config_files) == 0:
            raise ValueError(f"no config file found in {self.model_uri}")
        if len(self.safetensors_files) == 0:
            raise ValueError(f"no safetensors file found in {self.model_uri}")

    def download_config(self,
                        target_dir: str,
                        force_download: bool = False) -> Path:
        if self.model_source_type == "local":
            logger.info("local config no need to download")
            return Path(self.model_uri)

        for config_file in self.config_files:
            config_s3 = S3File(
                scheme=self.model_source_type,
                bucket_name=self.bucket_name,
                bucket_path=config_file,
                s3_client=self.s3_client,
            )

            config_s3.download_file(
                target_dir=target_dir,
                num_threads=self.num_threads,
                force_download=force_download,
            )

        target_path = Path(target_dir)
        return target_path

    def get_weights_iterator(self, device: Union[torch.device, str] = "cpu"):
        for safetensors_file in self.safetensors_files:
            safetensors_s3: LoadFile
            if self.model_source_type == "local":
                safetensors_s3 = LocalFile(str(safetensors_file))
            else:
                safetensors_s3 = S3File(
                    scheme=self.model_source_type,
                    bucket_name=self.bucket_name,
                    bucket_path=safetensors_file,
                    s3_client=self.s3_client,
                )
            safetensors_loader = StreamLoader(
                file=safetensors_s3,
                num_thread=self.num_threads,
                use_pinmem=self.use_pinmem,
                use_direct_io=self.use_direct_io,
            )
            for name, tensor in safetensors_loader.get_weights_iterator(
                    device="cpu"):
                yield name, tensor
