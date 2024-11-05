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

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import NoReturn, Optional

import numpy as np
from boto3.s3.transfer import TransferConfig

from .utils import (
    _create_s3_client,
    _parse_bucket_info_from_uri,
    read_to_bytes_io,
)

from vllm.logger import init_logger

logger = init_logger(__name__)


class LoadFile:
    def __init__(self, file_source: str) -> None:
        self.file_source = file_source

    def load_whole_file(self, num_threads: int = 1) -> NoReturn:
        raise NotImplementedError

    def load_to_bytes(self, offset: int, count: int) -> BytesIO:
        raise NotImplementedError

    def load_to_buffer(self, offset: int, count: int) -> memoryview:
        raise NotImplementedError
    
    def download(self, target_dir) -> NoReturn:
        raise NotImplementedError


class LocalFile(LoadFile):
    def __init__(self, file: str) -> None:
        if not Path(file).exists():
            raise ValueError(f"file {file} not exist")

        self.file = file
        super().__init__(file_source="local")

    def load_whole_file(self, num_threads: int = 1):
        if num_threads != 1:
            logger.warning(
                f"num_threads {num_threads} is not supported for local file."
            )

        tensor_bytes = np.memmap(
            self.file,
            dtype=np.uint8,
            mode="c",
        )
        return tensor_bytes.tobytes()

    def load_to_bytes(self, offset: int, count: int):
        return BytesIO(self.load_to_buffer(offset=offset, count=count))

    def load_to_buffer(self, offset: int, count: int):
        return np.memmap(
            self.file,
            dtype=np.uint8,
            mode="r",
            offset=offset,
            shape=count,
        )


class RemoteFile(LoadFile):
    def __init__(self, file: str, file_source: str) -> None:
        self.file = file
        super().__init__(file_source=file_source)

    def load_to_buffer(self, offset: int, count: int):
        tensor_bytes = self.load_to_bytes(offset=offset, count=count)
        return tensor_bytes.getbuffer()


class S3File(RemoteFile):
    def __init__(self, file: str) -> None:
        self.file = file
        scheme, bucket_name, bucket_path = _parse_bucket_info_from_uri(file)
        self.bucket_name = bucket_name
        self.bucket_path = bucket_path
        try:
            s3_client = _create_s3_client()
            s3_client.head_object(Bucket=bucket_name, Key=bucket_path)
        except Exception as e:
            raise ValueError(f"S3 bucket path {bucket_path} not exist for {e}.")
        super().__init__(file=file, file_source="s3")

    def load_whole_file(self, num_threads: int):
        s3_client = _create_s3_client()

        config_kwargs = {
            "max_concurrency": num_threads,
            "use_threads": True,
        }
        config = TransferConfig(**config_kwargs)

        data = BytesIO()
        s3_client.download_fileobj(
            Bucket=self.bucket_name,
            Key=self.bucket_path,
            Fileobj=data,
            Config=config,
        )
        return data.getbuffer()

    def load_to_bytes(self, offset: int, count: int):
        s3_client = _create_s3_client()

        range_header = f"bytes={offset}-{offset+count-1}"
        resp = s3_client.get_object(
            Bucket=self.bucket_name, Key=self.bucket_path, Range=range_header
        )
        return read_to_bytes_io(resp.get("Body"))


@dataclass
class StreamModel:
    model_uri: str
    num_threads: int = 16
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: Optional[str] = None
    aws_endpinit: Optional[str] = None

    def __post_init__(self):
        scheme, bucket_name, bucket_path = _parse_bucket_info_from_uri(self.model_uri)
        self.model_source_type = scheme
        self.bucket_name = bucket_name
        
        # list config and safetensors files in model_uri
        if self.model_source_type == "local":
            local_dir = Path(bucket_path)
            if not local_dir.exists():
                raise ValueError(f"local path {local_dir} not exist")
            files = [file for file in local_dir.iterdir() if file.is_file()]
            
            self.config_files = [file for file in files if file.suffix == ".json"]
            self.safetensors_files = [file for file in files if file.suffix == ".safetensors"]
        else:
            self.client = _create_s3_client(ak=self.aws_access_key_id,
                                            sk=self.aws_secret_access_key,
                                            endpoint=self.aws_endpinit,
                                            region=self.aws_endpinit)
            objects_out = self.client.list_objects_type2(
                self.bucket_name, prefix=bucket_path, delimiter="/"
            )
            files = [obj.key for obj in objects_out.contents]
            
            self.config_files = [file for file in files if file.endswith(".json")]
            self.safetensors_files = [file for file in files if file.endswith(".safetensors")]

        if len(self.config_files) == 0:
            raise ValueError(f"no config file found in {self.model_uri}")
        if len(self.safetensors_files) == 0:
            raise ValueError(f"no safetensors file found in {self.model_uri}")


    def download_config(self, target_dir: str) -> Path:
        if self.model_source_type == "local":
            logger.info("local config no need to download")
            return Path(self.model_uri)
        
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        for config_file in self.config_files:
            _file_name = config_file.split("/")[-1]
            local_file = target_path.joinpath(_file_name).absolute()
            config_kwargs = {
                "max_concurrency": self.num_threads,
                "use_threads": True,
            }
            config = TransferConfig(**config_kwargs)
            self.client.download_file(
                Bucket=self.bucket_name,
                Key=config_file,
                Filename=str(
                    local_file
                ),  # S3 client does not support Path, convert it to str
                Config=config,
            )
        return target_path

        