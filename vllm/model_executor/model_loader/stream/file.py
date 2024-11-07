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

from io import BytesIO
from pathlib import Path
from typing import Optional, Union

import boto3
import numpy as np
from boto3.s3.transfer import TransferConfig

from vllm.logger import init_logger

from .utils import (_create_s3_client, _parse_bucket_info_from_uri, meta_file,
                    need_to_download, read_to_bytes_io, save_meta_data)

logger = init_logger(__name__)


class LoadFile:

    def __init__(self, file_source: str) -> None:
        self.file_source = file_source

    def load_whole_file(self, num_threads: int = 1):
        raise NotImplementedError

    def load_to_bytes(self, offset: int, count: int) -> BytesIO:
        raise NotImplementedError

    def load_to_buffer(self, offset: int, count: int) -> memoryview:
        raise NotImplementedError

    def download(self, target_dir):
        raise NotImplementedError


class LocalFile(LoadFile):

    def __init__(self, file: Union[str, Path]) -> None:
        if not Path(file).exists():
            raise ValueError(f"file {file} not exist")

        self.file = file
        super().__init__(file_source="local")

    def load_whole_file(self, num_threads: int = 1):
        if num_threads != 1:
            logger.warning("num_threads %s is not supported for local file.",
                           num_threads)

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

    def download_file(self, target_dir: str, num_threads: int = 1):
        raise NotImplementedError


class S3File(RemoteFile):

    def __init__(
        self,
        scheme: str,
        bucket_name: str,
        bucket_path: str,
        s3_client: Optional[boto3.client] = None,
        s3_access_key_id: Optional[str] = None,
        s3_secret_access_key: Optional[str] = None,
        s3_region: Optional[str] = None,
        s3_endpinit: Optional[str] = None,
    ) -> None:
        self.bucket_name = bucket_name
        self.bucket_path = bucket_path
        if s3_client is None:
            try:
                s3_client = _create_s3_client(
                    ak=s3_access_key_id,
                    sk=s3_secret_access_key,
                    endpoint=s3_endpinit,
                    region=s3_region,
                )
            except Exception as e:
                raise ValueError(f"create s3 client failed for {e}.") from e
        self.s3_client = s3_client
        try:
            self.s3_client.head_object(Bucket=bucket_name, Key=bucket_path)
        except Exception as e:
            raise ValueError(
                f"S3 bucket path {bucket_path} not exist for {e}.") from e

        file = scheme + "://" + bucket_name + "/" + bucket_path
        super().__init__(file=file, file_source=scheme)

    @classmethod
    def from_uri(cls, file_uri: str, **kwargs):
        scheme, bucket_name, bucket_path = _parse_bucket_info_from_uri(
            file_uri)
        cls(scheme, bucket_name, bucket_path, **kwargs)

    def load_whole_file(self, num_threads: int = 1):
        config_kwargs = {
            "max_concurrency": num_threads,
            "use_threads": True,
        }
        config = TransferConfig(**config_kwargs)

        data = BytesIO()
        self.s3_client.download_fileobj(
            Bucket=self.bucket_name,
            Key=self.bucket_path,
            Fileobj=data,
            Config=config,
        )
        return data.getbuffer()

    def load_to_bytes(self, offset: int, count: int):
        range_header = f"bytes={offset}-{offset+count-1}"
        resp = self.s3_client.get_object(Bucket=self.bucket_name,
                                         Key=self.bucket_path,
                                         Range=range_header)
        return read_to_bytes_io(resp.get("Body"))

    def download_file(
        self,
        target_dir: str,
        num_threads: int = 1,
        force_download: bool = False,
    ):
        try:
            meta_data = self.s3_client.head_object(Bucket=self.bucket_name,
                                                   Key=self.bucket_path)
        except Exception as e:
            raise ValueError("S3 bucket path %s not exist for %s.",
                             self.bucket_path, e) from e

        # ensure target dir exist
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)

        _file_name = self.bucket_path.split("/")[-1]
        local_file = target_path.joinpath(_file_name).absolute()

        # check if file exist
        etag = meta_data.get("ETag", "")
        file_size = meta_data.get("ContentLength", 0)

        meta_data_file = meta_file(local_path=target_path,
                                   file_name=_file_name)
        if not need_to_download(local_file, meta_data_file, file_size, etag,
                                force_download):
            logger.info("file `%s` already exist.", self.bucket_path)
            return

        config_kwargs = {
            "max_concurrency": num_threads,
            "use_threads": True,
        }
        config = TransferConfig(**config_kwargs)
        self.s3_client.download_file(
            Bucket=self.bucket_name,
            Key=self.bucket_path,
            Filename=str(
                local_file
            ),  # S3 client does not support Path, convert it to str
            Config=config,
        )
        save_meta_data(meta_data_file, etag)
        logger.info(
            "download file from `%s` to `%s` success.",
            self.bucket_path,
            target_dir,
        )
