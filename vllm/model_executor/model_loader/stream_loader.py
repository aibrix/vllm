


from dataclasses import dataclass
from typing import Optional

from vllm.config import ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.stream.loader import StreamModel

logger = init_logger(__name__)

@dataclass
class StreamConfig:
    model_uri: str
    num_threads: int = 16
    s3_access_key_id: Optional[str] = None
    s3_secret_access_key: Optional[str] = None
    s3_region: Optional[str] = None
    s3_endpinit: Optional[str] = None
    use_pinmem: bool = False
    use_direct_io: bool = False    

    def verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:
        if parallel_config.tensor_parallel_size > 1:
            raise ValueError(
                "steam loader does not support tensor parallelism yet")

    def verify_with_model_config(self, model_config: "ModelConfig") -> None:
        if model_config.quantization is not None:
            logger.warning(
                "Loading a model using VeturboIO with quantization on vLLM"
                " is unstable and may lead to errors.")

    def construct_stream_model(self) -> StreamModel:
        return StreamModel(
            model_uri=self.model_uri,
            num_threads=self.num_threads,
            s3_access_key_id=self.s3_access_key_id,
            s3_secret_access_key=self.s3_secret_access_key,
            s3_endpinit=self.s3_endpinit,
            s3_region=self.s3_region,
            use_pinmem=self.use_pinmem,
            use_direct_io=self.use_direct_io
        )