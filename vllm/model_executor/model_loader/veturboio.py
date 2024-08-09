import os
import time
import torch
from torch import nn
from dataclasses import dataclass
from typing import Optional, Tuple

from vllm.config import ModelConfig, ParallelConfig
from vllm.logger import init_logger


veturboio_error_msg = None

try:
    import veturboio
    from veturboio.ops.load_utils import IOHelper
    if IOHelper is None:
        helper = None
    else:
        helper = veturboio.init_io_helper()
except ImportError as e:
    veturboio_error_msg = str(e)


logger = init_logger(__name__)


@dataclass
class VeturboIOConfig:
    model_files: Optional[Tuple[str, os.PathLike]] = None
    map_location: Optional[str] = "cpu"
    enable_fast_mode: Optional[bool] = True
    num_thread: Optional[int] = 32
    use_pinmem: Optional[bool] = False
    use_direct_io: Optional[bool] = False
    use_cipher: Optional[bool] = False  # not implemented yet

    def _construct_veturboio_args(self) -> "VeturboIOArgs":
        veturboio_args = {
            "map_location": self.map_location,
            "enable_fast_mode": self.enable_fast_mode,
            "num_thread": self.num_thread,
            "use_pinmem": self.use_pinmem,
            "use_direct_io": self.use_direct_io,
            "use_cipher": self.use_cipher,
        }
        return VeturboIOArgs(**veturboio_args)

    def verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:
        if parallel_config.tensor_parallel_size > 1:
            raise ValueError(
                "veturboio does not support tensor parallelism yet")

    def verify_with_model_config(self, model_config: "ModelConfig") -> None:
        if model_config.quantization is not None:
            logger.warning(
                "Loading a model using VeturboIO with quantization on vLLM"
                " is unstable and may lead to errors.")


class VeturboIOAgent:

    def __init__(self, veturboio_config: VeturboIOConfig):
        if veturboio_error_msg is not None:
            raise ImportError(
                "VeturboIO is not installed. Please install ImportError "
                "to use this feature with `pip install vllm[veturboio]`. "
                "Error message: {}".format(veturboio_error_msg))

        self.veturboio_config = veturboio_config
        self.veturboio_args = (
            self.veturboio_config._construct_veturboio_args())

    def deserialize(self, model):
        assert isinstance(model, torch.nn.Module)
        start = time.perf_counter()
        for model_file in self.veturboio_config.model_files:

            tensors_dict = veturboio.load(model_file, 
                                          helper=helper, 
                                          **self.veturboio_args.deserializer_params)
            model.load_weights(iter(tensors_dict.items()))
            del tensors_dict
            # gc.collect()  # do gc collect immediately
            torch.cuda.empty_cache()
                
        end = time.perf_counter()
        duration = end - start
        logger.info("Deserialized model in %0.2fs by VeturboIO", duration)


def load_with_veturboio_into_model(veturboio_config: VeturboIOConfig,
                        model: nn.Module):
    assert veturboio_config.model_files is not None, ("model files can not be None, "
                                                      "when load with veturboIO")
    veturboio = VeturboIOAgent(veturboio_config)
    return veturboio.deserialize(model)


@dataclass
class VeturboIOArgs:
    map_location: Optional[str] = "cpu"
    enable_fast_mode: Optional[bool] = True
    num_thread: Optional[int] = 32
    use_pinmem: Optional[bool] = False
    use_direct_io: Optional[bool] = False
    use_cipher: Optional[bool] = False  # not implemented yet

    def __post_init__(self):
        self.deserializer_params = {
            "map_location": self.map_location,
            "enable_fast_mode": self.enable_fast_mode,
            "num_thread": self.num_thread,
            "use_pinmem": self.use_pinmem,
            "use_direct_io": self.use_direct_io,
            "use_cipher": self.use_cipher,
        }
