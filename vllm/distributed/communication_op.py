from typing import Any, Dict, Optional, Union, List

import torch
import torch.distributed

from .parallel_state import get_tp_group


def tensor_model_parallel_all_reduce(input_: torch.Tensor, **kwargs) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_, **kwargs)


def tensor_model_parallel_all_gather(input_: torch.Tensor,
                                     dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_gather(input_: torch.Tensor,
                                 dst: int = 0,
                                 dim: int = -1) -> Optional[torch.Tensor]:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)

def tensor_model_parallel_broadcast(input_: torch.Tensor,
                                 src: int = 0,
                                 ) -> torch.Tensor:
    """Broadcast the input tensor across model parallel group."""
    return get_tp_group().broadcast(input_, src)

def tensor_model_parallel_broadcast_object_list(input_: List[Any],
                                 src: int = 0) -> List[Any]:
    """Broadcast object list across model parallel group."""
    return get_tp_group().broadcast_object_list(input_, src)

def broadcast_tensor_dict(tensor_dict: Optional[Dict[Any, Union[torch.Tensor,
                                                                Any]]] = None,
                          src: int = 0):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)
