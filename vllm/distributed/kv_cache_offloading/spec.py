import enum
import torch

from dataclasses import dataclass
from typing import Callable, List, Tuple


@dataclass
class KVCacheLayerSpec:
    """The specification of the kv cache tensor for each layer.

    Args:
        size: The size of the kv cache layer in bytes. For FullAttention,
            size = num_heads * head_dim * dtype_size
    """
    size: int

    def __post_init__(self):
        if self.size <= 0:
            raise ValueError(
                "The size of the kv cache layer must be greater than 0.")


@dataclass
class KVCacheTensorSpec:
    """The specification of the kv cache tensor.
    Args:
        heads: head ids. To support tensor parallelism.
        layers: layer ids. To support pipeline parallellism.
        layer_spec: layer specs.
    """
    heads: List[int]
    layers: List[int]
    layer_specs: List[KVCacheLayerSpec]

    def __post_init__(self):
        if len(self.heads) <= 0:
            raise ValueError("The number of heads must be greater than 0.")
        if len(self.layers) <= 0:
            raise ValueError("The number of layers must be greater than 0.")
        if len(self.layer_specs) != len(self.layers):
            raise ValueError(
                "The number of layer specs must be equal to the number of layers."
            )


class KVCacheBlockLayout(enum.Enum):
    """The layout of the kv cache block.

    Args:
        NCLD:
            This layout signifies that the shape would be [num_tokens, 2 (k & v), num_layers, layer_dim].
            |<-------------------------------- Block i ---------------------------------->|
            |...|<------------------------------ Token j ---------------------------->|...|
            |...|<---------------- K ------------->|<---------------- V ------------->|...|
            |...|<-- Layer 0 ->||<-- Layer 1 ->|...|<-- Layer 0 ->||<-- Layer 1 ->|...|...|

            For a heterogeneous tensor, its shape will be [num_tokens, 2, num_layers, [layer0_dim,
            layer1_dim, ...]].
        LCND:
            This layout signifies that the shape would be [num_layers, 2 (k & v), num_tokens, layer_dim].
            |<-------------------------------- Block i ---------------------------------->|
            |...|<------------------------------ Layer j ---------------------------->|...|
            |...|<---------------- K ------------->|<---------------- V ------------->|...|
            |...|<-- Token 0 ->||<-- Token 1 ->|...|<-- Token 0 ->||<-- Token 1 ->|...|...|

            For a heterogeneous tensor, its shape will be [num_layers, 2, num_tokens, [layer0_dim,
            layer1_dim, ...]].
    """
    NCLD = enum.auto()
    LCND = enum.auto()


@dataclass
class KVCacheBlockSpec:
    """The specification of the kv cache block.
    Args:
        block_ntokens: The number of tokens in each block.
        block_dtype: The dtype of the kv cache block.
        block_layout: The layout of the kv cache block.
        tensor_spec: The specification of the kv cache tensor.
    """
    block_ntokens: int
    block_dtype: torch.dtype
    block_layout: KVCacheBlockLayout
    tensor_spec: KVCacheTensorSpec

    def __post_init__(self):
        if self.block_ntokens <= 0:
            raise ValueError("block_ntokens must be greater than 0.")
        self.block_nbytes: int = (2 * self.block_ntokens *
                                  self.block_dtype.itemsize *
                                  sum(s.size
                                      for s in self.tensor_spec.layer_specs))
        self.block_shape: Tuple[int, ...] = self._get_block_shape()
        self.block_shape_token_dim: int = 0 if self.block_layout == KVCacheBlockLayout.NCLD else 2
        self.is_homogeneous: Callable[
            [], bool] = lambda: isinstance(self.block_shape[-1], int)

    def _get_block_shape(self) -> Tuple[int, ...]:
        if all(s.size == self.tensor_spec.layer_specs[0].size
               for s in self.tensor_spec.layer_specs):
            # Homogeneous block shape
            if self.block_layout == KVCacheBlockLayout.NCLD:
                return (
                    self.block_ntokens,
                    2,
                    len(self.tensor_spec.layers),
                    self.tensor_spec.layer_specs[0].size,
                )
            elif self.block_layout == KVCacheBlockLayout.LCND:
                return (
                    len(self.tensor_spec.layers),
                    2,
                    self.block_ntokens,
                    self.tensor_spec.layer_specs[0].size,
                )
        else:
            # Heterogenous block shape
            if self.block_layout == KVCacheBlockLayout.NCLD:
                return (
                    self.block_ntokens,
                    2,
                    len(self.tensor_spec.layers),
                    tuple(s.size for s in self.tensor_spec.layer_specs),
                )
            elif self.block_layout == KVCacheBlockLayout.LCND:
                return (
                    len(self.tensor_spec.layers),
                    2,
                    self.block_ntokens,
                    tuple(s.size for s in self.tensor_spec.layer_specs),
                )
