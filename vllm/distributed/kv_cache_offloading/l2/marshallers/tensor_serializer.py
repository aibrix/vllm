import io
import struct
import torch

from typing import Iterable, Tuple
from . import BaseMarshaller, Marshaller
from ...utils import tensor_to_bytes


class TensorSerializer(BaseMarshaller):

    def __init__(self, marshaller: Marshaller | None = None) -> None:
        super().__init__(marshaller)

    def _marshal(
            self,
            obj: torch.Tensor | Tuple[Iterable[int], torch.Tensor]) -> bytes:
        buffer = io.BytesIO()
        if isinstance(obj, torch.Tensor):
            # 0 indicates no indices
            buffer.write(struct.pack('i', 0))
            buffer.write(tensor_to_bytes(obj.view(torch.uint8)))
        else:
            indices, tensor = obj
            # non-zero indicates we have indices before tensor bytes
            buffer.write(struct.pack('i', len(indices)))
            for index in indices:
                buffer.write(struct.pack('i', index))
            buffer.write(tensor_to_bytes(tensor.view(torch.uint8)))
        return buffer.getvalue()

    def _unmarshal(
            self,
            data: bytes) -> torch.Tensor | Tuple[Iterable[int], torch.Tensor]:
        buffer = io.BytesIO(data)
        have_indices = struct.unpack('i', buffer.read(4))[0]

        if have_indices == 0:
            return torch.tensor(list(buffer.read()), dtype=torch.uint8)
        else:
            length_of_indices = have_indices
            indices = [
                struct.unpack("i", buffer.read(4))[0]
                for _ in range(length_of_indices)
            ]
            tensor = torch.tensor(list(buffer.read()), dtype=torch.uint8)
            return (indices, tensor)
