# SPDX-License-Identifier: Apache-2.0
from msgspec import msgpack

from . import BaseMarshaller, Marshaller


class StringSerializer(BaseMarshaller):

    def __init__(self, marshaller: Marshaller | None = None) -> None:
        super().__init__(marshaller)
        self._encoder = msgpack.Encoder()
        self._decoder = msgpack.Decoder(str)

    def _marshal(self, data: str) -> bytes:
        return self._encoder.encode(data)

    def _unmarshal(self, data: bytes) -> str:
        return self._decoder.decode(data)
