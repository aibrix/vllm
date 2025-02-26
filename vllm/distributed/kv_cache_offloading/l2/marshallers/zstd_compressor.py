import zstandard

from . import BaseMarshaller, Marshaller


class ZstdCompressor(BaseMarshaller):

    def __init__(self, marshaller: Marshaller | None = None) -> None:
        super().__init__(marshaller)
        self.compressor = zstandard.ZstdCompressor()
        self.decompressor = zstandard.ZstdDecompressor()

    def _marshal(self, data: bytes) -> bytes:
        return self.compressor.compress(data)

    def _unmarshal(self, data: bytes) -> bytes:
        return self.decompressor.decompress(data)
