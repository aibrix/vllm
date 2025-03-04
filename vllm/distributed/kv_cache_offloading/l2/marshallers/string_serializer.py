from . import BaseMarshaller, Marshaller


class StringSerializer(BaseMarshaller):

    def __init__(self, marshaller: Marshaller | None = None) -> None:
        super().__init__(marshaller)

    def _marshal(self, data: str) -> bytes:
        return data.encode('utf-8')

    def _unmarshal(self, data: bytes) -> str:
        return data.decode('utf-8')
