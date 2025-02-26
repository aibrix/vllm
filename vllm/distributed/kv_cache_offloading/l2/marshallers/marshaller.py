from abc import abstractmethod
from typing import Generic, TypeVar

V = TypeVar('V')
R = TypeVar('R')


class Marshaller(Generic[V, R]):
    """Marshaller is an abstraction of serializer, compressor, etc.
    """

    @abstractmethod
    def marshal(self, data: V) -> R:
        raise NotImplementedError

    @abstractmethod
    def unmarshal(self, data: R) -> V:
        raise NotImplementedError


class BaseMarshaller(Marshaller[V, R]):

    def __init__(self, marshaller: Marshaller | None = None) -> None:
        self._marshaller = marshaller

    def marshal(self, data: V) -> R:
        if self._marshaller is None:
            return self._marshal(data)
        else:
            return self._marshal(self._marshaller.marshal(data))

    def unmarshal(self, data: R) -> V:
        if self._marshaller is None:
            return self._unmarshal(data)
        else:
            return self._marshaller.unmarshal(self._unmarshal(data))

    @abstractmethod
    def _marshal(self, data: V) -> R:
        raise NotImplementedError

    @abstractmethod
    def _unmarshal(self, data: R) -> V:
        raise NotImplementedError
