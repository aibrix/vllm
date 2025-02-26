from abc import ABC, abstractmethod
from typing import Iterable, Tuple


class KeyBuilder(ABC):
    """KeyBuilder is used to build a sequence of keys from given tokens.
    """

    @abstractmethod
    def build(self, prefix: Iterable[int] | None,
              tokens: Iterable[int]) -> Tuple[Tuple[Iterable[int], str], ...]:
        """Build a sequnce of keys from given tokens.
        Args:
            prefix (Iterable[int] | None): prefix tokens
            tokens (Iterable[int]): tokens
        Returns:
            A sequence of keys.
        """
        raise NotImplementedError
