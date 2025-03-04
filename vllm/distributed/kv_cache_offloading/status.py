import asyncio
import functools
import traceback

from typing import Generic, TypeVar

T = TypeVar("T")


class StatusCodes:
    OK = "OK"
    ERROR = "ERROR"
    NOT_FOUND = "NOT_FOUND"
    INVALID = "INVALID"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    TIMEOUT = "TIMEOUT"
    DENIED = "DENIED"
    CANCELLED = "CANCELLED"


class Status(Generic[T]):

    def __init__(self, error_code: str | None = None, value: T | None = None):
        if error_code is None and value is None:
            raise ValueError("Either error_code or value must be provided")

        self.trace = None
        if error_code is None or error_code == StatusCodes.OK:
            self.error_code = StatusCodes.OK
            self.value = value
        else:
            self.error_code = error_code
            self.value = value
            if isinstance(value, Exception):
                self.trace = traceback.format_exc()

    def is_ok(self) -> bool:
        return self.error_code == StatusCodes.OK

    def __getattr__(self, name):
        if name.startswith("is_"):
            expected_code = name[3:].upper()
            return lambda: self.error_code == getattr(StatusCodes,
                                                      expected_code, None)
        raise AttributeError(f"{name} not found")

    def __repr__(self):
        if self.trace is not None:
            return f"Status(error_code={self.error_code}, value={self.value}), trace:\n{self.trace}"
        elif self.value is not None:
            return f"Status(error_code={self.error_code}, value={self.value})"
        else:
            return f"Status(error_code={self.error_code})"

    def __str__(self) -> str:
        return self.__repr__()

    def raise_if_has_exception(self) -> None:
        if self.value is not None and isinstance(self.value, Exception):
            raise self.value

    @staticmethod
    def capture_exception(func):
        """A decorator that converts exceptions raised by the decorated
        function into Status.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return Status(StatusCodes.ERROR, e)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                return Status(StatusCodes.ERROR, e)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
