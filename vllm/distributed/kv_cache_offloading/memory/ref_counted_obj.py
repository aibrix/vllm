import threading

from abc import ABC, abstractmethod


class RefCountedObj(ABC):

    def __init__(self):
        """Initializes the RefCountedObj.
        """
        self.ref_count = 1
        self._lock = threading.Lock()

    def ref_up(self):
        """Increments the reference count."""
        with self._lock:
            self.ref_count += 1

    def ref_down(self):
        """Decrements the reference count and invokes destroy if the count reaches zero."""
        with self._lock:
            if self.ref_count == 0:
                raise ValueError("Reference count is already zero.")
            self.ref_count -= 1
            if self.ref_count == 0:
                self.destroy_unsafe()

    @abstractmethod
    def destroy_unsafe(self):
        """Destroys the object."""
        pass

    def __repr__(self):
        with self._lock:
            return f"RefCountedObj(ref_count={self.ref_count})"

    def __str__(self) -> str:
        return self.__repr__()
