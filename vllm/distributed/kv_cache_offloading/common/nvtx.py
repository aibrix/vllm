import asyncio
import functools
import hashlib

_NVTX_COLORS = ["green", "blue", "yellow", "purple", "rapids", "red"]


@functools.lru_cache()
def _nvtx_get_color(name: str):
    m = hashlib.sha256()
    m.update(name.encode())
    hash_value = int(m.hexdigest(), 16)
    idx = hash_value % len(_NVTX_COLORS)
    return _NVTX_COLORS[idx]


try:

    import nvtx

    def nvtx_range(msg: str, domain: str):
        """ 
        Decorator and context manager for NVTX profiling.
        Supports both sync and async functions.

        Args:
            msg (str): Message associated with the NVTX range.
            domain (str): NVTX domain.
        """

        def decorator(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                nvtx.range_push(msg=msg,
                                domain=domain,
                                color=_nvtx_get_color(msg))
                try:
                    return await func(*args, **kwargs)
                finally:
                    nvtx.range_pop()

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                nvtx.range_push(msg=msg,
                                domain=domain,
                                color=_nvtx_get_color(msg))
                try:
                    return func(*args, **kwargs)
                finally:
                    nvtx.range_pop()

            return async_wrapper if asyncio.iscoroutinefunction(
                func) else sync_wrapper

        return decorator

except ImportError:

    def nvtx_range(msg: str, domain: str):

        def decorator(func):
            return func

        return decorator
