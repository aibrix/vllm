# SPDX-License-Identifier: Apache-2.0
import asyncio
import functools
from concurrent.futures import Executor


class AsyncBase:

    def __init__(
        self,
        executor: Executor,
        event_loop: asyncio.AbstractEventLoop | None = None,
    ):
        self._executor = executor
        self._event_loop = event_loop

    @property
    def event_loop(self):
        return self._event_loop or asyncio.get_running_loop()

    @staticmethod
    def async_wrap(**kwargs):

        def make_async_method(method_name):

            async def method(self, *args, **kwargs):
                cb = functools.partial(getattr(self, method_name), *args,
                                       **kwargs)
                return await self.event_loop.run_in_executor(
                    self._executor, cb)

            return method

        def cls_builder(cls):
            for async_method_name, orig_method_name in kwargs.items():
                setattr(cls, async_method_name,
                        make_async_method(orig_method_name))
            return cls

        return cls_builder
