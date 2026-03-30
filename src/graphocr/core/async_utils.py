"""Async utilities: semaphore pool, retry decorator."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from functools import wraps
from typing import Any, TypeVar

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

T = TypeVar("T")


class SemaphorePool:
    """Bounded concurrency pool for async tasks."""

    def __init__(self, max_concurrent: int = 64):
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def run(self, coro: Coroutine[Any, Any, T]) -> T:
        async with self._semaphore:
            return await coro

    async def gather(self, *coros: Coroutine[Any, Any, T]) -> list[T]:
        tasks = [self.run(c) for c in coros]
        return await asyncio.gather(*tasks)


def async_retry(
    max_attempts: int = 3,
    retry_on: type[Exception] | tuple[type[Exception], ...] = Exception,
) -> Callable:
    """Decorator for retrying async functions with exponential backoff."""

    def decorator(func: Callable[..., Coroutine]) -> Callable[..., Coroutine]:
        @wraps(func)
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(retry_on),
            reraise=True,
        )
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await func(*args, **kwargs)

        return wrapper

    return decorator
