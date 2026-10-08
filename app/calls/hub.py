"""Fan out call-log changes to live /monitor pages (Server-Sent Events).

Publishers may run on the event loop (the call bridge) or in worker threads
(console requests), so delivery goes through call_soon_threadsafe.
"""

import asyncio
import logging
import threading

LOGGER = logging.getLogger("univoice.monitor")

_QUEUE_SIZE = 1000

_lock = threading.Lock()
_subscribers: set[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = set()


def subscribe() -> asyncio.Queue:
    queue: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_SIZE)
    with _lock:
        _subscribers.add((asyncio.get_running_loop(), queue))
    return queue


def unsubscribe(queue: asyncio.Queue) -> None:
    with _lock:
        for entry in [e for e in _subscribers if e[1] is queue]:
            _subscribers.discard(entry)


def _offer(queue: asyncio.Queue, message: dict) -> None:
    try:
        queue.put_nowait(message)
    except asyncio.QueueFull:
        # A stalled browser tab must not hold memory or slow down calls.
        LOGGER.debug("Monitor subscriber is too slow; dropping an update")


def publish(message: dict) -> None:
    with _lock:
        subscribers = list(_subscribers)
    for loop, queue in subscribers:
        try:
            loop.call_soon_threadsafe(_offer, queue, message)
        except RuntimeError:
            unsubscribe(queue)  # its event loop has closed
