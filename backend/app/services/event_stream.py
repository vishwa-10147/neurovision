"""Per-run event history with live delivery to WebSocket subscribers.

Shared by built-in training jobs and external runs, so one WebSocket route
(WS /api/v1/training/{id}/events) can follow either.
"""

import asyncio
import threading
from typing import Any

TERMINAL_EVENTS = {"training.completed", "training.cancelled", "training.failed"}
Event = dict[str, Any]
Subscriber = tuple[asyncio.AbstractEventLoop, "asyncio.Queue[Event]"]


class EventStream:
    def __init__(self) -> None:
        self._events: list[Event] = []
        self._subscribers: set[Subscriber] = set()
        self._lock = threading.Lock()

    def publish(self, event: Event) -> None:
        """Record an event and deliver it to every subscriber (any thread)."""
        with self._lock:
            self._record(event)
            subscribers = list(self._subscribers)
        for loop, queue in subscribers:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, event)
            except RuntimeError:  # The subscriber's event loop has closed.
                self.unsubscribe(queue)

    def subscribe(self) -> tuple[list[Event], "asyncio.Queue[Event]"]:
        """Call from an event loop: past events plus a queue of future ones."""
        queue: asyncio.Queue[Event] = asyncio.Queue()
        with self._lock:
            history = list(self._events)
            self._subscribers.add((asyncio.get_running_loop(), queue))
        return history, queue

    def unsubscribe(self, queue: "asyncio.Queue[Event]") -> None:
        with self._lock:
            self._subscribers = {s for s in self._subscribers if s[1] is not queue}

    def _record(self, event: Event) -> None:
        """Add an event to the replay history; called with the lock held."""
        self._events.append(event)
