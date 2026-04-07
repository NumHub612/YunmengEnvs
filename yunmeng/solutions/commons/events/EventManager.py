# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

To provides a thread-safe event management system.
"""
from __future__ import annotations
from yunmeng.solutions.standards import (
    LinkableComponentChangeEventArgs,
    ExchangeItemChangeEventArgs,
)
from yunmeng.setting import logger
from weakref import ref
import threading
from typing import Any, Callable

# Define event handler type
EventHandler = Callable[..., Any]


class EventManager:
    """EventManager manages a list of event handlers and provides
    a thread-safe way to invoke them.
    """

    def __init__(self) -> None:
        self._handlers = set()
        self._lock = threading.RLock()

    @staticmethod
    def _make_key(handler: EventHandler):
        return ref(handler)

    def add_handler(self, handler: EventHandler):
        """Subscribes a new event handler."""
        with self._lock:
            handler_ref = self._make_key(handler)
            self._handlers.add(handler_ref)

    def invoke(self, *args: Any, **kwargs: Any):
        """Invokes all event handlers."""
        with self._lock:
            # Unpack the weak reference and call it.
            for weak_key in list(self._handlers):
                h = weak_key()
                if h is None:  # Garbage collected
                    self._handlers.discard(weak_key)
                    continue
                try:
                    h(*args, **kwargs)
                except Exception as e:
                    logger.exception(e)

    def remove_handler(self, handler: EventHandler):
        """Unsubscribes an event handler."""
        with self._lock:
            handler_ref = self._make_key(handler)
            self._handlers.discard(handler_ref)
