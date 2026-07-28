# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Lifecycle callback mechanism for linkable models.
Parameter-vector interface for calibration and training.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import numpy as np

from yunmeng.solutions.standards.IModel import ParamMeta

# ---------------------------------------------------
# region Callbacks
# ---------------------------------------------------


class CallbackEvent:
    """Standard lifecycle event names."""

    # model-level events
    BEFORE_INITIALIZE = "before_initialize"
    AFTER_INITIALIZE = "after_initialize"
    ON_PREPARE = "on_prepare"
    BEFORE_UPDATE = "before_update"
    AFTER_UPDATE = "after_update"
    ON_FINISH = "on_finish"
    ON_ERROR = "on_error"

    # scheduler-level events
    STEP_BEGIN = "step_begin"
    STEP_END = "step_end"


class ICallback(ABC):
    """Plugin invoked at model lifecycle points."""

    @abstractmethod
    def on_event(self, event: str, model: Any, context: dict):
        """Handle one lifecycle event.

        Args:
            event: one of :class:`CallbackEvent` type.
            model: the component firing the event.
            context: free-form payload.
        """
        pass


class ICallbackHost:
    """Callback registration and event firing."""

    def __init__(self):
        self._callbacks: list[ICallback] = []

    @property
    def callbacks(self) -> list[ICallback]:
        return list(self._callbacks)

    def add_callback(self, callback: ICallback):
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def remove_callback(self, callback: ICallback):
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def clear_callbacks(self):
        self._callbacks.clear()

    def _fire(self, event: str, **context):
        for cb in list(self._callbacks):
            cb.on_event(event, self, context)


# ---------------------------------------------------
# region Parametric
# ---------------------------------------------------


class IParametric(ABC):
    """Flat, named, bounded parameter vector plus run reset."""

    @abstractmethod
    def param_spec(self) -> list[ParamMeta]:
        """Parameter descriptors."""
        pass

    @abstractmethod
    def param_names(self) -> list[str]:
        """Ordered names."""
        pass

    @abstractmethod
    def get_param_vector(self, names: list[str] = None) -> np.ndarray:
        pass

    @abstractmethod
    def set_param_vector(
        self,
        values: np.ndarray,
        names: list[str] = None,
    ):
        pass

    @abstractmethod
    def reset_run(self):
        """Reset to the initial state for a fresh evaluation,
        keeping the current parameter values."""
        pass

    # -- helpers ------------------------------------

    def param_bounds(
        self,
        names: list[str] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """(lower, upper) bound arrays aligned with *names*;
        None bounds become ±inf."""
        spec = {p.name: p for p in self.param_spec()}
        names = names or self.param_names()
        lo, hi = [], []
        for n in names:
            b = spec[n].bounds if n in spec else (None, None)
            lo.append(-np.inf if b[0] is None else b[0])
            hi.append(np.inf if b[1] is None else b[1])

        lo = np.array(lo, dtype=float)
        hi = np.array(hi, dtype=float)
        return lo, hi


def split_namespaces(names: list[str]) -> dict[str, list[str]]:
    """Group namespaced names by their first segment.

    ["sub1.K", "sub1.CI", "sub2.K"] -> {"sub1": ["K", "CI"], "sub2": ["K"]}
    """
    groups: dict[str, list[str]] = {}
    for n in names:
        head, _, tail = n.partition(".")
        groups.setdefault(head, []).append(tail)
    return groups
