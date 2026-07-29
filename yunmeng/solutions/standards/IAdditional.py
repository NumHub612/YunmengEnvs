# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Parameter-vector interface for calibration and training.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from yunmeng.solutions.standards.IModel import ParamMeta

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
