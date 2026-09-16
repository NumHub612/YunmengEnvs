# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Estimation-layer contracts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from yunmeng.interfaces.types import ArrayLike, ParamMeta

# ---------------------------------------------------
# region IParameterized
# ---------------------------------------------------


class IParameterized:
    """Abstract base class for parameterized objects."""

    @abstractmethod
    def parameter_metas(self) -> list[ParamMeta]:
        """Parameter descriptors list."""
        ...

    def parameter_bounds(self, names: list[str] = None) -> tuple[list, list]:
        """(lower, upper) bound lists aligned with *names*."""
        specs = {p.name: p for p in self.parameter_metas()}
        names = names or specs
        inf = float("inf")
        lo, hi = [], []
        for n in names:
            b = specs[n].bounds if n in specs else (None, None)
            lo.append(-inf if b[0] is None else float(b[0]))
            hi.append(inf if b[1] is None else float(b[1]))
        return lo, hi

    @abstractmethod
    def get_parameters(self, names: list[str] = None) -> ArrayLike:
        """Flat parameter vector aligned with `names`."""
        ...

    @abstractmethod
    def set_parameters(self, values: ArrayLike, names: list[str]):
        """Write flat parameters."""
        ...


# ---------------------------------------------------
# region IEstimable
# ---------------------------------------------------


class IEstimable(IParameterized):
    """Estimation target contract: parameter-vector access + run +
    gradient capability declaration.

    Framework entry point is the MODEL layer only:
    users call estimator.fit(model, ...). A solver MAY also
    implement this for standalone testing, but that is not
    a framework-guaranteed path.
    """

    @abstractmethod
    def run(self, n_steps: int, **kwargs) -> Any:
        """Execute one forward pass of n_steps.

        Returns an object from which observation-equivalent output
        can be extracted. In TRAIN mode, the returned state must
        stay on the autograd graph.
        """
        ...

    @abstractmethod
    def reset(self):
        """Reset to the initial state for a fresh evaluation,
        keeping the current parameter values."""
        ...

    @abstractmethod
    def supports_gradients(self) -> bool:
        """Whether end-to-end backpropagation is available NOW."""
        ...


def split_namespaces(names: list[str]) -> dict[str, list[str]]:
    """Group namespaced names by their first segment.

    ["s1.K", "s1.W","s2.K"] -> {"s1": ["K", "W"], "s2": ["K"]}
    """
    groups: dict[str, list[str]] = {}
    for n in names:
        head, _, tail = n.partition(".")
        groups.setdefault(head, []).append(tail)
    return groups
