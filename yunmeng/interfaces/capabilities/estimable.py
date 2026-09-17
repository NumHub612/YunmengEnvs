# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Estimation-layer contracts.
"""

from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from yunmeng.interfaces.types import ArrayLike

# ---------------------------------------------------
# region IParameterized
# ---------------------------------------------------


@dataclass
class ParamMeta:
    """Tunable parameter descriptor."""

    name: str
    description: str = ""
    dtype: str = "float"
    bounds: tuple = (None, None)  # (min, max)
    default: Any = None
    required: bool = False


@runtime_checkable
class IParameterized(Protocol):
    """θ channel: named, bounded parameter access (COPY semantics).

    get returns value snapshots decoupled from any autograd graph;
    set writes values in. Serves archival, artifact registration and
    gradient-free calibration.

    Applicable layers: operator, solver, model.
    """

    @abstractmethod
    def parameter_metas(self) -> list[ParamMeta]:
        """Parameter descriptors list."""
        ...

    def parameter_names(self) -> list[str]:
        """Ordered parameter names."""
        return [p.name for p in self.parameter_metas()]

    @abstractmethod
    def get_parameters(self, names: list[str] = None) -> ArrayLike:
        """Flat parameter vector aligned with `names`."""
        ...

    @abstractmethod
    def set_parameters(self, values: ArrayLike, names: list[str]):
        """Write flat parameters aligned with *names*."""
        ...

    def parameter_bounds(self, names: list[str] = None) -> tuple:
        """(lower, upper) bound lists aligned with *names*."""
        specs = {p.name: p for p in self.parameter_metas()}
        names = names or list(specs.keys())
        inf = float("inf")
        lo, hi = [-inf] * len(names), [inf] * len(names)
        for i, n in enumerate(names):
            if n not in specs:
                raise ValueError(
                    f"Parameter {n} not found in {self.__class__.__name__}"
                )
            bo, bi = specs[n].bounds
            lo[i] = -inf if bo is None else float(bo)
            hi[i] = inf if bi is None else float(bi)
        return lo, hi


# ---------------------------------------------------
# region IEstimable
# ---------------------------------------------------


@runtime_checkable
class IEstimable(IParameterized, Protocol):
    """Estimation target contract: parameter-vector access + run +
    gradient capability declaration.

    Applicable layers: operator, solver.
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


# ---------------------------------------------------
# region IAssimilatable
# ---------------------------------------------------


@runtime_checkable
class IAssimilatable(Protocol):
    """Merge external observations into runtime STATE.

    Assimilation adjusts state; estimation adjusts parameters.

    Applicable layers: solver, model.
    """

    @abstractmethod
    def assimilate(self, observations: Any, **kwargs): ...
