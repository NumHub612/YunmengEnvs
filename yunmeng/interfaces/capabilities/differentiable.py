# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Differentiable-layer contracts.
"""

from __future__ import annotations
from abc import abstractmethod
from typing import List, Any, Protocol, runtime_checkable

from yunmeng.interfaces.types import ArrayLike, RunMode

# ---------------------------------------------------
# region IDifferentiable
# ---------------------------------------------------


@runtime_checkable
class IDifferentiable(Protocol):
    """Exposes LIVE autograd leaves for gradient training.

    Complements IParameterized diffs:
      - IParameterized  = copy semantics  -> archival / calibration
      - IDifferentiable = live semantics  -> optimizer updates in place

    Applicable layers: operator, solver.
    """

    @abstractmethod
    def grad_parameters(self) -> list[ArrayLike]:
        """Live, graph-participating parameter leaves, in stable order
        aligned with parameter_names() of the same object."""
        ...


# ---------------------------------------------------
# region IModeSwitchable
# ---------------------------------------------------


@runtime_checkable
class IModeSwitchable(Protocol):
    """Mode-sensitive behavior (TRAIN vs EVAL).

    TRAIN contract: forward paths must allocate fresh outputs and keep
    the autograd graph intact (no detach, no stale-cache replay).
    EVAL mode may reuse buffers.

    Applicable layers: operator, solver, datahub.
    """

    @abstractmethod
    def set_mode(self, mode: RunMode): ...


def propagate_mode(mode: RunMode, members: List[Any]):
    """Default mode propagation helper for containers (solver, hub)."""
    for m in members:
        if isinstance(m, IModeSwitchable):
            m.set_mode(mode)
