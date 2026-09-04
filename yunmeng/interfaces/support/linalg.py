# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Linear-system protocol for the interfaces layer (minimal surface).

Only the members referenced by interface signatures live here
(v2.0 §14.3: interface segregation). Assembly details, storage formats
and iterative solvers stay in numerics.linalgs.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from yunmeng.interfaces.types import ArrayLike


@runtime_checkable
class ILinearEqs(Protocol):
    """Assembled linear system A·x = b, opaque to the solver layer."""

    @property
    def size(self) -> int:
        """Number of unknowns."""
        ...

    @property
    def matrix(self) -> ArrayLike:
        """System matrix (backend array or backend sparse handle)."""
        ...

    @property
    def rhs(self) -> ArrayLike:
        """Right-hand side vector."""
        ...

    def solve(self, x0: ArrayLike | None = None) -> ArrayLike:
        """Solve and return the unknown vector.

        Note (v1.1 §13.1): under a differentiable backend this must be
        an implicit-function-theorem wrapper so the iteration process
        itself stays off the autograd graph.
        """
        ...
