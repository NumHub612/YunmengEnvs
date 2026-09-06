# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Linear-system protocol for the interfaces layer (minimal surface).
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
        """Solve and return the unknown vector."""
        ...
