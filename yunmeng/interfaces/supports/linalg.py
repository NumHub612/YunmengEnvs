# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Linear-system protocol for the interfaces layer (minimal surface).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from yunmeng.interfaces.types import ArrayLike, DeviceType
from yunmeng.interfaces.supports.field import IField


@runtime_checkable
class IMatrix(Protocol):
    """Abstract coefficient matrix."""

    # -----------------------------------------------
    # region Constructions
    # -----------------------------------------------

    @classmethod
    def from_data(
        cls,
        values: ArrayLike,
        indices: ArrayLike = None,
        shape: tuple[int, int] = None,
        device: DeviceType = None,
    ) -> "IMatrix":
        """Create from dense data, or from (rows, cols) sparse indices."""
        raise NotImplementedError()

    @classmethod
    def from_coo(
        cls,
        shape: tuple[int, int],
        values: ArrayLike,
        rows: ArrayLike,
        cols: ArrayLike,
        device: DeviceType = None,
    ) -> "IMatrix":
        """Create from COO-format triplets."""
        raise NotImplementedError()

    @classmethod
    def identity(cls, size: int, device: DeviceType = None) -> "IMatrix":
        """Create identity matrix."""
        raise NotImplementedError()

    @classmethod
    def zeros(
        cls,
        shape: tuple[int, int],
        device: DeviceType = None,
    ) -> "IMatrix":
        """Create an all-zero matrix."""
        raise NotImplementedError()

    # -----------------------------------------------
    # region Properties
    # -----------------------------------------------
    @property
    def backend(self) -> str:
        """Backend tag of the underlying storage: "numpy" | "torch"."""
        raise NotImplementedError()

    @property
    def data(self) -> ArrayLike:
        """The raw backend-native storage."""
        raise NotImplementedError()

    @property
    def shape(self) -> tuple[int, int]:
        raise NotImplementedError()


@runtime_checkable
class ILinearEqs(Protocol):
    """Assembled linear system A·x = b, opaque to the solver layer."""

    @property
    def size(self) -> int:
        """Number of unknowns."""
        ...

    @property
    def matrix(self) -> IMatrix:
        """System matrix (backend array or backend sparse handle)."""
        ...

    @property
    def rhs(self) -> IField:
        """Right-hand side vector."""
        ...

    def solve(self, x0: ArrayLike = None) -> IField:
        """Solve and return the unknown vector."""
        ...
