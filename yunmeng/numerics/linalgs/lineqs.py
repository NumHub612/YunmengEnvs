# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Linear equations container and interface.

This module defines LinearEqs — the high-level interface for linear systems.
Actual solving algorithms live in linalg_solvers.py and are selected
automatically based on matrix properties, or manually via algorithm= keyword.

Usage:
    eqs = LinearEqs(matrix, rhs_field)
    solution = eqs.solve()                    # auto-select solver
    solution = eqs.solve(algorithm="direct")  # force direct solve
    solution = eqs.solve(algorithm="cg")      # force CG
"""

from __future__ import annotations

from yunmeng.interfaces.supports import ILinearEqs, IMatrix, IField
from yunmeng.numerics.fields import Field
from yunmeng.numerics.linalgs.engines import (
    select_engine,
    LinearEngine,
)


class LinearEqs:
    """Linear equations container: A @ x = b.

    Responsibilities:
    - Hold matrix (A) and RHS field (b)
    - Algebraic operations (+, -, +=, -=)
    """

    def __init__(self, mat: IMatrix, rhs: IField):
        if mat.shape[0] != mat.shape[1] or mat.shape[0] != rhs.size:
            raise ValueError(
                f"The matrix is not square: {mat.shape}, or not "
                f"compatible with the rhs size: {rhs.size}."
            )
        if mat.backend != rhs.meta.btype:
            raise ValueError(
                f"Backend mismatch: matrix is {mat.backend!r}, rhs "
                f"field is {rhs.meta.btype!r}."
            )
        self._mat = mat
        self._rhs = rhs
        self._size = rhs.size

    # -----------------------------------------------
    # region properties
    # -----------------------------------------------

    @property
    def size(self) -> int:
        return self._size

    @property
    def matrix(self) -> IMatrix:
        return self._mat

    @property
    def rhs(self) -> IField:
        return self._rhs

    # -----------------------------------------------
    # region operations
    # -----------------------------------------------

    def __add__(self, other: "LinearEqs") -> "LinearEqs":
        self._check_compatible(other)
        return LinearEqs(self._mat + other.matrix, self._rhs + other.rhs)

    def __iadd__(self, other: "LinearEqs") -> "LinearEqs":
        self._check_compatible(other)
        self._mat = self._mat + other.matrix
        self._rhs = self._rhs + other.rhs
        return self

    def __sub__(self, other: "LinearEqs") -> "LinearEqs":
        self._check_compatible(other)
        return LinearEqs(self._mat - other.matrix, self._rhs - other.rhs)

    def __isub__(self, other: "LinearEqs") -> "LinearEqs":
        self._check_compatible(other)
        self._mat = self._mat - other.matrix
        self._rhs = self._rhs - other.rhs
        return self

    def _check_compatible(self, other: "LinearEqs"):
        if not isinstance(other, LinearEqs):
            raise ValueError(f"Require LinearEqs type, got {type(other)}.")
        if other.size != self.size:
            raise ValueError(f"LinearEqs size mismatch: {self.size} vs {other.size}.")
        if other.rhs.meta.vtype != self.rhs.meta.vtype:
            raise ValueError(
                f"LinearEqs vtype mismatch: {self.rhs.meta.vtype} vs "
                f"{other.rhs.meta.vtype}."
            )

    def reset_rhs(self, rhs: IField):
        """Reset the right-hand side field (e.g. between time steps)."""
        if rhs.size != self.size:
            raise ValueError(f"Invalid rhs size: {rhs.size} vs {self.size}.")
        if rhs.meta.vtype != self._rhs.meta.vtype:
            raise ValueError(
                f"Invalid rhs vtype: {rhs.meta.vtype} vs {self._rhs.meta.vtype}."
            )
        self._rhs = rhs

    # -----------------------------------------------
    # region solve
    # -----------------------------------------------

    def solve(self, algorithm: str = "auto") -> IField:
        """Solve the linear system.

        Args:
            algorithm: "auto" | "direct" | "cg" (torch) | "lu" (numpy).

        Returns:
            Solution Field with the same meta as the rhs. Values keep
            the rhs backend and — for differentiable engines — the
            autograd graph (no detach, no host round-trip).
        """
        engine = select_engine(self._mat, algorithm=algorithm)
        result = self._solve_batched(engine)

        n = self._size
        comp = self._rhs.meta.vtype.shape  # () | (3,) | (3, 3)
        values = result.reshape(n, *comp) if comp else result.reshape(n)
        return Field(self._rhs.meta, values)

    def _solve_batched(self, engine: LinearEngine):
        """Solve with rhs normalized to (N,) or (N, K)."""
        rhs = self._rhs.values
        if getattr(rhs, "requires_grad", False) and not engine.differentiable:
            raise RuntimeError(
                f"Engine {engine.name!r} is EVAL-only (breaks the "
                "autograd graph), but the rhs requires gradients. "
                "Use a differentiable engine (torch 'direct'/'cg') or "
                "run in EVAL mode."
            )
        rhs_2d = rhs.reshape(self._size, -1) if rhs.ndim > 1 else rhs
        result = engine.solve(self._mat.data, rhs_2d)
        if self._rhs.values.ndim > 1 and result.ndim == 1:
            result = result.reshape(self._size, 1)
        return result
