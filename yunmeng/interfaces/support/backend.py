# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Backend protocol (ArrayNamespace) for the interfaces layer.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from yunmeng.interfaces.types import ArrayLike, DeviceType


@runtime_checkable
class Backend(Protocol):
    """Array namespace + algorithmic primitives."""

    # -- identity -----------------------------------

    @property
    def name(self) -> str:
        """Backend identifier, e.g. "numpy", "torch"."""
        ...

    @property
    def xp(self) -> object:
        """The raw array namespace module."""
        ...

    @property
    def differentiable(self) -> bool:
        """Whether arrays on this backend can carry
        an autograd graph."""
        ...

    # -- construction / conversion ------------------

    def asarray(
        self,
        data: object,
        dtype: str | None = None,
        device: DeviceType | None = None,
    ) -> ArrayLike: ...

    def zeros(
        self,
        shape: Sequence[int],
        dtype: str | None = None,
        device: DeviceType | None = None,
    ) -> ArrayLike: ...

    def zeros_like(self, a: ArrayLike) -> ArrayLike: ...

    def to_device(self, a: ArrayLike, device: DeviceType) -> ArrayLike: ...

    def to_host(self, a: ArrayLike) -> ArrayLike:
        """Detach-free host copy for I/O."""
        ...

    # -- elementwise / selection --------------------

    def where(self, cond: ArrayLike, x: ArrayLike, y: ArrayLike) -> ArrayLike: ...

    def maximum(self, x: ArrayLike, y: ArrayLike) -> ArrayLike: ...

    def minimum(self, x: ArrayLike, y: ArrayLike) -> ArrayLike: ...

    def concatenate(self, arrays: Sequence[ArrayLike], axis: int = 0) -> ArrayLike: ...

    # -- algorithmic primitives ---------------------

    def matmul(self, a: ArrayLike, b: ArrayLike) -> ArrayLike: ...

    def einsum(self, equation: str, *operands: ArrayLike) -> ArrayLike: ...

    def scatter_add(
        self, target: ArrayLike, indices: ArrayLike, values: ArrayLike
    ) -> ArrayLike: ...

    def solve(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        """Dense differentiable solve only. Sparse/iterative paths belong
        to numerics.linalgs; under torch they must be wrapped via the
        implicit-function theorem, not raw AD through iterations."""
        ...
