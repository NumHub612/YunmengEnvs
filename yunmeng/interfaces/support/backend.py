# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Backend protocol (ArrayNamespace) for the interfaces layer.

Design docs: v1.0 §4.2 (abstraction layer never names a concrete
backend), v1.1 §13.3 (algorithmic primitives area, Array-API naming
convention, on-demand backend specialization).

The protocol below is the COMPLETE vocabulary the operator layer may
use. Implementations (numpy/torch/...) live in numerics; adding a new
primitive here is a deliberate, reviewed act (v2.0 §14.4).
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from yunmeng.interfaces.types import ArrayLike, DeviceType


@runtime_checkable
class Backend(Protocol):
    """Array namespace + algorithmic primitives.

    Acquisition: operators never import numpy/torch directly; they
    dispatch via ``backend = get_backend(field.meta.btype)`` and use
    the members below (v1.3 §19 pattern).
    """

    # -- identity -----------------------------------

    @property
    def name(self) -> str:
        """Backend identifier, e.g. "numpy", "torch"."""
        ...

    @property
    def xp(self) -> object:
        """The raw array namespace module (escape hatch; prefer the
        typed primitives below in operator code)."""
        ...

    @property
    def differentiable(self) -> bool:
        """Whether arrays on this backend can carry an autograd graph."""
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
        """Detach-free host copy for I/O. Callers in TRAIN mode must not
        route gradient-relevant values through here (v1.0 §4.1 L3)."""
        ...

    # -- elementwise / selection --------------------

    def where(self, cond: ArrayLike, x: ArrayLike, y: ArrayLike) -> ArrayLike: ...

    def maximum(self, x: ArrayLike, y: ArrayLike) -> ArrayLike: ...

    def minimum(self, x: ArrayLike, y: ArrayLike) -> ArrayLike: ...

    def concatenate(self, arrays: Sequence[ArrayLike], axis: int = 0) -> ArrayLike: ...

    # -- algorithmic primitives (v1.1 §13.3) --------

    def matmul(self, a: ArrayLike, b: ArrayLike) -> ArrayLike: ...

    def einsum(self, equation: str, *operands: ArrayLike) -> ArrayLike: ...

    def scatter_add(
        self, target: ArrayLike, indices: ArrayLike, values: ArrayLike
    ) -> ArrayLike:
        """FEM/FVM assembly and mesh-GNN message passing primitive.
        numpy: np.add.at; torch: index_add_ (on-device, differentiable)."""
        ...

    def solve(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        """Dense differentiable solve only. Sparse/iterative paths belong
        to numerics.linalgs; under torch they must be wrapped via the
        implicit-function theorem (v1.1 §13.1), not raw AD through
        iterations."""
        ...
