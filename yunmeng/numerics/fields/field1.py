# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Field minmal implementations.
"""

from __future__ import annotations

from yunmeng.interfaces.supports import FieldMeta, IField, IGrid
from yunmeng.interfaces.types import ArrayLike

# ---------------------------------------------------
# region Field
# ---------------------------------------------------


class Field:
    """Minimal distributed field over a mesh location (flat layout)."""

    def __init__(self, meta: FieldMeta, values: ArrayLike):
        self._meta = meta
        self._values = values

    @property
    def meta(self) -> FieldMeta:
        return self._meta

    @property
    def values(self) -> ArrayLike:
        return self._values

    @values.setter
    def values(self, v: ArrayLike):
        self._values = v


# ---------------------------------------------------
# region GridField
# ---------------------------------------------------


class GridField(Field):
    """Field bound to a structured grid.

    Storage stays flat (n, *comp) — fully compatible with solver-side
    scatter-write and trajectory stacking. The multi-dim view is
    derived on demand; both directions are zero-copy views.
    """

    def __init__(self, meta: FieldMeta, values: ArrayLike, grid: IGrid):
        super().__init__(meta, values)
        self._grid = grid

    @property
    def grid(self) -> IGrid:
        return self._grid

    @property
    def values_nd(self) -> ArrayLike:
        """Multi-dim view (nx, ny[, nz], *comp) of the flat storage."""
        comp = self._meta.vtype.shape
        return self._values.reshape(*self._grid.shape, *comp)

    @values_nd.setter
    def values_nd(self, v: ArrayLike):
        comp = self._meta.vtype.shape
        self._values = v.reshape(-1, *comp) if comp else v.reshape(-1)
