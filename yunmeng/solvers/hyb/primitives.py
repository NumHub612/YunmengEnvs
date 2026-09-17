# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Numerics primitives satisfying the support-layer protocols:
  - Field        (IField)
  - DataHub      (IDataHub, versioned samples, TRAIN keeps graph)
  - LinearEqs    (ILinearEqs)
  - UniformMesh1D (IMesh + topo/geom assistants + regions)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from yunmeng.interfaces.supports.field import DataProduct, FieldMeta, IField
from yunmeng.interfaces.types import (
    ArrayLike,
    ElementType,
    MeshDimension,
    RunMode,
    VariableType,
)

# ---------------------------------------------------
# region Field / Sample
# ---------------------------------------------------


class Field:
    """Minimal distributed field over a mesh location."""

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
    def values(self, v: ArrayLike) -> None:
        self._values = v


@dataclass
class Sample:
    """A versioned DataHub sample (graph-carrying under TRAIN)."""

    _value: ArrayLike
    _time: float
    _version: int

    @property
    def value(self) -> ArrayLike:
        return self._value

    @property
    def time(self) -> float:
        return self._time

    @property
    def version(self) -> int:
        return self._version


# ---------------------------------------------------
# region DataHub
# ---------------------------------------------------


class DataHub:
    """Versioned in-run store for fields and operator products.

    Contract: under TRAIN no detach, no stale-cache replay — publish()
    keeps the incoming array object (and its autograd graph) as-is.
    """

    def __init__(self, mode: RunMode = RunMode.EVAL):
        self._mode = mode
        self._fields: dict[str, Field] = {}
        self._products: dict[str, tuple[DataProduct, list[Sample]]] = {}

    @property
    def mode(self) -> RunMode:
        return self._mode

    def set_mode(self, mode: RunMode) -> None:
        self._mode = mode

    # -- fields -------------------------------------

    def register_field(self, field: Field) -> None:
        self._fields[field.meta.name] = field

    def get_field(self, name: str) -> IField:
        if name not in self._fields:
            raise KeyError(f"field {name!r} not registered in DataHub")
        return self._fields[name]

    def field_names(self) -> list[str]:
        return list(self._fields.keys())

    # -- products -----------------------------------

    def publish(self, product: str, value: IField, t: float):
        dp, samples = self._products.get(product, (None, []))
        if dp is None:
            dp = DataProduct(name=product, loc=value.meta.loc, dtype=value.meta.dtype)
        samples.append(Sample(value.values, t, len(samples)))
        if self._mode != RunMode.TRAIN:
            samples = samples[-8:]  # EVAL: keep a short ring for time_order
        self._products[product] = (dp, samples)

    def get(self, product: str, time_order: int = 0) -> Sample:
        if product not in self._products:
            raise KeyError(f"product {product!r} not published")
        samples = self._products[product][1]
        idx = -1 - time_order
        if len(samples) < -idx:
            raise IndexError(f"product {product!r} has {len(samples)} samples")
        return samples[idx]

    def products(self) -> Sequence[DataProduct]:
        return [dp for dp, _ in self._products.values()]

    def clear_products(self) -> None:
        self._products.clear()


# ---------------------------------------------------
# region LinearEqs
# ---------------------------------------------------


class LinearEqs:
    """Dense assembled system A x = b."""

    def __init__(self, matrix: ArrayLike, rhs: ArrayLike, backend):
        self._a = matrix
        self._b = rhs
        self._backend = backend

    @property
    def size(self) -> int:
        return int(self._a.shape[0])

    @property
    def matrix(self) -> ArrayLike:
        return self._a

    @property
    def rhs(self) -> ArrayLike:
        return self._b

    def solve(self, x0: ArrayLike | None = None) -> ArrayLike:
        return self._backend.solve(self._a, self._b)


# ---------------------------------------------------
# region UniformMesh1D
# ---------------------------------------------------


class Region:
    """A named set of mesh elements (boundary or interior)."""

    def __init__(self, rid: str, loc: ElementType, element_ids: ArrayLike):
        self._id = rid
        self._loc = loc
        self._ids = np.asarray(element_ids, dtype="int64")

    @property
    def id(self) -> str:
        return self._id

    @property
    def loc(self) -> ElementType:
        return self._loc

    @property
    def element_ids(self) -> ArrayLike:
        return self._ids


class _Topo1D:
    """Topology queries for a uniform 1D cell mesh."""

    def __init__(self, n_cells: int):
        self._n = n_cells

    def neighbors(self, element: int, loc: ElementType) -> Sequence[int]:
        out = []
        if element > 0:
            out.append(element - 1)
        if element < self._n - 1:
            out.append(element + 1)
        return out

    def boundary_faces(self) -> ArrayLike:
        # Canonical numbering: face i sits between cell i-1 and cell i;
        # boundary faces are 0 (left) and n (right).
        return np.array([0, self._n], dtype="int64")

    def interior_faces(self) -> ArrayLike:
        return np.arange(1, self._n, dtype="int64")


class _Geom1D:
    def __init__(self, x0: float, dx: float, n_cells: int):
        self._x0 = x0
        self._dx = dx
        self._n = n_cells

    def centroid(self, element: int, loc: ElementType) -> ArrayLike:
        if loc == ElementType.CELL:
            return np.array([self._x0 + (element + 0.5) * self._dx, 0.0, 0.0])
        return np.array([self._x0 + element * self._dx, 0.0, 0.0])

    def face_normal(self, face: int) -> ArrayLike:
        return np.array([1.0, 0.0, 0.0])

    def face_area(self, face: int) -> ArrayLike:
        return np.array(1.0)

    def cell_volume(self, cell: int) -> ArrayLike:
        return np.array(self._dx)


class UniformMesh1D:
    """Uniform 1D mesh of n cells on [x0, x0 + n*dx].

    Regions: "left" (cell 0 side), "right" (cell n-1 side),
    "interior" (cells 1..n-2), "all".
    """

    def __init__(self, n_cells: int, x0: float = 0.0, length: float = 1.0):
        self._n = int(n_cells)
        self._x0 = float(x0)
        self._dx = float(length) / self._n
        self._version = 0
        self._regions = {
            "left": Region("left", ElementType.CELL, [0]),
            "right": Region("right", ElementType.CELL, [self._n - 1]),
            "interior": Region("interior", ElementType.CELL, np.arange(1, self._n - 1)),
            "all": Region("all", ElementType.CELL, np.arange(self._n)),
        }

    @property
    def dimension(self) -> MeshDimension:
        return MeshDimension.D1

    @property
    def version(self) -> int:
        return self._version

    @property
    def n_cells(self) -> int:
        return self._n

    @property
    def dx(self) -> float:
        return self._dx

    @property
    def cell_centers(self) -> ArrayLike:
        return self._x0 + (np.arange(self._n) + 0.5) * self._dx

    def element_count(self, loc: ElementType) -> int:
        return {
            ElementType.CELL: self._n,
            ElementType.NODE: self._n + 1,
            ElementType.FACE: self._n + 1,
        }.get(loc, 0)

    def get_topo_assistant(self) -> _Topo1D:
        return _Topo1D(self._n)

    def get_geom_assistant(self) -> _Geom1D:
        return _Geom1D(self._x0, self._dx, self._n)

    def get_region(self, region_id: str) -> Region:
        if region_id not in self._regions:
            raise KeyError(f"unknown region {region_id!r}")
        return self._regions[region_id]

    def regions(self) -> Sequence[Region]:
        return list(self._regions.values())
