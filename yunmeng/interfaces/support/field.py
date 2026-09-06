# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Field/data value types and minimal protocols.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from yunmeng.interfaces.types import ArrayLike, DeviceType, ElementType, VariableType

# ---------------------------------------------------
# region Value types
# ---------------------------------------------------


@dataclass
class FieldMeta:
    """Descriptor of a distributed field."""

    name: str
    vtype: VariableType = VariableType.SCALAR
    loc: ElementType = ElementType.CELL
    unit: str = ""
    dtype: str = "float64"
    btype: str = "numpy"  # backend identifier
    device: DeviceType = "cpu"


@dataclass(frozen=True)
class DataProduct:
    """A named, located data product published into the DataHub."""

    name: str
    loc: ElementType
    dtype: str = "float64"


# ---------------------------------------------------
# region Minimal protocols
# ---------------------------------------------------


@runtime_checkable
class IField(Protocol):
    """Minimal field surface used by interface signatures."""

    @property
    def meta(self) -> FieldMeta: ...

    @property
    def values(self) -> ArrayLike:
        """Backend array of values.
        In TRAIN mode may carry a graph."""
        ...

    @values.setter
    def values(self, v: ArrayLike) -> None: ...


@runtime_checkable
class ISample(Protocol):
    """A versioned data sample retrieved from the DataHub.

    TRAIN-mode contract: samples handed to callbacks must
    stay on the autograd graph — no detach, no stale-cache replay.
    """

    @property
    def value(self) -> ArrayLike: ...

    @property
    def time(self) -> float: ...

    @property
    def version(self) -> int: ...
