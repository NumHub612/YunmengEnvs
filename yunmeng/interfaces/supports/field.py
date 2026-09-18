# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Field/data value types and minimal protocols.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from yunmeng.interfaces.types import (
    ArrayLike,
    DeviceType,
    ElementType,
    VariableType,
)


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


@runtime_checkable
class IField(Protocol):
    """Minimal field surface used by interface signatures."""

    @property
    def meta(self) -> FieldMeta: ...

    @property
    def values(self) -> ArrayLike:
        """Backend array of values."""
        ...

    @values.setter
    def values(self, v: ArrayLike): ...
