# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

DataHub protocol (minimal surface) for the interfaces layer.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable
from dataclasses import dataclass

from yunmeng.interfaces.supports.field import IField
from yunmeng.interfaces.types import RunMode, ElementType, ArrayLike


@dataclass(frozen=True)
class Sample:
    """A versioned DataHub sample (graph-carrying under TRAIN)."""

    value: ArrayLike
    time: float
    version: int


@dataclass(frozen=True)
class DataProduct:
    """A named, located data product published into the DataHub."""

    name: str
    loc: ElementType
    dtype: str = "float64"


@runtime_checkable
class IDataHub(Protocol):
    """Versioned field/sample store shared within a run."""

    @property
    def mode(self) -> RunMode: ...

    def get_field(self, name: str) -> IField: ...

    def get(self, product: str, time_order: int = 0) -> Sample:
        """Latest (or time-shifted) sample of a data product."""
        ...

    def publish(self, product: str, value: IField, t: float):
        """Publish an operator product.

        TRAIN mode: no detach, no stale-cache replay; the published
        value keeps its graph. EVAL mode: caching/aliasing allowed.
        """
        ...

    def products(self) -> Sequence[DataProduct]: ...
