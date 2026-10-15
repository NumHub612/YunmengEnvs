# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

DataHub protocol (minimal surface) for the interfaces layer.
"""

from __future__ import annotations

from typing import Protocol, List, runtime_checkable
from dataclasses import dataclass

from yunmeng.standard.supports.field import IField
from yunmeng.standard.types import RunMode, ElementType, ArrayLike

# ---------------------------------------------------
# region Components
# ---------------------------------------------------


@dataclass(frozen=True)
class Sample:
    """A versioned DataHub sample (graph-carrying under TRAIN)."""

    value: ArrayLike
    time: float
    version: int


@dataclass(frozen=True)
class DataProduct:
    """A structured, hashable dataflow declaration unit."""

    kind: str  # OperatorKinds value, the key namespace
    field: str  # target field name, e.g. "u"
    loc: ElementType = ElementType.CELL
    time_order: int = 0  # 0 = current; >0 = history
    qualifier: str = ""  # disambiguates same-kind variants

    @property
    def name(self) -> str:
        """Canonical cache/product key."""
        base = f"{self.kind}:{self.field}"
        return f"{base}@{self.qualifier}" if self.qualifier else base


@dataclass
class ComputedEntry:
    """A cached computed product with field versions."""

    product: str
    sample: Sample
    versions: dict[str, int]

    def is_stale(self, current: dict[str, int]) -> bool:
        """Check if the field versions match current."""
        return self.versions != current


# ---------------------------------------------------
# region IDataHub
# ---------------------------------------------------


@runtime_checkable
class IDataHub(Protocol):
    """Versioned in-run store for fields and operator products."""

    # -- mode ---------------------------------------

    @property
    def mode(self) -> RunMode: ...

    def set_mode(self, mode: RunMode):
        """Switch EVAL/TRAIN. Entering TRAIN clears the computed cache
        (autograd graphs must not be served from cache)."""
        ...

    # -- fields: register / field -------------------

    def register(self, field: IField):
        """Register a solution field (current state holder)."""
        ...

    def field(self, name: str) -> IField:
        """Current-state field by name."""
        ...

    def fields(self) -> list[str]:
        """Names of all registered solution fields."""
        ...

    # -- versions: touch / version ------------------

    def touch(self, name: str):
        """Bump a field's version after writing values; invalidates
        exactly the cached products that depend on it."""
        ...

    def version(self, name: str) -> int:
        """Current version stamp of a field."""
        ...

    # -- history line: commit / recall --------------

    def commit(self, field: IField, t: float):
        """Commit the field's current values as a new time level.
        Solver calls once per step after the solve."""
        ...

    def recall(self, name: str, order: int = 0) -> Sample:
        """Recall a committed field; order=0 is the most recent."""
        ...

    def depth(self, name: str) -> int:
        """Number of committed levels currently held for `name`."""
        ...

    # -- computed line: cache / fetch ---------------

    def cache(self, dp: DataProduct, sample: Sample, deps: list[str]):
        """Cache a derived product, stamping the current versions of
        its DECLARED source fields only. No-op in TRAIN mode."""
        ...

    def fetch(self, dp: DataProduct) -> Sample:
        """Fresh cached product, or None (always None in TRAIN).
        Stale entries (dependency version mismatch) are dropped."""
        ...

    # -- declaration-driven routing -----------------

    def resolve(self, dp: DataProduct) -> IField | Sample:
        """Route a declared DataProduct to its data line.
        Returns IField for current fields, Sample otherwise."""
        ...
