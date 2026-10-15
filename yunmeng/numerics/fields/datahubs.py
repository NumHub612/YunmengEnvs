# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

DataHub implementation with declaration-driven routing.

Two data lines:
    1. PRODUCTS line  — solver-published time history of solution fields
       (ring buffer, serves multi-step schemes via time_order).
    2. COMPUTED line  — same-timestep cross-operator derived products
       (EVAL-only cache, invalidated by dependency-field versions).

Operator declarations (DataProduct) ARE the routing table:
    - time_order > 0        -> products history (hub.get)
    - kind == MATH_VALUE    -> current solution field (hub.get_field)
    - otherwise             -> computed cache (hub.get_computed),
                               miss -> schedule the declaring producer.

=====================================================
Usage sketch
=====================================================
    hub = DataHub(mode=RunMode.EVAL, levels=3, grid=grid)

    # 1) register the solver solution field (auto-wrapped as GridField)
    hub.register(Field(meta, backend.zeros((n,))))
    u = hub.field("u")
    u_nd = u.values_nd

    # 2) solver publishes each step -> time history for multi-step
    #    schemes; time_order=1 retrieves the previous level
    hub.commit("u", u_field, t)
    u_prev = hub.recall("u", time_order=1).value

    # 3) operator A computes a gradient and caches the product
    hub.cache("grad:u", Sample(g, t, version=0))

    # 4) operator B reuses it (None in TRAIN or when stale)
    cached = hub.fetch("grad:u")

    # 5) solver writes new values -> touch invalidates stale cache
    u.values = u_new
    hub.touch("u")
    assert hub.fetch("grad:u") is None

    # 6) TRAIN mode: computed cache is bypassed, cleared on entry;
    #    publish keeps the array object (and its autograd graph).
    hub.set_mode(RunMode.TRAIN)
"""

from __future__ import annotations

from yunmeng.standard.solver import OperatorKinds
from yunmeng.standard.supports import (
    DataProduct,
    Sample,
    ComputedEntry,
    IField,
    IMesh,
    IDataHub,
)
from yunmeng.standard.types import RunMode
from yunmeng.numerics.fields.field1 import Field, GridField


class DataHub:
    """Versioned in-run store for fields and operator products."""

    def __init__(
        self,
        mode: RunMode = RunMode.EVAL,
        levels: int = 3,
        mesh: IMesh = None,
    ):
        self._mode = mode
        self._levels = max(1, int(levels))
        self._mesh = mesh
        self._fields: dict[str, Field] = {}
        self._versions: dict[str, int] = {}
        self._history: dict[str, list[Sample]] = {}
        self._cache: dict[str, ComputedEntry] = {}

    # -- mode ---------------------------------------

    @property
    def mode(self) -> RunMode:
        return self._mode

    def set_mode(self, mode: RunMode):
        self._mode = mode
        if mode == RunMode.TRAIN:
            self._cache.clear()

    # -- fields: register / field -------------------

    def register(self, field: Field):
        """Register a solution field (current state holder)."""
        if self._mesh and not isinstance(field, GridField):
            field = GridField(field.meta, field.values, self._mesh)
        self._fields[field.meta.name] = field
        self._versions[field.meta.name] = 0
        self._history.setdefault(field.meta.name, [])

    def field(self, name: str) -> IField:
        """Current-state field by name."""
        if name not in self._fields:
            raise KeyError(f"field {name!r} not registered")
        return self._fields[name]

    def fields(self) -> list[str]:
        """Names of all registered solution fields."""
        return list(self._fields.keys())

    def has_field(self, name: str) -> bool:
        """Whether a field is registered."""
        return name in self._fields

    # -- versions: touch / version ------------------

    def touch(self, name: str):
        """Bump a field's version after writing values."""
        self._versions[name] = self._versions.get(name, 0) + 1
        victims = [k for k, e in self._cache.items() if e.is_stale(self._versions)]
        for k in victims:
            del self._cache[k]

    def version(self, name: str) -> int:
        """Current version of a field."""
        return self._versions.get(name, 0)

    # -- history line: commit / recall --------------

    def commit(self, field: IField, t: float):
        """Commit the field's current values as a new time level."""
        name = field.meta.name
        if name not in self._fields:
            raise KeyError(f"field {name!r} not registered")
        ring = self._history.setdefault(name, [])
        ring.append(Sample(field.values, t, self._versions[name]))
        if self._mode != RunMode.TRAIN:
            del ring[: -self._levels]  # ring buffer

    def recall(self, name: str, order: int = 0) -> Sample:
        """Recall a committed field; order=0 is the most recent."""
        ring = self._history.get(name)
        if not ring:
            raise KeyError(f"field {name!r} has no committed levels")
        idx = -1 - order
        if len(ring) < -idx:
            raise IndexError(
                f"field {name!r} holds {len(ring)} levels, order={order} requested"
            )
        return ring[idx]

    def depth(self, name: str) -> int:
        """Number of committed levels currently held for `name`."""
        return len(self._history.get(name, ()))

    def clear_history(self, name: str):
        """Clear the committed history of a field."""
        self._history[name] = []

    # -- computed line: cache / fetch ---------------

    def cache(self, dp: DataProduct, sample: Sample, deps: list[str]):
        """Cache a derived product, stamping the current versions of
        its DECLARED source fields only. No-op in TRAIN mode."""
        if self._mode == RunMode.TRAIN:
            return
        stamps = {f: self._versions.get(f, 0) for f in deps}
        self._cache[dp.name] = ComputedEntry(sample=sample, deps=stamps)

    def fetch(self, dp: DataProduct) -> Sample:
        """Fresh cached product, or None (always None in TRAIN)."""
        if self._mode == RunMode.TRAIN:
            return None
        entry = self._cache.get(dp.name)
        if entry is None:
            return None
        if entry.is_stale(self._versions):
            del self._cache[dp.name]
            return None
        return entry.sample

    def clear_cache(self):
        """Clear the computed cache."""
        self._cache.clear()

    # -- declaration-driven routing -----------------

    def resolve(self, dp: DataProduct) -> IField | Sample:
        """Route a declared DataProduct to its data line.
        Returns IField for current fields, Sample otherwise."""
        if dp.time_order > 0:
            return self.recall(dp.field, dp.time_order)
        if dp.kind == OperatorKinds.MATH_VALUE or self.has_field(dp.field):
            return self.field(dp.field)
        sample = self.fetch(dp)
        if sample is None:
            raise KeyError(
                f"computed product {dp.name!r} unavailable: "
                "no fresh cache entry (producer not scheduled?)"
            )
        return sample
