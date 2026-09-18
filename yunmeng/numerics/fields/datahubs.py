# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

DataHub implementation — merged from the two legacy versions.

Centralizes solver-side data management across three concerns:
    1. Time history of PRODUCTS (multi-step schemes retrieve
       time-shifted samples via get(product, time_order)).
       NOTE: fields themselves hold only the CURRENT state
       (get_field); to keep a solution field's history,
       the solver publishes it as a product each step.
    2. Cross-operator reuse of computed products (EVAL-only cache,
       keyed by plain product strings, e.g. "laplacian:u").
    3. Version tracking with cache invalidation: the solver calls
       touch(field) after writing field values; cached entries
       whose recorded version stamps no longer match are dropped.

=====================================================
Usage sketch
=====================================================
    hub = DataHub(mode=RunMode.EVAL, levels=3, grid=grid)

    # 1) register the solver solution field (auto-wrapped as GridField)
    hub.register_field(Field(meta, backend.zeros((n,))))
    u = hub.get_field("u")
    u_nd = u.values_nd

    # 2) solver publishes each step -> time history for multi-step
    #    schemes; time_order=1 retrieves the previous level
    hub.publish("u", u_field, t)
    u_prev = hub.get("u", time_order=1).value

    # 3) operator A computes a gradient and caches the product
    hub.put_computed("grad:u", Sample(g, t, version=0))

    # 4) operator B reuses it (None in TRAIN or when stale)
    cached = hub.get_computed("grad:u")

    # 5) solver writes new values -> touch invalidates stale cache
    u.values = u_new
    hub.touch("u")
    assert hub.get_computed("grad:u") is None

    # 6) TRAIN mode: computed cache is bypassed, cleared on entry;
    #    publish keeps the array object (and its autograd graph).
    hub.set_mode(RunMode.TRAIN)
"""

from __future__ import annotations

from dataclasses import dataclass

from yunmeng.interfaces.supports import (
    DataProduct,
    Sample,
    IField,
    IMesh,
    IDataHub,
)
from yunmeng.interfaces.types import RunMode
from yunmeng.numerics.fields.field1 import Field, GridField

# ---------------------------------------------------
# region ComputedEntry
# ---------------------------------------------------


@dataclass
class ComputedEntry:
    """A cached computed product with source-field version stamps."""

    product: str
    sample: Sample
    versions: dict[str, int]

    def is_stale(self, current: dict[str, int]) -> bool:
        """Check if the source-field versions match  current ones."""
        return self.versions != current


# ---------------------------------------------------
# region DataHub
# ---------------------------------------------------


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
        self._products: dict[str, tuple[DataProduct, list[Sample]]] = {}
        self._versions: dict[str, int] = {}
        self._cache: dict[str, ComputedEntry] = {}

    # -- mode -----------------------------------------

    @property
    def mode(self) -> RunMode:
        return self._mode

    def set_mode(self, mode: RunMode):
        self._mode = mode
        if mode == RunMode.TRAIN:
            # computed cache must be bypassed, not reused
            self._cache.clear()

    # -- fields ---------------------------------------

    def register_field(self, field: Field):
        """Register a field with the DataHub."""
        if self._mesh is not None and not isinstance(field, GridField):
            # auto-wrap scalar fields as grid fields
            field = GridField(field.meta, field.values, self._mesh)
        self._fields[field.meta.name] = field
        self._versions[field.meta.name] = 0  # reset version stamp

    def get_field(self, name: str) -> IField:
        if name not in self._fields:
            raise KeyError(f"field {name!r} not registered in DataHub")
        return self._fields[name]

    def field_names(self) -> list[str]:
        return list(self._fields.keys())

    def touch(self, name: str):
        """Bump a field's version (solver calls after writing values);
        cascade-invalidates dependent cached products."""
        self._versions[name] = self._versions.get(name, 0) + 1  # bump version
        victims = [
            k
            for k, e in self._cache.items()
            if any(self._versions.get(f, 0) != v for f, v in e.versions.items())
        ]  # cascade-invalidates
        for k in victims:
            # cache entry is stale, remove it
            del self._cache[k]

    # -- products (history, EVAL only) --------------

    def publish(self, product: str, value: IField, t: float):
        dp, samples = self._products.get(product, (None, []))
        if dp is None:
            dp = DataProduct(name=product, loc=value.meta.loc, dtype=value.meta.dtype)
        samples.append(Sample(value.values, t, len(samples)))

        if self._mode != RunMode.TRAIN:  # only EVAL mode keeps history
            samples = samples[-self._levels :]  # EVAL: ring for time_order
        self._products[product] = (dp, samples)

    def get(self, product: str, time_order: int = 0) -> Sample:
        if product not in self._products:
            raise KeyError(f"product {product!r} not published")
        samples = self._products[product][1]

        idx = -1 - time_order  # time_order=0 is the most recent
        if len(samples) < -idx:
            raise IndexError(f"product {product!r} has {len(samples)} samples")
        return samples[idx]

    def products(self) -> list[DataProduct]:
        return [dp for dp, _ in self._products.values()]

    def clear_products(self):
        self._products.clear()
        self._cache.clear()

    # -- computed-entry (cache, EVAL only) ----------

    def get_computed(self, product: str) -> Sample:
        """Return a fresh cached product, or None. Always None in TRAIN."""
        if self._mode == RunMode.TRAIN:
            return None

        entry = self._cache.get(product)
        if entry is None:
            return None
        if entry.is_stale(self._versions):
            # cache entry is stale, remove it
            del self._cache[product]
            return None
        return entry.sample

    def put_computed(self, product: str, sample: Sample):
        """Cache a computed product. No-op in TRAIN."""
        if self._mode == RunMode.TRAIN:
            return
        self._cache[product] = ComputedEntry(
            product=product, sample=sample, versions=dict(self._versions)
        )

    def clear_cache(self):
        self._cache.clear()
