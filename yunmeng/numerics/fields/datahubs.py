# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Centralizes solver-side data management across three concerns:
    1. Time history of fields (multi-step schemes).
    2. Cross-operator reuse of computed products (cache).
    3. Version tracking with cascade cache invalidation.

=====================================================
Usage sketch
=====================================================
    hub = DataHub(fields=["u", "v"], levels=3, mode=RunMode.EVAL)

    # 1) push a time-stamped snapshot
    hub.push("u", Sample(t, Field(u_tensor)), ElementType.CELL)

    # 2) operator A computes a gradient and caches the product
    g = A(hub.latest("u", ElementType.CELL).data)
    hub.put_computed(
        DataProduct.for_producer("GradOp", "u", ElementType.CELL, otype="grad"),
        Sample(t, Field(g)),
    )

    # 3) operator B reuses it via a wildcard query (namespace="*")
    cached = hub.get_computed(
        DataProduct.for_query("u", ElementType.CELL, otype="grad")
    )

    # 4) pushing a new "u" bumps its version -> the cached gradient is
    #    cascade-invalidated automatically
    hub.push("u", Sample(t2, Field(u_new)), ElementType.CELL)
    assert hub.get_computed(...) is None

    # 5) TRAIN mode: get/put_computed are bypassed, so all data flows
    #    through the mode-agnostic *history* API (push/latest/field)
    #    or as plain return values between operators.
    hub.set_mode(RunMode.TRAIN)
    u = hub.latest("u", ElementType.CELL).data.tensor

"""

from __future__ import annotations
from dataclasses import dataclass, field

from yunmeng.numerics.enums import ElementType, RunMode
from yunmeng.numerics.fields import Field

# ---------------------------------------------------
# region Sample
# ---------------------------------------------------


@dataclass(slots=True)
class Sample:
    """A time-stamped field snapshot. Mutable — can be updated in-place."""

    timestamp: float
    data: Field

    def detach(self) -> Sample:
        """Return new Sample with data detached from computation graph."""
        return Sample(self.timestamp, self.data.detach())


# ---------------------------------------------------
# region DataProduct
# ---------------------------------------------------


@dataclass(frozen=True)
class DataProduct:
    """Identifies a computed product that can be reused by other operators.

    Factory methods (preferred over raw constructor):
        - for_producer(): production key with namespace=operator_name
        - for_query(): query key with namespace="*" (wildcard) by default

    Wildcard convention:
        namespace == "*"  =>  query matches ANY namespace.
    """

    otype: str  # "grad", "div", "lap", "rhs", ...
    fname: str  # which field this product is derived from
    etype: ElementType  # where the product lives
    namespace: str = "default"

    def __str__(self) -> str:
        return f"{self.otype}_{self.fname}_{self.etype.name}_{self.namespace}"

    @classmethod
    def for_producer(
        cls,
        operator_name: str,
        field_name: str,
        etype: ElementType,
        otype: str = "unknown",
    ) -> "DataProduct":
        """Create a production key."""
        return cls(otype, field_name, etype, operator_name)

    @classmethod
    def for_query(
        cls,
        field_name: str,
        etype: ElementType,
        otype: str,
        namespace: str = "*",
    ) -> "DataProduct":
        """Create a query key for cache lookup."""
        return cls(otype, field_name, etype, namespace)


# ---------------------------------------------------------------------------
# region ComputedEntry
# ---------------------------------------------------------------------------


@dataclass
class ComputedEntry:
    """A computed field cached for cross-operator reuse.

    Fields:
        product: What this entry represents.
        sample: The computed data + timestamp.
        versions: {field_name: version} at compute time.
        depends: Upstream DataProducts can be reused.
    """

    product: DataProduct
    sample: Sample
    versions: dict[str, int] = field(default_factory=dict)
    depends: list[DataProduct] = field(default_factory=list)

    def is_stale(self, current_versions: dict[str, int]) -> bool:
        """Check if any source field has changed."""
        return self.versions != current_versions

    def depends_on_field(self, field_name: str) -> bool:
        """Check if this entry depends on field_name."""
        return (
            self.product.fname == field_name
            or field_name in self.versions
            or any(dep.fname == field_name for dep in self.depends)
        )


# ---------------------------------------------------------------------------
# region TensorHistory
# ---------------------------------------------------------------------------


class TensorHistory:
    """Stores time history as Sample references, preserving computation graphs.

    Only Sample *references* are shifted on push (O(levels), no Field copy).
    If a Field's underlying data is a torch tensor with requires_grad, the
    computation graph connection is preserved across time levels.
    """

    def __init__(self, name: str, levels: int, shape_hint: tuple = None):
        self._name = name
        self._max_levels = levels
        self._shape_hint = shape_hint
        self._version = 0

        # Store Sample references (not stacked tensors)
        # [newest, t-1, t-2, ...]
        self._history: list[Sample] = [None] * levels

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> int:
        return self._version

    def push(self, sample: Sample):
        """Push new sample — shifts history, drops oldest."""
        if not isinstance(sample, Sample):
            raise TypeError(f"Expected Sample, got {type(sample)}")

        # Shift: [newest, t-1, t-2] -> [new_sample, newest, t-1]
        for i in range(self._max_levels - 1, 0, -1):
            self._history[i] = self._history[i - 1]
        self._history[0] = sample
        self._version += 1

    def latest(self) -> Sample:
        """Get the most recent sample (level=0)."""
        return self._history[0]

    def at(self, level: int = 0) -> Sample:
        """Get sample at time level (0=current, 1=previous, ...)."""
        if level < 0 or level >= self._max_levels:
            raise IndexError(f"Level {level} out of range [0, {self._max_levels})")
        return self._history[level]

    def at_time(self, t: float) -> Sample:
        """Find sample closest to given physical time."""
        best, best_dt = None, float("inf")
        for s in self._history:
            if s is None:
                continue
            dt = abs(s.timestamp - t)
            if dt < best_dt:
                best, best_dt = s, dt
        return best

    def samples(self, levels: int = None) -> list[Sample]:
        """Return valid samples [newest ... oldest] up to `levels`."""
        n = levels or self._max_levels
        return [self._history[i] for i in range(n) if self._history[i] is not None]

    def clear(self):
        self._history = [None] * self._max_levels
        self._version = 0

    def __len__(self) -> int:
        return sum(1 for s in self._history if s is not None)

    def __repr__(self) -> str:
        filled = len(self)
        return (
            f"TensorHistory({self._name}, levels={self._max_levels}, filled={filled})"
        )


# ---------------------------------------------------------------------------
# region DataHub
# ---------------------------------------------------------------------------


class DataHub:
    """Central data management for a solver.

    Three subsystems:
        1. Time history  — TensorHistory per (field, loc) for multi-step schemes
        2. Computed cache — operator products for cross-operator reuse
        3. Version tracking — automatic cascade cache invalidation

    Run mode:
        - EVAL (default): computed-product cache enabled; detach-style
          optimizations are allowed by consumers.
        - TRAIN: computed-product cache bypassed (get/put become no-ops),
          so every operator evaluation stays on the live autograd graph
          during unrolled backpropagation.
    """

    def __init__(
        self,
        fields: list[str],
        levels: int,
        mode: RunMode = RunMode.EVAL,
    ):
        self._fields = list(set(fields))
        self._levels = levels
        self._mode = mode

        # Time history: {field_name}_{loc.name} -> TensorHistory
        self._history: dict[str, TensorHistory] = {}
        for fname in self._fields:
            for loc in ElementType:
                key = self._key(fname, loc)
                self._history[key] = TensorHistory(key, levels)

        # Computed cache: str(DataProduct) -> ComputedEntry
        self._cache: dict[str, ComputedEntry] = {}

        # Version tracking: {field_name} -> version counter
        self._versions: dict[str, int] = {f: 0 for f in self._fields}

    # -- mode -----------------------------------------

    @property
    def mode(self) -> RunMode:
        """Current run mode."""
        return self._mode

    def set_mode(self, mode: RunMode):
        self._mode = mode

    # -- key helpers --------------------------------

    @staticmethod
    def _key(name: str, etype: ElementType) -> str:
        return f"{name}_{etype.name}"

    # -- time history management --------------------

    def push(self, name: str, sample: Sample, etype: ElementType):
        """Push a field snapshot into time history.

        Args:
            name: field name
            sample: the Sample to store
            etype: element type where the field lives
        """
        if not isinstance(sample, Sample):
            raise TypeError(f"Expected Sample, got {type(sample)}")

        key = self._key(name, etype)
        if key not in self._history:
            self._history[key] = TensorHistory(key, self._levels)

        self._history[key].push(sample)
        self._versions[name] = self._history[key].version

        # Cascade invalidation on push
        self._invalidate_cache_cascade(name)

    def field(
        self,
        name: str,
        etype: ElementType,
        level: int = 0,
    ) -> Sample:
        """Get field at given time level and location."""
        key = self._key(name, etype)
        hist = self._history.get(key)
        if hist is None:
            return None
        return hist.at(level)

    def latest(self, name: str, etype: ElementType) -> Sample:
        """Get the most recent sample of a field."""
        return self.field(name, etype, level=0)

    def has(self, name: str, etype: ElementType, level: int = 0) -> bool:
        """Check if field exists at given level."""
        return self.field(name, etype, level) is not None

    def history_depth(self, name: str, etype: ElementType) -> int:
        """Number of filled history levels for a field."""
        key = self._key(name, etype)
        hist = self._history.get(key)
        return len(hist) if hist else 0

    # -- cache management  --------------------------

    def get_computed(self, product: DataProduct) -> Sample:
        """Query cache for a previously computed product.

        Wildcard query: if ``product.namespace == "*"``, matches ANY
        namespace (newest fresh hit wins).
        """
        if self._mode == RunMode.TRAIN:
            return None

        # --- wildcard: match any namespace ---
        if product.namespace == "*":
            candidate_key = None
            candidate_ver = -1
            for key, entry in self._cache.items():
                if (
                    entry.product.otype == product.otype
                    and entry.product.fname == product.fname
                    and entry.product.etype == product.etype
                ):
                    if entry.is_stale(self._versions):
                        continue
                    ver_sum = sum(entry.versions.values())
                    if ver_sum > candidate_ver:
                        candidate_key = key
                        candidate_ver = ver_sum
            return self._cache[candidate_key].sample if candidate_key else None

        # --- exact match ---
        key = str(product)
        entry = self._cache.get(key)
        if entry is None:
            return None
        if entry.is_stale(self._versions):
            del self._cache[key]
            return None
        return entry.sample

    def put_computed(
        self,
        product: DataProduct,
        sample: Sample,
        depends: list[DataProduct] = None,
    ):
        """Store a computed product in cache for reuse."""
        if self._mode == RunMode.TRAIN:
            return
        key = str(product)
        self._cache[key] = ComputedEntry(
            product=product,
            sample=sample,
            versions=dict(self._versions),
            depends=list(depends) if depends else [],
        )

    def _invalidate_cache_cascade(self, changed_field: str):
        """Remove cache entries affected transitively."""
        to_remove: set[str] = set()

        # Phase 1 — direct victims (source changed)
        for key, entry in self._cache.items():
            if changed_field in entry.versions:
                to_remove.add(key)

        # Phase 2 — transitive victims (depend on removed products)
        changed = True
        while changed:
            changed = False
            for key, entry in self._cache.items():
                if key in to_remove:
                    continue
                dep_keys = {str(d) for d in entry.depends}
                if dep_keys & to_remove:
                    to_remove.add(key)
                    changed = True

        for key in to_remove:
            del self._cache[key]

    def clear_cache(self):
        """Clear all cached computed products."""
        self._cache.clear()

    # -- cache introspection ------------------------

    def cache_keys(self) -> list[str]:
        """Return all current cache keys."""
        return list(self._cache.keys())

    def cache_summary(self) -> dict[str, dict]:
        """Return cache summary."""
        return {
            key: {
                "product": str(entry.product),
                "source_versions": dict(entry.versions),
                "depends_on": [str(d) for d in entry.depends],
            }
            for key, entry in self._cache.items()
        }

    # -- cleanup ------------------------------------

    def clear(self):
        """Clear all history and cache."""
        for hist in self._history.values():
            hist.clear()
        self._cache.clear()
        self._versions = {f: 0 for f in self._fields}

    def __repr__(self) -> str:
        n_hist = sum(1 for h in self._history.values() if len(h) > 0)
        n_cache = len(self._cache)
        return f"DataHub(fields={self._fields}, history={n_hist}, cached={n_cache})"
