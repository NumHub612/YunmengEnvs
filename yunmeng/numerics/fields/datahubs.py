# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Datahubs for managing of the fields and its history.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch

from yunmeng.numerics.enums import ElementType
from yunmeng.numerics.fields import Field

# ---------------------------------------------------
# region Sample
# ---------------------------------------------------


@dataclass(slots=True)
class Sample2:
    """A time-stamped field snapshot. Mutable — can be updated in-place."""

    timestamp: float
    data: Field

    def detach(self) -> Sample2:
        """Return new Sample with data detached from computation graph."""
        return Sample2(self.timestamp, self.data.detach())

    def to_tensor(self) -> torch.Tensor:
        """Extract tensor from underlying Field data."""
        d = self.data
        if hasattr(d, "to_tensor"):
            return d.to_tensor()
        if hasattr(d, "values"):
            return torch.from_numpy(np.asarray(d.values))
        return torch.as_tensor(d)


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
    field_name: str  # which field this product is derived from
    etype: ElementType  # where the product lives
    namespace: str = "default"

    def __str__(self) -> str:
        return f"{self.otype}_{self.field_name}_{self.etype.name}_{self.namespace}"

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
    sample: Sample2
    versions: dict[str, int] = field(default_factory=dict)
    depends: list[DataProduct] = field(default_factory=list)

    def is_stale(self, current_versions: dict[str, int]) -> bool:
        """Check if any source field has changed."""
        return self.versions != current_versions

    def depends_on_field(self, field_name: str) -> bool:
        """Check if this entry depends on field_name."""
        return (
            self.product.field_name == field_name
            or field_name in self.versions
            or any(dep.field_name == field_name for dep in self.depends)
        )


# ---------------------------------------------------------------------------
# region TensorHistory
# ---------------------------------------------------------------------------


class TensorHistory:
    """Stores time history as stacked tensors, preserving computation graphs.

    History is stored as a single ``torch.Tensor`` of shape
    ``(levels, N, ...)`` so PyTorch tracks gradients across time steps
    natively.
    """

    def __init__(self, name: str, levels: int, shape_hint: tuple = None):
        self._name = name
        self._max_levels = levels
        self._shape_hint = shape_hint
        self._version = 0

        # Store Sample2 references (not stacked tensors)
        # [newest, t-1, t-2, ...]
        self._history: list[Optional[Sample2]] = [None] * levels

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> int:
        return self._version

    @property
    def dtype(self) -> Optional[torch.dtype]:
        return self._tensors.dtype if self._tensors is not None else None

    @property
    def device(self) -> Optional[torch.device]:
        return self._tensors.device if self._tensors is not None else None

    def push(self, sample: Sample2) -> None:
        """Push new sample — shifts history, drops oldest.

        Only the Sample2 *reference* is moved (O(levels)), no Field copy.
        If the Field's underlying data is a torch.Tensor with requires_grad,
        the computation graph connection is preserved.
        """
        if not isinstance(sample, Sample2):
            raise TypeError(f"Expected Sample2, got {type(sample)}")

        # Shift: [newest, t-1, t-2] -> [new_sample, newest, t-1]
        for i in range(self._max_levels - 1, 0, -1):
            self._history[i] = self._history[i - 1]
        self._history[0] = sample
        self._version += 1

    def latest(self) -> Optional[Sample2]:
        """Get the most recent sample (level=0)."""
        return self._history[0]

    def at(self, level: int = 0) -> Optional[Sample2]:
        """Get sample at time level (0=current, 1=previous, ...)."""
        if level < 0 or level >= self._max_levels:
            raise IndexError(f"Level {level} out of range [0, {self._max_levels})")
        return self._history[level]

    def at_time(self, t: float) -> Optional[Sample2]:
        """Find sample closest to given physical time."""
        best, best_dt = None, float("inf")
        for s in self._history:
            if s is None:
                continue
            dt = abs(s.timestamp - t)
            if dt < best_dt:
                best, best_dt = s, dt
        return best

    def as_stacked_tensor(self, levels: int = None) -> Optional[torch.Tensor]:
        """Convert history to a stacked tensor for neural network input.

        Shape: (L, N, ...) where L=time levels, N=nodes, C=components.
        This creates a NEW tensor (copy) — the original Field data in
        history is untouched.

        If the Field's backend is numpy, tensors are converted via
        ``torch.from_numpy()``.
        """
        n = levels or self._max_levels
        valid = [self._history[i] for i in range(n) if self._history[i] is not None]
        if not valid:
            return None

        tensors = []
        for s in valid:
            t = (
                s.data.to_tensor()
                if hasattr(s.data, "to_tensor")
                else torch.as_tensor(s.data)
            )
            tensors.append(t)

        return torch.stack(tensors, dim=0)

    def clear(self) -> None:
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


class DataHub2:
    """Central data management for a solver.

    Three subsystems:
        1. Time history  — TensorHistory per (field, loc) for multi-step schemes
        2. Computed cache — operator products for cross-operator reuse
        3. Version tracking — automatic cascade cache invalidation
    """

    def __init__(self, fields: list[str], levels: int):
        self._fields = list(set(fields))
        self._levels = levels

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

    # -- key helpers --------------------------------

    @staticmethod
    def _key(name: str, etype: ElementType) -> str:
        return f"{name}_{etype.name}"

    # -- time history management --------------------

    def push(self, name: str, sample: Sample2, etype: ElementType) -> None:
        """Push a field snapshot into time history.

        Args:
            name: field name
            sample: the Sample2 to store
            etype: element type where the field lives
        """
        if not isinstance(sample, Sample2):
            raise TypeError(f"Expected Sample2, got {type(sample)}")

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
    ) -> Optional[Sample2]:
        """Get field at given time level and location."""
        key = self._key(name, etype)
        hist = self._history.get(key)
        if hist is None:
            return None
        return hist.at(level)

    def latest(self, name: str, etype: ElementType) -> Optional[Sample2]:
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

    def get_computed(self, product: DataProduct) -> Optional[Sample2]:
        """Query cache for a previously computed product.

        Wildcard query: if ``product.namespace == "*"``, matches ANY
        namespace (newest fresh hit wins).
        """
        # --- wildcard: match any namespace ---
        if product.namespace == "*":
            candidate_key = None
            candidate_ver = -1
            for key, entry in self._cache.items():
                if (
                    entry.product.otype == product.otype
                    and entry.product.field_name == product.field_name
                    and entry.product.loc == product.loc
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
        sample: Sample2,
        depends: Optional[list[DataProduct]] = None,
    ) -> None:
        """Store a computed product in cache for reuse."""
        key = str(product)
        self._cache[key] = ComputedEntry(
            product=product,
            sample=sample,
            versions=dict(self._versions),
            depends=list(depends) if depends else [],
        )

    def _invalidate_cache_cascade(self, changed_field: str) -> None:
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

    def clear_cache(self) -> None:
        """Clear all cached computed products."""
        self._cache.clear()

    # -- tensor export for AI models ----------------

    def to_tensor_batch(
        self, name: str, etype: ElementType, levels: int = None
    ) -> Optional[torch.Tensor]:
        """Export field history as batched tensor (T, N, C). Detached."""
        key = self._key(name, etype)
        hist = self._history.get(key)
        if hist is None:
            return None
        stacked = hist.as_stacked_tensor(levels)
        return stacked.detach() if stacked is not None else None

    def to_training_sample(
        self, name: str, etype: ElementType, input_levels: int = 2
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Create (input, target) pair for supervised learning."""
        key = self._key(name, etype)
        hist = self._history.get(key)
        if hist is None or len(hist) < input_levels + 1:
            return None, None

        target_t = hist.at(0).to_tensor().detach()

        past = []
        for i in range(1, input_levels + 1):
            s = hist.at(i)
            if s is None:
                return None, None
            past.append(s.to_tensor().detach())
        input_tensor = torch.stack(past, dim=0)

        return input_tensor, target_t

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

    def clear(self) -> None:
        """Clear all history and cache."""
        for hist in self._history.values():
            hist.clear()
        self._cache.clear()
        self._versions = {f: 0 for f in self._fields}

    def __repr__(self) -> str:
        n_hist = sum(1 for h in self._history.values() if len(h) > 0)
        n_cache = len(self._cache)
        return f"DataHub(fields={self._fields}, history={n_hist}, cached={n_cache})"


# -----------------------------------------------
# region Datahub v1
# -----------------------------------------------


@dataclass(slots=True, frozen=True, order=False)
class Sample:
    """A sample of the field."""

    timestamp: float
    data: Field


class DataHub:
    """Datahub for managing the fields and its history."""

    def __init__(self, fields: list[str], levels: int):
        """Initialize with the given fields and levels."""
        self._buffs: dict[str, RingBuffer] = {}
        self._grads: dict[str, RingBuffer] = {}
        for f in list(set(fields)):
            for loc in ElementType:
                _name = self._inner_name(f, loc)
                self._buffs[_name] = RingBuffer(levels)
                self._grads[_name] = RingBuffer(levels)
        self._size = levels
        self._raws = fields

    def _inner_name(self, name: str, loc: ElementType):
        return f"{name}_{loc.name}"

    def field(
        self, name: str, level: int = 0, loc: ElementType = ElementType.CELL
    ) -> Sample:
        """Fetch the specified field at the given level.

        NOTE: level=0 present the latest data,
        level=1 present the previous data,
        and so on.
        """
        _name = self._inner_name(name, loc)
        return self._buffs[_name][level]

    def grad(
        self, name: str, level: int = 0, loc: ElementType = ElementType.CELL
    ) -> Sample:
        """Fetch the specified field's gradient at the given level.

        NOTE: level=0 present the latest gradient,
        level=1 present the previous gradient,
        and so on.
        """
        _name = self._inner_name(name, loc)
        return self._grads[_name][level]

    def has(
        self, name: str, loc: ElementType, level: int = 0, grad: bool = False
    ) -> bool:
        """Check if the datahub has the target field."""
        if name not in self._raws:
            return False
        if level >= self._size or level < 0:
            return False

        _name = self._inner_name(name, loc)
        if grad:
            return self._grads[_name][level] is not None
        else:
            return self._buffs[_name][level] is not None

    def clear(self):
        """Clear the datahub data."""
        for buff in self._buffs.values():
            buff.clear()
        for grad in self._grads.values():
            grad.clear()

    def push_field(self, name: str, sample: Sample):
        """Push origin field."""
        _name = self._inner_name(name, sample.data.meta.etype)
        self._buffs[_name].push(sample)

    def push_grad(self, name: str, sample: Sample):
        """Push gradient."""
        _name = self._inner_name(name, sample.data.meta.etype)
        self._grads[_name].push(sample)

    def push(self, name: str, field: Sample, grad: Sample):
        """Push both field and gradient."""
        self.push_field(name, field)
        self.push_grad(name, grad)


class RingBuffer:
    """Class RingBuffer for managing the history of field."""

    def __init__(self, size: int = 1):
        """Initialize the ring buffer."""
        if size < 1:
            raise ValueError("Size must be greater than 0.")
        self._i = 0
        self._size = size
        self._data = [None] * size

    def push(self, obj: Sample):
        if not isinstance(obj, Sample):
            raise TypeError(f"Invalid buffer {type(obj)}.")
        self._data[self._i] = obj
        self._i = (self._i + 1) % self._size

    def __getitem__(self, level: int):
        if level >= self._size or level < 0:
            raise IndexError(f"Level {level} out of range.")
        i = (self._i - 1 - level) % self._size
        return self._data[i]

    def __len__(self):
        return self._size

    def clear(self):
        self._data = [None] * self._size
        self._i = 0
