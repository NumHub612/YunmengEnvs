# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Datahubs for managing of the fields and its history.
"""

from yunmeng.numerics.enums import ElementType
from yunmeng.numerics.fields import Field
from dataclasses import dataclass

# -----------------------------------------------
# region TODO: Datahub v2
# -----------------------------------------------

"""
Redesigned DataHub — the solver-level data center.

Core responsibilities:
    1. Time-history storage for multi-step schemes (RK2, AB2, BDF2, etc.)
    2. Cross-operator data reuse (gradients, divergence, interpolated values)
    3. Differentiable computation graph node (stores tensors with gradients)

Design principles:
    - History is stored as stacked tensors (not Python lists), preserving gradients
    - Data reuse is explicit: operators declare products, consumers query availability
    - Lazy evaluation with automatic cache invalidation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import torch

from yunmeng.numerics.enums import ElementType
from yunmeng.numerics.fields import Field

# ---------------------------------------------------------------------------
# Sample — mutable, holds Fields that may have requires_grad
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Sample2:
    """A time-stamped field snapshot. Mutable — can be updated in-place."""

    timestamp: float
    data: Field

    def detach(self) -> Sample2:
        """Return a new Sample with data detached from computation graph."""
        return Sample2(self.timestamp, self.data.detach())


# ---------------------------------------------------------------------------
# DataProduct — what an operator produces, for reuse tracking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataProduct:
    """Identifies a computed product that can be reused by other operators.

    Examples:
        DataProduct("grad", "u", ElementType.NODE)   # gradient of u at nodes
        DataProduct("div", "u", ElementType.CELL)    # divergence of u at cells
        DataProduct("interp", "grad_u", ElementType.FACE)  # interpolated grad to faces
    """

    op_type: str  # "grad", "div", "lap", "interp", "limiter", etc.
    field_name: str  # which field this product is derived from
    loc: ElementType  # where the product lives (NODE, CELL, FACE)

    def __str__(self) -> str:
        return f"{self.op_type}_{self.field_name}_{self.loc.name}"


# ---------------------------------------------------------------------------
# ComputedEntry — a lazily-evaluated or cached computation
# ---------------------------------------------------------------------------


@dataclass
class ComputedEntry:
    """A computed field, either eagerly or lazily evaluated."""

    product: DataProduct
    sample: Sample2
    # Validity: which field versions this computation is based on
    # If upstream fields change, this entry becomes stale
    source_versions: dict[str, int] = field(default_factory=dict)

    def is_stale(self, current_versions: dict[str, int]) -> bool:
        """Check if the source fields have changed since computation."""
        return self.source_versions != current_versions


# ---------------------------------------------------------------------------
# TensorHistory — differentiable time history (replaces RingBuffer)
# ---------------------------------------------------------------------------


class TensorHistory:
    """Stores time history as stacked tensors, preserving computation graphs.

    Unlike RingBuffer (Python list of Samples), TensorHistory concatenates
    Field data along a time dimension, allowing PyTorch to track gradients
    across time steps for multi-step schemes.

    For RK2: needs 1 level (current + 1 intermediate)
    For AB2: needs 2 levels (current + 1 history)
    For BDF2: needs 2 levels (current + 2 history)
    """

    def __init__(self, name: str, levels: int, shape_hint: tuple = None):
        self._name = name
        self._max_levels = levels
        self._history: list[Sample2] = [None] * levels  # [newest, ..., oldest]
        self._shape_hint = shape_hint  # (size, *vtype_shape)
        self._version = 0  # increments on each push

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> int:
        return self._version

    def push(self, sample: Sample2) -> None:
        """Push new sample — shifts history, drops oldest."""
        if not isinstance(sample, Sample2):
            raise TypeError(f"Expected Sample, got {type(sample)}")

        # Shift: [newest, t-1, t-2] -> [new, newest, t-1]
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
        """Stack history levels into a single tensor for vectorized schemes.

        Returns tensor of shape (levels, N, ...) — useful for multi-step
        methods that operate on multiple time levels simultaneously.
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
                else torch.from_numpy(
                    s.data.values if hasattr(s.data, "values") else s.data
                )
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
# DataHub v2 — solver-level data center
# ---------------------------------------------------------------------------


class DataHub2:
    """
    Central data management for a solver.

    Three subsystems:
        1. Time history: stores field evolution over time (for multi-step schemes)
        2. Computed cache: stores operator products for cross-operator reuse
        3. Version tracking: automatic cache invalidation when source data changes

    Usage for multi-step schemes (AB2):
        # AB2: u^{n+1} = u^n + dt * (3*f(u^n) - f(u^{n-1})) / 2
        f_n = compute_rhs(datahub.field("u", level=0))   # current
        f_nm1 = datahub.field("u", level=1)               # history
        u_new = u_n + dt * (3*f_n - f_nm1) / 2

    Usage for data reuse:
        # Grad01 computes grad(u), stores in cache
        grad_u = grad_op.forward_field(u, dt, datahub)
        # Div01 needs div(u) = trace(grad(u)), checks cache first
        div_u = div_op.forward_field(u, dt, datahub)  # internally: div = trace(grad_u)
    """

    def __init__(self, fields: list[str], levels: int):
        """Initialize DataHub for given field names and history depth.

        Args:
            fields: field names to track (e.g., ["u", "p"])
            levels: time history depth (1=Euler, 2=AB2/RK2, 3=RK3/BDF2)
        """
        self._fields = list(set(fields))
        self._levels = levels

        # Time history: {field_name}_{loc} -> TensorHistory
        self._history: dict[str, TensorHistory] = {}
        for fname in self._fields:
            for loc in ElementType:
                key = self._key(fname, loc)
                self._history[key] = TensorHistory(key, levels)

        # Computed cache: DataProduct -> ComputedEntry
        self._cache: dict[str, ComputedEntry] = {}

        # Version tracking: {field_name} -> version counter
        # Incremented on push, used for cache invalidation
        self._versions: dict[str, int] = {f: 0 for f in self._fields}

    # -- key helpers --------------------------------------------------------

    @staticmethod
    def _key(name: str, loc: ElementType) -> str:
        return f"{name}_{loc.name}"

    # -- time history management --------------------------------------------

    def push(self, name: str, sample: Sample2, loc: ElementType = ElementType.NODE):
        """Push a field snapshot into time history."""
        key = self._key(name, loc)
        if key not in self._history:
            self._history[key] = TensorHistory(key, self._levels)

        self._history[key].push(sample)
        self._versions[name] = self._history[key].version

        # Invalidate cache entries that depend on this field
        self._invalidate_cache(name)

    def field(
        self, name: str, level: int = 0, loc: ElementType = ElementType.NODE
    ) -> Optional[Sample2]:
        """Get field at given time level and location.

        Args:
            name: field name
            level: 0=current, 1=previous, ...
            loc: element type where field lives

        Returns:
            Sample or None if not available at this level
        """
        key = self._key(name, loc)
        hist = self._history.get(key)
        if hist is None:
            return None
        return hist.at(level)

    def latest(
        self, name: str, loc: ElementType = ElementType.NODE
    ) -> Optional[Sample2]:
        """Get the most recent sample of a field."""
        return self.field(name, level=0, loc=loc)

    def has(
        self, name: str, level: int = 0, loc: ElementType = ElementType.NODE
    ) -> bool:
        """Check if field exists at given level."""
        return self.field(name, level, loc) is not None

    def history_depth(self, name: str, loc: ElementType = ElementType.NODE) -> int:
        """Number of filled history levels for a field."""
        key = self._key(name, loc)
        hist = self._history.get(key)
        return len(hist) if hist else 0

    # -- cache management (cross-operator data reuse) -----------------------

    def get_computed(self, product: DataProduct) -> Optional[Sample2]:
        """Query cache for a previously computed product.

        Returns the cached Sample if:
            1. The product was previously computed
            2. The source fields haven't changed since computation

        Returns None if cache miss or stale.
        """
        key = str(product)
        entry = self._cache.get(key)
        if entry is None:
            return None
        if entry.is_stale(self._versions):
            del self._cache[key]
            return None
        return entry.sample

    def put_computed(self, product: DataProduct, sample: Sample2) -> None:
        """Store a computed product in cache for reuse.

        Operators should call this after computing a reusable product.
        """
        key = str(product)
        self._cache[key] = ComputedEntry(
            product=product,
            sample=sample,
            source_versions=dict(self._versions),
        )

    def _invalidate_cache(self, changed_field: str) -> None:
        """Remove cache entries that depend on a changed field."""
        stale_keys = []
        for key, entry in self._cache.items():
            if entry.product.field_name == changed_field:
                stale_keys.append(key)
        for key in stale_keys:
            del self._cache[key]

    def clear_cache(self) -> None:
        """Clear all cached computed products."""
        self._cache.clear()

    # -- multi-step scheme helpers ------------------------------------------

    def ab2_rhs(
        self,
        name: str,
        compute_f: Callable[[Field], Field],
        dt: float,
        loc: ElementType = ElementType.NODE,
    ) -> Field:
        """Compute AB2 right-hand side: (3*f(u^n) - f(u^{n-1})) / 2.

        Args:
            name: field name
            compute_f: function to compute f(u) for a given u
            dt: time step
            loc: element type

        Returns:
            Combined RHS for AB2 time stepping.
        """
        u_n = self.field(name, level=0, loc=loc)
        u_nm1 = self.field(name, level=1, loc=loc)

        if u_n is None:
            raise ValueError(f"No current data for {name}")

        f_n = compute_f(u_n.data)

        if u_nm1 is None:
            # First step: fall back to Euler
            return f_n

        # Check if f(u^{n-1}) is cached
        prod_nm1 = DataProduct("rhs", name, loc)
        cached = self.get_computed(prod_nm1)
        if cached is not None:
            f_nm1 = cached.data
        else:
            f_nm1 = compute_f(u_nm1.data)
            self.put_computed(prod_nm1, Sample2(u_nm1.timestamp, f_nm1))

        # AB2 combination: (3*f_n - f_nm1) / 2
        return (f_n * 3.0 - f_nm1) * 0.5

    def rk2_step(
        self, name: str, compute_f: Callable[[Field, Field], Field], u: Field, dt: float
    ) -> Field:
        """Compute RK2 step: u + dt * (k1 + k2) / 2.

        Args:
            name: field name
            compute_f: function(u, u_stage) -> rhs for stage computation
            u: current field
            dt: time step

        Returns:
            Updated field after RK2 step.
        """
        # Stage 1
        k1 = compute_f(u, u)

        # Stage 2
        u_stage = u + dt * k1
        k2 = compute_f(u, u_stage)

        # Combine
        return u + dt * (k1 + k2) * 0.5

    # -- tensor export for AI models ----------------------------------------

    def to_tensor_batch(
        self, name: str, loc: ElementType = ElementType.NODE, levels: int = None
    ) -> Optional[torch.Tensor]:
        """Export field history as a batched tensor for neural network input.

        Shape: (T, N, C) where T=time levels, N=nodes, C=components.
        Automatically detaches from computation graph (for training data).
        """
        key = self._key(name, loc)
        hist = self._history.get(key)
        if hist is None:
            return None
        stacked = hist.as_stacked_tensor(levels)
        return stacked.detach() if stacked is not None else None

    def to_training_sample(
        self, name: str, loc: ElementType = ElementType.NODE, input_levels: int = 2
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create (input, target) pair for supervised learning.

        input: field at levels [1, 2, ...] (past states)
        target: field at level 0 (current state)

        Useful for training surrogate models that predict u^{n} from u^{n-1}, u^{n-2}...
        """
        key = self._key(name, loc)
        hist = self._history.get(key)
        if hist is None or len(hist) < input_levels + 1:
            return None, None

        # Target: current state
        target = hist.at(0).data.to_tensor().detach()

        # Input: past states stacked
        past = []
        for i in range(1, input_levels + 1):
            s = hist.at(i)
            if s is None:
                return None, None
            past.append(s.data.to_tensor().detach())
        input_tensor = torch.stack(past, dim=0)

        return input_tensor, target

    # -- cleanup ------------------------------------------------------------

    def clear(self) -> None:
        """Clear all history and cache."""
        for hist in self._history.values():
            hist.clear()
        self._cache.clear()
        self._versions = {f: 0 for f in self._fields}

    def __repr__(self) -> str:
        n_hist = sum(1 for h in self._history.values() if len(h) > 0)
        n_cache = len(self._cache)
        return (
            f"DataHub(fields={self._fields}, active_history={n_hist}, cached={n_cache})"
        )


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
