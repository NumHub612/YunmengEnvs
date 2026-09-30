# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Boundary provider that merges a list of boundary conditions.
"""

from dataclasses import dataclass
from typing import Sequence
import numpy as np

from yunmeng.interfaces.solver import (
    IBoundaryCondition,
    IBoundaryProvider,
    BoundaryValues,
    BoundaryChannel,
)


@dataclass
class _RuleSlot:
    """Bind-time plan for one rule inside one bucket."""

    rule: IBoundaryCondition
    span: slice
    """Where this rule's values land."""


@dataclass
class _Bucket:
    """Bind-time plan for one (channel, field) pair."""

    channel: BoundaryChannel
    field: str
    ids: np.ndarray
    """Concatenated int64 element ids."""

    slots: list[_RuleSlot]
    buffer: np.ndarray
    """Preallocated value buffer, aligned with ids."""


class ListBoundaryProvider(IBoundaryProvider):
    """Merge a fixed list of boundary conditions.

    uncovered boundary regions are treated as zero-flux boundaries.
    """

    def __init__(self, bcs: Sequence[IBoundaryCondition] = ()):
        self._bcs = list(bcs)
        self._buckets: list[_Bucket] = []
        self._dirty = True

    @property
    def rules(self) -> list[IBoundaryCondition]:
        return list(self._bcs)

    def bind(self):
        """build the merge plan."""
        groups = {}
        for bc in self._bcs:
            ids = np.asarray(bc.region.element_ids, dtype="int64")
            n = len(ids)

            # t0 shape probe: values must align with the region
            probe = np.asarray(bc.evaluate(0.0)).reshape(-1)
            if len(probe) != n:
                raise ValueError(
                    f"BC {bc.id} rule produced {len(probe)} values, "
                    f"doesn't match region with {n} elements."
                )

            key = (bc.channel, bc.target_field)
            groups.setdefault(key, []).append((bc, ids))

        buckets = []
        for (channel, field), entries in groups.items():
            ids = np.concatenate([_ids for _, _ids in entries])
            if len(np.unique(ids)) != len(ids):
                raise ValueError(
                    f"overlapping BC segments for field {field!r} "
                    f"on channel {channel.value!r}"
                )

            slots, cursor = [], 0
            for bc, entry_ids in entries:
                span = slice(cursor, cursor + len(entry_ids))
                slots.append(_RuleSlot(rule=bc, span=span))
                cursor += len(entry_ids)

            buckets.append(
                _Bucket(
                    channel=channel,
                    field=field,
                    ids=ids,
                    slots=slots,
                    buffer=np.zeros(cursor, dtype="float64"),
                )
            )

        self._buckets = buckets
        self._dirty = False

    def evaluate(self, t: float) -> BoundaryValues:
        if self._dirty:
            self.bind()

        result = BoundaryValues(time=t)
        for bucket in self._buckets:
            buffer = bucket.buffer
            for slot in bucket.slots:
                values = np.asarray(slot.rule.evaluate(t)).reshape(-1)
                buffer[slot.span] = values
            result.segment(bucket.channel)[bucket.field] = (
                bucket.ids,
                buffer.copy(),
            )
        return result
