# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

description
"""

from typing import Sequence
import numpy as np

from yunmeng.interfaces.solver import (
    IBoundaryCondition,
    IBoundaryProvider,
    BoundaryValues,
)


class ListBoundaryProvider(IBoundaryProvider):
    """Merge a fixed list of boundary-condition rules per evaluation."""

    def __init__(self, bcs: Sequence[IBoundaryCondition] = ()):
        self._bcs = list(bcs)

    @property
    def rules(self) -> list[IBoundaryCondition]:
        return list(self._bcs)

    def add(self, bc: IBoundaryCondition):
        self._bcs.append(bc)

    def evaluate(self, t: float) -> BoundaryValues:
        result = BoundaryValues(time=t)
        pending = {}
        for bc in self._bcs:
            channel, values = bc.evaluate(t)
            if channel not in ("constraints", "fluxes", "mixed"):
                raise ValueError(f"unknown BC channel {channel!r}")
            key = (channel, bc.target_field)
            ids = np.asarray(bc.region.element_ids, dtype="int64")
            pending.setdefault(key, []).append((ids, np.asarray(values)))

        for (channel, field), segments in pending.items():
            ids = np.concatenate([segment[0] for segment in segments])
            if len(np.unique(ids)) != len(ids):
                raise ValueError(f"overlapping BC segments for field {field!r}")
            values = np.concatenate([segment[1] for segment in segments])
            getattr(result, channel)[field] = (ids, values)
        return result
