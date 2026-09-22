# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Lightweight adapter registry for exchange items.
"""

import numpy as np

from yunmeng.interfaces.solution import IElementSet, IInput, IOutput, Quantity
from yunmeng.interfaces.types import ArrayLike
from yunmeng.solutions.commons.models import BaseAdapter
from yunmeng.numerics.algos import ym_register


# ---------------------------------------------------
# region WeightedAdapter
# ---------------------------------------------------
@ym_register("adapter", name="weighted")
class WeightedAdapter(BaseAdapter):
    """Weighted-sum reduction (e.g. Thiessen weights over stations)."""

    def __init__(
        self,
        adapter_id: str,
        weights: ArrayLike,
        adaptee: IOutput = None,
        target_elements: IElementSet = None,
    ):
        super().__init__(adapter_id, adaptee, elements=target_elements)
        w = np.asarray(weights, dtype=float).flatten()
        if not np.isclose(w.sum(), 1.0, atol=1e-3):
            raise ValueError(f"Weights must sum to 1 (got {w.sum()}).")
        self._w = w

    def adapt(self, data: ArrayLike):
        data = np.asarray(data, dtype=float).flatten()
        if data.size != self._w.size:
            raise ValueError(
                f"WeightedAdapter '{self._id}': adaptee has {data.size} "
                f"elements but {self._w.size} weights."
            )
        return np.array([float(np.dot(self._w, data))])


# ---------------------------------------------------
# region ElementMapAdapter
# ---------------------------------------------------
@ym_register("adapter", name="element_map")
class ElementMapAdapter(BaseAdapter):
    """Gather a subset of upstream elements and optionally scatter them
    into target positions (M -> N)."""

    def __init__(
        self,
        adapter_id: str,
        source_indices: ArrayLike = None,
        target_indices: ArrayLike = None,
        target_size: int = None,
        adaptee: IOutput = None,
        target_elements: IElementSet = None,
    ):
        super().__init__(adapter_id, adaptee, elements=target_elements)
        self._src = None
        self._tgt = None
        self._target_size = 0
        if source_indices is not None:
            self._configure(source_indices, target_indices, target_size)

    def _configure(
        self,
        source_indices: ArrayLike,
        target_indices: ArrayLike = None,
        target_size: int = None,
    ):
        self._src = np.asarray(source_indices, dtype=int).flatten()
        self._tgt = (
            np.asarray(target_indices, dtype=int).flatten()
            if target_indices is not None
            else None
        )
        if self._tgt is not None and self._tgt.size != self._src.size:
            raise ValueError(
                f"ElementMapAdapter {self._id}: source/target index "
                f"count mismatch {self._src.size} vs {self._tgt.size}."
            )
        self._target_size = target_size or self._src.size

    @staticmethod
    def _resolve(element_set: IElementSet, elements: list) -> list:
        ids = getattr(element_set, "element_ids", None)
        resolved = []
        for e in elements:
            if isinstance(e, (int, np.integer)):
                resolved.append(int(e))
            else:
                if ids is None:
                    raise ValueError("Element set exposes no ids.")
                resolved.append(ids.index(str(e)))
        return resolved

    def set_mapping(
        self,
        target_port: IInput,
        source_port: IOutput,
        source_elements: list,
        target_elements: list = None,
    ):
        """Configure in place and wire source -> this -> target."""
        src_idx = self._resolve(source_port.element_set, source_elements)
        tgt_idx, tgt_size = None, None
        if target_elements is not None:
            tgt_idx = self._resolve(target_port.element_set, target_elements)
            tgt_size = target_port.element_set.element_count
        self._configure(src_idx, tgt_idx, tgt_size)
        if self._upstream is not source_port:
            source_port.add_adapter(self)
        target_port.provider = self

    def adapt(self, data: ArrayLike):
        if self._src is None:
            raise ValueError(f"ElementMapAdapter '{self._id}' has no mapping.")
        data = np.asarray(data, dtype=float).flatten()
        gathered = data[self._src]
        if self._tgt is not None:
            out = np.full(self._target_size, np.nan)
            out[self._tgt] = gathered
            return out
        if gathered.size == self._target_size:
            return gathered
        out = np.full(self._target_size, np.nan)
        out[: gathered.size] = gathered
        return out


# ---------------------------------------------------
# region ScaleAdapter
# ---------------------------------------------------
@ym_register("adapter", name="scale")
class ScaleAdapter(BaseAdapter):
    """Constant factor scaling — stands in for unit conversion."""

    def __init__(
        self,
        adapter_id: str,
        factor: float,
        adaptee: IOutput = None,
        quantity: Quantity = None,
    ):
        super().__init__(adapter_id, adaptee, quantity=quantity)
        self._factor = float(factor)

    @property
    def factor(self) -> float:
        return self._factor

    def adapt(self, data: ArrayLike):
        return np.asarray(data, dtype=float) * self._factor


# ---------------------------------------------------
# region MeanAdapter
# ---------------------------------------------------
@ym_register("adapter", name="mean")
class MeanAdapter(BaseAdapter):
    """Spatial mean reduction to a scalar frame."""

    def __init__(
        self,
        adapter_id: str,
        adaptee: IOutput = None,
        target_elements: IElementSet = None,
    ):
        super().__init__(adapter_id, adaptee, elements=target_elements)

    def adapt(self, data: ArrayLike):
        return np.array([float(np.mean(np.asarray(data, dtype=float)))])


@ym_register("adapter", name="index")
class IndexAdapter(BaseAdapter):
    """Pick a single element by index."""

    def __init__(
        self,
        adapter_id: str,
        index: int,
        adaptee: IOutput = None,
        target_elements: IElementSet = None,
    ):
        super().__init__(adapter_id, adaptee, elements=target_elements)
        self._index = int(index)

    def adapt(self, data: ArrayLike):
        return np.atleast_1d(np.asarray(data, dtype=float).flatten()[self._index])
