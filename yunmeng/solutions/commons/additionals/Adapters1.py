# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Lightweight adapter registry for exchange items.
"""

from __future__ import annotations
import numpy as np

from yunmeng.solutions.standards import (
    IAdapterOutput,
    IInput,
    IOutput,
    GeometryType,
    IElementSet,
    Quantity,
)
from yunmeng.solutions.commons.models import BaseAdapter
from yunmeng.setting import logger

# ---------------------------------------------------
# region WeightedAdapter
# ---------------------------------------------------


class WeightedAdapter(BaseAdapter):
    """Weight adaptation (such as Thiessen weights)."""

    def __init__(
        self,
        adapter_id: str,
        weights: list[float] | np.ndarray,
        adaptee: IOutput = None,
        target_elements: IElementSet = None,
    ):
        super().__init__(adapter_id, adaptee, elements=target_elements)
        w = np.asarray(weights, dtype=float).flatten()
        if not np.isclose(w.sum(), 1.0, atol=1e-3):
            raise ValueError(f"Weights must sum to 1 (got {w.sum()}).")
        self._w = w

    def adapt(self, data: np.ndarray) -> np.ndarray:
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


class ElementMapAdapter(BaseAdapter):
    """Arbitrary element mapping: gather a subset of upstream elements
    and optionally scatter them into target positions (M -> N).
    """

    def __init__(
        self,
        adapter_id: str,
        source_indices: list[int] | np.ndarray,
        target_indices: list[int] | np.ndarray = None,
        target_size: int = None,
        adaptee: IOutput = None,
        target_elements: IElementSet = None,
    ):
        super().__init__(adapter_id, adaptee, elements=target_elements)
        self._src = np.asarray(source_indices, dtype=int).flatten()
        self._tgt = (
            np.asarray(target_indices, dtype=int).flatten()
            if target_indices is not None
            else None
        )
        if self._tgt is not None and self._tgt.size != self._src.size:
            raise ValueError(
                f"ElementMapAdapter {adapter_id}: source/target index "
                f"count mismatch {self._src.size} vs {self._tgt.size}."
            )
        self._target_size = target_size or self._src.size

    @staticmethod
    def _resolve(element_set, elements) -> list[int]:
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

    @classmethod
    def from_ids(
        cls,
        adapter_id: str,
        source: IOutput,
        source_elements: list,
        target: IOutput = None,
        target_elements: list = None,
    ) -> "ElementMapAdapter":
        """Resolve element ids to indices."""
        src_idx = cls._resolve(source.element_set, source_elements)
        tgt_idx, tgt_size = None, None
        if target_elements is not None and target is not None:
            tgt_idx = cls._resolve(
                target.element_set,
                target_elements,
            )
            tgt_size = target.element_set.element_count
        return cls(
            adapter_id,
            src_idx,
            tgt_idx,
            tgt_size,
            adaptee=source,
        )

    def adapt(self, data: np.ndarray) -> np.ndarray:
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


class ScaleOutput(BaseAdapter):
    """Constant factor scaling - stands in for unit conversion."""

    def __init__(
        self,
        adapter_id: str,
        factor: float,
        adaptee: IOutput = None,
        quantity: Quantity = None,
    ):
        super().__init__(adapter_id, adaptee, quantity=quantity)
        self._factor = factor

    def adapt(self, data: np.ndarray) -> np.ndarray:
        return np.asarray(data, dtype=float) * self._factor
