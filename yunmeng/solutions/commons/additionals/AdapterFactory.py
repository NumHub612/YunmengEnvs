# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Lightweight adapter registry for exchange items.
"""

from __future__ import annotations
import numpy as np

from yunmeng.solutions.standards import (
    IExchangeAdapter,
    IInput,
    IOutput,
    GeometryType,
)
from yunmeng.setting import logger

# ---------------------------------------------------
# region ArealMeanAdapter
# ---------------------------------------------------


class ArealMeanAdapter(IExchangeAdapter):
    """Weighted spatial mean from gauge points to scalar basin inputs.

    Implements Thiessen-style areal reduction.  One adapter instance is
    attached to a station output port and serves *all* consumer
    sub-basins, each with its own weight vector — this matches the
    framework's adapter-chain semantics, where adapters live on the
    output side and see the requesting input.

    Usage:
        adapter = ArealMeanAdapter("thiessen")
        adapter.set_weights("sub1.precipitation", [0.6, 0.4])
        adapter.set_weights("sub2.precipitation", [0.3, 0.7])
        rain_port.add_adapter(adapter)

    Weights must sum (approximately) to 1 and align with the source
    element (gauge) order.  A consumer without registered weights
    receives the arithmetic mean.
    """

    def __init__(self, adapter_id: str):
        self._id = adapter_id
        self._default = None
        self._per_consumer: dict[str, np.ndarray] = {}

    @property
    def id(self) -> str:
        return self._id

    def set_weights(self, consumer_input_id: str, weights: np.ndarray):
        """Register a Thiessen weight vector for one consumer input."""
        if consumer_input_id in self._per_consumer:
            logger.warning(
                f"Adapter {self._id}: overwriting weights for {consumer_input_id}."
            )
        self._per_consumer[consumer_input_id] = self._checked(weights)

    @staticmethod
    def _checked(weights: np.ndarray) -> np.ndarray:
        w = np.asarray(weights, dtype=float).flatten()
        if not np.isclose(w.sum(), 1.0, atol=1e-3):
            raise ValueError(f"Areal-mean weights must sum to 1 (got {w.sum()}).")
        return w

    def adapt(self, data: np.ndarray, source: IOutput, target: IInput) -> np.ndarray:
        data = np.asarray(data, dtype=float).flatten()
        w = self._per_consumer.get(getattr(target, "id", ""), self._default)
        if w is None:
            return np.array([float(data.mean())])
        if data.size != w.size:
            raise ValueError(
                f"Adapter '{self._id}': source has {data.size} gauges but "
                f"{w.size} weights were configured for '{target.id}'."
            )
        return np.array([float(np.dot(w, data))])

    @classmethod
    def can_adapt(cls, source: IOutput, target: IInput) -> bool:
        return (
            source is not None
            and target is not None
            and source.element_set.gtype == GeometryType.POINT
            and target.element_set.element_count == 1
        )


# ---------------------------------------------------
# region ElementMapAdapter
# ---------------------------------------------------


class ElementMapAdapter(IExchangeAdapter):
    """Element-to-element mapping between two exchange ports.

    This is the addressing mechanism for arbitrary element coupling:
    which elements of the source port feed which elements of the target
    port, resolved by *element id* (robust to ordering), e.g.

        basin outlet element "sub2"      -> reach element "sec_05"
        reservoir "gate_release"         -> 2-D grid cells {"c_31", "c_32"}

    One adapter instance attached to the source output serves all
    consumers, each with its own mapping (the same per-consumer pattern
    as :class:`ArealMeanAdapter`).  A consumer without a registered
    mapping receives the source array unchanged.

    Indices are resolved at construction; element sets are static by
    design, so the resolution stays valid for the run.
    """

    def __init__(self, adapter_id: str):
        self._id = adapter_id
        # target_input_id -> (source_indices, target_size)
        self._maps: dict[str, tuple[np.ndarray, int]] = {}

    @property
    def id(self) -> str:
        return self._id

    def set_mapping(
        self,
        target_input: IInput,
        source: IOutput,
        source_elements: list,
        target_elements: list = None,
    ):
        """Register an element mapping for one consumer.

        Args:
            target_input: the consuming input port.
            source: the providing output port.
            source_elements: element ids (or indices) of the source
                port's element set, in the order the consumer expects.
            target_elements: element ids (or indices) of the target
                element set; defaults to the first N elements.
        """
        src_idx = self._resolve(source.element_set, source_elements)
        if target_elements is None:
            tgt_size = len(src_idx)
        else:
            tgt_idx = self._resolve(target_input.element_set, target_elements)
            if len(tgt_idx) != len(src_idx):
                raise ValueError(
                    f"Adapter '{self._id}': source/target element count "
                    f"mismatch ({len(src_idx)} vs {len(tgt_idx)})."
                )
            tgt_size = target_input.element_set.element_count
        self._maps[target_input.id] = (np.asarray(src_idx, dtype=int), tgt_size)

    @staticmethod
    def _resolve(element_set, elements) -> list[int]:
        ids = getattr(element_set, "element_ids", None)
        resolved = []
        for e in elements:
            if isinstance(e, (int, np.integer)):
                resolved.append(int(e))
            else:
                if ids is None:
                    raise ValueError("Element set exposes no ids; use integer indices.")
                resolved.append(ids.index(str(e)))
        return resolved

    def adapt(self, data: np.ndarray, source: IOutput, target: IInput) -> np.ndarray:
        entry = self._maps.get(getattr(target, "id", ""))
        if entry is None:
            return data
        src_idx, tgt_size = entry
        data = np.asarray(data, dtype=float).flatten()
        gathered = data[src_idx]
        if gathered.size == tgt_size:
            return gathered
        out = np.full(tgt_size, np.nan)
        out[: gathered.size] = gathered
        return out

    @classmethod
    def can_adapt(cls, source: IOutput, target: IInput) -> bool:
        return source is not None and target is not None
