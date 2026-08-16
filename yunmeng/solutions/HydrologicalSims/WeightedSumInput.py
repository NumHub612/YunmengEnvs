# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Multi-provider fan-in input: `pull() = Σ wᵢ · providerᵢ`.
"""

from __future__ import annotations
import numpy as np

from yunmeng.solutions.standards import (
    IOutput,
    IElementSet,
    Quantity,
    TimeSpan,
)

from yunmeng.solutions.standards import ILinkableModel
from yunmeng.solutions.commons.models import BaseInput


class WeightedSumInput(BaseInput):
    """Multi-provider fan-in input: ``pull() = Σ wᵢ · providerᵢ``."""

    def __init__(
        self,
        item_id: str,
        quantity: Quantity,
        elements: IElementSet,
        weights: list[float],
        slot_names: list[str] = None,
        time_span: TimeSpan = None,
        owner: "ILinkableModel" = None,
        required: bool = True,
    ):
        super().__init__(
            item_id, quantity, elements, time_span, owner=owner, required=required
        )
        w = np.asarray(weights, dtype=float).flatten()
        if w.size == 0:
            raise ValueError(f"WeightedSumInput '{item_id}': no weights given.")
        if np.any(w <= 0):
            raise ValueError(f"WeightedSumInput '{item_id}': weights must be > 0.")
        if not np.isclose(w.sum(), 1.0, atol=1e-3):
            raise ValueError(
                f"WeightedSumInput '{item_id}': weights must sum to 1 "
                f"(got {w.sum():.4f})."
            )
        self._w = w
        self._slot_names = slot_names or [f"slot{i}" for i in range(w.size)]
        if len(self._slot_names) != w.size:
            raise ValueError("slot_names / weights length mismatch.")
        self._slots: list[IOutput] = [None] * w.size

    # -- multi-provider contract ------------------------

    @property
    def slot_names(self) -> list[str]:
        return list(self._slot_names)

    @property
    def weights(self) -> np.ndarray:
        return self._w.copy()

    def bind(self, slot: str, output: IOutput):
        """Attach a provider to a named slot (e.g. a gauge id)."""
        try:
            i = self._slot_names.index(slot)
        except ValueError:
            raise KeyError(
                f"{self._id}: unknown slot '{slot}' "
                f"(available: {self._slot_names})."
            ) from None
        if self._slots[i] is not None:
            raise ValueError(f"{self._id}: slot '{slot}' is already bound.")
        self._slots[i] = output

    def unbind(self, slot: str):
        i = self._slot_names.index(slot)
        self._slots[i] = None

    @property
    def providers(self) -> list[IOutput]:
        return [p for p in self._slots if p is not None]

    @property
    def unbound_slots(self) -> list[str]:
        return [n for n, p in zip(self._slot_names, self._slots) if p is None]

    # -- single-provider view (compatibility) ------------

    @property
    def provider(self) -> IOutput:
        bound = self.providers
        return bound[0] if len(bound) == 1 else None

    @provider.setter
    def provider(self, output: IOutput):
        if self._w.size != 1:
            raise TypeError(
                f"{self._id} is a {self._w.size}-provider fan-in; "
                f"use bind(slot, output) instead."
            )
        self._slots[0] = output

    @property
    def is_connected(self) -> bool:
        return all(p is not None for p in self._slots)

    # -- data flow ---------------------------------------

    def pull(self) -> np.ndarray:
        if not self.is_connected:
            raise ValueError(
                f"WeightedSumInput {self._id}: unbound slots " f"{self.unbound_slots}."
            )
        vals = [_as_frame(p.get_values(self)) for p in self._slots]
        shape = vals[0].shape
        for v in vals:
            if v.shape != shape:
                raise ValueError(
                    f"WeightedSumInput {self._id}: provider frame shape "
                    f"mismatch ({v.shape} vs {shape})."
                )
        self._values = sum(w * v for w, v in zip(self._w, vals))
        return self._values


def _as_frame(values) -> np.ndarray:
    return np.atleast_1d(np.asarray(values, dtype=float))
