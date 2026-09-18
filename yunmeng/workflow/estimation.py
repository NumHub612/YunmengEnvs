# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

description
"""

# -*- encoding: utf-8 -*-
"""Estimation protocols for the hybrid solver migration.

This module only adds contracts that are absent from the current interfaces
layer. IEstimable and IParameterized come from interfaces.capabilities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol

from yunmeng.interfaces.capabilities import IEstimable
from yunmeng.interfaces.types import ArrayLike


@dataclass(frozen=True)
class ModelRef:
    """Versioned reference to a model artifact."""

    model_id: str
    version: str

    def __str__(self) -> str:
        return f"{self.model_id}@{self.version}"


class MemoryStrategy(Enum):
    """Gradient memory strategy for unrolled long-horizon training."""

    FULL = "full"
    CHECKPOINT = "checkpoint"
    ADJOINT = "adjoint"


class IObservationSet(Protocol):
    """Observation data used by calibration or training."""

    def variables(self) -> list[str]: ...

    def times(self) -> ArrayLike: ...

    def values(self, variable: str) -> ArrayLike: ...


class ILoss(Protocol):
    """Loss between model output and observations."""

    def __call__(self, predicted: Any, observed: IObservationSet) -> ArrayLike: ...


@dataclass
class EstimationResult:
    """Outcome of one estimation run."""

    parameters: dict[str, Any] = field(default_factory=dict)
    model_refs: list[tuple[str, str]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    converged: bool = False
    message: str = ""


class IEstimator(ABC):
    """Unified calibration/training abstraction."""

    @classmethod
    @abstractmethod
    def get_name(cls) -> str: ...

    @abstractmethod
    def fit(
        self,
        target: IEstimable,
        data: IObservationSet,
        loss: ILoss,
        **kwargs,
    ) -> EstimationResult: ...

    @abstractmethod
    def evaluate(
        self,
        target: IEstimable,
        data: IObservationSet,
        **kwargs,
    ) -> dict[str, float]: ...


@dataclass
class TrainingMeta:
    """Provenance of a trained parameter set."""

    estimator: str = ""
    data_lineage: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    created_at: str = ""
    notes: str = ""


class IModelArtifactStore(Protocol):
    """Versioned persistence for parameter vectors."""

    def resolve(self, ref: ModelRef) -> dict[str, ArrayLike]: ...

    def register(
        self,
        params: Mapping[str, ArrayLike],
        meta: TrainingMeta,
    ) -> ModelRef: ...

    def meta(self, ref: ModelRef) -> TrainingMeta: ...

    def list_versions(self, model_id: str) -> list[str]: ...
