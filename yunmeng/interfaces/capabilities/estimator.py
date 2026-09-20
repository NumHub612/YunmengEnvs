# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Estimation protocols: observations, losses, estimators, artifact store.

Estimation is the θ-channel consumer layer: it drives an IEstimable
target (solver or standalone model) through trials and writes back
parameters. Two estimator families share IEstimator: gradient-free
calibrators (copy-semantic θ only) and gradient trainers (require
IDifferentiable leaves, see capabilities).

Design invariants of the estimation layer:

1. Copy semantics across the boundary. Observations enter the target
   through IObservationSet values and plain θ vectors; the estimation
   layer never holds autograd-carrying references INTO the target.
   Gradient flow lives entirely inside one fit() call, owned by the
   trainer and the target's solver layer.

2. Trial isolation via reset(). Between trials the estimator calls
   the target's reset() — never re-initialize() — so bindings and the
   DataHub allocation survive and trial parameters set via
   set_parameters() are preserved (ISolver invariant 4).

3. Standalone TRAIN only. fit() sets the target to TRAIN; a target
   participating in a composition must already have rejected that
   (IModel invariant 2). Estimators never run against coupled models.

4. Every trained result is provenance-tracked: a fit() that produces
   parameters must register them with IModelArtifactStore together
   with a TrainingMeta (estimator, data lineage, metrics), and return
   the ModelRef in EstimationResult.model_refs. Silent parameter
   hand-over without registration is a contract violation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol

from yunmeng.interfaces.capabilities import IEstimable
from yunmeng.interfaces.types import ArrayLike

# ---------------------------------------------------
# region Estimators
# ---------------------------------------------------


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


# ---------------------------------------------------
# region ModelArtifactStore
# ---------------------------------------------------
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
