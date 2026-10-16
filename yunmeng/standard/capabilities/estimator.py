# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Estimation protocols.

Estimation is the θ-channel consumer layer: it drives an IEstimable
target (solver or standalone model) through trials and writes back
parameters. Estimators are plain classes exposing a conventional
interface — no ABC is enforced:

    fit(target, data, loss, **kwargs) -> EstimationResult
    evaluate(target, data, **kwargs) -> dict[str, float]

where ``loss`` is any callable ``loss(predicted, observed) -> scalar``.
Two estimator families are expected: gradient-free calibrators
(copy-semantic θ only) and gradient trainers (require differentiable
leaves on the target).

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
"""

from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, Callable

from yunmeng.standard.capabilities import IEstimable
from yunmeng.standard.types import ArrayLike


class IObservationSet(Protocol):
    """Observation data used by calibration or training."""

    def variables(self) -> list[str]: ...

    def times(self) -> ArrayLike: ...

    def values(self, variable: str) -> ArrayLike: ...


@dataclass
class EstimationResult:
    """Outcome of one estimation run."""

    parameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    message: str = ""


class IEstimator:
    """Unified calibration/training abstraction."""

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """The unique name of this estimator."""
        ...

    @abstractmethod
    def fit(
        self,
        target: IEstimable,
        data: IObservationSet,
        loss: Callable,
        **kwargs,
    ) -> EstimationResult:
        """Fit the target to the data using the loss."""
        ...

    @abstractmethod
    def evaluate(
        self,
        target: IEstimable,
        data: IObservationSet,
        **kwargs,
    ) -> dict[str, float]:
        """Evaluate the target on the data."""
        ...
