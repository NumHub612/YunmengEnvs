# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Estimation-layer contracts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol

from yunmeng.interfaces.types import ArrayLike, ParamMeta

# ---------------------------------------------------
# region Shared values
# ---------------------------------------------------


@dataclass(frozen=True)
class ModelRef:
    """Versioned reference to a model artifact."""

    model_id: str
    version: str

    def __str__(self) -> str:
        return f"{self.model_id}@{self.version}"


# ---------------------------------------------------
# region IEstimable
# ---------------------------------------------------


class IEstimable(ABC):
    """Estimation target contract: parameter-vector access + run +
    gradient capability declaration.

    Framework entry point is the MODEL layer only:
    users call estimator.fit(model, ...). A solver MAY also
    implement this for standalone testing, but that is not
    a framework-guaranteed path.
    """

    # -- parameter vector ---------------------------

    @abstractmethod
    def param_spec(self) -> list[ParamMeta]:
        """Parameter descriptors list."""
        ...

    @abstractmethod
    def param_names(self) -> list[str]:
        """Ordered parameter names."""
        ...

    @abstractmethod
    def get_param_vector(self, names: list[str] = None) -> ArrayLike:
        """Flat parameter vector aligned with `names`."""
        ...

    @abstractmethod
    def set_param_vector(
        self,
        values: ArrayLike,
        names: list[str] = None,
    ):
        """Write a flat parameter vecto."""
        ...

    @abstractmethod
    def reset_run(self):
        """Reset to the initial state for a fresh evaluation,
        keeping the current parameter values."""
        ...

    def param_bounds(
        self,
        names: list[str] = None,
    ) -> tuple[list[float], list[float]]:
        """(lower, upper) bound lists aligned with *names*;
        None bounds become ±inf."""
        spec = {p.name: p for p in self.param_spec()}
        names = names or self.param_names()
        inf = float("inf")
        lo, hi = [], []
        for n in names:
            b = spec[n].bounds if n in spec else (None, None)
            lo.append(-inf if b[0] is None else float(b[0]))
            hi.append(inf if b[1] is None else float(b[1]))
        return lo, hi

    # -- run & gradient capability ------------------

    @abstractmethod
    def run(self, n_steps: int, **kwargs) -> Any:
        """Execute one forward pass of n_steps with current parameters.

        Returns an object from which observation-equivalent outputs
        can be extracted. In TRAIN mode the returned state must stay
        on the autograd graph.
        """
        ...

    @abstractmethod
    def supports_gradients(self) -> bool:
        """Whether end-to-end backpropagation is available NOW."""
        ...


def split_namespaces(names: list[str]) -> dict[str, list[str]]:
    """Group namespaced names by their first segment.

    ["sub1.K", "sub2.K"] -> {"sub1": ["K"], "sub2": ["K"]}
    """
    groups: dict[str, list[str]] = {}
    for n in names:
        head, _, tail = n.partition(".")
        groups.setdefault(head, []).append(tail)
    return groups


# ---------------------------------------------------
# region Estimator side
# ---------------------------------------------------


class MemoryStrategy(Enum):
    """Gradient memory strategy for unrolled long-horizon training.

    Full unrolled backprop stores the graph of every step: O(N) memory
    and exposure to gradient explosion/vanishing. Strategies trade
    compute for memory; they are orthogonal to the loss definition.
    """

    FULL = "full"
    """Store the whole graph. Only for short rollouts / debugging."""

    CHECKPOINT = "checkpoint"
    """Segment the rollout, store only segment boundaries, recompute
    within segments during backward (O(sqrt(N))-style memory)."""

    ADJOINT = "adjoint"
    """O(1) memory in time via adjoint-state integration backwards.
    Research-grade for coupled FVM; reserved, not yet implemented."""


class IObservationSet(Protocol):
    """Observation data for estimation (time series, gauges, fields)."""

    def variables(self) -> list[str]: ...

    def times(self) -> ArrayLike: ...

    def values(self, variable: str) -> ArrayLike: ...


class ILoss(Protocol):
    """Loss between model outputs and observations."""

    def __call__(
        self,
        predicted: Any,
        observed: IObservationSet,
    ) -> ArrayLike:
        """Returns a scalar array; differentiable under TRAIN."""
        ...


@dataclass
class EstimationResult:
    """Outcome of one estimation (calibration or training) run."""

    parameters: dict[str, Any] = field(default_factory=dict)
    """Inferred parameters theta (model refs, coefficients, ...)."""

    model_refs: list[tuple[str, str]] = field(default_factory=list)
    """Updated model artifact references: (model_id, version)."""

    metrics: dict[str, float] = field(default_factory=dict)
    """Evaluation metrics on validation/observation data."""

    history: list[dict] = field(default_factory=list)
    """Per-iteration loss / objective records."""

    converged: bool = False
    message: str = ""


class IEstimator(ABC):
    """Unified calibration/training abstraction: parameter inversion
    from observations (v1.0 §2 naming resolution).

    Implementations: GradientTrainer (unrolled AD, custom VJP hooks,
    MemoryStrategy) / Calibrator (SCE-UA, GLUE, ...; gradient-free).
    fit() entry sets the target to TRAIN (if applicable) and restores
    EVAL on exit or exception; entry MUST check supports_gradients()
    and fail with a suggested alternative estimator when unsatisfied.
    """

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """Unique estimator name."""
        pass

    @abstractmethod
    def fit(
        self,
        target: IEstimable,
        data: IObservationSet,
        loss: ILoss,
        **kwargs,
    ) -> EstimationResult:
        """Estimate model parameters."""
        ...

    @abstractmethod
    def evaluate(
        self,
        target: IEstimable,
        data: IObservationSet,
        **kwargs,
    ) -> dict[str, float]:
        """Evaluate model parameters."""
        pass


# ---------------------------------------------------
# region Model artifact
# ---------------------------------------------------


@dataclass
class TrainingMeta:
    """Provenance of a trained parameter set."""

    estimator: str = ""  # e.g. "GradientTrainer(checkpoint)"
    data_lineage: str = ""  # which observation set / episodes
    metrics: dict[str, float] = field(default_factory=dict)
    created_at: str = ""  # ISO-8601
    notes: str = ""


class IModelArtifactStore(Protocol):
    """Versioned persistence for parameters θ."""

    def resolve(self, ref: ModelRef) -> dict[str, ArrayLike]:
        """Fetch weights + normalization stats by reference.

        Must raise a error listing the required model_id@version
        when missing (snapshot load path relies on this)."""
        ...

    def register(
        self,
        params: Mapping[str, ArrayLike],
        meta: TrainingMeta,
    ) -> ModelRef:
        """Persist a newly trained parameter set as new version
        and return its reference."""
        ...

    def meta(self, ref: ModelRef) -> TrainingMeta: ...

    def list_versions(self, model_id: str) -> list[str]: ...
