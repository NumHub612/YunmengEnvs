# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

IEstimable: the estimation target abstraction (design doc v1.4).

MERGE NOTE: this interface ABSORBS the former
`solutions.standards.IAdditional.IParametric` (flat, named, bounded
parameter vector + run reset), which existed solely for calibration.
IParametric is kept as a deprecated alias — do not use it in new code.

Both layers can implement IEstimable:
  - ILinkableModel (end users, via the IEstimableModel mixin): parameters
    map to ModelMeta.parameters / PARAMS yaml declarations. Black-box,
    gradient-free calibration via Calibrator. A model need NOT be
    solver-based (e.g. lumped hydrological models).
  - ISolver (framework developers): white-box access, gradient-based
    end-to-end training via GradientTrainer.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import numpy as np

from yunmeng.numerics.enums import RunMode
from yunmeng.solutions.standards.IModel import ParamMeta

# ---------------------------------------------------
# region IEstimable
# ---------------------------------------------------


class IEstimable(ABC):
    """Mixin for linkable models that support estimation (calibration or
    training) with a unified user experience at the model layer.

    Contract = parameter-vector access (calibration-ready) + run +
    gradient capability declaration.
    """

    @property
    @abstractmethod
    def mode(self) -> RunMode:
        """Current run mode of the model (and internal components)."""
        pass

    # -- parameter vector ---------------------------

    @abstractmethod
    def param_spec(self) -> list[ParamMeta]:
        """Parameter descriptors (list[ParamMeta])."""
        pass

    @abstractmethod
    def param_names(self) -> list[str]:
        """Ordered parameter names."""
        pass

    @abstractmethod
    def get_param_vector(self, names: list[str] = None) -> np.ndarray:
        """Flat parameter vector aligned with `names` (default: all)."""
        pass

    @abstractmethod
    def set_param_vector(
        self,
        values: np.ndarray,
        names: list[str] = None,
    ):
        """Write a flat parameter vector."""
        pass

    @abstractmethod
    def reset_run(self):
        """Reset to the initial state for a fresh evaluation,
        keeping the current parameter values."""
        pass

    def param_bounds(
        self,
        names: list[str] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """(lower, upper) bound arrays aligned with *names*;
        None bounds become ±inf."""
        spec = {p.name: p for p in self.param_spec()}
        names = names or self.param_names()
        lo, hi = [], []
        for n in names:
            b = spec[n].bounds if n in spec else (None, None)
            lo.append(-np.inf if b[0] is None else b[0])
            hi.append(np.inf if b[1] is None else b[1])
        return np.array(lo, dtype=float), np.array(hi, dtype=float)

    # -- run & gradient capability ------------------

    @abstractmethod
    def run(self, **kwargs) -> Any:
        """Execute one full forward pass with current parameters.

        Returns an object from which observation-equivalent
        outputs can be extracted. In `TRAIN` mode the returned
        state must stay on the autograd graph.
        """
        pass

    @classmethod
    def supports_gradient(cls) -> bool:
        """Whether end-to-end backpropagation is available.

        True requires differentiable solver chain (torch
        backend + torch-ized operators). False for gradient-free.
        """
        return False

    def train(self):
        """Switch to TRAIN mode and propagate to internals."""
        self._set_mode_recursive(RunMode.TRAIN)

    def eval(self):
        """Switch to EVAL mode and propagate to internals."""
        self._set_mode_recursive(RunMode.EVAL)

    @abstractmethod
    def _set_mode_recursive(self, mode: RunMode):
        """Set own mode and propagate: model->solvers->datahubs."""
        pass


def split_namespaces(names: list[str]) -> dict[str, list[str]]:
    """Group namespaced names by their first segment.

    ["sub1.K", "sub1.CI", "sub2.K"] -> {"sub1": ["K", "CI"], "sub2": ["K"]}
    """
    groups: dict[str, list[str]] = {}
    for n in names:
        head, _, tail = n.partition(".")
        groups.setdefault(head, []).append(tail)
    return groups
