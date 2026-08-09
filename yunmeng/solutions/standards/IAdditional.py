# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Parameter-vector interface for calibration and training.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from yunmeng.solutions.standards.IModel import ParamMeta, ExchangeMeta
from yunmeng.solutions.standards.ITopology import ISpatialIndex, ITopologyLayer

# ---------------------------------------------------
# region Parametric
# ---------------------------------------------------


class IParametric(ABC):
    """Flat, named, bounded parameter vector plus run reset."""

    @abstractmethod
    def param_spec(self) -> list[ParamMeta]:
        """Parameter descriptors."""
        pass

    @abstractmethod
    def param_names(self) -> list[str]:
        """Ordered names."""
        pass

    @abstractmethod
    def get_param_vector(self, names: list[str] = None) -> np.ndarray:
        pass

    @abstractmethod
    def set_param_vector(
        self,
        values: np.ndarray,
        names: list[str] = None,
    ):
        pass

    @abstractmethod
    def reset_run(self):
        """Reset to the initial state for a fresh evaluation,
        keeping the current parameter values."""
        pass

    # -- helpers ------------------------------------

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

        lo = np.array(lo, dtype=float)
        hi = np.array(hi, dtype=float)
        return lo, hi


def split_namespaces(names: list[str]) -> dict[str, list[str]]:
    """Group namespaced names by their first segment.

    ["sub1.K", "sub1.CI", "sub2.K"] -> {"sub1": ["K", "CI"], "sub2": ["K"]}
    """
    groups: dict[str, list[str]] = {}
    for n in names:
        head, _, tail = n.partition(".")
        groups.setdefault(head, []).append(tail)
    return groups


# ---------------------------------------------------
# region IInternalTopology
# ---------------------------------------------------


class IInternalTopology(ABC):
    """Interface for components that contain internal topology."""

    @property
    def has_internal_topology(self) -> bool:
        """Whether this model has meaningful internal topology."""
        return False

    @abstractmethod
    def get_layers(self) -> list[ITopologyLayer]:
        """Return all topology layers, outermost first."""
        pass

    def get_layer(self, layer_id: str) -> ITopologyLayer:
        """Convenience: fetch a layer by its id."""
        for layer in self.get_layers():
            if layer.layer_id == layer_id:
                return layer
        return None

    def get_spatial_index(self, layer_id: str = "") -> ISpatialIndex:
        """Return a spatial index for the given layer.

        If *layer_id* is empty, the finest (innermost)
        layer is used. Returns None if the layer has
        no spatial extent (e.g. SCALAR).
        """
        return None

    def get_exposed_ports(self) -> list[ExchangeMeta]:
        """Return subset of internal nodes that are exposed
        as coupling ports to other components.
        """
        return []
