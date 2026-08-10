# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Pluggable hydrological algorithm interfaces and registry.

Four algorithm *kinds* exist, one per computational role:

  runoff    — RunoffGeneration:  rainfall/evap -> (RS, RI, RG) depths [mm]
  slope     — SlopeRouting:      (RS, RI, RG) over area -> outlet flow [m3/s]
  river     — RiverRouting:    inflow -> outflow along a reach [m3/s]
  release   — ReleasePolicy:     reservoir storage/inflow -> outflow [m3/s]

New algorithms register via ``register(kind, name, cls)`` and are
instantiated through ``create(kind, name, **params)`` — components only
ever talk to the interfaces, never to concrete classes.

All algorithms are stateful, expose a flat parameter vector (for
calibration), and support snapshot/restore (for iterative coupling and
run resets).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

from yunmeng.solutions.standards import ParamMeta

# ---------------------------------------------------
# region Base
# ---------------------------------------------------


class HydroAlgorithm(ABC):
    """Base class for all pluggable hydrological algorithms."""

    #: human-readable name
    algo_name: str = ""

    def __init__(self, **params):
        spec = {p.name: p for p in self.param_spec() + self.inits_spec()}
        unknown = set(params) - set(spec)
        if unknown:
            raise ValueError(
                f"{type(self).__name__}: unknown parameters {sorted(unknown)}; "
                f"known: {sorted(spec)}."
            )
        self._params: dict[str, float] = {}
        for name, meta in spec.items():
            if name in params:
                value = float(params[name])
            elif meta.default is not None:
                value = float(meta.default)
            elif meta.required:
                raise ValueError(
                    f"{type(self).__name__}: missing required parameter '{name}'."
                )
            else:
                continue
            lo, hi = meta.bounds
            if lo is not None and value < lo or hi is not None and value > hi:
                raise ValueError(
                    f"{type(self).__name__}: parameter '{name}'={value} "
                    f"outside bounds {meta.bounds}."
                )
            self._params[name] = value
        self._initial_state = None

    # -- parameters --------------------------------------

    @classmethod
    @abstractmethod
    def inits_spec(cls) -> list[ParamMeta]:
        """Initial state descriptors (name, bounds, defaults)."""
        pass

    @classmethod
    @abstractmethod
    def param_spec(cls) -> list[ParamMeta]:
        """Parameter descriptors (name, bounds, defaults)."""
        pass

    def get_params(self, names: list[str] = None) -> dict[str, float]:
        names = names or list(self._params)
        return {n: self._params[n] for n in names}

    def set_params(self, values: dict[str, float]):
        for n, v in values.items():
            if n not in self._params:
                raise KeyError(f"{type(self).__name__}: unknown parameter '{n}'.")
            self._params[n] = float(v)
        self.on_params_changed()

    def on_params_changed(self):
        """Recompute derived coefficients after parameter change."""
        pass

    def p(self, name: str) -> float:
        return self._params[name]

    # -- state --------------------------------------------

    @abstractmethod
    def state(self) -> dict[str, Any]:
        """Current internal state (flat dict of floats)."""
        pass

    @abstractmethod
    def set_state(self, state: dict[str, Any]):
        """Overwrite internal state."""
        pass

    def snapshot(self) -> dict:
        return dict(self.state())

    def restore(self, snapshot: dict):
        self.set_state(dict(snapshot))

    def reset(self):
        """Back to the post-construction state."""
        if self._initial_state is None:
            self._initial_state = self.snapshot()
        else:
            self.restore(self._initial_state)

    def freeze_initial_state(self):
        """Capture the current state as the reset point (called by
        components after warm-up configuration)."""
        self._initial_state = self.snapshot()


# ---------------------------------------------------
# region Interfaces
# ---------------------------------------------------


class RunoffGeneration(HydroAlgorithm):
    """Runoff generation algorithm: rainfall/evaporation -> runoff
    depths [mm]."""

    @abstractmethod
    def produce(
        self, rain: float, evap: float, dt: float
    ) -> tuple[float, float, float]:
        """Generate one step runoff depth.

        Args:
            rain: areal rainfall this step [mm].
            evap: potential evaporation depth this step [mm].
            dt: step size [s].

        Returns:
            (RS, RI, RG): surface / interflow / groundwater
            runoff depths over the whole sub-basin.
        """


class SurfaceRouting(HydroAlgorithm):
    """Surface routing algorithm: runoff depth -> sub-basin outlet
    discharge [m3/s]."""

    @abstractmethod
    def route(
        self, rs: float, ri: float, rg: float, area_km2: float, dt: float
    ) -> float:
        """Route one step of runoff depth to outlet discharge.

        Args:
            rs: surface runoff depth [mm].
            ri: interflow runoff depth [mm].
            rg: groundwater runoff depth [mm].
            area_km2: sub-basin area [km2].
            dt: step size [s].
        """
        pass


class RiverRouting(HydroAlgorithm):
    """River routing algorithm: inflow -> outflow."""

    @abstractmethod
    def route(self, q_in: float, dt: float) -> float:
        """Route one step of inflow to outflow."""
        pass


class ReleasePolicy(HydroAlgorithm):
    """Reservoir release policy:
    reservoir storage, inflow -> outflow.
    """

    @abstractmethod
    def release(
        self,
        storage: float,
        inflow: float,
        t: float,
        dt: float,
        context: dict,
    ) -> float:
        """Decide the outflow for this step.

        Args:
            storage: current reservoir storage [m3].
            inflow: total inflow this step [m3/s].
            t: current timestamp [s].
            dt: step size [s].
            context: extra node info
            (e.g. capacity, operation constraints).
        """
        pass

    @abstractmethod
    def plan(
        self,
        initial_storage: float,
        inflows: list[float],
        timestamps: list[float],
        dt: float,
        context: dict,
    ) -> list[float]:
        """Plan the complete outflow sequence based
        on the global inflow process.

        Args:
            initial_storage: The starting reservoir storage [m3].
            inflows: Total inflows for each step [m3/s].
            timestamps: Timestamps [s].
            dt: The uniform step size [s].
            context: Extra node info
            (e.g. capacity, operational constraints).
        """
        pass


# ---------------------------------------------------
# region Registry
# ---------------------------------------------------


_REGISTRY_HYDROS: dict[str, dict[str, type[HydroAlgorithm]]] = {
    "runoff": {},
    "surface": {},
    "river": {},
    "reservoir": {},
}


def register(kind: str, name: str, cls: type[HydroAlgorithm]):
    if kind not in _REGISTRY_HYDROS:
        raise KeyError(
            f"Unknown algorithm kind '{kind}' (known: {list(_REGISTRY_HYDROS)})."
        )
    _REGISTRY_HYDROS[kind][name] = cls


def available(kind: str = None) -> dict:
    if kind is not None:
        return dict(_REGISTRY_HYDROS[kind])
    return {k: sorted(v) for k, v in _REGISTRY_HYDROS.items()}


def create(kind: str, name: str, **params) -> HydroAlgorithm:
    try:
        cls = _REGISTRY_HYDROS[kind][name]
    except KeyError:
        raise KeyError(
            f"No {kind} algorithm named '{name}'; "
            f"available: {sorted(_REGISTRY_HYDROS.get(kind, {}))}."
        ) from None
    return cls(**params)
