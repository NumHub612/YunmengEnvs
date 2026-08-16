# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Muskingum river routing algorithm.
"""

from __future__ import annotations
from yunmeng.solutions.standards import ParamMeta
from yunmeng.solutions.HydrologicalSims.algorithms.Bases import RiverRouting, register


class MuskingumRouting(RiverRouting):
    """Muskingum river routing algorithm.

    O₂ = C₀·I₂ + C₁·I₁ + C₂·O₁  with

        C₀ = (Δt - 2Kx) / D,  C₁ = (Δt + 2Kx) / D,  C₂ = (2K(1-x) - Δt) / D
        D = 2K(1-x) + Δt

    K in seconds (same unit as dt).
    """

    algo_name = "muskingum"

    @classmethod
    def inits_spec(cls) -> list[ParamMeta]:
        return [
            ParamMeta(
                "I0",
                "initial inflow [m3/s]",
                bounds=(0.0, None),
                default=0.0,
            ),
            ParamMeta(
                "O0",
                "initial outflow [m3/s]",
                bounds=(0.0, None),
                default=0.0,
            ),
        ]

    @classmethod
    def param_spec(cls) -> list[ParamMeta]:
        return [
            ParamMeta(
                "K",
                "reach storage time constant [s]",
                bounds=(60.0, 30 * 86400.0),
                default=6 * 3600.0,
            ),
            ParamMeta(
                "X",
                "weighting factor",
                bounds=(0.0, 0.5),
                default=0.2,
            ),
        ]

    def __init__(self, **params):
        super().__init__(**params)
        self._i1 = self.p("I0")
        self._o1 = self.p("O0")
        self._coefs: tuple[float, float, float] | None = None
        self._coefs_dt: float | None = None
        self._neg_volume = 0.0  # cumulative clipped negative outflow [m3/s·steps]

    def on_params_changed(self):
        self._coefs = None
        self._coefs_dt = None

    def _coefficients(self, dt: float) -> tuple[float, float, float]:
        if self._coefs is not None and self._coefs_dt == dt:
            return self._coefs
        k, x = self.p("K"), self.p("X")
        d = 2.0 * k * (1.0 - x) + dt
        c0 = (dt - 2.0 * k * x) / d
        c1 = (dt + 2.0 * k * x) / d
        c2 = (2.0 * k * (1.0 - x) - dt) / d
        if c0 < 0:
            raise ValueError(
                f"MuskingumRouting: C0={c0:.4f} < 0 for dt={dt}s, K={k}s, "
                f"x={x}; decrease dt or x (need dt >= 2Kx)."
            )
        self._coefs = (c0, c1, c2)
        self._coefs_dt = dt
        return self._coefs

    @property
    def negative_volume(self) -> float:
        """Cumulative outflow volume clipped at zero (mass-balance audit)."""
        return self._neg_volume

    def state(self) -> dict:
        return {"I1": self._i1, "O1": self._o1, "NEG_VOLUME": self._neg_volume}

    def set_state(self, state: dict):
        self._i1 = float(state["I1"])
        self._o1 = float(state["O1"])
        self._neg_volume = float(state.get("NEG_VOLUME", 0.0))

    def route(self, q_in: float, dt: float) -> float:
        c0, c1, c2 = self._coefficients(dt)
        o2 = c0 * q_in + c1 * self._i1 + c2 * self._o1
        if o2 < 0.0:
            # numerical undershoot: clip but keep an audit trail
            self._neg_volume += -o2 * dt
            o2 = 0.0
        self._i1 = float(q_in)
        self._o1 = float(o2)
        return self._o1


register("river", MuskingumRouting.algo_name, MuskingumRouting)
