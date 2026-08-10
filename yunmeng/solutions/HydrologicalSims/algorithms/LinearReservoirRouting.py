# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Three-source linear reservoir routing algorithm.
"""

from __future__ import annotations
import numpy as np

from yunmeng.solutions.standards import ParamMeta
from yunmeng.solutions.HydrologicalSims.algorithms.Bases import SurfaceRouting


class ThreeSourceLinearReservoir(SurfaceRouting):
    """Three-source linear reservoir routing algorithm.

    Each source (surface / interflow / groundwater) is routed through
    its own linear reservoir with recession time constant KS/KI/KG
    [s]; the discrete transfer uses the exact constant-input
    coefficient ``c = exp(-dt / K)``::

        q_t = c · q_{t-1} + (1 - c) · I_t

    with ``I = depth[mm] · area[km²] · 1000 / dt[s]`` (m3/s per mm).
    """

    algo_name = "linear3"

    @classmethod
    def inits_spec(cls) -> list[ParamMeta]:
        return [
            ParamMeta(
                "QS0",
                "initial surface outflow [m3/s]",
                bounds=(0.0, None),
                default=0.0,
            ),
            ParamMeta(
                "QI0",
                "initial interflow outflow [m3/s]",
                bounds=(0.0, None),
                default=0.0,
            ),
            ParamMeta(
                "QG0",
                "initial groundwater outflow [m3/s]",
                bounds=(0.0, None),
                default=0.0,
            ),
        ]

    @classmethod
    def param_spec(cls) -> list[ParamMeta]:
        return [
            ParamMeta(
                "KS",
                "surface reservoir constant [s]",
                bounds=(60.0, 7 * 86400.0),
                default=6 * 3600.0,
            ),
            ParamMeta(
                "KI",
                "interflow reservoir constant [s]",
                bounds=(3600.0, 30 * 86400.0),
                default=3 * 86400.0,
            ),
            ParamMeta(
                "KG",
                "groundwater reservoir constant [s]",
                bounds=(86400.0, 180 * 86400.0),
                default=15 * 86400.0,
            ),
        ]

    def __init__(self, **params):
        super().__init__(**params)
        self._qs = self.p("QS0")
        self._qi = self.p("QI0")
        self._qg = self.p("QG0")

    def state(self) -> dict:
        return {"QS": self._qs, "QI": self._qi, "QG": self._qg}

    def set_state(self, state: dict):
        self._qs = float(state["QS"])
        self._qi = float(state["QI"])
        self._qg = float(state["QG"])

    @property
    def components(self) -> tuple[float, float, float]:
        return self._qs, self._qi, self._qg

    def route(
        self, rs: float, ri: float, rg: float, area_km2: float, dt: float
    ) -> float:
        u = area_km2 * 1000.0 / dt  # m3/s per mm of depth
        cs = np.exp(-dt / self.p("KS"))
        ci = np.exp(-dt / self.p("KI"))
        cg = np.exp(-dt / self.p("KG"))
        self._qs = cs * self._qs + (1.0 - cs) * max(rs, 0.0) * u
        self._qi = ci * self._qi + (1.0 - ci) * max(ri, 0.0) * u
        self._qg = cg * self._qg + (1.0 - cg) * max(rg, 0.0) * u
        return self._qs + self._qi + self._qg
