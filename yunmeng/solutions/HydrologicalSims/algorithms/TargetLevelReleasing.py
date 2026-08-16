# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Reservoir target water level balanced releasing algorithm.
"""

from __future__ import annotations

from yunmeng.solutions.standards import ParamMeta
from yunmeng.solutions.commons.datasets import Timeseries
from yunmeng.solutions.HydrologicalSims.algorithms.Bases import ReleasePolicy, register


class TargetLevelRelease(ReleasePolicy):
    """Release to return to the target storage at the end of each step:

        Q = clip( (S + I·Δt - S_target(t)) / Δt,  q_min,  q_max )

    Safety rules:
      * storage above ``flood_storage`` -> release at q_max;
      * storage at/below ``dead_storage`` -> no release.

    The target can be a constant ``target_storage`` or a rule curve
    ``target_series`` (Timeseries, interpolated at the current time).
    """

    algo_name = "target_level"

    @classmethod
    def inits_spec(cls) -> list[ParamMeta]:
        return []

    @classmethod
    def param_spec(cls) -> list[ParamMeta]:
        return [
            ParamMeta(
                "target_storage",
                "target storage [m3]",
                bounds=(0.0, None),
                default=None,
                required=False,
            ),
            ParamMeta(
                "q_min",
                "minimum release [m3/s]",
                bounds=(0.0, None),
                default=0.0,
            ),
            ParamMeta(
                "q_max",
                "maximum release [m3/s]",
                bounds=(0.0, None),
                default=1e12,
            ),
            ParamMeta(
                "flood_storage",
                "flood-control limit storage [m3]",
                bounds=(0.0, None),
                default=None,
                required=False,
            ),
            ParamMeta(
                "dead_storage",
                "dead storage [m3]",
                bounds=(0.0, None),
                default=0.0,
            ),
        ]

    def __init__(self, target_series: Timeseries = None, **params):
        super().__init__(**params)
        if "target_storage" not in self._params and target_series is None:
            raise ValueError(
                "TargetLevelRelease: give 'target_storage' or a target_series."
            )
        if self.p("q_min") > self.p("q_max"):
            raise ValueError("TargetLevelRelease: q_min must be <= q_max.")
        self._target_series = target_series

    def _target(self, t: float) -> float:
        if self._target_series is not None:
            return self._target_series.get_value(t)
        return self.p("target_storage")

    def state(self) -> dict:
        return {}

    def set_state(self, state: dict):
        pass

    def release(
        self, storage: float, inflow: float, t: float, dt: float, context: dict = None
    ) -> float:
        target = self._target(t)
        s_next = storage + inflow * dt  # FIX: negative inflow (pumping) allowed
        q = (s_next - target) / dt

        flood = self._params.get("flood_storage")
        if flood is not None and s_next > flood:
            q = self.p("q_max")
        if s_next <= self.p("dead_storage"):
            q = 0.0

        q = min(max(q, self.p("q_min")), self.p("q_max"))
        q = min(q, max(s_next, 0.0) / dt)  # never release more than available
        return max(q, 0.0)

    def plan(
        self,
        initial_storage: float,
        inflows: list[float],
        timestamps: list[float],
        dt: float,
        context: dict,
    ) -> list[float]:
        """Plan the full outflow sequence.  Does NOT mutate inputs."""
        if len(inflows) != len(timestamps):
            raise ValueError(
                f"TargetLevelRelease.plan: inflows ({len(inflows)}) and "
                f"timestamps ({len(timestamps)}) length mismatch."
            )
        outflows = []
        storage = initial_storage
        for inflow, t in zip(inflows, timestamps):  # FIX: no pop(0)
            q = self.release(storage, inflow, t, dt, context=context)
            outflows.append(q)
            storage += (inflow - q) * dt
        return outflows


register("reservoir", TargetLevelRelease.algo_name, TargetLevelRelease)
