# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Reservoir target water level balanced releasing algorithm.
"""

from __future__ import annotations

from yunmeng.solutions.standards import ParamMeta
from yunmeng.solutions.commons.datasets import Timeseries
from yunmeng.solutions.HydrologicalSims.algorithms.Bases import ReleasePolicy


class TargetLevelRelease(ReleasePolicy):
    """Reservoir target water level balanced releasing algorithm.

    Calculate the outbound inventory step by step according to
    the principle of
    'return to target storage capacity at the end':

        Q = clip( (S + I·Δt - S_target(t)) / Δt,  q_min,  q_max )

    And add two safety rules:
      * When the reservoir volume exceeds the flood control limit `S_flood`,
      release water at `q_max`;
      * When the reservoir volume is below the dead storage `S_dead`,
      only allow outflow, no inflow, changed to `q_min` (can be 0).

    The target storage can be a constant `target_storage`,
    or it can be a rule curve that changes over time `target_series`
    (Timeseries, interpolated at the current moment).
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

    def on_params_changed(self):
        pass

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
        s_next = storage + max(inflow, 0.0) * dt
        q = (s_next - target) / dt

        flood = self._params.get("flood_storage")
        if flood is not None and s_next > flood:
            q = self.p("q_max")
        dead = self.p("dead_storage")
        if s_next <= dead:
            q = 0.0

        # never release more water than is (or will be) available
        q = min(max(q, self.p("q_min")), self.p("q_max"))
        q = min(q, s_next / dt)
        return max(q, 0.0)

    def plan(
        self,
        initial_storage: float,
        inflows: list[float],
        timestamps: list[float],
        dt: float,
        context: dict,
    ) -> list[float]:
        outflows = []
        storage = initial_storage
        for inflow in inflows:
            outflows.append(
                self.release(storage, inflow, timestamps[0], dt, context=context)
            )
            storage += inflow * dt - outflows[-1] * dt
            timestamps.pop(0)
        return outflows
