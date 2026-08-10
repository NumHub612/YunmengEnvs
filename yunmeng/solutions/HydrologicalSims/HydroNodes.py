# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Watershed model internal nodes: subbasin / station / reservoir.
"""

from __future__ import annotations
from typing import Any, Optional

from yunmeng.solutions.HydrologicalSims.algorithms import (
    create,
    RunoffGeneration,
    SurfaceRouting,
    RiverRouting,
)

#: quantity names advertised by each variable/slot
Q_DISCHARGE = "discharge"
Q_RUNOFF = "runoff"
Q_SOIL = "soil_moisture"
Q_LEVEL = "water_level"
Q_STORAGE = "storage"
Q_PRECIP = "precipitation"
Q_EVAP = "evaporation"

# ---------------------------------------------------
# region HydrologyNode
# ---------------------------------------------------


class HydrologyNode:
    """Base class for an internal watershed node."""

    node_type = "node"

    #: variable name -> quantity name
    output_vars: dict[str, str] = {}

    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.cfg = cfg
        self._current: dict[str, float] = {}
        self._inflows: dict[str, float] = {}  # slot -> m3/s this step
        for slot in cfg.get("external_inflows", []):
            self._inflows[f"in.{slot}"] = 0.0

    # -- topology -----------------------------------

    def internal_inflow_slot(self, upstream_name: str) -> str:
        return f"in.{upstream_name}"

    def add_internal_inflow(self, upstream_name: str):
        self._inflows[self.internal_inflow_slot(upstream_name)] = 0.0

    def input_slots(self) -> dict[str, str]:
        """slot name -> quantity name, for ALL inflow slots."""
        return {s: Q_DISCHARGE for s in self._inflows}

    def external_slots(self) -> list[str]:
        """Slots not wired inside the model -> coupling input ports."""
        return [f"in.{s}" for s in self.cfg.get("external_inflows", [])]

    # -- data flow ----------------------------------

    def set_inflow(self, slot: str, q: float):
        if slot not in self._inflows:
            raise KeyError(f"{self.name}: unknown inflow slot '{slot}'.")
        self._inflows[slot] = float(q)

    def total_inflow(self) -> float:
        return sum(self._inflows.values())

    def current(self, var: str) -> float:
        try:
            return self._current[var]
        except KeyError:
            raise KeyError(
                f"{self.name}: no variable '{var}' "
                f"(available: {sorted(self._current)})."
            ) from None

    # -- lifecycle ----------------------------------

    def step(self, dt: float, t: float, inputs: dict[str, float]):
        raise NotImplementedError

    def algorithms(self) -> dict[str, Any]:
        """name -> pluggable algorithm instance."""
        return {}

    def snapshot(self) -> dict:
        return {
            "current": dict(self._current),
            "inflows": dict(self._inflows),
            "algos": {n: a.snapshot() for n, a in self.algorithms().items()},
        }

    def restore(self, snap: dict):
        self._current = dict(snap["current"])
        self._inflows = dict(snap["inflows"])
        for n, s in snap["algos"].items():
            self.algorithms()[n].restore(s)


# ---------------------------------------------------
# region SubBasinNode
# ---------------------------------------------------


class SubBasinNode(HydrologyNode):
    """Subbasin: precipitation/evaporation -> runoff -> surface -> river -> outlet.

    Input slots:  P [mm], E [mm], optional in.* [m3/s].
    Output vars:  Q [m3/s], R [mm], W [mm], S [mm].
    """

    node_type = "subbasin"
    output_vars = {"Q": Q_DISCHARGE, "R": Q_RUNOFF, "W": Q_SOIL, "S": Q_SOIL}

    def __init__(self, name: str, cfg: dict):
        super().__init__(name, cfg)
        self.area_km2 = float(cfg["area"])
        if self.area_km2 <= 0:
            raise ValueError(f"{name}: area must be positive.")
        self._runoff: RunoffGeneration = create(
            "runoff", cfg["runoff"]["algo"], **cfg["runoff"].get("params", {})
        )
        self._slope: SurfaceRouting = create(
            "surface", cfg["surface"]["algo"], **cfg["surface"].get("params", {})
        )
        self._channel: Optional[RiverRouting] = None
        if "river" in cfg:
            self._channel = create(
                "river", cfg["river"]["algo"], **cfg["river"].get("params", {})
            )
        self._accept_inflow = bool(cfg.get("accept_inflow", False))
        self._current = {"Q": 0.0, "R": 0.0, "W": self._wu0(), "S": 0.0}

    def _wu0(self) -> float:
        return 0.0

    def input_slots(self) -> dict[str, str]:
        slots = {"P": Q_PRECIP, "E": Q_EVAP}
        slots.update(super().input_slots())
        return slots

    def external_slots(self) -> list[str]:
        # P / E are always external (fed by gauges); inflow slots follow
        # the base rule.
        return ["P", "E"] + super().external_slots()

    def step(self, dt: float, t: float, inputs: dict[str, float]):
        p = max(inputs.get("P", 0.0), 0.0)
        e = max(inputs.get("E", 0.0), 0.0)

        rs, ri, rg = self._runoff.produce(p, e, dt)
        q = self._slope.route(rs, ri, rg, self.area_km2, dt)

        if self._accept_inflow or self._channel is not None:
            q += self.total_inflow()
        if self._channel is not None:
            q = self._channel.route(q, dt)

        st = self._runoff.state()
        self._current = {
            "Q": q,
            "R": rs + ri + rg,
            "W": st.get("WU", 0.0) + st.get("WL", 0.0) + st.get("WD", 0.0),
            "S": st.get("S", 0.0),
        }

    def algorithms(self) -> dict[str, Any]:
        algos = {"runoff": self._runoff, "surface": self._slope}
        if self._channel is not None:
            algos["river"] = self._channel
        return algos


# ---------------------------------------------------
# region StationNode
# ---------------------------------------------------


class StationNode(HydrologyNode):
    """Station: multi-river inflow -> river routing -> outlet.

    Input slots:  in.* [m3/s] (internal upstreams + external_inflows).
    Output vars:  Q [m3/s].
    """

    node_type = "station"
    output_vars = {"Q": Q_DISCHARGE}

    def __init__(self, name: str, cfg: dict):
        super().__init__(name, cfg)
        self._channel: Optional[RiverRouting] = None
        if "river" in cfg:
            self._channel = create(
                "river", cfg["river"]["algo"], **cfg["river"].get("params", {})
            )
        self._current = {"Q": 0.0}

    def step(self, dt: float, t: float, inputs: dict[str, float]):
        q = self.total_inflow()
        if self._channel is not None:
            q = self._channel.route(q, dt)
        self._current = {"Q": q}

    def algorithms(self) -> dict[str, Any]:
        return {"river": self._channel} if self._channel is not None else {}


NODE_TYPES = {
    "subbasin": SubBasinNode,
    "station": StationNode,
}
