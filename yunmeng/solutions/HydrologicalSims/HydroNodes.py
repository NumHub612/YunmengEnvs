# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Watershed model internal nodes: subbasin / station / reservoir.
"""

from __future__ import annotations
from typing import Any

from yunmeng.solutions.HydrologicalSims.algorithms import (
    create,
    RunoffGeneration,
    SurfaceRouting,
    RiverRouting,
    ReleasePolicy,
)
from yunmeng.solutions.commons.datasets import Curve

Q_DISCHARGE = "discharge"
Q_RUNOFF = "runoff"
Q_SOIL = "soil_moisture"
Q_LEVEL = "water_level"
Q_STORAGE = "storage"
Q_PRECIP = "precipitation"
Q_EVAP = "evaporation"


def normalize_inflow(entry) -> tuple[str, bool]:
    if isinstance(entry, dict):
        return str(entry["name"]), bool(entry.get("required", True))
    return str(entry), True


# ---------------------------------------------------
# region Storage curves
# ---------------------------------------------------


class LinearStorageCurve:
    """Linear storage-level relation:  S = S0 + A·(Z - Z0)."""

    def __init__(self, z0: float, s0: float, area_km2: float):
        self.z0 = float(z0)
        self.s0 = float(s0)
        self.area_m2 = float(area_km2) * 1e6
        if self.area_m2 <= 0:
            raise ValueError("storage curve area must be positive.")

    def level(self, storage: float) -> float:
        return self.z0 + (storage - self.s0) / self.area_m2

    def storage(self, level: float) -> float:
        return self.s0 + (level - self.z0) * self.area_m2


class TableStorageCurve:
    """Discrete Z~S relation (mono-increasing), linear interpolation."""

    def __init__(self, storages: list[float], levels: list[float]):
        self._curve = Curve("storage_curve", storages, levels)

    def level(self, storage: float) -> float:
        return self._curve.get_value(storage)

    def storage(self, level: float) -> float:
        return self._curve.inverse(level)


def build_storage_curve(cfg: dict):
    """Build a storage curve from node/model config."""
    method = cfg.get("method", "linear")
    params = cfg.get("params", {})
    if method == "linear":
        return LinearStorageCurve(**params)
    if method == "table":
        return TableStorageCurve(**params)
    raise ValueError(f"Unknown storage curve method '{method}'.")


# ---------------------------------------------------
# region HydrologyNode
# ---------------------------------------------------


class HydrologyNode:
    """Base class for an internal watershed node."""

    node_type = "node"
    output_vars: dict[str, str] = {}

    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.cfg = cfg
        self._current: dict[str, float] = {}
        self._inflows: dict[str, float] = {}  # slot -> m3/s this step
        self._external: dict[str, bool] = {}  # name -> required
        for entry in cfg.get("external_inflows", []):
            nm, req = normalize_inflow(entry)
            self._external[nm] = req
            self._inflows[f"in.{nm}"] = 0.0

    # -- topology -----------------------------------

    def internal_inflow_slot(self, upstream_name: str) -> str:
        return f"in.{upstream_name}"

    def add_internal_inflow(self, upstream_name: str):
        self._inflows[self.internal_inflow_slot(upstream_name)] = 0.0

    def external_slots(self) -> list[str]:
        """Slots not wired inside the model -> coupling input ports."""
        return [f"in.{nm}" for nm in self._external]

    def slot_required(self, slot: str) -> bool:
        """Whether an external slot must be fed (gauge or coupling)."""
        if slot.startswith("in."):
            return self._external.get(slot[3:], True)
        return True

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
    """Subbasin: precipitation/evaporation -> runoff -> routing -> outlet.

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
            "runoff",
            cfg["runoff"]["algo"],
            **cfg["runoff"].get(
                "params",
                {},
            ),
        )
        self._slope: SurfaceRouting = create(
            "surface",
            cfg["surface"]["algo"],
            **cfg["surface"].get(
                "params",
                {},
            ),
        )
        self._channel: RiverRouting = None
        if "river" in cfg:
            self._channel = create(
                "river",
                cfg["river"]["algo"],
                **cfg["river"].get(
                    "params",
                    {},
                ),
            )
        self._accept_inflow = bool(cfg.get("accept_inflow", False))
        self._current = {"Q": 0.0, "R": 0.0, "W": 0.0, "S": 0.0}

    def external_slots(self) -> list[str]:
        return ["P", "E"] + super().external_slots()

    def step(self, dt: float, t: float, inputs: dict[str, float]):
        p = max(inputs.get("P", 0.0), 0.0)
        e = max(inputs.get("E", 0.0), 0.0)

        rs, ri, rg = self._runoff.produce(p, e, dt)
        q = self._slope.route(rs, ri, rg, self.area_km2, dt)

        if self._accept_inflow or self._channel is not None or self._external:
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
        self._channel: RiverRouting = None
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


# ---------------------------------------------------
# region ReservoirNode
# ---------------------------------------------------


class ReservoirNode(HydrologyNode):
    """Reservoir: regulated storage node inside the basin tree.

    The release policy is a pluggable `ReleasePolicy` algorithm —
    "artificial operation" lives in the algorithm, not in the node
    structure, so naturalized rules and data-driven schedules are
    interchangeable.
    """

    node_type = "reservoir"
    output_vars = {
        "Q": Q_DISCHARGE,
        "QI": Q_DISCHARGE,
        "Z": Q_LEVEL,
        "S": Q_STORAGE,
    }

    def __init__(self, name: str, cfg: dict):
        super().__init__(name, cfg)
        sc = cfg["storage"]
        self._curve = build_storage_curve(sc["curve"])
        s0 = sc["curve"]["params"].get("s0", 0.0)
        self._storage = float(sc.get("s_init", s0))
        self._release: ReleasePolicy = create(
            "reservoir",
            cfg["reservoir"]["algo"],
            **cfg["reservoir"].get("params", {}),
        )
        self._current = {
            "Q": 0.0,
            "QI": 0.0,
            "Z": self._curve.level(self._storage),
            "S": self._storage,
        }

    @property
    def storage_curve(self):
        return self._curve

    @property
    def storage(self) -> float:
        return self._storage

    def step(self, dt: float, t: float, inputs: dict[str, float]):
        q_in = self.total_inflow()
        ctx = {"curve": self._curve, "node": self}
        q_out = self._release.release(self._storage, q_in, t, dt, ctx)
        self._storage = max(self._storage + (q_in - q_out) * dt, 0.0)
        self._current = {
            "Q": q_out,
            "QI": q_in,
            "Z": self._curve.level(self._storage),
            "S": self._storage,
        }

    def algorithms(self) -> dict[str, Any]:
        return {"release": self._release}

    def snapshot(self) -> dict:
        snap = super().snapshot()
        snap["storage"] = self._storage
        return snap

    def restore(self, snap: dict):
        super().restore(snap)
        self._storage = float(snap["storage"])


NODE_TYPES = {
    "subbasin": SubBasinNode,
    "station": StationNode,
    "reservoir": ReservoirNode,
}
