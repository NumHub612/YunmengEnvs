# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

ReservoirModel — independent reservoir regulation model.
"""

from __future__ import annotations
from typing import Optional

import numpy as np

from yunmeng.solutions.standards import (
    IStateful,
    IParametric,
    ModelMeta,
    ModelStatus,
    ParamMeta,
    TimeSpan,
)
from yunmeng.solutions.commons.models import BaseModel, BaseInput
from yunmeng.solutions.commons.datasets import ScalarElementSet, Quantities
from yunmeng.solutions.HydrologicalSims.algorithms import create, ReleasePolicy


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


class ReservoirModel(BaseModel, IStateful, IParametric):
    """Stand-alone reservoir regulation model."""

    def __init__(self, config: dict):
        model_id = config["id"]
        super().__init__(model_id, ModelMeta(name=model_id, category="reservoir"))
        self._cfg = config
        self._dt = float(config.get("dt", 86400.0))
        self._steps = int(config.get("steps", 0))
        self._start = float(config.get("start", 0.0))
        self._cursor = 0

        sc = config["storage"]
        self._curve = LinearStorageCurve(sc["z0"], sc["s0"], sc["area_km2"])
        self._storage = float(sc.get("s_init", sc["s0"]))
        release_cfg = config["release"]
        self._release: ReleasePolicy = create(
            "release", release_cfg["algo"], **release_cfg.get("params", {})
        )

        ts = TimeSpan(start=self._start, step=self._dt)
        self._inflow_ports: dict[str, BaseInput] = {}
        for entry in config.get("inflows", []):
            if isinstance(entry, dict):
                name, required = entry["name"], bool(entry.get("required", True))
            else:
                name, required = str(entry), True
            port = BaseInput(
                f"{model_id}.in.{name}",
                Quantities.DISCHARGE,
                ScalarElementSet(name),
                time_span=ts,
                owner=self,
                required=required,
            )
            self.add_input(port)
            self._inflow_ports[name] = port

        for var, q in (
            ("Q_out", Quantities.DISCHARGE),
            ("Q_in", Quantities.DISCHARGE),
            ("Z", Quantities.WATER_LEVEL),
            ("S", Quantities.STORAGE),
        ):
            self.create_output(
                q, ScalarElementSet(model_id), port_id=f"{model_id}.{var}"
            )

        self._current = {
            "Q_out": 0.0,
            "Q_in": 0.0,
            "Z": self._curve.level(self._storage),
            "S": self._storage,
        }
        self._initial_snapshot: Optional[dict] = None

    # -- info -------------------------------------------------

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def current_time(self) -> float:
        return self._start + self._cursor * self._dt

    @property
    def storage_curve(self) -> LinearStorageCurve:
        return self._curve

    # -- lifecycle ----------------------------------------------

    def _do_initialize(self):
        self._initial_snapshot = self.snapshot()

    def _do_update(self, inquirers=None):
        q_in = 0.0
        for name, port in self._inflow_ports.items():
            if port.is_connected:
                q_in += float(np.asarray(port.pull()).flat[0])
            elif port.required:
                raise ValueError(
                    f"{self._id}: required inflow port '{port.id}' is "
                    f"not connected."
                )

        ctx = {"curve": self._curve, "storage": self._storage, "model": self}
        q_out = self._release.release(
            self._storage, q_in, self.current_time, self._dt, ctx
        )
        self._storage = max(self._storage + (q_in - q_out) * self._dt, 0.0)

        self._current = {
            "Q_out": q_out,
            "Q_in": q_in,
            "Z": self._curve.level(self._storage),
            "S": self._storage,
        }
        for var, value in self._current.items():
            self.get_output(f"{self._id}.{var}").add_values([value])

        self._cursor += 1
        if 0 < self._steps <= self._cursor:
            self.mark_done()

    def _do_finish(self):
        self._cursor = 0

    # -- IStateful -------------------------------------------------

    def snapshot(self) -> dict:
        return {
            "cursor": self._cursor,
            "storage": self._storage,
            "release": self._release.snapshot(),
            "ports": {p.id: p._state() for p in self._outputs},
        }

    def restore(self, snapshot: dict):
        self._cursor = int(snapshot["cursor"])
        self._storage = float(snapshot["storage"])
        self._release.restore(snapshot["release"])
        for p in self._outputs:
            state = snapshot["ports"].get(p.id)
            if state is not None:
                p._set_state(state)
        if self._status != ModelStatus.CREATED:
            self._status = ModelStatus.READY

    # -- IParametric --------------------------------------------------

    def param_spec(self) -> list[ParamMeta]:
        spec = []
        for meta in self._release.param_spec():
            if meta.name in self._release.get_params():
                spec.append(
                    ParamMeta(
                        name=f"release.{meta.name}",
                        description=meta.description,
                        bounds=meta.bounds,
                        default=meta.default,
                    )
                )
        return spec

    def param_names(self) -> list[str]:
        return [m.name for m in self.param_spec()]

    def get_param_vector(self, names: list[str] = None) -> np.ndarray:
        names = names or self.param_names()
        params = self._release.get_params([n.split(".")[-1] for n in names])
        return np.array([params[n.split(".")[-1]] for n in names])

    def set_param_vector(self, values: np.ndarray, names: list[str] = None):
        names = names or self.param_names()
        if len(values) != len(names):
            raise ValueError("values / names length mismatch.")
        self._release.set_params(
            {n.split(".")[-1]: float(v) for n, v in zip(names, values)}
        )

    def reset_run(self):
        if self._initial_snapshot is None:
            raise RuntimeError(f"{self._id}: not initialized yet.")
        self.restore(self._initial_snapshot)
