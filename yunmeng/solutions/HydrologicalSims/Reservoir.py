# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

ReservoirModel — stand-alone reservoir regulation model.
"""

from __future__ import annotations
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
from yunmeng.solutions.HydrologicalSims.HydroNodes import ReservoirNode


class ReservoirModel(BaseModel, IStateful, IParametric):
    """Stand-alone reservoir regulation model (composes a ReservoirNode)."""

    def __init__(self, config: dict):
        model_id = config["id"]
        super().__init__(model_id, ModelMeta(name=model_id, category="reservoir"))
        self._cfg = config
        self._dt = float(config.get("dt", 86400.0))
        self._steps = int(config.get("steps", 0))
        self._start = float(config.get("start", 0.0))
        self._cursor = 0

        # single source of truth for the water balance
        self._node = ReservoirNode(
            model_id,
            {
                "storage": config["storage"],
                "reservoir": config["reservoir"],
                "external_inflows": config.get("inflows", []),
            },
        )

        ts = TimeSpan(start=self._start, step=self._dt)
        self._inflow_ports: dict[str, BaseInput] = {}
        for slot in self._node.external_slots():
            name = slot[3:]
            port = BaseInput(
                f"{model_id}.{slot}",
                Quantities.DISCHARGE,
                ScalarElementSet(name),
                time_span=ts,
                owner=self,
                required=self._node.slot_required(slot),
            )
            self.add_input(port)
            self._inflow_ports[name] = port

        for var, qname in self._node.output_vars.items():
            quantity = (
                getattr(Quantities, qname.upper(), None) or Quantities.CANONICAL[qname]
            )
            self.create_output(
                quantity, ScalarElementSet(model_id), port_id=f"{model_id}.{var}"
            )

        self._initial_snapshot: dict = None

    # -- info ---------------------------------------

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def current_time(self) -> float:
        return self._start + self._cursor * self._dt

    @property
    def node(self) -> ReservoirNode:
        """The composed reservoir node (storage curve, release policy)."""
        return self._node

    @property
    def storage_curve(self):
        return self._node.storage_curve

    # -- lifecycle ----------------------------------

    def _do_initialize(self):
        self._initial_snapshot = self.snapshot()

    def _do_update(self, inquirers=None):
        for slot in self._node.external_slots():
            port = self._inflow_ports[slot[3:]]
            if port.is_connected:
                value = float(np.asarray(port.pull()).flat[0])
            elif port.required:
                raise ValueError(
                    f"{self._id}: required inflow port '{port.id}' is "
                    f"not connected."
                )
            else:
                value = 0.0
            self._node.set_inflow(slot, value)

        self._node.step(self._dt, self.current_time, {})

        for var in self._node.output_vars:
            self.get_output(f"{self._id}.{var}").add_values([self._node.current(var)])

        self._cursor += 1
        if 0 < self._steps <= self._cursor:
            self.mark_done()

    def _do_finish(self):
        self._cursor = 0

    # -- IStateful ----------------------------------

    def snapshot(self) -> dict:
        return {
            "cursor": self._cursor,
            "node": self._node.snapshot(),
            "ports": {p.id: p._state() for p in self._outputs},
        }

    def restore(self, snapshot: dict):
        self._cursor = int(snapshot["cursor"])
        self._node.restore(snapshot["node"])
        for p in self._outputs:
            state = snapshot["ports"].get(p.id)
            if state is not None:
                p._set_state(state)
        if self._status != ModelStatus.CREATED:
            self._status = ModelStatus.READY

    # -- IParametric --------------------------------

    def param_spec(self) -> list[ParamMeta]:
        spec = []
        for algo_name, algo in self._node.algorithms().items():
            for meta in algo.param_spec():
                if meta.name in algo.get_params():
                    spec.append(
                        ParamMeta(
                            name=f"{algo_name}.{meta.name}",
                            description=meta.description,
                            bounds=meta.bounds,
                            default=meta.default,
                        )
                    )
        return spec

    def param_names(self) -> list[str]:
        return [m.name for m in self.param_spec()]

    def _algo(self, namespaced: str):
        algo_name = namespaced.split(".")[0]
        return self._node.algorithms()[algo_name]

    def get_param_vector(self, names: list[str] = None) -> np.ndarray:
        names = names or self.param_names()
        return np.array(
            [
                self._algo(n).get_params([n.split(".")[-1]])[n.split(".")[-1]]
                for n in names
            ]
        )

    def set_param_vector(self, values: np.ndarray, names: list[str] = None):
        names = names or self.param_names()
        if len(values) != len(names):
            raise ValueError("values / names length mismatch.")
        for n, v in zip(names, values):
            self._algo(n).set_params({n.split(".")[-1]: float(v)})

    def reset_run(self):
        if self._initial_snapshot is None:
            raise RuntimeError(f"{self._id}: not initialized yet.")
        self.restore(self._initial_snapshot)
