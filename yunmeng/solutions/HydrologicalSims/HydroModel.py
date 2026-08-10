# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

HydroModel — a semi-distributed watershed hydrological linkable model.

Inside the model is a *tree* (node = sub-basin/station/reservoir,
edge = river connection), validated by HydroTopology.
"""

from __future__ import annotations
from typing import Any

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
from yunmeng.solutions.HydrologicalSims.HydroNodes import NODE_TYPES, HydrologyNode
from yunmeng.solutions.HydrologicalSims.Topology import HydroTopology
from yunmeng.solutions.HydrologicalSims.Gauges import GaugeSet

_SLOT_QUANTITIES = {
    "P": Quantities.PRECIPITATION,
    "E": Quantities.EVAPORATION,
}


def _slot_quantity(slot: str):
    return _SLOT_QUANTITIES.get(slot, Quantities.DISCHARGE)


class HydrologyModel(BaseModel, IStateful, IParametric):
    """Semi-distributed watershed hydrological model (tree topology)."""

    def __init__(self, config: dict):
        model_id = config["id"]
        super().__init__(model_id, ModelMeta(name=model_id, category="hydrology"))
        self._cfg = config
        self._dt = float(config.get("dt", 86400.0))
        self._steps = int(config.get("steps", 0))
        self._start = float(config.get("start", 0.0))
        self._cursor = 0
        self._nodes: dict[str, HydrologyNode] = {}
        self._topo: list[HydrologyNode] = []
        self._root: HydrologyNode = None
        self._gauges = GaugeSet(config.get("gauges", {}))
        self._slot_ports: dict[tuple[str, str], Any] = {}
        self._var_ports: dict[tuple[str, str], Any] = {}
        self._initial_snapshot: dict = None
        self._build()

    # -- info ---------------------------------------

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def current_time(self) -> float:
        return self._start + self._cursor * self._dt

    @property
    def nodes(self) -> dict[str, HydrologyNode]:
        return dict(self._nodes)

    @property
    def root(self) -> HydrologyNode:
        return self._root

    @property
    def topology(self) -> HydroTopology:
        return self._topology

    @property
    def gauges(self) -> GaugeSet:
        return self._gauges

    def node(self, name: str) -> HydrologyNode:
        return self._nodes[name]

    def _do_initialize(self):
        # structure/ports built eagerly in __init__
        pass

    # -- build --------------------------------------

    def _build(self):
        cfg = self._cfg
        for ncfg in cfg.get("nodes", []):
            name = ncfg["name"]
            ntype = ncfg["type"]
            if name in self._nodes:
                raise ValueError(f"{self._id}: duplicate node name '{name}'.")
            try:
                cls = NODE_TYPES[ntype]
            except KeyError:
                raise ValueError(
                    f"{self._id}: unknown node type '{ntype}' "
                    f"(known: {sorted(NODE_TYPES)})."
                ) from None
            self._nodes[name] = cls(name, ncfg)

        if not self._nodes:
            raise ValueError(f"{self._id}: no nodes configured.")

        edges = [(u, v) for u, v in cfg.get("edges", [])]
        try:
            self._topology = HydroTopology(list(self._nodes), edges)
        except ValueError as e:
            raise ValueError(f"{self._id}: {e}") from None
        self._root = self._nodes[self._topology.root]
        self._topo = [self._nodes[n] for n in self._topology.topo_order]

        for u, v in edges:
            self._nodes[v].add_internal_inflow(u)

        self._bind_forcing()
        self._create_ports(cfg)
        self._initial_snapshot = self.snapshot()

    def _bind_forcing(self):
        """Register per-node gauge bindings (P/E) with the GaugeSet."""
        for node in self._topo:
            forcing = node.cfg.get("forcing") or {}
            for slot, binding in forcing.items():
                key = f"{node.name}.{slot}"
                self._gauges.bind(key, binding)

    def _forcing_of(self, node: HydrologyNode, slot: str) -> float:
        key = f"{node.name}.{slot}"
        if self._gauges.has_binding(key):
            return self._gauges.value(key, self._cursor)
        return None

    def _create_ports(self, cfg: dict):
        ts = TimeSpan(start=self._start, step=self._dt)

        for node in self._topo:
            for slot in node.external_slots():
                if self._gauges.has_binding(f"{node.name}.{slot}"):
                    continue  # fed internally by gauges -> no port
                port_id = f"{self._id}.{node.name}.{slot}"
                port = BaseInput(
                    port_id,
                    _slot_quantity(slot),
                    ScalarElementSet(node.name),
                    time_span=ts,
                    owner=self,
                    required=self._slot_required(node, slot),
                )
                self.add_input(port)
                self._slot_ports[(node.name, slot)] = port

        exposes = list(cfg.get("expose", []))
        default_ref = f"{self._root.name}.Q"
        if default_ref not in exposes:
            exposes.insert(0, default_ref)

        for ref in exposes:
            node_name, _, var = ref.partition(".")
            if not node_name or not var:
                raise ValueError(
                    f"{self._id}: expose entries must look like 'node.var', "
                    f"got '{ref}'."
                )
            if node_name not in self._nodes:
                raise ValueError(f"{self._id}: expose unknown node '{node_name}'.")
            node = self._nodes[node_name]
            if var not in node.output_vars:
                raise ValueError(
                    f"{self._id}: node '{node_name}' has no output variable "
                    f"'{var}' (has: {sorted(node.output_vars)})."
                )
            qname = node.output_vars[var]
            quantity = (
                getattr(Quantities, qname.upper(), None) or Quantities.CANONICAL[qname]
            )
            port = self.create_output(
                quantity,
                ScalarElementSet(node_name),
                port_id=f"{self._id}.{node_name}.{var}",
            )
            self._var_ports[(node_name, var)] = port

    def _slot_required(self, node: HydrologyNode, slot: str) -> bool:
        if slot in ("P", "E"):
            # required only when no internal forcing feeds it
            return not self._gauges.has_binding(f"{node.name}.{slot}")
        return node.slot_required(slot)

    # -- stepping ---------------------------------------------

    def _do_update(self, inquirers=None):
        t = self.current_time
        dt = self._dt

        for node in self._topo:
            inputs: dict[str, float] = {}
            for slot in node.external_slots():
                # 1) internal gauge forcing (P/E)
                forced = self._forcing_of(node, slot)
                if forced is not None:
                    inputs[slot] = forced
                    continue
                # 2) coupled input port
                port = self._slot_ports.get((node.name, slot))
                if port is not None and port.is_connected:
                    value = float(np.asarray(port.pull()).flat[0])
                elif port is not None and port.required:
                    raise ValueError(
                        f"{self._id}: required input port '{port.id}' is "
                        f"not connected."
                    )
                else:
                    value = 0.0
                if slot.startswith("in."):
                    # FIX: external inflows are actually injected now
                    node.set_inflow(slot, value)
                else:
                    inputs[slot] = value
            node.step(dt, t, inputs)

            # feed downstream via topology (O(1) lookup, no private scan)
            down = self._topology.downstream(node.name)
            if down is not None:
                target = self._nodes[down]
                target.set_inflow(
                    target.internal_inflow_slot(node.name), node.current("Q")
                )

        for (node_name, var), port in self._var_ports.items():
            port.add_values([self._nodes[node_name].current(var)])

        self._cursor += 1
        if 0 < self._steps <= self._cursor:
            self.mark_done()

    def _do_finish(self):
        self._cursor = 0

    # -- IStateful -----------------------------------------------

    def snapshot(self) -> dict:
        return {
            "cursor": self._cursor,
            "nodes": {n.name: n.snapshot() for n in self._topo},
            "ports": {p.id: p._state() for p in self._outputs},
        }

    def restore(self, snapshot: dict):
        self._cursor = int(snapshot["cursor"])
        for name, snap in snapshot["nodes"].items():
            self._nodes[name].restore(snap)
        for p in self._outputs:
            state = snapshot["ports"].get(p.id)
            if state is not None:
                p._set_state(state)
        if self._status != ModelStatus.CREATED:
            self._status = ModelStatus.READY

    # -- IParametric --------------------------------

    def _algo_params(self) -> list[tuple[str, Any, ParamMeta]]:
        out = []
        for node in self._topo:
            for algo_name, algo in node.algorithms().items():
                for meta in algo.param_spec():
                    if meta.name in algo.get_params():
                        out.append((f"{node.name}.{algo_name}.{meta.name}", algo, meta))
        return out

    def param_spec(self) -> list[ParamMeta]:
        return [
            ParamMeta(
                name=full,
                description=meta.description,
                bounds=meta.bounds,
                default=meta.default,
            )
            for full, _, meta in self._algo_params()
        ]

    def param_names(self) -> list[str]:
        return [full for full, _, _ in self._algo_params()]

    def get_param_vector(self, names: list[str] = None) -> np.ndarray:
        index = {full: algo for full, algo, _ in self._algo_params()}
        names = names or self.param_names()
        return np.array(
            [
                algo.get_params([n.split(".")[-1]])[n.split(".")[-1]]
                for n in names
                for algo in [index[n]]
            ]
        )

    def set_param_vector(self, values: np.ndarray, names: list[str] = None):
        names = names or self.param_names()
        if len(values) != len(names):
            raise ValueError("values / names length mismatch.")
        grouped: dict[int, dict[str, float]] = {}
        index = {
            full: (i, algo) for i, (full, algo, _) in enumerate(self._algo_params())
        }
        for n, v in zip(names, values):
            i, algo = index[n]
            grouped.setdefault(i, {})[n.split(".")[-1]] = float(v)
        algos = {i: algo for i, (_, algo, _) in enumerate(self._algo_params())}
        for i, params in grouped.items():
            algos[i].set_params(params)

    def reset_run(self):
        if self._initial_snapshot is None:
            raise RuntimeError(f"{self._id}: not initialized yet.")
        self.restore(self._initial_snapshot)

    # -- diagnostics --------------------------------

    def describe(self) -> str:
        lines = [f"HydroModel '{self._id}' (dt={self._dt}s, steps={self._steps})"]
        for node in self._topo:
            lines.append(f"  [{node.node_type}] {node.name}")
        lines.append(f"  root: {self._root.name if self._root else '?'}")
        lines.append(f"  gauges: {self._gauges.gauge_ids}")
        lines.append(f"  inputs:  {[p.id for p in self._inputs]}")
        lines.append(f"  outputs: {[p.id for p in self._outputs]}")
        return "\n".join(lines)
