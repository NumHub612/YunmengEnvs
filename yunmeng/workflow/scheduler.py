# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Scheduler: drives a coupling graph of linkable models step by step.

  - PULL links define a dependency DAG; models update in topological
    order (data sources first) and pull through their input ports.
  - LOOP links define component pairs executed by an IIterativeCoupler
    (FixedPointCoupler by default).  LOOP back-edges are excluded from
    the topological ordering — that is what makes cyclic feedback
    graphs schedulable at all.

Usage:
    sched = Scheduler()
    sched.add(rain); sched.add(sub); sched.add(reach)
    sched.link(rain.output_port, sub.inputs[0])                    # PULL
    sched.link(sub.discharge_port, reach.inputs[0])                # PULL
    sched.link(reach.water_level_port, sub.tailwater_port,
               CouplingConfig(mode=CouplingMode.LOOP, ...))        # LOOP
    sched.initialize()
    sched.run(n_steps)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from yunmeng.solutions.standards import (
    ILinkableModel,
    IInput,
    IOutput,
    IIterativeCoupler,
    CouplingMode,
    CouplingConfig,
    IterationResult,
    ModelStatus,
    CallbackEvent,
)
from yunmeng.solutions.commons.additionals import (
    PullCoupler,
    FixedPointCoupler,
    ElementMapAdapter,
)


@dataclass
class Link:
    """One coupling link between an output port and an input port."""

    source_port: IOutput
    target_port: IInput
    config: CouplingConfig
    target_model: Optional[ILinkableModel] = None

    @property
    def source_model(self) -> Optional[ILinkableModel]:
        return getattr(self.source_port, "component", None)


class Scheduler:
    """Step-wise driver for a mixed PULL / LOOP coupling graph."""

    def __init__(self, loop_coupler: IIterativeCoupler = None):
        self._models: list[ILinkableModel] = []
        self._links: list[Link] = []
        self._loop_coupler = loop_coupler or FixedPointCoupler()
        self._pull_strategy = PullCoupler()
        self._order: list[ILinkableModel] = []
        self._loop_of: dict[int, tuple] = {}
        self.results: list[IterationResult] = []

    # -- graph construction --------------------------------------------

    def add(self, model: ILinkableModel):
        if model not in self._models:
            self._models.append(model)

    def link(
        self,
        source_port: IOutput,
        target_port: IInput,
        config: CouplingConfig = None,
        source_elements: list = None,
        target_elements: list = None,
    ):
        """Connect two ports with the given coupling config (PULL by
        default).  The target model is resolved from the port's owner.

        When *source_elements* (and optionally *target_elements*) are
        given, an :class:`ElementMapAdapter` is attached to the source
        output so that only the addressed elements reach this consumer
        — this is how arbitrary element-to-element coupling is wired
        (e.g. basin outlet "sub2" -> reach section "sec_05")."""
        config = config or CouplingConfig()
        target_port.provider = source_port
        if source_elements is not None:
            adapter = self._element_adapter_for(source_port)
            adapter.set_mapping(
                target_port, source_port, source_elements, target_elements
            )
        target_model = self._resolve_owner(target_port)
        if source_port.component is not None:
            self.add(source_port.component)
        if target_model is not None:
            self.add(target_model)
        self._links.append(Link(source_port, target_port, config, target_model))

    @staticmethod
    def _element_adapter_for(source_port: IOutput) -> ElementMapAdapter:
        """Reuse an existing ElementMapAdapter on the port or attach one."""
        for a in getattr(source_port, "_adapters", []):
            if isinstance(a, ElementMapAdapter):
                return a
        adapter = ElementMapAdapter(f"element_map.{source_port.id}")
        source_port.add_adapter(adapter)
        return adapter

    def unlink(self, source_port: IOutput, target_port: IInput) -> bool:
        """Remove a link: detach the provider and drop the link record.

        Call :meth:`rebuild` afterwards to refresh the execution plan."""
        for i, l in enumerate(self._links):
            if l.source_port is source_port and l.target_port is target_port:
                del self._links[i]
                if target_port.provider is source_port:
                    target_port.provider = None
                return True
        return False

    def rebuild(self):
        """Rebuild the execution plan after dynamic graph changes."""
        self._build_execution_plan()

    def _resolve_owner(self, port: IInput) -> Optional[ILinkableModel]:
        for m in self._models:
            if port in m.inputs:
                return m
        return None

    @property
    def models(self) -> list[ILinkableModel]:
        return self._models

    # -- lifecycle -------------------------------------------------------

    def initialize(self):
        errors = []
        for m in self._models:
            errors += m.validate()
        if errors:
            raise ValueError("Validation failed:\n  " + "\n  ".join(errors))
        for m in self._models:
            m.initialize()
        self._build_execution_plan()

    def _build_execution_plan(self):
        """Topological order over PULL links; group LOOP pairs."""
        pull_links = [l for l in self._links if l.config.mode != CouplingMode.LOOP]
        loop_links = [l for l in self._links if l.config.mode == CouplingMode.LOOP]

        model_by_id = {id(m): m for m in self._models}
        indeg = {mid: 0 for mid in model_by_id}
        downstream: dict[int, list[int]] = {}
        for l in pull_links:
            s, t = l.source_model, l.target_model
            if s is None or t is None or s is t:
                continue
            downstream.setdefault(id(s), []).append(id(t))
            indeg[id(t)] += 1

        ready = [mid for mid, d in indeg.items() if d == 0]
        order = []
        while ready:
            mid = ready.pop()
            order.append(model_by_id[mid])
            for nid in downstream.get(mid, []):
                indeg[nid] -= 1
                if indeg[nid] == 0:
                    ready.append(nid)
        if len(order) != len(self._models):
            raise ValueError(
                "PULL dependency graph contains a cycle; feedback links "
                "must be declared with CouplingMode.LOOP."
            )
        self._order = order

        self._loop_of = {}
        pairs: dict[frozenset, tuple] = {}
        for l in loop_links:
            s, t = l.source_model, l.target_model
            key = frozenset((id(s), id(t)))
            pairs[key] = (s, t, l.config)
        for key, (s, t, cfg) in pairs.items():
            entry = (s, t, cfg, key)
            self._loop_of[id(s)] = entry
            self._loop_of[id(t)] = entry

    # -- stepping ----------------------------------------------------------

    def step(self) -> list[IterationResult]:
        """Advance every model by one time step."""
        results = []
        done_pairs = set()
        self._fire(CallbackEvent.STEP_BEGIN)
        for m in self._order:
            if m.status in (ModelStatus.DONE, ModelStatus.FAILED):
                continue
            entry = self._loop_of.get(id(m))
            if entry is None:
                m.update()
                continue
            comp_a, comp_b, cfg, key = entry
            if key in done_pairs:
                continue
            done_pairs.add(key)
            results.append(self._loop_coupler.iterate(comp_a, comp_b, cfg))
        self.results.extend(results)
        self._fire(CallbackEvent.STEP_END, results=results)
        return results

    # -- graph-level state (calibration / ensemble support) ----------------

    def snapshot_graph(self) -> dict:
        """Snapshot every model in the graph (keyed by model id)."""
        return {m.id: m.snapshot() for m in self._models if hasattr(m, "snapshot")}

    def restore_graph(self, snapshot: dict):
        """Restore every model from a graph snapshot — the reset half of
        an objective-function evaluation loop."""
        for m in self._models:
            if hasattr(m, "restore") and m.id in snapshot:
                m.restore(snapshot[m.id])

    # -- callbacks ------------------------------------------------------------

    def add_callback(self, callback):
        if not hasattr(self, "_callbacks"):
            self._callbacks = []
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def remove_callback(self, callback):
        if hasattr(self, "_callbacks") and callback in self._callbacks:
            self._callbacks.remove(callback)

    def _fire(self, event: str, **context):
        for cb in list(getattr(self, "_callbacks", [])):
            cb.on_event(event, self, context)

    def run(self, max_steps: int = None) -> list[IterationResult]:
        """Run until all models are DONE/FAILED or *max_steps* reached."""
        all_results = []
        n = 0
        while True:
            active = [
                m
                for m in self._models
                if m.status not in (ModelStatus.DONE, ModelStatus.FAILED)
            ]
            if not active:
                break
            if max_steps is not None and n >= max_steps:
                break
            all_results.extend(self.step())
            n += 1
        return all_results
