# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Scheduler: drives a coupling graph of linkable models step by step.

Usage:

"""

import time
from dataclasses import dataclass, field

from yunmeng.interfaces.capabilities import IScheduler, ISnapshottable
from yunmeng.interfaces.solution import (
    CouplingConfig,
    CouplingKinds,
    ILinkableModel,
    IModelCallback,
    ModelEvent,
    ModelStatus,
)
from yunmeng.numerics.algos import ym_register
from yunmeng.setting import logger
from yunmeng.solutions.commons.additionals import (
    ElementMapAdapter,
    FixedPointCoupler,
    PullCoupler,
)
from yunmeng.solutions.commons.models import BaseAdapter, BaseInput, BaseOutput


@dataclass
class Link:
    """One coupling link between an output port and an input port."""

    source_port: object
    target_port: object
    config: CouplingConfig = field(default_factory=CouplingConfig)

    @property
    def source_model(self) -> ILinkableModel | None:
        return self.source_port.owner

    @property
    def target_model(self) -> ILinkableModel | None:
        return self.target_port.owner


class ExecutionPlan:
    """Topological order over PULL links + LOOP component clusters."""

    def __init__(self) -> None:
        self.order: list = []
        self.clusters: dict = (
            {}
        )  # model identity -> list of (cluster_key, members, config)

    @classmethod
    def build(cls, models: list, links: list) -> "ExecutionPlan":
        plan = cls()
        pull_links = [l for l in links if l.config.mode != CouplingKinds.LOOP]
        loop_links = [l for l in links if l.config.mode == CouplingKinds.LOOP]

        model_by_id = {id(m): m for m in models}
        indeg = {mid: 0 for mid in model_by_id}
        downstream = {}
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
        if len(order) != len(models):
            cyclic = [m.id for mid, m in model_by_id.items() if indeg[mid] > 0]
            raise ValueError(
                "PULL dependency graph contains a cycle involving "
                f"{cyclic}; feedback links must be declared with mode LOOP."
            )
        plan.order = order

        # LOOP clusters: connected components over loop links (union-find)
        parent = {id(m): id(m) for m in models}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for l in loop_links:
            s, t = l.source_model, l.target_model
            if s is None or t is None:
                continue
            parent[find(id(s))] = find(id(t))

        cluster_cfg = {}
        for l in loop_links:
            s = l.source_model
            if s is None:
                continue
            cluster_cfg.setdefault(find(id(s)), l.config)

        members = {}
        for m in models:
            members.setdefault(find(id(m)), []).append(m)
        for root, ms in members.items():
            if root not in cluster_cfg or len(ms) < 2:
                continue
            for m in ms:
                plan.clusters.setdefault(id(m), []).append(
                    (root, tuple(ms), cluster_cfg[root])
                )
        return plan


@ym_register("scheduler")
class Scheduler(IScheduler):
    """Step-wise driver for a mixed PULL / LOOP coupling graph."""

    @classmethod
    def get_name(cls) -> str:
        return "Scheduler"

    def __init__(self, loop_coupler: FixedPointCoupler | None = None) -> None:
        self._models: list = []
        self._links: list = []
        self._loop_coupler = loop_coupler or FixedPointCoupler()
        self._pull_strategy = PullCoupler()
        self._plan = ExecutionPlan()
        self.results: list = []
        self.settings: dict = {}
        self._callbacks: list = []
        self._step_count = 0
        self._started_at: float | None = None

    # -- graph construction -------------------------

    def add(self, model: ILinkableModel) -> None:
        if model not in self._models:
            self._models.append(model)

    def link(
        self,
        source_port: BaseOutput,
        target_port: BaseInput,
        config: CouplingConfig | None = None,
        source_elements: list | None = None,
        target_elements: list | None = None,
        adapter_chain: list | None = None,
    ) -> None:
        """Connect two ports (PULL by default).

        ``adapter_chain``: ordered list of adapters applied between source
        and target (replaces the legacy data_operations). ``source_elements``
        without an explicit chain inserts an ElementMapAdapter.
        """
        config = config or CouplingConfig()
        adapted = adapter_chain is not None or source_elements is not None
        self._check_compatible(source_port, target_port, adapted=adapted)

        provider = source_port
        if adapter_chain:
            for adapter in adapter_chain:
                provider.add_adapter(adapter)
                provider = adapter
        if source_elements is not None:
            adapter = ElementMapAdapter(
                f"element_map.{source_port.id}.{target_port.id}"
            )
            adapter.set_mapping(target_port, provider, source_elements, target_elements)
            provider = adapter
        elif target_port.is_connected:
            raise ValueError(
                f"Input port '{target_port.id}' already has a provider; "
                f"an input accepts exactly one provider "
                f"(use an adapter chain for fan-in reduction)."
            )
        if not adapter_chain and source_elements is None:
            source_port.add_consumer(target_port)
        else:
            provider.add_consumer(target_port)

        for model in (source_port.owner, target_port.owner):
            if model is not None:
                self.add(model)
        self._links.append(Link(source_port, target_port, config))

    @staticmethod
    def _check_compatible(
        source_port: BaseOutput, target_port: BaseInput, adapted: bool
    ) -> None:
        sq, tq = source_port.quantity, target_port.quantity
        if not adapted and sq.name != tq.name:
            raise ValueError(
                f"Quantity mismatch: '{source_port.id}' provides "
                f"'{sq.name}' but '{target_port.id}' expects '{tq.name}'. "
                f"Insert an adapter (e.g. ScaleOutput) to convert."
            )
        if sq.unit and tq.unit and sq.unit != tq.unit:
            logger.warning(
                f"Unit mismatch on link {source_port.id} -> {target_port.id}: "
                f"'{sq.unit}' vs '{tq.unit}'; make sure an adapter converts."
            )
        s_step = getattr(source_port.time_span, "step", None)
        t_step = getattr(target_port.time_span, "step", None)
        if s_step and t_step and s_step != t_step:
            logger.warning(
                f"Time-step mismatch on link {source_port.id} ({s_step}s) -> "
                f"{target_port.id} ({t_step}s); values are pulled as-is."
            )

    def unlink(self, source_port: BaseOutput, target_port: BaseInput) -> bool:
        for i, l in enumerate(self._links):
            if l.source_port is source_port and l.target_port is target_port:
                del self._links[i]
                if target_port.provider is source_port:
                    source_port.remove_consumer(target_port)
                return True
        return False

    def rebuild(self) -> None:
        self._plan = ExecutionPlan.build(self._models, self._links)

    @property
    def models(self) -> list:
        return self._models

    @property
    def links(self) -> list:
        return list(self._links)

    @property
    def execution_order(self) -> list:
        return [m.id for m in self._plan.order]

    # -- lifecycle ----------------------------------

    def initialize(self) -> None:
        errors = []
        for m in self._models:
            errors += m.validate()
        if errors:
            raise ValueError("Validation failed:\n  " + "\n  ".join(errors))
        for m in self._models:
            m.initialize()
        self.rebuild()
        self._step_count = 0

    def finish(self) -> None:
        for m in self._models:
            if m.status not in (ModelStatus.CREATED,):
                try:
                    m.finish()
                except Exception as e:
                    logger.warning(f"finish() of '{m.id}' failed: {e}")

    # -- stepping -----------------------------------

    def step(self) -> list:
        results = []
        done_clusters = set()
        self._fire(ModelEvent.STEP_BEGIN, step=self._step_count)
        for m in self._plan.order:
            if m.status in (ModelStatus.DONE, ModelStatus.FAILED):
                continue
            entries = self._plan.clusters.get(id(m))
            if not entries:
                m.update()
                continue
            for key, members, cfg in entries:
                if key in done_clusters:
                    continue
                done_clusters.add(key)
                if len(members) == 2:
                    results.append(
                        self._loop_coupler.iterate(members[0], members[1], cfg)
                    )
                else:
                    results.append(self._iterate_cluster(members, cfg))
        self.results.extend(results)
        self._step_count += 1
        self._fire(ModelEvent.STEP_END, step=self._step_count, results=results)
        return results

    def _iterate_cluster(self, members: tuple, cfg: CouplingConfig):
        """Pairwise fixed-point sweep for >2-member LOOP clusters."""
        last = None
        for k in range(cfg.max_iterations):
            residuals = []
            for a, b in zip(members, members[1:] + members[:1]):
                last = self._loop_coupler.iterate(a, b, cfg)
                residuals.append(last.residual)
            if last is not None and last.converged:
                return last
        return last

    # -- graph-level state --------------------------

    def snapshot_graph(self) -> dict:
        return {
            m.id: m.snapshot() for m in self._models if isinstance(m, ISnapshottable)
        }

    def restore_graph(self, snapshot: dict) -> None:
        for m in self._models:
            if isinstance(m, ISnapshottable) and m.id in snapshot:
                m.restore(snapshot[m.id])

    # -- callbacks ----------------------------------

    def add_callback(self, callback: IModelCallback) -> None:
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def remove_callback(self, callback: IModelCallback) -> None:
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _fire(self, event: str, **context) -> None:
        for cb in list(self._callbacks):
            cb.on_event(event, self, context)

    # -- run ----------------------------------------

    def run(self, max_steps: int | None = None) -> list:
        timeout = self.settings.get("timeout")
        log_freq = int(self.settings.get("log_freq", 0) or 0)
        all_results = []
        n = 0
        self._started_at = time.monotonic()
        try:
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
                if timeout and time.monotonic() - self._started_at > float(timeout):
                    raise TimeoutError(f"Scheduler run exceeded timeout={timeout}s.")
                all_results.extend(self.step())
                n += 1
                if log_freq and n % log_freq == 0:
                    logger.info(
                        f"step {n}: "
                        + ", ".join(f"{m.id}={m.status.value}" for m in self._models)
                    )
            failed = [m for m in self._models if m.status == ModelStatus.FAILED]
            for m in failed:
                logger.error(f"model '{m.id}' FAILED: {m.get_last_error()}")
            return all_results
        finally:
            self.finish()
