# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Lightweight scheduler for running linkable models.
"""

from __future__ import annotations
from enum import Enum
from typing import Union, Tuple, Callable
import datetime as dt
import networkx as nx

from yunmeng.solutions.standards import (
    ILinkableModel,
    IOutput,
    IInput,
    ModelStatus,
    CouplingMode,
    CouplingConfig,
    ICouplingStrategy,
    IterationResult,
)
from yunmeng.solutions.commons.models import LoopController, AdapterFactory


class SchedulerStatus(Enum):
    """Scheduler status."""

    CREATED = 1
    LOADING = 2
    READY = 3
    RUNNING = 4
    PAUSED = 5
    DONE = 6
    FAILED = 7


Breakpoint = Union[float, Tuple[str, float], Callable[["Scheduler"], bool]]


class Scheduler:
    """Minimal scheduler that initializes, links and runs models."""

    def __init__(self, registered_components: dict[str, type[ILinkableModel]] = None):
        self._registered_components = registered_components or {}
        self._status = SchedulerStatus.CREATED
        self._breakpoints: set[float] = set()
        self._paused = False

        self._models: dict[str, ILinkableModel] = {}
        self._topo: nx.DiGraph = nx.DiGraph()
        self._trigger: ILinkableModel = None
        self._links: list[dict] = []

        self._start_time = dt.datetime.now()
        self._timeout: float = None
        self._log_freq = 60.0
        self._log_time = 0.0
        self._verbose = True

    @property
    def components(self) -> list[ILinkableModel]:
        return list(self._models.values())

    @property
    def status(self) -> SchedulerStatus:
        return self._status

    @property
    def elapsed_time(self) -> float:
        return (dt.datetime.now() - self._start_time).total_seconds()

    def setup(
        self,
        model_configs: dict[str, dict],
        link_configs: list[dict] = None,
        schedule_settings: dict = None,
    ):
        """Instantiate models and build the topology graph."""
        link_configs = link_configs or []
        schedule_settings = schedule_settings or {}

        for mid, cfg in model_configs.items():
            model_type = cfg.pop("TYPE", mid)
            factory = self._registered_components.get(model_type)
            if factory is None:
                raise ValueError(f"Unknown model type: {model_type}")
            model = factory(mid, cfg)
            self._models[mid] = model
            self._topo.add_node(mid, model=model)

        for link in link_configs:
            src = link.get("source", {}).get("model")
            tgt = link.get("target", {}).get("model")
            if src and tgt:
                self._topo.add_edge(src, tgt, link=link)
        self._links = link_configs

        self._timeout = schedule_settings.get("timeout")
        self._log_freq = schedule_settings.get("log_freq", 60.0)
        self._verbose = schedule_settings.get("verbose", True)

        self._log("setup done", True)
        self._status = SchedulerStatus.LOADING

    def initialize(self):
        """Initialize all models and establish links."""
        for cid in self._topo_order():
            comp = self._models[cid]
            comp.initialize()
            if comp.status == ModelStatus.FAILED:
                raise RuntimeError(f"{cid} initialize failed")

        self._establish_links()
        self._trigger = self._models.get(self._topo_order()[-1])
        self._log("initialize done", True)

    def _establish_links(self):
        adapter_factory = AdapterFactory()
        for link in self._links:
            if not link.get("is_use", True):
                continue
            src_id = link.get("source", {}).get("model")
            tgt_id = link.get("target", {}).get("model")
            src_item = link.get("source", {}).get("item")
            tgt_item = link.get("target", {}).get("item")
            if not (src_id and tgt_id and src_item and tgt_item):
                continue

            src_model = self._models[src_id]
            tgt_model = self._models[tgt_id]
            output = self._find_port(src_model.outputs, src_item)
            input_ = self._find_port(tgt_model.inputs, tgt_item)
            if output is None or input_ is None:
                raise ValueError(f"Link item not found: {link}")

            input_.provider = output
            # TODO: apply adapter_factory chain if data_operations specified

    @staticmethod
    def _find_port(ports: list, port_id: str):
        for p in ports:
            if p.id == port_id:
                return p
        return None

    def _topo_order(self) -> list[str]:
        try:
            return list(nx.topological_sort(self._topo))
        except nx.NetworkXError as e:
            raise RuntimeError("cycle detected") from e

    def validate(self) -> dict[str, list[str]]:
        errors = {}
        for cid, comp in self._models.items():
            res = comp.validate()
            if res:
                errors[cid] = res
        if errors:
            self._status = SchedulerStatus.FAILED
        self._log(f"validate done with {len(errors)} errors", True)
        return errors

    def prepare(self):
        for cid in self._topo_order():
            self._models[cid].prepare()
        self._status = SchedulerStatus.READY
        self._log("prepare done", True)

    def run(self):
        if self.status == SchedulerStatus.PAUSED:
            self.resume()
            return
        if self.status not in (SchedulerStatus.READY, SchedulerStatus.CREATED):
            raise RuntimeError(f"Cannot run from status {self.status}")
        self._status = SchedulerStatus.RUNNING
        self._main_loop()
        self._log("run done", True)

    def _progress_token(self):
        """Return a hashable snapshot of current model statuses and output versions."""
        statuses = tuple(sorted((cid, m.status) for cid, m in self._models.items()))
        versions = []
        for m in self._models.values():
            for out in m.outputs:
                versions.append((m.id, out.id, getattr(out, "version", 0)))
        return (statuses, tuple(versions))

    def _propagate_done(self) -> bool:
        """Mark downstream models as done once all their input providers are done."""
        changed = False
        for cid in self._topo_order():
            m = self._models[cid]
            if m.status in (ModelStatus.DONE, ModelStatus.FAILED):
                continue
            if not m.inputs:
                continue
            if all(
                inp.is_connected
                and inp.provider is not None
                and inp.provider.component is not None
                and inp.provider.component.status == ModelStatus.DONE
                for inp in m.inputs
            ):
                m.mark_done()
                changed = True
        return changed

    def _main_loop(self):
        token = self._progress_token()
        while True:
            statuses = {cid: m.status for cid, m in self._models.items()}
            if all(s in (ModelStatus.DONE, ModelStatus.FAILED) for s in statuses.values()):
                self._status = SchedulerStatus.DONE
                break
            if any(s == ModelStatus.FAILED for s in statuses.values()):
                self._status = SchedulerStatus.FAILED
                break

            if self._hit_breakpoint() or self._paused or (
                self._timeout and self.elapsed_time >= self._timeout
            ):
                self._status = SchedulerStatus.PAUSED
                break

            for cid in self._topo_order():
                self._models[cid].update([])

            propagated = self._propagate_done()
            if all(
                m.status in (ModelStatus.DONE, ModelStatus.FAILED)
                for m in self._models.values()
            ):
                self._status = SchedulerStatus.DONE
                self._log("all models done after propagation")
                break

            new_token = self._progress_token()
            if not propagated and new_token == token:
                # No observable progress in a full pass: the remaining models are
                # effectively finished (e.g. downstream consumers of DONE sources).
                for m in self._models.values():
                    if m.status not in (ModelStatus.DONE, ModelStatus.FAILED):
                        m.mark_done()
                self._status = SchedulerStatus.DONE
                self._log("no progress: marking remaining models as done")
                break
            token = new_token
            self._log("updated")

    def _hit_breakpoint(self) -> bool:
        t = self.elapsed_time
        for bp in list(self._breakpoints):
            hit = False
            if isinstance(bp, float) and t >= bp:
                hit = True
            elif isinstance(bp, tuple):
                mid, tgt = bp
                if t >= tgt and self._models.get(mid, self._trigger).status in (
                    ModelStatus.DONE,
                    ModelStatus.FAILED,
                ):
                    hit = True
            elif callable(bp) and bp(self):
                hit = True
            if hit:
                self._breakpoints.remove(bp)
                return True
        return False

    def resume(self):
        if self.status != SchedulerStatus.PAUSED:
            raise RuntimeError("Not paused, cannot resume")
        self._status = SchedulerStatus.RUNNING
        self._paused = False
        self._main_loop()
        self._log("resume done", True)

    def finish(self):
        for cid in reversed(self._topo_order()):
            self._models[cid].finish()
        self._status = SchedulerStatus.DONE
        self._log("finish done", True)

    def pause(self):
        self._paused = True

    def breakpoint(self, bp: Breakpoint):
        self._breakpoints.add(bp)

    def _log(self, message: str = "", force: bool = False):
        if not self._verbose or not self._log_freq:
            return
        elapsed = self.elapsed_time
        if force or elapsed - self._log_time >= self._log_freq:
            date_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            print(f"{date_str}, {self.status.name}: {message}")
            self._log_time = elapsed

    def execute_link(
        self,
        source: ILinkableModel,
        target: ILinkableModel,
        strategy: ICouplingStrategy,
        config: CouplingConfig = None,
    ) -> IterationResult:
        """Execute a coupling strategy between two models."""
        controller = LoopController(strategy)
        return controller.execute(source, target, config)
