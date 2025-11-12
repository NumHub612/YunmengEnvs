# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

To provide linking network analysis and management functionalities.
"""
from core.solutions.standards import (
    ILinkableComponent,
    IOutput,
    IInput,
    LinkableComponentStatus,
)
from core.solutions.commons.links import LoopController, AdapterFactory
from configs.orchestrator import Orchestrator

import importlib
from enum import Enum
import networkx as nx
import pickle


class SchedulerStatus(Enum):
    """Scheduler status."""

    CREATED = 1
    LOADING = 2
    READY = 3
    RUNNING = 4
    PAUSED = 5
    DONE = 6
    FAILED = 7


class Scheduler:
    """The scheduler is responsible for managing the initializing、coupling
    and scheduling of the linking components."""

    def __init__(self, regietered_components: dict[str, ILinkableComponent]):
        self._registered_components = regietered_components
        self._system_config = None
        self._models: dict[str, ILinkableComponent] = {}
        self._topo: nx.DiGraph = nx.DiGraph()
        self._trigger: ILinkableComponent = None
        self._status = SchedulerStatus.CREATED
        self._breakpoints: set[float] = set()
        self._pause_req = False

    @property
    def components(self) -> list[ILinkableComponent]:
        """All registered components"""
        return list(self._models.values())

    @property
    def status(self) -> SchedulerStatus:
        """Scheduler status"""
        return self._status

    def setup(self, system: Orchestrator):
        """Setup scheduler with system configs."""
        self._status = SchedulerStatus.LOADING

        self._system_config = system
        # Instantiate all components
        for mid, mcfg in system.models.items():
            model_type = mcfg.pop("TYPE")
            io_items = mcfg.pop("IOS")
            model = self._registered_components[model_type](mcfg, io_items)
            self._models[mid] = model
            self._topo.add_node(mid, model=model)
        self._trigger = self._models[list(reversed(self._topo_order()))[0]]

        # Set scheduler config
        ...

        self._status = SchedulerStatus.READY

    def initialize(self) -> None:
        """Initialze all components in order or topological order."""
        self._status = SchedulerStatus.READY

        # Initialize all components in topological order
        for cid in self._topo_order():
            comp = self._models[cid]
            comp.initialize()
            if comp.status == LinkableComponentStatus.FAILED:
                raise RuntimeError(f"{cid} initialize failed")

        # Establish links
        for lid, lcfg in self._system_config.links.items():
            ...

    def _topo_order(self) -> list[str]:
        try:
            return list(nx.topological_sort(self._topo))
        except nx.NetworkXError as e:
            raise RuntimeError("cycle detected") from e

    def validate(self) -> dict[str, list[str]]:
        """Validate all components and return errors."""
        errors = {}
        for cid, comp in self._models.items():
            res = comp.validate()
            if res:
                errors[cid] = res
        return errors

    def prepare(self) -> None:
        """Prepare all components for running."""
        for cid in self._topo_order():
            self._models[cid].prepare()

    def run(self) -> None:
        """Run scheduler until all components are done or failed."""
        self._status = SchedulerStatus.RUNNING
        self._pause_req = False
        while not self._trigger.status == LinkableComponentStatus.DONE:
            if self._pause_requested():
                self._status = SchedulerStatus.PAUSED
                return
            self._trigger.update([])
            if self._trigger.status == LinkableComponentStatus.FAILED:
                self._status = SchedulerStatus.FAILED
                return
        self._status = SchedulerStatus.DONE

    def _pause_requested(self) -> bool:
        return False

    def finish(self) -> None:
        """Finish all components in order or reverse order."""
        for cid in reversed(self._topo_order()):
            self._models[cid].finish()
        self._status = SchedulerStatus.DONE

    def breakpoint(self, time: float, model_id: str = None) -> None:
        """Debug: set breakpoint at specified time or component."""
        self._breakpoints.add(time)

    def snapshot(self, tag: str) -> dict:
        """Generate current global state snapshot."""
        snapshot = {}
        for mid, model in self._models.items():
            if hasattr(model, "keep_current_state"):
                snapshot[mid] = model.keep_current_state()
        return snapshot
