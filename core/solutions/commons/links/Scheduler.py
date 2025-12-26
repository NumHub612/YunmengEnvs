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
from core.solutions.commons.links import Orchestrator, LoopController, AdapterFactory

from typing import Union, Tuple, Callable
from enum import Enum
import networkx as nx
import datetime as dt


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
    """The scheduler is responsible for managing the initializing、coupling
    and scheduling of the linking components."""

    def __init__(self, regietered_components: dict[str, ILinkableComponent]):
        self._registered_components = regietered_components
        self._status = SchedulerStatus.CREATED
        self._breakpoints: set[float] = set()
        self._paused = False

        self._system_config = None
        self._models: dict[str, ILinkableComponent] = {}
        self._topo: nx.DiGraph = nx.DiGraph()
        self._trigger: ILinkableComponent = None

        self._start_time = dt.datetime.now()
        self._timeout = None
        self._log_freq = 60.0
        self._log_time = 0.0
        self._verbose = True

    @property
    def components(self) -> list[ILinkableComponent]:
        """All registered components"""
        return list(self._models.values())

    @property
    def status(self) -> SchedulerStatus:
        """Scheduler status"""
        return self._status

    @property
    def elapsed_time(self) -> float:
        """Elapsed time since start"""
        time = dt.datetime.now() - self._start_time
        return time.total_seconds()

    def setup(self, system: Orchestrator):
        """Setup scheduler with system configs."""
        self._system_config = system
        # Instantiate all components
        for mid, mcfg in system.models.items():
            model_type = mcfg.pop("TYPE")
            io_items = mcfg.pop("IOS")
            model = self._registered_components[model_type](mid, mcfg, io_items)
            self._models[mid] = model
            self._topo.add_node(mid, model=model)

        # Set scheduler config
        settings = system.schedules
        if "timeout" in settings:
            self._timeout = settings["timeout"]
        if "log_freq" in settings:
            self._log_freq = settings["log_freq"]
        if "verbose" in settings:
            self._verbose = settings["verbose"]

        self._log("setup done", True)
        self._status = SchedulerStatus.LOADING

    def initialize(self):
        """Initialze all components in order or topological order."""
        # Initialize all components in topological order
        for cid in self._topo_order():
            comp = self._models[cid]
            comp.initialize()
            if comp.status == LinkableComponentStatus.FAILED:
                raise RuntimeError(f"{cid} initialize failed")

        # Establish links
        for link in self._system_config.links:
            lid = link["id"]
            is_used = link.get("is_use", True)
            mode = link.get("mode", "PULL")
            if not is_used:
                continue
            if not ({"source", "target"} <= set(link.keys())):
                continue

            provider = link["source"]
            if provider["model"] not in self._models:
                raise ValueError(f"Link {lid}: provider model not found")
            src_model = self._models[provider["model"]]
            provider_id = provider["item"]
            idx = None
            for i, o in enumerate(src_model.outputs):
                if o.id == provider_id:
                    idx = i
                    break
            if idx is None:
                raise ValueError(f"Link {lid}: provider item not found")
            output = src_model.outputs[idx]

            consumer = link["target"]
            if consumer["model"] not in self._models:
                raise ValueError(f"Link {lid}: consumer model not found")
            tar_model = self._models[consumer["model"]]
            consumer_id = consumer["item"]
            idx = None
            for i, i_ in enumerate(tar_model.inputs):
                if i_.id == consumer_id:
                    idx = i
                    break
            if idx is None:
                raise ValueError(f"Link {lid}: consumer item not found")
            input = tar_model.inputs[idx]

            if "data_operations" in link:
                adapters = AdapterFactory("")
                # TODO： configure adapters

            if mode == "PULL":
                input.provider = output
            elif mode == "LOOP":
                looper = LoopController()
                # TODO： configure looper
            else:
                raise ValueError(f"Link {lid}: invalid mode {mode}")

        # Analyze trigger
        self._analyze_trigger()

        self._log("initialize done", True)

    def _topo_order(self) -> list[str]:
        try:
            return list(nx.topological_sort(self._topo))
        except nx.NetworkXError as e:
            raise RuntimeError("cycle detected") from e

    def _analyze_trigger(self):
        """Analyze trigger from system network."""
        self._trigger = self._models[self._topo_order()[-1]]

    def validate(self) -> dict[str, list[str]]:
        """Validate all components and return errors."""
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
        """Prepare all components for running."""
        for cid in self._topo_order():
            self._models[cid].prepare()

        self._status = SchedulerStatus.READY
        self._log("prepare done", True)

    def run(self):
        """Run scheduler until all components are done or failed."""
        if self.status == SchedulerStatus.PAUSED:
            self.resume()
            return
        if self.status not in (SchedulerStatus.READY, SchedulerStatus.CREATED):
            raise RuntimeError(f"Cannot run from status {self.status}")

        self._status = SchedulerStatus.RUNNING
        self._main_loop()

        self._log("run done", True)

    def _main_loop(self):
        while True:
            # Check if trigger are done or failed
            if self._trigger.status == LinkableComponentStatus.DONE:
                self._status = SchedulerStatus.DONE
                break
            if self._trigger.status == LinkableComponentStatus.FAILED:
                self._status = SchedulerStatus.FAILED
                break

            # Stop if user paused or hit breakpoint or timeout.
            if self._hit_breakpoint():
                self._status = SchedulerStatus.PAUSED
                break
            if self._paused:
                self._status = SchedulerStatus.PAUSED
                break
            if self._timeout and self.elapsed_time >= self._timeout:
                self._status = SchedulerStatus.PAUSED
                break

            # Update to next step
            self._trigger.update([])
            self._log("updated")

    def _hit_breakpoint(self) -> bool:
        """Check if scheduler hits any breakpoint."""
        t = self.elapsed_time
        for bp in list(self._breakpoints):
            is_hint = False
            # timestamp breakpoint
            if isinstance(bp, float) and t >= bp:
                is_hint = True
            # componenet breakpoint
            if isinstance(bp, tuple):
                mid, tgt = bp
                if t >= tgt and self._models[mid].status in (
                    LinkableComponentStatus.DONE,
                    LinkableComponentStatus.FAILED,
                ):
                    is_hint = True
            # conditional breakpoint
            if callable(bp) and bp(self):
                is_hint = True
            # remove bp if hit
            if is_hint:
                self._breakpoints.remove(bp)
                return True
        return False

    def resume(self):
        if self.status != SchedulerStatus.PAUSED:
            raise RuntimeError("Not paused, cannot resume")

        self._status = SchedulerStatus.RUNNING
        self._main_loop()
        self._log("resume done", True)

    def finish(self):
        """Finish all components in order or reverse order."""
        for cid in reversed(self._topo_order()):
            self._models[cid].finish()
        self._status = SchedulerStatus.DONE
        self._log("finish done", True)

    def _log(self, message: str = "", force: bool = False):
        """Log current status."""
        if not self._verbose or not self._log_freq:
            return

        elapsed_time = self.elapsed_time
        date_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        if elapsed_time - self._log_time >= 1e-6 or force:
            time_str = self._format_time(elapsed_time)
            print(f"{date_str}, {time_str}, {self.status.name}: {message}")
            while elapsed_time >= self._log_time:
                self._log_time += self._log_freq

    def _format_time(self, elapsed_time: float) -> str:
        """Format time in readable string."""
        if elapsed_time < 60.0:
            time_str = f"{elapsed_time:06.3f}s"
        elif elapsed_time < 3600.0:
            mins = int(elapsed_time // 60.0)
            secs = elapsed_time - 60.0 * mins
            time_str = f"{mins:02d}m{secs:06.3f}s"
        elif elapsed_time < 86400.0:
            hours = int(elapsed_time // 3600.0)
            mins = int((elapsed_time % 3600.0) // 60.0)
            secs = elapsed_time - 3600.0 * hours - 60.0 * mins
            time_str = f"{hours:02d}h{mins:02d}m{secs:06.3f}s"
        else:
            days = int(elapsed_time // 86400.0)
            hours = int((elapsed_time % 86400.0) // 3600.0)
            mins = int((elapsed_time % 3600.0) // 60.0)
            secs = elapsed_time - 86400.0 * days - 3600.0 * hours - 60.0 * mins
            time_str = f"{days}d{hours:02d}h{mins:02d}m{secs:06.3f}s"

        return time_str

    def pause(self):
        """Pause scheduler."""
        self._paused = True

    def breakpoint(self, bp: Breakpoint):
        """Debug: add breakpoint."""
        self._breakpoints.add(bp)

    def snapshot(self, tag: str) -> dict:
        """Generate current global state snapshot."""
        snapshot = {}
        for mid, model in self._models.items():
            if hasattr(model, "keep_current_state"):
                snapshot[mid] = model.keep_current_state()
        return {tag: snapshot}
