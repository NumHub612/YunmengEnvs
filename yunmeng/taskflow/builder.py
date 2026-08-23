# -*- encoding: utf-8 -*-
"""Copyright (C) 2026, The YunMengEnvs Project Contributors. Welcome aboard YunmengEnvs!

To assemble a new Scheduler from the orchestrator config.
"""

from typing import Any, Dict
import importlib

from yunmeng.setting import logger
from yunmeng.solutions.standards import CouplingConfig, CouplingMode, IEstimable
from yunmeng.taskflow.estimator import Optimizer, Calibrator
from yunmeng.taskflow.scheduler import Scheduler

# ---------------------------------------------------
# region Builder & Assembler
# ---------------------------------------------------


class ClassBuilder:
    """Builder for dynamic class instantiation."""

    def build(self, config: Dict[str, Any]) -> Any:
        """Parse the config and instantiate the corresponding class."""
        class_path = config["class"]
        params = config.get("params", {})

        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)

        return cls(**params)


class PipelineAssembler:
    """Assembler for multiple components into an execution pipeline."""

    def __init__(self, builder: ClassBuilder):
        self.builder = builder

    def assemble(self, pipeline_config: list) -> list:
        """Assemble components in order."""
        return [self.builder.build(step) for step in pipeline_config]


# ---------------------------------------------------
# region SchedulerBuilder
# ---------------------------------------------------


class SchedulerBuilder:
    """Builder for a Scheduler from orchestrator config."""

    def __init__(self, model_registry: dict):
        self._registry = model_registry
        self._class_builder = ClassBuilder()
        self._instances: Dict[str, Any] = {}
        self._model_configs: Dict[str, dict] = {}

    # -- public API ---------------------------------

    def build(self, orchestrator) -> Scheduler:
        """Build and return a Scheduler that has been added/linked
        (but not yet initialized)."""
        sched = Scheduler()

        self._model_configs = dict(orchestrator.models or {})
        self._build_models(sched)
        self._build_links(orchestrator.links, sched)
        self._apply_schedules(orchestrator.schedules or {}, sched)
        return sched

    @property
    def model_instances(self) -> Dict[str, Any]:
        return dict(self._instances)

    # -- model construction -------------------------

    def _build_models(self, sched: Scheduler):
        for model_id, cfg in self._model_configs.items():
            model = self._instantiate_model(model_id, cfg)
            self._instances[model_id] = model
            sched.add(model)
            logger.info(f"Model '{model_id}' (TYPE={cfg.get('TYPE')}) built.")

    def _instantiate_model(self, model_id: str, cfg: dict):
        mtype = cfg.get("TYPE")
        entry = self._registry.get(mtype)

        if entry is None:
            entry = self._resolve_model_class(mtype)
        if entry is None:
            raise ValueError(
                f"Model '{model_id}' has unknown TYPE '{mtype}'; "
                f"registered types: {list(self._registry)}. "
                f"Register it in ym_models or name the model class '{mtype}' "
                f"under yunmeng.solutions."
            )

        if isinstance(entry, dict):
            params = dict(entry.get("params", {}))
            params.setdefault("config", cfg)
            return self._class_builder.build(
                {"class": entry["class"], "params": params}
            )

        return entry(self._translate_config(model_id, cfg))

    # -- config translation -------------------------

    def _translate_config(self, model_id: str, cfg: dict) -> dict:
        runtime = dict(cfg.get("CUSTOMS") or {})

        runtime["id"] = cfg.get("PROJECT", model_id)

        temporal = cfg.get("TEMPORAL") or {}
        dt = float(temporal.get("time_step", 86400.0))
        runtime["dt"] = dt
        runtime["start"] = 0.0
        runtime["steps"] = self._count_steps(temporal, dt)

        gauges = self._build_gauges(cfg.get("DATAS") or {})
        if gauges:
            runtime["gauges"] = gauges

        return runtime

    @staticmethod
    def _count_steps(temporal: dict, dt: float) -> int:
        start, end = temporal.get("start_time"), temporal.get("end_time")
        if not start or not end:
            return 0
        import datetime as _dt

        def _parse(t):
            if isinstance(t, _dt.datetime):
                return t
            return _dt.datetime.strptime(str(t), "%Y-%m-%d %H:%M:%S")

        return max(int((_parse(end) - _parse(start)).total_seconds() / dt), 0)

    @staticmethod
    def _build_gauges(datas: dict) -> dict:
        series = datas.get("timeseries") or []
        if not series:
            return {}
        import numpy as np
        from yunmeng.solutions.commons.datasets import Timeseries

        gauges = {}
        for ts in series:
            gid = ts["id"]
            if ts.get("xs") is not None and ts.get("ys") is not None:
                gauges[gid] = Timeseries(
                    gid, np.asarray(ts["xs"], float), np.asarray(ts["ys"], float)
                )
            elif ts.get("expr"):
                func = eval(ts["expr"]["func"], {"__builtins__": {}})
                start, end = ts["expr"].get("start"), ts["expr"].get("end")
                step = float(ts["expr"].get("step", 3600))
                import datetime as _dt

                def _parse(t):
                    if isinstance(t, _dt.datetime):
                        return t
                    return _dt.datetime.strptime(str(t), "%Y-%m-%d %H:%M:%S")

                n = max(int((_parse(end) - _parse(start)).total_seconds() / step), 1)
                xs = np.arange(n) * step
                ys = np.array([float(func(x)) for x in xs])
                gauges[gid] = Timeseries(gid, xs, ys)
            else:
                raise ValueError(f"Timeseries '{gid}' needs xs/ys or an expr source.")
        return gauges

    @staticmethod
    def _resolve_model_class(mtype: str):
        import pkgutil
        import yunmeng.solutions as sol_pkg

        for info in pkgutil.walk_packages(
            sol_pkg.__path__, prefix=sol_pkg.__name__ + "."
        ):
            try:
                module = importlib.import_module(info.name)
            except Exception:
                continue
            cls = getattr(module, mtype, None)
            if isinstance(cls, type):
                logger.info(f"TYPE '{mtype}' resolved to {info.name}.{mtype}.")
                return cls
        return None

    # -- link construction --------------------------

    def _build_links(self, link_configs: list, sched: Scheduler):
        for link in link_configs or []:
            lid = link["id"]

            if link.get("if_use", True) is False:
                logger.info(f"Link '{lid}' disabled (if_use=False), skipped.")
                continue

            src_model = self._model_of(lid, link["source"])
            tgt_model = self._model_of(lid, link["target"])

            src_port = self._port_of(lid, src_model, link["source"]["item"], True)
            tgt_port = self._port_of(lid, tgt_model, link["target"]["item"], False)

            config = self._coupling_config(link)
            src_elems, tgt_elems = self._element_mapping(link)

            self._warn_data_operations(link)

            sched.link(
                src_port,
                tgt_port,
                config=config,
                source_elements=src_elems,
                target_elements=tgt_elems,
            )
            logger.info(
                f"Link '{lid}': {src_port.id} -> {tgt_port.id} "
                f"(mode={config.mode.name})"
            )

    def _model_of(self, lid: str, endpoint: dict):
        mid = endpoint["model"]
        if mid not in self._instances:
            raise ValueError(f"Link '{lid}' references unknown model '{mid}'.")
        return self._instances[mid]

    @staticmethod
    def _port_of(lid: str, model, item: str, is_output: bool):
        full_id = f"{model.id}.{item}"
        port = model.get_output(full_id) if is_output else model.get_input(full_id)
        if port is None:
            kind = "output" if is_output else "input"
            raise ValueError(
                f"Link '{lid}': model '{model.id}' has no {kind} port '{full_id}'."
            )
        return port

    @staticmethod
    def _coupling_config(link: dict) -> CouplingConfig:
        mode = str(link.get("mode", "PULL")).strip().upper()
        if mode == "LOOP":
            params = dict(link.get("loop", {}))
            return CouplingConfig(mode=CouplingMode.LOOP, **params)
        return CouplingConfig(mode=CouplingMode.PULL)

    def _element_mapping(self, link: dict):
        src_ids = self._io_item_ids(
            link["source"]["model"], link["source"]["item"], "outputs"
        )
        tgt_ids = self._io_item_ids(
            link["target"]["model"], link["target"]["item"], "inputs"
        )
        return src_ids, tgt_ids

    def _io_item_ids(self, model_id: str, item: str, direction: str):
        ios = (self._model_configs.get(model_id) or {}).get("IOS") or {}
        for it in ios.get(direction) or []:
            if it.get("id") == item:
                return (it.get("position") or {}).get("ids")
        return None

    @staticmethod
    def _warn_data_operations(link: dict):
        ops = link.get("data_operations") or []
        if ops:
            names = [op.get("type") for op in ops]
            logger.warning(
                f"Link '{link['id']}' declares legacy data_operations {names}; "
                f"the new Scheduler applies adapters at link time instead — "
                f"please migrate these to an adapter chain. Ignored for now."
            )

    # -- schedules ----------------------------------

    def _apply_schedules(self, schedules: dict, sched: Scheduler):
        sched.settings = dict(schedules)
        for cfg in schedules.get("callbacks") or []:
            sched.add_callback(self._class_builder.build(cfg))


# ---------------------------------------------------
# region EstimatorBuilder
# ---------------------------------------------------


class EstimatorBuilder(SchedulerBuilder):
    """Build (estimator, target_model, data, loss) for TASK=estimation.

    TODO: not yet implemented.
    """

    def build(self, orchestrator):
        # -- structural guards --------------------------
        if getattr(orchestrator, "links", None):
            raise ValueError(
                "Estimation task forbids LINKS (coupling is EVAL-only). "
                "Train/calibrate the model standalone, then couple it in a "
                "simulation task."
            )

        est_cfg = getattr(orchestrator, "estimation", None)
        if not est_cfg:
            raise ValueError("TASK=estimation requires an ESTIMATION section.")

        # -- instantiate models (reused as-is) ----------
        self._model_configs = dict(orchestrator.models or {})
        self._build_models(sched=None)  # no scheduler in estimation tasks

        # -- resolve target -----------------------------
        target_id = est_cfg["target"]
        if target_id not in self._instances:
            raise ValueError(
                f"Estimation target '{target_id}' not found in MODELS: "
                f"{list(self._instances)}."
            )
        target = self._instances[target_id]

        if not isinstance(target, IEstimable):
            raise TypeError(
                f"Model '{target_id}' ({type(target).__name__}) is not "
                f"estimable. Mix in IEstimableModel to support "
                f"calibration/training."
            )

        # -- resolve estimator & loss by short name -----
        # estimator = self._build_by_name(
        #     ym_estimators, est_cfg["estimator"], kind="estimator"
        # )
        # loss = self._build_by_name(ym_losses, est_cfg["loss"], kind="loss")

        # -- parameter subset & data binding ------------
        param_names = est_cfg.get("parameters") or target.param_names()
        data = est_cfg.get("data")  # observations binding

        logger.info(
            f"Estimation task: target='{target_id}', "
            f"estimator='{est_cfg['estimator']['name']}', "
            f"params={param_names}."
        )
        # return estimator, target, data, loss, param_names

    # -- internals --------------------------------------

    def _build_models(self, sched):
        """Reuse SchedulerBuilder model instantiation; `sched` is unused
        here (no coupling graph in estimation tasks)."""
        for model_id, cfg in self._model_configs.items():
            model = self._instantiate_model(model_id, cfg)
            self._instances[model_id] = model
            logger.info(f"Model '{model_id}' (TYPE={cfg.get('TYPE')}) built.")

    @staticmethod
    def _build_by_name(registry: dict, cfg: dict, kind: str) -> Any:
        """Resolve a short unique name from the registry and instantiate.

        cfg: {"name": "sceua", "params": {...}} — class paths are NOT
        accepted; the framework guarantees name uniqueness via registry.
        """
        name = cfg.get("name")
        if "." in str(name):
            raise ValueError(
                f"Invalid {kind} reference '{name}': yaml must use short "
                f"registered names, not class paths. Available: "
                f"{sorted(registry)}."
            )
        cls = registry.get(name)
        if cls is None:
            raise ValueError(
                f"Unknown {kind} '{name}'. Registered: {sorted(registry)}."
            )
        return cls(**(cfg.get("params") or {}))
