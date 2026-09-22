# -*- encoding: utf-8 -*-
"""Copyright (C) 2026, The YunMengEnvs Project Contributors. Welcome aboard YunmengEnvs!

To assemble a new Scheduler from the orchestrator config.
"""

import importlib

from yunmeng.interfaces.solution import CouplingConfig, CouplingKinds, ILinkableModel
from yunmeng.numerics.algos import get_class, kind_of, available
from yunmeng.setting import logger
from yunmeng.workflow.optimizer import TrajectoryObservationSet, ym_losses
from yunmeng.workflow.parser import Orchestrator
from yunmeng.workflow.scheduler import Scheduler

import numpy as np


def resolve_dotted(path: str) -> type:
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _resolve(name: str, kind: str) -> type:
    """Resolve a registered component class with kind checking."""
    cls = get_class(name)  # raises KeyError listing available names
    actual = kind_of(name)
    if actual != kind:
        raise ValueError(f"'{name}' is a {actual}, not a {kind}.")
    return cls


class SchedulerBuilder:
    """Build a Scheduler from an Orchestrator (TASK=simulation)."""

    def __init__(self) -> None:
        self._instances: dict = {}
        self._model_configs: dict = {}

    @property
    def model_instances(self) -> dict:
        return dict(self._instances)

    def build(self, orchestrator: Orchestrator) -> Scheduler:
        sched = Scheduler()
        self._model_configs = dict(orchestrator.models or {})
        self._build_models(sched)
        self._build_links(orchestrator.links, sched)
        sched.settings = dict(orchestrator.schedules or {})
        for cb_cfg in sched.settings.get("callbacks") or []:
            sched.add_callback(self._build_callback(cb_cfg))
        return sched

    # -- models --------------------------------------

    def _build_models(self, sched: Scheduler) -> None:
        for model_id, cfg in self._model_configs.items():
            model = self._instantiate_model(model_id, cfg)
            self._instances[model_id] = model
            sched.add(model)
            logger.info(f"Model '{model_id}' (TYPE={cfg.get('TYPE')}) built.")

    def _instantiate_model(self, model_id: str, cfg: dict):
        mtype = cfg.get("TYPE")
        try:
            cls = _resolve(mtype, "model")
        except KeyError:
            if "." in str(mtype):
                cls = resolve_dotted(mtype)
            else:
                raise
        return cls(cfg)

    # -- links ---------------------------------------

    def _build_links(self, link_configs: list | None, sched: Scheduler) -> None:
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
            adapters = self._build_adapter_chain(lid, link)

            if link.get("data_operations"):
                logger.warning(
                    f"Link '{lid}': legacy 'data_operations' is deprecated; "
                    f"use 'adapters' instead. Ignored."
                )

            sched.link(
                src_port,
                tgt_port,
                config=config,
                source_elements=src_elems,
                target_elements=tgt_elems,
                adapter_chain=adapters or None,
            )
            logger.info(f"Link '{lid}': {src_port.id} -> {tgt_port.id} ({config.mode})")

    def _model_of(self, lid: str, endpoint: dict):
        mid = endpoint["model"]
        if mid not in self._instances:
            raise ValueError(f"Link '{lid}' references unknown model '{mid}'.")
        return self._instances[mid]

    @staticmethod
    def _port_of(lid: str, model: ILinkableModel, item: str, is_output: bool):
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
            if "divergence_action" in params:
                from yunmeng.interfaces.solution import DivergenceAction

                params["divergence_action"] = DivergenceAction[
                    str(params["divergence_action"]).upper()
                ]
            return CouplingConfig(mode=CouplingKinds.LOOP, **params)
        return CouplingConfig(mode=CouplingKinds.PULL)

    def _element_mapping(self, link: dict) -> tuple:
        src = self._io_item_ids(
            link["source"]["model"], link["source"]["item"], "outputs"
        )
        tgt = self._io_item_ids(
            link["target"]["model"], link["target"]["item"], "inputs"
        )
        return src, tgt

    def _io_item_ids(self, model_id: str, item: str, direction: str) -> list | None:
        ios = (self._model_configs.get(model_id) or {}).get("IOS") or {}
        for it in ios.get(direction) or []:
            if it.get("id") == item:
                return (it.get("position") or {}).get("ids")
        return None

    @staticmethod
    def _build_adapter_chain(lid: str, link: dict) -> list:
        """Instantiate the ordered adapter chain declared under `adapters:`."""
        adapters = []
        for i, op in enumerate(link.get("adapters") or []):
            otype = op["type"]
            cls = get_class(otype)
            if cls is None and "." in str(otype):
                cls = resolve_dotted(otype)
            if cls is None:
                raise ValueError(
                    f"Link '{lid}': unknown adapter '{otype}' "
                    f"(registered: {available("adapter")})."
                )
            adapters.append(cls(f"{lid}.adapter{i}", **(op.get("params") or {})))
        return adapters

    @staticmethod
    def _build_callback(cfg: dict):
        cls = resolve_dotted(cfg["class"])
        return cls(**(cfg.get("params") or {}))


class EstimatorBuilder(SchedulerBuilder):
    """Build (estimator, target, data, loss) for TASK=estimation."""

    def build(self, orchestrator: Orchestrator) -> tuple:
        est_cfg = orchestrator.estimation
        if not est_cfg:
            raise ValueError("TASK=estimation requires an ESTIMATION section.")

        self._model_configs = dict(orchestrator.models or {})
        for model_id, cfg in self._model_configs.items():
            model = self._instantiate_model(model_id, cfg)
            model.initialize()
            self._instances[model_id] = model
            logger.info(f"Model '{model_id}' (TYPE={cfg.get('TYPE')}) built.")

        target_id = est_cfg["target"]
        if target_id not in self._instances:
            raise ValueError(
                f"Estimation target '{target_id}' not found in MODELS: "
                f"{list(self._instances)}."
            )
        target = self._instances[target_id]
        if not target.supports_gradients():
            logger.warning(
                f"Target '{target_id}' reports no gradient support; "
                f"gradient estimators will reject it."
            )

        estimator = self._named(est_cfg["estimator"], "estimator")
        loss = self._named(est_cfg["loss"], "loss")
        data = self._build_observations(est_cfg.get("data") or {})
        param_names = est_cfg.get("parameters") or target.param_names()

        logger.info(
            f"Estimation task: target='{target_id}', "
            f"estimator={type(estimator).__name__}, params={len(param_names)}."
        )
        return estimator, target, data, loss, param_names

    @staticmethod
    def _named(cfg: dict, kind: str):
        """Resolve by registered name (estimators via the unified registry;
        losses via the loss table, which is not a registry kind)."""
        name = cfg.get("name")
        if "." in str(name):
            raise ValueError(
                f"Invalid {kind} reference '{name}': use short registered "
                f"names, not class paths."
            )
        if kind == "estimator":
            cls = _resolve(name, "estimator")
        else:
            cls = ym_losses.get(name)
            if cls is None:
                raise ValueError(
                    f"Unknown loss '{name}'. Registered: {sorted(ym_losses)}."
                )
        return cls(**(cfg.get("params") or {}))

    def _build_observations(self, cfg: dict) -> TrajectoryObservationSet:
        """Observation declaration -> TrajectoryObservationSet.

        Supported: inline xs/matrix, or from npz file, or generated from
        another configured model run (``generated_by``).
        """
        if "generated_by" in cfg:
            ref_id = cfg["generated_by"]
            if ref_id not in self._instances:
                raise ValueError(f"Observation generator '{ref_id}' not in MODELS.")
            ref = self._instances[ref_id]
            n = int(cfg.get("n_steps"))
            ref.reset_run()
            traj = np.asarray(ref.run(n), dtype="float64")
            field = cfg.get("field", "u")
            return TrajectoryObservationSet(
                times=np.arange(1, n + 1) * float(cfg.get("dt", 1.0)),
                fields={field: traj},
            )
        if "from" in cfg:
            data = np.load(cfg["from"])
            return TrajectoryObservationSet(
                times=data["times"], fields={v: data[v] for v in cfg["variables"]}
            )
        field = cfg.get("field", "u")
        return TrajectoryObservationSet(
            times=np.asarray(cfg["xs"], float),
            fields={field: np.asarray(cfg["ys"], float)},
        )
