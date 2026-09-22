# -*- encoding: utf-8 -*-
"""Copyright (C) 2026, The YunMengEnvs Project Contributors. Welcome aboard YunmengEnvs!

Orchestrator: load and validate yml configurations.

Refactored into three explicit phases per document:
  load (yaml -> dict) -> structural check (required keys)
  -> semantic check (cross references, per-section rules).
No instantiation happens here.
"""

import argparse
import os

import yaml

from yunmeng.setting import logger, settings


class ConfigError(ValueError):
    """Configuration error with a yaml path for orientation."""

    def __init__(self, path: str, message: str) -> None:
        super().__init__(f"[{path}] {message}")
        self.path = path


class Orchestrator:
    """Loads and validates the task (links) configuration."""

    TASKS = ("simulation", "estimation")

    def __init__(self, configs: argparse.Namespace) -> None:
        self._config_file = configs.config
        self._root: str | None = None
        self._config: dict = {}
        self._load()

    # -- public views ---------------------------------

    @property
    def task(self) -> str:
        return self._config.get("TASK", "simulation")

    @property
    def schedules(self) -> dict:
        return self._config.get("SCHEDULES", {})

    @property
    def envs(self) -> dict:
        return self._config.get("ENVS", {}) or {}

    @property
    def links(self) -> list:
        return self._config.get("LINKS", [])

    @property
    def models(self) -> dict:
        return self._config.get("MODELS", {})

    @property
    def estimation(self) -> dict | None:
        return self._config.get("ESTIMATION")

    def summary(self) -> str:
        models_ids = list(self.models.keys())
        links_ids = [link["id"] for link in self.links]
        return (
            f"Task: {self.task}, Models: {len(models_ids)} ({models_ids}), "
            f"Links: {len(links_ids)} ({links_ids})"
        )

    # -- phase 1: load ----------------------------------

    def _load(self) -> None:
        if not os.path.exists(self._config_file):
            raise ConfigError(self._config_file, "config file does not exist.")
        self._root = os.path.dirname(os.path.abspath(self._config_file))
        with open(self._config_file, "r", encoding="utf-8") as f:
            raw = yaml.load(f, Loader=yaml.FullLoader)
        if not isinstance(raw, dict):
            raise ConfigError(self._config_file, "top level must be a mapping.")

        self._check_structure(raw)
        self._parse_models(raw.get("MODELS") or [])
        self._parse_links(raw.get("LINKS") or [])
        self._config["SCHEDULES"] = raw.get("SCHEDULES") or {}
        self._config["ENVS"] = raw.get("ENVS") or raw.get("ENV") or {}
        self._config["TASK"] = str(raw.get("TASK", "simulation")).lower()
        self._config["ESTIMATION"] = raw.get("ESTIMATION")
        self._check_task()

        settings.load(self.envs)
        logger.info(f"Yunmeng configs loaded. {self.summary()}.")

    # -- phase 2: structure -------------------------------

    def _check_structure(self, raw: dict) -> None:
        for field in ("MODELS", "LINKS", "SCHEDULES"):
            if field not in raw:
                raise ConfigError(field, f"missing required top-level section.")
        task = str(raw.get("TASK", "simulation")).lower()
        if task not in self.TASKS:
            raise ConfigError("TASK", f"unknown task '{task}' (known: {self.TASKS}).")

    def _check_task(self) -> None:
        if self.task == "estimation":
            if not self._config.get("ESTIMATION"):
                raise ConfigError(
                    "ESTIMATION", "TASK=estimation requires this section."
                )
            if self._config.get("LINKS"):
                raise ConfigError(
                    "LINKS", "estimation task forbids coupling links (EVAL-only)."
                )

    # -- models -------------------------------------------

    def _parse_models(self, configs: list) -> None:
        models = {}
        for entry in configs:
            mid = entry.get("id")
            if not mid:
                raise ConfigError("MODELS", f"entry {entry} misses 'id'.")
            if mid in models:
                raise ConfigError(f"MODELS.{mid}", "duplicated model id.")
            models[mid] = self._load_model_file(mid, entry)
        self._config["MODELS"] = models

    def _load_model_file(self, mid: str, entry: dict) -> dict:
        rel = entry.get("from")
        if not rel:
            raise ConfigError(f"MODELS.{mid}", "misses 'from' (model config file).")
        path = rel if os.path.exists(rel) else os.path.join(self._root, rel)
        if not os.path.exists(path):
            raise ConfigError(f"MODELS.{mid}.from", f"file '{rel}' does not exist.")
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        if cfg.get("PROJECT") != mid:
            raise ConfigError(
                f"MODELS.{mid}", f"PROJECT '{cfg.get('PROJECT')}' != id '{mid}'."
            )
        for field in ("TYPE", "SPATIAL", "TEMPORAL"):
            if field not in cfg:
                raise ConfigError(f"MODELS.{mid}.{field}", "missing required field.")
        cfg.setdefault("DATAS", None)
        cfg.setdefault("SOLVER", None)
        cfg.setdefault("OPERATORS", None)
        cfg.setdefault("CUSTOMS", None)
        cfg["id"] = mid
        cfg["IOS"] = {
            "inputs": entry.get("inputs") or [],
            "outputs": entry.get("outputs") or [],
        }

        self._check_temporal(mid, cfg["TEMPORAL"])
        self._check_spatial(mid, cfg["SPATIAL"])
        if cfg.get("DATAS"):
            self._check_datas(mid, cfg["DATAS"])
        if cfg.get("SOLVER"):
            self._check_solver(mid, cfg["SOLVER"])
        if cfg.get("OPERATORS"):
            self._check_operators(mid, cfg["OPERATORS"])
        self._check_io(mid, cfg["IOS"])
        return cfg

    # -- semantic checks -----------------------------------

    def _check_temporal(self, mid: str, cfg: dict) -> None:
        for f in ("start_time", "end_time", "time_step"):
            if f not in cfg:
                raise ConfigError(f"MODELS.{mid}.TEMPORAL", f"misses '{f}'.")
        if float(cfg["time_step"]) <= 0:
            raise ConfigError(f"MODELS.{mid}.TEMPORAL.time_step", "must be > 0.")

    def _check_spatial(self, mid: str, cfg: dict) -> None:
        if "type" not in cfg:
            raise ConfigError(f"MODELS.{mid}.SPATIAL", "misses 'type'.")
        if not cfg.get("load_from") and "params" not in cfg:
            raise ConfigError(f"MODELS.{mid}.SPATIAL", "misses 'params'.")

    def _check_datas(self, mid: str, cfg: dict) -> None:
        for kind in ("timeseries", "curves", "patterns"):
            for item in cfg.get(kind) or []:
                if "id" not in item:
                    raise ConfigError(
                        f"MODELS.{mid}.DATAS.{kind}", f"{item} misses 'id'."
                    )
                has_inline = {"xs", "ys"} <= item.keys()
                has_source = bool({"expr", "from"} & item.keys())
                if not (has_inline or has_source):
                    raise ConfigError(
                        f"MODELS.{mid}.DATAS.{kind}.{item['id']}",
                        "needs xs/ys, an expr, or a from source.",
                    )

    def _check_solver(self, mid: str, cfg: dict) -> None:
        if not ({"id", "type"} <= cfg.keys()):
            raise ConfigError(f"MODELS.{mid}.SOLVER", "misses 'id' or 'type'.")
        for ic in cfg.get("ics") or []:
            if not ({"id", "field", "method"} <= ic.keys()):
                raise ConfigError(
                    f"MODELS.{mid}.SOLVER.ics", f"{ic} misses id/field/method."
                )
        for bc in cfg.get("bcs") or []:
            if not ({"id", "field", "method"} <= bc.keys()):
                raise ConfigError(
                    f"MODELS.{mid}.SOLVER.bcs", f"{bc} misses id/field/method."
                )
            if not (bc.get("patches") or bc.get("region")):
                raise ConfigError(
                    f"MODELS.{mid}.SOLVER.bcs.{bc['id']}", "misses patches/region."
                )
        for cbc in cfg.get("coupled_bcs") or []:
            if not ({"id", "field"} <= cbc.keys()):
                raise ConfigError(
                    f"MODELS.{mid}.SOLVER.coupled_bcs", f"{cbc} misses id/field."
                )

    def _check_operators(self, mid: str, cfg: list) -> None:
        for op in cfg:
            if "method" not in op:
                raise ConfigError(f"MODELS.{mid}.OPERATORS", f"{op} misses 'method'.")

    def _check_io(self, mid: str, ios: dict) -> None:
        for direction in ("inputs", "outputs"):
            for it in ios[direction]:
                if "id" not in it:
                    raise ConfigError(
                        f"MODELS.{mid}.IOS.{direction}", f"{it} misses 'id'."
                    )

    # -- links ---------------------------------------------

    def _parse_links(self, configs: list) -> None:
        links = []
        for link in configs:
            lid = link.get("id")
            if not lid:
                raise ConfigError("LINKS", f"{link} misses 'id'.")
            for end in ("source", "target"):
                if end not in link:
                    raise ConfigError(f"LINKS.{lid}", f"misses '{end}'.")
                if not ({"model", "item"} <= link[end].keys()):
                    raise ConfigError(f"LINKS.{lid}.{end}", "misses model/item.")
            for op in link.get("adapters") or []:
                if "type" not in op:
                    raise ConfigError(f"LINKS.{lid}.adapters", f"{op} misses 'type'.")
            links.append(link)
        self._config["LINKS"] = links
