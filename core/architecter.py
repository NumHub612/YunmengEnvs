# -*- encoding: utf-8 -*-
"""
Support yml configuration.
"""
from configs.settings import settings, logger
import os
import yaml


class Orchestrator:
    """
    Orchestrator class for loading and parsing yml configuration.

    This class sets and checks the model configuration standards,
    but it does not instantiate any components.
    """

    REQUIRED_LINK_FIELDS = ["MODELS", "ENV", "LINKS", "SCHEDULES"]
    REQUIRED_MODEL_FIELDS = [
        "PROJECT",
        "TYPE",
        "SPATIAL",
        "TEMPORAL",
        "DATAS",
        "SOLVER",
        "OPERATORS",
        "CUSTOMS",
    ]

    def __init__(self, configs):
        self._config_file = configs.config
        self._root = None
        self._config = {}

        self._load()

    @property
    def schedules(self) -> dict:
        """Schedule configurations."""
        return self._config.get("SCHEDULES", {})

    @property
    def envs(self) -> dict:
        """Environment configurations."""
        return self._config.get("ENV", {})

    @property
    def links(self) -> list:
        """Link configurations."""
        return self._config.get("LINKS", [])

    @property
    def models(self) -> dict:
        """Model configurations."""
        return self._config.get("MODELS", {})

    def _load(self):
        """Load and activate the configurations."""
        # Load the configurations
        if not os.path.exists(self._config_file):
            raise ValueError(f"Config file {self._config_file} doesn't exist.")

        self._root = os.path.dirname(self._config_file)
        raw_configs = yaml.load(
            open(self._config_file, "r", encoding="utf-8"),
            Loader=yaml.FullLoader,
        )
        self._parse_configs(raw_configs)

        # Activate the settings
        self._activate_env_settings(self.envs)

        # Log
        summary = self.summary()
        logger.info(f"Yunmeng Configs loaded. {summary}.")

    def _activate_env_settings(self, configs: dict):
        """Activate the settings."""
        for key, value in configs.items():
            settings[key] = value

    def _parse_configs(self, configs: dict):
        """
        Parse the link configurations.

        This method checks the coupling settings and parses model configurations.
        """
        for field in self.REQUIRED_LINK_FIELDS:
            if field not in configs:
                raise ValueError(f"Field {field} is missing in LINK configs.")

        # Parse `MODELS` section
        self._parse_model_configs(configs["MODELS"])

        # Parse `LINKS` section
        self._parse_link_configs(configs["LINKS"])

        # Parse `SCHEDULES` section
        self._parse_schedule_configs(configs["SCHEDULES"])

        # Check `ENV` section.
        self._config["SCHEDULES"] = configs.get("ENV", {})

    def _parse_schedule_configs(self, configs: dict):
        """
        Parse the spatial configurations.
        """
        if not configs:
            self._config["SCHEDULES"] = {}
            return

        self._config["SCHEDULES"] = configs

    def _parse_model_configs(self, configs: dict):
        """
        Parse the model configurations.
        """
        # Load the models and build the IO exchange items.
        model_configs = {}
        for task in configs:
            model_id = task["id"]
            model = self._parse_task_configs(task)
            model_configs[model_id] = model

        self._config["MODELS"] = model_configs

    def _parse_link_configs(self, configs: dict):
        """
        Parse the temporal configurations.
        """
        if not configs:
            self._config["LINKS"] = {}
            return

        for link in configs:
            if "id" not in link:
                raise ValueError(f"Link configs {link} miss id.")
            lid = link["id"]
            if not ({"source", "target"} <= link.keys()):
                raise ValueError(f"Link {lid} miss source or target.")
            if not ({"model", "item"} <= link["source"].keys()):
                raise ValueError(f"Link {lid} provider miss model or item.")
            if not ({"model", "item"} <= link["target"].keys()):
                raise ValueError(f"Link {lid} consumer miss model or item.")
            if "data_operations" in link:
                for op in link["data_operations"]:
                    if not ({"type", "params"} <= op.keys()):
                        raise ValueError(
                            f"Link {lid} data op {op} miss type or params."
                        )
        self._config["LINKS"] = configs

    def _parse_task_configs(self, configs: dict):
        """
        Parse the solver configurations.

        This method checks the model settings and builds the IO exchange items.
        """
        # Load the model configuration
        model_root, model_file = self._search_file(configs["from"], self._root)
        model_root = os.path.abspath(model_root)
        if model_file is None:
            raise ValueError(f"Model file {configs['from']} does not exist.")

        model_config = yaml.load(
            open(model_file, "r", encoding="utf-8"),
            Loader=yaml.FullLoader,
        )

        model_id = configs["id"]
        if model_config["PROJECT"] != model_id:
            raise ValueError(f"Model ID {model_id} doesn't match project.")
        if "TYPE" not in model_config:
            raise ValueError(f"Model {model_id} doesn't have TYPE.")

        # Check required fields.
        for field in self.REQUIRED_MODEL_FIELDS:
            if field not in model_config:
                raise ValueError(f"Field {field} is missing in {model_file}.")

        # Check each field.
        self._check_spatial_configs(model_config["SPATIAL"])
        self._check_temporal_configs(model_config["TEMPORAL"])
        self._check_datas_configs(model_config["DATAS"])
        self._check_solver_configs(model_config["SOLVER"])
        self._check_operators_configs(model_config["OPERATORS"])

        # Build the Input-Output exchange items.
        io_config = self._parse_io_configs(configs)
        model_config.update({"IOS": io_config})

        return model_config

    def _check_temporal_configs(self, config: dict):
        """
        Check the temporal configurations.
        """
        assert config is not None, "Temporal configs should not be None."

        required_fields = ["start_time", "end_time", "time_step"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Temporal configs miss {field}.")

    def _check_spatial_configs(self, config: dict):
        """
        Check the spatial configurations.
        """
        assert config is not None, "Spatial configs should not be None."

        if "type" not in config:
            raise ValueError("Spatial configs miss type.")

        if "load_from" not in config or config["load_from"] is None:
            if "params" not in config:
                raise ValueError("Spatial configs miss params.")

        if "patches" in config:
            for patch in config["patches"]:
                if not ({"id", "etype"} <= patch.keys()):
                    raise ValueError("Patch configs miss patch id or etype.")
                if not ({"expr", "spec", "from"} & patch.keys()):
                    raise ValueError(f"Patch {patch['id']} has no data source.")

        if "zones" in config:
            for zone in config["zones"]:
                if not ({"id", "etype"} <= zone.keys()):
                    raise ValueError("Zone configs miss zone id or etype.")
                if not ({"expr", "contour", "from"} & zone.keys()):
                    raise ValueError(f"Zone {zone['id']} has no data source.")

    def _check_datas_configs(self, config: dict):
        """
        Check the data configurations.
        """
        if not config:
            return

        if "timeseries" in config:
            for ts in config["timeseries"]:
                if "id" not in ts:
                    raise ValueError(f"Timeseries configs {ts} miss id.")
                if not ({"xs", "ys"} <= ts.keys()) and not (
                    {"expr", "from"} & ts.keys()
                ):
                    raise ValueError(f"Timeseries {ts['id']} has no data source.")

        if "curves" in config:
            for curve in config["curves"]:
                if "id" not in curve:
                    raise ValueError(f"Curve configs {curve} miss id.")
                if not ({"xs", "ys"} <= curve.keys()) and not (
                    {"expr", "from"} & curve.keys()
                ):
                    raise ValueError(f"Curve {curve['id']} has no data source.")

        if "patterns" in config:
            for pattern in config["patterns"]:
                if "id" not in pattern:
                    raise ValueError(f"Pattern configs {pattern} miss id.")
                if not ({"xs", "ys"} <= curve.keys()) and not (
                    {"expr", "from"} & curve.keys()
                ):
                    raise ValueError(f"Pattern {pattern['id']} has no data source.")

        if "tables" in config:
            for table in config["tables"]:
                if "id" not in table:
                    raise ValueError(f"Table configs {table} miss id.")
                if not ({"xs", "ys", "zs", "vs"} <= table.keys()) and not (
                    {"expr", "from"} & table.keys()
                ):
                    raise ValueError(f"Table {table['id']} has no data source.")

        if "fields" in config:
            for field in config["fields"]:
                if not ({"id", "var", "etype", "dtype"} <= field.keys()):
                    raise ValueError(
                        f"Field configs {field} miss id, var, etype or dtype."
                    )
                if not ({"expr", "from"} & field.keys()):
                    raise ValueError(f"Field {field['id']} has no data source.")

    def _check_solver_configs(self, config: dict):
        """
        Check the solver configurations.
        """
        if not config:
            return

        if not ({"id", "type"} <= config.keys()):
            raise ValueError("Solver configs miss id and type.")
        sid = config["id"]

        if "load_from" not in config or config["load_from"] is None:
            if "params" not in config:
                raise ValueError(f"Solver {sid} configs miss params.")

        if "ics" not in config:
            raise ValueError(f"Solver {sid} configs miss ics.")
        if config["ics"] is not None:
            required_ic_fields = {"id", "field", "method"}
            for ic in config["ics"]:
                if not (required_ic_fields <= ic.keys()):
                    raise ValueError(f"IC {ic} miss {required_ic_fields}.")

        if "bcs" not in config:
            raise ValueError(f"Solver {sid} configs miss bcs.")
        if config["bcs"] is not None:
            required_bc_fields = {"id", "field", "method", "patches"}
            for bc in config["bcs"]:
                if not (required_bc_fields <= bc.keys()):
                    raise ValueError(f"BC {bc} miss {required_bc_fields}.")

        if "cbs" not in config:
            raise ValueError(f"Solver {sid} configs miss cbs.")
        if config["cbs"] is not None:
            required_cb_fields = {"id", "method"}
            for cb in config["cbs"]:
                if not (required_cb_fields <= cb.keys()):
                    raise ValueError(f"CB {cb} miss {required_cb_fields}.")

    def _check_operators_configs(self, config: dict):
        """
        Check the operator configurations.
        """
        if not config:
            return

        for op in config:
            if not ({"method", "params"} <= op.keys()):
                raise ValueError(f"Operator {op} miss method or params.")

    def _parse_io_configs(self, configs: dict):
        """
        Parse the IO configurations.
        """
        inputs = [] if "inputs" not in configs else configs["inputs"]
        outputs = [] if "outputs" not in configs else configs["outputs"]

        def _check(items, is_input=True):
            item_type = "input" if is_input else "output"
            for it in items:
                if "id" not in it:
                    raise ValueError(f"{item_type} {it} has no id.")
                it_name = it["id"]
                if not ({"quantity", "quality"} & it.keys()):
                    raise ValueError(f"{item_type} {it_name} miss quantity or quality.")
                if "quantity" in it:
                    if "variable" not in it["quantity"]:
                        raise ValueError(f"{item_type} {it_name} miss variable.")
                if not ({"element", "ids"} & it["position"].keys()):
                    raise ValueError(f"{item_type} {it_name} miss element or ids.")

        _check(inputs)
        _check(outputs, is_input=False)

        io_config = {"inputs": inputs, "outputs": outputs}
        return io_config

    def _search_file(self, file_name: str, extra_path: str):
        """
        Search the file with relative path.
        """
        if os.path.exists(file_name):
            return os.path.dirname(file_name), file_name

        if extra_path is None:
            return None, None

        file_path = os.path.join(extra_path, file_name)
        if os.path.exists(file_path):
            return extra_path, file_path

        return None, None

    def summary(self) -> str:
        """
        Print the summary of the configurations.
        """
        models_ids = list(self.models.keys())
        models_num = len(models_ids)

        links_ids = [link["id"] for link in self.links]
        links_num = len(links_ids)

        summary = (
            f"Models: {models_num} ({models_ids}), \tLinks: {links_num} ({links_ids})"
        )
        return summary
