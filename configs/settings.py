# -*- encoding: utf-8 -*-
"""
Project configurations.
"""
import os
import logging
import logging.handlers
import json


class Settings:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Settings, cls).__new__(cls)
            cls._instance._configs = {}
        return cls._instance

    def load_configs(self, config_file="configs.json"):
        with open(config_file, "r") as f:
            self._configs = json.load(f)

    def save_configs(self, config_file="configs.json"):
        with open(config_file, "w") as f:
            json.dump(self._configs, f, indent=4)

    def __getattr__(self, name):
        if name in self._configs:
            return self._configs[name]
        else:
            raise AttributeError(f"No such attribute: {name}")

    def __setattr__(self, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._configs[name] = value

    @property
    def NUMERIC_TOLERANCE(self):
        """Numeric tolerance for floating point comparison."""
        return self._configs.get("NUMERIC_TOLERANCE", 1e-6)

    @NUMERIC_TOLERANCE.setter
    def NUMERIC_TOLERANCE(self, value):
        self._configs["NUMERIC_TOLERANCE"] = abs(value)


# global settings object
settings = Settings()


# set up logging system
logger = logging.getLogger("yunmengenvs")
formatter = logging.Formatter(
    "[%(asctime)s][%(name)s][%(levelname)s][%(process)d][%(thread)d]: %(message)s"
)

log_file = os.path.abspath(os.path.join("./", "yunmeng.log"))
file_handler = logging.handlers.RotatingFileHandler(log_file, "a", 1024 * 1024 * 10, 10)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.WARNING)
logger.addHandler(console_handler)

logger.setLevel(logging.INFO)
